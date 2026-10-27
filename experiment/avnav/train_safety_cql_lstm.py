"""Train safety-input CQL-LSTM policies with optional collision reward costs."""

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import yaml

from experiment.avnav.config import resolve_path
from experiment.avnav.model.hybrid import Network
from experiment.avnav.offline_cql_lstm import (
    LSTMQNetwork,
    LSTMActor,
    SequenceDataset,
    _batch,
    _evaluate_validation,
)
from experiment.avnav.offline_rl import _require_clean_data_audit, _sha256


# In the cached safety state schema, indices 256:260 hold the previous action,
# index 260 holds previous_collision, and 261 onward contains progress, depth,
# and learned action-risk features. The next state at the final sequence slot
# therefore carries the collision label for the sampled transition.
COLLISION_HISTORY_INDEX = 260


def train(config_path, feature_cache, output_dir, cost_weight=0.0,
          train_cost_critic=False, collision_reward_penalty=0.0):
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit = _require_clean_data_audit(config)
    rl, features = config["OFFLINE_RL"], config["FEATURES"]
    if features.get("COLLISION_HISTORY") is not True:
        raise ValueError("Safety CQL-LSTM must include previous-collision history")
    if features.get("DISTANCE_DELTA_FEATURE") is not True:
        raise ValueError("Safety CQL-LSTM must include distance progress")
    if features.get("LOCAL_GEOMETRY") is not True:
        raise ValueError("Safety CQL-LSTM must include local depth geometry")
    if train_cost_critic and cost_weight <= 0:
        raise ValueError("Cost-critic training requires a positive, predeclared cost weight")
    collision_reward_penalty = float(collision_reward_penalty)
    if collision_reward_penalty < 0:
        raise ValueError("Collision reward penalty must be non-negative")

    feature_cache = Path(feature_cache).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty training output: {output}")
    train_data = SequenceDataset(feature_cache / "train")
    validation_data = {
        name: SequenceDataset(feature_cache / name)
        for name in ("validation", "validation_unheard")
    }
    feature_dim = int(train_data.data["states"].shape[-1])
    sequence_length = int(train_data.sequence_length)
    if sequence_length != int(rl.get("SEQUENCE_LENGTH", 5)):
        raise ValueError("Cached sequence length differs from the frozen training config")
    if feature_dim not in (271, 275):
        raise ValueError(f"Unexpected safety feature width {feature_dim}")

    foundation_path = resolve_path(config["FOUNDATION"]["CHECKPOINT"])
    device = torch.device(rl["DEVICE"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    seed = int(config.get("SEED", 20261025))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(foundation_path, map_location=device,
                                          weights_only=True))
    foundation.eval().requires_grad_(False)
    actor = LSTMActor(feature_dim, int(rl["HIDDEN_DIM"]), foundation=foundation).to(device)
    q1 = LSTMQNetwork(feature_dim, int(rl["HIDDEN_DIM"])).to(device)
    q2 = LSTMQNetwork(feature_dim, int(rl["HIDDEN_DIM"])).to(device)
    target_q1, target_q2 = deepcopy(q1).eval(), deepcopy(q2).eval()
    for target in (target_q1, target_q2):
        target.requires_grad_(False)

    cost_q1 = cost_q2 = target_cost_q1 = target_cost_q2 = None
    cost_data = None
    if train_cost_critic:
        payload = torch.load(feature_cache / "counterfactual_anchors.pt",
                             map_location="cpu", weights_only=True)
        if len(payload["states"]) < 32:
            raise ValueError("Too few complete train-split counterfactual anchors for a cost critic")
        if payload["states"].shape[-1] != feature_dim:
            raise ValueError("Counterfactual anchor feature width differs from offline cache")
        cost_data = {"states": payload["states"], "costs": payload["costs"]}
        cost_q1 = LSTMQNetwork(feature_dim, int(rl["HIDDEN_DIM"])).to(device)
        cost_q2 = LSTMQNetwork(feature_dim, int(rl["HIDDEN_DIM"])).to(device)
        target_cost_q1, target_cost_q2 = deepcopy(cost_q1).eval(), deepcopy(cost_q2).eval()
        for target in (target_cost_q1, target_cost_q2):
            target.requires_grad_(False)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(rl["LEARNING_RATE"]))
    critic_optimizer = torch.optim.Adam([*q1.parameters(), *q2.parameters()],
                                        lr=float(rl["LEARNING_RATE"]))
    cost_optimizer = (torch.optim.Adam([*cost_q1.parameters(), *cost_q2.parameters()],
                                       lr=float(rl["LEARNING_RATE"]))
                      if train_cost_critic else None)
    output.mkdir(parents=True, exist_ok=True)
    history, best_nll = [], float("inf")
    train_tensors = train_data.data
    epochs = int(rl["EPOCHS"])
    updates_per_epoch = int(rl["UPDATES_PER_EPOCH"])
    batch_size = int(rl["BATCH_SIZE"])
    cfg = {
        "algorithm": "cql_lstm_safety",
        "architecture": "lstm_actor_twin_reward_q" +
                        ("_twin_supervised_collision_cost_q" if train_cost_critic else ""),
        "feature_dim": feature_dim, "sequence_length": sequence_length,
        "padding": "left-zero; reset per episode",
        "input_schema": (
            "embedding256+previous_action4+previous_collision1+distance_delta1+"
            "depth_sector_quantiles5+immediate_collision_risk4" +
            ("+future_collision_risk4" if feature_dim == 275 else "")
        ),
        "seed": seed, "epochs": epochs, "updates_per_epoch": updates_per_epoch,
        "batch_size": batch_size, "learning_rate": float(rl["LEARNING_RATE"]),
        "gamma": float(rl["GAMMA"]), "alpha": float(rl["ALPHA"]),
        "cql_weight": float(rl["CQL_WEIGHT"]), "bc_weight": float(rl["BC_WEIGHT"]),
        "target_tau": float(rl["TARGET_TAU"]), "hidden_dim": int(rl["HIDDEN_DIM"]),
        "device": str(device), "foundation_checkpoint": str(foundation_path),
        "foundation_sha256": _sha256(foundation_path),
        "collision_history": True, "distance_delta_feature": True,
        "local_geometry": True, "action_history": True,
        "collision_history_feature_index": COLLISION_HISTORY_INDEX,
        "collision_risk_actions": True,
        "future_collision_risk_actions": bool(feature_dim == 275),
        "collision_predictor_checkpoint": str(resolve_path(
            rl["COLLISION_PREDICTOR_CHECKPOINT"])),
        "collision_predictor_sha256": _sha256(resolve_path(
            rl["COLLISION_PREDICTOR_CHECKPOINT"])),
        "feature_cache": str(feature_cache), "data_source_audit": audit,
        "data_source_audit_sha256": _sha256(audit) if audit else None,
        "cost_critic": bool(train_cost_critic),
        "collision_cost_weight": float(cost_weight),
        "collision_cost_target": "five-step collision count divided by requested horizon",
        "collision_reward_penalty": collision_reward_penalty,
        "collision_reward_transform": (
            "recorded_reward - collision_reward_penalty * current_transition_collision"
        ),
        "collision_reward_source": (
            "train-only recorded collision label aligned from next-state collision-history feature"
        ),
        "checkpoint_selection": "validation action NLL for training snapshots; policy selected by frozen validation SPL/safety rule",
    }
    (output / "training_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for epoch in range(1, epochs + 1):
        actor.train(); q1.train(); q2.train()
        if train_cost_critic:
            cost_q1.train(); cost_q2.train()
        sums = {key: 0.0 for key in (
            "critic_loss", "actor_loss", "td_loss", "cql_gap", "bc_loss", "cost_loss"
        )}
        for _ in tqdm(range(updates_per_epoch), desc=f"Safety CQL-LSTM {epoch}/{epochs}",
                      mininterval=10):
            indices = torch.randint(len(train_tensors["actions"]), (batch_size,))
            data = _batch(train_tensors, indices, device)
            collision_rewards = data["rewards"]
            if collision_reward_penalty:
                collision = data["next_states"][:, -1, COLLISION_HISTORY_INDEX]
                if torch.any((collision < 0.0) | (collision > 1.0)):
                    raise ValueError("Cached collision-history labels are outside [0, 1]")
                collision_rewards = collision_rewards - collision_reward_penalty * collision
            with torch.no_grad():
                next_logits = actor(data["next_states"])
                next_log_probs = F.log_softmax(next_logits, dim=-1)
                next_probs = next_log_probs.exp()
                next_q = torch.minimum(target_q1(data["next_states"]),
                                       target_q2(data["next_states"]))
                next_value = (next_probs * (next_q - float(rl["ALPHA"]) *
                                             next_log_probs)).sum(-1)
                target = collision_rewards + float(rl["GAMMA"]) * (
                    1.0 - data["dones"]
                ) * next_value
            q_values_1, q_values_2 = q1(data["states"]), q2(data["states"])
            chosen_1 = q_values_1.gather(1, data["actions"].unsqueeze(1)).squeeze(1)
            chosen_2 = q_values_2.gather(1, data["actions"].unsqueeze(1)).squeeze(1)
            td_loss = F.mse_loss(chosen_1, target) + F.mse_loss(chosen_2, target)
            cql_gap = (torch.logsumexp(q_values_1, dim=-1) - chosen_1).mean() + (
                torch.logsumexp(q_values_2, dim=-1) - chosen_2
            ).mean()
            critic_loss = td_loss + float(rl["CQL_WEIGHT"]) * cql_gap
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_([*q1.parameters(), *q2.parameters()], 10.0)
            critic_optimizer.step()

            cost_loss = data["rewards"].new_zeros(())
            if train_cost_critic:
                cost_indices = torch.randint(len(cost_data["costs"]), (batch_size,))
                cost_batch = cost_data["states"][cost_indices].to(device, dtype=torch.float32)
                targets = cost_data["costs"][cost_indices].to(device, dtype=torch.float32)
                cost_values_1, cost_values_2 = cost_q1(cost_batch), cost_q2(cost_batch)
                cost_loss = F.mse_loss(cost_values_1, targets) + F.mse_loss(cost_values_2, targets)
                cost_optimizer.zero_grad(set_to_none=True)
                cost_loss.backward()
                nn.utils.clip_grad_norm_([*cost_q1.parameters(), *cost_q2.parameters()], 10.0)
                cost_optimizer.step()

            logits = actor(data["states"])
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            with torch.no_grad():
                policy_q = torch.minimum(q1(data["states"]), q2(data["states"]))
            sac_loss = (probs * (float(rl["ALPHA"]) * log_probs - policy_q)).sum(-1).mean()
            bc_loss = F.cross_entropy(logits, data["actions"])
            actor_loss = sac_loss + float(rl["BC_WEIGHT"]) * bc_loss
            if train_cost_critic:
                with torch.no_grad():
                    predicted_cost = torch.maximum(cost_q1(data["states"]),
                                                   cost_q2(data["states"]))
                actor_loss = actor_loss + float(cost_weight) * (
                    probs * predicted_cost
                ).sum(-1).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            actor_optimizer.step()

            with torch.no_grad():
                pairs = [(target_q1, q1), (target_q2, q2)]
                if train_cost_critic:
                    pairs += [(target_cost_q1, cost_q1), (target_cost_q2, cost_q2)]
                for target_net, source_net in pairs:
                    for target_param, source_param in zip(target_net.parameters(),
                                                          source_net.parameters()):
                        target_param.lerp_(source_param, float(rl["TARGET_TAU"]))
            for key, value in (("critic_loss", critic_loss), ("actor_loss", actor_loss),
                               ("td_loss", td_loss), ("cql_gap", cql_gap),
                               ("bc_loss", bc_loss), ("cost_loss", cost_loss)):
                sums[key] += float(value.detach().item())

        metrics = {key: value / updates_per_epoch for key, value in sums.items()}
        metrics["epoch"] = epoch
        metrics["validation"] = _evaluate_validation(actor, validation_data, device)
        history.append(metrics)
        temporary = output / "metrics.json.tmp"
        temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
        temporary.replace(output / "metrics.json")
        checkpoint = {
            "epoch": epoch, "actor": actor.state_dict(),
            "q1": q1.state_dict(), "q2": q2.state_dict(),
            "target_q1": target_q1.state_dict(), "target_q2": target_q2.state_dict(),
            "foundation_sha256": _sha256(foundation_path), "config": cfg,
            "metrics": metrics, "algorithm": "CQL_LSTM",
        }
        if train_cost_critic:
            checkpoint.update(
                cost_q1=cost_q1.state_dict(), cost_q2=cost_q2.state_dict(),
                target_cost_q1=target_cost_q1.state_dict(),
                target_cost_q2=target_cost_q2.state_dict(),
            )
        torch.save(checkpoint, output / f"epoch_{epoch:03d}.pt")
        if metrics["validation"]["action_nll"] < best_nll:
            best_nll = metrics["validation"]["action_nll"]
            torch.save(checkpoint, output / "best_policy_by_action_nll.pt")
        torch.save(checkpoint, output / "latest.pt")
        print(json.dumps(metrics), flush=True)
    summary = {
        "status": "completed", "training_config": cfg, "epochs": len(history),
        "best_action_nll_epoch": min(history, key=lambda row: row["validation"]["action_nll"])["epoch"],
        "train_samples": train_data.samples,
        "validation_samples": {name: dataset.samples for name, dataset in validation_data.items()},
        "cost_critic_samples": len(cost_data["costs"]) if cost_data is not None else 0,
        "checkpoints": [str(path) for path in sorted(output.glob("epoch_*.pt"))],
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cost-weight", type=float, default=0.0)
    parser.add_argument("--train-cost-critic", action="store_true")
    parser.add_argument("--collision-reward-penalty", type=float, default=0.0,
                        help="Subtract this fixed amount from the recorded reward on train-only collision transitions")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.feature_cache, args.output,
                           args.cost_weight, args.train_cost_critic,
                           args.collision_reward_penalty),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
