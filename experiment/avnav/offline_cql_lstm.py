"""Train discrete CQL with a bounded LSTM history over collision-risk states."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import yaml

from experiment.avnav.config import resolve_path
from experiment.avnav.model.hybrid import Network
from experiment.avnav.offline_rl import _require_clean_data_audit, _sha256


class SequenceDataset:
    def __init__(self, root):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        collected = {key: [] for key in ("states", "next_states", "actions", "rewards", "dones")}
        for shard in manifest["shards"]:
            payload = torch.load(root / shard["file"], map_location="cpu", weights_only=True)
            for key in collected:
                if len(payload[key]) != shard["samples"]:
                    raise ValueError(f"{shard['file']}: manifest sample count mismatch")
                collected[key].append(payload[key])
        self.data = {key: torch.cat(rows, dim=0) for key, rows in collected.items()}
        self.samples = int(manifest["samples"])
        self.sequence_length = int(manifest["sequence_length"])


class LSTMActor(nn.Module):
    def __init__(self, feature_dim=264, hidden_dim=256, action_count=4, foundation=None):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        if foundation is None:
            self.head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(),
                                      nn.LayerNorm(128), nn.Linear(128, action_count))
        else:
            self.head = deepcopy(foundation.final.action_head)

    def forward(self, sequence):
        outputs, _ = self.lstm(sequence)
        return self.head(outputs[:, -1])


class LSTMQNetwork(nn.Module):
    def __init__(self, feature_dim=264, hidden_dim=256, action_count=4):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, action_count))

    def forward(self, sequence):
        outputs, _ = self.lstm(sequence)
        return self.head(outputs[:, -1])


def _batch(data, indices, device):
    return {key: value[indices].to(device, dtype=(torch.float32 if value.is_floating_point()
                                                   else value.dtype),
                                   non_blocking=True)
            for key, value in data.items()}


def _evaluate_actor(actor, data, device, batch_size=1024):
    actor.eval()
    loss_sum, correct, seen = 0.0, 0, 0
    with torch.inference_mode():
        for start in range(0, len(data["actions"]), batch_size):
            stop = min(len(data["actions"]), start + batch_size)
            states = data["states"][start:stop].to(device, dtype=torch.float32)
            actions = data["actions"][start:stop].to(device)
            logits = actor(states)
            loss_sum += F.cross_entropy(logits, actions, reduction="sum").item()
            correct += (logits.argmax(-1) == actions).sum().item()
            seen += len(actions)
    actor.train()
    return {"action_nll": loss_sum / max(seen, 1),
            "action_accuracy": correct / max(seen, 1), "samples": seen}


def _evaluate_validation(actor, datasets, device):
    splits = {name: _evaluate_actor(actor, dataset.data, device)
              for name, dataset in datasets.items()}
    return {"action_nll": max(row["action_nll"] for row in splits.values()),
            "splits": splits}


def train(train_data, validation_data, foundation, cfg, output):
    device = torch.device(cfg["device"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.manual_seed_all(cfg["seed"])

    actor = LSTMActor(cfg["feature_dim"], cfg["hidden_dim"], foundation=foundation).to(device)
    q1 = LSTMQNetwork(cfg["feature_dim"], cfg["hidden_dim"]).to(device)
    q2 = LSTMQNetwork(cfg["feature_dim"], cfg["hidden_dim"]).to(device)
    target_q1, target_q2 = deepcopy(q1).eval(), deepcopy(q2).eval()
    for target in (target_q1, target_q2):
        target.requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"])
    critic_optimizer = torch.optim.Adam([*q1.parameters(), *q2.parameters()],
                                        lr=cfg["learning_rate"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history, best_nll = [], float("inf")
    train_tensors = train_data.data
    for epoch in range(1, cfg["epochs"] + 1):
        actor.train(); q1.train(); q2.train()
        sums = {key: 0.0 for key in ("critic_loss", "actor_loss", "td_loss", "cql_gap", "bc_loss")}
        for _ in tqdm(range(cfg["updates_per_epoch"]), desc=f"CQL-LSTM {epoch}/{cfg['epochs']}",
                      mininterval=10):
            indices = torch.randint(len(train_tensors["actions"]), (cfg["batch_size"],))
            data = _batch(train_tensors, indices, device)
            with torch.no_grad():
                next_logits = actor(data["next_states"])
                next_log_probs = F.log_softmax(next_logits, dim=-1)
                next_probs = next_log_probs.exp()
                next_q = torch.minimum(target_q1(data["next_states"]),
                                       target_q2(data["next_states"]))
                next_value = (next_probs * (next_q - cfg["alpha"] * next_log_probs)).sum(-1)
                target = data["rewards"] + cfg["gamma"] * (1.0 - data["dones"]) * next_value

            q_values_1, q_values_2 = q1(data["states"]), q2(data["states"])
            chosen_1 = q_values_1.gather(1, data["actions"].unsqueeze(1)).squeeze(1)
            chosen_2 = q_values_2.gather(1, data["actions"].unsqueeze(1)).squeeze(1)
            td_loss = F.mse_loss(chosen_1, target) + F.mse_loss(chosen_2, target)
            cql_gap = (torch.logsumexp(q_values_1, dim=-1) - chosen_1).mean() + (
                torch.logsumexp(q_values_2, dim=-1) - chosen_2).mean()
            critic_loss = td_loss + cfg["cql_weight"] * cql_gap
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_([*q1.parameters(), *q2.parameters()], 10.0)
            critic_optimizer.step()

            logits = actor(data["states"])
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            with torch.no_grad():
                policy_q = torch.minimum(q1(data["states"]), q2(data["states"]))
            sac_loss = (probs * (cfg["alpha"] * log_probs - policy_q)).sum(-1).mean()
            bc_loss = F.cross_entropy(logits, data["actions"])
            actor_loss = sac_loss + cfg["bc_weight"] * bc_loss
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            actor_optimizer.step()

            with torch.no_grad():
                for target_net, source_net in ((target_q1, q1), (target_q2, q2)):
                    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
                        target_param.lerp_(source_param, cfg["target_tau"])
            for key, value in (("critic_loss", critic_loss), ("actor_loss", actor_loss),
                               ("td_loss", td_loss), ("cql_gap", cql_gap), ("bc_loss", bc_loss)):
                sums[key] += float(value.detach().item())

        metrics = {key: value / cfg["updates_per_epoch"] for key, value in sums.items()}
        metrics["epoch"] = epoch
        metrics["validation"] = _evaluate_validation(actor, validation_data, device)
        history.append(metrics)
        temporary = output / "metrics.json.tmp"
        temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
        temporary.replace(output / "metrics.json")
        checkpoint = {
            "epoch": epoch, "actor": actor.state_dict(), "q1": q1.state_dict(),
            "q2": q2.state_dict(), "target_q1": target_q1.state_dict(),
            "target_q2": target_q2.state_dict(),
            "foundation_sha256": _sha256(cfg["foundation_checkpoint"]),
            "config": cfg, "metrics": metrics, "algorithm": "CQL_LSTM",
        }
        torch.save(checkpoint, output / f"epoch_{epoch:03d}.pt")
        if metrics["validation"]["action_nll"] < best_nll:
            best_nll = metrics["validation"]["action_nll"]
            torch.save(checkpoint, output / "best_policy.pt")
        torch.save(checkpoint, output / "latest.pt")
        print(json.dumps(metrics), flush=True)
    return history


def run(config_path, output_override=None):
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit = _require_clean_data_audit(config)
    rl = config["OFFLINE_RL"]
    foundation_path = resolve_path(config["FOUNDATION"]["CHECKPOINT"])
    output = resolve_path(output_override or rl["OUTPUT"])
    cache_root = resolve_path(rl["SEQUENCE_CACHE_ROOT"]) / _sha256(foundation_path)[:16]
    train_data = SequenceDataset(cache_root / "train")
    validation_data = {name: SequenceDataset(cache_root / name)
                       for name in ("validation", "validation_unheard")}
    feature_dim = int(train_data.data["states"].shape[-1])
    if feature_dim != 264 or train_data.sequence_length != int(rl["SEQUENCE_LENGTH"]):
        raise ValueError("Training sequence cache schema differs from the frozen config")
    predictor_path = resolve_path(rl["COLLISION_PREDICTOR_CHECKPOINT"])
    predictor = torch.load(predictor_path, map_location="cpu", weights_only=True)
    if predictor.get("foundation_sha256") != _sha256(foundation_path):
        raise ValueError("Collision predictor and foundation checkpoint hashes differ")
    device = torch.device(rl["DEVICE"])
    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(foundation_path, map_location=device, weights_only=True))
    foundation.eval().requires_grad_(False)
    training_config = {
        "algorithm": "cql_lstm", "architecture": "lstm_actor_and_double_q",
        "sequence_length": train_data.sequence_length, "padding": "left-zero; reset per episode",
        "feature_dim": feature_dim, "seed": int(config.get("SEED", 42)),
        "epochs": int(rl["EPOCHS"]), "updates_per_epoch": int(rl["UPDATES_PER_EPOCH"]),
        "batch_size": int(rl["BATCH_SIZE"]), "learning_rate": float(rl["LEARNING_RATE"]),
        "gamma": float(rl["GAMMA"]), "alpha": float(rl["ALPHA"]),
        "cql_weight": float(rl["CQL_WEIGHT"]), "bc_weight": float(rl["BC_WEIGHT"]),
        "target_tau": float(rl["TARGET_TAU"]), "hidden_dim": int(rl["HIDDEN_DIM"]),
        "device": str(device), "foundation_checkpoint": str(foundation_path),
        "foundation_sha256": _sha256(foundation_path), "action_history": True,
        "collision_history": False, "collision_risk_actions": True,
        "collision_predictor_checkpoint": str(predictor_path),
        "collision_predictor_sha256": _sha256(predictor_path),
        "input_schema": "embedding256+previous_action_onehot4+predicted_collision_probability_by_action4",
        "feature_cache": str(cache_root), "data_source_audit": audit,
    }
    config["OFFLINE_RL"]["OUTPUT"] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "training_protocol.json").write_text(
        json.dumps(training_config, indent=2) + "\n", encoding="utf-8")
    history = train(train_data, validation_data, foundation, training_config, output)
    best_nll = min(history, key=lambda row: row["validation"]["action_nll"])
    summary = {
        "status": "completed", "algorithm": "cql_lstm", "training_config": training_config,
        "epochs": len(history), "best_action_nll_epoch": best_nll["epoch"],
        "train_samples": train_data.samples,
        "validation_samples": {name: dataset.samples for name, dataset in validation_data.items()},
        "best_policy_by_validation_action_nll": str(output / "best_policy.pt"),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output or None), indent=2), flush=True)


if __name__ == "__main__":
    main()
