"""Train the frozen-feature LSTM SAC-CQL policy on AVNav codex transitions."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiment.avnav.offline_rl_v1_5_lstm.OfflineNet import SAC_LSTM_CQL_v1_5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_state(path: Path, model) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
    temporary.replace(path)
    return _sha256(path)


class TransitionFeatures:
    """Flatten episodes while retaining episode-local LSTM window indices."""

    def __init__(self, path, sequence_length, collision_extra_penalty=0.0, training=False,
                 previous_action=False, collision_bc_weight=1.0):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        episodes = payload["episodes"]
        if not episodes:
            raise ValueError(f"No episodes in frozen-feature cache: {path}")
        dimensions = {tuple(row["features"].shape[1:]) for row in episodes}
        if len(dimensions) != 1 or next(iter(dimensions))[0] < 1:
            raise ValueError(f"Unexpected foundation feature dimensions in {path}: {dimensions}")
        self.feature_dim = next(iter(dimensions))[0]
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 1 or not 0 <= collision_bc_weight <= 1:
            raise ValueError('Invalid sequence length or collision BC weight')
        self.state_dim = self.feature_dim + (4 if previous_action else 0)
        feature_chunks = [torch.zeros((1, self.state_dim), dtype=torch.float32)]
        feature_offset = 1
        state_rows, next_state_rows = [], []
        actions, rewards, dones, bc_weights = [], [], [], []
        self.episode_count = len(episodes)
        self.transitions = 0
        self.recorded_collisions = 0
        self.additional_collision_penalty = float(collision_extra_penalty if training else 0.0)
        for row in episodes:
            features = row["features"].float().contiguous()
            action = row["actions"].long().reshape(-1)
            reward = row["rewards"].float().reshape(-1)
            done = row["dones"].float().reshape(-1)
            collisions = row["collisions"].bool().reshape(-1)
            count = len(action)
            if features.shape != (count + 1, 256) or any(
                    len(value) != count for value in (reward, done, collisions)):
                raise ValueError(f"Misaligned feature transitions: {row.get('task_id')}")
            if count == 0:
                continue
            if previous_action:
                previous = torch.zeros((count + 1, 4), dtype=torch.float32)
                previous[1:] = F.one_hot(action, 4).float()
                features = torch.cat((features, previous), dim=-1)
            base = feature_offset
            feature_chunks.append(features)
            feature_offset += len(features)
            self.recorded_collisions += int(collisions.sum())
            for step in range(count):
                current = [base + offset if offset >= 0 else 0
                           for offset in range(step - self.sequence_length + 1, step + 1)]
                following = [base + offset if offset >= 0 else 0
                             for offset in range(step - self.sequence_length + 2, step + 2)]
                state_rows.append(current)
                next_state_rows.append(following)
            actions.append(action)
            bc_weights.append(torch.where(collisions, float(collision_bc_weight), 1.0))
            adjusted = reward - collisions.float() * self.additional_collision_penalty
            rewards.append(adjusted)
            dones.append(done)
            self.transitions += count
        self.features = torch.cat(feature_chunks, dim=0).contiguous()
        self.state_indices = torch.tensor(state_rows, dtype=torch.long)
        self.next_state_indices = torch.tensor(next_state_rows, dtype=torch.long)
        self.actions = torch.cat(actions).contiguous()
        self.rewards = torch.cat(rewards).contiguous()
        self.dones = torch.cat(dones).contiguous()
        self.bc_weights = torch.cat(bc_weights).contiguous()

    def batch(self, indices, device):
        state_ids = self.state_indices[indices]
        next_ids = self.next_state_indices[indices]
        states = self.features[state_ids].to(device, non_blocking=True)
        next_states = self.features[next_ids].to(device, non_blocking=True)
        actions = self.actions[indices].to(device, non_blocking=True)
        rewards = self.rewards[indices].to(device, non_blocking=True)
        dones = self.dones[indices].to(device, non_blocking=True)
        return states, actions, rewards, next_states, dones, self.bc_weights[indices].to(device)


def _optimizer_update(model, states, actions, rewards, next_states, dones, sample_bc_weights, bc_weight):
    """One discrete SAC-CQL update using the latest LSTM token per transition."""
    alpha = model.log_alpha.exp().detach()
    with torch.no_grad():
        next_actor_latent = model.temporal_encoder(next_states)[0][:, -1]
        next_latent = getattr(model, 'target_temporal_encoder', model.temporal_encoder)(next_states)[0][:, -1]
        next_prob = model.actor(next_actor_latent)
        next_log_prob = torch.log(next_prob.clamp_min(1e-8))
        next_q = torch.minimum(
            model.target_critic_1(next_latent), model.target_critic_2(next_latent)
        )
        next_value = (next_prob * (next_q - alpha * next_log_prob)).sum(dim=-1)
        target = rewards + model.gamma * (1.0 - dones) * next_value

    latent = model.temporal_encoder(states)[0][:, -1]
    q1 = model.critic_1(latent)
    q2 = model.critic_2(latent)
    q1_data = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
    q2_data = q2.gather(1, actions.unsqueeze(1)).squeeze(1)
    td1 = F.mse_loss(q1_data, target)
    td2 = F.mse_loss(q2_data, target)
    cql1 = (torch.logsumexp(q1, dim=-1) - q1_data).mean()
    cql2 = (torch.logsumexp(q2, dim=-1) - q2_data).mean()
    critic_loss = td1 + td2 + model.beta * (cql1 + cql2)
    if not torch.isfinite(critic_loss):
        raise FloatingPointError("Non-finite SAC-CQL critic loss")
    model.temporal_optimizer.zero_grad(set_to_none=True)
    model.critic_1_optimizer.zero_grad(set_to_none=True)
    model.critic_2_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.temporal_encoder.parameters(), model.clip_grad_param)
    torch.nn.utils.clip_grad_norm_(model.critic_1.parameters(), model.clip_grad_param)
    torch.nn.utils.clip_grad_norm_(model.critic_2.parameters(), model.clip_grad_param)
    model.temporal_optimizer.step()
    model.critic_1_optimizer.step()
    model.critic_2_optimizer.step()

    with torch.no_grad():
        actor_latent = model.temporal_encoder(states)[0][:, -1]
        policy_q = torch.minimum(model.critic_1(actor_latent), model.critic_2(actor_latent))
    probabilities = model.actor(actor_latent)
    log_probabilities = torch.log(probabilities.clamp_min(1e-8))
    sac_loss = (probabilities * (alpha * log_probabilities - policy_q)).sum(dim=-1).mean()
    per_action_nll = -log_probabilities.gather(1, actions.unsqueeze(1)).squeeze(1)
    bc_loss = (per_action_nll * sample_bc_weights).sum() / sample_bc_weights.sum().clamp_min(1.0)
    actor_loss = sac_loss + float(bc_weight) * bc_loss
    if not torch.isfinite(actor_loss):
        raise FloatingPointError("Non-finite SAC-CQL actor loss")
    model.actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.actor.parameters(), model.clip_grad_param)
    model.actor_optimizer.step()

    entropy = -(probabilities.detach() * log_probabilities.detach()).sum(dim=-1).mean()
    alpha_loss = (model.log_alpha * (entropy - model.target_entropy)).mean()
    model.log_alpha_optimizer.zero_grad(set_to_none=True)
    alpha_loss.backward()
    model.log_alpha_optimizer.step()
    with torch.no_grad():
        model.log_alpha.clamp_(-10.0, 2.0)
        for target_net, source_net in (
            (model.target_critic_1, model.critic_1),
            (model.target_critic_2, model.critic_2),
        ):
            for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
                target_param.lerp_(source_param, model.tau)
        if hasattr(model, 'target_temporal_encoder'):
            for target_param, param in zip(model.target_temporal_encoder.parameters(), model.temporal_encoder.parameters()):
                target_param.lerp_(param, model.tau)
    return {
        "critic_loss": float(critic_loss.detach()), "td_loss": float((td1 + td2).detach()),
        "cql_gap": float((cql1 + cql2).detach()), "actor_loss": float(actor_loss.detach()),
        "bc_loss": float(bc_loss.detach()), "alpha_loss": float(alpha_loss.detach()),
        "alpha": float(model.log_alpha.detach().exp()), "entropy": float(entropy),
    }


@torch.inference_mode()
def _evaluate_action_nll(model, dataset, device, batch_size):
    model.eval()
    total_nll = correct = seen = 0
    for start in range(0, dataset.transitions, batch_size):
        stop = min(dataset.transitions, start + batch_size)
        ids = torch.arange(start, stop)
        state_indices = dataset.state_indices[ids]
        states = dataset.features[state_indices].to(device, non_blocking=True)
        actions = dataset.actions[ids].to(device, non_blocking=True)
        latent = model.temporal_encoder(states)[0][:, -1]
        probabilities = model.actor(latent).clamp_min(1e-8)
        total_nll += F.nll_loss(torch.log(probabilities), actions, reduction="sum").item()
        correct += int((probabilities.argmax(dim=-1) == actions).sum())
        seen += len(actions)
    return {"action_nll": total_nll / max(seen, 1),
            "action_accuracy": correct / max(seen, 1), "samples": seen}


def run(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    feature_dir = run_dir / "offline/features"
    feature_manifest_path = feature_dir / "manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("status") != "completed" or feature_manifest.get("test_data_loaded") is not False:
        raise RuntimeError("Feature cache must be complete and test-free before offline training")
    audit_path = run_dir / "audit/data_source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Offline training blocked: data-source audit is not pass")
    foundation_path = run_dir / "foundation/action_stage/best.pth"
    if _sha256(foundation_path) != feature_manifest["foundation_sha256"]:
        raise ValueError("Foundation checkpoint hash differs from frozen feature cache")
    output = Path(args.output_dir).resolve() if args.output_dir else run_dir / "offline/training"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists():
        raise FileExistsError(f"Offline training output already has metrics: {output}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    train_data = TransitionFeatures(feature_dir / "train.pt", args.sequence_length,
                                    args.collision_extra_penalty, training=True,
                                    previous_action=args.previous_action, collision_bc_weight=args.collision_bc_weight)
    validation_data = {
        "heard": TransitionFeatures(feature_dir / "validation.pt", args.sequence_length, previous_action=args.previous_action),
        "unheard": TransitionFeatures(feature_dir / "validation_unheard.pt", args.sequence_length, previous_action=args.previous_action),
    }
    model = SAC_LSTM_CQL_v1_5(
        state_dim=train_data.state_dim, hidden_dim=args.hidden_dim, action_dim=4,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr,
        target_entropy=args.target_entropy, tau=args.tau, gamma=args.gamma,
        beta=args.cql_weight, device=device, lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=1, use_target_encoder=True,
    ).to(device)
    for target in (model.target_critic_1, model.target_critic_2):
        target.requires_grad_(False)
        target.eval()

    training_config = {
        "algorithm": "discrete_sac_cql_lstm_v1_5",
        "architecture": "experiment.avnav.offline_rl_v1_5_lstm.OfflineNet.SAC_LSTM_CQL_v1_5",
        "foundation_checkpoint": str(foundation_path),
        "foundation_sha256": _sha256(foundation_path),
        "feature_cache_manifest": str(feature_manifest_path),
        "feature_cache_manifest_sha256": _sha256(feature_manifest_path),
        "data_source_audit": str(audit_path), "data_source_audit_sha256": _sha256(audit_path),
        "seed": args.seed, "device": str(device), "feature_dim": train_data.feature_dim,
        "state_dim": train_data.state_dim, "previous_action": args.previous_action,
        "collision_bc_weight": args.collision_bc_weight,
        "use_target_encoder": True,
        "checkpoint_epochs": args.checkpoint_epochs,
        "sequence_length": args.sequence_length, "padding": "left-zero, reset per episode",
        "hidden_dim": args.hidden_dim, "lstm_hidden_dim": args.lstm_hidden_dim,
        "epochs": args.epochs, "updates_per_epoch": args.updates_per_epoch,
        "batch_size": args.batch_size, "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr, "alpha_lr": args.alpha_lr,
        "target_entropy": args.target_entropy, "target_tau": args.tau,
        "gamma": args.gamma, "cql_weight": args.cql_weight,
        "behavior_cloning_weight": args.bc_weight,
        "collision_reward": {
            "source_field": "recorded shaped reward (environment reward plus train-only recorded adjustments)",
            "collision_flag_source": "per-transition collision flag stored by the training collector",
            "extra_train_only_penalty": args.collision_extra_penalty,
            "formula": "reward_used = recorded_reward - extra_train_only_penalty * collision_flag",
            "validation_rewards_used_for_gradient_updates": False,
        },
        "selection_rule": "final selection by complete heard/unheard validation rollout mean SPL, then mean SR, then stable candidate/epoch order",
        "nll_diagnostic_rule": "minimum maximum validation action NLL; diagnostic only, not final model selection",
        "validation_used_for_gradient_updates": False,
        "test_data_used": False,
        "train_episodes": train_data.episode_count, "train_transitions": train_data.transitions,
        "train_recorded_collision_steps": train_data.recorded_collisions,
        "validation_episodes": {key: val.episode_count for key, val in validation_data.items()},
        "validation_transitions": {key: val.transitions for key, val in validation_data.items()},
    }
    _atomic_json(output / "training_config.json", training_config)
    _atomic_json(output / "training_protocol.json", {
        "checkpoint": "offline/best_policy.pt selected on validation only",
        "actor_action_handling": f"raw actor argmax over episode-local {args.sequence_length}-step sequence",
        "evaluation_action_postprocessing": "none",
        "feature_extractor": "frozen foundation checkpoint",
        "data_split_rule": "gradient updates use codex train episodes only; validation for selection; test untouched",
    })

    history = []
    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {}
        started = time.monotonic()
        for _ in range(args.updates_per_epoch):
            ids = torch.randint(train_data.transitions, (args.batch_size,))
            batch = train_data.batch(ids, device)
            metrics = _optimizer_update(model, *batch, args.bc_weight)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value
        train_metrics = {key: value / args.updates_per_epoch for key, value in sums.items()}
        val_metrics = {name: _evaluate_action_nll(model, data, device, args.validation_batch_size)
                       for name, data in validation_data.items()}
        score = max(row["action_nll"] for row in val_metrics.values())
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics,
               "selection_score_worst_validation_action_nll": score,
               "seconds": time.monotonic() - started}
        history.append(row)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        latest_hash = _save_state(output / "latest.pt", model)
        if epoch in args.checkpoint_epochs:
            checkpoint = output / f"epoch_{epoch:03d}.pt"
            checkpoint_hash = _save_state(checkpoint, model)
            _atomic_json(checkpoint.with_suffix('.json'), {'epoch': epoch, 'sha256': checkpoint_hash, 'validation_nll_diagnostic': val_metrics})
        _atomic_json(output / "latest.json", {"epoch": epoch, "checkpoint_sha256": latest_hash})
        if score < best_score:
            best_score = score
            best_hash = _save_state(output / "best_nll_policy.pt", model)
            _atomic_json(output / "best_nll.json", {
                "epoch": epoch, "selection_score_worst_validation_action_nll": score,
                "validation": val_metrics, "checkpoint": str(output / "best_nll_policy.pt"),
                "checkpoint_sha256": best_hash,
            })
    _atomic_json(output / "summary.json", {
        "status": "completed", "epochs": args.epochs,
        "updates": args.epochs * args.updates_per_epoch,
        "best_nll_diagnostic": json.loads((output / "best_nll.json").read_text(encoding="utf-8")),
        "training_config": str(output / "training_config.json"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="result/2026-10-27-avnav-codex-full-training")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument('--output-dir')
    parser.add_argument('--previous-action', action='store_true')
    parser.add_argument('--collision-bc-weight', type=float, default=0.0)
    parser.add_argument('--checkpoint-epochs', type=int, nargs='+', default=[10, 20, 30])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--updates-per-epoch", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lstm-hidden-dim", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument("--critic-lr", type=float, default=5e-5)
    parser.add_argument("--alpha-lr", type=float, default=1e-5)
    parser.add_argument("--target-entropy", type=float, default=1.08)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--cql-weight", type=float, default=0.5)
    parser.add_argument("--bc-weight", type=float, default=0.5)
    parser.add_argument("--collision-extra-penalty", type=float, default=0.5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
