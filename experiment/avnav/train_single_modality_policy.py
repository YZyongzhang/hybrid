"""Train one frozen-feature SAC-CQL-LSTM policy on train episodes only."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from experiment.avnav.train_codex_saccql_lstm import (
    TransitionFeatures, _atomic_json, _optimizer_update, _save_state,
)
from experiment.avnav.offline_rl_v1_5_lstm.OfflineNet import SAC_LSTM_CQL_v1_5


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    run_dir = Path(args.run_dir).resolve()
    feature_dir = run_dir / "offline/single_modality" / args.modality / "features"
    feature_manifest_path = feature_dir / "manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if (feature_manifest.get("status") != "completed"
            or feature_manifest.get("modality") != args.modality
            or feature_manifest.get("validation_data_loaded") is not False
            or feature_manifest.get("test_data_loaded") is not False
            or feature_manifest.get("other_modality_encoder_called") is not False):
        raise RuntimeError("Expected a completed train-only feature cache from one encoder branch")
    if _sha256(feature_manifest["output"]["path"]) != feature_manifest["output"]["sha256"]:
        raise RuntimeError("Train feature cache hash changed after extraction")
    audit_path = run_dir / "audit/data_source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Offline training blocked by failed source provenance audit")
    foundation_path = run_dir / "foundation/action_stage/best.pth"
    if _sha256(foundation_path) != feature_manifest["foundation_sha256"]:
        raise RuntimeError("Feature cache uses a different foundation checkpoint")

    output = run_dir / "offline/single_modality" / args.modality / "training"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists():
        raise FileExistsError(f"Training output already exists: {output}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(args.cpu_threads)

    train_data = TransitionFeatures(
        feature_manifest["output"]["path"],
        sequence_length=args.sequence_length,
        collision_extra_penalty=args.collision_extra_penalty,
        training=True,
        previous_action=False,
        collision_bc_weight=args.collision_bc_weight,
    )
    if train_data.feature_dim != feature_manifest["feature_dim"]:
        raise ValueError("Cached feature dimension differs from the feature manifest")

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
        "algorithm": "discrete_sac_cql_lstm_v1_5_single_modality",
        "policy_modality": args.modality,
        "architecture": "SAC_LSTM_CQL_v1_5 over one frozen 128-dimensional encoder branch",
        "foundation_checkpoint": str(foundation_path),
        "foundation_sha256": _sha256(foundation_path),
        "feature_cache_manifest": str(feature_manifest_path),
        "feature_cache_manifest_sha256": _sha256(feature_manifest_path),
        "feature_cache": feature_manifest["output"]["path"],
        "feature_cache_sha256": feature_manifest["output"]["sha256"],
        "data_source_audit": str(audit_path),
        "data_source_audit_sha256": _sha256(audit_path),
        "seed": args.seed, "device": str(device),
        "feature_dim": train_data.feature_dim, "state_dim": train_data.state_dim,
        "previous_action": False, "collision_bc_weight": args.collision_bc_weight,
        "use_target_encoder": True,
        "checkpoint_epochs": args.checkpoint_epochs,
        "epochs": args.epochs, "updates_per_epoch": args.updates_per_epoch,
        "batch_size": args.batch_size, "sequence_length": args.sequence_length,
        "hidden_dim": args.hidden_dim, "lstm_hidden_dim": args.lstm_hidden_dim,
        "actor_lr": args.actor_lr, "critic_lr": args.critic_lr, "alpha_lr": args.alpha_lr,
        "target_entropy": args.target_entropy, "target_tau": args.tau,
        "gamma": args.gamma, "cql_weight": args.cql_weight,
        "behavior_cloning_weight": args.bc_weight,
        "collision_reward": {
            "source_field": "recorded train reward",
            "collision_flag_source": "per-transition training collector collision flag",
            "extra_train_only_penalty": args.collision_extra_penalty,
            "formula": "reward_used = recorded_reward - extra_train_only_penalty * collision_flag",
        },
        "selection_rule": "predeclared final epoch 30; no validation data used during training or checkpoint selection",
        "train_episodes": train_data.episode_count,
        "train_transitions": train_data.transitions,
        "train_recorded_collision_steps": train_data.recorded_collisions,
        "validation_data_loaded": False, "test_data_used": False,
    }
    _atomic_json(output / "training_config.json", training_config)
    _atomic_json(output / "training_protocol.json", {
        "checkpoint": "epoch_030.pt; fixed final epoch declared before validation evaluation",
        "encoder": feature_manifest["feature_extraction"],
        "encoder_frozen": True,
        "other_modality_encoder_called": False,
        "action_handling": "deterministic actor argmax at validation; no action post-processing",
        "training_sources": "audited train split trajectories only",
        "validation_or_test_used_for_gradient_updates": False,
    })

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {}
        started = time.monotonic()
        for _ in range(args.updates_per_epoch):
            indices = torch.randint(train_data.transitions, (args.batch_size,))
            metrics = _optimizer_update(model, *train_data.batch(indices, device), args.bc_weight)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value
        row = {
            "epoch": epoch,
            "train": {key: value / args.updates_per_epoch for key, value in sums.items()},
            "seconds": time.monotonic() - started,
        }
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({"modality": args.modality, **row}), flush=True)
        latest_hash = _save_state(output / "latest.pt", model)
        if epoch in args.checkpoint_epochs:
            checkpoint = output / f"epoch_{epoch:03d}.pt"
            checkpoint_hash = _save_state(checkpoint, model)
            _atomic_json(checkpoint.with_suffix(".json"), {
                "epoch": epoch, "sha256": checkpoint_hash, "checkpoint_rule": "fixed epoch",
            })
        _atomic_json(output / "latest.json", {"epoch": epoch, "sha256": latest_hash})

    final_checkpoint = output / f"epoch_{args.epochs:03d}.pt"
    if not final_checkpoint.exists():
        raise RuntimeError("Predeclared final checkpoint was not saved")
    best_policy = output / "best_policy.pt"
    best_policy.write_bytes(final_checkpoint.read_bytes())
    _atomic_json(output / "summary.json", {
        "status": "completed", "modality": args.modality,
        "epochs": args.epochs, "updates": args.epochs * args.updates_per_epoch,
        "checkpoint": str(final_checkpoint), "checkpoint_sha256": _sha256(final_checkpoint),
        "training_config": str(output / "training_config.json"),
        "validation_data_used_for_training_or_selection": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--modality", choices=["vision_only", "audio_only"], required=True)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--updates-per-epoch", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
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
    parser.add_argument("--collision-bc-weight", type=float, default=1.0)
    parser.add_argument("--collision-extra-penalty", type=float, default=0.5)
    parser.add_argument("--checkpoint-epochs", type=int, nargs="+", default=[10, 20, 30])
    run(parser.parse_args())
