"""Train IQL on states augmented with learned per-action collision risks."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from experiment.avnav.config import resolve_path
from experiment.avnav.model.hybrid import Network
from experiment.avnav.offline_rl import (
    FeatureDataset, _require_clean_data_audit, _sha256, train_offline_iql,
)


def train(config_path, output_override=None):
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit = _require_clean_data_audit(config)
    seed = int(config.get("SEED", 20261025))
    device = torch.device(config["FOUNDATION"]["DEVICE"])
    torch.set_num_threads(2)
    foundation_path = resolve_path(config["FOUNDATION"]["CHECKPOINT"])
    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(foundation_path, map_location=device,
                                          weights_only=True))
    foundation.eval().requires_grad_(False)
    rl = config["OFFLINE_RL"]
    output = resolve_path(output_override or rl["OUTPUT"])
    feature_root = (resolve_path(rl["FEATURE_CACHE_ROOT"]) /
                    _sha256(foundation_path)[:16])
    train_data = FeatureDataset(feature_root / "train")
    validation_data = {
        "validation": FeatureDataset(feature_root / "validation"),
        "validation_unheard": FeatureDataset(feature_root / "validation_unheard"),
    }
    for name, dataset in [("train", train_data), *validation_data.items()]:
        width = int(dataset.data["states"].shape[-1])
        if width != 264:
            raise ValueError(f"{name} collision-risk state width should be 264, got {width}")
    predictor_path = resolve_path(rl["COLLISION_PREDICTOR_CHECKPOINT"])
    predictor = torch.load(predictor_path, map_location="cpu", weights_only=True)
    if predictor.get("foundation_sha256") != _sha256(foundation_path):
        raise ValueError("Feature cache, predictor, and foundation checkpoints do not match")
    config["OFFLINE_RL"]["OUTPUT"] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    training_config = {
        "seed": seed,
        "epochs": int(rl["EPOCHS"]),
        "updates_per_epoch": int(rl["UPDATES_PER_EPOCH"]),
        "batch_size": int(rl["BATCH_SIZE"]),
        "learning_rate": float(rl["LEARNING_RATE"]),
        "gamma": float(rl["GAMMA"]),
        "target_tau": float(rl["TARGET_TAU"]),
        "hidden_dim": int(rl["HIDDEN_DIM"]),
        "device": rl["DEVICE"],
        "foundation_checkpoint": str(foundation_path),
        "action_history": True,
        "collision_history": False,
        "collision_risk_actions": True,
        "collision_predictor_checkpoint": str(predictor_path),
        "collision_predictor_sha256": _sha256(predictor_path),
        "input_schema": "embedding256+previous_action_onehot4+predicted_collision_probability_by_action4",
        "algorithm": "iql",
        "expectile": float(rl.get("EXPECTILE", 0.7)),
        "beta": float(rl.get("IQL_BETA", 3.0)),
    }
    history = train_offline_iql(train_data, validation_data, foundation,
                                training_config, output)
    best_nll = min(history, key=lambda row: row["validation"]["action_nll"])
    result = {
        "status": "completed",
        "algorithm": "iql",
        "training_config": training_config,
        "epochs": len(history),
        "minimum_validation_action_nll_epoch": best_nll["epoch"],
        "train_samples": train_data.samples,
        "validation_samples": {name: data.samples for name, data in validation_data.items()},
        "data_source_audit": audit,
        "feature_cache": str(feature_root),
        "best_policy_by_action_nll": str(output / "best_policy.pt"),
    }
    (output / "training_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.output or None), indent=2), flush=True)


if __name__ == "__main__":
    main()
