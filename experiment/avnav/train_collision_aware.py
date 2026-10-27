"""Train IQL with replay and reward penalties from train-only collision transitions."""
import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.avnav.config import resolve_path
from experiment.avnav.model.hybrid import Network
from experiment.avnav.offline_rl import (
    FeatureDataset, _require_clean_data_audit, _sha256, train_offline_iql,
)


FEATURE_KEYS = ("states", "next_states", "actions", "rewards", "dones")


def _repeat_selected(data, mask, repeats, reward_penalty=0.0):
    selected = {key: data[key][mask].contiguous() for key in FEATURE_KEYS}
    if not len(selected["actions"]) or repeats < 1:
        return None
    if reward_penalty:
        selected["rewards"] = selected["rewards"] - float(reward_penalty)
    return {key: value.repeat((repeats,) + (1,) * (value.ndim - 1))
            for key, value in selected.items()}


def _append_replay(base, replay, collision_repeat, post_collision_repeat,
                   collision_penalty, action_history=False, collision_history=False,
                   successful_post_collision_only=False):
    collision = replay["collision"].bool()
    post_collision = replay["post_collision"].bool() & ~collision
    replay_states = replay["states"]
    replay_next_states = replay["next_states"]
    if action_history and replay_states.shape[-1] == 256:
        previous_actions = replay["previous_actions"].long()
        previous_one_hot = torch.zeros((len(previous_actions), 4), dtype=torch.float32)
        valid = previous_actions >= 0
        if valid.any():
            previous_one_hot[valid] = torch.nn.functional.one_hot(
                previous_actions[valid], 4).float()
        next_one_hot = torch.nn.functional.one_hot(replay["actions"].long(), 4).float()
        replay_states = torch.cat((replay_states, previous_one_hot), dim=-1)
        replay_next_states = torch.cat((replay_next_states, next_one_hot), dim=-1)
    elif action_history and replay_states.shape[-1] not in (260, 261):
        raise ValueError(f"Replay feature width incompatible with action history: {replay_states.shape[-1]}")
    if collision_history:
        current_collision = post_collision.float().unsqueeze(-1)
        next_collision = collision.float().unsqueeze(-1)
        replay_states = torch.cat((replay_states, current_collision), dim=-1)
        replay_next_states = torch.cat((replay_next_states, next_collision), dim=-1)
    replay = dict(replay)
    replay["states"], replay["next_states"] = replay_states, replay_next_states
    if successful_post_collision_only:
        selected = _repeat_selected(replay, post_collision, post_collision_repeat)
        pieces = [selected] if selected is not None else []
    else:
        pieces = [piece for piece in (
            _repeat_selected(replay, collision, collision_repeat, collision_penalty),
            _repeat_selected(replay, post_collision, post_collision_repeat),
        ) if piece is not None]
    if not pieces:
        raise ValueError("Collision replay has no colliding or post-collision transitions")
    augmented = {key: torch.cat([piece[key] for piece in pieces]) for key in FEATURE_KEYS}
    result = {key: torch.cat((base.data[key], augmented[key])) for key in FEATURE_KEYS}
    data = FeatureDataset.__new__(FeatureDataset)
    data.data = result
    data.samples = len(result["actions"])
    return data, {"base_transitions": base.samples,
                  "collision_transitions": int(collision.sum().item()),
                  "post_collision_transitions": int(post_collision.sum().item()),
                  "collision_repeats": int(collision_repeat),
                  "post_collision_repeats": int(post_collision_repeat),
                  "collision_reward_penalty": float(collision_penalty),
                  "action_history": bool(action_history),
                  "collision_history": bool(collision_history),
                  "successful_post_collision_only": bool(successful_post_collision_only),
                  "augmented_transitions": len(augmented["actions"]),
                  "total_training_transitions": data.samples}


def run(config_path, replay_path, output_override=None, collision_repeat=8,
        post_collision_repeat=4, collision_penalty=1.0,
        successful_post_collision_only=False):
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_audit = _require_clean_data_audit(config)
    seed = int(config.get("SEED", 42))
    foundation_path = resolve_path(config["FOUNDATION"]["CHECKPOINT"])
    device = torch.device(config["FOUNDATION"]["DEVICE"])
    torch.set_num_threads(2)
    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(foundation_path, map_location=device, weights_only=True))
    foundation.eval().requires_grad_(False)

    rl = config["OFFLINE_RL"]
    output = (resolve_path(output_override) if output_override
              else resolve_path(rl["OUTPUT"]))
    cache_root = resolve_path(rl["FEATURE_CACHE_ROOT"])
    feature_root = cache_root / _sha256(foundation_path)[:16]
    base = FeatureDataset(feature_root / "train")
    validation = {name: FeatureDataset(feature_root / name)
                  for name in ("validation", "validation_unheard")
                  if (feature_root / name / "manifest.json").is_file()}
    if not validation:
        raise FileNotFoundError(f"Validation feature caches are missing under {feature_root}")
    if len(validation) == 1:
        validation = next(iter(validation.values()))

    replay = torch.load(resolve_path(replay_path), map_location="cpu", weights_only=False)
    train, augmentation = _append_replay(base, replay, collision_repeat,
                                         post_collision_repeat, collision_penalty,
                                         action_history=bool(rl.get("ACTION_HISTORY", False)),
                                         collision_history=bool(rl.get("COLLISION_HISTORY", False)),
                                         successful_post_collision_only=successful_post_collision_only)
    config["OFFLINE_RL"]["OUTPUT"] = str(output.resolve())
    training_config = dict(
        seed=seed, epochs=int(rl["EPOCHS"]),
        updates_per_epoch=int(rl["UPDATES_PER_EPOCH"]),
        batch_size=int(rl["BATCH_SIZE"]), learning_rate=float(rl["LEARNING_RATE"]),
        gamma=float(rl["GAMMA"]), target_tau=float(rl["TARGET_TAU"]),
        hidden_dim=int(rl["HIDDEN_DIM"]), device=rl["DEVICE"],
        foundation_checkpoint=str(foundation_path), algorithm="IQL",
        expectile=float(rl.get("EXPECTILE", 0.7)), beta=float(rl.get("IQL_BETA", 3.0)),
        collision_augmentation=augmentation,
        action_history=bool(rl.get("ACTION_HISTORY", False)),
        collision_history=bool(rl.get("COLLISION_HISTORY", False)),
        collision_replay=str(resolve_path(replay_path)),
        successful_post_collision_only=bool(successful_post_collision_only),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "augmentation.json").write_text(
        json.dumps(augmentation, indent=2) + "\n", encoding="utf-8")
    history = train_offline_iql(train, validation, foundation, training_config, output)
    best = min(history, key=lambda row: row["validation"]["action_nll"])
    summary = dict(status="completed", algorithm="iql", training_config=training_config,
                   epochs=len(history), best_epoch=best["epoch"],
                   best_validation=best["validation"], train_samples=train.samples,
                   validation_samples=(
                       {name: item.samples for name, item in validation.items()}
                       if isinstance(validation, dict) else validation.samples),
                   foundation_checkpoint=str(foundation_path),
                   data_source_audit=source_audit,
                   best_policy=str(output / "best_policy.pt"))
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--collision-repeat", type=int, default=24)
    parser.add_argument("--post-collision-repeat", type=int, default=4,
                        help="Number of extra copies of the recorded next transition")
    parser.add_argument("--collision-penalty", type=float, default=1.0)
    parser.add_argument("--successful-post-collision-only", action="store_true",
                        help="Use only non-colliding post-collision actions from successful train rollouts")
    args = parser.parse_args(argv)
    result = run(args.config, args.replay, args.output or None,
                 args.collision_repeat, args.post_collision_repeat, args.collision_penalty,
                 args.successful_post_collision_only)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
