"""Build episode-local CQL-LSTM features, including train counterfactual anchors."""

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from Dataset.dataset_avnav import get_files
from experiment.avnav.collision_predictor import (
    ActionCollisionPredictor,
    MultiHorizonActionCollisionPredictor,
)
from experiment.avnav.config import resolve_path
from experiment.avnav.model.hybrid import Network
from experiment.avnav.offline_rl import (
    _collision_labels,
    _read_observation,
    _require_clean_data_audit,
)
from experiment.avnav.safety_features import (
    depth_sector_quantiles,
    previous_collision_features,
    state_progress_delta,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(sources):
    sources = sources if isinstance(sources, (list, tuple)) else [sources]
    result = []
    for source in sources:
        root = resolve_path(source)
        result.extend(Path(item).resolve() for item in get_files(root, (".pkl", ".pickle")))
    if not result:
        raise FileNotFoundError(f"No trajectory pickles found under {sources}")
    if len(result) != len(set(result)):
        raise ValueError("A training/validation feature source includes duplicate file paths")
    return sorted(result)


def _episode_key(path, episode):
    scene = episode.get("parent_scene", path.parent.name)
    episode_id = episode.get("parent_episode_id", path.stem)
    return str(scene), str(episode_id)


def _distance_trace(episode, count):
    explicit = episode.get("distance_to_goal")
    if explicit is not None:
        values = np.asarray(explicit, dtype=np.float32).reshape(-1)
        if len(values) == count + 1:
            return values.tolist()
    info = episode.get("info")
    if info is not None and len(info) >= count:
        post = []
        for row in info[:count]:
            value = None
            if hasattr(row, "get"):
                value = row.get("distance_to_goal")
            elif isinstance(row, dict):
                value = row.get("distance_to_goal")
            post.append(float(value) if value is not None else np.nan)
        finite = [value for value in post if np.isfinite(value)]
        fallback = finite[0] if finite else 0.0
        post = [value if np.isfinite(value) else fallback for value in post]
        return [post[0] if post else 0.0, *post]
    return [0.0] * (count + 1)


def _sequence_window(states, state_index, length):
    width = states.shape[-1]
    result = torch.zeros((length, width), dtype=states.dtype)
    start = max(0, state_index - length + 1)
    selected = states[start:state_index + 1]
    result[-len(selected):] = selected
    return result


def _transition_windows(states, length):
    count = len(states) - 1
    current = torch.stack([_sequence_window(states, index, length)
                           for index in range(count)])
    following = torch.stack([_sequence_window(states, index + 1, length)
                             for index in range(count)])
    return current, following


def _features(episode, foundation, predictor, device, cfg, batch_size):
    observations = episode["obs"]
    actions = np.asarray(episode["action_id"]).reshape(-1).astype(np.int64)
    count = len(observations) - 1
    if count < 1 or len(actions) != count:
        raise ValueError("Episode observation/action arrays have inconsistent lengths")
    if np.any((actions < 0) | (actions > 3)):
        raise ValueError("AVNav action IDs must be in [0,3]")
    rewards = np.asarray(episode["reward"], dtype=np.float32).reshape(-1)
    dones = np.asarray(episode["done"], dtype=np.bool_).reshape(-1)
    if len(rewards) != count or len(dones) != count:
        raise ValueError("Episode reward/done arrays have inconsistent lengths")
    collision = _collision_labels(episode, actions)
    if len(collision) != count:
        raise ValueError("Collision labels and actions have inconsistent lengths")

    audio_rows, rgb_rows, depth_rows = [], [], []
    previous_rgb = previous_depth = None
    geometries = []
    for observation in observations:
        audio, rgb, depth = _read_observation(observation)
        if previous_rgb is None:
            previous_rgb, previous_depth = torch.zeros_like(rgb), torch.zeros_like(depth)
        audio_rows.append(audio)
        rgb_rows.append(torch.cat((previous_rgb, rgb), dim=-1))
        depth_rows.append(torch.cat((previous_depth, depth), dim=-1))
        geometries.append(depth_sector_quantiles(observation))
        previous_rgb, previous_depth = rgb, depth

    embeddings = []
    foundation.eval()
    with torch.inference_mode():
        for start in range(0, count + 1, batch_size):
            stop = min(count + 1, start + batch_size)
            embeddings.append(foundation.embedding_forward(
                torch.stack(audio_rows[start:stop]).to(device),
                torch.stack(rgb_rows[start:stop]).to(device),
                torch.stack(depth_rows[start:stop]).to(device),
            ).float())
    embeddings = torch.cat(embeddings)
    previous_ids = torch.zeros(count + 1, dtype=torch.long)
    previous_ids[1:] = torch.as_tensor(actions, dtype=torch.long)
    previous_actions = F.one_hot(previous_ids.to(device), 4).to(embeddings.dtype)
    previous_actions[0].zero_()

    pieces = [embeddings]
    if cfg.get("ACTION_HISTORY", True):
        pieces.append(previous_actions)
    if cfg.get("COLLISION_HISTORY", True):
        pieces.append(torch.tensor(previous_collision_features(collision),
                                   dtype=embeddings.dtype, device=device)[:, None])
    distance_trace = _distance_trace(episode, count)
    if cfg.get("DISTANCE_DELTA_FEATURE", True):
        pieces.append(torch.tensor(state_progress_delta(distance_trace),
                                   dtype=embeddings.dtype, device=device)[:, None])
    if cfg.get("LOCAL_GEOMETRY", True):
        pieces.append(torch.tensor(geometries, dtype=embeddings.dtype, device=device))

    predictor.eval()
    risks = predictor.all_action_probabilities(embeddings, previous_actions)
    pieces.append(risks.to(embeddings.dtype))
    if cfg.get("FUTURE_COLLISION_RISK_ACTIONS", False):
        pieces.append(predictor.all_future_action_probabilities(
            embeddings, previous_actions).to(embeddings.dtype))
    states = torch.cat(pieces, dim=-1).cpu()
    current, following = _transition_windows(states, int(cfg.get("SEQUENCE_LENGTH", 5)))
    return {
        "states": current.to(torch.float16),
        "next_states": following.to(torch.float16),
        "actions": torch.from_numpy(actions).long(),
        "rewards": torch.from_numpy(rewards).float(),
        "dones": torch.from_numpy(dones.astype(np.float32)),
        "state_features": states,
        "collision": torch.from_numpy(collision.astype(np.float32)),
        "distance_trace": torch.tensor(distance_trace, dtype=torch.float32),
    }


def _read_counterfactual_labels(label_root):
    groups = {}
    for path in sorted(Path(label_root).rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["scene"]), str(row["episode_id"]), int(row["anchor_step"]))
            groups.setdefault(key, {})[int(row["candidate_action_id"])] = row
    for key, actions in groups.items():
        if set(actions) != {0, 1, 2, 3}:
            raise ValueError(f"Counterfactual anchor {key} lacks the complete four-action set")
    return groups


def _append_shard(destination, buffers, shard_rows, shard_size, split):
    if not buffers["actions"]:
        return 0
    index = len(shard_rows)
    name = f"sequences_{index:05d}.pt"
    payload = {key: torch.cat(rows, dim=0) for key, rows in buffers.items()}
    torch.save(payload, destination / name)
    samples = len(payload["actions"])
    shard_rows.append({"file": name, "samples": samples})
    for values in buffers.values():
        values.clear()
    return samples


def _build_split(sources, destination, foundation, predictor, device, cfg,
                 foundation_path, predictor_path, split_name,
                 counterfactual_groups=None, label_root=None,
                 batch_size=32, shard_size=8192, seed=20261025):
    paths = _paths(sources)
    destination = Path(destination).resolve()
    signature = {
        "split": split_name, "sources": [str(path) for path in paths],
        "foundation_sha256": _sha256(foundation_path),
        "collision_predictor_sha256": _sha256(predictor_path),
        "feature_flags": {key: cfg.get(key) for key in (
            "ACTION_HISTORY", "COLLISION_HISTORY", "DISTANCE_DELTA_FEATURE",
            "LOCAL_GEOMETRY", "FUTURE_COLLISION_RISK_ACTIONS", "SEQUENCE_LENGTH")},
        "state_schema": "embedding256+previous_action4+previous_collision1+distance_delta1+depth_sector5+immediate_risk4+optional_future_risk4",
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(previous.get(key) == value for key, value in signature.items()):
            return previous
        raise FileExistsError(f"Feature cache exists with a different signature: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to reuse incomplete feature cache: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    random.Random(seed).shuffle(paths)
    keys = ("states", "next_states", "actions", "rewards", "dones")
    buffers = {key: [] for key in keys}
    shard_rows, samples = [], 0
    counterfactual_states, counterfactual_rows = [], []
    matched_groups = set()
    for path in tqdm(paths, desc=f"Safety features {split_name}", mininterval=10):
        with path.open("rb") as handle:
            episode = pickle.load(handle)
        features = _features(episode, foundation, predictor, device, cfg, batch_size)
        for key in keys:
            buffers[key].append(features[key])
        if sum(len(rows) for rows in buffers["actions"]) >= shard_size:
            samples += _append_shard(destination, buffers, shard_rows, shard_size, split_name)

        if counterfactual_groups:
            scene, episode_id = _episode_key(path, episode)
            for group_key, action_rows in counterfactual_groups.items():
                if group_key[:2] != (scene, episode_id):
                    continue
                anchor_step = group_key[2]
                if not 0 <= anchor_step < len(features["state_features"]) - 1:
                    continue
                counterfactual_states.append(_sequence_window(
                    features["state_features"], anchor_step,
                    int(cfg.get("SEQUENCE_LENGTH", 5))).to(torch.float16))
                costs = [float(action_rows[action_id]["future_collision_count"]) /
                         max(int(action_rows[action_id]["future_horizon_requested"]), 1)
                         for action_id in range(4)]
                immediate = [float(action_rows[action_id]["immediate_collision"])
                             for action_id in range(4)]
                future = [float(action_rows[action_id]["future_any_collision_after_first"])
                          for action_id in range(4)]
                progress = [float(action_rows[action_id]["movement_progress_m"])
                            for action_id in range(4)]
                counterfactual_rows.append({
                    "scene": scene, "episode_id": episode_id,
                    "anchor_step": anchor_step,
                    "event_tags": action_rows[0]["event_tags"],
                    "costs": costs, "immediate_collision": immediate,
                    "future_collision": future, "movement_progress_m": progress,
                })
                matched_groups.add(group_key)
    samples += _append_shard(destination, buffers, shard_rows, shard_size, split_name)
    manifest = {**signature, "episodes": len(paths), "samples": samples,
                "sequence_length": int(cfg.get("SEQUENCE_LENGTH", 5)),
                "feature_dim": int(shard_rows[0]["samples"] >= 0) and
                               int(torch.load(destination / shard_rows[0]["file"],
                                              map_location="cpu", weights_only=True)["states"].shape[-1]),
                "shards": shard_rows}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if counterfactual_groups is not None:
        counterfactual = {
            "states": (torch.stack(counterfactual_states) if counterfactual_states else
                       torch.empty(0, int(cfg.get("SEQUENCE_LENGTH", 5)), manifest["feature_dim"])),
            "costs": torch.tensor([row["costs"] for row in counterfactual_rows],
                                   dtype=torch.float32),
            "immediate_collision": torch.tensor(
                [row["immediate_collision"] for row in counterfactual_rows], dtype=torch.float32),
            "future_collision": torch.tensor(
                [row["future_collision"] for row in counterfactual_rows], dtype=torch.float32),
            "movement_progress_m": torch.tensor(
                [row["movement_progress_m"] for row in counterfactual_rows], dtype=torch.float32),
            "rows": counterfactual_rows,
            "label_root": str(Path(label_root).resolve()) if label_root else None,
            "matched_anchor_groups": len(matched_groups),
            "unmatched_anchor_groups": [list(key) for key in
                                        sorted(set(counterfactual_groups) - matched_groups)],
        }
        torch.save(counterfactual, destination.parent / "counterfactual_anchors.pt")
    return manifest


def build(config_path, output_root, predictor_path=None, sequence_length=5,
          device="cuda:0", include_counterfactual=False):
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit = _require_clean_data_audit(config)
    foundation_path = resolve_path(config["FOUNDATION"]["CHECKPOINT"])
    predictor_path = resolve_path(predictor_path or config["OFFLINE_RL"]["COLLISION_PREDICTOR_CHECKPOINT"])
    device = torch.device(device)
    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(foundation_path, map_location=device,
                                          weights_only=True))
    foundation.eval().requires_grad_(False)
    predictor_state = torch.load(predictor_path, map_location="cpu", weights_only=True)
    if predictor_state.get("foundation_sha256") != _sha256(foundation_path):
        raise ValueError("Collision predictor and foundation checkpoint hashes differ")
    feature_flags = config["FEATURES"]
    predictor_class = (MultiHorizonActionCollisionPredictor
                       if feature_flags.get("FUTURE_COLLISION_RISK_ACTIONS", False)
                       else ActionCollisionPredictor)
    predictor = predictor_class().to(device)
    predictor.load_state_dict(predictor_state["predictor"])
    predictor.eval().requires_grad_(False)
    feature_config = dict(feature_flags, SEQUENCE_LENGTH=int(sequence_length))
    output_root = Path(output_root).expanduser().resolve()
    signature = hashlib.sha256(
        (config_path.read_bytes() + Path(predictor_path).read_bytes()).strip()
    ).hexdigest()[:16]
    root = output_root / signature
    root.mkdir(parents=True, exist_ok=True)
    data = config["DATA"]
    result = {
        "status": "running", "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "data_source_audit": audit,
        "foundation_checkpoint": str(foundation_path),
        "foundation_sha256": _sha256(foundation_path),
        "collision_predictor_checkpoint": str(predictor_path),
        "collision_predictor_sha256": _sha256(predictor_path),
        "sequence_length": int(sequence_length), "cache_root": str(root),
    }
    counterfactual_groups = None
    label_root = None
    if include_counterfactual:
        label_root = resolve_path(data["COUNTERFACTUAL_LABEL_ROOT"])
        counterfactual_groups = _read_counterfactual_labels(label_root)
    splits = {
        "train": data["TRAIN_EPISODES"],
        "validation": data["VALIDATION_EPISODES"],
        "validation_unheard": data["VALIDATION_UNHEARD_EPISODES"],
    }
    for name, sources in splits.items():
        result[name] = _build_split(
            sources, root / name, foundation, predictor, device, feature_config,
            foundation_path, predictor_path, name,
            counterfactual_groups if name == "train" else None,
            label_root if name == "train" else None,
            batch_size=int(config["FOUNDATION"].get("EMBED_BATCH_SIZE", 32)),
            shard_size=int(config["FOUNDATION"].get("FEATURE_SHARD_SIZE", 8192)),
            seed=int(config.get("SEED", 20261025)),
        )
    result["status"] = "completed"
    (root / "feature_build_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--predictor", default="")
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-counterfactual", action="store_true")
    args = parser.parse_args()
    result = build(args.config, args.output_root, args.predictor or None,
                   args.sequence_length, args.device, args.include_counterfactual)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
