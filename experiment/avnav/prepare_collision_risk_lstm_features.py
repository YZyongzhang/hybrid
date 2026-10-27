"""Build episode-bounded LSTM windows for the learned collision-risk state."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random

import torch
import yaml
from tqdm import tqdm

from Dataset.dataset_avnav import get_files
from experiment.avnav.collision_predictor import ActionCollisionPredictor
from experiment.avnav.config import resolve_path
from experiment.avnav.model.hybrid import Network
from experiment.avnav.prepare_collision_risk_features import episode_risk_features


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_windows(transitions, length):
    states = transitions["states"].float()
    next_rows = transitions["next_states"].float()
    count, width = states.shape
    current = states.new_zeros((count, length, width))
    for offset in range(length):
        if offset < count:
            current[offset:, length - 1 - offset] = states[:count - offset]
    following = torch.empty_like(current)
    following[:, :-1] = current[:, 1:]
    following[:, -1] = next_rows
    result = dict(transitions)
    result["states"] = current.to(torch.float16)
    result["next_states"] = following.to(torch.float16)
    return result


def _build(source, destination, foundation, predictor, foundation_path,
           predictor_path, device, length=5, batch_size=32, shard_size=8192,
           seed=42):
    source_paths = source if isinstance(source, (list, tuple)) else [source]
    resolved_sources = [resolve_path(item) for item in source_paths]
    destination = Path(destination).resolve()
    signature = {
        "source": [str(path) for path in resolved_sources],
        "foundation_sha256": _sha256(foundation_path),
        "collision_predictor_sha256": _sha256(predictor_path),
        "feature_dim": 264,
        "sequence_length": int(length),
        "padding": "left-zero; windows reset at each episode boundary",
        "state_schema": "embedding256+previous_action_onehot4+predicted_collision_probability_by_action4",
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in signature.items()):
            return manifest
        raise FileExistsError(f"Existing sequence cache has a different signature: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to reuse partial sequence cache: {destination}")
    files = []
    for root in resolved_sources:
        files.extend(get_files(root, (".pkl", ".pickle")))
    files = [Path(path).resolve() for path in files]
    if len(files) != len(set(files)):
        raise ValueError("Training sources include duplicate trajectory files")
    random.Random(seed).shuffle(files)
    destination.mkdir(parents=True, exist_ok=True)
    keys = ("states", "next_states", "actions", "rewards", "dones")
    buffers = {key: [] for key in keys}
    shards, samples = [], 0

    def flush():
        nonlocal samples
        if not buffers["actions"]:
            return
        name = f"sequences_{len(shards):05d}.pt"
        payload = {key: torch.cat(rows, dim=0) for key, rows in buffers.items()}
        torch.save(payload, destination / name)
        count = len(payload["actions"])
        shards.append({"file": name, "samples": count})
        samples += count
        for rows in buffers.values():
            rows.clear()

    foundation.eval().requires_grad_(False)
    predictor.eval().requires_grad_(False)
    for path in tqdm(files, desc=f"LSTM risk features: {destination.name}", mininterval=10):
        with path.open("rb") as handle:
            episode = pickle.load(handle)
        windows = _make_windows(
            episode_risk_features(episode, foundation, predictor, device, batch_size), length
        )
        for key in keys:
            buffers[key].append(windows[key])
        if sum(len(rows) for rows in buffers["actions"]) >= shard_size:
            flush()
    flush()
    manifest = {**signature, "episodes": len(files), "samples": samples, "shards": shards}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run(config_path, output_root, sequence_length=5, device="cuda:0"):
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rl = config["OFFLINE_RL"]
    foundation_path = resolve_path(config["FOUNDATION"]["CHECKPOINT"])
    predictor_path = resolve_path(rl["COLLISION_PREDICTOR_CHECKPOINT"])
    device = torch.device(device)
    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(foundation_path, map_location=device,
                                          weights_only=True))
    predictor_state = torch.load(predictor_path, map_location="cpu", weights_only=True)
    if predictor_state.get("foundation_sha256") != _sha256(foundation_path):
        raise ValueError("Collision predictor and foundation checkpoint hashes differ")
    predictor = ActionCollisionPredictor().to(device)
    predictor.load_state_dict(predictor_state["predictor"])
    root = Path(output_root).expanduser().resolve() / _sha256(foundation_path)[:16]
    root.mkdir(parents=True, exist_ok=True)
    source_splits = {
        "train": config["DATA"]["TRAIN_EPISODES"],
        "validation": config["DATA"]["VALIDATION_EPISODES"],
        "validation_unheard": config["DATA"]["VALIDATION_UNHEARD_EPISODES"],
    }
    result = {
        "foundation_checkpoint": str(foundation_path),
        "foundation_sha256": _sha256(foundation_path),
        "collision_predictor_checkpoint": str(predictor_path),
        "collision_predictor_sha256": _sha256(predictor_path),
        "sequence_length": int(sequence_length),
        "cache_root": str(root),
    }
    for split, sources in source_splits.items():
        result[split] = _build(
            sources, root / split, foundation, predictor, foundation_path,
            predictor_path, device, length=sequence_length,
            batch_size=int(config["FOUNDATION"]["EMBED_BATCH_SIZE"]),
            shard_size=int(config["FOUNDATION"]["FEATURE_SHARD_SIZE"]),
            seed=int(config.get("SEED", 42)),
        )
    result["status"] = "completed"
    (root / "feature_build_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_root, args.sequence_length, args.device),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
