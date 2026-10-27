"""Cache AVNav features with learned per-action collision-risk probabilities."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from Dataset.dataset_avnav import get_files
from experiment.avnav.collision_predictor import ActionCollisionPredictor
from experiment.avnav.config import resolve_path
from experiment.avnav.model.hybrid import Network
from experiment.avnav.offline_rl import _read_observation


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_risk_features(data, foundation, predictor, device, batch_size=32):
    observations = data["obs"]
    actions = np.asarray(data["action_id"]).reshape(-1)
    rewards = np.asarray(data["reward"], dtype=np.float32).reshape(-1)
    dones = np.asarray(data["done"], dtype=np.bool_).reshape(-1)
    count = len(observations) - 1
    if (count < 1 or len(actions) != count or len(rewards) != count or
            len(dones) != count):
        raise ValueError("Episode obs/action/reward/done arrays have inconsistent lengths")
    actions = actions.astype(np.int64)
    if np.any((actions < 0) | (actions > 3)):
        raise ValueError("AVNav action IDs must be in [0,3]")

    audio_rows, rgb_rows, depth_rows = [], [], []
    previous_rgb = previous_depth = None
    for observation in observations:
        audio, rgb, depth = _read_observation(observation)
        if previous_rgb is None:
            previous_rgb, previous_depth = torch.zeros_like(rgb), torch.zeros_like(depth)
        audio_rows.append(audio)
        rgb_rows.append(torch.cat((previous_rgb, rgb), dim=-1))
        depth_rows.append(torch.cat((previous_depth, depth), dim=-1))
        previous_rgb, previous_depth = rgb, depth

    embeddings = []
    foundation.eval()
    predictor.eval()
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
    previous_action = torch.zeros((count + 1, 4), dtype=embeddings.dtype, device=device)
    previous_action[1:] = F.one_hot(previous_ids[1:].to(device), 4).to(embeddings.dtype)
    risks = predictor.all_action_probabilities(embeddings, previous_action).to(embeddings.dtype)
    states = torch.cat((embeddings, previous_action, risks), dim=-1)
    return {
        "states": states[:-1].cpu(),
        "next_states": states[1:].cpu(),
        "actions": torch.as_tensor(actions, dtype=torch.long),
        "rewards": torch.as_tensor(rewards, dtype=torch.float32),
        "dones": torch.as_tensor(dones, dtype=torch.float32),
    }


def build(source, destination, foundation, predictor, foundation_path,
          predictor_path, device, seed=42, batch_size=32, shard_size=8192):
    source_paths = source if isinstance(source, (list, tuple)) else [source]
    resolved_sources = [resolve_path(path) for path in source_paths]
    destination = Path(destination).resolve()
    signature = {
        "source": [str(path) for path in resolved_sources],
        "foundation_sha256": _sha256(foundation_path),
        "collision_predictor_sha256": _sha256(predictor_path),
        "feature_dim": 264,
        "action_count": 4,
        "state_schema": "embedding256+previous_action_onehot4+predicted_collision_probability_by_action4",
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all(manifest.get(key) == value for key, value in signature.items()):
            return manifest
        raise FileExistsError(f"Existing collision-risk cache has a different signature: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to reuse partial feature cache: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for root in resolved_sources:
        files.extend(get_files(root, (".pkl", ".pickle")))
    files = [Path(path).resolve() for path in files]
    if len(files) != len(set(files)):
        raise ValueError("Training sources include duplicate trajectory files")
    random.Random(seed).shuffle(files)
    buffers = {key: [] for key in ("states", "next_states", "actions", "rewards", "dones")}
    shards, samples = [], 0

    def flush():
        nonlocal samples
        if not buffers["actions"]:
            return
        name = f"transitions_{len(shards):05d}.pt"
        payload = {key: torch.cat(value, dim=0) for key, value in buffers.items()}
        torch.save(payload, destination / name)
        count = len(payload["actions"])
        shards.append({"file": name, "samples": count})
        samples += count
        for rows in buffers.values():
            rows.clear()

    foundation.eval().requires_grad_(False)
    predictor.eval().requires_grad_(False)
    for path in tqdm(files, desc=f"Risk features: {destination.name}", mininterval=10):
        with path.open("rb") as handle:
            episode = pickle.load(handle)
        rows = episode_risk_features(episode, foundation, predictor, device, batch_size)
        for key in buffers:
            buffers[key].append(rows[key])
        if sum(len(item) for item in buffers["actions"]) >= shard_size:
            flush()
    flush()
    manifest = {**signature, "episodes": len(files), "samples": samples,
                "shards": shards}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run(config_path, foundation_path, predictor_path, output_root, device="cuda:0",
        batch_size=32, shard_size=8192):
    config = yaml.safe_load(Path(config_path).read_text())
    train_sources = config["DATA"]["TRAIN_EPISODES"]
    train_sources = train_sources if isinstance(train_sources, list) else [train_sources]
    validation_sources = {
        "validation": config["DATA"]["VALIDATION_EPISODES"],
        "validation_unheard": config["DATA"]["VALIDATION_UNHEARD_EPISODES"],
    }
    device = torch.device(device)
    foundation_path = resolve_path(foundation_path)
    predictor_path = resolve_path(predictor_path)
    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(foundation_path, map_location=device, weights_only=True))
    predictor_state = torch.load(predictor_path, map_location="cpu", weights_only=True)
    if predictor_state.get("foundation_sha256") != _sha256(foundation_path):
        raise ValueError("Collision predictor and foundation checkpoint hashes differ")
    predictor = ActionCollisionPredictor().to(device)
    predictor.load_state_dict(predictor_state["predictor"])
    cache = Path(output_root).resolve() / _sha256(foundation_path)[:16]
    cache.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "foundation_checkpoint": str(foundation_path),
        "foundation_sha256": _sha256(foundation_path),
        "collision_predictor_checkpoint": str(predictor_path),
        "collision_predictor_sha256": _sha256(predictor_path),
        "cache_root": str(cache),
        "train": build(train_sources, cache / "train", foundation, predictor,
                        foundation_path, predictor_path, device,
                        batch_size=batch_size, shard_size=shard_size),
        "validation": build(validation_sources["validation"], cache / "validation",
                            foundation, predictor, foundation_path, predictor_path,
                            device, batch_size=batch_size, shard_size=shard_size),
        "validation_unheard": build(validation_sources["validation_unheard"],
                                    cache / "validation_unheard", foundation, predictor,
                                    foundation_path, predictor_path, device,
                                    batch_size=batch_size, shard_size=shard_size),
    }
    result["status"] = "completed"
    summary_path = cache / "feature_build_summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--foundation", required=True)
    parser.add_argument("--collision-predictor", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=8192)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.foundation, args.collision_predictor,
                         args.output_root, args.device, args.batch_size,
                         args.shard_size), indent=2), flush=True)


if __name__ == "__main__":
    main()
