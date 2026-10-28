"""Extract frozen AVNav foundation features from train and validation episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from experiment.avnav.hybrid.foundation_model import Network


CODEX_ROOT = Path("/data2/getuanhui/avnav-codex")
VAL_PICKLES = {
    "validation": Path("/data/Getuanhui/avnav-hybrid-mutiple/pickle/hybird(val_multiple)"),
    "validation_unheard": Path(
        "/data/Getuanhui/avnav-hybrid-mutiple/pickle/hybird(val_multiple_unheard)"
    ),
}


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


def _read_audio(observation) -> torch.Tensor:
    spectrogram = observation["spectrogram"]
    if isinstance(spectrogram, tuple) and len(spectrogram) == 2:
        spectrogram = spectrogram[0]
    audio = torch.as_tensor(np.asarray(spectrogram)).float()
    if audio.ndim == 4 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim != 3 or audio.shape[-1] != 2:
        raise ValueError(f"Expected (frequency,time,2) binaural audio, got {tuple(audio.shape)}")
    return audio


def _read_visual(observation):
    rgb = torch.as_tensor(np.asarray(observation["rgb"])).float() / 255.0
    depth = torch.as_tensor(np.asarray(observation["depth"])).float()
    if depth.ndim == 2:
        depth = depth.unsqueeze(-1)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or depth.shape != (*rgb.shape[:2], 1):
        raise ValueError(f"Invalid RGB/depth observation dimensions: {tuple(rgb.shape)}, {tuple(depth.shape)}")
    return rgb, depth


@torch.inference_mode()
def _episode_features(model, observations, device, batch_size, modality="multimodal"):
    audios, rgb_pairs, depth_pairs = [], [], []
    previous_rgb = previous_depth = None
    for observation in observations:
        if modality in ("multimodal", "audio_only"):
            audios.append(_read_audio(observation))
        if modality in ("multimodal", "vision_only"):
            rgb, depth = _read_visual(observation)
            if previous_rgb is None:
                previous_rgb, previous_depth = torch.zeros_like(rgb), torch.zeros_like(depth)
            rgb_pairs.append(torch.cat((previous_rgb, rgb), dim=-1))
            depth_pairs.append(torch.cat((previous_depth, depth), dim=-1))
            previous_rgb, previous_depth = rgb, depth
        if modality not in ("multimodal", "vision_only", "audio_only"):
            raise ValueError(f"Unknown feature modality: {modality}")

    batches = []
    count = len(audios) if modality in ("multimodal", "audio_only") else len(rgb_pairs)
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        audio = torch.stack(audios[start:stop]).to(device) if audios else None
        rgb = torch.stack(rgb_pairs[start:stop]).to(device) if rgb_pairs else None
        depth = torch.stack(depth_pairs[start:stop]).to(device) if depth_pairs else None
        if modality == "multimodal":
            embedding = model.embedding_forward(audio, rgb, depth)
        else:
            embedding = model.embedding_forward_modality(audio, rgb, depth, modality)
        embedding = embedding.float()
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        batches.append(embedding.cpu())
    return torch.cat(batches, dim=0).contiguous()


def _load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _pack_episode(model, path, device, batch_size, expected=None, modality="multimodal"):
    data = _load_pickle(path)
    actions = torch.as_tensor(np.asarray(data["action_id"]).reshape(-1), dtype=torch.long)
    rewards = torch.as_tensor(np.asarray(data.get("reward", []), dtype=np.float32).reshape(-1))
    dones = torch.as_tensor(np.asarray(data.get("done", []), dtype=np.bool_).reshape(-1))
    collisions = torch.as_tensor(np.asarray(data["collision"], dtype=np.bool_).reshape(-1))
    observations = data.get("obs", [])
    if len(observations) != len(actions) + 1:
        raise ValueError(f"{path}: expected one terminal observation, got {len(observations)} obs for {len(actions)} actions")
    if any(len(value) != len(actions) for value in (rewards, dones, collisions)):
        raise ValueError(f"{path}: transition field lengths do not align")
    if actions.numel() and (int(actions.min()) < 0 or int(actions.max()) >= 4):
        raise ValueError(f"{path}: action ID outside [0,3]")
    scene = str(data.get("parent_scene", Path(path).parent.name))
    episode_id = str(data.get("parent_episode_id", Path(path).stem))
    task_id = str(data.get("task_id", Path(path).stem))
    if expected is not None and (scene, episode_id, task_id) != (
            str(expected["scene"]), str(expected["episode_id"]), str(expected["task_id"])):
        raise ValueError(f"{path}: serialized provenance does not match the selection manifest")
    if not len(actions):
        raise ValueError(f"{path}: empty trajectory")
    if not torch.isfinite(rewards).all():
        raise ValueError(f"{path}: non-finite recorded reward")
    if dones[:-1].any():
        raise ValueError(f'{path}: transitions continue after terminal state')
    distances = torch.as_tensor(data['distance_to_goal'], dtype=torch.float32)
    if distances.shape != (len(actions) + 1,):
        raise ValueError(f'{path}: distance trajectory misaligned')
    if not torch.isfinite(distances).all():
        raise ValueError(f'{path}: nonfinite distance')
    recovery_delays = []
    for step in collisions.nonzero().flatten().tolist():
        for later in range(step + 1, min(step + 17, len(actions))):
            if not collisions[later] and distances[later + 1] < distances[later] - 1e-4:
                recovery_delays.append(later - step)
                break
    features = _episode_features(model, observations, device, batch_size, modality=modality)
    return {
        "features": features,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "collisions": collisions,
        "scene": scene,
        "episode_id": episode_id,
        "task_id": task_id,
        "behavior_source": data.get("behavior_source"),
        "action_handling": data.get("action_handling"),
        "source_path": str(path), "source_sha256": _sha256(path),
        "coverage": {'collisions': int(collisions.sum()), 'progress_within_16_steps': len(recovery_delays),
                     'delay_sum': sum(recovery_delays), 'episode_success': bool(data['episode_success'])},
    }


def _extract_category(model, category, device, batch_size, modality="multimodal"):
    manifest_path = CODEX_ROOT / category / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    episodes = []
    for index, row in enumerate(rows, 1):
        if row.get("status") != "saved" or row.get("error") is not None:
            raise ValueError(f"{category}: refusing unavailable trajectory {row.get('task_id')}")
        episodes.append(_pack_episode(model, row["file"], device, batch_size, row, modality=modality))
        if index % 100 == 0 or index == len(rows):
            print(json.dumps({"stage": "features", "split": category,
                              "episodes": index, "total": len(rows)}), flush=True)
    return episodes


def _extract_validation(model, name, source, device, batch_size):
    files = sorted(Path(source).rglob("*.pkl"))
    episodes = []
    for index, path in enumerate(files, 1):
        data = _load_pickle(path)
        actions = torch.as_tensor(np.asarray(data["action_id"]).reshape(-1), dtype=torch.long)
        rewards = torch.as_tensor(np.asarray(data.get("reward", []), dtype=np.float32).reshape(-1))
        dones = torch.as_tensor(np.asarray(data.get("done", []), dtype=np.bool_).reshape(-1))
        if not actions.numel() or len(data.get("obs", [])) != len(actions) + 1:
            raise ValueError(f"{path}: invalid validation episode structure")
        if any(len(value) != len(actions) for value in (rewards, dones)):
            raise ValueError(f"{path}: validation transition fields are misaligned")
        features = _episode_features(model, data["obs"], device, batch_size)
        episodes.append({
            "features": features, "actions": actions, "rewards": rewards,
            "dones": dones, "collisions": torch.zeros_like(dones),
            "scene": path.parent.name, "episode_id": path.stem,
            "task_id": f"{name}_{path.parent.name}_{path.stem}",
            "behavior_source": None, "action_handling": "recorded validation demonstration",
        })
        if index % 100 == 0 or index == len(files):
            print(json.dumps({"stage": "features", "split": name,
                              "episodes": index, "total": len(files)}), flush=True)
    return episodes


def _save_split(path, episodes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"episodes": episodes}, temporary)
    temporary.replace(path)


def run(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    audit_path = run_dir / "audit/data_source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Feature extraction blocked: data-source audit is not pass")
    foundation_path = run_dir / "foundation/action_stage/best.pth"
    if not foundation_path.is_file():
        raise FileNotFoundError(f"Selected frozen foundation checkpoint is missing: {foundation_path}")
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    model = Network().to(device)
    model.load_state_dict(torch.load(foundation_path, map_location=device, weights_only=True))
    model.eval().requires_grad_(False)

    out = run_dir / "offline/features"
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    categories = {}
    train_episodes = []
    for category in ("greedy", "deverse", "collision"):
        episodes = _extract_category(model, category, device, args.batch_size)
        manifest_path = CODEX_ROOT / category / "manifest.jsonl"
        categories[category] = {
            "manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
            "episodes": len(episodes),
            "transitions": sum(len(row["actions"]) for row in episodes),
            "recorded_collisions": sum(int(row["collisions"].sum()) for row in episodes),
        }
        train_episodes.extend(episodes)
        print(json.dumps({"stage": "features", "split": category,
                          "status": "completed", "episodes": len(episodes),
                          "transitions": categories[category]["transitions"]}), flush=True)
    train_path = out / "train.pt"
    coverage = [row['coverage'] for row in train_episodes]
    collision_count = sum(row['collisions'] for row in coverage)
    progress_count = sum(row['progress_within_16_steps'] for row in coverage)
    _atomic_json(run_dir / 'audit/transition_coverage.json', {
        'scope': 'train only', 'episodes': len(coverage), 'collision_steps': collision_count,
        'collision_steps_followed_by_progress_within_16': progress_count,
        'mean_progress_delay_conditional': sum(r['delay_sum'] for r in coverage) / max(progress_count, 1),
        'successful_episodes_with_collision': sum(r['episode_success'] and r['collisions'] > 0 for r in coverage),
        'definition': 'subsequent noncolliding transition decreases recorded train geodesic distance; diagnostics only',
    })
    _save_split(train_path, train_episodes)
    outputs["train"] = {"path": str(train_path), "episodes": len(train_episodes),
                        "transitions": sum(len(row["actions"]) for row in train_episodes)}
    del train_episodes

    validation_details = {}
    for name, source in VAL_PICKLES.items():
        episodes = _extract_validation(model, name, source, device, args.batch_size)
        path = out / f"{name}.pt"
        _save_split(path, episodes)
        validation_details[name] = {
            "path": str(path), "episodes": len(episodes),
            "transitions": sum(len(row["actions"]) for row in episodes),
            "pickle_source": str(source),
        }
        outputs[name] = {"path": str(path), "episodes": len(episodes),
                         "transitions": validation_details[name]["transitions"]}
        del episodes

    manifest = {
        "status": "completed", "feature_dim": 256, "foundation_checkpoint": str(foundation_path),
        "foundation_sha256": _sha256(foundation_path), "foundation_frozen": True,
        "feature_schema": "repaired-v2-sample-local-audio-spatial-tokens",
        "data_source_audit": str(audit_path), "data_source_audit_sha256": _sha256(audit_path),
        "train_sources": categories, "validation_sources": validation_details,
        "outputs": outputs, "test_data_loaded": False,
        "action_handling": "recorded train/validation actions only; no action post-processing",
    }
    _atomic_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="result/2026-10-27-avnav-codex-full-training")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--cpu-threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
