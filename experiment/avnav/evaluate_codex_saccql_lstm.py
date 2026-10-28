"""Evaluate a frozen AVNav foundation + raw LSTM SAC-CQL actor on Habitat splits."""
from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import torch

from experiment.avnav.hybrid.foundation_model import Network
from experiment.avnav.offline_rl_v1_5_lstm.OfflineNet import SAC_LSTM_CQL_v1_5


DEFAULT_RUN = Path("result/2026-10-27-avnav-codex-full-training")


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path):
    digest = __import__("hashlib").sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_runtime_dataset_manifest(split, data_root, run_dir, source_manifest=None):
    """Resolve the benchmark manifest and gzip plain JSON only for Habitat's loader."""
    source_dir = Path(data_root).expanduser().resolve() / "datasets/audionav/mp3d/v1" / split
    compressed_source = source_dir / f"{split}.json.gz"
    plain_source = source_dir / f"{split}.json"
    source_path = (Path(source_manifest).resolve() if source_manifest is not None else
                   compressed_source if compressed_source.is_file() else plain_source)
    if not source_path.is_file():
        raise FileNotFoundError(f'Audited benchmark manifest missing: {source_path}')
    if source_path.suffix == '.gz':
        runtime_path = source_path
        conversion = "none; source already gzip-compressed"
    else:
        runtime_path = Path(run_dir) / "evaluation/runtime_dataset_manifests" / f"{split}.json.gz"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = runtime_path.with_suffix(runtime_path.suffix + ".tmp")
        with source_path.open("rb") as source, gzip.open(temporary, "wb") as destination:
            shutil.copyfileobj(source, destination)
        temporary.replace(runtime_path)
        conversion = "gzip wrapper only; JSON payload copied byte-for-byte for Habitat gzip loader"
    return {
        "source_manifest": str(source_path),
        "source_manifest_sha256": _sha256_file(source_path),
        "runtime_manifest": str(runtime_path),
        "runtime_manifest_sha256": _sha256_file(runtime_path),
        "runtime_conversion": conversion,
    }


def _read_split_ids(path):
    return {(str(row["scene"]), str(row["episode_id"]))
            for row in (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())}


def _read_dataset_episode_ids(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        (Path(str(row["scene_id"])).stem, str(row["episode_id"]))
        for row in payload.get("episodes", [])
    }


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensors(observation):
    rgb = torch.as_tensor(np.asarray(observation["rgb"])).float().div_(255.0)
    depth = torch.as_tensor(np.asarray(observation["depth"])).float()
    if depth.ndim == 2:
        depth = depth.unsqueeze(-1)
    audio = observation["spectrogram"]
    if isinstance(audio, tuple) and len(audio) == 2:
        audio = audio[0]
    audio = torch.as_tensor(np.asarray(audio)).float()
    if audio.ndim == 4 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim != 3 or audio.shape[-1] != 2:
        raise ValueError(f"Expected binaural spectrogram (F,T,2), got {tuple(audio.shape)}")
    return audio, rgb, depth


def _episode_identity(env):
    episode = env._env.current_episode
    scene = Path(str(episode.scene_id)).stem
    return scene, str(episode.episode_id)


def _load_models(run_dir, device, config, checkpoint=None, training_config=None):
    foundation_path = run_dir / "foundation/action_stage/best.pth"
    policy_path = Path(checkpoint) if checkpoint else run_dir / "offline/training/best_policy.pt"
    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(foundation_path, map_location=device, weights_only=True))
    foundation.eval().requires_grad_(False)
    with (Path(training_config) if training_config else run_dir / "offline/training/training_config.json").open(encoding="utf-8") as handle:
        rl_config = json.load(handle)
    policy = SAC_LSTM_CQL_v1_5(
        state_dim=int(rl_config.get('state_dim', 256)), hidden_dim=int(rl_config["hidden_dim"]), action_dim=4,
        actor_lr=float(rl_config["actor_lr"]), critic_lr=float(rl_config["critic_lr"]),
        alpha_lr=float(rl_config["alpha_lr"]), target_entropy=float(rl_config["target_entropy"]),
        tau=float(rl_config["target_tau"]), gamma=float(rl_config["gamma"]),
        beta=float(rl_config["cql_weight"]), device=device,
        lstm_hidden_dim=int(rl_config["lstm_hidden_dim"]), lstm_num_layers=1,
        use_target_encoder=rl_config.get('use_target_encoder', False),
    ).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    policy.eval()
    return foundation, policy, foundation_path, policy_path, rl_config


@torch.inference_mode()
def _run_episode(env, foundation, policy, device, sequence_length, max_steps, input_ablation="none",
                 policy_modality="multimodal"):
    observation = env.reset()
    scene, episode_id = _episode_identity(env)
    state_history = deque(maxlen=sequence_length)
    previous_rgb = previous_depth = None
    total_reward = 0.0
    collisions = 0
    previous_action = None
    action_counts = [0, 0, 0, 0]
    collision_run = max_collision_run = 0
    path_length = 0.0
    collision_ages, recovery_delays = [], []
    info = {}
    step_count = 0
    done = False
    while not done and step_count < max_steps:
        audio, rgb, depth = _to_tensors(observation)
        # Fixed sensor ablations alter model inputs only. The policy's action
        # distribution and the benchmark's action handling remain untouched.
        if input_ablation == "vision_only":
            audio = torch.zeros_like(audio)
        elif input_ablation == "audio_only":
            rgb, depth = torch.zeros_like(rgb), torch.zeros_like(depth)
        elif input_ablation != "none":
            raise ValueError(f"Unknown input ablation: {input_ablation}")
        if previous_rgb is None:
            prev_rgb, prev_depth = torch.zeros_like(rgb), torch.zeros_like(depth)
        else:
            prev_rgb, prev_depth = previous_rgb, previous_depth
        rgb_pair = torch.cat((prev_rgb, rgb), dim=-1).unsqueeze(0).to(device)
        depth_pair = torch.cat((prev_depth, depth), dim=-1).unsqueeze(0).to(device)
        if policy_modality == "multimodal":
            embedding = foundation.embedding_forward(
                audio.unsqueeze(0).to(device), rgb_pair, depth_pair
            )
        else:
            if input_ablation != "none":
                raise ValueError("Trained single-modality policies cannot also use input ablations")
            embedding = foundation.embedding_forward_modality(
                audio.unsqueeze(0).to(device), rgb_pair, depth_pair, policy_modality
            )
        embedding = embedding.float().reshape(-1)
        if policy.temporal_encoder.input_size == embedding.numel() + 4:
            action_encoding = embedding.new_zeros(4)
            if previous_action is not None:
                action_encoding[previous_action] = 1.0
            embedding = torch.cat((embedding, action_encoding))
        state_history.append(embedding)
        sequence = embedding.new_zeros((sequence_length, embedding.shape[-1]))
        sequence[-len(state_history):] = torch.stack(list(state_history))
        action = int(policy.get_action(sequence.unsqueeze(0), eval=True))
        sim = env._env._sim
        before_position = np.asarray(sim.get_agent_state().position).copy()
        # The actor's discrete output is sent directly to the benchmark environment.
        observation, reward, done, info = env.step(action=action)
        sim = getattr(env._env, "_sim", None)
        collided = action == 1 and bool(getattr(sim, "previous_step_collided", False))
        collisions += int(collided)
        action_counts[action] += 1
        collision_run = collision_run + 1 if collided else 0
        max_collision_run = max(max_collision_run, collision_run)
        movement = float(np.linalg.norm(np.asarray(sim.get_agent_state().position) - before_position))
        path_length += movement
        collision_ages = [age + 1 for age in collision_ages if age < 16]
        if movement > 1e-4:
            recovery_delays.extend(collision_ages)
            collision_ages = []
        if collided:
            collision_ages.append(0)
        previous_action = action
        total_reward += float(reward)
        step_count += 1
        previous_rgb, previous_depth = rgb, depth
    values = dict(info or {})
    try:
        metrics = env._env.get_metrics()
        if isinstance(metrics, dict):
            values.update(metrics)
    except (AttributeError, RuntimeError):
        pass
    spl = float(values['spl'] if 'spl' in values else values['SPL'])
    success_value = values['success'] if 'success' in values else values['SUCCESS']
    if not np.isfinite(spl) or not 0 <= spl <= 1.00001:
        raise ValueError(f'Invalid SPL {spl}')
    try:
        success = float(success_value) > 0.0
    except (TypeError, ValueError):
        success = bool(success_value)
    return {
        "scene": scene, "episode_id": episode_id, "success": bool(success),
        "spl": spl, "steps": step_count, "reward": total_reward,
        "collisions": collisions, "done": bool(done),
        'action_counts': action_counts, 'max_consecutive_collisions': max_collision_run,
        'path_length': path_length, 'timeout': step_count >= max_steps and previous_action != 0,
        'collision_events_followed_by_movement_within_16': len(recovery_delays),
        'recovery_delay_sum': sum(recovery_delays),
        'success_path_efficiency': spl if success else None,
    }


def _evaluate_split(split, split_ids_path, run_dir, device, seed, max_steps, output_dir, config_path, data_root,
                    dataset_manifest_override=None, checkpoint=None, training_config=None, episode_limit=None,
                    input_ablation="none", episode_start=0, episode_count=None,
                    policy_modality="multimodal"):
    # Habitat is imported only after the current working directory has been set to finnal_exp.
    from experiment.avnav.benchmark_runtime import get_benchmark_config
    from ss_baselines.common.environments import AudioNavRLEnv as Env
    from habitat.datasets import make_dataset

    split_ids = _read_split_ids(split_ids_path)
    _set_seed(seed)
    data_root = Path(data_root).expanduser().resolve()
    if episode_limit is not None and not split.startswith('val'):
        raise ValueError('Partial evaluation is allowed only for validation smoke checks')
    manifest = dataset_manifest_override or data_root / f'datasets/audionav/mp3d/v1/{split}/{split}.json.gz'
    config = get_benchmark_config(config_path, data_root, split, manifest, max_steps, seed)
    dataset_manifest_path = Path(config.TASK_CONFIG.DATASET.DATA_PATH.format(
        version=config.TASK_CONFIG.DATASET.VERSION, split=split
    ))
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(f"Dataset split manifest not found: {dataset_manifest_path}")
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    actual_ids = {(Path(e.scene_id).stem, str(e.episode_id)) for e in dataset.episodes}
    if actual_ids != split_ids or len(dataset.episodes) != len(split_ids):
        raise ValueError(f'{split}: dataset identities differ from audited manifest')
    full_expected_count = len(split_ids)
    if episode_start < 0 or (episode_count is not None and episode_count < 1):
        raise ValueError('Episode shard start/count must be nonnegative/positive')
    if episode_count is not None and episode_limit is not None:
        raise ValueError('Specify episode_count shards or episode_limit smoke checks, not both')
    if (episode_start or episode_count is not None) and not split.startswith('val'):
        raise ValueError('Episode sharding is allowed only for validation splits')
    if episode_count is not None:
        episode_end = min(episode_start + episode_count, full_expected_count)
        dataset.episodes = dataset.episodes[episode_start:episode_end]
        split_ids = {(Path(e.scene_id).stem, str(e.episode_id)) for e in dataset.episodes}
    if episode_limit is not None:
        dataset.episodes = sorted(dataset.episodes, key=lambda e: (e.scene_id, str(e.episode_id)))[:episode_limit]
        split_ids = {(Path(e.scene_id).stem, str(e.episode_id)) for e in dataset.episodes}
    elif episode_count is None and episode_start:
        raise ValueError('episode_start requires episode_count')
    env = Env(config, dataset=dataset)
    env.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f'{split}_environment.yaml').write_text(config.dump())
    expected_count = len(split_ids)
    actual_count = int(env._env.number_of_episodes)
    if actual_count != expected_count:
        env.close()
        raise RuntimeError(f"{split}: environment has {actual_count} episodes, expected {expected_count}")

    foundation, policy, foundation_path, policy_path, rl_config = _load_models(run_dir, device, config, checkpoint, training_config)
    if policy_modality not in ("multimodal", "vision_only", "audio_only"):
        env.close()
        raise ValueError(f"Unknown policy modality: {policy_modality}")
    expected_feature_dim = int(rl_config.get("feature_dim", 256))
    if policy.temporal_encoder.input_size != expected_feature_dim:
        env.close()
        raise ValueError("Policy recurrent input dimension does not match its training configuration")
    if policy_modality != "multimodal" and expected_feature_dim != 128:
        env.close()
        raise ValueError("Trained single-modality branch policies must use the 128-dimensional branch feature")
    sequence_length = int(rl_config["sequence_length"])
    episodes, errors = [], []
    seen = set()
    try:
        for index in range(expected_count):
            try:
                row = _run_episode(env, foundation, policy, device, sequence_length, max_steps,
                                   input_ablation=input_ablation, policy_modality=policy_modality)
                key = (row["scene"], row["episode_id"])
                if key in seen:
                    raise RuntimeError(f"Duplicate evaluated episode: {key}")
                seen.add(key)
                episodes.append(row)
            except Exception as exc:
                errors.append({"index": index, "type": type(exc).__name__, "message": str(exc)})
                print(json.dumps({"stage": "evaluation", "split": split,
                                  "episode_index": index, "error": type(exc).__name__,
                                  "message": str(exc)}), flush=True)
                # A fresh reset begins the next benchmark episode; no action recovery is applied.
                continue
            if (index + 1) % 25 == 0 or index + 1 == expected_count:
                print(json.dumps({"stage": "evaluation", "split": split,
                                  "episodes": index + 1, "completed": len(episodes),
                                  "errors": len(errors)}), flush=True)
    finally:
        env.close()

    missing = sorted(split_ids - seen)
    unexpected = sorted(seen - split_ids)
    completed = len(episodes)
    success_count = sum(int(row["success"]) for row in episodes)
    result = {
        "status": "pass" if completed == expected_count and not errors and not missing and not unexpected else "fail",
        "split": split, "expected_episodes": expected_count,
        "full_expected_episodes": full_expected_count,
        "evaluated_episodes": completed, "evaluation_errors": len(errors),
        "missing_expected_ids": len(missing), "unexpected_ids": len(unexpected),
        "missing_id_examples": missing[:30], "unexpected_id_examples": unexpected[:30],
        "SR": success_count / max(expected_count, 1),
        "SPL": sum(row["spl"] for row in episodes) / max(expected_count, 1),
        "mean_reward": sum(row["reward"] for row in episodes) / max(completed, 1),
        "mean_steps": sum(row["steps"] for row in episodes) / max(completed, 1),
        "collisions": sum(row["collisions"] for row in episodes),
        "checkpoint": str(policy_path),
        "checkpoint_sha256": __import__("hashlib").sha256(policy_path.read_bytes()).hexdigest(),
        "foundation_checkpoint": str(foundation_path),
        "foundation_checkpoint_sha256": __import__("hashlib").sha256(foundation_path.read_bytes()).hexdigest(),
        "dataset_root": str(data_root),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "split_id_manifest": str(split_ids_path),
        "split_id_manifest_sha256": __import__("hashlib").sha256(Path(split_ids_path).read_bytes()).hexdigest(),
        "seed": seed, "max_steps": max_steps,
        "policy_modality": policy_modality,
        "input_ablation": input_ablation,
        "input_handling": {
            "none": "native audio, RGB, and depth observations",
            "vision_only": "binaural spectrogram replaced with zeros at every step; RGB and depth retained",
            "audio_only": "RGB and depth replaced with zeros at every step; binaural spectrogram retained",
        }[input_ablation],
        "feature_handling": {
            "multimodal": "frozen audio and visual branch features passed through the trained AV fusion encoder",
            "vision_only": "only frozen visual encoder and visual transformer called; spatial tokens mean-pooled",
            "audio_only": "only frozen audio encoder called; its single branch token used",
        }[policy_modality],
        "action_handling": f"raw actor argmax over episode-local {sequence_length}-step sequence; direct discrete action; no overrides",
        "training_config": rl_config,
        "environment_sha256": _sha256_file(output_dir / f'{split}_environment.yaml'),
        "environment_path": str(output_dir / f'{split}_environment.yaml'),
        'partial_validation_smoke': episode_limit is not None,
        'partial_validation_shard': episode_count is not None,
        'evaluation_episode_range': [episode_start, episode_start + expected_count],
        'forward_collision_rate': sum(r['collisions'] for r in episodes) / max(sum(r['action_counts'][1] for r in episodes), 1),
        'timeout_rate': sum(r['timeout'] for r in episodes) / max(expected_count, 1),
        'mean_success_path_efficiency': sum(r['spl'] for r in episodes) / max(success_count, 1),
        "episode_metrics": episodes,
        "errors": errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / f"{split}.json", result)
    with (output_dir / f"{split}_episode_ids.jsonl").open("w", encoding="utf-8") as handle:
        for row in episodes:
            handle.write(json.dumps({"scene": row["scene"], "episode_id": row["episode_id"]}) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "split", "expected_episodes", "evaluated_episodes", "evaluation_errors", "SR", "SPL"
    )}), flush=True)
    return result


def main():
    # A single entry point enforces the frozen-selection gate for every test run.
    from experiment.avnav.evaluate_repaired import main as repaired_main
    repaired_main()


if __name__ == "__main__":
    main()
