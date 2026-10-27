"""Collect train-only CQL-LSTM rollouts and all-action simulator labels."""

import argparse
import hashlib
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.avnav.collision_predictor import ActionCollisionPredictor
from experiment.avnav.config import resolve_path
from experiment.avnav.environment.avnav_env import AudioNavRLEnv
from experiment.avnav.environment.config import get_config
from experiment.avnav.environment.evaluation_episodes import load_episodes
from experiment.avnav.evaluate_online import OnlinePolicy
from experiment.avnav.model.hybrid import Network
from experiment.avnav.offline_cql_lstm import LSTMActor


ACTION_NAMES = ("STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT")
DEPTH_MAX_M = 10.0  # Habitat depth is normalized by the sensor's 10 m max range.
WALL_THRESHOLD_M = 1.0
NARROW_SIDE_THRESHOLD_M = 2.5
NO_PROGRESS_STEPS = 5
NO_PROGRESS_TOLERANCE_M = 0.25


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(row):
    return str(row["scene"]), str(row["episode_id"])


def _depth_sectors(observation):
    depth = np.asarray(observation["depth"], dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth after squeeze, got {depth.shape}")
    height, width = depth.shape
    depth = np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    depth = depth[int(height * 0.25):int(height * 0.85)]
    sectors = []
    for index in range(5):
        left, right = round(index * width / 5), round((index + 1) * width / 5)
        sector = depth[:, left:right]
        sectors.append(float(np.quantile(sector, 0.2)))
    return sectors


def _geometry_tags(sectors):
    distances = [value * DEPTH_MAX_M for value in sectors]
    close_side = min(distances[0], distances[4]) <= WALL_THRESHOLD_M
    close_forward = distances[2] <= WALL_THRESHOLD_M
    wall = close_side or close_forward
    narrow = (distances[0] <= NARROW_SIDE_THRESHOLD_M and
              distances[4] <= NARROW_SIDE_THRESHOLD_M and
              distances[2] > WALL_THRESHOLD_M)
    return wall, narrow


def _distance(env):
    metrics = env._env.get_metrics()
    for name in ("distance_to_goal", "DISTANCE_TO_GOAL"):
        if name in metrics:
            return float(metrics[name])
    raise KeyError(f"Distance-to-goal measure unavailable; metrics are {sorted(metrics)}")


def _collided(env):
    sim = getattr(env._env, "_sim", None)
    return bool(getattr(sim, "previous_step_collided", False))


def _episode_anchors(observations, actions, collisions, distances, geometries,
                     dones, max_anchors):
    tags_by_step = {}

    def add(step, tag):
        can_act = (0 <= step < len(actions) and
                   (step == 0 or not bool(dones[step - 1])))
        if can_act:
            tags_by_step.setdefault(step, set()).add(tag)

    for step, collided in enumerate(collisions):
        if collided:
            add(step, "collision_pre")
            add(step + 1, "post_collision")

    for step in range(NO_PROGRESS_STEPS, len(observations)):
        window = distances[step - NO_PROGRESS_STEPS:step + 1]
        if window[0] - min(window) < NO_PROGRESS_TOLERANCE_M:
            add(step, "no_progress")

    for step, sectors in enumerate(geometries):
        wall, narrow = _geometry_tags(sectors)
        if wall:
            add(step, "wall_proximity")
        if narrow:
            add(step, "narrow_passage")

    priority = ("collision_pre", "post_collision", "no_progress",
                "narrow_passage", "wall_proximity")
    candidates = []
    used = set()
    for tag in priority:
        matching = [step for step, tags in tags_by_step.items() if tag in tags]
        if not matching:
            continue
        # Spread collision samples across an episode and use the clearest
        # geometry/stall state when a category has many eligible rows.
        if tag in ("wall_proximity", "narrow_passage"):
            matching.sort(key=lambda step: min(geometries[step][0], geometries[step][2],
                                               geometries[step][4]))
        elif tag == "no_progress":
            matching.sort(key=lambda step: distances[step] -
                          min(distances[max(0, step - NO_PROGRESS_STEPS):step + 1]))
        else:
            matching.sort()
        chosen = next((step for step in matching
                       if all(abs(step - prior) > 2 for prior in used)), None)
        if chosen is not None:
            candidates.append(chosen)
            used.add(chosen)
        if len(candidates) >= max_anchors:
            break
    if not candidates and actions:
        fallback = min(len(actions) - 1, len(actions) // 2)
        while fallback > 0 and bool(dones[fallback - 1]):
            fallback -= 1
        candidates = [fallback]
        tags_by_step[fallback] = {"policy_state"}
    return [{"step": step, "event_tags": sorted(tags_by_step[step])}
            for step in sorted(candidates)]


def _set_episode(env, episode, expected_key):
    env._env.current_episode = episode
    observation = env.reset()
    active = env._env.current_episode
    active_key = (Path(str(active.scene_id)).stem, str(active.episode_id))
    if active_key != expected_key:
        raise RuntimeError(f"Reset changed episode identity to {active_key}; expected {expected_key}")
    return observation


def _branch_at_anchor(env, policy, episode, expected_key, anchor_step,
                      episode_actions, episode_collisions, target_observation,
                      target_distance, horizon):
    observation = _set_episode(env, episode, expected_key)
    policy.reset()
    replay_divergences = 0
    for step in range(anchor_step):
        predicted = policy.act(observation)
        replay_divergences += int(predicted != int(episode_actions[step]))
        observation, _, done, _ = env.step(action=int(episode_actions[step]))
        if done and step + 1 < anchor_step:
            raise RuntimeError("Recorded prefix terminated before the selected anchor")
        policy.record_collision(_collided(env))

    replay_distance = _distance(env)
    target_depth = np.asarray(target_observation["depth"], dtype=np.float32)
    replay_depth = np.asarray(observation["depth"], dtype=np.float32)
    depth_mae = float(np.nanmean(np.abs(target_depth - replay_depth)))
    distance_error = abs(replay_distance - float(target_distance))
    if distance_error > 0.05 or depth_mae > 0.02:
        raise RuntimeError(
            f"Counterfactual replay did not recover anchor: distance_error={distance_error:.4f}, "
            f"depth_mae={depth_mae:.5f}"
        )
    return observation, replay_divergences, depth_mae


def _counterfactuals(env, policy, episode, key, anchors, trajectory,
                     horizon):
    labels = []
    for anchor in anchors:
        step = int(anchor["step"])
        for action_id, action_name in enumerate(ACTION_NAMES):
            observation, divergences, depth_mae = _branch_at_anchor(
                env, policy, episode, key, step,
                trajectory["action_id"], trajectory["collision"],
                trajectory["obs"][step], trajectory["distance_to_goal"][step],
                horizon,
            )
            # Compute the reference actor's anchor output to advance its LSTM
            # context, then execute the counterfactual candidate verbatim.
            policy.act(observation)
            policy.previous_action = action_id
            distances = [float(trajectory["distance_to_goal"][step])]
            branch_collisions, branch_actions, branch_rewards = [], [], []
            branch_dones, policy_continuations = [], []
            done = False
            for future_step in range(horizon):
                if future_step == 0:
                    executed = action_id
                    policy_continuations.append(None)
                else:
                    executed = policy.act(observation)
                    policy_continuations.append(int(executed))
                observation, reward, done, _ = env.step(action=int(executed))
                collided = _collided(env)
                policy.record_collision(collided)
                branch_actions.append(int(executed))
                branch_rewards.append(float(reward))
                branch_dones.append(bool(done))
                branch_collisions.append(bool(collided))
                distances.append(_distance(env))
                if done:
                    break
            labels.append({
                "scene": key[0], "episode_id": key[1],
                "anchor_step": step, "event_tags": anchor["event_tags"],
                "candidate_action_id": action_id, "candidate_action": action_name,
                "immediate_collision": branch_collisions[0],
                "future_collision_count": int(sum(branch_collisions)),
                "future_any_collision": bool(any(branch_collisions)),
                "future_collision_count_after_first": int(sum(branch_collisions[1:])),
                "future_any_collision_after_first": bool(any(branch_collisions[1:])),
                "future_horizon_requested": int(horizon),
                "future_horizon_observed": len(branch_collisions),
                "movement_progress_m": float(max(
                    0.0, distances[0] - min(distances[1:])) if len(distances) > 1 else 0.0),
                "distance_after_first_action_m": float(distances[1]),
                "distance_after_horizon_m": float(distances[-1]),
                "distance_trace_m": distances,
                "branch_actions": branch_actions,
                "reference_policy_continuations": policy_continuations,
                "branch_rewards": branch_rewards,
                "branch_dones": branch_dones,
                "anchor_replay_policy_divergences": divergences,
                "anchor_replay_depth_mae": depth_mae,
                "action_handling": "counterfactual candidate label; raw frozen actor continuation",
            })
    return labels


def _write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def collect(config_path, checkpoint_path, episode_manifest, output_dir,
            episode_role="train", limit_episodes=0, split="train_multiple",
            max_steps=200, horizon=5, max_anchors_per_episode=5):
    config_path = Path(config_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    episode_manifest = Path(episode_manifest).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    run_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    permitted_splits = {
        "train": {"train_multiple"},
        "validation": {"val_multiple", "val_multiple_unheard"},
    }
    if episode_role not in permitted_splits or split not in permitted_splits[episode_role]:
        raise ValueError("Collection is limited to train_multiple/train or validation splits")
    rows = [json.loads(line) for line in episode_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = [row for row in rows if row.get("role") == episode_role]
    if any(row.get("split") != split for row in rows):
        raise ValueError(f"The {episode_role} selection manifest contains a different split")
    keys = [_key(row) for row in rows]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Episode selection is empty or contains duplicate keys")
    if limit_episodes:
        if limit_episodes < 1:
            raise ValueError("limit_episodes must be positive")
        rows = rows[:limit_episodes]
        keys = keys[:limit_episodes]

    foundation_path = resolve_path(run_config["FOUNDATION"]["CHECKPOINT"])
    rl = run_config["OFFLINE_RL"]
    predictor_path = resolve_path(rl["COLLISION_PREDICTOR_CHECKPOINT"])
    device = torch.device(run_config["EVALUATION"]["DEVICE"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    seed = int(run_config["SEED"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(foundation_path, map_location=device,
                                          weights_only=True))
    foundation.eval().requires_grad_(False)
    predictor_state = torch.load(predictor_path, map_location="cpu", weights_only=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    foundation_sha = _sha256(foundation_path)
    predictor_sha = _sha256(predictor_path)
    if predictor_state.get("foundation_sha256") != foundation_sha:
        raise ValueError("Collision predictor and foundation checkpoints do not match")
    if checkpoint.get("foundation_sha256") != foundation_sha:
        raise ValueError("CQL-LSTM checkpoint uses a different foundation")
    policy_config = checkpoint.get("config") or {}
    if policy_config.get("algorithm") != "cql_lstm":
        raise ValueError("Expected the frozen CQL-LSTM checkpoint")
    if policy_config.get("collision_predictor_sha256") != predictor_sha:
        raise ValueError("CQL-LSTM checkpoint uses a different collision predictor")
    actor = LSTMActor(264, int(policy_config.get("hidden_dim", 256)),
                      foundation=foundation).to(device)
    actor.load_state_dict(checkpoint["actor"])
    predictor = ActionCollisionPredictor().to(device)
    predictor.load_state_dict(predictor_state["predictor"])
    policy = OnlinePolicy(foundation, actor, device, action_history=True,
                          collision_history=False, collision_predictor=predictor,
                          collision_risk_actions=True, sequence_length=5)

    config = get_config(task_config_path=str(resolve_path(run_config["EVALUATION"]["CONFIG"])),
                        opts=["TASK_CONFIG.DATASET.SPLIT", split])
    config.defrost()
    config.TASK_CONFIG.TRAIN.OPEN = False
    config.TASK_CONFIG.COLLECT.OPEN = False
    config.TASK_CONFIG.TASK.SENSORS = ["SPECTROGRAM_SENSOR"]
    config.TASK_CONFIG.TASK.MEASUREMENTS = [
        name for name in config.TASK_CONFIG.TASK.MEASUREMENTS if name != "TOP_DOWN_MAP"
    ]
    config.freeze()
    dataset = load_episodes(config.TASK_CONFIG.DATASET, max_episodes=0, seed=seed,
                            episode_keys=keys)
    episodes = list(dataset.episodes)
    episode_by_key = {(Path(str(row.scene_id)).stem, str(row.episode_id)): row
                      for row in episodes}
    if set(episode_by_key) != set(keys):
        raise ValueError("Loaded episodes do not match the frozen selection manifest")

    context = {
        "status": "frozen_before_collection",
        "collector_script": str(Path(__file__).resolve()),
        "collector_script_sha256": _sha256(Path(__file__).resolve()),
        "config": str(config_path), "config_sha256": _sha256(config_path),
        "policy_checkpoint": str(checkpoint_path),
        "policy_checkpoint_sha256": _sha256(checkpoint_path),
        "policy_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection": "frozen reference CQL-LSTM checkpoint",
        "foundation_checkpoint": str(foundation_path), "foundation_sha256": foundation_sha,
        "collision_predictor_checkpoint": str(predictor_path),
        "collision_predictor_sha256": predictor_sha,
        "episode_manifest": str(episode_manifest),
        "episode_manifest_sha256": _sha256(episode_manifest),
        "episode_role": episode_role, "split": split,
        "selected_episodes": [{"scene": scene, "episode_id": episode_id}
                               for scene, episode_id in keys],
        "seed": seed, "max_steps": int(max_steps), "future_horizon": int(horizon),
        "max_anchors_per_episode": int(max_anchors_per_episode),
        "state_selection": {
            "collision_pre_and_post": True,
            "no_progress_steps": NO_PROGRESS_STEPS,
            "no_progress_tolerance_m": NO_PROGRESS_TOLERANCE_M,
            "depth_max_distance_m": DEPTH_MAX_M,
            "wall_threshold_m": WALL_THRESHOLD_M,
            "narrow_side_threshold_m": NARROW_SIDE_THRESHOLD_M,
            "depth_summary": "20th percentile of five horizontal sectors over rows 25%-85%",
        },
        "candidate_action_ids": list(range(4)),
        "candidate_action_order": list(ACTION_NAMES),
        "action_handling": ("raw CQL-LSTM argmax rollouts; candidate branches used only for "
                            f"{episode_role} labels; raw actor continuation; no action adapter"),
        "replay_protocol": ("reset exact selected episode and replay recorded prefix actions; "
                            "require distance error <= 0.05m and depth MAE <= 0.02"),
        "validation_labels_used_for_training": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    context_path = output_dir / "collection_protocol.json"
    if context_path.exists():
        existing = json.loads(context_path.read_text(encoding="utf-8"))
        if existing != context:
            raise FileExistsError("Existing collection directory has a different frozen protocol")
    elif any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty collection output: {output_dir}")
    else:
        _write_json_atomic(context_path, context)

    trajectory_root = output_dir / "trajectories"
    label_root = output_dir / "candidate_labels"
    parent_manifest = output_dir / "parent_episode_manifest.jsonl"
    completed = {}
    if parent_manifest.exists():
        for line in parent_manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[(str(row["scene"]), str(row["episode_id"]))] = row
    unexpected = set(completed) - set(keys)
    if unexpected:
        raise ValueError(f"Output parent manifest contains unselected episodes: {sorted(unexpected)[:5]}")

    env = AudioNavRLEnv(config, dataset)
    try:
        for index, key in enumerate(keys):
            scene, episode_id = key
            if key in completed:
                print(f"resume: {index + 1}/{len(keys)} already complete {key}", flush=True)
                continue
            episode = episode_by_key[key]
            try:
                observation = _set_episode(env, episode, key)
                policy.reset()
                observations = [observation]
                actions, rewards, dones, collisions = [], [], [], []
                distances = [_distance(env)]
                geometries = [_depth_sectors(observation)]
                done, info = False, {}
                while not done and len(actions) < max_steps:
                    action = policy.act(observation)
                    observation, reward, done, info = env.step(action=int(action))
                    collided = _collided(env)
                    policy.record_collision(collided)
                    actions.append(int(action))
                    rewards.append(float(reward))
                    dones.append(bool(done))
                    collisions.append(collided)
                    observations.append(observation)
                    distances.append(_distance(env))
                    geometries.append(_depth_sectors(observation))
                trajectory = {
                    "obs": observations, "action_id": actions, "reward": rewards,
                    "done": dones, "collision": collisions,
                    "distance_to_goal": distances,
                    "distance_delta": [0.0] + [distances[i] - distances[i + 1]
                                                for i in range(len(actions))],
                    "depth_sectors": geometries,
                    "parent_scene": scene, "parent_episode_id": episode_id,
                    "episode_success": bool(float((info or {}).get("success", 0.0)) > 0.0),
                    "policy_checkpoint": str(checkpoint_path),
                    "policy_epoch": int(checkpoint["epoch"]),
                    "action_handling": "raw CQL-LSTM argmax; no action override or recovery",
                }
                anchors = _episode_anchors(observations, actions, collisions,
                                           distances, geometries, dones,
                                           max_anchors_per_episode)
                labels = _counterfactuals(env, policy, episode, key, anchors,
                                          trajectory, horizon)

                trajectory_path = trajectory_root / scene / f"{episode_id}.pkl"
                trajectory_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = trajectory_path.with_suffix(".pkl.tmp")
                with temporary.open("wb") as handle:
                    pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
                temporary.replace(trajectory_path)
                label_path = label_root / scene / f"{episode_id}.jsonl"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_label = label_path.with_suffix(".jsonl.tmp")
                with temporary_label.open("w", encoding="utf-8") as handle:
                    for row in labels:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                temporary_label.replace(label_path)
                parent = {
                    "scene": scene, "episode_id": episode_id, "split": split,
                    "role": episode_role, "steps": len(actions),
                    "collisions": int(sum(collisions)), "anchors": len(anchors),
                    "candidate_labels": len(labels),
                    "trajectory_file": str(trajectory_path.resolve()),
                    "candidate_label_file": str(label_path.resolve()),
                    "error": None,
                }
            except Exception as error:
                parent = {"scene": scene, "episode_id": episode_id,
                          "split": split, "role": episode_role,
                          "steps": 0, "collisions": 0, "anchors": 0,
                          "candidate_labels": 0,
                          "error": f"{type(error).__name__}: {error}"}
                print(json.dumps(parent, ensure_ascii=False), flush=True)
            with parent_manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(parent, ensure_ascii=False) + "\n")
                handle.flush()
            completed[key] = parent
            print(f"{split}: {index + 1}/{len(keys)} episodes; "
                  f"steps={sum(row.get('steps', 0) for row in completed.values())}; "
                  f"candidate_labels={sum(row.get('candidate_labels', 0) for row in completed.values())}; "
                  f"errors={sum(row.get('error') is not None for row in completed.values())}",
                  flush=True)
    finally:
        env.close()

    counts = Counter()
    for key in keys:
        row = completed[key]
        counts["episodes"] += 1
        counts["steps"] += int(row.get("steps", 0))
        counts["collisions"] += int(row.get("collisions", 0))
        counts["anchors"] += int(row.get("anchors", 0))
        counts["candidate_labels"] += int(row.get("candidate_labels", 0))
        counts["errors"] += int(row.get("error") is not None)
    summary = {
        "status": "completed" if counts["errors"] == 0 else "completed_with_errors",
        "protocol": str(context_path),
        "parent_episode_manifest": str(parent_manifest.resolve()),
        "candidate_action_order": list(ACTION_NAMES),
        "totals": dict(counts),
        "error_rows": [row for row in completed.values() if row.get("error")],
    }
    _write_json_atomic(output_dir / "collection_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-role", default="train")
    parser.add_argument("--limit-episodes", type=int, default=0,
                        help="Optional deterministic prefix limit for a train-only pilot")
    parser.add_argument("--split", default="train_multiple")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-anchors-per-episode", type=int, default=5)
    args = parser.parse_args()
    result = collect(args.config, args.checkpoint, args.episode_manifest,
                     args.output, args.episode_role, args.limit_episodes,
                     args.split, args.max_steps, args.horizon,
                     args.max_anchors_per_episode)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
