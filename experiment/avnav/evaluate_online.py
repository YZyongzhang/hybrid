"""Evaluate an AVNav policy in closed-loop Habitat/SoundSpaces episodes."""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.avnav.config import resolve_path
from experiment.avnav.collision_predictor import (
    ActionCollisionPredictor,
    MultiHorizonActionCollisionPredictor,
)
from experiment.avnav.model.hybrid import Network
from experiment.avnav.offline_rl import _initial_policy_head
from experiment.avnav.offline_cql_lstm import LSTMActor
from experiment.avnav.safety_features import depth_sector_quantiles


def _audio(obs):
    value = obs["spectrogram"]
    if isinstance(value, (tuple, list)):
        value = value[0]
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.shape != (65, 26, 2):
        raise ValueError(f"Expected spectrogram [65, 26, 2], got {value.shape}")
    return torch.from_numpy(np.ascontiguousarray(value))


class OnlinePolicy:
    def __init__(self, foundation, actor, device, action_history=False,
                 collision_history=False, collision_predictor=None,
                 collision_risk_actions=False, sequence_length=1,
                 distance_delta_feature=False, local_geometry=False,
                 future_collision_risk_actions=False):
        self.foundation = foundation.eval()
        self.actor = actor.eval()
        self.device = device
        self.action_history = bool(action_history)
        self.collision_history = bool(collision_history)
        self.collision_predictor = collision_predictor
        self.collision_risk_actions = bool(collision_risk_actions)
        self.distance_delta_feature = bool(distance_delta_feature)
        self.local_geometry = bool(local_geometry)
        self.future_collision_risk_actions = bool(future_collision_risk_actions)
        self.sequence_length = int(sequence_length)
        self.state_history = []
        self.previous_rgb = None
        self.previous_depth = None
        self.previous_action = None
        self.previous_collision = False
        self.previous_distance = None
        self.last_collision_probability = None
        self.last_future_collision_probability = None
        self.state_history.clear()

    def reset(self):
        self.previous_rgb = self.previous_depth = None
        self.previous_action = None
        self.previous_collision = False
        self.previous_distance = None
        self.last_collision_probability = None
        self.last_future_collision_probability = None
        # The LSTM context is episode-local. Keeping rows from the previous
        # episode changes the first four decisions after every reset.
        self.state_history.clear()

    def record_collision(self, collided):
        """Store the previous transition's collision as learned policy context."""
        self.previous_collision = bool(collided)

    @torch.inference_mode()
    def act(self, observation, sample=False, temperature=1.0, current_distance=None):
        rgb = torch.as_tensor(np.asarray(observation["rgb"])).float().div_(255.0)
        depth = torch.as_tensor(np.asarray(observation["depth"])).float()
        if depth.ndim == 2:
            depth = depth.unsqueeze(-1)
        if self.previous_rgb is None:
            previous_rgb, previous_depth = torch.zeros_like(rgb), torch.zeros_like(depth)
        else:
            previous_rgb, previous_depth = self.previous_rgb, self.previous_depth
        audio_batch = _audio(observation).unsqueeze(0).to(self.device)
        rgb_batch = torch.cat((previous_rgb, rgb), dim=-1).unsqueeze(0).to(self.device)
        depth_batch = torch.cat((previous_depth, depth), dim=-1).unsqueeze(0).to(self.device)
        features = self.foundation.embedding_forward(audio_batch, rgb_batch, depth_batch)
        action_context = torch.zeros((1, 4), dtype=features.dtype, device=self.device)
        if self.previous_action is not None:
            action_context[0, self.previous_action] = 1.0
        if self.action_history:
            features = torch.cat((features, action_context), dim=-1)
        if self.collision_history:
            collision_context = torch.tensor(
                [[float(self.previous_collision)]], dtype=features.dtype,
                device=self.device)
            features = torch.cat((features, collision_context), dim=-1)
        if self.distance_delta_feature:
            distance_delta = 0.0
            if current_distance is not None and self.previous_distance is not None:
                distance_delta = float(self.previous_distance) - float(current_distance)
            progress_context = torch.tensor(
                [[distance_delta]], dtype=features.dtype, device=self.device
            )
            features = torch.cat((features, progress_context), dim=-1)
        if self.local_geometry:
            geometry = torch.tensor(
                [depth_sector_quantiles(observation)], dtype=features.dtype,
                device=self.device,
            )
            features = torch.cat((features, geometry), dim=-1)
        if self.collision_risk_actions:
            if self.collision_predictor is None:
                raise RuntimeError("Collision-risk input is enabled without a predictor")
            risks = self.collision_predictor.all_action_probabilities(
                features[:, :256], action_context
            )
            features = torch.cat((features, risks), dim=-1)
            future_risks = None
            if self.future_collision_risk_actions:
                if not hasattr(self.collision_predictor, "all_future_action_probabilities"):
                    raise RuntimeError("Future collision-risk input needs a two-horizon predictor")
                future_risks = self.collision_predictor.all_future_action_probabilities(
                    features[:, :256], action_context
                )
                features = torch.cat((features, future_risks), dim=-1)
        else:
            risks = None
            future_risks = None
        if self.sequence_length > 1:
            self.state_history.append(features[0])
            self.state_history = self.state_history[-self.sequence_length:]
            sequence = features.new_zeros((self.sequence_length, features.shape[-1]))
            sequence[-len(self.state_history):] = torch.stack(self.state_history)
            logits = self.actor(sequence.unsqueeze(0))
        else:
            logits = self.actor(features)
        if sample:
            temperature = float(temperature)
            if temperature <= 0:
                raise ValueError("Action sampling temperature must be positive")
            action = int(torch.distributions.Categorical(
                logits=logits / temperature).sample().item())
        else:
            action = int(logits.argmax(dim=-1).item())
        self.previous_rgb, self.previous_depth = rgb, depth
        self.previous_action = action
        self.previous_distance = (float(current_distance)
                                  if current_distance is not None else None)
        self.last_collision_probability = (
            float(risks[0, action].item()) if risks is not None else None
        )
        self.last_future_collision_probability = (
            float(future_risks[0, action].item()) if future_risks is not None else None
        )
        return action


def _metrics(env, info):
    values = dict(info or {})
    try:
        current = env._env.get_metrics()
        if isinstance(current, dict):
            values.update(current)
    except (AttributeError, RuntimeError):
        pass
    spl = float(values.get("spl", 0.0))
    success = values.get("success", values.get("SUCCESS", False))
    try:
        success = float(success) > 0
    except (TypeError, ValueError):
        success = bool(success)
    return spl, success


def _distance_to_goal(env):
    metrics = env._env.get_metrics()
    for name in ("distance_to_goal", "DISTANCE_TO_GOAL"):
        if name in metrics:
            return float(metrics[name])
    raise KeyError(f"Distance-to-goal measure unavailable: {sorted(metrics)}")


def _episode_safety_metrics(collisions, distances):
    max_run = run = repeated_steps = repeated_runs = 0
    for collided in collisions:
        run = run + 1 if collided else 0
        max_run = max(max_run, run)
    run = 0
    for collided in collisions:
        if collided:
            run += 1
        else:
            if run >= 2:
                repeated_steps += run
                repeated_runs += 1
            run = 0
    if run >= 2:
        repeated_steps += run
        repeated_runs += 1
    no_progress = 0
    for state_index in range(5, min(len(distances), len(collisions) + 1)):
        window = distances[state_index - 5:state_index + 1]
        if window[0] - min(window) < 0.25:
            no_progress += 1
    return {"max_consecutive_collisions": max_run,
            "collision_streaks_ge_2": repeated_runs,
            "collision_steps_in_streaks_ge_2": repeated_steps,
            "no_progress_windows": no_progress,
            "no_progress_window_rate": no_progress / max(len(collisions), 1)}


def _summarize_rows(split, rows):
    count = max(len(rows), 1)
    steps = sum(int(row.get("steps", 0)) for row in rows)
    collisions = sum(int(row.get("collisions", 0)) for row in rows)
    valid = [row for row in rows if not row.get("error")]
    episodes_with_streak = sum(bool(row.get("collision_streaks_ge_2", 0)) for row in valid)
    episodes_with_stall = sum(bool(row.get("no_progress_windows", 0)) for row in valid)
    result = dict(split=split, episodes=len(rows),
                  errors=sum(row.get("error") is not None for row in rows),
                  success_rate=sum(bool(row.get("success")) for row in rows) / count,
                  spl=sum(float(row.get("spl", 0.0)) for row in rows) / count,
                  mean_reward=sum(float(row.get("reward", 0.0)) for row in rows) / count,
                  mean_steps=steps / count, mean_collisions=collisions / count,
                  collisions_per_action=collisions / max(steps, 1),
                  repeated_collision_episode_rate=episodes_with_streak / max(len(valid), 1),
                  repeated_collision_steps_per_action=sum(
                      int(row.get("collision_steps_in_streaks_ge_2", 0)) for row in valid
                  ) / max(steps, 1),
                  no_progress_episode_rate=episodes_with_stall / max(len(valid), 1),
                  no_progress_windows_per_action=sum(
                      int(row.get("no_progress_windows", 0)) for row in valid
                  ) / max(steps, 1))
    action_names = ("STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT")
    action_counts = [0, 0, 0, 0]
    action_collisions = [0, 0, 0, 0]
    for row in rows:
        for action_id, value in enumerate(row.get("action_counts", [0, 0, 0, 0])):
            action_counts[action_id] += int(value)
        for action_id, value in enumerate(row.get("action_collisions", [0, 0, 0, 0])):
            action_collisions[action_id] += int(value)
    result["action_collision_rates"] = {
        name: {"actions": action_counts[action_id],
               "collisions": action_collisions[action_id],
               "collision_rate": action_collisions[action_id] /
                                 max(action_counts[action_id], 1)}
        for action_id, name in enumerate(action_names)
    }
    prediction = {name: sum(int(row.get(f"collision_prediction_{name}", 0))
                            for row in rows)
                  for name in ("tp", "fp", "tn", "fn")}
    prediction_samples = sum(prediction.values())
    if prediction_samples:
        recall = prediction["tp"] / max(prediction["tp"] + prediction["fn"], 1)
        specificity = prediction["tn"] / max(prediction["tn"] + prediction["fp"], 1)
        result["selected_action_collision_prediction"] = {
            "samples": prediction_samples,
            "recall_at_0_5": recall,
            "precision_at_0_5": prediction["tp"] / max(prediction["tp"] + prediction["fp"], 1),
            "specificity_at_0_5": specificity,
            "balanced_accuracy_at_0_5": (recall + specificity) / 2,
            "brier_score": sum(float(row.get("collision_prediction_brier_sum", 0.0))
                                for row in rows) / prediction_samples,
            "confusion_matrix": {"tp": prediction["tp"], "fp": prediction["fp"],
                                 "tn": prediction["tn"], "fn": prediction["fn"]},
        }
    future_prediction = {
        name: sum(int(row.get(f"future_collision_prediction_{name}", 0)) for row in rows)
        for name in ("tp", "fp", "tn", "fn")
    }
    future_samples = sum(future_prediction.values())
    if future_samples:
        recall = future_prediction["tp"] / max(
            future_prediction["tp"] + future_prediction["fn"], 1
        )
        specificity = future_prediction["tn"] / max(
            future_prediction["tn"] + future_prediction["fp"], 1
        )
        result["selected_action_future_collision_prediction"] = {
            "samples": future_samples,
            "recall_at_0_5": recall,
            "precision_at_0_5": future_prediction["tp"] / max(
                future_prediction["tp"] + future_prediction["fp"], 1
            ),
            "specificity_at_0_5": specificity,
            "balanced_accuracy_at_0_5": (recall + specificity) / 2,
            "brier_score": sum(float(row.get("future_collision_prediction_brier_sum", 0.0))
                                for row in rows) / future_samples,
            "confusion_matrix": {"tp": future_prediction["tp"],
                                 "fp": future_prediction["fp"],
                                 "tn": future_prediction["tn"],
                                 "fn": future_prediction["fn"]},
        }
    return result


def evaluate_split(split, policy, settings, seed, output, index_range=None):
    from experiment.avnav.environment.avnav_env import AudioNavRLEnv
    from experiment.avnav.environment.config import get_config
    from experiment.avnav.environment.evaluation_episodes import load_episodes

    config = get_config(task_config_path=settings["env_config"],
                        opts=["TASK_CONFIG.DATASET.SPLIT", split])
    config.defrost()
    config.TASK_CONFIG.TRAIN.OPEN = False
    config.TASK_CONFIG.COLLECT.OPEN = False
    # The AVNav primitive-action policy consumes RGB, depth, and audio.
    config.TASK_CONFIG.TASK.SENSORS = ["SPECTROGRAM_SENSOR"]
    config.TASK_CONFIG.TASK.MEASUREMENTS = [
        name for name in config.TASK_CONFIG.TASK.MEASUREMENTS if name != "TOP_DOWN_MAP"
    ]
    config.freeze()
    dataset = load_episodes(config.TASK_CONFIG.DATASET,
                            max_episodes=int(settings["episodes_per_split"]), seed=seed)
    selected_count = len(dataset.episodes)
    index_start, index_end = index_range or (0, selected_count)
    if not 0 <= index_start <= index_end <= selected_count:
        raise ValueError(
            f"Invalid episode range [{index_start}, {index_end}) for {selected_count} episodes"
        )
    dataset.episodes = dataset.episodes[index_start:index_end]
    # Keep an immutable ordering: Habitat's EpisodeIterator shuffles the
    # dataset list while AudioNavRLEnv is constructed.
    selected_episodes = list(dataset.episodes)
    expected = len(selected_episodes)
    expected_keys = [
        (Path(str(episode.scene_id)).stem, str(episode.episode_id))
        for episode in selected_episodes
    ]
    episodes_path = output / f"{split}_episodes.jsonl"
    rows = []
    if episodes_path.exists():
        rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) > expected or any(
                row.get("index") != index_start + offset
                for offset, row in enumerate(rows)):
            raise ValueError(f"Invalid or stale partial episode log: {episodes_path}")
        for offset, row in enumerate(rows):
            recorded_key = (str(row.get("scene", "unknown")),
                            str(row.get("episode_id", "")))
            if recorded_key != expected_keys[offset] and not (
                    row.get("error") and recorded_key[0] == "unknown"):
                raise ValueError(
                    f"Episode identity mismatch at index {index_start + offset}: "
                    f"saved {recorded_key}, expected {expected_keys[offset]} in {episodes_path}"
                )
    if len(rows) == expected:
        metrics = _summarize_rows(split, rows)
        (output / f"{split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"{split}: reusing {expected}/{expected} saved episodes", flush=True)
        return metrics

    env = AudioNavRLEnv(config, dataset)
    if rows:
        print(f"{split}: resuming after {len(rows)}/{expected} saved episodes", flush=True)
    max_steps = int(settings["max_steps"])
    try:
        for offset in range(expected):
            index = index_start + offset
            episode = selected_episodes[offset]
            try:
                # Habitat's reset may choose an arbitrary episode unless we
                # explicitly pin the one represented by this shard index.
                env._env.current_episode = episode
                observation = env.reset()
                active_episode = env._env.current_episode
                active_key = (Path(str(active_episode.scene_id)).stem,
                              str(active_episode.episode_id))
                if active_key != expected_keys[offset]:
                    raise RuntimeError(
                        f"Habitat reset to {active_key}, expected {expected_keys[offset]}"
                    )
            except Exception as error:
                row = dict(split=split, index=index,
                           scene=Path(str(episode.scene_id)).stem,
                           episode_id=str(episode.episode_id),
                           reward=0.0, steps=0, spl=0.0, collisions=0,
                           success=False, error=f"reset {type(error).__name__}: {error}")
                if offset < len(rows):
                    continue
                rows.append(row)
                with episodes_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                continue
            if offset < len(rows):
                continue
            policy.reset()
            episode = getattr(env._env, "current_episode", None)
            row = dict(split=split, index=index,
                       scene=Path(str(getattr(episode, "scene_id", "unknown"))).stem,
                       episode_id=str(getattr(episode, "episode_id", index)),
                       reward=0.0, steps=0, spl=0.0, collisions=0,
                       action_counts=[0, 0, 0, 0],
                       action_collisions=[0, 0, 0, 0],
                       collision_prediction_n=0, collision_prediction_tp=0,
                       collision_prediction_fp=0, collision_prediction_tn=0,
                       collision_prediction_fn=0, collision_prediction_brier_sum=0.0,
                       future_collision_prediction_n=0,
                       future_collision_prediction_tp=0,
                       future_collision_prediction_fp=0,
                       future_collision_prediction_tn=0,
                       future_collision_prediction_fn=0,
                       future_collision_prediction_brier_sum=0.0,
                       max_consecutive_collisions=0, collision_streaks_ge_2=0,
                       collision_steps_in_streaks_ge_2=0, no_progress_windows=0,
                       no_progress_window_rate=0.0,
                       success=False, error=None)
            info = {}
            collision_trace = []
            future_probability_trace = []
            distance_trace = []
            try:
                current_distance = _distance_to_goal(env)
                distance_trace.append(current_distance)
                done = False
                while not done and row["steps"] < max_steps:
                    action = policy.act(observation, current_distance=current_distance)
                    action_id = int(action)
                    row["action_counts"][action_id] += 1
                    observation, reward, done, info = env.step(action=action)
                    sim = getattr(env._env, "_sim", None)
                    collided = bool(getattr(sim, "previous_step_collided", False))
                    row["collisions"] += int(collided)
                    row["action_collisions"][action_id] += int(collided)
                    collision_trace.append(collided)
                    collision_probability = policy.last_collision_probability
                    if collision_probability is not None:
                        predicted_collision = collision_probability >= 0.5
                        row["collision_prediction_n"] += 1
                        row["collision_prediction_tp"] += int(predicted_collision and collided)
                        row["collision_prediction_fp"] += int(predicted_collision and not collided)
                        row["collision_prediction_tn"] += int(not predicted_collision and not collided)
                        row["collision_prediction_fn"] += int(not predicted_collision and collided)
                        row["collision_prediction_brier_sum"] += (
                            collision_probability - float(collided)
                        ) ** 2
                    future_probability_trace.append(
                        policy.last_future_collision_probability
                    )
                    policy.record_collision(collided)
                    row["reward"] += float(reward)
                    row["steps"] += 1
                    current_distance = _distance_to_goal(env)
                    distance_trace.append(current_distance)
                row["spl"], row["success"] = _metrics(env, info)
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                print(json.dumps(row), flush=True)
            row.update(_episode_safety_metrics(collision_trace, distance_trace))
            for index, probability in enumerate(future_probability_trace):
                if probability is None:
                    continue
                target = bool(any(collision_trace[index + 1:index + 5]))
                predicted = float(probability) >= 0.5
                row["future_collision_prediction_n"] += 1
                row["future_collision_prediction_tp"] += int(predicted and target)
                row["future_collision_prediction_fp"] += int(predicted and not target)
                row["future_collision_prediction_tn"] += int(not predicted and not target)
                row["future_collision_prediction_fn"] += int(not predicted and target)
                row["future_collision_prediction_brier_sum"] += (
                    float(probability) - float(target)
                ) ** 2
            rows.append(row)
            with episodes_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            if (offset + 1) % 10 == 0 or offset + 1 == expected:
                print(f"{split}: {offset + 1}/{expected} shard episodes "
                      f"(global index {index})", flush=True)
    finally:
        env.close()

    errors = sum(row["error"] is not None for row in rows)
    metrics = _summarize_rows(split, rows)
    (output / f"{split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run(config_path, checkpoint=None, output_dir=None, episodes=None, splits=None,
        index_range=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    raw_settings = config["EVALUATION"]
    settings = dict(episodes_per_split=int(raw_settings["EPISODES_PER_SPLIT"]),
                    max_steps=int(raw_settings["MAX_STEPS"]),
                    env_config=str(resolve_path(raw_settings["CONFIG"])))
    if episodes is not None:
        settings["episodes_per_split"] = int(episodes)
    split_names = list(splits) if splits else list(raw_settings["SPLITS"])
    device = torch.device(raw_settings["DEVICE"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    random.seed(int(raw_settings["SEED"]))
    np.random.seed(int(raw_settings["SEED"]))
    torch.manual_seed(int(raw_settings["SEED"]))

    foundation_path = resolve_path(config["FOUNDATION"]["CHECKPOINT"])
    offline_path = Path(checkpoint).expanduser() if checkpoint else resolve_path(config["OFFLINE_RL"]["OUTPUT"]) / "best_policy.pt"
    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(foundation_path, map_location=device, weights_only=True))
    state = torch.load(offline_path, map_location="cpu", weights_only=True)
    if state["foundation_sha256"] != __import__("hashlib").sha256(foundation_path.read_bytes()).hexdigest():
        raise ValueError("Offline policy was trained with a different foundation checkpoint")
    algorithm = str(state.get("algorithm") or (state.get("config") or {}).get("algorithm", "cql")).upper()
    training_config = state.get("config") or {}
    action_history = bool(training_config.get("action_history", False))
    collision_history = bool(training_config.get("collision_history", False))
    collision_risk_actions = bool(training_config.get("collision_risk_actions", False))
    distance_delta_feature = bool(training_config.get("distance_delta_feature", False))
    local_geometry = bool(training_config.get("local_geometry", False))
    future_collision_risk_actions = bool(
        training_config.get("future_collision_risk_actions", False)
    )
    predictor = None
    predictor_path = training_config.get("collision_predictor_checkpoint")
    if collision_risk_actions:
        if not predictor_path:
            raise ValueError("Collision-risk policy checkpoint lacks predictor provenance")
        predictor_path = resolve_path(predictor_path)
        predictor_sha256 = __import__("hashlib").sha256(
            predictor_path.read_bytes()).hexdigest()
        expected_predictor_sha256 = training_config.get("collision_predictor_sha256")
        if expected_predictor_sha256 and predictor_sha256 != expected_predictor_sha256:
            raise ValueError("Collision predictor checkpoint hash differs from the IQL manifest")
        predictor_state = torch.load(predictor_path, map_location="cpu", weights_only=True)
        if predictor_state.get("foundation_sha256") != __import__("hashlib").sha256(
                foundation_path.read_bytes()).hexdigest():
            raise ValueError("Collision predictor and policy use different foundation checkpoints")
        predictor_class = (MultiHorizonActionCollisionPredictor
                           if future_collision_risk_actions else ActionCollisionPredictor)
        predictor = predictor_class().to(device)
        predictor.load_state_dict(predictor_state["predictor"])
        predictor.eval().requires_grad_(False)
    input_components = ["embedding"]
    if action_history:
        input_components.append("previous action")
    if collision_history:
        input_components.append("previous collision")
    if distance_delta_feature:
        input_components.append("distance progress")
    if local_geometry:
        input_components.append("local depth geometry")
    if collision_risk_actions:
        input_components.append("per-action immediate collision risk")
    if future_collision_risk_actions:
        input_components.append("per-action short-horizon collision risk")
    if algorithm == "CQL_LSTM":
        action_handling = (
            "raw policy actor argmax over a 5-step episode-local LSTM of " +
            ", ".join(input_components) + "; no action override or recovery"
        )
    else:
        action_handling = "raw actor argmax over " + ", ".join(input_components)
    feature_dim = (256 + (4 if action_history else 0) +
                   (1 if collision_history else 0) +
                   (1 if distance_delta_feature else 0) +
                   (5 if local_geometry else 0) +
                   (4 if collision_risk_actions else 0) +
                   (4 if collision_risk_actions and future_collision_risk_actions else 0))
    sequence_length = int(training_config.get("sequence_length", 1))
    if algorithm == "CQL_LSTM":
        actor = LSTMActor(feature_dim, int(training_config["hidden_dim"]),
                          foundation=foundation).to(device)
    else:
        actor = _initial_policy_head(foundation, feature_dim).to(device)
    actor.load_state_dict(state["actor"])
    policy = OnlinePolicy(foundation, actor, device, action_history=action_history,
                          collision_history=collision_history,
                          collision_predictor=predictor,
                          collision_risk_actions=collision_risk_actions,
                          sequence_length=sequence_length,
                          distance_delta_feature=distance_delta_feature,
                          local_geometry=local_geometry,
                          future_collision_risk_actions=future_collision_risk_actions)
    output = (resolve_path(output_dir) if output_dir else
              resolve_path(config["OFFLINE_RL"]["OUTPUT"]) / "online_test")
    output.mkdir(parents=True, exist_ok=True)
    run_record = dict(
        foundation_checkpoint=str(foundation_path.resolve()),
        offline_checkpoint=str(offline_path.resolve()),
        offline_checkpoint_sha256=__import__("hashlib").sha256(
            offline_path.read_bytes()).hexdigest(),
        collision_predictor_checkpoint=(str(predictor_path.resolve())
                                        if collision_risk_actions else None),
        collision_predictor_sha256=(__import__("hashlib").sha256(
            predictor_path.read_bytes()).hexdigest() if collision_risk_actions else None),
        offline_epoch=state["epoch"], policy_training_config=state.get("config"),
        episodes_per_split=settings["episodes_per_split"],
        splits=split_names,
        index_range=list(index_range) if index_range is not None else None,
        config=config,
        action_handling=action_handling,
    )
    run_path = output / "run.json"
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        for key in ("foundation_checkpoint", "offline_checkpoint", "offline_epoch",
                    "offline_checkpoint_sha256", "collision_predictor_checkpoint",
                    "collision_predictor_sha256", "episodes_per_split", "splits",
                    "index_range", "action_handling"):
            if previous.get(key) != run_record.get(key):
                raise ValueError(f"Existing online evaluation output belongs to a different run: {key}")
    else:
        run_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    reports = [evaluate_split(split, policy, settings, int(raw_settings["SEED"]), output,
                              index_range)
               for split in split_names]
    report = dict(policy=f"AVNav frozen foundation + discrete {algorithm}",
                  offline_checkpoint=str(offline_path.resolve()), splits=reports)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiment/avnav/config/offline_rl.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--index-range", nargs=2, type=int, default=None,
                        metavar=("START", "END"),
                        help="Evaluate selected-set episode indices in [START, END)")
    args = parser.parse_args(argv)
    run(args.config, args.checkpoint or None, args.output or None,
        args.episodes, args.splits,
        tuple(args.index_range) if args.index_range is not None else None)


if __name__ == "__main__":
    main()
