"""Collect diverse AVNav primitive-action episodes from the train split."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import traceback

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _audio(observation):
    value = observation["spectrogram"]
    if isinstance(value, (tuple, list)):
        value = value[0]
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.shape != (65, 26, 2):
        raise ValueError(f"Expected spectrogram [65, 26, 2], got {value.shape}")
    return torch.from_numpy(np.ascontiguousarray(value))


class AVNavCollector:
    def __init__(self, env, model, settings, device):
        self.env = env
        self.model = model
        self.settings = settings
        self.device = device
        self.model_probability = float(settings["MODEL_PROBABILITY"])
        self.oracle_probability = float(settings["ORACLE_PROBABILITY"])
        self.random_probability = float(settings["RANDOM_PROBABILITY"])
        if min(self.model_probability, self.oracle_probability, self.random_probability) < 0:
            raise ValueError("Policy mixture probabilities must be nonnegative")
        if abs(self.model_probability + self.oracle_probability + self.random_probability - 1) > 1e-6:
            raise ValueError("Policy mixture probabilities must sum to 1")
        self.recovery_steps = int(settings["RECOVERY_STEPS"])
        self.max_steps = int(settings["MAX_STEPS"])

    def _model_action(self, observation, previous_rgb, previous_depth):
        rgb = torch.as_tensor(np.asarray(observation["rgb"])).float().div_(255.0)
        depth = torch.as_tensor(np.asarray(observation["depth"])).float()
        if depth.ndim == 2:
            depth = depth.unsqueeze(-1)
        if previous_rgb is None:
            previous_rgb, previous_depth = torch.zeros_like(rgb), torch.zeros_like(depth)
        audio_batch = _audio(observation).unsqueeze(0).to(self.device)
        rgb_batch = torch.cat((previous_rgb, rgb), dim=-1).unsqueeze(0).to(self.device)
        depth_batch = torch.cat((previous_depth, depth), dim=-1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            _, _, logits = self.model(audio_batch, rgb_batch, depth_batch)
        return int(logits.argmax(dim=-1).item()), rgb, depth

    def _oracle_action(self):
        actions = self.env._env._sim.compute_oracle_actions() or []
        for action in actions:
            action = int(action)
            if action != 0:
                return action
        return 0

    def _choose_action(self, observation, previous_rgb, previous_depth,
                       recovery_remaining, recovery_success):
        if recovery_remaining > 0:
            # Turn away from the obstacle, then retry forward. A later policy
            # action resumes as soon as motion is no longer colliding.
            return (2 if recovery_remaining in (4, 2) else 1), "recovery", previous_rgb, previous_depth
        mode = random.random()
        if mode < self.model_probability:
            action, rgb, depth = self._model_action(observation, previous_rgb, previous_depth)
            return action, "model", rgb, depth
        if mode < self.model_probability + self.oracle_probability:
            return self._oracle_action(), "oracle", previous_rgb, previous_depth
        action = 0 if random.random() < 0.03 else random.choice((1, 2, 3))
        return action, "explore_stop" if action == 0 else "explore", previous_rgb, previous_depth

    def collect_episode(self, episode):
        self.env._env.current_episode = episode
        observation = self.env.reset()
        observations = [observation]
        actions, rewards, dones, infos = [], [], [], []
        behavior, collisions = [], []
        previous_rgb = previous_depth = None
        recovery_remaining = 0
        recovery_success = 0
        sim = self.env._env._sim
        for _ in range(self.max_steps):
            action, source, previous_rgb, previous_depth = self._choose_action(
                observation, previous_rgb, previous_depth,
                recovery_remaining, recovery_success)
            if source != "model":
                previous_rgb = torch.as_tensor(np.asarray(observation["rgb"])).float().div_(255.0)
                previous_depth = torch.as_tensor(np.asarray(observation["depth"])).float()
                if previous_depth.ndim == 2:
                    previous_depth = previous_depth.unsqueeze(-1)
            observation, reward, done, info = self.env.step(action=action)
            collided = bool(getattr(sim, "previous_step_collided", False))
            actions.append(int(action))
            observations.append(observation)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(dict(info or {}))
            behavior.append(source)
            collisions.append(collided)
            if source == "recovery":
                recovery_remaining = max(0, recovery_remaining - 1)
                if action == 1 and not collided:
                    recovery_success += 1
                    recovery_remaining = 0
                elif collided:
                    recovery_remaining = max(recovery_remaining, 2)
            elif collided:
                recovery_remaining = self.recovery_steps
            if done:
                break
        info = infos[-1] if infos else {}
        success = bool(info.get("success", False))
        source_sound = (episode.info or {}).get("sound", "unknown")
        return dict(sound_id=source_sound, obs=observations, action_id=actions,
                    reward=rewards, done=dones, info=infos,
                    behavior_policy=behavior, collision=collisions,
                    success=success, recovery_actions=behavior.count("recovery"),
                    recovery_success=recovery_success,
                    initial_distance=float(np.linalg.norm(
                        np.asarray(episode.start_position, dtype=np.float32) -
                        np.asarray(episode.goals[0].position, dtype=np.float32))),
                    status="success" if success else "timeout" if len(actions) >= self.max_steps else "failure")


def run(config_path):
    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    collection_name = str(settings.get("COLLECTION_NAME", ""))
    referenced = str(settings.get("COLLECT_CONFIG", "")).strip()
    if referenced:
        referenced_path = Path(referenced).expanduser()
        if not referenced_path.is_absolute():
            local_reference = config_path.parent / referenced_path
            referenced_path = (local_reference if local_reference.exists()
                               else ROOT / referenced_path)
        settings = yaml.safe_load(referenced_path.read_text(encoding="utf-8"))
    if collection_name:
        settings["COLLECTION_NAME"] = collection_name
    seed = int(settings["SEED"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(settings["DEVICE"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for AVNav collection but unavailable")
    torch.set_num_threads(2)

    from experiment.avwan.config.default import get_config
    from ss_baselines.common.environments import AudioNavRLEnv
    from soundspaces.datasets.audionav_dataset import AudioNavDataset
    from experiment.avnav.model.hybrid import Network

    env_config_path = Path(settings["ENV_CONFIG"])
    if not env_config_path.is_absolute():
        env_config_path = ROOT / env_config_path
    config = get_config(str(env_config_path))
    config.defrost()
    task = config.TASK_CONFIG.TASK
    task.SENSORS = ["SPECTROGRAM_SENSOR", "AngleSensor"]
    task.MEASUREMENTS = [name for name in task.MEASUREMENTS if name != "TOP_DOWN_MAP"]
    config.TASK_CONFIG.TRAIN.OPEN = False
    config.TASK_CONFIG.TEST.OPEN = False
    config.TASK_CONFIG.COLLECT.OPEN = False
    config.TASK_CONFIG.DATASET.SPLIT = "train_multiple"
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = list(settings["CONTENT_SCENES"])
    config.freeze()

    dataset = AudioNavDataset(config.TASK_CONFIG.DATASET)
    selection_path = Path(settings["SELECTION_PATH"])
    if not selection_path.is_absolute():
        selection_path = ROOT / selection_path
    requested = json.loads(selection_path.read_text(encoding="utf-8"))
    requested_keys = {(str(row["scene"]), str(row["episode_id"])) for row in requested}
    selected = [episode for episode in dataset.episodes
                if (Path(episode.scene_id).stem, str(episode.episode_id)) in requested_keys]
    selected_keys = {(Path(episode.scene_id).stem, str(episode.episode_id)) for episode in selected}
    if selected_keys != requested_keys:
        missing = sorted(requested_keys - selected_keys)[:10]
        raise ValueError(f"Selection has {len(requested_keys - selected_keys)} episodes absent "
                         f"from train_multiple: {missing}")
    if not selected:
        raise ValueError("No selected AVNav training episodes")

    checkpoint = Path(settings["CHECKPOINT"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    model = Network().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    env = AudioNavRLEnv(config, dataset)
    output = Path(settings["OUTPUT"]).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    episodes_dir = output / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    index_path = output / "episodes.jsonl"
    completed = set()
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") != "error" and (output / row["file"]).is_file():
                    completed.add((row["scene"], str(row["episode_id"])))
    run_record = dict(format_version=1, collection_name=collection_name,
                      policy="AVNavFoundationMixedRecovery",
                      seed=seed, split="train_multiple", planned=len(selected),
                      selection=str(selection_path), checkpoint=str(checkpoint),
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      content_scenes=list(settings["CONTENT_SCENES"]),
                      config=settings)
    _write_json(output / "run.json", run_record)
    collector = AVNavCollector(env, model, settings, device)
    summary = dict(format_version=1, planned=len(selected), completed=len(completed),
                   saved=len(completed), success=0, errors=0, transitions=0,
                   episodes_index="episodes.jsonl")
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            previous = json.loads(line)
            summary["success"] += int(previous.get("success", False))
            summary["errors"] += int(previous.get("status") == "error")
            summary["transitions"] += int(previous.get("transitions", 0))
    with index_path.open("a", encoding="utf-8") as index:
        for progress, episode in enumerate(selected, 1):
            scene, episode_id = Path(episode.scene_id).stem, str(episode.episode_id)
            if (scene, episode_id) in completed:
                continue
            relative_file = Path("episodes") / scene / f"{episode_id}.pkl"
            destination = output / relative_file
            destination.parent.mkdir(parents=True, exist_ok=True)
            row = dict(scene=scene, episode_id=episode_id,
                       file=relative_file.as_posix(), status="error", saved=False)
            try:
                data = collector.collect_episode(episode)
                row.update({key: data[key] for key in
                            ("status", "success", "initial_distance", "recovery_actions",
                             "recovery_success")})
                row["transitions"] = len(data["action_id"])
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                with temporary.open("wb") as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(destination)
                row["saved"] = True
            except Exception:
                row["error"] = traceback.format_exc()
            index.write(json.dumps(row) + "\n")
            index.flush()
            summary["completed"] += 1
            summary["saved"] += int(row["saved"])
            summary["success"] += int(row.get("success", False))
            summary["errors"] += int(row["status"] == "error")
            summary["transitions"] += int(row.get("transitions", 0))
            _write_json(output / "manifest.json", summary)
            print(f"{progress}/{len(selected)} {scene}_{episode_id}: {row['status']}, "
                  f"saved={row['saved']}", flush=True)
    env.close()
    return int(summary["errors"] > 0)
