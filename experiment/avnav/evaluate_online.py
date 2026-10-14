"""Evaluate an AVNav CQL policy in closed-loop Habitat/SoundSpaces episodes."""
import argparse
from collections import defaultdict
from copy import deepcopy
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
from experiment.avnav.model.hybrid import Network


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
    def __init__(self, foundation, actor, device):
        self.foundation = foundation.eval()
        self.actor = actor.eval()
        self.device = device
        self.previous_rgb = None
        self.previous_depth = None

    def reset(self):
        self.previous_rgb = self.previous_depth = None

    @torch.inference_mode()
    def act(self, observation):
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
        action = int(self.actor(features).argmax(dim=-1).item())
        self.previous_rgb, self.previous_depth = rgb, depth
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


def evaluate_split(split, policy, settings, seed, output):
    from ss_baselines.common.environments import AudioNavRLEnv
    from experiment.avwan.config.default import get_config
    from experiment.avwan.evaluate import load_episodes

    config = get_config(opts=["TASK_CONFIG.DATASET.SPLIT", split])
    config.defrost()
    config.TASK_CONFIG.TRAIN.OPEN = False
    config.TASK_CONFIG.COLLECT.OPEN = False
    # The AVNAV primitive-action policy only consumes RGB, depth, and audio.
    # AV-WaN's optional map sensors are not registered by AudioNavRLEnv.
    config.TASK_CONFIG.TASK.SENSORS = ["SPECTROGRAM_SENSOR"]
    config.TASK_CONFIG.TASK.MEASUREMENTS = [
        name for name in config.TASK_CONFIG.TASK.MEASUREMENTS if name != "TOP_DOWN_MAP"
    ]
    config.freeze()
    dataset = load_episodes(config, max_episodes=int(settings["episodes_per_split"]), seed=seed)
    expected = len(dataset.episodes)
    env = AudioNavRLEnv(config, dataset)
    episodes_path = output / f"{split}_episodes.jsonl"
    rows = []
    if episodes_path.exists():
        rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) > expected or any(row.get("index") != index for index, row in enumerate(rows)):
            env.close()
            raise ValueError(f"Invalid or stale partial episode log: {episodes_path}")
    if rows:
        print(f"{split}: resuming after {len(rows)}/{expected} saved episodes", flush=True)
    max_steps = int(settings["max_steps"])
    try:
        for index in range(expected):
            try:
                observation = env.reset()
            except Exception as error:
                row = dict(split=split, index=index, scene="unknown",
                           episode_id=str(index), reward=0.0, steps=0, spl=0.0,
                           success=False, error=f"reset {type(error).__name__}: {error}")
                if index < len(rows):
                    continue
                rows.append(row)
                with episodes_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                continue
            if index < len(rows):
                continue
            policy.reset()
            episode = getattr(env._env, "current_episode", None)
            row = dict(split=split, index=index,
                       scene=Path(str(getattr(episode, "scene_id", "unknown"))).stem,
                       episode_id=str(getattr(episode, "episode_id", index)),
                       reward=0.0, steps=0, spl=0.0, success=False, error=None)
            info = {}
            try:
                done = False
                while not done and row["steps"] < max_steps:
                    action = policy.act(observation)
                    observation, reward, done, info = env.step(action=action)
                    row["reward"] += float(reward)
                    row["steps"] += 1
                row["spl"], row["success"] = _metrics(env, info)
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                print(json.dumps(row), flush=True)
            rows.append(row)
            with episodes_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            if (index + 1) % 10 == 0 or index + 1 == expected:
                print(f"{split}: {index + 1}/{expected} episodes", flush=True)
    finally:
        env.close()

    errors = sum(row["error"] is not None for row in rows)
    metrics = dict(split=split, episodes=len(rows), errors=errors,
                   success_rate=sum(row["success"] for row in rows) / max(1, len(rows)),
                   spl=sum(row["spl"] for row in rows) / max(1, len(rows)),
                   mean_reward=sum(row["reward"] for row in rows) / max(1, len(rows)),
                   mean_steps=sum(row["steps"] for row in rows) / max(1, len(rows)))
    (output / f"{split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run(config_path, checkpoint=None, output_dir=None, episodes=None, splits=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    raw_settings = config["EVALUATION"]
    settings = dict(episodes_per_split=int(raw_settings["EPISODES_PER_SPLIT"]),
                    max_steps=int(raw_settings["MAX_STEPS"]))
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
    actor = deepcopy(foundation.final.action_head).to(device)
    actor.load_state_dict(state["actor"])
    policy = OnlinePolicy(foundation, actor, device)
    output = (resolve_path(output_dir) if output_dir else
              resolve_path(config["OFFLINE_RL"]["OUTPUT"]) / "online_test")
    output.mkdir(parents=True, exist_ok=True)
    run_record = dict(
        foundation_checkpoint=str(foundation_path.resolve()),
        offline_checkpoint=str(offline_path.resolve()),
        offline_epoch=state["epoch"], policy_training_config=state.get("config"),
        episodes_per_split=settings["episodes_per_split"],
        splits=split_names, config=config,
    )
    run_path = output / "run.json"
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        for key in ("foundation_checkpoint", "offline_checkpoint", "offline_epoch",
                    "episodes_per_split", "splits"):
            if previous.get(key) != run_record.get(key):
                raise ValueError(f"Existing online evaluation output belongs to a different run: {key}")
    else:
        run_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    reports = [evaluate_split(split, policy, settings, int(raw_settings["SEED"]), output)
               for split in split_names]
    report = dict(policy="AVNav frozen foundation + discrete SAC-CQL",
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
    args = parser.parse_args(argv)
    run(args.config, args.checkpoint or None, args.output or None,
        args.episodes, args.splits)


if __name__ == "__main__":
    main()
