"""Waypoint transition storage and a simulator-free PyTorch dataset."""
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
import math

import numpy as np
import torch
from torch.utils.data import Dataset

FORMAT = "avwan_waypoint_v1"
# Exclude oracle goal position, angle and other privileged collector sensors.
POLICY_OBSERVATIONS = ("rgb", "depth", "spectrogram", "gm", "am", "action_map")
TRAIN_FIELDS = ("action", "reward", "discount", "terminated", "truncated", "num_steps",
                "action_mask", "next_action_mask")


def _to_cpu(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_cpu(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported transition value: {type(value).__name__}")


def _validate(transition):
    for key in (*TRAIN_FIELDS, "observation", "next_observation", "gamma", "done",
                "low_level_rewards", "low_level_actions", "undiscounted_reward", "episode_id", "scene_id"):
        if key not in transition:
            raise ValueError(f"Missing transition field: {key}")
    t = transition
    if not isinstance(t["num_steps"], int) or t["num_steps"] < 1:
        raise ValueError("num_steps must be positive")
    if t["num_steps"] != len(t["low_level_rewards"]) or t["num_steps"] != len(t["low_level_actions"]):
        raise ValueError("Primitive trace length differs from num_steps")
    if t["terminated"] and t["truncated"]:
        raise ValueError("A transition cannot be both terminated and truncated")
    if bool(t["done"]) != bool(t["terminated"] or t["truncated"]):
        raise ValueError("done must equal terminated or truncated")
    gamma = float(t["gamma"])
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    expected = {"discount": 0.0 if t["terminated"] else gamma ** t["num_steps"],
                "reward": sum(gamma ** i * r for i, r in enumerate(t["low_level_rewards"])),
                "undiscounted_reward": sum(t["low_level_rewards"])}
    for key, value in expected.items():
        if not math.isfinite(float(t[key])) or not math.isclose(t[key], value, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"Inconsistent SMDP {key}")
    mask = torch.as_tensor(t["action_mask"])
    next_mask = torch.as_tensor(t["next_action_mask"])
    if mask.ndim != 1 or next_mask.shape != mask.shape:
        raise ValueError("Action masks must have the same flat shape")
    if not isinstance(t["action"], int) or not 0 <= t["action"] < mask.numel():
        raise ValueError("Invalid waypoint action index")


def save_transitions(transitions, path):
    """Save collected transitions without pickle-only custom objects; never overwrite."""
    rows = [_to_cpu(t) for t in transitions]
    if not rows:
        raise ValueError("Cannot save an empty transition file")
    for row in rows:
        _validate(row)
    path = Path(path)
    if path.suffix != ".pt":
        raise ValueError("Transition files must use the .pt suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save({"format": FORMAT, "transitions": rows}, handle)
    return path


def load_transitions(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{path}: expected {FORMAT}; primitive-action AVNav PT is not waypoint RL data")
    rows = payload["transitions"]
    if not rows:
        raise ValueError(f"{path}: empty transitions")
    for row in rows:
        _validate(row)
    return rows


class AVWANOfflineDataset(Dataset):
    """Read saved waypoint transitions without importing Habitat or starting a sim.

    Returns fixed training fields; variable-length primitive traces remain in the
    file and are available through load_transitions(). Bellman target:
        target = batch['reward'] + batch['discount'] * next_value
    No additional gamma or (1 - done) factor should be applied.
    """
    def __init__(self, path, observation_keys=POLICY_OBSERVATIONS, cache_size=2):
        path = Path(path)
        self.files = [path] if path.is_file() else sorted(path.rglob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No offline waypoint PT files at {path}")
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.cache_size = cache_size
        self.observation_keys = tuple(observation_keys)
        self._cache = OrderedDict()
        self.ends = []
        count = 0
        for file in self.files:
            count += len(load_transitions(file))
            self.ends.append(count)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect_right(self.ends, index)
        if shard not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            self._cache[shard] = load_transitions(self.files[shard])
        self._cache.move_to_end(shard)
        row = self._cache[shard][index - (self.ends[shard - 1] if shard else 0)]
        sample = {key: torch.as_tensor(row[key]).clone() for key in TRAIN_FIELDS}
        for field in ("observation", "next_observation"):
            sample[field] = {}
            for key in self.observation_keys:
                if key not in row[field]:
                    continue
                value = row[field][key]
                # Multiple-source SoundSpaces stores (spectrogram, waveform).
                if key == "spectrogram" and isinstance(value, (list, tuple)):
                    value = value[0]
                sample[field][key] = torch.as_tensor(value).clone()
            if not sample[field]:
                raise ValueError(f"No requested policy observations in {field}")
        return sample
