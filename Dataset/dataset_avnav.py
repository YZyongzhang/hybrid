"""Convert trusted AVNav episode pickles and read tensor shards lazily."""
from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

FIELDS = ("audios", "rgb", "depth", "angles", "actions")


def get_files(path, suffixes):
    path = Path(path)
    files = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.is_file() and item.suffix in suffixes
    )
    files = [item for item in files if item.suffix in suffixes]
    if not files:
        raise FileNotFoundError(f"No {suffixes} files found at {path}")
    return files


def _observation(obs):
    rgb = torch.as_tensor(np.asarray(obs["rgb"])).float() / 255.0
    depth = torch.as_tensor(np.asarray(obs["depth"])).float()
    if depth.ndim == 2:
        depth = depth.unsqueeze(-1)
    audio = torch.as_tensor(np.asarray(obs["spectrogram"])).float()
    if audio.ndim == 4 and audio.shape[0] == 1:
        audio = audio[0]
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {tuple(rgb.shape)}")
    if depth.shape != (*rgb.shape[:2], 1):
        raise ValueError(f"Expected matching HWC depth, got {tuple(depth.shape)}")
    if audio.ndim != 3 or audio.shape[-1] != 2:
        raise ValueError(f"Expected (frequency, time, 2) spectrogram, got {tuple(audio.shape)}")
    angle = np.asarray(obs["angle"]).reshape(-1)
    if len(angle) < 2:
        raise ValueError("obs.angle must contain [distance, azimuth_radians]")
    return rgb, depth, audio, torch.tensor(np.degrees(angle[1]), dtype=torch.float32)


def convert_pickle_to_pt(source, destination, shard_size=256, shuffle=True, seed=42):
    """Preserve episode boundaries; shard_size is a target, not a hard limit.

    The last observation is terminal and has no action. At each episode start,
    the previous RGB/depth frame is zero. Only load pickles from trusted sources.
    """
    if shard_size < 1:
        raise ValueError("SHARD_SIZE must be positive")
    files = get_files(source, (".pkl", ".pickle"))
    if shuffle:
        random.Random(seed).shuffle(files)
    destination = Path(destination)
    if destination.exists() and any(destination.glob("*.pt")):
        raise FileExistsError(f"PT output already contains shards: {destination}; use a new directory")
    destination.mkdir(parents=True, exist_ok=True)
    buffer = {key: [] for key in (*FIELDS, "sound_ids")}
    sound_labels, shards = {}, []
    total = 0

    def flush():
        nonlocal total
        if not buffer["actions"]:
            return
        name = f"foundation_model_shard_{len(shards):05d}.pt"
        torch.save({key: torch.stack(value) for key, value in buffer.items()}, destination / name)
        size = len(buffer["actions"])
        shards.append({"file": name, "samples": size})
        total += size
        for value in buffer.values():
            value.clear()

    for file in tqdm(files, desc="Converting AVNav episodes"):
        with file.open("rb") as handle:
            data = pickle.load(handle)
        obs = data["obs"]
        steps = max(0, len(obs) - 1)
        actions = np.asarray(data["action_id"]).reshape(-1)
        if len(actions) != steps:
            raise ValueError(f"{file}: {steps} transitions but {len(actions)} actions")
        if buffer["actions"] and len(buffer["actions"]) + steps > shard_size:
            flush()
        sounds = np.asarray(data.get("sound_id", "unknown"), dtype=object).reshape(-1)
        if len(sounds) not in (1, steps, steps + 1):
            raise ValueError(f"{file}: sound_id must be an episode label or per-step labels")
        previous_rgb = previous_depth = None
        for index in range(steps):
            rgb, depth, audio, angle = _observation(obs[index])
            if previous_rgb is None:
                previous_rgb, previous_depth = torch.zeros_like(rgb), torch.zeros_like(depth)
            action = int(actions[index])
            if action != actions[index] or not 0 <= action < 4:
                raise ValueError(f"{file}: action must be an integer in [0, 3], got {actions[index]}")
            label = str(sounds[0 if len(sounds) == 1 else index])
            sound_id = sound_labels.setdefault(label, len(sound_labels))
            sample = (audio, torch.cat((previous_rgb, rgb), dim=-1),
                      torch.cat((previous_depth, depth), dim=-1), angle,
                      torch.tensor(action, dtype=torch.long), torch.tensor(sound_id, dtype=torch.long))
            for key, value in zip(buffer, sample):
                buffer[key].append(value)
            previous_rgb, previous_depth = rgb, depth
        if len(buffer["actions"]) >= shard_size:
            flush()
    flush()
    if not total:
        raise ValueError(f"No transitions found in {source}")
    manifest = {"samples": total, "shards": shards, "sound_label_to_id": sound_labels}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Converted {total} transitions into {len(shards)} shards at {destination}")
    return [destination / shard["file"] for shard in shards]


def load_two_frame_pt(cls, path, config):
    """Compatibility wrapper for the previous conversion entry point."""
    return convert_pickle_to_pt(path, config.TO_PATH, getattr(config, "SHARD_SIZE", 256))


class AVNavDataset(Dataset):
    """Return (audio, rgb, depth, angle_degrees, action) with an LRU shard cache."""
    def __init__(self, path, cache_size=2):
        if cache_size < 1:
            raise ValueError("CACHE_SIZE must be positive")
        self.files = list(map(Path, path)) if isinstance(path, (list, tuple)) else get_files(path, (".pt",))
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self.ends = []
        total = 0
        for file in self.files:
            shard = self._read(file)
            total += len(shard["actions"])
            self.ends.append(total)
        if not total:
            raise ValueError("PT dataset contains no samples")

    @staticmethod
    def _read(file):
        shard = torch.load(file, map_location="cpu", weights_only=True)
        if not isinstance(shard, dict) or any(key not in shard for key in FIELDS):
            raise ValueError(f"{file}: expected tensor fields {FIELDS}")
        count = shard["actions"].shape[0]
        for key in FIELDS:
            if not isinstance(shard[key], torch.Tensor) or shard[key].shape[0] != count:
                raise ValueError(f"{file}: inconsistent tensor field {key}")
        return shard

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.ends, index)
        offset = index - (self.ends[shard_index - 1] if shard_index else 0)
        if shard_index not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            self._cache[shard_index] = self._read(self.files[shard_index])
        self._cache.move_to_end(shard_index)
        shard = self._cache[shard_index]
        return tuple(shard[key][offset].float() if key != "actions" else shard[key][offset].long()
                     for key in FIELDS)
