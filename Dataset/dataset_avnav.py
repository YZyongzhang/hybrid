"""Convert trusted AVNav episode pickles and read tensor shards lazily."""
from bisect import bisect_right
from collections import OrderedDict
import json
import hashlib
from pathlib import Path
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
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


def _observation(obs, angle_override=None):
    rgb = torch.as_tensor(np.asarray(obs["rgb"])).float() / 255.0
    depth = torch.as_tensor(np.asarray(obs["depth"])).float()
    if depth.ndim == 2:
        depth = depth.unsqueeze(-1)
    spectrogram = obs["spectrogram"]
    # The collector stores (spectrogram, waveform) for multiple-source audio.
    if isinstance(spectrogram, tuple) and len(spectrogram) == 2:
        spectrogram = spectrogram[0]
    audio = torch.as_tensor(np.asarray(spectrogram)).float()
    if audio.ndim == 4 and audio.shape[0] == 1:
        audio = audio[0]
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {tuple(rgb.shape)}")
    if depth.shape != (*rgb.shape[:2], 1):
        raise ValueError(f"Expected matching HWC depth, got {tuple(depth.shape)}")
    if audio.ndim != 3 or audio.shape[-1] != 2:
        raise ValueError(f"Expected (frequency, time, 2) spectrogram, got {tuple(audio.shape)}")
    angle_value = obs.get("angle") if angle_override is None else angle_override
    if angle_value is None:
        raise KeyError("obs.angle is required; provide a train-only angle label map if absent")
    angle = np.asarray(angle_value).reshape(-1)
    if len(angle) < 2:
        raise ValueError("obs.angle must contain [distance, azimuth_radians]")
    return rgb, depth, audio, torch.tensor(np.degrees(angle[1]), dtype=torch.float32)


def convert_pickle_to_pt(source, destination, shard_size=256, shuffle=True, seed=42,
                         angle_labels_path=None, sound_labels_path=None,
                         file_list=None):
    """Preserve episode boundaries; shard_size is a target, not a hard limit.

    The last observation is terminal and has no action. At each episode start,
    the previous RGB/depth frame is zero. Only load pickles from trusted sources.
    """
    if shard_size < 1:
        raise ValueError("SHARD_SIZE must be positive")
    files = ([Path(path).expanduser().resolve() for path in file_list]
             if file_list is not None else get_files(source, (".pkl", ".pickle")))
    if not files or any(not path.is_file() or path.suffix not in (".pkl", ".pickle")
                        for path in files):
        raise FileNotFoundError("Explicit AVNav episode file list is empty or invalid")
    if shuffle:
        random.Random(seed).shuffle(files)
    angle_labels = None
    parent_sound_labels = None
    angle_labels_path = Path(angle_labels_path).expanduser().resolve() if angle_labels_path else None
    sound_labels_path = Path(sound_labels_path).expanduser().resolve() if sound_labels_path else None
    if angle_labels_path:
        angle_labels = json.loads(angle_labels_path.read_text(encoding="utf-8"))
    if sound_labels_path:
        parent_sound_labels = json.loads(sound_labels_path.read_text(encoding="utf-8"))
    destination = Path(destination)
    if destination.exists() and any(destination.glob("*.pt")):
        raise FileExistsError(f"PT output already contains shards: {destination}; use a new directory")
    destination.mkdir(parents=True, exist_ok=True)
    buffer = {key: [] for key in (*FIELDS, "sound_ids")}
    sound_label_to_id, shards = {}, []
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
        task_id = str(data.get("task_id", file.stem))
        episode_angles = angle_labels.get(task_id) if angle_labels is not None else None
        if episode_angles is not None and len(episode_angles) != len(obs):
            raise ValueError(f"{file}: angle labels must cover every observation")
        if any("angle" not in row for row in obs) and episode_angles is None:
            raise ValueError(f"{file}: missing obs.angle and train-only angle labels")
        actions = np.asarray(data["action_id"]).reshape(-1)
        if len(actions) != steps:
            raise ValueError(f"{file}: {steps} transitions but {len(actions)} actions")
        if buffer["actions"] and len(buffer["actions"]) + steps > shard_size:
            flush()
        sound_value = data.get("sound_id")
        parent_key = f"{data.get('parent_scene', '')}/{data.get('parent_episode_id', '')}"
        if parent_sound_labels is not None and parent_key in parent_sound_labels:
            sound_value = [parent_sound_labels[parent_key]]
        if sound_value is None:
            sound_value = "unknown"
        sounds = np.asarray(sound_value, dtype=object).reshape(-1)
        if len(sounds) not in (1, steps, steps + 1):
            raise ValueError(f"{file}: sound_id must be an episode label or per-step labels")
        previous_rgb = previous_depth = None
        for index in range(steps):
            override = episode_angles[index] if episode_angles is not None else None
            rgb, depth, audio, angle = _observation(obs[index], angle_override=override)
            if previous_rgb is None:
                previous_rgb, previous_depth = torch.zeros_like(rgb), torch.zeros_like(depth)
            action = int(actions[index])
            if action != actions[index] or not 0 <= action < 4:
                raise ValueError(f"{file}: action must be an integer in [0, 3], got {actions[index]}")
            label = str(sounds[0 if len(sounds) == 1 else index])
            sound_id = sound_label_to_id.setdefault(label, len(sound_label_to_id))
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
    manifest = {"samples": total, "shards": shards, "sound_label_to_id": sound_label_to_id,
                "source": str(Path(source).resolve()), "episodes": len(files)}
    if angle_labels_path:
        manifest["angle_labels_source"] = str(angle_labels_path)
        manifest["angle_labels_sha256"] = hashlib.sha256(angle_labels_path.read_bytes()).hexdigest()
    if sound_labels_path:
        manifest["sound_labels_source"] = str(sound_labels_path)
        manifest["sound_labels_sha256"] = hashlib.sha256(sound_labels_path.read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Converted {total} transitions into {len(shards)} shards at {destination}")
    return [destination / shard["file"] for shard in shards]


def load_two_frame_pt(cls, path, config):
    """Compatibility wrapper for the previous conversion entry point."""
    return convert_pickle_to_pt(path, config.TO_PATH, getattr(config, "SHARD_SIZE", 256))


class AVNavDataset(Dataset):
    """Return (audio, rgb, depth, angle_degrees, action) with an LRU shard cache."""
    def __init__(self, path, cache_size=2, include_sound_ids=False):
        if cache_size < 1:
            raise ValueError("CACHE_SIZE must be positive")
        self.files = list(map(Path, path)) if isinstance(path, (list, tuple)) else get_files(path, (".pt",))
        self.cache_size = cache_size
        self.include_sound_ids = bool(include_sound_ids)
        self._cache = OrderedDict()
        self.ends = []
        total = 0
        manifests = {}
        for file in self.files:
            if file.parent not in manifests:
                manifest = file.parent / "manifest.json"
                manifests[file.parent] = ({item["file"]: item["samples"] for item in
                                          json.loads(manifest.read_text())["shards"]}
                                         if manifest.exists() else {})
            count = manifests[file.parent].get(file.name)
            if count is None:
                count = len(self._read(file)["actions"])
            if not isinstance(count, int) or count < 1:
                raise ValueError(f"{file}: invalid sample count {count}")
            total += count
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
            expected = self.ends[shard_index] - (self.ends[shard_index - 1] if shard_index else 0)
            if len(self._cache[shard_index]["actions"]) != expected:
                raise ValueError(f"{self.files[shard_index]}: manifest sample count mismatch")
        self._cache.move_to_end(shard_index)
        shard = self._cache[shard_index]
        values = tuple(shard[key][offset].float() if key != "actions" else shard[key][offset].long()
                       for key in FIELDS)
        if self.include_sound_ids:
            if "sound_ids" not in shard:
                raise ValueError(f"{self.files[shard_index]}: staged audio training requires sound_ids")
            return (*values, shard["sound_ids"][offset].long())
        return values


class ShardBatchSampler(Sampler):
    """Shuffle shards and their samples, reading each shard only once per worker/epoch."""
    def __init__(self, dataset, batch_size, seed=42):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.ends = dataset.ends
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        shards = list(range(len(self.ends)))
        rng.shuffle(shards)
        for shard in shards:
            start = self.ends[shard - 1] if shard else 0
            indices = list(range(start, self.ends[shard]))
            rng.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                yield indices[offset:offset + self.batch_size]

    def __len__(self):
        starts = [0] + self.ends[:-1]
        return sum((end - start + self.batch_size - 1) // self.batch_size
                   for start, end in zip(starts, self.ends))
