"""Select a diverse, train-only AVNav episode set for additional collection."""
from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _distance_bin(distance):
    if distance < 5.0:
        return "short"
    if distance < 10.0:
        return "medium"
    return "long"


def _orientation_bin(rotation):
    # Habitat stores quaternions as x, y, z, w.
    x, y, z, w = (float(value) for value in rotation)
    yaw = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
    return int((yaw % (2.0 * math.pi)) / (math.pi / 4.0)) % 8


def select(config_path, count, output, seed=20261018, exclude_index=None,
           distance_fractions=None):
    import yaml

    settings = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_config_path = Path(settings["ENV_CONFIG"]).expanduser()
    if not env_config_path.is_absolute():
        env_config_path = ROOT / env_config_path
    env_config = yaml.safe_load(env_config_path.read_text(encoding="utf-8"))
    dataset = env_config["DATASET"]
    split_path = Path(dataset["DATA_PATH"].format(
        version=dataset["VERSION"], split="train_multiple"))
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    if not split_path.exists() and split_path.suffix == ".gz":
        split_path = split_path.with_suffix("")
    opener = gzip.open if split_path.suffix == ".gz" else open
    with opener(split_path, "rt", encoding="utf-8") as handle:
        master_records = json.load(handle).get("episodes", [])
    content_paths = sorted((split_path.parent / "content").glob("*.json*"))
    if master_records:
        content_paths = []

    excluded = set()
    if exclude_index:
        with Path(exclude_index).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    excluded.add((str(row["scene"]), str(row["episode_id"])))
    group_counts, sound_counts_by_group, orientation_counts_by_group = {}, {}, {}
    eligible = 0

    def rows_from_content():
        if master_records:
            yield from master_records
            return
        for content in content_paths:
            if content.suffix not in (".json", ".gz"):
                continue
            content_opener = gzip.open if content.suffix == ".gz" else open
            with content_opener(content, "rt", encoding="utf-8") as handle:
                yield from json.load(handle).get("episodes", [])

    def metadata(record):
        info = record.get("info") or {}
        distance = float(info.get("geodesic_distance", 0.0))
        return {
            "scene": Path(record["scene_id"]).stem,
            "episode_id": str(record["episode_id"]),
            "sound": str(info.get("sound", "unknown")),
            "distance": distance,
            "distance_bin": _distance_bin(distance),
            "orientation_bin": _orientation_bin(record["start_rotation"]),
        }

    for record in rows_from_content():
        row = metadata(record)
        if (row["scene"], row["episode_id"]) in excluded:
            continue
        eligible += 1
        group = (row["distance_bin"], row["scene"])
        group_counts[group] = group_counts.get(group, 0) + 1
        sound_key = (*group, row["sound"])
        orient_key = (*group, row["orientation_bin"])
        sound_counts_by_group[sound_key] = sound_counts_by_group.get(sound_key, 0) + 1
        orientation_counts_by_group[orient_key] = orientation_counts_by_group.get(orient_key, 0) + 1
    if count <= 0 or count > eligible:
        raise ValueError(f"Requested {count} episodes but only {eligible} train episodes are eligible")

    rng = random.Random(seed)
    distance_fraction = distance_fractions or {
        "short": 0.10, "medium": 0.15, "long": 0.75
    }
    if set(distance_fraction) != {"short", "medium", "long"}:
        raise ValueError("Distance fractions must define short, medium and long")
    if any(value < 0 for value in distance_fraction.values()) or not math.isclose(
            sum(distance_fraction.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Distance fractions must be nonnegative and sum to 1")
    quotas = {name: int(count * fraction) for name, fraction in distance_fraction.items()}
    remainder = count - sum(quotas.values())
    remainder_order = sorted(distance_fraction,
                             key=lambda name: (-(count * distance_fraction[name] - quotas[name]), name))
    for name in remainder_order[:remainder]:
        quotas[name] += 1

    group_quotas = {}
    for distance_bin, quota in quotas.items():
        available = {scene: value for (distance, scene), value in group_counts.items()
                     if distance == distance_bin}
        allocation = {scene: 0 for scene in available}
        for _ in range(quota):
            candidates = [scene for scene, capacity in available.items()
                          if allocation[scene] < capacity]
            if not candidates:
                raise ValueError(f"Not enough {distance_bin} training episodes for quota {quota}")
            scene = min(candidates, key=lambda key: (allocation[key], rng.random()))
            allocation[scene] += 1
        group_quotas.update({(distance_bin, scene): value
                             for scene, value in allocation.items() if value})

    heaps = {group: [] for group in group_quotas}
    selected_serial = 0
    for record in rows_from_content():
        row = metadata(record)
        if (row["scene"], row["episode_id"]) in excluded:
            continue
        group = (row["distance_bin"], row["scene"])
        if group not in group_quotas:
            continue
        sound_count = sound_counts_by_group[(*group, row["sound"])]
        orientation_count = orientation_counts_by_group[(*group, row["orientation_bin"])]
        group_count = group_counts[group]
        log_weight = (
            0.7 * math.log(max(group_count / sound_count, 1.0))
            + 0.3 * math.log(max(group_count / orientation_count, 1.0))
        )
        exponential = -math.log(max(rng.random(), 1e-15))
        score = math.log(exponential) - log_weight
        entry = (-score, selected_serial, row)
        selected_serial += 1
        heap = heaps[group]
        quota = group_quotas[group]
        if len(heap) < quota:
            heapq.heappush(heap, entry)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    selected = [entry[2] for heap in heaps.values() for entry in heap]
    if len(selected) != count:
        raise ValueError(f"Selected {len(selected)} episodes, expected {count}")
    rng.shuffle(selected)
    scene_counts, sound_counts = {}, {}
    distance_counts, orientation_counts = {}, {}
    for row in selected:
        scene_counts[row["scene"]] = scene_counts.get(row["scene"], 0) + 1
        sound_counts[row["sound"]] = sound_counts.get(row["sound"], 0) + 1
        distance_counts[row["distance_bin"]] = distance_counts.get(row["distance_bin"], 0) + 1
        orientation_counts[row["orientation_bin"]] = orientation_counts.get(row["orientation_bin"], 0) + 1

    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    summary = {
        "split": "train_multiple",
        "seed": seed,
        "eligible_episodes": eligible,
        "selected_episodes": len(selected),
        "scene_count": len(scene_counts),
        "sound_count": len(sound_counts),
        "scene_counts": dict(sorted(scene_counts.items())),
        "sound_counts": dict(sorted(sound_counts.items())),
        "distance_counts": distance_counts,
        "orientation_counts": orientation_counts,
        "output": str(output),
    }
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiment/avnav/config/collect_broad_20261018.yaml")
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--output", default="result_avnav/2026-10-18-avnav-diversity/data/selected_train_episodes.json")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--exclude-index", default="")
    parser.add_argument("--distance-fractions", default="long=0.75,medium=0.15,short=0.10",
                        help="Comma-separated distance-bin proportions")
    args = parser.parse_args(argv)
    fractions = {}
    for item in args.distance_fractions.split(","):
        name, value = item.split("=", 1)
        fractions[name.strip()] = float(value)
    select(args.config, args.count, args.output, args.seed,
           args.exclude_index or None, fractions)


if __name__ == "__main__":
    main()
