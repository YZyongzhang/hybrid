"""Audit AVNav offline trajectory sources against canonical val/test manifests."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import yaml


def _digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _manifest_keys(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    keys = {(Path(str(row["scene_id"])).stem, str(row["episode_id"]))
            for row in payload["episodes"]}
    return keys


def audit(config_path, output_dir, dataset_root, additional_provenance=None):
    config_path = Path(config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sources = config["DATA"]["TRAIN_EPISODES"]
    if not isinstance(sources, (tuple, list)):
        sources = [sources]

    train_rows = []
    source_keys = {}
    for source in sources:
        root = Path(source).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Training source is missing: {root}")
        keys = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in (".pkl", ".pickle"):
                key = (path.parent.name, path.stem)
                keys.append(key)
                train_rows.append({"source": str(root), "file": str(path.resolve()),
                                   "scene": key[0], "episode_id": key[1]})
        source_keys[str(root)] = keys

    split_files = {
        "val_multiple": dataset_root / "val_multiple/val_multiple.json.gz",
        "val_multiple_unheard": dataset_root / "val_multiple_unheard/val_multiple_unheard.json.gz",
        "test_multiple": dataset_root / "test_multiple/test_multiple.json",
        "test_multiple_unheard": dataset_root / "test_multiple_unheard/test_multiple_unheard.json.gz",
    }
    split_keys = {name: _manifest_keys(path) for name, path in split_files.items()}
    additional_rows = []
    if additional_provenance:
        for manifest in additional_provenance:
            with Path(manifest).expanduser().resolve().open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    additional_rows.append({
                        "source_manifest": str(Path(manifest).expanduser().resolve()),
                        "scene": str(row["scene"]),
                        "episode_id": str(row["episode_id"]),
                    })

    all_train = {(row["scene"], row["episode_id"]) for row in train_rows}
    additional_train = {(row["scene"], row["episode_id"]) for row in additional_rows}
    all_train_with_additional = all_train | additional_train
    train_scenes = {scene for scene, _ in all_train}
    additional_train_scenes = {scene for scene, _ in additional_train}
    all_train_scenes = train_scenes | additional_train_scenes
    duplicate_counts = {}
    for row in train_rows:
        key = (row["scene"], row["episode_id"])
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
    duplicate_keys = {key: count for key, count in duplicate_counts.items() if count > 1}

    overlaps = {}
    for name, keys in split_keys.items():
        scenes = {scene for scene, _ in keys}
        overlaps[name] = {
            "manifest_episodes": len(keys),
            "episode_key_overlap_count": len(all_train & keys),
            "episode_key_overlap_examples": sorted(all_train & keys)[:100],
            "additional_rollout_parent_overlap_count": len(additional_train & keys),
            "additional_rollout_parent_overlap_examples": sorted(additional_train & keys)[:100],
            "additional_rollout_scene_overlap_count": len(additional_train_scenes & scenes),
            "additional_rollout_scene_overlap_examples": sorted(
                additional_train_scenes & scenes),
            "scene_overlap_count": len(all_train_scenes & scenes),
            "scene_overlap_examples": sorted(all_train_scenes & scenes),
            "manifest_sha256": _digest(split_files[name]),
        }

    report = {
        "status": "pass" if all(not row["episode_key_overlap_count"] and
                                 not row["additional_rollout_parent_overlap_count"] and
                                 not row["additional_rollout_scene_overlap_count"] and
                                 not row["scene_overlap_count"]
                                 for row in overlaps.values()) else "fail",
        "config": str(config_path),
        "training_sources": [
            {"path": source, "episodes": len(keys),
             "unique_episode_keys": len(set(keys)),
             "duplicate_episode_keys": len(keys) - len(set(keys))}
            for source, keys in source_keys.items()
        ],
        "training_episode_files": len(train_rows),
        "unique_training_episode_keys": len(all_train),
        "additional_train_only_rollout_parent_rows": len(additional_rows),
        "unique_additional_train_rollout_parent_keys": len(additional_train),
        "additional_rollout_parent_overlap_with_original_train_count": len(
            additional_train & all_train),
        "additional_rollout_parent_overlap_with_original_train_examples": sorted(
            additional_train & all_train)[:100],
        "duplicate_training_keys_across_sources": [
            {"scene": scene, "episode_id": episode_id, "copies": count}
            for (scene, episode_id), count in sorted(duplicate_keys.items())
        ],
        "split_audit": overlaps,
        "protocol": {
            "training_samples_read_from": "configured TRAIN_EPISODES only",
            "validation_and_test_used_for_training": False,
            "episode_identity": "scene directory stem + pickle filename stem",
            "scene_identity": "scene directory stem vs manifest scene_id stem",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_source_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "training_episode_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if report["status"] != "pass":
        raise RuntimeError("AVNav train source overlaps validation/test data; see audit report")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", default="data/datasets/audionav/mp3d/v1")
    parser.add_argument("--additional-provenance", nargs="*", default=None,
                        help="JSONL parent episode manifests for generated train-only rollouts")
    args = parser.parse_args()
    print(json.dumps(audit(args.config, args.output_dir, args.dataset_root,
                           args.additional_provenance),
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
