"""Audit and prepare train-only greedy supervision from avnav-codex."""
import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/data2/getuanhui/avnav-codex")
DEFAULT_RUN_DIR = DEFAULT_DATA_ROOT / "rl_experiment/greedy_foundation_saccql_g0p65_20261027"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _episode_manifest_keys(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {(Path(str(row["scene_id"])).stem, str(row["episode_id"]))
            for row in payload["episodes"]}


def _sound_label(episode_path):
    with Path(episode_path).open("rb") as handle:
        episode = pickle.load(handle)
    values = np.asarray(episode.get("sound_id", []), dtype=object).reshape(-1)
    if len(values) != 1 or not str(values[0]):
        raise ValueError(f"Expected one train-split sound_id in {episode_path}")
    return str(values[0])


def audit_and_link(data_root, run_dir, dataset_root):
    data_root = Path(data_root).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    audit_dir = run_dir / "audit"
    protocol_path = data_root / "collection_protocol.json"
    repair_path = data_root / "collision_repair_protocol.json"
    selection_path = data_root / "selection_manifest.jsonl"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    selection = _read_jsonl(selection_path)
    if protocol.get("status") != "frozen_before_collection":
        raise ValueError("Collection protocol was not frozen before collection")
    if protocol.get("source", {}).get("split") != "train_multiple" or \
            protocol.get("source", {}).get("validation_or_test_used") is not False:
        raise ValueError("Collection provenance is not explicitly train-only")
    if _sha256(selection_path) != protocol["selection_manifest_sha256"]:
        raise ValueError("Selection manifest differs from the frozen collection protocol")

    source_manifest_path = Path(protocol["source"]["manifest"]).expanduser().resolve()
    source_audit_path = Path(protocol["source"]["audit"]).expanduser().resolve()
    if _sha256(source_manifest_path) != protocol["source"]["manifest_sha256"]:
        raise ValueError("Parent training episode manifest hash mismatch")
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if source_audit.get("status") != "pass":
        raise ValueError("Parent training data source audit did not pass")
    parent_rows = _read_jsonl(source_manifest_path)
    source_parent_paths = {}
    for row in parent_rows:
        key = (str(row["scene"]), str(row["episode_id"]))
        if key in source_parent_paths:
            raise ValueError(f"Duplicate parent train episode key: {key}")
        source_parent_paths[key] = Path(row["file"]).resolve()

    selected_by_category = {name: {} for name in ("greedy", "deverse", "collision")}
    for row in selection:
        category = row["category"]
        if category not in selected_by_category:
            raise ValueError(f"Unexpected collection category: {category}")
        task_id = row["task_id"]
        if task_id in selected_by_category[category]:
            raise ValueError(f"Duplicate selection task: {task_id}")
        key = (str(row["scene"]), str(row["episode_id"]))
        if key not in source_parent_paths:
            raise ValueError(f"Selected episode is not in the source train manifest: {key}")
        selected_by_category[category][task_id] = row

    counts = {name: len(rows) for name, rows in selected_by_category.items()}
    expected_counts = protocol["category_counts"]
    if counts != expected_counts:
        raise ValueError(f"Selection counts differ from frozen plan: {counts} vs {expected_counts}")

    canonical_splits = {
        "val_multiple": dataset_root / "val_multiple/val_multiple.json.gz",
        "val_multiple_unheard": dataset_root / "val_multiple_unheard/val_multiple_unheard.json.gz",
        "test_multiple": dataset_root / "test_multiple/test_multiple.json",
        "test_multiple_unheard": dataset_root / "test_multiple_unheard/test_multiple_unheard.json.gz",
    }
    parent_keys = {(str(row["scene"]), str(row["episode_id"])) for row in selection}
    parent_scenes = {scene for scene, _ in parent_keys}
    split_details = {}
    for split, manifest_path in canonical_splits.items():
        keys = _episode_manifest_keys(manifest_path)
        split_scenes = {scene for scene, _ in keys}
        episode_overlap = sorted(parent_keys & keys)
        scene_overlap = sorted(parent_scenes & split_scenes)
        split_details[split] = {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "episodes": len(keys),
            "parent_episode_overlap_count": len(episode_overlap),
            "parent_episode_overlap_examples": episode_overlap[:20],
            "scene_overlap_count": len(scene_overlap),
            "scene_overlap_examples": scene_overlap[:20],
        }
        if episode_overlap or scene_overlap:
            raise ValueError(f"Train parent data overlaps {split}: {split_details[split]}")

    category_manifests = {}
    all_rows = {}
    for category in ("greedy", "deverse", "collision"):
        manifest_path = data_root / category / "manifest.jsonl"
        rows = _read_jsonl(manifest_path)
        if _sha256(manifest_path) != repair.get("result_manifest_sha256") and category == "collision":
            raise ValueError("Repaired collision manifest differs from its frozen repair protocol")
        manifest_by_task = {row["task_id"]: row for row in rows}
        if len(manifest_by_task) != len(rows):
            raise ValueError(f"Duplicate {category} result task IDs")
        if set(manifest_by_task) != set(selected_by_category[category]):
            raise ValueError(f"{category} manifest task IDs differ from the selection manifest")
        accepted = []
        for task_id, selected in selected_by_category[category].items():
            row = manifest_by_task[task_id]
            if (str(row["scene"]) != str(selected["scene"]) or
                    str(row["episode_id"]) != str(selected["episode_id"])):
                raise ValueError(f"{task_id}: result provenance differs from selection")
            if row.get("status") != "saved" or row.get("error") is not None:
                raise ValueError(f"{task_id}: missing or errored saved trajectory")
            path = Path(row["file"]).expanduser().resolve()
            if data_root not in path.parents or not path.is_file():
                raise ValueError(f"{task_id}: episode path is missing or outside data root")
            if category == "greedy" and not (
                    row.get("success") is True and int(row.get("collisions", 0)) == 0 and
                    int(row.get("recoveries", 0)) == 0 and int(row.get("unresolved", 0)) == 0):
                raise ValueError(f"{task_id}: non-expert/failed data reached greedy supervision")
            accepted.append({**row, "file": str(path), "parent_source": selected["parent_source"],
                             "parent_replicate": selected["parent_replicate"]})
        category_manifests[category] = {
            "manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
            "episodes_path": str(data_root / category / "episodes"),
            "episodes": len(rows), "accepted_episodes": len(accepted),
            "source_bytes": sum(Path(row["file"]).stat().st_size for row in accepted),
            "successes": sum(row.get("success") is True for row in accepted),
            "collision_steps": sum(int(row.get("collisions", 0)) for row in accepted),
            "recovery_events": sum(int(row.get("recoveries", 0)) for row in accepted),
        }
        all_rows[category] = accepted
    if repair.get("status") != "repair_completed":
        raise ValueError("Collision collection repair did not complete")

    expert_rows = all_rows["greedy"]
    expert_manifest_path = audit_dir / "greedy_expert_manifest.jsonl"
    parent_rows_path = audit_dir / "selected_train_parent_manifest.jsonl"
    sound_label_path = audit_dir / "parent_sound_labels.json"
    manifest_records = []
    for row in expert_rows:
        source = Path(row["file"]).resolve()
        manifest_records.append({
            "task_id": row["task_id"], "scene": str(row["scene"]),
            "episode_id": str(row["episode_id"]), "file": str(source),
            "success": True, "collisions": 0, "expert_source": "expert_oracle",
        })
    _write_jsonl(expert_manifest_path, manifest_records)
    _write_jsonl(parent_rows_path, selection)

    parent_sound_labels = {}
    for key, path in source_parent_paths.items():
        if key in parent_keys:
            parent_sound_labels[f"{key[0]}/{key[1]}"] = _sound_label(path)
    _write_json(sound_label_path, parent_sound_labels)

    report = {
        "status": "pass",
        "data_root": str(data_root),
        "run_dir": str(run_dir),
        "collection_protocol": str(protocol_path),
        "collection_protocol_sha256": _sha256(protocol_path),
        "collision_repair_protocol": str(repair_path),
        "collision_repair_protocol_sha256": _sha256(repair_path),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "parent_training_manifest": str(source_manifest_path),
        "parent_training_manifest_sha256": _sha256(source_manifest_path),
        "parent_training_source_audit": str(source_audit_path),
        "parent_training_source_audit_sha256": _sha256(source_audit_path),
        "selected_unique_parent_episodes": len(parent_keys),
        "selected_parent_scenes": sorted(parent_scenes),
        "source_categories": category_manifests,
        "greedy_supervision": {
            "episodes": len(expert_rows),
            "successful": sum(row.get("success") is True for row in expert_rows),
            "collisions": sum(int(row.get("collisions", 0)) for row in expert_rows),
            "errors": 0,
            "manifest": str(expert_manifest_path),
            "manifest_sha256": _sha256(expert_manifest_path),
            "source_mode": "successful, collision-free greedy expert-oracle trajectories only",
        },
        "parent_sound_labels": {
            "path": str(sound_label_path), "sha256": _sha256(sound_label_path),
            "unique_classes": len(set(parent_sound_labels.values())),
            "source_split": "train_multiple parent pickles only",
        },
        "split_audit": split_details,
        "validation_or_test_training_data_used": False,
        "action_adapter": "none; recorded training actions are retained as collected",
    }
    _write_json(audit_dir / "data_preparation_audit.json", report)
    return report


def _capture_angles(data_root, run_dir, seed):
    """Replay expert actions in train scenes using the registered AngleSensor."""
    data_root = Path(data_root).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    audit_path = run_dir / "audit/data_preparation_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise ValueError("Train-only data audit must pass before extracting angle labels")
    rows = _read_jsonl(run_dir / "audit/greedy_expert_manifest.jsonl")
    rows.sort(key=lambda row: (row["scene"], int(row["episode_id"]), row["task_id"]))
    keys = sorted({(row["scene"], row["episode_id"]) for row in rows})
    labels_path = run_dir / "inputs/angle_labels.json"
    if labels_path.is_file():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if set(labels) == {row["task_id"] for row in rows}:
            return labels_path
        raise ValueError("Existing angle label file does not match the greedy manifest")

    # Load LLVM before Habitat-Sim's audio propagation library, as required by
    # this host's SoundSpaces runtime.
    import llvmlite.binding  # noqa: F401
    from experiment.avnav.config import resolve_path
    from experiment.avnav.environment.avnav_env import AudioNavRLEnv
    from experiment.avnav.environment.config import get_config
    from experiment.avnav.environment.evaluation_episodes import load_episodes

    config = get_config(task_config_path=str(resolve_path("experiment/avnav/config/audiogoal.yaml")))
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.SENSORS = []
    config.TASK_CONFIG.TASK.SENSORS = ["AngleSensor"]
    config.TASK_CONFIG.TRAIN.OPEN = False
    config.TASK_CONFIG.TEST.OPEN = False
    config.TASK_CONFIG.COLLECT.OPEN = False
    config.TASK_CONFIG.DATASET.SPLIT = "train_multiple"
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = sorted({scene for scene, _ in keys})
    config.freeze()
    dataset = load_episodes(config.TASK_CONFIG.DATASET, max_episodes=0, seed=int(seed),
                            episode_keys=set(keys))
    episode_by_key = {
        (Path(str(episode.scene_id)).stem, str(episode.episode_id)): episode
        for episode in dataset.episodes
    }
    if set(episode_by_key) != set(keys):
        raise ValueError("Angle replay did not load exactly the selected train episodes")
    env = AudioNavRLEnv(config, dataset)
    angle_labels = {}
    cached = {}
    try:
        for index, row in enumerate(rows, 1):
            path = Path(row["file"])
            with path.open("rb") as handle:
                episode = pickle.load(handle)
            task_id = str(episode.get("task_id", ""))
            parent_key = (str(episode.get("parent_scene", "")),
                          str(episode.get("parent_episode_id", "")))
            if task_id != row["task_id"] or parent_key != (row["scene"], row["episode_id"]):
                raise ValueError(f"{path}: collection parent identity differs from the audit")
            if episode.get("episode_success") is not True or int(episode.get("collision_count", -1)) != 0:
                raise ValueError(f"{path}: greedy training episode is not a clean expert success")
            if any(source != "expert_oracle" for source in episode.get("behavior_source", [])):
                raise ValueError(f"{path}: supervised actions include non-expert behavior")
            actions = [int(action) for action in episode["action_id"]]
            if len(episode["obs"]) != len(actions) + 1:
                raise ValueError(f"{path}: observation/action lengths do not align")
            if parent_key in cached and cached[parent_key][0] == actions:
                angle_labels[task_id] = cached[parent_key][1]
                continue

            env._env.current_episode = episode_by_key[parent_key]
            observation = env.reset()
            actual_key = (Path(str(env._env.current_episode.scene_id)).stem,
                          str(env._env.current_episode.episode_id))
            if actual_key != parent_key:
                raise ValueError(f"Replay reset changed train episode identity: {parent_key} -> {actual_key}")
            angles = []

            def add_angle(obs):
                if "angle" not in obs:
                    raise KeyError("Registered AngleSensor was not returned by SoundSpaces")
                value = np.asarray(obs["angle"], dtype=np.float32).reshape(-1)
                if len(value) != 2 or not np.isfinite(value).all():
                    raise ValueError(f"Invalid train-only angle label for {task_id}: {value}")
                angles.append(value.tolist())

            add_angle(observation)
            sim = env._env._sim
            for step, action in enumerate(actions):
                oracle = sim.compute_oracle_actions() or []
                if not oracle or int(oracle[0]) != action:
                    raise ValueError(f"{path}: step {step} is not the raw oracle action")
                observation, _, done, _ = env.step(action=action)
                if bool(getattr(sim, "previous_step_collided", False)):
                    raise ValueError(f"{path}: replayed greedy expert action collided at step {step}")
                if bool(done) != bool(episode["done"][step]):
                    raise ValueError(f"{path}: replay terminal flag differs at step {step}")
                add_angle(observation)
            if len(angles) != len(episode["obs"]):
                raise ValueError(f"{path}: replay angle labels do not cover the observations")
            angle_labels[task_id] = angles
            cached[parent_key] = (actions, angles)
            if index % 100 == 0 or index == len(rows):
                print(json.dumps({"stage": "angle_labels", "episodes": index,
                                  "total": len(rows), "unique_parents": len(cached)}),
                      flush=True)
    finally:
        env.close()

    if set(angle_labels) != {row["task_id"] for row in rows}:
        raise ValueError("Angle label coverage differs from the accepted greedy manifest")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(labels_path, angle_labels)
    _write_json(run_dir / "inputs/angle_labels_metadata.json", {
        "source": "SoundSpaces AngleSensor replay of raw greedy expert actions",
        "source_split": "train_multiple only",
        "episodes": len(angle_labels),
        "observations": sum(len(values) for values in angle_labels.values()),
        "labels_path": str(labels_path), "labels_sha256": _sha256(labels_path),
        "action_check": "every replayed greedy action matched compute_oracle_actions()[0]",
        "collision_check": "zero replayed collisions",
        "validation_or_test_used": False,
    })
    return labels_path


def convert_foundation_data(run_dir, shard_size=256, seed=20261027):
    run_dir = Path(run_dir).expanduser().resolve()
    from Dataset.dataset_avnav import convert_pickle_to_pt

    audit = json.loads((run_dir / "audit/data_preparation_audit.json").read_text(
        encoding="utf-8"))
    source = Path(audit["source_categories"]["greedy"]["episodes_path"])
    accepted_rows = _read_jsonl(run_dir / "audit/greedy_expert_manifest.jsonl")
    accepted_files = [row["file"] for row in accepted_rows]
    destination = run_dir / "inputs/greedy_foundation_pt"
    angle_labels = run_dir / "inputs/angle_labels.json"
    sound_labels = run_dir / "audit/parent_sound_labels.json"
    if not angle_labels.is_file() or not sound_labels.is_file():
        raise FileNotFoundError("Train-only angle and parent sound labels must be prepared first")
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("angle_labels_sha256") != _sha256(angle_labels) or
                manifest.get("sound_labels_sha256") != _sha256(sound_labels) or
                manifest.get("episodes") != 5700):
            raise FileExistsError("Existing greedy PT dataset has mismatched provenance")
        return manifest
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to reuse partial PT output: {destination}")
    convert_pickle_to_pt(
        source, destination, shard_size=int(shard_size), shuffle=True, seed=int(seed),
        angle_labels_path=angle_labels, sound_labels_path=sound_labels,
        file_list=accepted_files,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("episodes") != 5700 or len(manifest.get("sound_label_to_id", {})) < 2:
        raise ValueError("Greedy PT conversion lost expert episodes or sound-class labels")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--dataset-root", default=str(ROOT / "data/datasets/audionav/mp3d/v1"))
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--stage", choices=("audit", "angles", "convert", "all"), default="all")
    parser.add_argument("--shard-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.stage in ("audit", "all"):
        print(json.dumps(audit_and_link(args.data_root, args.run_dir, args.dataset_root),
                         indent=2, ensure_ascii=False), flush=True)
    if args.stage in ("angles", "all"):
        angle_path = _capture_angles(args.data_root, args.run_dir, args.seed)
        print(json.dumps({"angle_labels": str(angle_path),
                          "sha256": _sha256(angle_path)}, ensure_ascii=False), flush=True)
    if args.stage in ("convert", "all"):
        print(json.dumps(convert_foundation_data(args.run_dir, args.shard_size, args.seed),
                         indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
