"""Check completeness, action coverage, and replay integrity of train labels."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(collection_root, selection_manifest, output_path=None, episode_role="train"):
    collection_root = Path(collection_root).expanduser().resolve()
    selection_manifest = Path(selection_manifest).expanduser().resolve()
    parent_path = collection_root / "parent_episode_manifest.jsonl"
    parent_rows = [json.loads(line) for line in parent_path.read_text().splitlines()
                   if line.strip()]
    selected_rows = [json.loads(line) for line in selection_manifest.read_text().splitlines()
                     if line.strip() and json.loads(line).get("role") == episode_role]
    selected_keys = {(str(row["scene"]), str(row["episode_id"])) for row in selected_rows}
    parent_keys = {(str(row["scene"]), str(row["episode_id"])) for row in parent_rows}
    errors = [row for row in parent_rows if row.get("error")]
    labels = []
    for path in sorted((collection_root / "candidate_labels").rglob("*.jsonl")):
        labels.extend(json.loads(line) for line in path.read_text().splitlines()
                      if line.strip())
    anchors = {}
    for row in labels:
        key = (row["scene"], row["episode_id"], int(row["anchor_step"]))
        anchors.setdefault(key, []).append(row)
    incomplete = []
    for key, rows in anchors.items():
        action_ids = [int(row["candidate_action_id"]) for row in rows]
        if len(action_ids) != 4 or set(action_ids) != {0, 1, 2, 3}:
            incomplete.append({"anchor": list(key), "candidate_action_ids": action_ids})
    tag_counts = Counter()
    for rows in anchors.values():
        for tag in rows[0].get("event_tags", []):
            tag_counts[tag] += 1
    replay_divergences = [int(row.get("anchor_replay_policy_divergences", 0)) for row in labels]
    replay_depth_mae = [float(row.get("anchor_replay_depth_mae", 0.0)) for row in labels]
    immediate = [bool(row["immediate_collision"]) for row in labels]
    future = [bool(row["future_any_collision_after_first"]) for row in labels]
    progress = [float(row["movement_progress_m"]) for row in labels]
    expected_episodes = len(selected_keys)
    missing = sorted(selected_keys - parent_keys)
    extra = sorted(parent_keys - selected_keys)
    trajectories = list((collection_root / "trajectories").rglob("*.pkl"))
    status = "pass" if (not errors and not missing and not extra and not incomplete and
                         len(parent_rows) == expected_episodes and
                         len(trajectories) == expected_episodes and
                         all(value == 0 for value in replay_divergences) and
                         (not replay_depth_mae or max(replay_depth_mae) <= 0.02)) else "fail"
    report = {
        "status": status,
        "collection_protocol": str((collection_root / "collection_protocol.json").resolve()),
        "collection_protocol_sha256": _sha256(collection_root / "collection_protocol.json"),
        "selection_manifest": str(selection_manifest),
        "selection_manifest_sha256": _sha256(selection_manifest),
        "parent_episode_manifest": str(parent_path),
        "parent_episode_manifest_sha256": _sha256(parent_path),
        "episode_role": episode_role,
        "selected_episode_count": expected_episodes,
        "train_episode_count": expected_episodes if episode_role == "train" else 0,
        "completed_parent_episode_count": len(parent_rows),
        "trajectory_file_count": len(trajectories),
        "train_parent_key_missing_count": len(missing),
        "train_parent_key_extra_count": len(extra),
        "episode_errors": errors,
        "anchor_count": len(anchors),
        "candidate_action_label_count": len(labels),
        "incomplete_four_action_anchor_count": len(incomplete),
        "incomplete_anchor_examples": incomplete[:50],
        "event_anchor_counts": dict(tag_counts),
        "candidate_actions": {
            str(action_id): sum(int(row["candidate_action_id"] == action_id) for row in labels)
            for action_id in range(4)
        },
        "immediate_collision_labels": sum(immediate),
        "immediate_collision_rate": sum(immediate) / max(len(immediate), 1),
        "future_collision_labels": sum(future),
        "future_collision_rate": sum(future) / max(len(future), 1),
        "mean_movement_progress_m": sum(progress) / max(len(progress), 1),
        "max_anchor_replay_policy_divergences": max(replay_divergences, default=0),
        "max_anchor_replay_depth_mae": max(replay_depth_mae, default=0.0),
        "protocol": {
            "train_only": episode_role == "train",
            "validation_only_for_model_selection": episode_role == "validation",
            "validation_labels_used_for_training": False,
            "candidate_actions_per_anchor": 4,
            "action_names": ["STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT"],
            "evaluation_actions_overridden": False,
        },
    }
    destination = Path(output_path) if output_path else collection_root / "collection_quality_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    if status != "pass":
        raise RuntimeError(f"Counterfactual collection audit failed: {destination}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--episode-role", default="train")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    print(json.dumps(audit(args.collection_root, args.selection_manifest,
                           args.output or None, args.episode_role),
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
