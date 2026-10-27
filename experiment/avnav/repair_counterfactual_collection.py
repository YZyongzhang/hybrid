"""Retry failed train-only counterfactual episodes and merge verified labels."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from experiment.avnav.collect_collision_counterfactuals import collect


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repair(collection_root, repair_root=None):
    collection_root = Path(collection_root).expanduser().resolve()
    protocol = json.loads((collection_root / "collection_protocol.json").read_text())
    parent_path = collection_root / "parent_episode_manifest.jsonl"
    original_rows = [json.loads(line) for line in parent_path.read_text().splitlines()
                     if line.strip()]
    failed = [row for row in original_rows if row.get("error")]
    if not failed:
        return {"status": "no_repair_needed", "episodes": len(original_rows)}
    repair_root = Path(repair_root or collection_root.parent / "counterfactual_repair_train").resolve()
    repair_root.mkdir(parents=True, exist_ok=True)
    selection_manifest = repair_root / "failed_train_episodes.jsonl"
    selection_manifest.write_text("".join(json.dumps({
        "scene": row["scene"], "episode_id": row["episode_id"],
        "split": "train_multiple", "role": "train",
    }) + "\n" for row in failed), encoding="utf-8")
    repaired_root = repair_root / "collected"
    if not (repaired_root / "collection_summary.json").exists():
        summary = collect(
            protocol["config"], protocol["policy_checkpoint"],
            selection_manifest, repaired_root,
            episode_role="train", limit_episodes=0,
            split="train_multiple", max_steps=int(protocol["max_steps"]),
            horizon=int(protocol["future_horizon"]),
            max_anchors_per_episode=int(protocol["max_anchors_per_episode"]),
        )
    else:
        summary = json.loads((repaired_root / "collection_summary.json").read_text())
    if summary["totals"]["errors"]:
        raise RuntimeError(f"Train-only counterfactual retry still has errors: {summary['error_rows']}")
    retry_rows = [json.loads(line) for line in
                  (repaired_root / "parent_episode_manifest.jsonl").read_text().splitlines()
                  if line.strip()]
    retry_by_key = {(row["scene"], row["episode_id"]): row for row in retry_rows}
    if set(retry_by_key) != {(row["scene"], row["episode_id"]) for row in failed}:
        raise ValueError("Retry output does not exactly cover the failed train episodes")

    for key, row in retry_by_key.items():
        if row.get("error"):
            raise RuntimeError(f"Retry failed for {key}: {row['error']}")
        for source_key in ("trajectory_file", "candidate_label_file"):
            source = Path(row[source_key])
            relative = source.relative_to(repaired_root)
            destination = collection_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            row[source_key] = str(destination.resolve())

    merged = []
    for row in original_rows:
        if row.get("error"):
            merged.append(retry_by_key[(row["scene"], row["episode_id"])])
        else:
            merged.append(row)
    temporary = parent_path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                         for row in merged), encoding="utf-8")
    temporary.replace(parent_path)

    totals = Counter()
    for row in merged:
        totals["episodes"] += 1
        totals["steps"] += int(row.get("steps", 0))
        totals["collisions"] += int(row.get("collisions", 0))
        totals["anchors"] += int(row.get("anchors", 0))
        totals["candidate_labels"] += int(row.get("candidate_labels", 0))
        totals["errors"] += int(row.get("error") is not None)
    if totals["errors"]:
        raise RuntimeError("Merged counterfactual train collection is still incomplete")
    merged_summary = {
        "status": "completed", "protocol": str((collection_root / "collection_protocol.json").resolve()),
        "parent_episode_manifest": str(parent_path.resolve()),
        "candidate_action_order": ["STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT"],
        "totals": dict(totals), "error_rows": [],
    }
    (collection_root / "collection_summary.json").write_text(
        json.dumps(merged_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fingerprints_path = collection_root / "source_fingerprints.json"
    fingerprints = json.loads(fingerprints_path.read_text()) if fingerprints_path.exists() else {}
    fingerprints["repair_collector_script"] = str(Path(__file__).resolve())
    fingerprints["repair_collector_script_sha256"] = _sha256(Path(__file__).resolve())
    fingerprints["repair_selection_manifest_sha256"] = _sha256(selection_manifest)
    fingerprints["repair_output_parent_manifest_sha256"] = _sha256(
        repaired_root / "parent_episode_manifest.jsonl")
    fingerprints_path.write_text(json.dumps(fingerprints, indent=2) + "\n")
    merge_record = {
        "status": "completed", "failed_episode_count": len(failed),
        "repaired_episode_keys": [{"scene": row["scene"], "episode_id": row["episode_id"]}
                                   for row in retry_rows],
        "repair_collection": str(repaired_root),
        "main_parent_manifest_sha256": _sha256(parent_path),
        "repair_selection_manifest_sha256": _sha256(selection_manifest),
        "errors_after_merge": int(totals["errors"]),
    }
    (repair_root / "repair_merge.json").write_text(
        json.dumps(merge_record, indent=2) + "\n", encoding="utf-8")
    return merge_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", required=True)
    parser.add_argument("--repair-root", default="")
    args = parser.parse_args()
    result = repair(args.collection_root, args.repair_root or None)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
