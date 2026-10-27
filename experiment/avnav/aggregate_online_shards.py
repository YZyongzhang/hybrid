"""Merge independently evaluated AVNav episode shards into a complete report."""
import argparse
import json
from pathlib import Path

from experiment.avnav.evaluate_online import _summarize_rows


def _read_rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def aggregate(output_dir, shard_root, episodes=1000,
              splits=("test_multiple", "test_multiple_unheard")):
    output_dir = Path(output_dir).expanduser().resolve()
    shard_root = Path(shard_root).expanduser().resolve()
    reports = []
    for split in splits:
        episode_path = output_dir / f"{split}_episodes.jsonl"
        sources = [episode_path, *sorted(shard_root.rglob(f"{split}_episodes.jsonl"))]
        by_index = {}
        for source in sources:
            for row in _read_rows(source):
                if row.get("split") != split:
                    raise ValueError(f"Wrong split in {source}: {row.get('split')}")
                index = int(row["index"])
                if not 0 <= index < episodes:
                    raise ValueError(f"Episode index {index} outside [0, {episodes})")
                previous = by_index.get(index)
                if previous is not None and previous != row:
                    raise ValueError(f"Conflicting duplicate results for {split} index {index}")
                by_index[index] = row
        missing = sorted(set(range(episodes)) - set(by_index))
        if missing:
            raise ValueError(f"{split}: missing {len(missing)} episode results; first={missing[:10]}")
        rows = [by_index[index] for index in range(episodes)]
        temporary = episode_path.with_suffix(episode_path.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in rows),
                             encoding="utf-8")
        temporary.replace(episode_path)
        prediction_n = sum(int(row.get("collision_prediction_n", 0)) for row in rows)
        prediction_tp = sum(int(row.get("collision_prediction_tp", 0)) for row in rows)
        prediction_fp = sum(int(row.get("collision_prediction_fp", 0)) for row in rows)
        prediction_tn = sum(int(row.get("collision_prediction_tn", 0)) for row in rows)
        prediction_fn = sum(int(row.get("collision_prediction_fn", 0)) for row in rows)
        prediction_brier_sum = sum(
            float(row.get("collision_prediction_brier_sum", 0.0)) for row in rows
        )
        prediction_metrics = {
            "samples": prediction_n,
            "collision_rate": (prediction_tp + prediction_fn) / max(prediction_n, 1),
            "accuracy_at_0_5": (prediction_tp + prediction_tn) / max(prediction_n, 1),
            "precision_at_0_5": prediction_tp / max(prediction_tp + prediction_fp, 1),
            "recall_at_0_5": prediction_tp / max(prediction_tp + prediction_fn, 1),
            "specificity_at_0_5": prediction_tn / max(prediction_tn + prediction_fp, 1),
            "balanced_accuracy_at_0_5": (
                prediction_tp / max(prediction_tp + prediction_fn, 1) +
                prediction_tn / max(prediction_tn + prediction_fp, 1)
            ) / 2,
            "f1_at_0_5": 2 * prediction_tp / max(
                2 * prediction_tp + prediction_fp + prediction_fn, 1
            ),
            "brier_score": prediction_brier_sum / max(prediction_n, 1),
            "confusion_matrix": {
                "tn": prediction_tn, "fp": prediction_fp,
                "fn": prediction_fn, "tp": prediction_tp,
            },
        }
        error_count = sum(row.get("error") is not None for row in rows)
        valid_count = len(rows) - error_count
        total_steps = sum(float(row["steps"]) for row in rows)
        total_collisions = sum(float(row.get("collisions", 0)) for row in rows)
        safety = _summarize_rows(split, rows)
        report = dict(
            split=split,
            episodes=len(rows),
            errors=error_count,
            valid_episodes=valid_count,
            coverage_rate=valid_count / max(len(rows), 1),
            success_rate=sum(bool(row["success"]) for row in rows) / len(rows),
            spl=sum(float(row["spl"]) for row in rows) / len(rows),
            mean_reward=sum(float(row["reward"]) for row in rows) / len(rows),
            mean_steps=total_steps / len(rows),
            mean_collisions=total_collisions / len(rows),
            collisions_per_step=total_collisions / max(total_steps, 1.0),
            collisions_per_action=total_collisions / max(total_steps, 1.0),
            repeated_collision_episode_rate=safety["repeated_collision_episode_rate"],
            repeated_collision_steps_per_action=safety[
                "repeated_collision_steps_per_action"],
            no_progress_episode_rate=safety["no_progress_episode_rate"],
            no_progress_windows_per_action=safety["no_progress_windows_per_action"],
            action_collision_rates=safety["action_collision_rates"],
            selected_action_collision_prediction=prediction_metrics,
        )
        if "selected_action_future_collision_prediction" in safety:
            report["selected_action_future_collision_prediction"] = safety[
                "selected_action_future_collision_prediction"]
        (output_dir / f"{split}.json").write_text(json.dumps(report, indent=2) + "\n",
                                                   encoding="utf-8")
        reports.append(report)

    run_path = output_dir / "run.json"
    shard_run_paths = sorted(shard_root.rglob("run.json"))
    if run_path.is_file():
        run_record = json.loads(run_path.read_text(encoding="utf-8"))
    elif shard_run_paths:
        run_record = json.loads(shard_run_paths[0].read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"No run.json found in {output_dir} or {shard_root}")
    shard_ranges = sorted({
        tuple(record.get("index_range") or (0, episodes))
        for path in shard_run_paths
        for record in [json.loads(path.read_text(encoding="utf-8"))]
    })
    if not run_path.is_file():
        run_record["index_range"] = None
        run_record["index_ranges"] = [list(item) for item in shard_ranges]
        run_record["splits"] = list(splits)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_path.write_text(json.dumps(run_record, indent=2) + "\n", encoding="utf-8")
    result = dict(
        policy="AVNav frozen foundation + discrete " +
        str(run_record.get("policy_training_config", {}).get("algorithm", "policy")).upper(),
        offline_checkpoint=run_record["offline_checkpoint"],
        splits=reports,
        evaluation_shards=str(shard_root),
    )
    (output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n",
                                            encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True,
                        help="Final evaluation directory with the full selection and run.json")
    parser.add_argument("--shards", required=True,
                        help="Root directory containing independent shard outputs")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--splits", nargs="+",
                        default=["test_multiple", "test_multiple_unheard"])
    args = parser.parse_args(argv)
    print(json.dumps(aggregate(args.output, args.shards, args.episodes, args.splits),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
