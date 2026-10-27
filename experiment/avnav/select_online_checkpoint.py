"""Select an AVNav IQL epoch using raw-action validation SPL only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select(config, checkpoint_dir, output, episodes=30, workers=2):
    config = Path(config).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints in {checkpoint_dir}")

    evaluated = []
    for checkpoint in checkpoints:
        name = checkpoint.stem
        candidate_output = output / name
        log_path = output / f"{name}.log"
        command = [
            sys.executable, "-m", "experiment.avnav.run_full_test_parallel",
            "--config", str(config), "--checkpoint", str(checkpoint),
            "--output", str(candidate_output),
            "--splits", "val_multiple", "val_multiple_unheard",
            "--episodes", str(episodes), "--workers", str(workers),
            "--max-concurrent", str(workers),
        ]
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                # Habitat can abort while releasing its native GL context after
                # it has already flushed all episode records. A resume run
                # validates those shard files and aggregates them; incomplete
                # shards are rerun by the same command.
                completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise subprocess.CalledProcessError(completed.returncode, command)
        report_path = candidate_output / "final/report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        splits = {row["split"]: row for row in report["splits"]}
        required = ("val_multiple", "val_multiple_unheard")
        if any(name not in splits for name in required):
            raise ValueError(f"Validation report misses required splits: {report_path}")
        rows = [splits[name] for name in required]
        mean_spl = sum(row["spl"] for row in rows) / len(rows)
        mean_sr = sum(row["success_rate"] for row in rows) / len(rows)
        errors = sum(row["errors"] for row in rows)
        evaluated.append({
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "epoch": int(checkpoint.stem.split("_")[-1]),
            "validation": {name: splits[name] for name in required},
            "mean_validation_spl": mean_spl,
            "mean_validation_success_rate": mean_sr,
            "errors": errors,
        })
        print(json.dumps(evaluated[-1], ensure_ascii=False), flush=True)

    evaluated.sort(key=lambda row: (row["errors"] == 0,
                                    row["mean_validation_spl"],
                                    row["mean_validation_success_rate"]), reverse=True)
    chosen = evaluated[0]
    result = {
        "status": "selected",
        "selection_rule": "maximize mean SPL across 30 held validation episodes per split; "
                          "tie-break by mean SR; any error-free candidate outranks one with errors",
        "validation_splits": ["val_multiple", "val_multiple_unheard"],
        "episodes_per_split": episodes,
        "action_handling": "raw policy outputs, no rule-based post-processing",
        "selected": chosen,
        "ranking": evaluated,
    }
    (output / "validation_selection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(select(args.config, args.checkpoint_dir, args.output,
                            args.episodes, args.workers), indent=2,
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
