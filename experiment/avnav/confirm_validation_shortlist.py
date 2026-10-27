"""Confirm the top raw-policy epoch candidates on a larger validation sample."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def confirm(config, selection_path, output, episodes=100, finalists=3, workers=2):
    config = Path(config).expanduser().resolve()
    selection_path = Path(selection_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    ranking = selection["ranking"]
    shortlist = ranking[:finalists]
    results = []
    for candidate in shortlist:
        checkpoint = Path(candidate["checkpoint"]).resolve()
        name = f"epoch_{candidate['epoch']:03d}"
        candidate_output = output / name
        log_path = output / f"{name}.log"
        output.mkdir(parents=True, exist_ok=True)
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
                completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise subprocess.CalledProcessError(completed.returncode, command)
        report = json.loads((candidate_output / "final/report.json").read_text(
            encoding="utf-8"))
        splits = {row["split"]: row for row in report["splits"]}
        expected = ("val_multiple", "val_multiple_unheard")
        if any(name not in splits or splits[name]["episodes"] != episodes for name in expected):
            raise ValueError(f"Incomplete full-validation shortlist report: {candidate_output}")
        rows = [splits[name] for name in expected]
        result = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "epoch": candidate["epoch"],
            "validation": {name: splits[name] for name in expected},
            "mean_validation_spl": sum(row["spl"] for row in rows) / len(rows),
            "mean_validation_success_rate": sum(row["success_rate"] for row in rows) / len(rows),
            "errors": sum(row["errors"] for row in rows),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    results.sort(key=lambda row: (row["errors"] == 0,
                                  row["mean_validation_spl"],
                                  row["mean_validation_success_rate"]), reverse=True)
    confirmation = {
        "status": "selected",
        "selection_rule": "screen all epochs at 30 episodes per validation split; confirm top "
                          f"{len(shortlist)} on {episodes} episodes per split; maximize mean SPL, "
                          "tie-break by mean SR, and prefer error-free results",
        "validation_splits": ["val_multiple", "val_multiple_unheard"],
        "episodes_per_split": episodes,
        "action_handling": "raw policy outputs, no rule-based post-processing",
        "shortlist": results,
        "selected": results[0],
    }
    (output / "validation_confirmation.json").write_text(
        json.dumps(confirmation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    selection["screen30_selected"] = selection["selected"]
    selection["screen30_episodes_per_split"] = selection["episodes_per_split"]
    selection["selection_rule"] = confirmation["selection_rule"]
    selection["episodes_per_split"] = episodes
    selection["finalist_confirmation"] = confirmation
    selection["selected"] = results[0]
    selection_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    return confirmation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(confirm(args.config, args.selection, args.output,
                             args.episodes, args.finalists, args.workers),
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
