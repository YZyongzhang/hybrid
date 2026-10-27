"""Screen completed offline AVNav policies on held-out simulator validation splits."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def screen(sweep_path, episodes=100, top_k=0):
    sweep_path = Path(sweep_path).expanduser().resolve()
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    rows = [row for row in sweep["candidates"]
            if (Path(row["output"]) / "training_summary.json").is_file()]
    if top_k:
        rows = sorted(rows, key=lambda row: row["best_validation"]["action_nll"])[:top_k]
    results_path = sweep_path.with_name("validation_screen.json")
    previous = {}
    if results_path.is_file():
        previous = {row["name"]: row for row in json.loads(results_path.read_text(encoding="utf-8")).get("candidates", [])}
    results = {"status": "running", "episodes_per_split": int(episodes),
               "splits": ["val_multiple", "val_multiple_unheard"], "candidates": []}

    for candidate in rows:
        output = Path(candidate["output"]).resolve()
        validation_dir = output / "validation_rollouts"
        report_path = validation_dir / "report.json"
        if candidate["name"] in previous and report_path.is_file():
            results["candidates"].append(previous[candidate["name"]])
            continue
        config_path = output / "candidate_config.yaml"
        checkpoint = output / "best_policy.pt"
        log_path = output / "validation_screen.log"
        command = [sys.executable, "-m", "experiment.avnav.evaluate_online",
                   "--config", str(config_path), "--checkpoint", str(checkpoint),
                   "--output", str(validation_dir), "--episodes", str(episodes),
                   "--splits", "val_multiple", "val_multiple_unheard"]
        print(f"Screening AVNav candidate {candidate['name']}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       check=False)
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            row = dict(name=candidate["name"], status="completed",
                       exit_code=completed.returncode, output=str(output),
                       splits=report["splits"])
        else:
            row = dict(name=candidate["name"], status="failed",
                       exit_code=completed.returncode, output=str(output),
                       log=str(log_path))
        results["candidates"].append(row)
        _write_json(results_path, results)

    results["status"] = "completed" if all(
        row["status"] == "completed" for row in results["candidates"]
    ) else "incomplete"
    _write_json(results_path, results)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="result_avnav/2026-10-18-avnav-diversity/offline/candidates/sweep.json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=0,
                        help="Screen all candidates by default; set 0 to include every trained run")
    args = parser.parse_args(argv)
    print(json.dumps(screen(args.sweep, args.episodes, args.top_k), indent=2), flush=True)


if __name__ == "__main__":
    main()
