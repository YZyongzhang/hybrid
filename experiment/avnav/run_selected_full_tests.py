"""Run complete test splits only after all raw-policy validation selections are frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "result_avnav/2026-10-22-avnav-rawpolicy"
VARIANTS = {
    "clean_iql": {
        "config": ROOT / "experiment/avnav/config/integrity_clean_iql_20261022.yaml",
        "selection": ARCHIVE / "validation/clean_iql_screen30/validation_selection.json",
    },
    "collision_replay_iql": {
        "config": ROOT / "experiment/avnav/config/integrity_clean_iql_20261022.yaml",
        "selection": ARCHIVE / "validation/collision_replay_iql_screen30/validation_selection.json",
    },
    "action_history_collision_iql": {
        "config": ROOT / "experiment/avnav/config/integrity_action_history_iql_20261022.yaml",
        "selection": ARCHIVE / "validation/action_history_collision_iql_screen30/validation_selection.json",
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _complete_report(path):
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = {row["split"]: row for row in data["splits"]}
        return all(metrics.get(split, {}).get("episodes") == 1000
                   for split in ("test_multiple", "test_multiple_unheard"))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(workers=2):
    missing = [str(row["selection"]) for row in VARIANTS.values()
               if not row["selection"].is_file()]
    if missing:
        raise FileNotFoundError(
            "Full test is locked until validation selects every model: " + ", ".join(missing)
        )
    frozen = {}
    for name, spec in VARIANTS.items():
        selection = json.loads(spec["selection"].read_text(encoding="utf-8"))
        confirmation = selection.get("finalist_confirmation")
        if (not confirmation or confirmation.get("status") != "selected" or
                confirmation.get("episodes_per_split") != 500 or
                selection.get("episodes_per_split") != 500):
            raise ValueError(
                f"Full test is locked until {name} has been selected on all 500 "
                "episodes of each validation split"
            )
        chosen = selection["selected"]
        if chosen.get("checkpoint") != confirmation.get("selected", {}).get("checkpoint"):
            raise ValueError(f"Selected checkpoint does not match full-validation result: {name}")
        checkpoint = Path(chosen["checkpoint"]).resolve()
        if _sha256(checkpoint) != chosen["checkpoint_sha256"]:
            raise ValueError(f"Selected checkpoint changed after validation: {checkpoint}")
        if selection.get("action_handling") != "raw policy outputs, no rule-based post-processing":
            raise ValueError(f"Unexpected action handling in {spec['selection']}")
        output = ARCHIVE / "test" / name
        output.mkdir(parents=True, exist_ok=True)
        context = {
            "variant": name,
            "config": str(spec["config"].resolve()),
            "selection": str(spec["selection"].resolve()),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": chosen["checkpoint_sha256"],
            "validation_mean_spl": chosen["mean_validation_spl"],
            "validation_mean_success_rate": chosen["mean_validation_success_rate"],
            "validation_episodes_per_split": selection["episodes_per_split"],
            "test_splits": ["test_multiple", "test_multiple_unheard"],
            "test_episodes_per_split": 1000,
            "workers": workers,
            "action_handling": selection["action_handling"],
            "test_used_for_selection_or_training": False,
        }
        context_path = output / "run_context.json"
        if context_path.exists():
            old = json.loads(context_path.read_text(encoding="utf-8"))
            if old != context:
                raise ValueError(f"Existing test output belongs to a different frozen run: {output}")
        else:
            context_path.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
        frozen[name] = (spec, checkpoint, output)

    running = []
    for name, (spec, checkpoint, output) in frozen.items():
        report_path = output / "final/report.json"
        if _complete_report(report_path):
            continue
        command = [
            sys.executable, "-m", "experiment.avnav.run_full_test_parallel",
            "--config", str(spec["config"].resolve()),
            "--checkpoint", str(checkpoint), "--output", str(output),
            "--splits", "test_multiple", "test_multiple_unheard",
            "--episodes", "1000", "--workers", str(workers),
            "--max-concurrent", str(workers),
        ]
        log = (output / "runner.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        running.append((name, command, process, log, output))

    failed = []
    for name, command, process, log, output in running:
        code = process.wait()
        log.close()
        report_path = output / "final/report.json"
        if code and not _complete_report(report_path):
            with (output / "runner.log").open("a", encoding="utf-8") as retry_log:
                retry = subprocess.run(command, cwd=ROOT, stdout=retry_log,
                                       stderr=subprocess.STDOUT, check=False)
            code = retry.returncode
        if code and not _complete_report(report_path):
            failed.append({"variant": name, "exit_code": code,
                           "output": str(output)})
    if failed:
        raise RuntimeError(f"Incomplete full-test outputs: {failed}")

    comparison = {"status": "completed", "test_metrics_used_for_selection": False,
                  "action_handling": "raw policy outputs, no rule-based post-processing",
                  "variants": {}}
    for name, (spec, checkpoint, output) in frozen.items():
        report = json.loads((output / "final/report.json").read_text(encoding="utf-8"))
        comparison["variants"][name] = {
            "checkpoint": str(checkpoint),
            "validation_selection": str(spec["selection"].resolve()),
            "test": report["splits"],
        }
    (ARCHIVE / "test/comparison_summary.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.workers), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
