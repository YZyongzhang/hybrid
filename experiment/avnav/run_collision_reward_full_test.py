"""Run the full AVNav test only after collision-reward validation selection."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

from experiment.avnav.evaluation_protocol import episode_manifest_hashes

ROOT = Path(__file__).resolve().parents[2]
TEST_SPLITS = ("test_multiple", "test_multiple_unheard")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(selection_path, output_root, episodes=1000, workers=4):
    selection_path = Path(selection_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "selected" or selection.get("test_data_used") is not False:
        raise ValueError("Test is locked until validation-only selection is complete")

    protocol_path = Path(selection["selection_protocol"]).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("test_data_used") is not False:
        raise ValueError("Validation protocol does not certify test isolation")
    if int(protocol.get("confirmation_episodes_per_split", 0)) != 500:
        raise ValueError("Full test requires 500-episode confirmation on both validation splits")
    chosen = selection.get("selected", {})
    checkpoint = Path(chosen.get("checkpoint", "")).resolve()
    config = Path(chosen.get("config", "")).resolve()
    if not checkpoint.is_file() or _sha256(checkpoint) != chosen.get("checkpoint_sha256"):
        raise ValueError("Selected checkpoint is missing or changed after validation")
    if not config.is_file():
        raise FileNotFoundError(f"Selected evaluation config is missing: {config}")

    confirmation_context = Path(chosen["output"]).resolve() / "validation_context.json"
    context = json.loads(confirmation_context.read_text(encoding="utf-8"))
    if context.get("phase") != "confirm500" or context.get("episodes_per_split") != 500:
        raise ValueError("Selected checkpoint has no complete validation confirmation context")
    if context.get("config_sha256") != _sha256(config):
        raise ValueError("Selected evaluation config changed after validation")
    if context.get("checkpoint_sha256") != chosen["checkpoint_sha256"]:
        raise ValueError("Validation confirmation checkpoint hash does not match selection")
    if context.get("action_handling") != "raw policy actor argmax; no action override or recovery":
        raise ValueError("Validation used an unexpected action-handling protocol")

    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    test_manifests = episode_manifest_hashes(
        config_payload["EVALUATION"]["CONFIG"], TEST_SPLITS
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_context = {
        "status": "test_protocol_frozen_before_evaluation",
        "validation_selection": str(selection_path),
        "validation_selection_sha256": _sha256(selection_path),
        "validation_selection_protocol": str(protocol_path),
        "validation_selection_protocol_sha256": _sha256(protocol_path),
        "selected_candidate": chosen.get("candidate"),
        "config": str(config),
        "config_sha256": _sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": chosen["checkpoint_sha256"],
        "validation_confirmation_episodes_per_split": 500,
        "test_splits": list(TEST_SPLITS),
        "canonical_test_episode_manifests": test_manifests,
        "test_episodes_per_split": int(episodes),
        "workers": int(workers),
        "test_metrics_used_for_selection_or_training": False,
        "action_handling": "raw policy actor argmax; no action override or recovery",
    }
    context_path = output_root / "run_context.json"
    if context_path.exists():
        if json.loads(context_path.read_text(encoding="utf-8")) != run_context:
            raise ValueError("Existing test output belongs to a different frozen run")
    else:
        context_path.write_text(json.dumps(run_context, indent=2) + "\n", encoding="utf-8")

    report_path = output_root / "final" / "report.json"
    if not report_path.is_file():
        command = [sys.executable, "-m", "experiment.avnav.run_full_test_parallel",
                   "--config", str(config), "--checkpoint", str(checkpoint),
                   "--output", str(output_root), "--splits", *TEST_SPLITS,
                   "--episodes", str(episodes), "--workers", str(workers),
                   "--max-concurrent", str(workers)]
        with (output_root / "runner.log").open("a", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        if completed.returncode and not report_path.is_file():
            raise subprocess.CalledProcessError(completed.returncode, command)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = {row["split"]: row for row in report.get("splits", [])}
    if any(rows.get(split, {}).get("episodes") != int(episodes) for split in TEST_SPLITS):
        raise ValueError(f"Test report is incomplete: {report_path}")
    result = {
        "status": "completed",
        "test_metrics_used_for_selection_or_training": False,
        "action_handling": run_context["action_handling"],
        "selected_candidate": chosen.get("candidate"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": chosen["checkpoint_sha256"],
        "test": [rows[split] for split in TEST_SPLITS],
        "evaluated_episode_manifests": {
            split: {
                "path": str((output_root / "final" / f"{split}_episodes.jsonl").resolve()),
                "sha256": _sha256(output_root / "final" / f"{split}_episodes.jsonl"),
            }
            for split in TEST_SPLITS
        },
    }
    (output_root / "comparison_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.selection, args.output, args.episodes, args.workers),
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
