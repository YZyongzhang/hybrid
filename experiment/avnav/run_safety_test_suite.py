"""Run the validation-frozen safety CQL-LSTM ablations on complete test splits."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

from experiment.avnav.evaluation_protocol import episode_manifest_hashes

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_VARIANTS = {
    "current_cql_lstm", "collision_history",
    "counterfactual_labels", "collision_cost_critic",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(selection_path, output_root, episodes=1000, workers=2):
    selection_path = Path(selection_path).expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "selected" or selection.get("test_data_used") is not False:
        raise ValueError("Test is locked until a validation-only selection is complete")
    protocol_path = Path(selection["selection_protocol"]).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("test_data_used") is not False:
        raise ValueError("Validation protocol does not certify test isolation")
    confirmed = selection.get("confirmation", [])
    by_variant = {row.get("variant"): row for row in confirmed}
    if set(by_variant) != REQUIRED_VARIANTS:
        raise ValueError(f"Expected validation confirmation for every ablation: {set(by_variant)}")
    if int(protocol.get("confirmation_episodes_per_split", 0)) != 500:
        raise ValueError("Full test requires 500 validation episodes per split for every ablation")

    reference_config = Path(by_variant["current_cql_lstm"]["config"]).resolve()
    reference_config_payload = yaml.safe_load(reference_config.read_text(encoding="utf-8"))
    canonical_test_manifests = episode_manifest_hashes(
        reference_config_payload["EVALUATION"]["CONFIG"],
        ("test_multiple", "test_multiple_unheard"),
    )

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    suite_context = {
        "status": "test_protocol_frozen_before_evaluation",
        "validation_selection": str(selection_path),
        "validation_selection_sha256": _sha256(selection_path),
        "validation_selection_protocol": str(protocol_path),
        "validation_selection_protocol_sha256": _sha256(protocol_path),
        "test_splits": ["test_multiple", "test_multiple_unheard"],
        "canonical_test_episode_manifests": canonical_test_manifests,
        "episodes_per_split": int(episodes), "workers": int(workers),
        "test_metrics_used_for_selection_or_training": False,
        "action_handling": "raw policy actor argmax; no action override or recovery",
    }
    suite_path = output_root / "test_suite_context.json"
    if suite_path.exists():
        if json.loads(suite_path.read_text(encoding="utf-8")) != suite_context:
            raise ValueError("Existing test suite has a different frozen protocol")
    else:
        suite_path.write_text(json.dumps(suite_context, indent=2) + "\n", encoding="utf-8")

    results = {}
    for variant in sorted(REQUIRED_VARIANTS):
        row = by_variant[variant]
        checkpoint = Path(row["checkpoint"]).resolve()
        config = Path(row["config"]).resolve()
        if _sha256(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after validation selection: {checkpoint}")
        validation_context_path = Path(row["output"]).resolve() / "validation_context.json"
        validation_context = json.loads(validation_context_path.read_text(encoding="utf-8"))
        if validation_context.get("config_sha256") != _sha256(config):
            raise ValueError(f"Evaluation config changed after validation: {config}")
        if validation_context.get("checkpoint_sha256") != row["checkpoint_sha256"]:
            raise ValueError(f"Validation checkpoint hash does not match selection: {checkpoint}")
        context = {
            "variant": variant, "candidate": row["candidate"],
            "config": str(config), "config_sha256": _sha256(config),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "validation_safety_spl_score": row["safety_spl_score"],
            "validation_mean_spl": row["mean_spl"],
            "validation_mean_success_rate": row["mean_success_rate"],
            "validation_mean_collision_rate": row["mean_collisions_per_action"],
            "validation_confirmation_episodes_per_split": 500,
            "test_splits": suite_context["test_splits"],
            "canonical_test_episode_manifests": canonical_test_manifests,
            "test_episodes_per_split": int(episodes),
            "test_metrics_used_for_selection": False,
            "action_handling": suite_context["action_handling"],
        }
        output = output_root / variant
        output.mkdir(parents=True, exist_ok=True)
        context_path = output / "run_context.json"
        if context_path.exists():
            if json.loads(context_path.read_text(encoding="utf-8")) != context:
                raise ValueError(f"Existing test output belongs to a different frozen run: {output}")
        else:
            context_path.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
        report_path = output / "final" / "report.json"
        if not report_path.is_file():
            command = [sys.executable, "-m", "experiment.avnav.run_full_test_parallel",
                       "--config", str(config), "--checkpoint", str(checkpoint),
                       "--output", str(output), "--splits", "test_multiple",
                       "test_multiple_unheard", "--episodes", str(episodes),
                       "--workers", str(workers), "--max-concurrent", str(workers)]
            with (output / "runner.log").open("a", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, check=False)
            if completed.returncode and not report_path.is_file():
                raise subprocess.CalledProcessError(completed.returncode, command)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        splits = {item["split"]: item for item in report["splits"]}
        for split in suite_context["test_splits"]:
            if splits.get(split, {}).get("episodes") != int(episodes):
                raise ValueError(f"Incomplete {split} test report for {variant}")
        results[variant] = {
            "checkpoint": str(checkpoint), "validation_candidate": row,
            "test": [splits[name] for name in suite_context["test_splits"]],
            "evaluated_episode_manifests": {
                name: {
                    "path": str((output / "final" / f"{name}_episodes.jsonl").resolve()),
                    "sha256": _sha256(output / "final" / f"{name}_episodes.jsonl"),
                }
                for name in suite_context["test_splits"]
            },
        }
        print(json.dumps({"variant": variant, **results[variant]}, ensure_ascii=False), flush=True)

    comparison = {
        "status": "completed", "selection_source": "validation only",
        "test_metrics_used_for_selection_or_training": False,
        "action_handling": suite_context["action_handling"],
        "test_episodes_per_split": int(episodes), "variants": results,
    }
    (output_root / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.selection, args.output, args.episodes, args.workers),
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
