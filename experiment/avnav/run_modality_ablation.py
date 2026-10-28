"""Run paired, zero-input AVNav modality ablations on validation splits only."""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


SPLITS = ("val_multiple", "val_multiple_unheard")
ABLATIONS = ("vision_only", "audio_only")
SHARD_COUNT = 2


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manifest_episode_order(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [(Path(str(row["scene_id"])).stem, str(row["episode_id"]))
            for row in payload.get("episodes", [])]


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _merge_split_shards(output_root, run_dir, ablation, split, shard_records, expected_ids):
    episodes = []
    errors = []
    shard_artifacts = []
    seen = set()
    shard_metadata = None
    for shard_index, shard_path, expected_count, episode_range in shard_records:
        record = _read_json(shard_path)
        if (record.get("status") != "pass" or not record.get("partial_validation_shard")
                or record.get("input_ablation") != ablation
                or record.get("checkpoint_sha256") is None
                or record.get("expected_episodes") != expected_count
                or record.get("evaluated_episodes") != expected_count
                or record.get("full_expected_episodes") != len(expected_ids)
                or record.get("evaluation_episode_range") != episode_range
                or record.get("evaluation_errors") != 0):
            raise RuntimeError(f"Invalid or incomplete validation shard: {shard_path}")
        shard_metadata = record
        shard_episodes = record["episode_metrics"]
        shard_seen = set()
        for row in shard_episodes:
            key = (str(row["scene"]), str(row["episode_id"]))
            if key in seen:
                raise RuntimeError(f"Duplicate episode across shards: {key}")
            seen.add(key)
            shard_seen.add(key)
            episodes.append(row)
        if len(shard_seen) != expected_count:
            raise RuntimeError(f"Shard has duplicate or missing episode rows: {shard_path}")
        errors.extend(record.get("errors", []))
        shard_artifacts.append({
            "shard": shard_index, "path": str(shard_path), "sha256": _sha256(shard_path),
            "episode_range": episode_range, "episodes": len(shard_episodes),
        })

    expected_id_set = set(expected_ids)
    missing = sorted(expected_id_set - seen)
    unexpected = sorted(seen - expected_id_set)
    if missing or unexpected or len(episodes) != len(expected_ids):
        raise RuntimeError(f"Shard identity coverage failed for {split}: missing={len(missing)}, unexpected={len(unexpected)}")
    if shard_metadata is None:
        raise RuntimeError(f"No shard results found for {ablation} {split}")
    episodes.sort(key=lambda row: (row["scene"], str(row["episode_id"])))
    success_count = sum(int(row["success"]) for row in episodes)
    collision_count = sum(int(row["collisions"]) for row in episodes)
    forward_actions = sum(int(row["action_counts"][1]) for row in episodes)
    result = {
        **shard_metadata,
        "status": "pass" if not errors and not missing and not unexpected else "fail",
        "expected_episodes": len(expected_ids), "full_expected_episodes": len(expected_ids),
        "evaluated_episodes": len(episodes), "evaluation_errors": len(errors),
        "missing_expected_ids": len(missing), "unexpected_ids": len(unexpected),
        "missing_id_examples": missing[:30], "unexpected_id_examples": unexpected[:30],
        "SR": success_count / max(len(expected_ids), 1),
        "SPL": sum(float(row["spl"]) for row in episodes) / max(len(expected_ids), 1),
        "mean_reward": sum(float(row["reward"]) for row in episodes) / max(len(episodes), 1),
        "mean_steps": sum(int(row["steps"]) for row in episodes) / max(len(episodes), 1),
        "collisions": collision_count,
        "forward_collision_rate": collision_count / max(forward_actions, 1),
        "timeout_rate": sum(int(bool(row["timeout"])) for row in episodes) / max(len(expected_ids), 1),
        "mean_success_path_efficiency": sum(float(row["spl"]) for row in episodes if row["success"])
        / max(success_count, 1),
        "episode_metrics": episodes, "errors": errors,
        "partial_validation_smoke": False, "partial_validation_shard": False,
        "evaluation_episode_range": [0, len(expected_ids)],
        "shard_aggregation": shard_artifacts,
    }
    output_dir = output_root / ablation
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_source = Path(shard_metadata["environment_path"])
    environment_destination = output_dir / f"{split}_environment.yaml"
    shutil.copy2(environment_source, environment_destination)
    result["environment_sha256"] = _sha256(environment_destination)
    result_path = output_dir / f"{split}.json"
    _atomic_json(result_path, result)
    with (output_dir / f"{split}_episode_ids.jsonl").open("w", encoding="utf-8") as handle:
        for row in episodes:
            handle.write(json.dumps({"scene": row["scene"], "episode_id": row["episode_id"]}) + "\n")
    return result_path


def run(args):
    run_dir = Path(args.run_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    training_config = Path(args.training_config).resolve()
    protocol = _read_json(run_dir / "run_protocol.json")
    source_audit = _read_json(run_dir / "audit/data_source_audit.json")
    output_root = run_dir / "evaluation/modality_ablation"
    output_root.mkdir(parents=True, exist_ok=True)

    # Pair the ablations with the only checkpoint currently backed by complete
    # heard/unheard closed-loop validation results in this interrupted run.
    baseline_root = run_dir / "evaluation/candidates/A_repaired_baseline/epoch_010"
    baseline = {}
    for split in SPLITS:
        baseline_path = baseline_root / f"{split}.json"
        record = _read_json(baseline_path)
        if (record.get("status") != "pass" or record.get("partial_validation_smoke")
                or record.get("evaluated_episodes") != record.get("expected_episodes")
                or record.get("evaluation_errors") != 0
                or record.get("checkpoint_sha256") != _sha256(checkpoint)):
            raise RuntimeError(f"Complete matching baseline evaluation missing or invalid: {baseline_path}")
        baseline[split] = {
            "path": str(baseline_path), "sha256": _sha256(baseline_path),
            "expected_episodes": record["expected_episodes"], "SR": record["SR"], "SPL": record["SPL"],
        }

    validation_manifests = {}
    for split in SPLITS:
        declared = source_audit["validation_and_test_split_manifests"][split]
        manifest_path = Path(declared["path"]).resolve()
        actual_hash = _sha256(manifest_path)
        if actual_hash != declared["sha256"]:
            raise RuntimeError(f"Validation manifest changed since provenance audit: {split}")
        id_path = run_dir / "audit/split_episode_ids" / f"{split}.jsonl"
        validation_manifests[split] = {
            "path": str(manifest_path), "sha256": actual_hash,
            "episode_count": declared["episode_count"],
            "split_id_manifest": str(id_path), "split_id_manifest_sha256": _sha256(id_path),
        }
        if declared["episode_count"] != baseline[split]["expected_episodes"]:
            raise RuntimeError(f"Baseline count differs from audited validation count: {split}")

    ablation_protocol = {
        "status": "running",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "objective": "Measure AVNav policy dependence on audio and RGB-D inputs with frozen weights",
        "scope": "Complete validation splits only; no test rollout, training, or checkpoint selection",
        "execution_history": [
            {
                "attempt": "single-process vision_only val_multiple",
                "status": "superseded_before_completion",
                "completed_episodes_before_stop": 300,
                "log": str(output_root / "logs/vision_only_val_multiple.log"),
                "final_metrics_used": False,
                "reason": "Restarted as disjoint parallel shards to use available machine resources.",
            },
            {
                "attempt": "four concurrent full-split workers",
                "status": "superseded_during_startup",
                "final_metrics_used": False,
                "reason": "Replaced by eight concurrent episode-shard workers before any split completed.",
            },
        ],
        "checkpoint_selection": (
            "Fixed A_repaired_baseline epoch_010: it is the checkpoint paired with complete, error-free "
            "heard and unheard validation baseline artifacts already present when this ablation was launched. "
            "This is not a claim that epoch_010 is the final candidate selected by the interrupted pipeline."
        ),
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "foundation_checkpoint": str(run_dir / "foundation/action_stage/best.pth"),
        "foundation_checkpoint_sha256": _sha256(run_dir / "foundation/action_stage/best.pth"),
        "training_config": str(training_config), "training_config_sha256": _sha256(training_config),
        "evaluation_code": {
            str(path.resolve()): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("evaluate_repaired.py").resolve(),
                Path(__file__).with_name("evaluate_codex_saccql_lstm.py").resolve(),
                Path(__file__).with_name("benchmark_runtime.py").resolve(),
            )
        },
        "ablation_inputs": {
            "vision_only": "Zero binaural spectrogram at every step; retain RGB and depth.",
            "audio_only": "Zero RGB and depth at every step; retain binaural spectrogram.",
        },
        "action_handling": "Deterministic actor argmax sent directly to the environment; no action overrides or post-processing",
        "evaluation_seed": protocol["evaluation"]["seed"],
        "max_steps": protocol["evaluation"]["max_steps"],
        "parallelism": {
            "concurrent_jobs": len(ABLATIONS) * len(SPLITS) * SHARD_COUNT,
            "unit": "one episode shard per job, with two equal disjoint shards per split",
            "episodes_per_shard": {
                split: (validation_manifests[split]["episode_count"] + SHARD_COUNT - 1) // SHARD_COUNT
                for split in SPLITS
            },
            "per_job_threads": {"OMP_NUM_THREADS": 2, "MKL_NUM_THREADS": 2},
        },
        "validation_manifests": validation_manifests,
        "multimodal_baseline": baseline,
        "test_data_used": False,
    }
    _atomic_json(output_root / "protocol.json", ablation_protocol)

    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    shard_records_by_evaluation = {}
    for ablation in ABLATIONS:
        output_dir = output_root / ablation
        for split in SPLITS:
            expected_total = validation_manifests[split]["episode_count"]
            shard_size = (expected_total + SHARD_COUNT - 1) // SHARD_COUNT
            result_path = output_dir / f"{split}.json"
            if result_path.exists():
                existing = _read_json(result_path)
                if (existing.get("status") == "pass"
                        and existing.get("input_ablation") == ablation
                        and existing.get("checkpoint_sha256") == _sha256(checkpoint)
                        and existing.get("dataset_manifest_sha256") == validation_manifests[split]["sha256"]
                        and existing.get("evaluated_episodes") == expected_total
                        and existing.get("evaluation_errors") == 0
                        and existing.get("missing_expected_ids") == 0
                        and existing.get("unexpected_ids") == 0):
                    continue

            evaluation_key = (ablation, split)
            shard_records_by_evaluation[evaluation_key] = []
            for shard_index in range(SHARD_COUNT):
                episode_start = shard_index * shard_size
                if episode_start >= expected_total:
                    continue
                episode_count = min(shard_size, expected_total - episode_start)
                episode_range = [episode_start, episode_start + episode_count]
                shard_output_dir = output_root / "shards" / ablation / split / f"shard_{shard_index:02d}"
                shard_path = shard_output_dir / f"{split}.json"
                shard_record = (shard_index, shard_path, episode_count, episode_range)
                shard_records_by_evaluation[evaluation_key].append(shard_record)
                if shard_path.exists():
                    prior = _read_json(shard_path)
                    if (prior.get("status") == "pass" and prior.get("partial_validation_shard")
                            and prior.get("input_ablation") == ablation
                            and prior.get("checkpoint_sha256") == _sha256(checkpoint)
                            and prior.get("dataset_manifest_sha256") == validation_manifests[split]["sha256"]
                            and prior.get("expected_episodes") == episode_count
                            and prior.get("evaluation_episode_range") == episode_range
                            and prior.get("evaluation_errors") == 0):
                        continue

                command = [
                    sys.executable, "-m", "experiment.avnav.evaluate_repaired",
                    "--run-dir", str(run_dir), "--split", split,
                    "--checkpoint", str(checkpoint), "--training-config", str(training_config),
                    "--output-dir", str(shard_output_dir), "--input-ablation", ablation,
                    "--episode-start", str(episode_start), "--episode-count", str(episode_count),
                ]
                log_path = log_dir / f"{ablation}_{split}_shard_{shard_index:02d}.log"
                log_handle = log_path.open("a", encoding="utf-8")
                environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                                   MKL_NUM_THREADS="2", MAGNUM_LOG="quiet",
                                   HABITAT_SIM_LOG="quiet", MPLCONFIGDIR="/tmp/avnav-modality-ablation-mpl")
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                           env=environment, stdout=log_handle, stderr=subprocess.STDOUT)
                pending.append({"ablation": ablation, "split": split, "shard_index": shard_index,
                                "command": command, "log_path": log_path,
                                "log_handle": log_handle, "process": process})

    failures = []
    for job in pending:
        return_code = job["process"].wait()
        job["log_handle"].close()
        if return_code:
            failures.append({"command": job["command"], "log": str(job["log_path"]),
                             "return_code": return_code})
            continue
        result_path = (output_root / "shards" / job["ablation"] / job["split"]
                       / f"shard_{job['shard_index']:02d}" / f"{job['split']}.json")
        result = _read_json(result_path)
        if (result.get("status") != "pass" or result.get("partial_validation_smoke")
                or not result.get("partial_validation_shard")
                or result.get("evaluation_errors") != 0 or result.get("input_ablation") != job["ablation"]):
            failures.append({"command": job["command"], "log": str(job["log_path"]),
                             "return_code": "incomplete result"})
    if failures:
        ablation_protocol["status"] = "failed"
        ablation_protocol["failures"] = failures
        _atomic_json(output_root / "protocol.json", ablation_protocol)
        raise RuntimeError(f"One or more validation ablations failed: {failures}")

    merged_paths = {}
    for (ablation, split), shard_records in shard_records_by_evaluation.items():
        expected_id_path = run_dir / "audit/split_episode_ids" / f"{split}.jsonl"
        audited_ids = {
            (str(row["scene"]), str(row["episode_id"]))
            for row in (json.loads(line) for line in expected_id_path.read_text(encoding="utf-8").splitlines() if line.strip())
        }
        expected_ids = _manifest_episode_order(validation_manifests[split]["path"])
        if len(expected_ids) != len(audited_ids) or set(expected_ids) != audited_ids:
            raise RuntimeError(f"Validation manifest order does not match audited IDs: {split}")
        merged_paths[(ablation, split)] = _merge_split_shards(
            output_root, run_dir, ablation, split, shard_records, expected_ids)

    results = {}
    for ablation in ABLATIONS:
        results[ablation] = {}
        for split in SPLITS:
            path = merged_paths.get((ablation, split), output_root / ablation / f"{split}.json")
            record = _read_json(path)
            if record.get("status") != "pass" or record.get("evaluation_errors") != 0:
                raise RuntimeError(f"Ablation result is not complete: {path}")
            results[ablation][split] = {
                "path": str(path), "sha256": _sha256(path),
                "expected_episodes": record["expected_episodes"],
                "evaluated_episodes": record["evaluated_episodes"],
                "evaluation_errors": record["evaluation_errors"],
                "SR": record["SR"], "SPL": record["SPL"],
            }

    summary = {
        "status": "completed", "protocol": str(output_root / "protocol.json"),
        "baseline": baseline, "ablations": results,
        "test_data_used": False,
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _atomic_json(output_root / "summary.json", summary)
    ablation_protocol["status"] = "completed"
    ablation_protocol["completed_at"] = summary["completed_at"]
    _atomic_json(output_root / "protocol.json", ablation_protocol)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-config", required=True)
    run(parser.parse_args())
