"""Extract, train, and validate independent AVNav single-modality policies."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


MODALITIES = ("vision_only", "audio_only")
SPLITS = ("val_multiple", "val_multiple_unheard")
SHARD_COUNT = 2


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_parallel(jobs, cwd, environment):
    running = []
    for job in jobs:
        job["log"].parent.mkdir(parents=True, exist_ok=True)
        handle = job["log"].open("a", encoding="utf-8")
        process = subprocess.Popen(job["command"], cwd=cwd, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        running.append((job, handle, process))
    failures = []
    for job, handle, process in running:
        code = process.wait()
        handle.close()
        if code:
            failures.append({"command": job["command"], "log": str(job["log"]), "return_code": code})
    if failures:
        raise RuntimeError(f"Parallel AVNav jobs failed: {failures}")


def _baseline_record(run_dir, split):
    path = run_dir / "evaluation/candidates/A_repaired_baseline/epoch_010" / f"{split}.json"
    record = _json(path)
    if (record.get("status") != "pass" or record.get("evaluated_episodes") != 500
            or record.get("evaluation_errors") != 0 or record.get("missing_expected_ids") != 0
            or record.get("unexpected_ids") != 0):
        raise RuntimeError(f"Paired multimodal baseline is incomplete: {path}")
    return {
        "path": str(path), "sha256": _sha256(path),
        "checkpoint": record["checkpoint"], "checkpoint_sha256": record["checkpoint_sha256"],
        "expected_episodes": record["expected_episodes"], "SR": record["SR"], "SPL": record["SPL"],
    }


def _merge_policy_shards(output_root, modality, split, shard_records, expected_ids,
                         checkpoint, training_config):
    seen, episodes, errors, artifacts = set(), [], [], []
    base = None
    for shard_index, result_path, count, episode_range in shard_records:
        record = _json(result_path)
        if (record.get("status") != "pass" or not record.get("partial_validation_shard")
                or record.get("policy_modality") != modality
                or record.get("input_ablation") != "none"
                or record.get("checkpoint_sha256") != _sha256(checkpoint)
                or record.get("expected_episodes") != count
                or record.get("evaluated_episodes") != count
                or record.get("evaluation_episode_range") != episode_range
                or record.get("evaluation_errors") != 0):
            raise RuntimeError(f"Incomplete or mismatched trained-policy shard: {result_path}")
        base = record
        for row in record["episode_metrics"]:
            key = (str(row["scene"]), str(row["episode_id"]))
            if key in seen:
                raise RuntimeError(f"Duplicate validation episode across shards: {key}")
            seen.add(key)
            episodes.append(row)
        errors.extend(record.get("errors", []))
        artifacts.append({
            "shard": shard_index, "path": str(result_path), "sha256": _sha256(result_path),
            "episode_range": episode_range, "evaluated_episodes": len(record["episode_metrics"]),
        })
    missing, unexpected = sorted(set(expected_ids) - seen), sorted(seen - set(expected_ids))
    if len(episodes) != len(set(expected_ids)) or missing or unexpected:
        raise RuntimeError(
            f"{modality}/{split} validation ID coverage failed: "
            f"episodes={len(episodes)}/{len(set(expected_ids))}, missing={len(missing)}, unexpected={len(unexpected)}"
        )
    if base is None:
        raise RuntimeError(f"No validation shards for {modality}/{split}")

    episodes.sort(key=lambda row: (row["scene"], str(row["episode_id"])))
    successes = sum(int(row["success"]) for row in episodes)
    collision_count = sum(int(row["collisions"]) for row in episodes)
    action_counts = [sum(int(row["action_counts"][i]) for row in episodes) for i in range(4)]
    decisions = sum(action_counts)
    forward_actions = action_counts[1]
    if decisions != sum(int(row["steps"]) for row in episodes):
        raise RuntimeError(f"Action count and step count disagree for {modality}/{split}")
    result = {
        **base,
        "status": "pass" if not errors else "fail",
        "policy_modality": modality,
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "training_config_path": str(training_config),
        "expected_episodes": len(set(expected_ids)), "full_expected_episodes": len(set(expected_ids)),
        "evaluated_episodes": len(episodes), "evaluation_errors": len(errors),
        "missing_expected_ids": len(missing), "unexpected_ids": len(unexpected),
        "missing_id_examples": missing[:30], "unexpected_id_examples": unexpected[:30],
        "SR": successes / max(len(set(expected_ids)), 1),
        "SPL": sum(float(row["spl"]) for row in episodes) / max(len(set(expected_ids)), 1),
        "mean_steps": sum(int(row["steps"]) for row in episodes) / max(len(episodes), 1),
        "collisions": collision_count,
        "mean_collisions_per_episode": collision_count / max(len(episodes), 1),
        "collision_rate_per_action": collision_count / max(decisions, 1),
        "forward_collision_rate": collision_count / max(forward_actions, 1),
        "STOP_actions": action_counts[0],
        "STOP_rate": action_counts[0] / max(decisions, 1),
        "action_counts": dict(zip(("STOP", "FORWARD", "LEFT", "RIGHT"), action_counts)),
        "timeout_rate": sum(int(bool(row["timeout"])) for row in episodes) / max(len(set(expected_ids)), 1),
        "episode_metrics": episodes, "errors": errors,
        "partial_validation_smoke": False, "partial_validation_shard": False,
        "evaluation_episode_range": [0, len(set(expected_ids))], "shard_aggregation": artifacts,
    }
    output_dir = output_root / modality
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_source = Path(base["environment_path"])
    environment_destination = output_dir / f"{split}_environment.yaml"
    environment_destination.write_bytes(environment_source.read_bytes())
    result["environment_sha256"] = _sha256(environment_destination)
    result_path = output_dir / f"{split}.json"
    _atomic_json(result_path, result)
    with (output_dir / f"{split}_episode_ids.jsonl").open("w", encoding="utf-8") as handle:
        for row in episodes:
            handle.write(json.dumps({"scene": row["scene"], "episode_id": row["episode_id"]}) + "\n")
    return result_path


def _write_report(path, summary):
    lines = [
        "# Retrained AVNav single-modality baselines",
        "",
        "Separate vision-only and audio-only policies were trained. Each policy calls only its own frozen encoder branch; the other modality is not zero-filled into the fusion network.",
        "",
        "| Split | Model | Episodes | SR | SPL | Forward collision rate | Mean collisions/episode | Mean steps | STOP rate | Errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        for modality in MODALITIES:
            row = summary["policies"][modality][split]
            lines.append(
                f"| {split} | {modality} | {row['evaluated_episodes']} | {row['SR']:.4f} | "
                f"{row['SPL']:.4f} | {row['forward_collision_rate']:.4f} | "
                f"{row['mean_collisions_per_episode']:.2f} | {row['mean_steps']:.2f} | "
                f"{row['STOP_rate']:.4f} | {row['evaluation_errors']} |"
            )
    lines.extend([
        "",
        "Forward collision rate is collision count divided by forward actions. STOP rate is STOP actions divided by all policy decisions.",
        "",
        "## Training and evaluation protocol",
        "",
        "- The two SAC-CQL-LSTM policies were initialized and trained independently for 30 epochs, 300 updates per epoch, with the same train episodes, seed, and hyperparameters.",
        "- Only audited greedy/deverse/collision train trajectories were used. Encoders remained frozen. The visual branch uses RGB-D features; the audio branch uses binaural audio features. Each branch outputs 128 dimensions.",
        "- Epoch 30 was fixed before validation. Validation did not contribute gradients or tune settings. Each policy was evaluated on all 500 heard and 500 unheard validation episodes.",
        "- The environment received raw actor argmax actions with no override or post-processing. No test observations, trajectories, or metrics were loaded, and no test split was run.",
        "",
        "Source manifests, checkpoint/config/code hashes, and split ID hashes are in protocol.json. Per-episode results are in the vision_only/ and audio_only/ directories.",
        "",
        "The earlier zero-input diagnostic remains in ../modality_ablation/ for reference; it is not a retrained single-modality baseline.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    run_dir = Path(args.run_dir).resolve()
    output_root = run_dir / "evaluation/trained_single_modality"
    log_dir = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists() and _json(protocol_path).get("status") == "completed":
        print((output_root / "summary.json").read_text(encoding="utf-8"))
        return

    audit_path = run_dir / "audit/data_source_audit.json"
    audit = _json(audit_path)
    if audit.get("status") != "pass":
        raise RuntimeError("Single-modality runs require a passing train/validation source audit")
    for split in SPLITS:
        if (audit["selected_parent_episode_overlap"][split]["count"] != 0
                or audit["selected_parent_scene_overlap"][split]["count"] != 0):
            raise RuntimeError(f"Training episode/scene overlap with {split}")
    for category, source in audit["categories"].items():
        if _sha256(source["manifest"]) != source["manifest_sha256"]:
            raise RuntimeError(f"Training manifest hash changed: {category}")

    foundation_path = run_dir / "foundation/action_stage/best.pth"
    training = {
        "seed": 20261027, "epochs": 30, "updates_per_epoch": 300, "batch_size": 256,
        "sequence_length": 5, "hidden_dim": 128, "lstm_hidden_dim": 256,
        "actor_lr": 2e-5, "critic_lr": 5e-5, "alpha_lr": 1e-5,
        "target_entropy": 1.08, "tau": 0.005, "gamma": 0.75,
        "cql_weight": 0.5, "bc_weight": 0.5,
        "collision_bc_weight": 1.0, "collision_extra_penalty": 0.5,
        "checkpoint_rule": "fixed final epoch 30 before validation evaluation",
    }
    validation_sources = {}
    for split in SPLITS:
        declared = audit["validation_and_test_split_manifests"][split]
        path = Path(declared["path"])
        if _sha256(path) != declared["sha256"]:
            raise RuntimeError(f"Validation manifest changed since audit: {split}")
        id_path = run_dir / "audit/split_episode_ids" / f"{split}.jsonl"
        validation_sources[split] = {
            "manifest": str(path), "manifest_sha256": declared["sha256"],
            "episode_count": declared["episode_count"],
            "episode_id_manifest": str(id_path), "episode_id_manifest_sha256": _sha256(id_path),
        }
        _baseline_record(run_dir, split)

    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("extract_single_modality_features.py").resolve(),
        Path(__file__).with_name("train_single_modality_policy.py").resolve(),
        Path(__file__).with_name("extract_codex_features.py").resolve(),
        Path(__file__).with_name("train_codex_saccql_lstm.py").resolve(),
        Path(__file__).with_name("evaluate_repaired.py").resolve(),
        Path(__file__).with_name("evaluate_codex_saccql_lstm.py").resolve(),
        Path(__file__).with_name("hybrid/foundation_model.py").resolve(),
    ]
    protocol = {
        "status": "running",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "objective": "Train separate visual-branch-only and audio-branch-only AVNav policies",
        "modalities": {
            "vision_only": "RGB-D visual encoder and visual transformer; 64 spatial tokens mean-pooled to 128D",
            "audio_only": "binaural AudioCRNN encoder token only, 128D",
        },
        "zero_filled_inputs": False,
        "foundation_checkpoint": str(foundation_path), "foundation_sha256": _sha256(foundation_path),
        "foundation_frozen": True,
        "training": training,
        "data_source_audit": str(audit_path), "data_source_audit_sha256": _sha256(audit_path),
        "train_manifests": audit["categories"],
        "train_validation_episode_scene_overlap": "zero, verified from passing audit before extraction",
        "validation_sources": validation_sources,
        "multimodal_reference": {split: _baseline_record(run_dir, split) for split in SPLITS},
        "evaluation": {
            "seed": 20261027, "max_steps": 200,
            "actions": "raw deterministic actor argmax sent directly to the benchmark environment",
            "splits": list(SPLITS), "episodes_per_split": {split: 500 for split in SPLITS},
            "parallel_workers": len(MODALITIES) * len(SPLITS) * SHARD_COUNT,
            "disjoint_shards_per_modality_split": SHARD_COUNT,
        },
        "validation_data_used_for_gradient_updates": False,
        "test_data_loaded_or_evaluated": False,
        "code_sha256": {str(path): _sha256(path) for path in code_paths},
    }
    _atomic_json(protocol_path, protocol)

    environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet",
                       MPLCONFIGDIR="/tmp/avnav-single-modality-mpl")
    cwd = Path(__file__).resolve().parents[2]

    extraction_jobs = []
    for modality in MODALITIES:
        extraction_jobs.append({
            "command": [
                sys.executable, "-m", "experiment.avnav.extract_single_modality_features",
                "--run-dir", str(run_dir), "--modality", modality,
                "--device", args.device, "--batch-size", str(args.feature_batch_size),
                "--cpu-threads", "2", "--seed", str(training["seed"]),
            ],
            "log": log_dir / f"features_{modality}.log",
        })
    _run_parallel(extraction_jobs, cwd, environment)

    train_jobs = []
    for modality in MODALITIES:
        train_jobs.append({
            "command": [
                sys.executable, "-m", "experiment.avnav.train_single_modality_policy",
                "--run-dir", str(run_dir), "--modality", modality,
                "--device", args.device, "--cpu-threads", "2",
                "--seed", str(training["seed"]), "--epochs", str(training["epochs"]),
                "--updates-per-epoch", str(training["updates_per_epoch"]),
                "--batch-size", str(training["batch_size"]),
                "--sequence-length", str(training["sequence_length"]),
                "--hidden-dim", str(training["hidden_dim"]),
                "--lstm-hidden-dim", str(training["lstm_hidden_dim"]),
                "--actor-lr", str(training["actor_lr"]), "--critic-lr", str(training["critic_lr"]),
                "--alpha-lr", str(training["alpha_lr"]), "--target-entropy", str(training["target_entropy"]),
                "--tau", str(training["tau"]), "--gamma", str(training["gamma"]),
                "--cql-weight", str(training["cql_weight"]), "--bc-weight", str(training["bc_weight"]),
                "--collision-bc-weight", str(training["collision_bc_weight"]),
                "--collision-extra-penalty", str(training["collision_extra_penalty"]),
            ],
            "log": log_dir / f"training_{modality}.log",
        })
    _run_parallel(train_jobs, cwd, environment)

    pending, shard_records = [], {}
    for modality in MODALITIES:
        training_dir = run_dir / "offline/single_modality" / modality / "training"
        checkpoint = training_dir / "epoch_030.pt"
        config_path = training_dir / "training_config.json"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for split in SPLITS:
            manifest = validation_sources[split]
            shard_size = (manifest["episode_count"] + SHARD_COUNT - 1) // SHARD_COUNT
            records = []
            shard_records[(modality, split)] = records
            for shard_index in range(SHARD_COUNT):
                start = shard_index * shard_size
                count = min(shard_size, manifest["episode_count"] - start)
                episode_range = [start, start + count]
                shard_dir = output_root / "shards" / modality / split / f"shard_{shard_index:02d}"
                result_path = shard_dir / f"{split}.json"
                records.append((shard_index, result_path, count, episode_range))
                command = [
                    sys.executable, "-m", "experiment.avnav.evaluate_repaired",
                    "--run-dir", str(run_dir), "--split", split,
                    "--checkpoint", str(checkpoint), "--training-config", str(config_path),
                    "--output-dir", str(shard_dir), "--policy-modality", modality,
                    "--episode-start", str(start), "--episode-count", str(count),
                    "--device", args.device,
                ]
                pending.append({
                    "command": command, "log": log_dir / f"{modality}_{split}_shard_{shard_index:02d}.log",
                })
    _run_parallel(pending, cwd, environment)

    policies = {}
    for modality in MODALITIES:
        policies[modality] = {}
        for split in SPLITS:
            id_path = Path(validation_sources[split]["episode_id_manifest"])
            expected = [
                (str(row["scene"]), str(row["episode_id"]))
                for row in (json.loads(line) for line in id_path.read_text(encoding="utf-8").splitlines() if line.strip())
            ]
            training_dir = run_dir / "offline/single_modality" / modality / "training"
            output_path = _merge_policy_shards(
                output_root, modality, split, shard_records[(modality, split)],
                expected, training_dir / "epoch_030.pt", training_dir / "training_config.json",
            )
            row = _json(output_path)
            policies[modality][split] = {
                "path": str(output_path), "sha256": _sha256(output_path),
                "expected_episodes": row["expected_episodes"],
                "evaluated_episodes": row["evaluated_episodes"],
                "evaluation_errors": row["evaluation_errors"],
                "missing_expected_ids": row["missing_expected_ids"],
                "unexpected_ids": row["unexpected_ids"],
                "SR": row["SR"], "SPL": row["SPL"],
                "collisions": row["collisions"],
                "mean_collisions_per_episode": row["mean_collisions_per_episode"],
                "collision_rate_per_action": row["collision_rate_per_action"],
                "forward_collision_rate": row["forward_collision_rate"],
                "mean_steps": row["mean_steps"], "STOP_actions": row["STOP_actions"],
                "STOP_rate": row["STOP_rate"], "timeout_rate": row["timeout_rate"],
                "action_counts": row["action_counts"],
            }

    summary = {
        "status": "completed", "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "multimodal_reference": protocol["multimodal_reference"],
        "policies": policies,
        "validation_data_used_for_training": False,
        "test_data_loaded_or_evaluated": False,
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _atomic_json(output_root / "summary.json", summary)
    protocol["status"] = "completed"
    protocol["completed_at"] = summary["completed_at"]
    _atomic_json(protocol_path, protocol)
    summary["protocol_sha256"] = _sha256(protocol_path)
    _atomic_json(output_root / "summary.json", summary)
    _write_report(output_root / "REPORT.md", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    run(parser.parse_args())
