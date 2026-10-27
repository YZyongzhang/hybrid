"""Select safety-aware CQL-LSTM ablations using validation only."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

from experiment.avnav.config import resolve_path
from experiment.avnav.evaluation_protocol import episode_manifest_hashes

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reports(output):
    report_path = Path(output) / "final" / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Validation report was not produced: {report_path}")
    data = json.loads(report_path.read_text(encoding="utf-8"))
    splits = {row["split"]: row for row in data["splits"]}
    required = ("val_multiple", "val_multiple_unheard")
    if any(name not in splits for name in required):
        raise ValueError(f"Validation report misses required splits: {report_path}")
    return splits


def _score(splits, coefficient):
    rows = [splits[name] for name in ("val_multiple", "val_multiple_unheard")]
    mean_spl = sum(float(row["spl"]) for row in rows) / 2
    mean_sr = sum(float(row["success_rate"]) for row in rows) / 2
    mean_collision_rate = sum(float(row.get("collisions_per_action",
                                                  row.get("collisions_per_step", 0.0)))
                              for row in rows) / 2
    errors = sum(int(row["errors"]) for row in rows)
    return {
        "mean_spl": mean_spl,
        "mean_success_rate": mean_sr,
        "mean_collisions_per_action": mean_collision_rate,
        "safety_spl_score": mean_spl - coefficient * mean_collision_rate,
        "errors": errors,
    }


def _run_candidate(config, checkpoint, name, output, episodes, workers,
                   coefficient, phase, variant):
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    config = Path(config).expanduser().resolve()
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    manifest_hashes = episode_manifest_hashes(
        config_payload["EVALUATION"]["CONFIG"],
        ("val_multiple", "val_multiple_unheard"),
    )
    context = {
        "phase": phase, "candidate": name,
        "variant": variant,
        "config": str(config), "config_sha256": _sha256(config),
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "splits": ["val_multiple", "val_multiple_unheard"],
        "canonical_episode_manifests": manifest_hashes,
        "episodes_per_split": int(episodes), "workers": int(workers),
        "selection_score": "mean_SPL - 0.1 * mean_collision_rate_per_action",
        "collision_rate_coefficient": float(coefficient),
        "action_handling": "raw policy actor argmax; no action override or recovery",
        "test_data_used": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    context_path = output / "validation_context.json"
    if context_path.exists():
        if json.loads(context_path.read_text(encoding="utf-8")) != context:
            raise ValueError(f"Validation output belongs to a different run: {output}")
    elif any(path.name != "validation_context.json" for path in output.iterdir()):
        raise FileExistsError(f"Refusing to reuse validation output: {output}")
    else:
        context_path.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")

    report = output / "final" / "report.json"
    if not report.is_file():
        command = [sys.executable, "-m", "experiment.avnav.run_full_test_parallel",
                   "--config", str(config), "--checkpoint", str(checkpoint),
                   "--output", str(output), "--splits", "val_multiple",
                   "val_multiple_unheard", "--episodes", str(episodes),
                   "--workers", str(workers), "--max-concurrent", str(workers)]
        with (output / "runner.log").open("a", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        if completed.returncode and not report.is_file():
            raise subprocess.CalledProcessError(completed.returncode, command)
    splits = _reports(output)
    score = _score(splits, coefficient)
    evaluated_manifests = {
        split: {
            "path": str((output / "final" / f"{split}_episodes.jsonl").resolve()),
            "sha256": _sha256(output / "final" / f"{split}_episodes.jsonl"),
        }
        for split in ("val_multiple", "val_multiple_unheard")
    }
    return {"candidate": name, "variant": variant, "checkpoint": str(checkpoint),
            "checkpoint_sha256": context["checkpoint_sha256"],
            "config": str(config), "output": str(output),
            "validation": splits,
            "evaluated_episode_manifests": evaluated_manifests, **score}


def select(candidate_manifest, output_root, screen_episodes=30,
           confirmation_episodes=500, top_k=3, workers=2,
           collision_rate_coefficient=0.1):
    candidate_manifest = Path(candidate_manifest).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    payload = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    candidates = payload["candidates"]
    if not candidates:
        raise ValueError("Candidate manifest is empty")
    output_root.mkdir(parents=True, exist_ok=True)
    config_payload = yaml.safe_load(
        Path(candidates[0]["config"]).expanduser().resolve().read_text(encoding="utf-8")
    )
    canonical_manifests = episode_manifest_hashes(
        config_payload["EVALUATION"]["CONFIG"],
        ("val_multiple", "val_multiple_unheard"),
    )
    protocol = {
        "status": "validation_protocol_frozen_before_evaluation",
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": _sha256(candidate_manifest),
        "validation_splits": ["val_multiple", "val_multiple_unheard"],
        "canonical_episode_manifests": canonical_manifests,
        "screen_episodes_per_split": int(screen_episodes),
        "confirmation_episodes_per_split": int(confirmation_episodes),
        "confirmation_top_k": int(top_k),
        "selection_score": "mean of (split SPL - 0.1 * split collisions_per_action)",
        "collision_rate_coefficient": float(collision_rate_coefficient),
        "ranking": "zero-error candidates first; maximize the frozen safety-SPL score; tie-break by mean SR then lower collision rate",
        "test_data_used": False,
        "action_handling": "raw policy actor argmax; no action override or recovery",
    }
    protocol_path = output_root / "selection_protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Validation selection protocol changed after it was frozen")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    # Screening ranks many checkpoints, but the confirmation set must still
    # cover every requested ablation. Keep the screening protocol immutable and
    # freeze this confirmation rule before starting any 500-episode runs.
    confirmation_protocol = {
        "status": "validation_confirmation_protocol_frozen_before_confirmation",
        "screening_protocol": str(protocol_path),
        "screening_protocol_sha256": _sha256(protocol_path),
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": _sha256(candidate_manifest),
        "validation_splits": ["val_multiple", "val_multiple_unheard"],
        "canonical_episode_manifests": canonical_manifests,
        "screen_episodes_per_split": int(screen_episodes),
        "confirmation_episodes_per_split": int(confirmation_episodes),
        "confirmation_top_k": int(top_k),
        "confirmation_candidate_policy": (
            "include the highest-ranked screened checkpoint from every variant; "
            "fill any remaining slots by global screening rank"
        ),
        "protocol_amendment_reason": (
            "the global screening top-k did not cover all requested ablations; "
            "one confirmed checkpoint per variant is required for the planned comparison"
        ),
        "selection_score": "mean of (split SPL - 0.1 * split collisions_per_action)",
        "collision_rate_coefficient": float(collision_rate_coefficient),
        "ranking": "zero-error candidates first; maximize the frozen safety-SPL score; tie-break by mean SR then lower collision rate",
        "test_data_used": False,
        "action_handling": "raw policy actor argmax; no action override or recovery",
    }
    confirmation_protocol_path = output_root / "confirmation_protocol.json"
    if confirmation_protocol_path.exists():
        if json.loads(confirmation_protocol_path.read_text(encoding="utf-8")) != confirmation_protocol:
            raise ValueError("Validation confirmation protocol changed after it was frozen")
    else:
        confirmation_protocol_path.write_text(
            json.dumps(confirmation_protocol, indent=2) + "\n", encoding="utf-8"
        )

    screened = []
    for row in candidates:
        name = str(row["name"])
        screened.append(_run_candidate(
            row["config"], row["checkpoint"], name,
            output_root / "screen30" / name, screen_episodes,
            workers, collision_rate_coefficient, "screen30",
            str(row.get("variant", name)),
        ))
        print(json.dumps({"phase": "screen30", **screened[-1]}, ensure_ascii=False), flush=True)

    def rank(row):
        return (row["errors"] == 0, row["safety_spl_score"],
                row["mean_success_rate"], -row["mean_collisions_per_action"])

    screened.sort(key=rank, reverse=True)
    best_by_variant = {}
    for row in screened:
        candidate = next(item for item in candidates if item["name"] == row["candidate"])
        variant = str(candidate.get("variant", candidate["name"]))
        current = best_by_variant.get(variant)
        if current is None or rank(row) > rank(current):
            best_by_variant[variant] = row
    if top_k < len(best_by_variant):
        raise ValueError(
            f"top_k={top_k} cannot confirm all {len(best_by_variant)} variants"
        )
    finalists = sorted(best_by_variant.values(), key=rank, reverse=True)
    if len(finalists) < top_k:
        finalist_names = {row["candidate"] for row in finalists}
        finalists.extend(
            row for row in sorted(screened, key=rank, reverse=True)
            if row["candidate"] not in finalist_names
        )
        finalists = finalists[:top_k]
    confirmed = []
    for row in finalists:
        name = row["candidate"]
        config = next(candidate["config"] for candidate in candidates
                      if candidate["name"] == name)
        confirmed.append(_run_candidate(
            config, row["checkpoint"], name,
            output_root / "confirm500" / name, confirmation_episodes,
            workers, collision_rate_coefficient, "confirm500",
            str(next(candidate.get("variant", name) for candidate in candidates
                     if candidate["name"] == name)),
        ))
        print(json.dumps({"phase": "confirm500", **confirmed[-1]}, ensure_ascii=False), flush=True)
    confirmed.sort(key=rank, reverse=True)
    selected = confirmed[0]
    result = {
        "status": "selected",
        "selection_protocol": str(confirmation_protocol_path),
        "candidate_manifest": str(candidate_manifest),
        "screening": screened,
        "confirmation": confirmed,
        "selected": selected,
        "test_data_used": False,
        "action_handling": "raw policy actor argmax; no action override or recovery",
    }
    (output_root / "validation_selection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--screen-episodes", type=int, default=30)
    parser.add_argument("--confirmation-episodes", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--collision-rate-coefficient", type=float, default=0.1)
    args = parser.parse_args()
    result = select(args.candidates, args.output, args.screen_episodes,
                    args.confirmation_episodes, args.top_k, args.workers,
                    args.collision_rate_coefficient)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
