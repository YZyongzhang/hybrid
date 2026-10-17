"""Paired validation comparison for the parent-anchored curriculum run."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


SPLITS = {
    "val_multiple": ("val_multiple", "val_multiple.json"),
    "val_multiple_unheard": ("val_multiple_unheard", "result.json"),
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def binomial_two_sided(gains, losses):
    discordant = gains + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def compare_split(split, baseline_root, candidate_root, rng):
    folder_name, result_name = SPLITS[split]
    baseline_dir = baseline_root / folder_name
    candidate_dir = candidate_root / folder_name
    baseline_run = json.loads((baseline_dir / "run.json").read_text())
    candidate_run = json.loads((candidate_dir / "run.json").read_text())
    baseline_result = json.loads((baseline_root / result_name).read_text())
    candidate_result = json.loads((candidate_root / result_name).read_text())
    baseline_rows = read_rows(baseline_dir / "episodes.jsonl")
    candidate_rows = read_rows(candidate_dir / "episodes.jsonl")
    if (baseline_result["episodes"] != 100 or candidate_result["episodes"] != 100 or
            baseline_run["selection_sha256"] != candidate_run["selection_sha256"] or
            baseline_result["selection_sha256"] != candidate_result["selection_sha256"]):
        raise ValueError(f"Validation split or episode selections differ for {split}")
    baseline = {(row["scene"], str(row["episode_id"])): row for row in baseline_rows}
    candidate = {(row["scene"], str(row["episode_id"])): row for row in candidate_rows}
    if set(baseline) != set(candidate) or len(baseline) != 100 or len(candidate) != 100:
        raise ValueError(f"Validation episodes do not pair exactly for {split}")
    if any(row.get("error") for row in baseline_rows + candidate_rows):
        raise ValueError(f"Evaluation errors found in {split}")

    keys = sorted(baseline)
    old_spl = np.asarray([baseline[key]["spl"] for key in keys], dtype=np.float64)
    new_spl = np.asarray([candidate[key]["spl"] for key in keys], dtype=np.float64)
    old_success = np.asarray([bool(baseline[key]["success"]) for key in keys])
    new_success = np.asarray([bool(candidate[key]["success"]) for key in keys])
    delta = new_spl - old_spl
    boot = delta[rng.integers(0, len(delta), size=(20000, len(delta)))].mean(axis=1)
    gains = int((new_success & ~old_success).sum())
    losses = int((old_success & ~new_success).sum())
    long_mask = np.asarray([float(baseline[key].get("initial_distance", 0.0)) >= 24.0
                            for key in keys])
    long_old, long_new = old_spl[long_mask], new_spl[long_mask]
    return {
        "split": split,
        "episodes": len(keys),
        "selection_sha256": baseline_result["selection_sha256"],
        "baseline_checkpoint_sha256": baseline_result["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate_result["checkpoint_sha256"],
        "baseline_spl": float(old_spl.mean()),
        "candidate_spl": float(new_spl.mean()),
        "delta_spl": float(delta.mean()),
        "paired_bootstrap_95pct_ci_delta_spl": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "baseline_success_rate": float(old_success.mean()),
        "candidate_success_rate": float(new_success.mean()),
        "delta_success_rate": float(new_success.mean() - old_success.mean()),
        "success_gain_episodes": gains,
        "success_loss_episodes": losses,
        "success_mcnemar_exact_p": binomial_two_sided(gains, losses),
        "ge_24m_episodes": int(long_mask.sum()),
        "ge_24m_baseline_spl": float(long_old.mean()) if len(long_old) else None,
        "ge_24m_candidate_spl": float(long_new.mean()) if len(long_new) else None,
        "ge_24m_delta_spl": float((long_new - long_old).mean()) if len(long_new) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Parent validation root containing split folders and result JSONs")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="Candidate validation root containing split folders and result JSONs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-label", default="curriculum checkpoint",
                        help="Label recorded in the comparison report")
    args = parser.parse_args()
    rng = np.random.default_rng(20261018)
    comparisons = [compare_split(split, args.baseline, args.candidate, rng) for split in SPLITS]
    report = {
        "baseline": "Oct 14 best_policy parent checkpoint",
        "candidate": args.candidate_label,
        "validation_only": True,
        "heldout_test_used_for_selection": False,
        "bootstrap": {"method": "paired episode bootstrap", "replicates": 20000,
                      "seed": 20261018, "interval": "percentile 95% CI"},
        "success_test": "exact two-sided McNemar test",
        "splits": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Parent-anchored curriculum validation", "",
             f"{args.candidate_label} is compared with its exact parent on the same 100 episodes in each held-in validation split. SPL intervals use paired episode bootstrap.", "",
             "| Split | n | Parent SPL | Curriculum SPL | Δ SPL (95% CI) | Parent success | Curriculum success | ≥24 m n | ≥24 m parent→curriculum SPL |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        low, high = row["paired_bootstrap_95pct_ci_delta_spl"]
        long_old = row["ge_24m_baseline_spl"]
        long_new = row["ge_24m_candidate_spl"]
        long_text = f"{long_old:.4f}→{long_new:.4f}" if long_old is not None else "n/a"
        lines.append(
            f"| {row['split']} | {row['episodes']} | {row['baseline_spl']:.4f} | "
            f"{row['candidate_spl']:.4f} | {row['delta_spl']:+.4f} "
            f"([{low:+.4f}, {high:+.4f}]) | {row['baseline_success_rate']:.1%} | "
            f"{row['candidate_success_rate']:.1%} | {row['ge_24m_episodes']} | {long_text} |"
        )
    lines.extend(["", "Paired success changes:",
                  *[f"- {row['split']}: {row['success_gain_episodes']} gains, "
                    f"{row['success_loss_episodes']} losses; exact McNemar p="
                    f"{row['success_mcnemar_exact_p']:.4f}." for row in comparisons],
                  "", "No held-out test population was used in training or validation.", ""])
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
