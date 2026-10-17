"""Paired held-in validation comparison for the hard-correction SAC-CQL run."""
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "result/2026-10-15-avwan-hard-corrections/offline/parent_best_validation/validation"
CANDIDATE = ROOT / "result/2026-10-15-avwan-hard-corrections/offline/latest_candidate_validation/validation"
OUT = ROOT / "result/2026-10-15-avwan-hard-corrections"
SPLITS = {
    "val_multiple": ("val_multiple", "val_multiple.json"),
    "val_multiple_unheard": ("val_multiple_unheard", "result.json"),
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def binomial_two_sided(gains, losses):
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def compare_split(split, rng):
    baseline_folder, baseline_result_name = SPLITS[split]
    candidate_folder, candidate_result_name = SPLITS[split]
    baseline_dir = BASELINE / baseline_folder
    candidate_dir = CANDIDATE / candidate_folder
    baseline_run = json.loads((baseline_dir / "run.json").read_text())
    candidate_run = json.loads((candidate_dir / "run.json").read_text())
    baseline_result = json.loads((BASELINE / baseline_result_name).read_text())
    candidate_result = json.loads((CANDIDATE / candidate_result_name).read_text())
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
    sampled = rng.integers(0, len(delta), size=(20000, len(delta)))
    boot_delta = delta[sampled].mean(axis=1)
    gains = int((new_success & ~old_success).sum())
    losses = int((old_success & ~new_success).sum())

    long_mask = np.asarray([float(baseline[key].get("initial_distance", 0.0)) >= 24.0
                            for key in keys])
    long_count = int(long_mask.sum())
    if long_count:
        long_old_spl = old_spl[long_mask]
        long_new_spl = new_spl[long_mask]
        long_old_success = old_success[long_mask]
        long_new_success = new_success[long_mask]
    else:
        long_old_spl = long_new_spl = np.asarray([], dtype=np.float64)
        long_old_success = long_new_success = np.asarray([], dtype=bool)

    return {
        "split": split,
        "episodes": len(keys),
        "selection_sha256": baseline_result["selection_sha256"],
        "baseline_checkpoint_sha256": baseline_result["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate_result["checkpoint_sha256"],
        "baseline_spl": float(old_spl.mean()),
        "candidate_spl": float(new_spl.mean()),
        "delta_spl": float(delta.mean()),
        "paired_bootstrap_95pct_ci_delta_spl": [float(x) for x in np.quantile(boot_delta, [0.025, 0.975])],
        "baseline_success_rate": float(old_success.mean()),
        "candidate_success_rate": float(new_success.mean()),
        "delta_success_rate": float(new_success.mean() - old_success.mean()),
        "success_gain_episodes": gains,
        "success_loss_episodes": losses,
        "success_mcnemar_exact_p": binomial_two_sided(gains, losses),
        "ge_24m_episodes": long_count,
        "ge_24m_baseline_spl": float(long_old_spl.mean()) if long_count else None,
        "ge_24m_candidate_spl": float(long_new_spl.mean()) if long_count else None,
        "ge_24m_delta_spl": float((long_new_spl - long_old_spl).mean()) if long_count else None,
        "ge_24m_baseline_success_rate": float(long_old_success.mean()) if long_count else None,
        "ge_24m_candidate_success_rate": float(long_new_success.mean()) if long_count else None,
    }


def main():
    rng = np.random.default_rng(20261015)
    comparisons = [compare_split(split, rng) for split in SPLITS]
    result = {
        "baseline": "Exact Oct 14 best_policy initialization checkpoint on matched validation population",
        "candidate": "Oct 15 SAC-CQL latest checkpoint after 20 epochs with hard corrections",
        "validation_only": True,
        "heldout_test_used_for_selection": False,
        "bootstrap": {"method": "paired episode bootstrap", "replicates": 20000,
                      "seed": 20261015, "interval": "percentile 95% CI"},
        "success_test": "exact two-sided McNemar test",
        "splits": comparisons,
    }
    (OUT / "validation_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# 困难纠错微调闭环验证", "",
             "比较 10 月 15 日 SAC-CQL 的 epoch 20 latest checkpoint 与 10 月 14 日父策略。两边使用同一批 100 集验证 episode；" 
             "表中 Δ 为候选减父模型，SPL 的区间用逐集配对 bootstrap 计算。", "",
             "| Split | n | 父模型 SPL | 微调 SPL | Δ SPL（95% CI） | 父模型成功率 | 微调成功率 | ≥24 m n | ≥24 m 父→微调 SPL |", 
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        low, high = row["paired_bootstrap_95pct_ci_delta_spl"]
        lines.append(
            f"| {row['split']} | {row['episodes']} | {row['baseline_spl']:.4f} | "
            f"{row['candidate_spl']:.4f} | {row['delta_spl']:+.4f} "
            f"([{low:+.4f}, {high:+.4f}]) | {row['baseline_success_rate']:.1%} | "
            f"{row['candidate_success_rate']:.1%} | {row['ge_24m_episodes']} | "
            f"{row['ge_24m_baseline_spl']:.4f}→{row['ge_24m_candidate_spl']:.4f} |"
        )
    lines.extend(["", "已配对成功结果变化：",
                  *[f"- {row['split']}：成功新增 {row['success_gain_episodes']} 集，成功退化 "
                    f"{row['success_loss_episodes']} 集，McNemar 双侧精确检验 p="
                    f"{row['success_mcnemar_exact_p']:.4f}。" for row in comparisons],
                  "", "epoch 0 的 checkpoint 通过离线验证选择，但 actor 参数与父策略逐项相同；训练后的 latest 权重已独立做本表验证。",
                  "两个 split 的 SPL 均未提升，因此不启动全量 held-out 测试。", ""])
    (OUT / "VALIDATION_COMPARISON.md").write_text("\n".join(lines))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
