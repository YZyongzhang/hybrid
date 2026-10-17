"""Summarize saved AV-WaN held-out rollouts without using them for training."""
import json
from collections import Counter
import gzip
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "result/2026-10-14-avwan-diverse-retraining"
REPORT_DIR = ROOT / "result/2026-10-15-avwan-hard-corrections"
TRAIN_INDEX = Path("/data/Getuanhui/avwan_multiple_greedy_train/episodes.jsonl")
BINS = [(0, 8, "<8 m"), (8, 16, "8–<16 m"), (16, 24, "16–<24 m"), (24, float("inf"), "≥24 m")]
STOP_ACTION = 40


def load_rollouts(split):
    rows = []
    for path in sorted((RUN / "full_test" / split).glob("shard_*/episodes.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if len(rows) != 1000 or len({(r["scene"], str(r["episode_id"])) for r in rows}) != 1000:
        raise ValueError(f"Expected 1,000 unique saved episodes for {split}, got {len(rows)}")
    if any(row.get("error") for row in rows):
        raise ValueError(f"Rollout errors present in {split}")
    return rows


def summarize(rows):
    failure = [row for row in rows if not row["success"]]
    timeouts = [row for row in failure if row.get("truncated", False)]
    # Saved rollout rows do not store the environment termination reason.
    # The center action (40) is STOP, so this is a directly observed marker,
    # while collision and stuck-loop causes need primitive-level collection.
    failed_stop = [row for row in failure if row.get("actions") and row["actions"][-1] == STOP_ACTION]
    return {
        "episodes": len(rows),
        "spl": sum(row["spl"] for row in rows) / max(len(rows), 1),
        "success_rate": sum(bool(row["success"]) for row in rows) / max(len(rows), 1),
        "mean_primitive_steps": sum(row["primitive_steps"] for row in rows) / max(len(rows), 1),
        "failures": len(failure),
        "timeouts": len(timeouts),
        "failed_stop_last_action": len(failed_stop),
        "other_failures": len(failure) - len(timeouts) - len(failed_stop),
        "mean_final_distance_on_failures": (
            sum(row.get("final_distance", 0) for row in failure) / len(failure) if failure else 0
        ),
    }


def by_distance(rows):
    result = []
    for low, high, label in BINS:
        group = [row for row in rows if low <= row.get("initial_distance", -1) < high]
        if not group:
            continue
        summary = summarize(group)
        summary["distance_bin"] = label
        result.append(summary)
    return result


def previous_longrange_stats():
    selection_path = RUN / "collection" / "selection.json"
    collected_path = RUN / "collection" / "episodes.jsonl"
    if not selection_path.exists() or not collected_path.exists() or not TRAIN_INDEX.exists():
        return {"available": False}
    keys = {(row["scene"], str(row["episode_id"])) for row in json.loads(selection_path.read_text())}
    train_index = [json.loads(line) for line in TRAIN_INDEX.read_text().splitlines() if line.strip()]
    train_keys = {(row["scene"], str(row["episode_id"])) for row in train_index}
    if not keys <= train_keys:
        raise ValueError("Previously selected episodes are not all present in the train source")
    selected = []
    for line in collected_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row["scene"], str(row["episode_id"])) in keys:
            selected.append(row)
    if len(selected) != len(keys):
        raise ValueError("Previously selected long-range episodes do not match their collection index")
    return {
        "available": True,
        "episodes": len(selected),
        "episodes_ge_8m": sum(row["initial_distance"] >= 8 for row in selected),
        "episodes_ge_16m": sum(row["initial_distance"] >= 16 for row in selected),
        "episodes_ge_24m": sum(row["initial_distance"] >= 24 for row in selected),
        "distance_bins": {label: sum(low <= row["initial_distance"] < high for row in selected)
                          for low, high, label in BINS},
        "collection_outcomes": dict(Counter(row["status"] for row in selected)),
        "selection_file": str(selection_path.relative_to(ROOT)),
    }


def previous_dagger_stats():
    folder = ROOT / "result/10-11/total/2026-10-03-unheard-generalization-study/training_corrections"
    selection_path = folder / "selection.json"
    if not selection_path.exists():
        return {"available": False}
    selected = json.loads(selection_path.read_text())["episodes"]
    metadata = []
    for row in selected:
        path = folder / "episodes" / row["scene"] / (str(row["episode_id"]) + ".pkl.gz")
        if not path.exists():
            raise FileNotFoundError(path)
        with gzip.open(path, "rb") as f:
            metadata.append(pickle.load(f)["metadata"])
    transitions = sum(row["transitions"] for row in metadata)
    disagreements = sum(row.get("teacher_disagreements", 0) for row in metadata)
    return {
        "available": True,
        "episodes": len(metadata),
        "episodes_ge_24m": sum(row["initial_distance"] >= 24 for row in selected),
        "distance_bins": {label: sum(low <= row["initial_distance"] < high for row in selected)
                          for low, high, label in BINS},
        "successful_episodes": sum(bool(row.get("success")) for row in metadata),
        "transitions": transitions,
        "policy_expert_disagreements": disagreements,
        "disagreement_rate": disagreements / max(transitions, 1),
        "collection_policy_sha256": metadata[0].get("policy_sha256") if metadata else None,
    }


def main():
    splits = {}
    for split in ("test_multiple", "test_multiple_unheard"):
        rows = load_rollouts(split)
        splits[split] = {"overall": summarize(rows), "by_initial_distance": by_distance(rows)}
    result = {
        "held_out_rollout_source": "result/2026-10-14-avwan-diverse-retraining/full_test",
        "training_data_used_from_held_out": False,
        "stop_action": STOP_ACTION,
        "failure_reason_limits": (
            "The saved rows identify time limits and whether the final chosen waypoint was STOP. "
            "They do not contain primitive collision traces or poses, so they cannot establish "
            "whether a timeout came from collision, oscillation, or poor progress."
        ),
        "splits": splits,
        "previous_targeted_collection": previous_longrange_stats(),
        "previous_dagger_collection": previous_dagger_stats(),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "failure_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# AV-WaN 距离与失败类型统计", "",
             "统计对象为 2026-10-14 保存的 held-out 全量 rollout，每个 split 1,000 集。这里只做诊断，未将 held-out 数据用于训练、采样或参数选择。", ""]
    for split, title in (("test_multiple", "Heard"), ("test_multiple_unheard", "Unheard")):
        data = splits[split]
        lines.extend([f"## {title}", "",
                      "| 起点距离 | 集数 | SPL | 成功率 | 超时 | 失败且最后动作为 STOP | 其他失败 | 平均步数 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for row in data["by_initial_distance"]:
            lines.append(f"| {row['distance_bin']} | {row['episodes']} | {row['spl']:.4f} | "
                         f"{row['success_rate']:.1%} | {row['timeouts']} | "
                         f"{row['failed_stop_last_action']} | {row['other_failures']} | "
                         f"{row['mean_primitive_steps']:.1f} |")
        row = data["overall"]
        lines.extend([f"", f"总体：SPL **{row['spl']:.4f}**，成功率 **{row['success_rate']:.1%}**；"
                      f"{row['failures']} 集失败，其中 {row['timeouts']} 集超时，"
                      f"{row['failed_stop_last_action']} 集以 STOP 结束。", ""])
    prev = result["previous_targeted_collection"]
    if prev.get("available"):
        lines.extend(["## 上一批长路径采集的覆盖", "",
                      f"250 集中，起点 ≥8 m 有 {prev['episodes_ge_8m']} 集，≥16 m 有 "
                      f"{prev['episodes_ge_16m']} 集，≥24 m 有 **{prev['episodes_ge_24m']} 集**。"
                      f"采集结果为 {prev['collection_outcomes'].get('success', 0)} 成功、"
                      f"{prev['collection_outcomes'].get('timeout', 0)} 超时、"
                      f"{prev['collection_outcomes'].get('failure', 0)} 失败。"
                      "它的总体成功率很高，但没有覆盖测试中最弱的 ≥24 m 区间。", ""])
    dagger = result["previous_dagger_collection"]
    if dagger.get("available"):
        lines.extend(["## 先前策略纠错轨迹覆盖", "",
                      f"先前 DAgger 采集有 {dagger['episodes']} 集，其中起点 ≥24 m 只有 "
                      f"{dagger['episodes_ge_24m']} 集；共 {dagger['transitions']} 个 waypoint 转移，"
                      f"策略与专家动作分歧 {dagger['policy_expert_disagreements']} 次 "
                      f"（{dagger['disagreement_rate']:.1%}）。这批轨迹有有效纠错信号，但长路径样本仍少。", ""])
    lines.extend(["## 解释范围", "",
                  "保存的逐集日志能区分超时和以 STOP 结束的失败；它没有逐 primitive 的碰撞与位置序列，"
                  "所以不能把超时直接归因于碰撞或原地循环。新一轮训练集采集会保存每个 waypoint 的碰撞标记、"
                  "距离变化、专家动作、策略动作和两者分歧，用于把这些原因拆开统计。", ""])
    (REPORT_DIR / "FAILURE_DIAGNOSTICS.md").write_text("\n".join(lines))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
