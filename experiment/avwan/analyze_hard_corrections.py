"""Audit train-only long-distance DAgger data and summarize correction yield."""
import gzip
import json
from collections import Counter
from pathlib import Path
import pickle

from .offline.features import read_episode


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "result/2026-10-15-avwan-hard-corrections/collection"
OUT = ROOT / "result/2026-10-15-avwan-hard-corrections"
BINS = [(24, 32, "24–<32 m"), (32, float("inf"), "≥32 m")]
STOP = 40


def summarize(rows, transition_rows):
    n = len(rows)
    t = len(transition_rows)
    disagreements = sum(row["policy_action"] != row["expert_action"] for row in transition_rows)
    interventions = sum(bool(row["expert_control"]) for row in transition_rows)
    false_stop = sum(row["policy_action"] == STOP and row["expert_action"] != STOP
                     for row in transition_rows)
    collisions = sum(int(row.get("collision_count", 0)) for row in transition_rows)
    collision_actions = sum(int(row.get("collision_count", 0)) > 0 for row in transition_rows)
    unreachable = sum(bool(row.get("cant_reach_waypoint", False)) for row in transition_rows)
    nonpositive_progress = sum(float(row.get("distance_progress", 0.0)) <= 0 and
                               not bool(row.get("reaching_waypoint", False))
                               for row in transition_rows)
    useful = sum(bool(row["expert_control"]) and
                 (bool(row.get("correction_progress", False)) or
                  (bool(row.get("episode_success", False)) and int(row["action"]) == STOP and
                   bool(row.get("done", False)))) and
                 int(row.get("collision_count", 0)) == 0 for row in transition_rows)
    reasons = Counter(row["intervention_reason"] for row in transition_rows
                      if bool(row["expert_control"]))
    primitive_steps = sum(int(row["num_steps"]) for row in transition_rows)
    return {
        "episodes": n,
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / max(n, 1),
        "timeouts": sum(row["status"] == "timeout" for row in rows),
        "blocked": sum(row["status"] == "blocked" for row in rows),
        "waypoint_transitions": t,
        "primitive_steps": primitive_steps,
        "mean_primitive_steps_per_episode": primitive_steps / max(n, 1),
        "policy_expert_disagreements": disagreements,
        "disagreement_rate": disagreements / max(t, 1),
        "expert_interventions_executed": interventions,
        "expert_intervention_rate": interventions / max(t, 1),
        "predicted_false_stops_corrected": false_stop,
        "primitive_collision_events": collisions,
        "waypoint_actions_with_collision": collision_actions,
        "unreachable_waypoints": unreachable,
        "nonpositive_goal_distance_progress": nonpositive_progress,
        "useful_correction_proxy": useful,
        "useful_correction_proxy_rate": useful / max(interventions, 1),
        "intervention_reasons": dict(reasons),
    }


def candidate_population(selection):
    source = Path(selection["source"])
    if not source.is_file():
        raise FileNotFoundError(source)
    from hashlib import sha256
    digest = sha256(source.read_bytes()).hexdigest()
    if digest != selection["source_sha256"]:
        raise ValueError("Training selection source index changed after episode selection")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    audit_path = ROOT / "result/10-11/total/2026-10-03-unheard-generalization-study/split_audit.json"
    audit = json.loads(audit_path.read_text())
    heldout_scenes = set(audit["test_multiple"]["scenes"]) | set(audit["test_multiple_unheard"]["scenes"])
    validation_scenes = set(audit["val_multiple"]["scenes"]) | set(audit["val_multiple_unheard"]["scenes"])
    eligible = [row for row in rows if row.get("saved") and row["initial_distance"] >= 24
                and row["scene"] not in heldout_scenes | validation_scenes]
    return {
        "train_source_episodes": len(rows),
        "eligible_ge_24m": len(eligible),
        "eligible_24_to_32m": sum(24 <= row["initial_distance"] < 32 for row in eligible),
        "eligible_ge_32m": sum(row["initial_distance"] >= 32 for row in eligible),
        "eligible_scenes": len({row["scene"] for row in eligible}),
        "eligible_training_sounds": len({row["sound"] for row in eligible}),
        "heldout_or_validation_scenes_removed": True,
    }


def main():
    manifest_path = DATA / "manifest.json"
    index_path = DATA / "episodes.jsonl"
    if not manifest_path.exists() or not index_path.exists():
        raise FileNotFoundError("Collection manifest and episode index must be complete first")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed" or not manifest.get("training_only"):
        raise ValueError("Refusing to analyze an incomplete or non-training collection")
    records = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    if len(records) != manifest["episodes"]:
        raise ValueError("Collection index count differs from its manifest")
    selection = json.loads((DATA / "selection.json").read_text())
    selected_keys = {(row["scene"], str(row["episode_id"])) for row in selection["episodes"]}
    record_keys = {(row["scene"], str(row["episode_id"])) for row in records}
    if len(record_keys) != len(records) or record_keys != selected_keys:
        raise ValueError("Collected episode keys do not exactly match the train-only selection")

    all_transitions = []
    per_episode = []
    for row in records:
        path = DATA / "episodes" / row["scene"] / (str(row["episode_id"]) + ".pkl.gz")
        with gzip.open(path, "rb") as f:
            episode = pickle.load(f)
        transitions = episode["transitions"]
        read_episode(path, 81)  # Validates masks, SMDP discounts, boundaries, and state continuity.
        for t in transitions:
            if t["action"] != (t["expert_action"] if t["expert_control"] else t["policy_action"]):
                raise ValueError(f"Executed action provenance mismatch in {path}")
            if len(t["primitive_collisions"]) != t["num_steps"]:
                raise ValueError(f"Collision trace length mismatch in {path}")
            t["episode_success"] = bool(row["success"])
        per_episode.append((row, transitions))
        all_transitions.extend(transitions)

    overall = summarize(records, all_transitions)
    by_distance = []
    for low, high, label in BINS:
        selected = [(row, transitions) for row, transitions in per_episode
                    if low <= row["initial_distance"] < high]
        rows = [row for row, _ in selected]
        transitions = [t for _, group in selected for t in group]
        data = summarize(rows, transitions)
        data["distance_bin"] = label
        by_distance.append(data)
    result = {
        "training_only": True,
        "heldout_data_used": False,
        "selection_sha256": manifest["selection_sha256"],
        "policy_sha256": manifest["policy_sha256"],
        "episodes": len(records),
        "candidate_population": candidate_population(selection),
        "overall": overall,
        "by_distance": by_distance,
        "definition_of_useful_correction_proxy": (
            "An executed expert action followed by a reached waypoint, decreased goal distance, or "
            "successful terminal STOP, with no collision in that macro action. This is a local progress "
            "marker, not a causal comparison."
        ),
        "quality_audit": {
            "episode_keys_match_selection": True,
            "all_transitions_pass_smdp_and_state_continuity_validation": True,
            "executed_actions_match_behavior_provenance": True,
            "collision_trace_lengths_match_primitive_steps": True,
        },
    }
    (OUT / "correction_statistics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    population = result["candidate_population"]
    lines = ["# 长距离策略纠错数据统计", "",
             f"训练集合中有 {population['eligible_ge_24m']} 条 ≥24 m 候选轨迹 "
             f"（24–<32 m: {population['eligible_24_to_32m']}，≥32 m: {population['eligible_ge_32m']}），"
             f"覆盖 {population['eligible_scenes']} 个训练场景。按距离段均衡选取 250 集；"
             "验证集与测试场景从候选集中排除，held-out 数据没有用于采集或训练。", "",
             "| 距离段 | 集数 | 成功率 | waypoint 转移 | 平均 primitive 步数 | 策略/专家分歧 | 执行专家纠错 | 纠正策略误报 STOP | 碰撞事件 | 不可达 waypoint | 非正向距离变化 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for data in by_distance:
        lines.append(f"| {data['distance_bin']} | {data['episodes']} | {data['success_rate']:.1%} | "
                     f"{data['waypoint_transitions']} | {data['mean_primitive_steps_per_episode']:.1f} | "
                     f"{data['policy_expert_disagreements']} "
                     f"({data['disagreement_rate']:.1%}) | {data['expert_interventions_executed']} "
                     f"({data['expert_intervention_rate']:.1%}) | {data['predicted_false_stops_corrected']} | "
                     f"{data['primitive_collision_events']} | {data['unreachable_waypoints']} | "
                     f"{data['nonpositive_goal_distance_progress']} |")
    lines.extend(["", f"合计 {overall['episodes']} 集、{overall['waypoint_transitions']} 个 waypoint 转移；"
                  f"策略与专家分歧 {overall['policy_expert_disagreements']} 次 "
                  f"（{overall['disagreement_rate']:.1%}），执行专家动作 {overall['expert_interventions_executed']} 次。",
                  f"专家接管类型：{json.dumps(overall['intervention_reasons'], ensure_ascii=False)}。",
                  f"策略误报 STOP 并由专家继续前进 {overall['predicted_false_stops_corrected']} 次；"
                  f"采集记录 {overall['primitive_collision_events']} 次 primitive 碰撞，"
                  f"{overall['unreachable_waypoints']} 次 waypoint 不可达。",
                  f"按局部定义，{overall['useful_correction_proxy']} 次专家干预满足‘推进目标距离、到达 waypoint "
                  "或成功 STOP，且无碰撞’"
                  f"（占干预的 {overall['useful_correction_proxy_rate']:.1%}）。这个指标是纠错是否产生局部推进的代理量，"
                  "不表示同一状态下与策略动作的因果差值。", "",
                  "所有原始 transition 保留实际执行动作、reward、next state、terminated/truncated 和 discount；"
                  "每个 episode 通过动作掩码、SMDP discount、连续状态与碰撞轨迹长度审计。", ""])
    (OUT / "CORRECTION_STATISTICS.md").write_text("\n".join(lines))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
