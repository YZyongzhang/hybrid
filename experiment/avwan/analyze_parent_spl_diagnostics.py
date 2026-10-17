"""Summarize train-only parent rollouts and expert-assisted correction yield."""
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"result/2026-10-15-avwan-spl-improvement"
DIAG=OUT/"diagnostics"
CORR=OUT/"corrections"
BINS=("under_16m","16_to_24m","24_to_32m","32m_plus")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def diagnostic_stats(items):
    n=len(items)
    ts=[t for r in items for t in r.get("trace",[])]
    false_stops=sum(t["policy_action"]==40 and not (r["success"] and t["done"])
                    for r in items for t in r.get("trace",[]))
    correct_stops=sum(t["policy_action"]==40 and r["success"] and t["done"]
                      for r in items for t in r.get("trace",[]))
    nonpositive=sum(t["distance_progress"]<=0 and not t["reaching_waypoint"] and
                    not (r["success"] and t["policy_action"]==40 and t["done"])
                    for r in items for t in r.get("trace",[]))
    return {"episodes":n,"actual_initial_distance_mean_m":sum(r["initial_distance"] for r in items)/max(n,1),
        "successes":sum(r["success"] for r in items),"success_rate":sum(r["success"] for r in items)/max(n,1),
        "mean_spl":sum(r["spl"] for r in items)/max(n,1),
        "mean_primitive_steps":sum(r["primitive_steps"] for r in items)/max(n,1),
        "false_stop_transitions":false_stops,"episodes_with_false_stop":sum(r["false_stop_actions"]>0 for r in items),
        "correct_stop_transitions":correct_stops,"waypoint_transitions":len(ts),
        "policy_expert_disagreements":sum(r["policy_expert_disagreements"] for r in items),
        "disagreement_rate":sum(r["policy_expert_disagreements"] for r in items)/max(len(ts),1),
        "primitive_collision_events":sum(r["primitive_collision_events"] for r in items),
        "waypoint_actions_with_collision":sum(r["waypoint_actions_with_collision"] for r in items),
        "unreachable_waypoints":sum(r["unreachable_waypoints"] for r in items),
        "nonpositive_progress_actions":nonpositive,
        "mean_repeated_endpoint_fraction":sum(r["repeated_endpoint_fraction"] for r in items)/max(n,1)}


def correction_stats(items):
    n=len(items)
    return {"attempted_episodes":n,"parent_successes":sum(r["diagnostic_parent_success"] for r in items),
        "assisted_successes":sum(r["success"] for r in items),
        "mean_parent_spl":sum(r["diagnostic_parent_spl"] for r in items)/max(n,1),
        "mean_assisted_spl":sum(r["spl"] for r in items)/max(n,1),
        "mean_spl_gain":sum(r["spl"]-r["diagnostic_parent_spl"] for r in items)/max(n,1),
        "mean_parent_primitive_steps":sum(r["diagnostic_parent_primitive_steps"] for r in items)/max(n,1),
        "mean_assisted_primitive_steps":sum(r["primitive_steps"] for r in items)/max(n,1),
        "expert_interventions":sum(r["expert_interventions"] for r in items),
        "useful_executed_expert_actions":sum(r["useful_interventions"] for r in items)}


def main():
    diag=rows(DIAG/"episodes.jsonl")
    corr=rows(CORR/"episodes.jsonl") if (CORR/"episodes.jsonl").is_file() else []
    train_rows=rows(CORR/"training_source"/"episodes.jsonl") if (CORR/"training_source"/"episodes.jsonl").is_file() else []
    corr_manifest=json.loads((CORR/"manifest.json").read_text()) if (CORR/"manifest.json").is_file() else {}
    result={"training_only":True,"heldout_used_for_training":False,
        "parent_checkpoint_sha256":json.loads((DIAG/"selection.json").read_text())["policy_sha256"],
        "selection_sha256":json.loads((DIAG/"manifest.json").read_text())["selection_sha256"],
        "parent_pure_rollout":{"overall":diagnostic_stats(diag),
            "by_distance_bin":{b:diagnostic_stats([r for r in diag if r["distance_bin"]==b]) for b in BINS}},
        "expert_assisted_replays":{"overall":correction_stats(corr),
            "by_distance_bin":{b:correction_stats([r for r in corr if r["distance_bin"]==b]) for b in BINS},
            "productivity_rule":"assisted success and assisted SPL >= pure-parent SPL + 0.02",
            "productive_training_episodes":len(train_rows),
            "executed_expert_action_reasons":corr_manifest.get("executed_expert_action_reasons",{}),
            "productive_expert_action_reasons":corr_manifest.get("productive_expert_action_reasons",{}),
            "bc_eligible_executed_expert_transitions_productive_pool":corr_manifest.get(
                "bc_eligible_executed_expert_transitions_productive_pool",0),
            "training_transitions":sum(r["waypoint_transitions"] for r in corr
                                        if (r["scene"],str(r["episode_id"])) in
                                        {(x["scene"],str(x["episode_id"])) for x in train_rows})}}
    (OUT/"DIAGNOSTICS_AND_CORRECTIONS.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# 纯父策略诊断与有效纠错产出","",
        "所有 rollout 和纠错重跑均来自 scene-disjoint 的 train_multiple；纯诊断中专家只给动作对照标签，没有执行专家动作。","",
        "## 父策略训练集诊断","",
        "| 实测起点距离 | 集数 | 成功率 | SPL | 原语步数 | 误报 STOP | 策略/专家分歧率 | 碰撞 | 不可达 waypoint | 非正向推进 | 端点重复率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b in BINS:
        r=result["parent_pure_rollout"]["by_distance_bin"][b]
        lines.append(f"| {b} | {r['episodes']} | {r['success_rate']:.1%} | {r['mean_spl']:.4f} | {r['mean_primitive_steps']:.1f} | {r['false_stop_transitions']} | {r['disagreement_rate']:.1%} | {r['primitive_collision_events']} | {r['unreachable_waypoints']} | {r['nonpositive_progress_actions']} | {r['mean_repeated_endpoint_fraction']:.1%} |")
    lines.extend(["","## 失败轨迹专家辅助重跑","",
        "纳入条件：父策略失败，或起点 ≥24 m 且父策略 SPL <0.20。对策略/专家分歧执行专家动作，其他状态执行父策略动作。训练池只收录 reset 实测距离 ≥24 m、辅助成功且 SPL 至少提高 0.02 的 episode；BC mask 只开放实际执行、有效推进且无碰撞的专家动作。","",
        "| 距离档 | 重跑集数 | 父策略成功 | 辅助成功 | 父 SPL | 辅助 SPL | SPL 增量 | 专家接管 | 有效纠错动作 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for b in BINS:
        r=result["expert_assisted_replays"]["by_distance_bin"][b]
        lines.append(f"| {b} | {r['attempted_episodes']} | {r['parent_successes']} | {r['assisted_successes']} | {r['mean_parent_spl']:.4f} | {r['mean_assisted_spl']:.4f} | {r['mean_spl_gain']:+.4f} | {r['expert_interventions']} | {r['useful_executed_expert_actions']} |")
    lines.extend(["",f"可用于微调的生产性纠错 episode：**{len(train_rows)}**。","",
        f"生产性训练池中符合 BC 条件的实际执行专家转移：**{result['expert_assisted_replays']['bc_eligible_executed_expert_transitions_productive_pool']}**。",
        f"按接管原因统计：`{json.dumps(result['expert_assisted_replays']['productive_expert_action_reasons'],ensure_ascii=False)}`。",
        "实际 reset 初始距离与训练索引距离、源 episode overlap、哈希与动作/reward/next-state 保真审计见 diagnostics 和 corrections 下的 manifest。",""])
    (OUT/"DIAGNOSTICS_AND_CORRECTIONS.md").write_text("\n".join(lines))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
