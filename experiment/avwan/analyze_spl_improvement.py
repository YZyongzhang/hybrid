"""Paired, validation-only comparison for parent-anchored correction runs."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/"result/2026-10-15-avwan-hard-corrections/offline/parent_best_validation/validation"
SPLITS=("val_multiple","val_multiple_unheard")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare(candidate_root, label, seed=20261016):
    rng=np.random.default_rng(seed)
    results=[]
    for split in SPLITS:
        parent_dir=PARENT/split
        candidate_dir=Path(candidate_root)/split
        parent_run=read_json(parent_dir/"run.json")
        candidate_run=read_json(candidate_dir/"run.json")
        old_rows=read_rows(parent_dir/"episodes.jsonl")
        new_rows=read_rows(candidate_dir/"episodes.jsonl")
        old={(r["scene"],str(r["episode_id"])):r for r in old_rows}
        new={(r["scene"],str(r["episode_id"])):r for r in new_rows}
        if parent_run["selection_sha256"]!=candidate_run["selection_sha256"] or set(old)!=set(new):
            raise ValueError(f"Parent/candidate validation episodes are not paired for {split}")
        if len(old)!=100 or any(r.get("error") for r in old_rows+new_rows):
            raise ValueError(f"Expected 100 valid paired episodes for {split}")
        keys=sorted(old)
        old_spl=np.asarray([old[k]["spl"] for k in keys],dtype=np.float64)
        new_spl=np.asarray([new[k]["spl"] for k in keys],dtype=np.float64)
        old_success=np.asarray([bool(old[k]["success"]) for k in keys])
        new_success=np.asarray([bool(new[k]["success"]) for k in keys])
        delta=new_spl-old_spl
        sample=rng.integers(0,len(delta),size=(20000,len(delta)))
        boot=delta[sample].mean(1)
        long=np.asarray([float(old[k].get("initial_distance",0.))>=24. for k in keys])
        gains=int((new_success & ~old_success).sum());losses=int((old_success & ~new_success).sum())
        results.append({"split":split,"episodes":len(keys),
            "selection_sha256":parent_run["selection_sha256"],
            "baseline_checkpoint_sha256":parent_run["checkpoint_sha256"],
            "candidate_checkpoint_sha256":candidate_run["checkpoint_sha256"],
            "baseline_spl":float(old_spl.mean()),"candidate_spl":float(new_spl.mean()),
            "delta_spl":float(delta.mean()),
            "paired_bootstrap_95pct_ci_delta_spl":[float(x) for x in np.quantile(boot,[.025,.975])],
            "baseline_success_rate":float(old_success.mean()),"candidate_success_rate":float(new_success.mean()),
            "delta_success_rate":float(new_success.mean()-old_success.mean()),
            "success_gain_episodes":gains,"success_loss_episodes":losses,
            "success_mcnemar_exact_p":_mcnemar(gains,losses),
            "ge_24m_episodes":int(long.sum()),
            "ge_24m_baseline_spl":float(old_spl[long].mean()) if long.any() else None,
            "ge_24m_candidate_spl":float(new_spl[long].mean()) if long.any() else None,
            "ge_24m_delta_spl":float(delta[long].mean()) if long.any() else None})
    passed=all(r["paired_bootstrap_95pct_ci_delta_spl"][0]>0 and
               (r["ge_24m_delta_spl"] is None or r["ge_24m_delta_spl"]>=-0.02)
               for r in results)
    return {"label":label,"baseline":"Oct 14 best_policy checkpoint",
        "validation_only":True,"heldout_test_used_for_selection":False,
        "bootstrap":{"method":"paired episode bootstrap","replicates":20000,"seed":seed,"interval":"percentile 95% CI"},
        "promotion_gate":"both split 95% paired-bootstrap lower bounds > 0 and >=24m SPL delta >= -0.02",
        "promotion_gate_passed":passed,"splits":results}


def _mcnemar(gains,losses):
    discordant=gains+losses
    if discordant==0:return 1.
    tail=sum(math.comb(discordant,k) for k in range(min(gains,losses)+1))
    return min(1.,2.*tail/(2**discordant))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate_validation_dir",type=Path)
    p.add_argument("--label",required=True)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=compare(args.candidate_validation_dir,args.label)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        lines=[f"# {args.label} 验证对照","",
            "候选与 10 月 14 日父策略在固定 heard/unheard 各 100 集上逐集配对；区间由 20,000 次 paired bootstrap 计算。","",
            "| Split | n | 父 SPL | 候选 SPL | Δ SPL（95% CI） | 父成功率 | 候选成功率 | ≥24 m n | ≥24 m SPL 父→候选 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in result["splits"]:
            lo,hi=r["paired_bootstrap_95pct_ci_delta_spl"]
            long_text=(f"{r['ge_24m_baseline_spl']:.4f}→{r['ge_24m_candidate_spl']:.4f}"
                       if r["ge_24m_episodes"] else "—")
            lines.append(f"| {r['split']} | {r['episodes']} | {r['baseline_spl']:.4f} | {r['candidate_spl']:.4f} | {r['delta_spl']:+.4f} ([{lo:+.4f}, {hi:+.4f}]) | {r['baseline_success_rate']:.1%} | {r['candidate_success_rate']:.1%} | {r['ge_24m_episodes']} | {long_text} |")
        lines.extend(["",f"预先设定的提升门槛：**{'通过' if result['promotion_gate_passed'] else '未通过'}**。", ""])
        args.output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
