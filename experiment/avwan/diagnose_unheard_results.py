"""Describe completed rollouts; never changes selection or policy parameters."""
import csv
import json
from statistics import mean

from .unheard_study import OUT
from .run_gamma_study import write


def summarize(rows):
    successful = [r for r in rows if r['success']]
    return dict(
        episodes=len(rows),
        spl=mean(r['spl'] for r in rows),
        successes=len(successful),
        successful_mean_spl=mean(r['spl'] for r in successful) if successful else None,
        failed_timeout=sum(not r['success'] and bool(r.get('truncated')) for r in rows),
        failed_stop=sum(not r['success'] and bool(r['actions']) and r['actions'][-1] == 40 for r in rows),
        early_failed_stop=sum(not r['success'] and len(r['actions']) <= 2 and bool(r['actions'])
                              and r['actions'][-1] == 40 for r in rows),
        mean_initial_distance=mean(r['initial_distance'] for r in rows),
        mean_final_distance=mean(r['final_distance'] for r in rows),
        mean_distance_reduction=mean(r['initial_distance']-r['final_distance'] for r in rows),
        mean_primitive_steps=mean(r['primitive_steps'] for r in rows),
    )


def diagnose():
    summaries, strata = [], []
    bounds = [(0, 5), (5, 10), (10, 20), (20, float('inf'))]
    for folder in sorted(OUT.glob('exp*')):
        for split in ['val_multiple_unheard', 'test_multiple', 'test_multiple_unheard']:
            path = folder/split/'episodes.jsonl'
            if not path.exists():
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != 100 or any(r.get('error') for r in rows):
                continue
            summaries.append(dict(experiment=folder.name, split=split, **summarize(rows)))
            for lower, upper in bounds:
                subset = [r for r in rows if lower <= r['initial_distance'] < upper]
                if subset:
                    strata.append(dict(experiment=folder.name, split=split,
                                       distance_bin=f'[{lower}, {upper})', **summarize(subset)))
    for name, records in [('failure_breakdown', summaries), ('distance_strata', strata)]:
        if records:
            write(OUT/(name+'.json'), records)
            with (OUT/(name+'.csv')).open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    lines = ['# 失败诊断', '',
             '只描述已完成的固定100条评估；本分析不用于修改模型、选择测试样本或调节参数。', '',
             '| 配置 | Unheard SPL | 成功数 | 成功episode平均SPL | 失败超时 | 失败STOP | 前两次决策错误STOP |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summaries:
        if row['split'] != 'test_multiple_unheard':
            continue
        efficiency = row['successful_mean_spl']
        text_efficiency = '—' if efficiency is None else f'{efficiency:.4f}'
        lines.append(f"| {row['experiment']} | {row['spl']:.4f} | {row['successes']} | {text_efficiency} | "
                     f"{row['failed_timeout']} | {row['failed_stop']} | {row['early_failed_stop']} |")
    lines += ['', '平均SPL = 成功率 × 成功episode平均SPL。后者反映成功轨迹的路径效率；只改善路径效率无法突破成功率所限定的上界。',
              '失败超时和失败STOP是可重叠的观测标记，不能相加当作失败总数。距离分层见distance_strata.csv；所有距离单位与环境一致。',
              '这里没有保存完整位姿轨迹，不能仅凭重复动作把超时断言为原地循环。']
    (OUT/'FAILURE_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    equivalence = OUT/'exp012-ratio-corrective-recurrent-bc/same_observation_equivalence.json'
    if equivalence.exists():
        check = json.loads(equivalence.read_text())
        comparisons = []
        for split in ['val_multiple_unheard', 'test_multiple', 'test_multiple_unheard']:
            paths = [OUT/name/split/'episodes.jsonl' for name in
                     ['exp011-foundation-binaural-ratio-adapter', 'exp012-ratio-corrective-recurrent-bc']]
            if not all(p.exists() for p in paths):
                continue
            tables = [{(r['scene'], r['episode_id']): r for r in
                       (json.loads(line) for line in p.read_text().splitlines())} for p in paths]
            a, b = tables
            if len(a) != 100 or a.keys() != b.keys():
                continue
            comparisons.append(dict(split=split, episodes=100,
                first_action_disagreements=sum(a[k]['actions'][0] != b[k]['actions'][0] for k in a),
                full_action_sequence_disagreements=sum(a[k]['actions'] != b[k]['actions'] for k in a),
                success_disagreements=sum(a[k]['success'] != b[k]['success'] for k in a),
                offline_minus_foundation_spl=mean(b[k]['spl']-a[k]['spl'] for k in a)))
        write(OUT/'rollout_equivalence_comparison.json', comparisons)
        notes = ['# 仿真一致性限制', '',
                 f"exp012选中epoch0。同一批验证观测上的两套完整部署策略对照：{check['observations']}个状态、"
                 f"{check['action_mismatches']}个动作差异；先验权重完全相同，循环输出残差为零。",
                 '但是独立导航运行中的首动作和成功结果有差异，不能把这些差异解释为LSTM收益，也不能仅当作微小舍入误差。', '',
                 '| 分组 | 首动作不同/100 | 全部动作序列不同/100 | 成功结果不同/100 | offline减foundation SPL |',
                 '| --- | ---: | ---: | ---: | ---: |']
        for r in comparisons:
            notes.append(f"| {r['split']} | {r['first_action_disagreements']} | "
                         f"{r['full_action_sequence_disagreements']} | {r['success_disagreements']} | "
                         f"{r['offline_minus_foundation_spl']:.4f} |")
        notes += ['', '两组使用相同episode清单、逐episode种子、步数上限和决策间隔，但分片数分别为2和4，仿真进程与场景访问历史不同。',
                  '源码检查：当前使用实时声学渲染，temporalCoherence=True；场景变化时重建底层Simulator。评估在env.reset前调用seed，而场景重建发生在reset内部。'
                  '声学缓存也只在场景或声音变化时清理。这些是需要隔离验证的运行状态因素，目前没有证明哪一项是根因。环境源码快照见environment_audit。',
                  '本轮未修改这些环境行为，保留所有原始结果。固定清单和种子尚不足以保证相同观测；bootstrap区间不包含这类运行间变化。'
                  '因此最高测试SPL属于单次探索性观察，不能据此声称稳定提升。下一步应先统一仿真重置/声学渲染条件，并重复基线和候选模型的运行，再验证收益。']
        replay = OUT/'probes/simulator_initial_observation_replay/status.json'
        if replay.exists():
            result = json.loads(replay.read_text())
            notes += ['', '## 初始观测复测', '',
                f"系统抽取固定验证清单中10条episode，以相同进程级Python/NumPy/PyTorch种子和逐episode种子、相反访问顺序分别重置。"
                f"10对观测中RGB一致{result['identical_rgb']}对，深度一致{result['identical_depth']}对，"
                f"声谱图一致{result['identical_spectrograms']}对。这确认存在声学观测的运行间变化，尚未隔离其底层原因。",
                '这20次初始重置不执行完整导航，不计入正式验证/测试rollout总数。数组、哈希、顺序、种子和恢复记录保存在probes/simulator_initial_observation_replay。'
                '首次尝试在同一进程重建第二个环境时遇到GL上下文错误，已归档；随后补齐进程级种子重新执行20次重置，结论相同。'
                '有效对照使用独立进程，未改动正式实验环境。此前仅固定逐episode种子的诊断保存在simulator_initial_observation_replay_before_global_seed_control，'
                '失败尝试保存在simulator_initial_observation_replay_failed_attempt1。']
        (OUT/'REPRODUCIBILITY.md').write_text('\n'.join(notes)+'\n')


if __name__ == '__main__':
    diagnose()
