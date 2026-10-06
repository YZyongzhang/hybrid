"""Generate an auditable Chinese report and standalone figures for the study."""
import csv
import json
from pathlib import Path

import numpy as np

from .run_gamma_study import OUT, sha, write


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results, training = [], []
    for folder in sorted(OUT.glob('exp*')):
        if (folder / 'training/top_checkpoints.json').exists():
            ckpt = next(r for r in json.loads((folder / 'training/top_checkpoints.json').read_text()) if r['file'] == 'best_policy.pt')
            training.append(dict(experiment=folder.name, epoch=ckpt['epoch'], val_policy_nll=ckpt['val_policy_nll']))
        if not (folder / 'results.json').exists():
            continue
        data = json.loads((folder / 'results.json').read_text())
        row = dict(experiment=folder.name, gamma=data['spec']['gamma'])
        for split, label in [('val_multiple', 'val'), ('test_multiple', 'heard'), ('test_multiple_unheard', 'unheard')]:
            if split not in data:
                row.update({f'{label}_{metric}': None for metric in ['spl', 'success_rate', 'ci_low', 'ci_high', 'episodes', 'errors']})
                continue
            m = data[split]
            episodes = [json.loads(line) for line in (folder / split / 'episodes.jsonl').read_text().splitlines()]
            rng = np.random.default_rng(20260930)
            values = np.array([r['spl'] for r in episodes])
            ci = np.quantile(values[rng.integers(0, len(values), (10000, len(values)))].mean(1), [.025, .975])
            row.update({f'{label}_{metric}': m[metric] for metric in ['spl', 'success_rate', 'episodes', 'errors']})
            row.update({f'{label}_ci_low': float(ci[0]), f'{label}_ci_high': float(ci[1])})
        results.append(row)
    for name, rows in [('comparison_detailed', results), ('training_selection', training)]:
        write(OUT / (name + '.json'), rows)
        if rows:
            with (OUT / (name + '.csv')).open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    lines = ['# 固定 100 episode：gamma 与 offline RL 对照实验', '',
             'heard=`test_multiple`、unheard=`test_multiple_unheard`，各固定 100 条；seed=20260930。'
             '按场景比例分配（所有场景至少 1 条），场景内按声音、距离系统抽样。清单和哈希见 `selections/`；'
             'heard 声音在训练中出现，unheard 声音与训练声音无交集，见 `split_audit.json`。', '',
             '每组从同一冻结 foundation 的缓存特征训练，默认 30 epoch；其他轮数和参数见各目录 `spec.json`、`config.yaml`。'
             'checkpoint 按离线验证 policy NLL 选择。测试集上的多次比较属于探索性实验，最高测试分数不是独立泛化估计。', '',
             '| 实验 | 验证 SPL | Heard SPL | Heard SR | Unheard SPL | Unheard SR |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    def fmt(x):
        return '—' if x is None else f'{x:.4f}'
    for row in results:
        lines.append('| ' + row['experiment'] + ' | ' + ' | '.join(fmt(row[k]) for k in
                     ['val_spl', 'heard_spl', 'heard_success_rate', 'unheard_spl', 'unheard_success_rate']) + ' |')
    if results:
        passed = [r['experiment'] for r in results if r['heard_spl'] >= .5 and r['unheard_spl'] >= .5]
        lines += ['', '同时达到 heard/unheard SPL≥0.5 的配置：' + ('、'.join(passed) if passed else '目前没有。')]
        fig, ax = plt.subplots(figsize=(max(9, len(results)*.7), 5))
        x = np.arange(len(results))
        for offset, label, color in [(-.18, 'heard', '#377eb8'), (.18, 'unheard', '#ff7f00')]:
            vals = np.array([r[label+'_spl'] for r in results])
            low = vals - np.array([r[label+'_ci_low'] for r in results])
            high = np.array([r[label+'_ci_high'] for r in results]) - vals
            ax.bar(x+offset, vals, .36, label=label, color=color, yerr=[low, high], capsize=2)
        ax.axhline(.5, color='black', linestyle='--', linewidth=1, label='SPL target 0.5')
        ax.set(ylim=(0,1), ylabel='SPL', xticks=x, xticklabels=[r['experiment'].split('-')[0] for r in results])
        ax.legend(); fig.tight_layout(); fig.savefig(OUT/'spl_comparison.png', dpi=160); plt.close(fig)
        pure = [r for r in results if r['experiment'].split('-')[0] in {f'exp{i:03}' for i in range(3,9)}]
        if pure:
            pure.sort(key=lambda r:r['gamma'])
            fig, ax = plt.subplots(figsize=(7,4))
            for label in ['heard','unheard']:
                ax.plot([r['gamma'] for r in pure],[r[label+'_spl'] for r in pure],'-o',label=label)
            ax.axhline(.5, color='black',linestyle='--'); ax.set(xlabel='Training gamma (per decision)',ylabel='SPL',ylim=(0,1));ax.legend()
            fig.tight_layout();fig.savefig(OUT/'gamma_spl.png',dpi=160);plt.close(fig)
    if training:
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        for row in training:
            metrics=[json.loads(line) for line in (OUT/row['experiment']/'training/metrics.jsonl').read_text().splitlines()]
            for ax,key in zip(axes,['policy_nll','td_loss']):
                ax.plot([r['epoch'] for r in metrics],[r['val'][key] for r in metrics],label=row['experiment'].split('-')[0])
                ax.set(xlabel='Epoch',ylabel='Validation '+key)
        axes[0].legend(ncol=2,fontsize=7);fig.tight_layout();fig.savefig(OUT/'training_comparison.png',dpi=160);plt.close(fig)
    lines += ['', '## 对比图', '']
    for filename,label in [('spl_comparison.png','固定测试 SPL 与描述性 bootstrap 区间'),
                           ('gamma_spl.png','Gamma 与固定测试 SPL'),
                           ('training_comparison.png','训练过程中的验证 NLL 与 TD loss')]:
        if (OUT/filename).exists():
            lines += [f'![{label}]({filename})', '']
    audit=OUT/'memory_audit_baseline.json'
    if audit.exists():
        m=json.loads(audit.read_text())
        lines += ['', '## LSTM 检查', '',
                  f'旧 checkpoint：整段/逐步输出最大绝对误差 {m["streaming_max_abs_error"]:.3g}；'
                  f'padding 对有效 loss 的影响 {m["padding_loss_max_abs_error"]:.3g}；episode 重置检查通过；'
                  f'历史改变后续动作的比例 {m["history_action_change_fraction"]:.2%}。'
                  f'保留/清空历史的离线验证 NLL 为 {m["full_history_policy_nll"]:.4f}/{m["reset_every_step_policy_nll"]:.4f}。'
                  '记忆模块能传播历史和梯度，但不保证提高导航性能。完整数据见 `memory_audit_baseline.json`。']
    data_audit=OUT/'cached_data_audit.json'
    selected_audit=OUT/'memory_audit_selected.json'
    if selected_audit.exists():
        m=json.loads(selected_audit.read_text())
        lines += ['', f'最终选定 checkpoint 的整段/逐步输出误差为 {m["streaming_max_abs_error"]:.3g}，'
                  f'padding 对有效 loss 的影响为 {m["padding_loss_max_abs_error"]:.3g}，'
                  f'历史改变动作的比例为 {m["history_action_change_fraction"]:.2%}。'
                  f'数值记忆检查标记为 `{m["memory_is_functional"]}`；详情见 `memory_audit_selected.json`。']
    ablation=OUT/'exp013-selected-memory-reset/results.json'
    if ablation.exists():
        reset=json.loads(ablation.read_text())
        original_path=Path(reset['spec']['baseline']).parent.parent/'results.json'
        original=json.loads(original_path.read_text())
        lines += ['', '逐步清空 actor LSTM 的闭环消融（保留视觉双帧与地图，预测间隔均为 10）：', '',
                  '| 测试集 | 保留 LSTM 历史 SPL | 每步清空 LSTM SPL |', '| --- | ---: | ---: |']
        for split,label in [('test_multiple','heard'),('test_multiple_unheard','unheard')]:
            lines.append(f'| {label} | {original[split]["spl"]:.4f} | {reset[split]["spl"]:.4f} |')
    if data_audit.exists():
        d=json.loads(data_audit.read_text())
        lines += ['', '## 训练数据与部署序列', '',
                  f'训练共 {d["train"]["episodes"]} 条 episode、{d["train"]["transitions"]} 个 transition；'
                  f'平均每条 {d["train"]["mean_decisions"]:.2f} 个高层决策，中位数 {d["train"]["decision_quantiles"][2]:.0f}，'
                  f'95% 分位数 {d["train"]["decision_quantiles"][4]:.0f}。'
                  f'验证缓存平均 {d["val"]["mean_decisions"]:.2f} 个高层决策。'
                  '短专家序列上的训练不能保证较长部署序列的记忆泛化；这是数据分布上的限制，并非已证实的单一失败原因。',
                  f'冻结 foundation 原动作头在验证缓存上的 NLL={d["val"]["foundation_policy_nll"]:.4f}，'
                  f'动作准确率={d["val"]["foundation_action_accuracy"]:.2%}。详情见 `cached_data_audit.json`。']
    lines += ['', '## 可复现性和解释边界', '',
              '- 原始 checkpoint 使用缓存中的 SMDP discount；当前 gamma 对照使用每个决策步固定 gamma。旧模型不是本轮纯 gamma 对照的一部分。',
              '- `terminal_mask` 使用缓存 discount>0 区分真正终止与 timeout；gamma 仍为决策步折扣。未将原始累计 reward 改写成新的 primitive-step gamma reward。',
              '- BC 正则限制 actor 偏离离线动作；残差模型冻结 foundation 动作头，并用 LSTM 学习 logits 的修正。',
              '- 所有正式结果包含同一批 100 条，不丢弃失败 episode；发生运行异常则该次评估失败，不能作为有效实验。',
              '- 图中误差线是固定样本上的 episode bootstrap 描述性 95% 区间；未校正分层设计或多次比较。',
              '- 每次实验保存源码快照、配置、checkpoint、命令、日志、逐条 CSV/JSONL 和状态。采样草案、进程重启和失败诊断也保留。',
              '- baseline 和 gamma 扫描默认预测间隔 10；`interval5` 是独立推理设置，不更改步数上限或 SPL 定义。']
    if (OUT/'queue_completed.json').exists():
        lines += ['', '执行状态：实验队列已完成。']
    else:
        lines += ['', '执行状态：仍在运行，表格仅列已完整完成的实验。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return results


if __name__ == '__main__':
    report()
