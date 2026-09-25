"""Export SAC loss tables and plots; optionally include completed rollout metrics."""
import argparse
import csv
import json
from pathlib import Path


def report(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run = Path(run)
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    keys = ('actor_loss', 'critic_loss', 'td_loss', 'cql_loss', 'entropy', 'alpha',
            'alpha_loss', 'target_entropy', 'action_accuracy', 'policy_nll')
    with (run / 'sac_losses.csv').open('w', newline='') as handle:
        fields = ['epoch', 'global_step'] + [f'{split}_{key}' for split in ('train', 'val') for key in keys]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(epoch=row['epoch'], global_step=row['global_step'],
                                  **{f'{split}_{key}': row[split][key] for split in ('train', 'val') for key in keys}))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, key in zip(axes.flat, keys[:6]):
        for split in ('train', 'val'):
            ax.plot([r['epoch'] for r in rows], [r[split][key] for r in rows], label=split)
        ax.set(title=key, xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Epoch-10 frozen foundation + LSTM SAC-CQL')
    fig.tight_layout()
    fig.savefig(run / 'sac_losses.png', dpi=160)
    plt.close(fig)
    final_path = run / 'rollouts/final_results.json'
    if final_path.exists():
        final = json.loads(final_path.read_text())
        test = final['test']
        rows = ['# 第十轮 foundation + offline RL 测试结果', '',
                f'- 选模：{final["selection"]}。',
                f'- 选定模型：`{final["selected"]}`，副本 `best_spl.pt`。',
                f'- 测试集：`{test["split"]}`，随机种子 {test["seed"]}，{test["episodes"]} 个 episode。',
                f'- 平均 SPL：**{test["spl"]:.6f}**。',
                f'- 成功率：**{test["success_rate"]:.1%}**；运行错误：{test["errors"]}。', '',
                '| 验证候选 | Epoch | 验证 SPL | 验证成功率 |', '| --- | ---: | ---: | ---: |']
        for item in final['validation']:
            rows.append(f'| {item["file"]} | {item["epoch"]} | {item["rollout"]["spl"]:.6f} | '
                        f'{item["rollout"]["success_rate"]:.1%} |')
        rows += ['', '训练日志：`steps.jsonl`（每次更新）、`metrics.jsonl`（每轮）、`sac_losses.csv`、'
                 '`sac_losses.png`、`tensorboard/`。', '',
                 '测试明细：`rollouts/test_selected/episodes.jsonl`；固定抽样：`selection.json`；'
                 '聚合结果：`rollouts/final_results.json`。失败 episode 的 SPL 为 0，按全部 episode 平均。', '',
                 '测试前已按验证集选定模型；未使用 test 结果选模。']
        (run / 'RESULTS.md').write_text('\n'.join(rows) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    report(parser.parse_args().run)
