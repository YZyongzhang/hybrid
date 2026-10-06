"""Summarize completed groups without discarding unsuccessful episodes."""
import csv
import json
from pathlib import Path

from .unheard_study import OUT


def report():
    records = []
    splits = ['val_multiple_unheard', 'test_multiple', 'test_multiple_unheard']
    for path in sorted(OUT.glob('exp*/results.json')):
        result = json.loads(path.read_text())
        row = dict(experiment=path.parent.name)
        for split in splits:
            values = result.get(split, {})
            for key in ['spl', 'success_rate', 'episodes', 'errors']:
                row[f'{split}_{key}'] = values.get(key)
        records.append(row)
    if not records:
        return
    with (OUT/'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    text = ['# Unheard 泛化改进实验', '',
            '目标：固定100条 unheard 测试平均 SPL≥0.45。heard/unheard沿用上一轮清单，各100条、seed=20260930、最多200步。',
            '新增独立 unseen-sound 验证集：100条，验证声音与训练及测试unheard声音互不重叠。仅训练集用于更新模型。', '',
            '| 实验 | Unheard验证 SPL | Heard测试 SPL | Unheard测试 SPL | Unheard成功率 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for r in records:
        keys = ['val_multiple_unheard_spl', 'test_multiple_spl', 'test_multiple_unheard_spl', 'test_multiple_unheard_success_rate']
        text.append('| '+r['experiment']+' | '+' | '.join('—' if r[k] is None else f'{r[k]:.4f}' for k in keys)+' |')
    text += ['', 'exp001为上一轮最终推荐offline模型，测试指标引用原始逐episode结果；本轮补测其unheard验证集。',
             'exp002/003/004：从预训练foundation微调，分别为共同双耳增益与音色增强、再加相对声强地图、再加保留双耳差异的频谱白化。',
             '每个模型内部按独立unheard验证动作NLL选checkpoint，最终配置按unheard验证导航SPL选择；测试比较为固定子集上的探索性结果。',
             '', '配置/训练曲线/权重在各exp目录，逐episode CSV/JSONL、调用命令和日志在各split子目录。空白表示尚未完成该组评估。']
    (OUT/'RESULTS.md').write_text('\n'.join(text)+'\n')


if __name__ == '__main__':
    report()
