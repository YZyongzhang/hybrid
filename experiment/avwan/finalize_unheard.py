"""Audit fixed manifests, export validation-selected weights, and summarize."""
import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .unheard_study import OUT, ROOT, PYTHON
from .run_gamma_study import sha, write, now
from .report_unheard import report


def finalize():
    for marker in ['initial_queue_complete.json', 'offline_queue_complete.json',
                   'correction_queue_complete.json', 'detail_queue_complete.json',
                   'corrective_memory_queue_complete.json', 'ensemble_queue_complete.json',
                   'ratio_queue_complete.json', 'ratio_offline_queue_complete.json', 'revisit_queue_complete.json']:
        assert (OUT/marker).exists(), marker
    paths = sorted(OUT.glob('exp*/results.json'))
    assert len(paths) == 14
    assert {p.parent.name.split('-')[0] for p in paths} == {f'exp{i:03d}' for i in range(1, 15)}
    splits = ['val_multiple_unheard', 'test_multiple', 'test_multiple_unheard']
    expected = {};manifest_hashes = {}
    for split in splits:
        path = OUT/'selections'/split/'episodes.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        expected[split] = {(r['scene'], r['episode_id']) for r in rows}
        assert len(rows) == len(expected[split]) == 100
        manifest_hashes[split] = sha(path)
        manifest = json.loads((path.parent/'manifest.json').read_text())
        assert sha(Path(manifest['source'])) == manifest['source_sha256']
    audit, arrays, candidates, by_sound = [], {}, [], []
    for path in paths:
        folder = path.parent;results = json.loads(path.read_text())
        for split in splits:
            summary = results[split]
            rows = [json.loads(x) for x in (folder/split/'episodes.jsonl').read_text().splitlines()]
            assert len(rows) == len({(r['scene'], r['episode_id']) for r in rows}) == 100
            assert {(r['scene'], r['episode_id']) for r in rows} == expected[split]
            assert summary['selection_sha256'] == manifest_hashes[split]
            assert sha(Path(summary['checkpoint'])) == summary['checkpoint_sha256']
            assert summary['errors'] == 0 and all(not r.get('error') for r in rows)
            assert all(0 <= r['spl'] <= 1 and r['primitive_steps'] <= 200 for r in rows)
            assert np.isclose(summary['spl'], np.mean([r['spl'] for r in rows]))
            assert np.isclose(summary['success_rate'], np.mean([r['success'] for r in rows]))
            for run_path in (folder/split).glob('worker_*/run.json'):
                run = json.loads(run_path.read_text())
                assert run['seed'] == 20260930 and run['max_episode_steps'] == 200 and run['prediction_interval'] == 5
                assert run['checkpoint_sha256'] == summary['checkpoint_sha256']
                if 'foundation_checkpoint' in run:
                    assert sha(Path(run['foundation_checkpoint'])) == run['foundation_sha256']
                assert run.get('revisit_penalty', 0.) == (1. if folder.name.startswith('exp014') else 0.)
            values = np.array([r['spl'] for r in sorted(rows, key=lambda r:(r['scene'], r['episode_id']))])
            arrays[folder.name, split] = values
            rng = np.random.default_rng(20261003)
            interval = np.quantile(values[rng.integers(0, 100, (10000, 100))].mean(1), [.025, .975])
            audit.append(dict(experiment=folder.name, split=split, episodes=100, spl=summary['spl'],
                              ci95_low=float(interval[0]), ci95_high=float(interval[1]), passed=True))
            for sound in sorted({r['sound'] for r in rows}):
                subset = [r for r in rows if r['sound'] == sound]
                by_sound.append(dict(experiment=folder.name, split=split, sound=sound, episodes=len(subset),
                                     spl=float(np.mean([r['spl'] for r in subset])),
                                     success_rate=float(np.mean([r['success'] for r in subset])),
                                     timeout=sum(bool(r.get('truncated')) for r in subset),
                                     failed_stop=sum(not r['success'] and r['actions'][-1] == 40 for r in subset)))
        candidates.append(dict(experiment=folder.name, validation_spl=results['val_multiple_unheard']['spl'],
                               heard_spl=results['test_multiple']['spl'], unheard_spl=results['test_multiple_unheard']['spl'],
                               checkpoint=results['val_multiple_unheard']['checkpoint']))
        metrics = folder/'training/metrics.jsonl'
        if metrics.exists():
            records = []
            for line in metrics.read_text().splitlines():
                source = json.loads(line);row = dict(epoch=source['epoch'])
                for split, values in source.items():
                    if isinstance(values, dict):row.update({f'{split}_{k}':v for k,v in values.items()})
                records.append(row)
            with (folder/'training_metrics.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (OUT/'per_sound.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(by_sound[0]));writer.writeheader();writer.writerows(by_sound)
    pairs = []
    for folder in [p.parent for p in paths]:
        for split in ['test_multiple', 'test_multiple_unheard']:
            delta = arrays[folder.name, split]-arrays['exp001-previous-offline-interval5', split]
            rng = np.random.default_rng(20261003)
            interval=np.quantile(delta[rng.integers(0,100,(10000,100))].mean(1),[.025,.975])
            pairs.append(dict(experiment=folder.name,split=split,mean_spl_difference=float(delta.mean()),
                              ci95_low=float(interval[0]),ci95_high=float(interval[1])))
    write(OUT/'paired_comparisons.json',pairs)
    selected = max(candidates, key=lambda r:r['validation_spl'])
    selected.update(selection='highest independent unheard validation rollout SPL',
                    target_unheard_spl=.45, best_observed_unheard=max(r['unheard_spl'] for r in candidates),
                    any_test_configuration_reached_target=any(r['unheard_spl'] >= .45 for r in candidates),
                    stable_improvement_confirmed=False,
                    evaluation_caveat='Single-run exploratory scores: equivalent policies agree on identical observations but differ across simulator runs; see REPRODUCIBILITY.md')
    folder = OUT/selected['experiment']
    run = json.loads((folder/'test_multiple_unheard/worker_0/run.json').read_text())
    shutil.copy2(selected['checkpoint'], OUT/'best_policy.pt')
    assert sha(OUT/'best_policy.pt') == sha(Path(selected['checkpoint']))
    selected['exported_policy'] = str(OUT/'best_policy.pt')
    selected['revisit_penalty'] = run.get('revisit_penalty', 0.)
    if 'foundation_checkpoint' in run:
        shutil.copy2(run['foundation_checkpoint'], OUT/'best_foundation.pt')
        shutil.copy2(selected['checkpoint'], OUT/'best_offlinerl.pt')
        selected['policy_type'] = 'frozen_foundation_recurrent_actor'
        selected['foundation'] = str(OUT/'best_foundation.pt')
    elif run['policy'] == 'foundation_probability_ensemble_masked_argmax':
        shutil.copy2(selected['checkpoint'], OUT/'best_ensemble.pt')
        selected['policy_type'] = 'foundation_probability_ensemble'
    else:
        shutil.copy2(selected['checkpoint'], OUT/'best_foundation.pt')
        selected['policy_type'] = 'fine_tuned_foundation_policy'
    shutil.copy2(folder/'evaluation.yaml', OUT/'selected_evaluation.yaml')
    commands = [];shards_by_split = {}
    for split in ['test_multiple', 'test_multiple_unheard']:
        metadata=json.loads((folder/split/'worker_0/run.json').read_text())
        shards_by_split[split]=metadata.get('shards',1)
        for shard in range(shards_by_split[split]):
            cmd = [PYTHON, '-m', 'experiment.avwan.evaluate', '--checkpoint', str(OUT/'best_policy.pt'),
                   '--config', str(OUT/'selected_evaluation.yaml'), '--split', split,
                   '--selection-source', str(OUT/'selections'/split), '--seed', '20260930',
                   '--shards', str(shards_by_split[split]), '--shard-index', str(shard), '--resume', '--max-new-episodes', '25',
                   '--max-rss-mb', '8192', '--output', str(OUT/'recheck'/split/f'worker_{shard}')]
            if 'foundation' in selected:cmd += ['--foundation-checkpoint', selected['foundation']]
            if selected['revisit_penalty']:cmd += ['--revisit-penalty', str(selected['revisit_penalty'])]
            commands.append(cmd)
    write(OUT/'reproduction_commands.json', commands)
    write(OUT/'recommendation.json', selected)
    (OUT/'reproduce_selected.py').write_text(
        'from pathlib import Path\nimport json,os,sys\n'
        'OUT=Path(__file__).resolve().parent\nROOT=OUT.parents[1]\nsys.path.insert(0,str(ROOT))\nos.chdir(ROOT)\n'
        'from experiment.avwan import run_gamma_study as driver\n'
        'selected=json.loads((OUT/"recommendation.json").read_text())\ndriver.OUT=OUT\n'
        'offline="foundation" in selected\n'
        'if offline:driver.FOUNDATION=OUT/"best_foundation.pt"\n'
        f'shards={shards_by_split!r}\n'
        'extra=["--revisit-penalty",str(selected["revisit_penalty"])] if selected["revisit_penalty"] else None\n'
        'for split,count in shards.items():\n os.environ["AVWAN_EVAL_SHARDS"]=str(count)\n'
        ' driver.evaluate(OUT/"recheck",OUT/"best_policy.pt",split,offline,OUT/"selected_evaluation.yaml",extra)\n')
    write(OUT/'validation_comparison.json', candidates)
    report()
    from .diagnose_unheard_results import diagnose
    diagnose()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [p.parent.name for p in paths];x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5))
    for shift, split, label in [(-.18,'test_multiple','heard'),(.18,'test_multiple_unheard','unheard')]:
        means = np.array([arrays[n,split].mean() for n in names])
        checks = [next(a for a in audit if a['experiment']==n and a['split']==split) for n in names]
        errors = np.array([[m-a['ci95_low'] for m,a in zip(means,checks)], [a['ci95_high']-m for m,a in zip(means,checks)]])
        ax.bar(x+shift,means,.36,label=label,yerr=errors,capsize=3)
    ax.axhline(.45,color='black',linestyle='--',label='unheard target 0.45')
    ax.set_xticks(x,[n.split('-')[0] for n in names]);ax.set_ylim(0,1);ax.set_ylabel('SPL');ax.legend()
    fig.tight_layout();fig.savefig(OUT/'spl_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for path in paths:
        metrics=path.parent/'training/metrics.jsonl'
        if not metrics.exists():continue
        rows=[json.loads(line) for line in metrics.read_text().splitlines()]
        key='action_loss' if 'action_loss' in rows[0]['val'] else 'policy_nll'
        xs=[r['epoch'] for r in rows];ys=[r['val'][key] for r in rows]
        initial=metrics.parent/'initial_validation.json'
        if initial.exists():
            first=json.loads(initial.read_text());first=first.get('val',first)
            if key in first:xs.insert(0,0);ys.insert(0,first[key])
        axis=axes[0 if key=='action_loss' else 1]
        axis.plot(xs,ys,label=path.parent.name.split('-')[0])
    for axis,title in zip(axes,['Foundation adaptations','Recurrent actors']):
        axis.set_title(title);axis.set_xlabel('Epoch');axis.set_ylabel('Unheard validation NLL');axis.legend()
    fig.tight_layout();fig.savefig(OUT/'training_comparison.png',dpi=180);plt.close(fig)
    report_path = OUT/'RESULTS.md'
    report_path.write_text(report_path.read_text()+
        '\n## 后续方法与最终选择\n\nexp005：新foundation上的SAC-CQL+BC。exp006：冻结动作先验的纯BC循环残差。'
        'exp007：训练场景的纠正轨迹微调。exp008：增加完整声学网格和局部几何细节分支。'
        'exp009：在原始及纠正轨迹上训练仅优化专家动作交叉熵的循环残差策略。'
        'exp010：三种声学处理模型的动作概率等权集成。'
        'exp011：逐时频双耳比值、固定尺度及冻结视觉/融合主干的适配分支；仅非STOP样本使用角度辅助损失。'
        'exp012：双耳比值foundation上的纠正轨迹循环BC。exp013：同一foundation上的SAC-CQL，gamma=.01、BC=5、冻结动作先验。'
        'exp014：对验证最好的模型增加重复访问移动动作调整，保持STOP与移动的判定间隔。\n\n'+
        f'最终按独立unheard验证SPL选择 `{selected["experiment"]}`，验证SPL={selected["validation_spl"]:.4f}，'
        f'heard测试SPL={selected["heard_spl"]:.4f}，unheard测试SPL={selected["unheard_spl"]:.4f}。\n\n'+
        ('至少一组固定unheard测试达到0.45。' if selected['any_test_configuration_reached_target'] else '本轮固定unheard测试仍未达到0.45。')+
        '\n\n![固定测试比较](spl_comparison.png)\n\n![训练验证曲线](training_comparison.png)\n\n每个测试分组均100条、零运行错误。exp001的200条测试引用上一轮，'
        '其余13组共新增2600条测试，另有1400条验证导航。采样、原始失败尝试和恢复记录均保留。'
        '配对bootstrap区间是固定子集上的描述性统计，未校正重复比较或仿真数值波动。'
        '导出权重见recommendation.json；运行reproduce_selected.py可按原分片数复测并自动续跑，原始调用见reproduction_commands.json。\n')
    report_path.write_text(report_path.read_text()+
        '\n失败类型和成功路径效率见[FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md)，距离分层见distance_strata.csv。\n')
    report_path.write_text(report_path.read_text()+
        '\n**运行一致性限制：**同一观测下，exp011与epoch0的exp012在595个状态上动作完全一致；'
        '独立仿真运行却出现大量首动作和成功结果差异。不能将这些差异归因于记忆收益；'
        '10对初始观测复测中RGB/深度全部相同，7对声谱图不同。'
        '最高测试分数是单次探索性观察，尚未确认稳定收益，bootstrap区间也未包含运行间变化。'
        '详情见[REPRODUCIBILITY.md](REPRODUCIBILITY.md)。\n')
    assert json.loads((OUT/'exp012-ratio-corrective-recurrent-bc/same_observation_equivalence.json').read_text())['passed']
    for name in ['exp009-corrective-recurrent-bc', 'exp012-ratio-corrective-recurrent-bc']:
        memory_best=json.loads((OUT/name/'training/best.json').read_text())
        if memory_best['epoch']==0:
            report_path.write_text(report_path.read_text()+
                f'\n{name}的验证选择为epoch0，即未修改的foundation动作先验。该结果不能证明循环模块带来收益；'
                '零残差是checkpoint选择的结果，不能据此判断LSTM实现断开。循环模型的数值检查见各实验memory_audit.json及latest_memory_audit.json。\n')
    for name in ['exp012-ratio-corrective-recurrent-bc', 'exp013-ratio-sac-g0p01-prior-bc5']:
        assert json.loads((OUT/name/'latest_memory_audit.json').read_text())['memory_is_functional']
    assert json.loads((OUT/'exp011-foundation-binaural-ratio-adapter/training/frozen_backbone_audit.json').read_text())['passed']
    hashes={}
    for source in (ROOT/'experiment/avwan').rglob('*'):
        relative=source.relative_to(ROOT/'experiment/avwan')
        if (not source.is_file() or source.suffix not in ['.py','.md','.yaml']
                or any(p in ['data_cache','runs','__pycache__'] for p in relative.parts)):continue
        relative=source.relative_to(ROOT);destination=OUT/'final_source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        hashes[str(relative)]=sha(source)
    write(OUT/'source_hashes.json',hashes)
    (OUT/'source_changes.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=ROOT))
    write(OUT/'git_revision.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                                      status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)))
    write(OUT/'final_audit.json', dict(status='passed',completed_at=now(),experiments=14,
          test_results=2800,new_test_rollouts=2600,reused_test_rollouts=200,validation_rollouts=1400,
          checks=audit,selection_hashes=manifest_hashes,scope='record integrity and declared evaluation protocol',
          simulator_reproducibility_guaranteed=False,identical_observation_policy_equivalence_passed=True,
          controlled_diagnostic_initial_resets=20,additional_diagnostic_attempts_preserved=True,
          diagnostic_resets_scored_as_rollouts=False))
    print(json.dumps(selected,indent=2),flush=True)


if __name__ == '__main__':
    finalize()
