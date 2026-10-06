"""Verify all completed runs, quantify paired differences, and publish report."""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from .audit_offline_memory import audit
from .report_gamma_study import report
from .run_gamma_study import OUT, FOUNDATION, CACHE, sha, write, summarize, evaluate


def finalize():
    assert (OUT/'queue_completed.json').exists(), 'Evaluation queue is unfinished'
    folders = sorted(p.parent for p in OUT.glob('exp*/results.json'))
    assert len(folders) == 14, [p.name for p in folders]
    selected = json.loads((OUT/'recommended_checkpoint.json').read_text())
    candidates = json.loads((OUT/'final_validation_candidates.json').read_text())['candidates']
    for candidate in candidates:
        folder=OUT/candidate['experiment']
        result=json.loads((folder/'results.json').read_text())
        if 'val_multiple' not in result:
            result['val_multiple']=evaluate(folder,Path(candidate['checkpoint']),'val_multiple',True,folder/'evaluation.yaml')
            write(folder/'results.json',result)
    expected = {}
    dataset_hashes = {}
    for split in ['test_multiple','test_multiple_unheard','val_multiple']:
        manifest=json.loads((OUT/'selections'/split/'manifest.json').read_text())
        dataset_hashes[split]=sha(Path(manifest['source']))
        assert dataset_hashes[split]==manifest['source_sha256'], f'Dataset changed: {split}'
    for split in ['test_multiple', 'test_multiple_unheard']:
        path = OUT/'selections'/split/'episodes.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        expected[split] = {(r['scene'], r['episode_id']) for r in rows}
        assert len(rows) == len(expected[split]) == 100
    arrays, checks = {}, []
    for folder in folders:
        result = json.loads((folder/'results.json').read_text())
        metrics_path = folder/'training/metrics.jsonl'
        if metrics_path.exists():
            metrics=[]
            for line in metrics_path.read_text().splitlines():
                source=json.loads(line)
                row=dict(epoch=source['epoch'],global_step=source['global_step'])
                for split in ['train','val']:
                    row.update({f'{split}_{key}':value for key,value in source[split].items()})
                metrics.append(row)
            with (folder/'training_metrics.csv').open('w') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
        assert result['spec']['foundation_sha256'] == sha(FOUNDATION)
        assert json.loads((folder/'status.json').read_text())['status'] == 'completed'
        for split in expected:
            summary = result[split]
            rows = [json.loads(line) for line in (folder/split/'episodes.jsonl').read_text().splitlines()]
            assert len(rows) == len({(r['scene'], r['episode_id']) for r in rows}) == 100
            assert {(r['scene'], r['episode_id']) for r in rows} == expected[split]
            assert summary['selection_sha256'] == sha(OUT/'selections'/split/'episodes.jsonl')
            assert summary['checkpoint_sha256'] == sha(Path(summary['checkpoint']))
            for run_path in (folder/split).glob('worker_*/run.json'):
                run_metadata=json.loads(run_path.read_text())
                assert run_metadata['seed']==result['spec']['seed']
                assert run_metadata['max_episode_steps']==200
                assert run_metadata['prediction_interval']==result['spec']['prediction_interval']
                assert run_metadata['checkpoint_sha256']==summary['checkpoint_sha256']
                assert run_metadata.get('reset_memory_every_step',False)==('--reset-memory' in (result['spec']['eval_extra'] or []))
                if 'foundation_sha256' in run_metadata:
                    assert run_metadata['foundation_sha256']==sha(FOUNDATION)
            assert summary['errors'] == 0 and all(r.get('error') is None for r in rows)
            assert all(0 <= r['spl'] <= 1 and r['primitive_steps'] <= 200 for r in rows)
            assert np.isclose(summary['spl'], np.mean([r['spl'] for r in rows]), atol=1e-12)
            assert np.isclose(summary['success_rate'], np.mean([r['success'] for r in rows]), atol=1e-12)
            arrays[folder.name, split] = np.array([r['spl'] for r in sorted(rows, key=lambda r:(r['scene'],r['episode_id']))])
            checks.append(dict(experiment=folder.name, split=split, episodes=100, spl=summary['spl'], passed=True))
    pairs = []
    for folder in folders:
        for baseline in ['exp001-baseline-foundation', 'exp002-baseline-offline']:
            for split in expected:
                delta = arrays[folder.name, split] - arrays[baseline, split]
                rng = np.random.default_rng(20260930)
                ci = np.quantile(delta[rng.integers(0,100,(10000,100))].mean(1), [.025,.975])
                pairs.append(dict(experiment=folder.name, baseline=baseline, split=split,
                                  mean_spl_difference=float(delta.mean()), ci95_low=float(ci[0]),ci95_high=float(ci[1]),
                                  improved_episodes=int((delta>0).sum()), worsened_episodes=int((delta<0).sum())))
    write(OUT/'paired_comparisons.json', pairs)
    cadence = OUT/'exp014-selected-interval5'
    variants = []
    for folder in [OUT/c['experiment'] for c in candidates] + [cadence]:
        result = json.loads((folder/'results.json').read_text())
        variants.append(dict(experiment=folder.name, validation_spl=result['val_multiple']['spl'],
                             heard_spl=result['test_multiple']['spl'],unheard_spl=result['test_multiple_unheard']['spl'],
                             prediction_interval=result['spec']['prediction_interval'],
                             checkpoint=selected['checkpoint'] if folder==cadence else str(folder/'training/best_policy.pt')))
    recommended = max(variants,key=lambda r:r['validation_spl'])
    recommended.update(selection='top three configurations by offline validation policy NLL plus interval-5 variant; final choice by validation rollout SPL')
    write(OUT/'validation_comparison.json',variants)
    audit(recommended['checkpoint'], CACHE, OUT/'memory_audit_selected.json')
    for source,name in [(Path(recommended['checkpoint']),'best_offlinerl.pt'),(FOUNDATION,'best_foundation.pt')]:
        destination=OUT/name
        temporary=destination.with_suffix('.pt.tmp')
        shutil.copy2(source,temporary);temporary.replace(destination)
        assert sha(destination)==sha(source)
    shutil.copy2(OUT/recommended['experiment']/'evaluation.yaml',OUT/'selected_evaluation.yaml')
    recommended.update(exported_checkpoint=str(OUT/'best_offlinerl.pt'),foundation_checkpoint=str(OUT/'best_foundation.pt'))
    write(OUT/'recommendation.json',recommended)
    summarize(); report()
    path=OUT/'RESULTS.md'
    path.write_text(path.read_text()+'\n## 推荐配置\n\n'+
                   f'`{recommended["experiment"]}`：验证 SPL={recommended["validation_spl"]:.4f}，'
                   f'heard SPL={recommended["heard_spl"]:.4f}，unheard SPL={recommended["unheard_spl"]:.4f}。'
                   '选模依据见 `recommendation.json`，配对差值见 `paired_comparisons.json`。'
                   '导出权重为 `best_offlinerl.pt` 和 `best_foundation.pt`，推理配置为 `selected_evaluation.yaml`。\n')
    for src in ['/tmp/avwan-gamma-baseline.log','/tmp/avwan-gamma-queue.log','/tmp/avwan-gamma-pretrain.log',
                '/tmp/avwan-alternative-pretrain.log','/tmp/avwan-alternative-queue.log']:
        if Path(src).exists():shutil.copy2(src,OUT/Path(src).name)
    write(OUT/'final_audit.json',dict(status='passed',experiments=len(folders),test_rollouts=2800,
           checks=checks,dataset_hashes=dataset_hashes,foundation_sha256=sha(FOUNDATION),completed_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(recommended,indent=2),flush=True)


if __name__ == '__main__':
    finalize()
