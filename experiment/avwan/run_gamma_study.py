"""Reproducible fixed-episode study; every invocation is retained under result/."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'result/2026-09-30-multiple-gamma-study'
FOUNDATION = ROOT / 'result/9-29-multiple-foundation/best_foundation.pt'
CACHE = ROOT / 'result/9-29-multiple-foundation/features'
SEED = 20260930


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    evaluation = yaml.safe_load((ROOT / 'result/9-29-multiple-foundation/evaluation.yaml').read_text())
    evaluation['BASE_TASK_CONFIG_PATH'] = str(ROOT / 'result/9-29-multiple-foundation/foundation.yaml')
    (OUT / 'evaluation.yaml').write_text(yaml.safe_dump(evaluation))
    for split in ['test_multiple', 'test_multiple_unheard', 'val_multiple']:
        dest = OUT / 'selections' / split
        if (dest / 'manifest.json').exists():
            continue
        path = ROOT / f'data/datasets/audionav/mp3d/v1/{split}/{split}.json.gz'
        if not path.exists():
            path = path.with_suffix('')
        with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
            episodes = json.load(f)['episodes']
        # Allocate scene quotas first so rare scenes cannot disappear through
        # rounding hundreds of tiny joint strata. No policy outcomes are read.
        groups = {}
        for ep in episodes:
            info = ep['info']
            distance_bin = int(np.digitize(info['geodesic_distance'], [5, 10, 20]))
            key = Path(ep['scene_id']).stem
            groups.setdefault(key, []).append(ep)
        rng = random.Random(SEED)
        keys = sorted(groups)
        rng.shuffle(keys)
        quota = {k: max(1, int(100 * len(groups[k]) / len(episodes))) for k in keys}
        while sum(quota.values()) < 100:
            k = max(keys, key=lambda k: 100 * len(groups[k]) / len(episodes) - quota[k])
            quota[k] += 1
        while sum(quota.values()) > 100:
            k = max((k for k in keys if quota[k] > 1), key=lambda k: quota[k] - 100 * len(groups[k]) / len(episodes))
            quota[k] -= 1
        chosen = []
        for k in keys:
            pool = sorted(groups[k], key=lambda ep: (ep['info']['sound'], ep['info']['geodesic_distance'], str(ep['episode_id'])))
            start = rng.random()
            chosen.extend(pool[int((i + start) * len(pool) / quota[k])] for i in range(quota[k]))
        rows = sorted([dict(scene=Path(ep['scene_id']).stem, episode_id=str(ep['episode_id']),
                            sound=ep['info']['sound'], distance=ep['info']['geodesic_distance'], saved=True)
                       for ep in chosen], key=lambda r: (r['scene'], r['episode_id']))
        assert len(rows) == len({(r['scene'], r['episode_id']) for r in rows}) == 100
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        write(dest / 'manifest.json', dict(seed=SEED, method='proportional scene quotas (minimum one); systematic sampling sorted by sound and distance within scene',
              bins=[5, 10, 20], source=str(path), source_sha256=sha(path), population=len(episodes), selected=100,
              selection_sha256=sha(dest / 'episodes.jsonl'), scenes=dict(Counter(r['scene'] for r in rows)),
              sounds=dict(Counter(r['sound'] for r in rows))))


def command_run(command, folder, label):
    write(folder / (label + '_command.json'), command)
    with (folder / (label + '.log')).open('a') as log:
        subprocess.run(command, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       MPLBACKEND='Agg', MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet', TF_CPP_MIN_LOG_LEVEL='3',
                       PYTHONUNBUFFERED='1'), stdout=log, stderr=subprocess.STDOUT, check=True)


def evaluate(folder, checkpoint, split, offline, evaluation=None, extra=None):
    target = folder / split
    if (target / 'results.json').exists():
        return json.loads((target / 'results.json').read_text())
    target.mkdir(parents=True, exist_ok=True)
    prior_run = target / 'worker_0/run.json'
    shards = (json.loads(prior_run.read_text()).get('shards', 1) if prior_run.exists()
              else int(os.environ.get('AVWAN_EVAL_SHARDS', '4')))
    if shards < 1:
        raise ValueError('AVWAN_EVAL_SHARDS must be positive')
    def worker(index):
        worker_dir = target / f'worker_{index}'
        cmd = [sys.executable, '-m', 'experiment.avwan.evaluate', '--checkpoint', str(checkpoint),
               '--config', str(evaluation or OUT / 'evaluation.yaml'), '--output', str(worker_dir), '--split', split,
               '--selection-source', str(OUT / 'selections' / split), '--seed', str(SEED), '--shards', str(shards),
               '--shard-index', str(index), '--max-new-episodes', '25', '--max-rss-mb', '8192', '--resume']
        if offline:
            cmd += ['--foundation-checkpoint', str(FOUNDATION)]
        cmd += extra or []
        previous = -1
        while True:
            command_run(cmd, target, f'worker_{index}')
            s = json.loads((worker_dir / 'summary.json').read_text())
            if s['errors']:
                raise RuntimeError(f'Rollout errors: {worker_dir}')
            if s['episodes'] == s['planned']:
                return
            if s['episodes'] <= previous:
                raise RuntimeError('Evaluation made no progress')
            previous = s['episodes']
    with ThreadPoolExecutor(max_workers=shards) as pool:
        list(pool.map(worker, range(shards)))
    rows = []
    for index in range(shards):
        rows.extend(json.loads(line) for line in (target / f'worker_{index}/episodes.jsonl').read_text().splitlines())
    expected = [json.loads(line) for line in (OUT / 'selections' / split / 'episodes.jsonl').read_text().splitlines()]
    mapping = {(r['scene'], r['episode_id']): r for r in expected}
    assert len(rows) == len({(r['scene'], r['episode_id']) for r in rows}) == 100
    assert {(r['scene'], r['episode_id']) for r in rows} == set(mapping)
    rows.sort(key=lambda r: (r['scene'], r['episode_id']))
    for row in rows:
        row['sound'] = mapping[(row['scene'], row['episode_id'])]['sound']
    result = dict(episodes=100, spl=float(np.mean([r['spl'] for r in rows])),
                  success_rate=float(np.mean([r['success'] for r in rows])),
                  errors=sum(r.get('error') is not None for r in rows), checkpoint=str(checkpoint),
                  checkpoint_sha256=sha(checkpoint), selection_sha256=sha(OUT / 'selections' / split / 'episodes.jsonl'))
    write(target / 'results.json', result)
    (target / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with (target / 'episodes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['scene', 'episode_id', 'sound', 'success', 'spl', 'initial_distance',
                                'final_distance', 'primitive_steps', 'waypoint_steps', 'truncated'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(folder.name, split, json.dumps(result), flush=True)
    return result


def run(name, gamma=None, overrides=None, baseline=None, epochs=30, eval_extra=None, interval=10, train_only=False):
    prepare()
    folder = OUT / name
    if (folder / 'results.json').exists():
        print('Already complete:', name, flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    spec = dict(name=name, gamma=gamma, overrides=overrides, baseline=baseline, epochs=epochs,
                eval_extra=eval_extra, prediction_interval=interval, seed=SEED, foundation_sha256=sha(FOUNDATION))
    if (folder / 'spec.json').exists():
        assert json.loads((folder / 'spec.json').read_text()) == spec
    write(folder / 'spec.json', spec)
    write(folder / 'status.json', dict(status='running', started_at=now()))
    snapshot = folder / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for p in [Path(__file__), *Path(__file__).parent.joinpath('offline').glob('*.py'), Path(__file__).parent / 'evaluate.py']:
        if not (snapshot / p.name).exists():
            (snapshot / p.name).write_text(p.read_text())
    try:
        offline = baseline != 'foundation'
        if baseline:
            checkpoint = FOUNDATION if baseline == 'foundation' else (
                ROOT / 'result/9-29-multiple-offlinerl/best_offlinerl.pt' if baseline == 'offline' else Path(baseline))
        else:
            config = yaml.safe_load((ROOT / 'result/9-29-multiple-offlinerl/offline_rl.yaml').read_text())
            config.update(seed=SEED)
            config['features'].update(checkpoint=str(FOUNDATION), cache=str(CACHE))
            config['training'].update(output=str(folder / 'training'), gamma=gamma, epochs=epochs, tensorboard=False)
            config['training'].update(overrides or {})
            (folder / 'config.yaml').write_text(yaml.safe_dump(config))
            if not (folder / 'training/completed.json').exists():
                cmd = [sys.executable, '-m', 'experiment.avwan.train_offline_rl', '--config', str(folder / 'config.yaml'), '--stage', 'train']
                if (folder / 'training/latest.pt').exists():
                    cmd += ['--resume', str(folder / 'training/latest.pt')]
                command_run(cmd, folder, 'train')
            # Validation policy NLL determines epoch, before looking at test results.
            checkpoint = folder / 'training/best_policy.pt'
        if train_only:
            write(folder / 'status.json', dict(status='trained_pending_evaluation', time=now()))
            return
        config = yaml.safe_load((OUT / 'evaluation.yaml').read_text())
        config['PREDICTION_INTERVAL'] = interval
        (folder / 'evaluation.yaml').write_text(yaml.safe_dump(config))
        # Offline validation already selects the epoch. Reserve expensive
        # validation rollouts for baselines and the final recommended model.
        splits = ['test_multiple', 'test_multiple_unheard']
        if name in ('exp001-baseline-foundation', 'exp002-baseline-offline') or (folder / 'val_multiple').exists():
            splits.insert(0, 'val_multiple')
        results = {split: evaluate(folder, checkpoint, split, offline, folder / 'evaluation.yaml', eval_extra)
                   for split in splits}
        write(folder / 'results.json', dict(spec=spec, selection='minimum offline validation policy NLL' if not baseline else 'existing checkpoint', **results))
        write(folder / 'status.json', dict(status='completed', completed_at=now()))
    except BaseException as error:
        write(folder / 'status.json', dict(status='failed', error=repr(error), time=now()))
        raise
    summarize()


def summarize():
    rows = []
    for path in sorted(OUT.glob('*/results.json')):
        result = json.loads(path.read_text())
        row = dict(experiment=path.parent.name)
        for split, label in [('val_multiple', 'val'), ('test_multiple', 'heard'), ('test_multiple_unheard', 'unheard')]:
            row[label + '_spl'] = result.get(split, {}).get('spl')
            row[label + '_success'] = result.get(split, {}).get('success_rate')
        rows.append(row)
    if rows:
        with (OUT / 'comparison.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(OUT / 'comparison.json', rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--baseline')
    parser.add_argument('--overrides', default='{}')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--eval-extra', default='[]')
    parser.add_argument('--train-only', action='store_true')
    args = parser.parse_args()
    run(args.name, args.gamma, json.loads(args.overrides), args.baseline, args.epochs, json.loads(args.eval_extra), args.interval, args.train_only)
