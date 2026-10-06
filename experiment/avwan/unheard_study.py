"""Fixed-sample, validation-selected experiments for unseen-sound generalization."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import run_gamma_study as common

ROOT = common.ROOT
OUT = ROOT / 'result/2026-10-03-unheard-generalization-study'
OLD = ROOT / 'result/2026-09-30-multiple-gamma-study'
PYTHON = '/home/getuanhui/anaconda3/envs/ss/bin/python'


def configure():
    common.OUT = OUT
    return OUT


def evaluate(name, checkpoint, foundation=None, splits=None, extra=None):
    configure()
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    configuration = folder/'evaluation.yaml'
    if not configuration.exists():
        shutil.copy2(OUT/'evaluation.yaml', configuration)
    if foundation:
        common.FOUNDATION = Path(foundation).resolve()
    common.write(folder/'status.json', dict(status='evaluating', updated_at=common.now()))
    snapshot = folder/'source_snapshot'
    if not snapshot.exists():
        shutil.copytree(ROOT/'experiment/avwan', snapshot,
                        ignore=shutil.ignore_patterns('__pycache__', 'data_cache', 'runs'))
    results = json.loads((folder/'results.json').read_text()) if (folder/'results.json').exists() else {}
    for split in splits or ['val_multiple_unheard', 'test_multiple', 'test_multiple_unheard']:
        results[split] = common.evaluate(folder, Path(checkpoint).resolve(), split,
                                        bool(foundation), configuration, extra)
        common.write(folder/'results.json', results)
    common.write(folder/'status.json', dict(status='completed', updated_at=common.now()))
    return results


def collect_worker(index, shards=4):
    # Import torch before Habitat's native audio plugin, as in the evaluator.
    import torch
    import numpy as np
    from .evaluate import load_episodes
    from .config.default import get_config
    from .collector.collect import collect_episode, save_episode
    from experiment import AVWANRLEnv
    torch.set_num_threads(2)
    config = get_config(str(OUT/'evaluation.yaml'), opts=['TASK_CONFIG.DATASET.SPLIT', 'val_multiple_unheard'])
    config.defrost()
    config.PREDICTION_INTERVAL = 10  # match expert training cache cadence
    config.TASK_CONFIG.TASK.MEASUREMENTS = [x for x in config.TASK_CONFIG.TASK.MEASUREMENTS if x != 'TOP_DOWN_MAP']
    config.freeze()
    dataset = load_episodes(config, str(OUT/'selections/val_multiple_unheard'))
    episodes = list(dataset.episodes)
    env = AVWANRLEnv(config, dataset)
    new = 0
    try:
        for i in range(index, len(episodes), shards):
            episode = episodes[i]
            scene, eid = Path(episode.scene_id).stem, str(episode.episode_id)
            path = OUT/'validation_demonstrations/episodes'/scene/(eid+'.pkl.gz')
            if path.exists():
                continue
            env.habitat_env.current_episode = episode
            env.seed(20260930+i)
            transitions, metadata, _ = collect_episode(env)
            metadata.update(scene=scene, episode_id=eid, split='val_multiple_unheard', sound=episode.info['sound'])
            if not transitions:
                raise ValueError(f'Empty demonstration: {scene}/{eid}')
            save_episode(path, transitions, metadata)
            print(json.dumps(metadata), flush=True)
            new += 1
            import resource
            if new >= 10 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 >= 6000:
                break
    finally:
        env.close()


def collect():
    configure()
    folder = OUT/'validation_demonstrations'
    folder.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (OUT/'selections/val_multiple_unheard/episodes.jsonl').read_text().splitlines()]
    def worker(index):
        expected = rows[index::4]
        while not all((folder/'episodes'/r['scene']/(r['episode_id']+'.pkl.gz')).exists() for r in expected):
            common.command_run([PYTHON, '-m', 'experiment.avwan.unheard_study', 'collect-worker', '--index', str(index)],
                               folder, f'worker_{index}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))
    common.write(folder/'status.json', dict(status='completed', episodes=100, training_use=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['evaluate', 'collect', 'collect-worker'])
    p.add_argument('--name')
    p.add_argument('--checkpoint')
    p.add_argument('--foundation')
    p.add_argument('--splits', nargs='+')
    p.add_argument('--index', type=int)
    a = p.parse_args()
    if a.command == 'evaluate':
        evaluate(a.name, a.checkpoint, a.foundation, a.splits)
    elif a.command == 'collect':
        collect()
    else:
        collect_worker(a.index)


if __name__ == '__main__':
    main()
