"""Training-only DAgger-style recovery data; never query experts during testing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import random
import time

from .unheard_study import OUT, ROOT, PYTHON, configure
from . import run_gamma_study as common

FOLDER = OUT/'training_corrections'


def prepare():
    FOLDER.mkdir(parents=True, exist_ok=True)
    marker = FOLDER/'selection.json'
    if marker.exists():
        return json.loads(marker.read_text())
    source = Path('/data/Getuanhui/avwan_multiple_greedy_train/episodes.jsonl')
    records = [json.loads(x) for x in source.read_text().splitlines()]
    groups = {}
    for row in records:
        if row.get('saved'):
            groups.setdefault(row['sound'], []).append(row)
    # Equal coverage of training sounds, systematic scene/distance coverage.
    rng = random.Random(20261003)
    selected = []
    for sound, pool in sorted(groups.items()):
        pool.sort(key=lambda r: (r['scene'], r['initial_distance'], r['episode_id']))
        count = min(4, len(pool))
        start = rng.random()
        selected.extend(pool[int((i+start)*len(pool)/count)] for i in range(count))
    selected.sort(key=lambda r: (r['scene'], r['episode_id']))
    keys = {(r['scene'], str(r['episode_id'])) for r in selected}
    scenes = {r['scene'] for r in selected}
    dataset = []
    root = ROOT/'data/datasets/audionav/mp3d/v1/train_multiple/content'
    for scene in sorted(scenes):
        files = [p for p in root.glob(scene+'.json*') if p.name.endswith(('.json', '.json.gz'))]
        if len(files) != 1:
            raise ValueError(f'Expected one scene dataset: {scene}: {files}')
        p = files[0]
        with (gzip.open(p, 'rt') if p.suffix == '.gz' else p.open()) as f:
            eps = json.load(f)['episodes']
        dataset.extend(e for e in eps if (Path(e['scene_id']).stem, str(e['episode_id'])) in keys)
    assert len(dataset) == len(keys) == len(selected)
    audit = json.loads((OUT/'split_audit.json').read_text())
    assert not scenes & set(audit['test_multiple_unheard']['scenes'])
    assert not scenes & set(audit['val_multiple_unheard']['scenes'])
    assert not set(groups) & set(audit['test_multiple_unheard']['sound_classes'])
    common.write(FOLDER/'dataset.json', dict(episodes=dataset))
    result = dict(episodes=selected, source=str(source), source_sha256=common.sha(source),
                  dataset_sha256=common.sha(FOLDER/'dataset.json'), seed=20261003,
                  method='up to four episodes per training sound, systematic scene/distance order',
                  expert_mixture_probability=.3, stop_correction_training_only=True,
                  random_nonstop_probability=.05, validation_test_disjoint=True)
    common.write(marker, result)
    return result


def worker(index):
    import torch
    import numpy as np
    from .evaluate import LearnedPolicy, load_episodes
    from .offline.features import load_foundation
    from .config.default import get_config
    from .collector.collect import save_episode
    from .environment.BFSExpert import NoReachableWaypointError
    from experiment import AVWANRLEnv
    torch.set_num_threads(2)
    chosen = json.loads((OUT/'offline_foundation_selection.json').read_text())
    policy = LearnedPolicy(load_foundation(chosen['checkpoint'], 'cuda:0'), 'cuda:0')
    config = get_config(str(OUT/'evaluation.yaml'), opts=['TASK_CONFIG.DATASET.SPLIT', 'train_multiple'])
    config.defrost()
    config.TASK_CONFIG.DATASET.DATA_PATH = str(FOLDER/'dataset.json')
    config.TASK_CONFIG.TASK.MEASUREMENTS = [m for m in config.TASK_CONFIG.TASK.MEASUREMENTS if m != 'TOP_DOWN_MAP']
    config.freeze()
    dataset = load_episodes(config)
    episodes = list(dataset.episodes)
    env = AVWANRLEnv(config, dataset)
    new = 0
    try:
        for i in range(index, len(episodes), 2):
            ep = episodes[i];scene = Path(ep.scene_id).stem;eid = str(ep.episode_id)
            path = FOLDER/'episodes'/scene/(eid+'.pkl.gz')
            if path.exists():
                continue
            rng = np.random.default_rng(20261003+i)
            torch.manual_seed(20261003+i)
            env.habitat_env.current_episode = ep
            env.seed(20261003+i)
            obs = env.reset();policy.reset()
            rows = [];done = False;blocked = None
            while not done:
                mask = env.action_mask(obs)
                predicted = policy.act(obs, mask)
                try:
                    teacher = env.get_expert_map_action()
                except NoReachableWaypointError as error:
                    blocked = str(error)
                    if rows:
                        rows[-1].update(done=True, truncated=True, collector_truncated=True)
                    break
                action = teacher if (rng.random() < .3 or predicted == 40 or teacher == 40) else predicted
                if teacher != 40 and rng.random() < .05:
                    choices = np.flatnonzero(mask);choices = choices[choices != 40]
                    if len(choices):action = int(rng.choice(choices))
                obs, _, done, info = env.step(action)
                transition = env.last_transition
                # Actual action/next-state/reward stay coherent for later offline RL.
                # Only supervised foundation training reads the separate label.
                transition.update(expert_action=int(teacher), learner_action=int(predicted))
                rows.append(transition)
            if not rows:
                raise ValueError(f'No transitions for {scene}/{eid}: {blocked}')
            metadata = dict(scene=scene, episode_id=eid, sound=ep.info['sound'], split='train_multiple',
                            policy_checkpoint=chosen['checkpoint'], policy_sha256=chosen['checkpoint_sha256'],
                            transitions=len(rows), primitive_steps=sum(t['num_steps'] for t in rows),
                            success=bool(info.get('success', False)), blocked=blocked,
                            teacher_disagreements=sum(t['expert_action'] != t['learner_action'] for t in rows),
                            purpose='training_only_recovery_data')
            save_episode(path, rows, metadata)
            print(json.dumps(metadata), flush=True)
            new += 1
            import resource
            if new >= 10 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 >= 6500:
                break
    finally:
        env.close()


def collect():
    configure();selection = prepare()
    common.write(FOLDER/'status.json', dict(status='collecting', episodes=len(selection['episodes'])))
    def run(index):
        expected = selection['episodes'][index::2]
        previous = -1
        while True:
            count = sum((FOLDER/'episodes'/r['scene']/(str(r['episode_id'])+'.pkl.gz')).exists() for r in expected)
            if count == len(expected):return
            if count <= previous:raise RuntimeError('Collector made no progress')
            previous = count
            common.command_run([PYTHON, '-m', 'experiment.avwan.collect_corrections', '--worker', str(index)],
                               FOLDER, f'worker_{index}')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, range(2)))
    common.write(FOLDER/'status.json', dict(status='completed', episodes=len(selection['episodes']), training_only=True))


def supervised_cache():
    """BC labels are expert actions; raw transition files keep behavior actions."""
    import numpy as np
    import pickle
    from .data_pipeline.prepare import read_episode, INPUTS
    cache = OUT/'correction_cache'
    if (cache/'manifest.json').exists():return cache
    assert json.loads((FOLDER/'status.json').read_text())['status'] == 'completed'
    manifest = json.loads((OUT/'domain_cache/manifest.json').read_text())
    cache.mkdir(parents=True, exist_ok=True)
    buffer, shards = [], []
    def flush():
        if not buffer:return
        folder = cache/'train'/f'{len(shards):04d}';folder.mkdir(parents=True, exist_ok=True)
        count = len(buffer)
        for key in ['rgb', 'depth']:
            frames = [r[key] for r in buffer]
            frames += [np.zeros_like(r[key]) if r['prev_'+key] is None else r['prev_'+key] for r in buffer]
            np.save(folder/(key+'.npy'), np.stack(frames))
        np.save(folder/'previous_index.npy', np.arange(count, dtype=np.int64)+count)
        for key in ['spectrogram', 'gm', 'am', 'action', 'action_mask', 'angle_label', 'behavior_action']:
            np.save(folder/(key+'.npy'), np.stack([r[key] for r in buffer]))
        shards.append(dict(path=str(folder), samples=count, purpose='training_expert_labels_at_learner_visited_states'))
        buffer.clear()
    for path in sorted((FOLDER/'episodes').glob('*/*.pkl.gz')):
        rows, _, _ = read_episode(path, INPUTS)
        with gzip.open(path, 'rb') as f:transitions = pickle.load(f)['transitions']
        for row, transition in zip(rows, transitions):
            teacher = transition['expert_action']
            assert row['action_mask'][teacher]
            row['behavior_action'] = row['action']
            row['action'] = np.int64(teacher)
            buffer.append(row)
            if len(buffer) == 256:flush()
    flush()
    # Two copies of shard references implement explicit 2x correction weighting.
    manifest['splits']['train'] += shards + shards
    count = sum(s['samples'] for s in shards)
    manifest['samples']['train'] += 2*count
    manifest['correction_data'] = dict(unique_samples=count, sampling_weight=2,
                                      actual_actions_retained_in_raw_transitions=True,
                                      use='supervised BC only; not a transition cache for TD learning')
    common.write(cache/'manifest.json', manifest)
    return cache


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=int)
    a = p.parse_args()
    if a.worker is None:collect()
    else:worker(a.worker)
