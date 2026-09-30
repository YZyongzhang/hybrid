"""Closed-loop learned-policy evaluation in Habitat/SoundSpaces (no expert actions)."""
import argparse
from collections import defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import random
import resource
import time
import traceback

import numpy as np
import torch

from experiment.avwan.config.default import get_config
from experiment.avwan.data_pipeline.prepare import write_json
from experiment.avwan.model.hybrid import Network

ROOT = Path(__file__).resolve().parents[2]


class LearnedPolicy:
    def __init__(self, model, device):
        self.model, self.device = model.eval(), device
        self.previous = None

    def reset(self):
        self.previous = None

    @torch.inference_mode()
    def act(self, obs, action_mask):
        previous = self.previous
        rgb = np.concatenate((previous['rgb'] if previous else np.zeros_like(obs['rgb']), obs['rgb']), -1)
        depth = np.concatenate((previous['depth'] if previous else np.zeros_like(obs['depth']), obs['depth']), -1)
        values = [obs['spectrogram'], rgb.astype(np.float32) / 255., depth, obs['gm'], obs['am'], action_mask]
        tensors = [torch.as_tensor(np.ascontiguousarray(value), device=self.device).unsqueeze(0) for value in values]
        logits, _ = self.model.forward_joint(*tensors)
        action = int(logits.argmax(-1).item())
        self.previous = {key: np.asarray(obs[key]).copy() for key in ('rgb', 'depth')}
        return action


def load_episodes(config, selection_source=None, max_episodes=0, seed=20260925):
    from soundspaces.datasets.audionav_dataset import AudioNavDataset
    cfg = config.TASK_CONFIG.DATASET
    path = Path(cfg.DATA_PATH.format(version=cfg.VERSION, split=cfg.SPLIT))
    # Some installed splits provide plain JSON instead of a gzip archive.
    if not path.exists() and path.suffix == '.gz' and path.with_suffix('').exists():
        path = path.with_suffix('')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        raw = json.load(handle)
    records = raw['episodes']
    if not records:
        contents = sorted((path.parent / 'content').glob('*.json*'))
        for content in contents:
            if not (content.name.endswith('.json') or content.name.endswith('.json.gz')):
                continue
            opener = gzip.open if content.suffix == '.gz' else open
            with opener(content, 'rt') as handle:
                records.extend(json.load(handle)['episodes'])
    def key(ep):
        return Path(ep['scene_id']).stem, str(ep['episode_id'])
    if selection_source:
        index = Path(selection_source) / 'episodes.jsonl'
        requested = {(str(row['scene']), str(row['episode_id'])) for row in
                     (json.loads(line) for line in index.read_text().splitlines()) if row.get('saved')}
        records = [ep for ep in records if key(ep) in requested]
        if {key(ep) for ep in records} != requested:
            raise ValueError('Some collected validation episodes are absent from simulator split')
    records.sort(key=key)
    if max_episodes:
        random.Random(seed).shuffle(records)
        records = records[:max_episodes]
        records.sort(key=key)
    if not records or len({key(ep) for ep in records}) != len(records):
        raise ValueError('Empty or duplicate evaluation episodes')
    dataset = AudioNavDataset()
    dataset._config = cfg
    # This SoundSpaces reader truncates each call to 100; chunks preserve ALL
    # selected records instead of silently evaluating only the first 100.
    for offset in range(0, len(records), 100):
        dataset.from_json(json.dumps({'episodes': records[offset:offset+100]}), scenes_dir=cfg.SCENES_DIR)
    dataset.episodes.sort(key=lambda ep: (Path(ep.scene_id).stem, str(ep.episode_id)))
    return dataset


def aggregate(rows, metadata):
    scenes = defaultdict(list)
    for row in rows:
        scenes[row['scene']].append(row)
    def metrics(items):
        n = len(items)
        return dict(episodes=n, spl=sum(x['spl'] for x in items)/max(n, 1),
                    success_rate=sum(x['success'] for x in items)/max(n, 1),
                    errors=sum(x.get('error') is not None for x in items),
                    mean_primitive_steps=sum(x['primitive_steps'] for x in items)/max(n, 1))
    return dict(**metadata, **metrics(rows), per_scene={key: metrics(items) for key, items in scenes.items()})


def evaluate(args):
    from experiment import AVWANRLEnv
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint).resolve()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    foundation_path = getattr(args, 'foundation_checkpoint', None)
    if foundation_path:
        from experiment.avwan.offline.policy import OfflinePolicy
        policy = OfflinePolicy.from_checkpoint(state, foundation_path, device)
    else:
        model = Network(action_count=state['signature']['actions']).to(device)
        model.load_state_dict(state['model'])
        policy = LearnedPolicy(model, device)
    config = get_config(args.config, opts=['TASK_CONFIG.DATASET.SPLIT', args.split])
    config.defrost()
    config.TASK_CONFIG.TRAIN.OPEN = False
    config.TASK_CONFIG.COLLECT.OPEN = False
    # Diagnostic top-down images are unnecessary for SPL and expensive each step.
    config.TASK_CONFIG.TASK.MEASUREMENTS = [m for m in config.TASK_CONFIG.TASK.MEASUREMENTS if m != 'TOP_DOWN_MAP']
    config.freeze()
    dataset = load_episodes(config, args.selection_source, args.max_episodes, args.seed)
    # Habitat may shuffle the dataset list during construction. Freeze the planned
    # traversal first so checkpoint comparisons use identical order and seeds.
    planned_episodes = list(dataset.episodes)
    global_indices = {(Path(ep.scene_id).stem, str(ep.episode_id)): i for i, ep in enumerate(planned_episodes)}
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError('Invalid evaluation shard index/count')
    planned_episodes = planned_episodes[args.shard_index::args.shards]
    selection = [(Path(ep.scene_id).stem, str(ep.episode_id)) for ep in planned_episodes]
    metadata = dict(checkpoint=str(checkpoint), epoch=state['epoch'], split=args.split, seed=args.seed,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    planned=len(selection), selection_sha256=hashlib.sha256(json.dumps(selection).encode()).hexdigest(),
                    policy='masked_argmax', max_episode_steps=config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS,
                    prediction_interval=config.PREDICTION_INTERVAL)
    if foundation_path:
        metadata.update(policy='frozen_foundation_lstm_sac_masked_argmax',
                        foundation_checkpoint=str(Path(foundation_path).resolve()),
                        foundation_sha256=state['signature']['features']['checkpoint_sha256'])
    if args.shards > 1:
        metadata.update(shards=args.shards, shard_index=args.shard_index)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / 'run.json'
    if run_path.exists():
        if not args.resume or json.loads(run_path.read_text()) != metadata:
            raise ValueError('Existing evaluation output requires matching --resume')
    elif any(output.iterdir()):
        raise ValueError('Evaluation output must be empty')
    write_json(run_path, metadata)
    write_json(output / 'selection.json', selection)
    (output / 'environment.yaml').write_text(config.dump())
    index = output / 'episodes.jsonl'
    rows = [json.loads(line) for line in index.read_text().splitlines()] if index.exists() else []
    complete = {(r['scene'], r['episode_id']) for r in rows}
    if len(complete) != len(rows) or not complete.issubset(set(selection)):
        raise ValueError('Duplicate or unexpected episodes in evaluation output')
    if complete == set(selection):
        if not (output / 'summary.json').exists():
            write_json(output / 'summary.json', aggregate(rows, metadata))
        print('Already complete: ' + json.dumps(aggregate(rows, metadata)), flush=True)
        return
    env = AVWANRLEnv(config, dataset)
    started = time.monotonic()
    completed_this_process = 0
    try:
        for i, episode in enumerate(planned_episodes):
            scene, eid = Path(episode.scene_id).stem, str(episode.episode_id)
            if (scene, eid) in complete:
                continue
            env.habitat_env.current_episode = episode
            env.seed(args.seed + global_indices[(scene, eid)])
            policy.reset()
            actions, steps = [], 0
            row = dict(scene=scene, episode_id=eid, success=False, spl=0., primitive_steps=0)
            try:
                obs = env.reset()
                info = env.get_info(obs)
                row['initial_distance'] = float(info['distance_to_goal'])
                done = False
                while not done:
                    action = policy.act(obs, env.action_mask(obs))
                    actions.append(action)
                    obs, reward, done, info = env.step(action)
                    steps += int(info['num_steps'])
                row.update(success=bool(info.get('success', False)), spl=float(info['spl']),
                           final_distance=float(info['distance_to_goal']), truncated=bool(info.get('truncated', False)))
            except Exception:
                row['error'] = traceback.format_exc()
            row.update(primitive_steps=steps, actions=actions, waypoint_steps=len(actions))
            rows.append(row)
            with index.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            summary = aggregate(rows, metadata)
            summary['seconds_this_run'] = time.monotonic() - started
            write_json(output / 'summary.json', summary)
            print(f'{len(rows)}/{len(selection)} {scene}/{eid} success={row["success"]} '
                  f'spl={row["spl"]:.4f} mean_SPL={summary["spl"]:.4f} errors={summary["errors"]}', flush=True)
            completed_this_process += 1
            # Release native acoustic-simulator allocations by ending the process
            # only between episodes. The coordinator resumes the saved selection.
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            if ((args.max_new_episodes and completed_this_process >= args.max_new_episodes)
                    or (args.max_rss_mb and rss_mb >= args.max_rss_mb)):
                print(f'Process budget reached: new_episodes={completed_this_process}, peak_RSS_MiB={rss_mb:.0f}', flush=True)
                break
    finally:
        env.close()
    print('RESULT ' + json.dumps(aggregate(rows, metadata)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--foundation-checkpoint', help='Frozen encoder checkpoint for an offline RL SAC policy')
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='val_telephone')
    parser.add_argument('--selection-source', help='Evaluate exactly these collected episode IDs')
    parser.add_argument('--max-episodes', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--max-new-episodes', type=int, default=0,
                        help='End this process after this many new episodes; resume to continue')
    parser.add_argument('--max-rss-mb', type=int, default=0,
                        help='End between episodes when peak process RSS reaches this many MiB')
    parser.add_argument('--profile', help='Optional cProfile output for rollout performance diagnosis')
    args = parser.parse_args()
    import os
    os.chdir(ROOT)
    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        try:
            profiler.runcall(evaluate, args)
        finally:
            profiler.dump_stats(args.profile)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
