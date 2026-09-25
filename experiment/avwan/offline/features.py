"""Lossless episode metadata with frozen float32 foundation embeddings.

Raw collector files and PyTorch checkpoints must be trusted local artifacts.
Each episode stores T+1 observations and T transitions. Completion markers
are published atomically, allowing extraction to resume after interruption.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from ..data_pipeline.prepare import INPUTS, encode_observation, snapshot, scene_name, write_json
from ..model.hybrid import Network
from ..model.maps import unpack_geometry


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_episode(path, action_count):
    with gzip.open(path, 'rb') as handle:
        episode = pickle.load(handle)
    transitions = episode['transitions']
    if episode.get('format_version') != 1 or not transitions:
        raise ValueError(f'{path}: unsupported or empty episode')
    observations, masks = [], []
    for i, t in enumerate(transitions):
        action = int(t['action'])
        for key in ('action_mask', 'next_action_mask'):
            mask = np.asarray(t[key])
            if mask.shape != (action_count,) or not np.isin(mask, [0, 1]).all() or not mask.any():
                raise ValueError(f'{path}: invalid {key} at {i}')
        if action != t['action'] or not 0 <= action < action_count or not t['action_mask'][action]:
            raise ValueError(f'{path}: invalid dataset action at {i}')
        terminated, truncated = bool(t['terminated']), bool(t['truncated'])
        if (terminated and truncated or bool(t['done']) != (terminated or truncated)
                or (t['done'] and i != len(transitions) - 1)):
            raise ValueError(f'{path}: invalid episode boundary at {i}')
        steps, gamma = int(t['num_steps']), float(t['gamma'])
        expected = 0. if terminated else gamma ** steps
        if (steps < 1 or steps != t['num_steps'] or not 0 < gamma <= 1
                or not np.isfinite(t['reward']) or not np.isclose(t['discount'], expected)):
            raise ValueError(f'{path}: invalid SMDP reward/discount at {i}')
        if i:
            previous = transitions[i - 1]
            if (not np.array_equal(previous['next_action_mask'], t['action_mask']) or
                    any(not np.array_equal(previous['next_observation'][key], t['observation'][key])
                        for key in INPUTS)):
                raise ValueError(f'{path}: noncontiguous observations at {i}')
        observations.append(encode_observation(t['observation'], INPUTS)[0])
        masks.append(np.asarray(t['action_mask'], dtype=bool))
    if not transitions[-1]['done']:
        raise ValueError(f'{path}: unfinished episode; explicit terminated/truncated required')
    observations.append(encode_observation(transitions[-1]['next_observation'], INPUTS)[0])
    masks.append(np.asarray(transitions[-1]['next_action_mask'], dtype=bool))
    arrays = {key: np.asarray([t[key] for t in transitions], dtype=dtype)
              for key, dtype in [('action', 'int64'), ('reward', 'float32'), ('discount', 'float32'),
                                 ('terminated', 'bool'), ('truncated', 'bool'), ('num_steps', 'int64'),
                                 ('gamma', 'float32')]}
    arrays['action_mask'] = np.stack(masks)
    return observations, arrays


def load_foundation(checkpoint, device):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights = state.get('model', state)
    actions = weights['mask_encoder.0.weight'].shape[1]
    model = Network(action_count=actions).to(device)
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


@torch.inference_mode()
def embed(model, episodes, batch_size, device):
    """Batch across episodes; RGB/depth history resets exactly at episode start."""
    rows = []
    for observations, arrays in episodes:
        for i, obs in enumerate(observations):
            previous = observations[i - 1] if i else None
            rows.append((obs, previous, arrays['action_mask'][i]))
    features = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        values = {key: torch.from_numpy(np.stack([row[0][key] for row in batch])).to(device)
                  for key in INPUTS}
        for key in ('rgb', 'depth'):
            previous = torch.from_numpy(np.stack([
                np.zeros_like(row[0][key]) if row[1] is None else row[1][key] for row in batch])).to(device)
            values[key] = torch.cat((previous, values[key]), dim=-1).float()
        values['rgb'] /= 255.
        mask = torch.from_numpy(np.stack([row[2] for row in batch])).to(device)
        out = model.embedding_forward(values['spectrogram'], values['rgb'], values['depth'],
                                      unpack_geometry(values['gm']), values['am'], mask)
        if not torch.isfinite(out).all():
            raise FloatingPointError('Nonfinite foundation embedding')
        features.append(out.float().cpu().numpy())
    features = np.concatenate(features)
    offset = 0
    result = []
    for observations, arrays in episodes:
        count = len(observations)
        result.append(dict(arrays, features=features[offset:offset + count]))
        offset += count
    return result


def extract(config, device):
    cfg = config['features']
    cache = Path(cfg['cache'])
    checkpoint = Path(cfg['checkpoint'])
    sources, partitions, records, fingerprints = {}, {}, {}, {}
    for split, key in [('train', 'source'), ('val', 'val_source')]:
        sources[split], partitions[split], records[split], fingerprints[split] = snapshot(
            cfg[key], cfg['max_episodes'])
    keys = [{(scene_name(p), p.name) for p in partitions[split]} for split in ('train', 'val')]
    if keys[0] & keys[1] or set(partitions['train']) & set(partitions['val']):
        raise ValueError('Training and validation episodes overlap')
    identity = dict(version=1, purpose='avwan_frozen_rl_episodes', checkpoint_sha256=sha256(checkpoint),
                    sources={k: str(v) for k, v in sources.items()}, fingerprints=fingerprints,
                    preprocessing='spatial_v3_two_frames_zero_start_fp32', max_episodes=cfg['max_episodes'])
    marker = cache / 'manifest.json'
    pending = cache / 'identity.json'
    if pending.exists() and json.loads(pending.read_text()) != identity:
        raise ValueError('Feature cache checkpoint/source changed; choose a new cache directory')
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if manifest['identity'] != identity:
            raise ValueError('Feature cache identity mismatch')
        validate_cache(cache, manifest)
        print(f'Reusing complete feature cache: {cache}', flush=True)
        return manifest
    if cache.exists() and any(cache.iterdir()) and not pending.exists():
        raise FileExistsError(f'Unrecognized nonempty feature directory: {cache}')
    cache.mkdir(parents=True, exist_ok=True)
    write_json(pending, identity)
    write_json(cache / 'sources.json', records)
    model = load_foundation(checkpoint, device)
    manifest = dict(identity=identity, feature_dim=256, action_count=model.action_count, splits={})
    started = time.monotonic()
    for split, paths in partitions.items():
        folder = cache / split
        folder.mkdir(exist_ok=True)
        entries, work = [], []
        for index, path in enumerate(paths):
            relative = f'{split}/{index:06d}.npz'
            entry = dict(path=relative, source=str(path.relative_to(sources[split])))
            entries.append(entry)
            target = cache / relative
            if target.exists():
                with np.load(target, allow_pickle=False) as data:
                    entry.update(transitions=len(data['action']), bytes=target.stat().st_size)
            else:
                work.append((index, path))
        # Bounded prefetch stores compact observations, not decoded whole datasets.
        with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
            iterator, futures = iter(work), deque()

            def submit():
                item = next(iterator, None)
                if item is not None:
                    index, path = item
                    futures.append((index, path, pool.submit(read_episode, path, model.action_count)))

            for _ in range(cfg['workers']):
                submit()
            completed = len(paths) - len(work)
            while futures:
                group, indices = [], []
                for _ in range(cfg['episode_batch_size']):
                    if not futures:
                        break
                    index, path, future = futures.popleft()
                    group.append(future.result())
                    stat = path.stat()
                    if (stat.st_size, stat.st_mtime_ns) != tuple(records[split][index][1:]):
                        raise RuntimeError(f'Source changed while extracting: {path}')
                    indices.append(index)
                    submit()
                for index, arrays in zip(indices, embed(model, group, cfg['batch_size'], device)):
                    target = cache / entries[index]['path']
                    temporary = target.with_suffix('.npz.tmp')
                    with temporary.open('wb') as handle:
                        np.savez(handle, **arrays)
                    temporary.replace(target)
                    entries[index].update(transitions=len(arrays['action']), bytes=target.stat().st_size)
                completed += len(indices)
                if completed % 100 < len(indices) or completed == len(paths):
                    print(f'Extract {split}: {completed}/{len(paths)} episodes, '
                          f'{time.monotonic()-started:.1f}s', flush=True)
        manifest['splits'][split] = entries
    # Never publish a cache mixing snapshots if collection was still running.
    for split in partitions:
        if snapshot(sources[split], cfg['max_episodes'])[3] != fingerprints[split]:
            raise RuntimeError('Source changed during extraction; use a new cache')
    if sha256(checkpoint) != identity['checkpoint_sha256']:
        raise RuntimeError('Foundation checkpoint changed during extraction')
    manifest['transitions'] = {k: sum(e['transitions'] for e in v) for k, v in manifest['splits'].items()}
    write_json(marker, manifest)
    print(f'Features complete: {manifest["transitions"]}', flush=True)
    return manifest


def validate_cache(cache, manifest):
    if manifest['identity']['purpose'] != 'avwan_frozen_rl_episodes':
        raise ValueError('Not an offline RL feature cache')
    for entries in manifest['splits'].values():
        for entry in entries:
            path = Path(cache) / entry['path']
            if not path.exists() or path.stat().st_size != entry['bytes']:
                raise ValueError(f'Missing/truncated feature file: {path}')
