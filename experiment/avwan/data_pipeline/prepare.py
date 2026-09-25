"""Convert trusted collector pickles once into lossless, mmap-friendly shards."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np

VERSION = 3
PURPOSE = 'avwan_foundation_spatial'
INPUTS = ('rgb', 'depth', 'spectrogram', 'gm', 'am')
SHAPES = {'rgb': (128, 128, 3), 'depth': (128, 128, 1), 'spectrogram': (65, 26, 2),
          'gm': (400, 400, 2), 'am': (20, 20, 1)}


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def snapshot(source, max_episodes=0):
    source = Path(source).resolve()
    files = sorted((source / 'episodes').rglob('*.pkl.gz'))
    if max_episodes:
        # Round-robin scenes so a small diagnostic subset can still be split.
        groups = {}
        for path in files:
            groups.setdefault(scene_name(path), deque()).append(path)
        files = []
        while len(files) < max_episodes and any(groups.values()):
            for queue in groups.values():
                if queue and len(files) < max_episodes:
                    files.append(queue.popleft())
    if not files:
        raise FileNotFoundError(f'No episodes/**/*.pkl.gz in {source}')
    records = [(str(p.relative_to(source)), p.stat().st_size, p.stat().st_mtime_ns)
               for p in files]
    digest = hashlib.sha256(json.dumps(records).encode()).hexdigest()
    return source, files, records, digest


def scene_name(path):
    return (path.parent.name if path.parent.name != 'episodes'
            else path.name.removesuffix('.pkl.gz').rsplit('_', 1)[0])


def split_files(files, fraction, seed, split_by):
    key = scene_name if split_by == 'scene' else lambda p: str(p)
    groups = sorted({key(p) for p in files})
    if len(groups) < 2:
        raise ValueError(f'Need at least two {split_by} groups for train/validation')
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    validation = set(groups[:count])
    result = {'train': [], 'val': []}
    for path in files:
        result['val' if key(path) in validation else 'train'].append(path)
    for paths in result.values():
        random.Random(seed).shuffle(paths)
    return result, sorted(validation)


def angle_sector(radians):
    """Same [-180,180) sector convention as AVNav's angle_to_sector_label."""
    return int(np.floor(((np.degrees(float(radians)) + 180.) % 360.) / 45.))


def encode_observation(obs, inputs):
    encoded, layout = {}, {}
    for key in inputs:
        value = np.asarray(obs[key])
        dtype = np.dtype('uint8' if key == 'rgb' else 'float32')
        if value.shape != SHAPES[key] or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f'{key}: expected finite {SHAPES[key]} {dtype}, got {value.shape} {value.dtype}')
        if key == 'gm':
            if not np.all((value == 0) | (value == 1)):
                raise ValueError('gm must be binary for lossless bit packing')
            encoded[key] = np.packbits(value.reshape(-1).astype(np.uint8))
        else:
            encoded[key] = value.copy()
        layout[key] = dict(shape=list(value.shape), storage='packbits' if key == 'gm' else str(dtype))
    return encoded, layout


def read_episode(path, inputs):
    with gzip.open(path, 'rb') as handle:
        episode = pickle.load(handle)
    if episode.get('format_version') != 1 or not episode['transitions']:
        raise ValueError(f'{path}: unsupported or empty episode')
    rows, previous, actions = [], None, None
    for t in episode['transitions']:
        mask = np.asarray(t['action_mask'])
        action = int(t['action'])
        if (mask.ndim != 1 or not np.isin(mask, [0, 1]).all() or
                action != t['action'] or not 0 <= action < len(mask) or not mask[action] or
                (actions is not None and len(mask) != actions)):
            raise ValueError(f'{path}: invalid waypoint action/mask')
        actions = len(mask)
        arrays, schema = encode_observation(t['observation'], inputs)
        angle = np.asarray(t['observation']['angle']).reshape(-1)
        if angle.size != 2 or not np.isfinite(angle).all():
            raise ValueError(f'{path}: angle must contain finite [distance, azimuth_radians]')
        row = dict(**arrays, action=np.int64(action), action_mask=mask.astype(bool),
                   angle_label=np.int64(angle_sector(angle[1])), episode_start=previous is None,
                   prev_rgb=None if previous is None else previous['rgb'],
                   prev_depth=None if previous is None else previous['depth'])
        rows.append(row)
        previous = arrays
    return rows, schema, actions


def bounded_episodes(paths, inputs, workers):
    # Bounded futures avoid accumulating the entire decompressed dataset in RAM.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator, pending = iter(paths), deque()
        for _ in range(workers):
            path = next(iterator, None)
            if path is not None:
                pending.append((path, pool.submit(read_episode, path, inputs)))
        while pending:
            path, future = pending.popleft()
            result = future.result()
            yield path, result
            following = next(iterator, None)
            if following is not None:
                pending.append((following, pool.submit(read_episode, following, inputs)))


def prepare(config):
    """A completed cache is immutable; reject changed source files or options."""
    cfg = config['data']
    inputs = cfg['inputs']
    if list(inputs) != list(INPUTS):
        raise ValueError(f'Foundation inputs must be {list(INPUTS)}')
    source, files, records, fingerprint = snapshot(cfg['source'], cfg['max_episodes'])
    options = {key: cfg[key] for key in ('inputs', 'shard_size', 'val_fraction', 'split_by', 'max_episodes')}
    options['seed'] = config['seed']
    identity = dict(version=VERSION, purpose=PURPOSE, source=str(source), fingerprint=fingerprint, options=options)
    cache = Path(cfg['cache'])
    manifest_path = cache / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity:
            raise ValueError('Cache/source/config changed; choose a new data.cache directory')
        for split in manifest['splits'].values():
            for shard in split:
                for name, size in shard['files'].items():
                    path = cache / shard['path'] / name
                    if not path.exists() or path.stat().st_size != size:
                        raise ValueError(f'Missing/truncated cache file: {path}')
        print(f'Reusing cache: {cache}', flush=True)
        return manifest
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError(f'Incomplete/nonempty cache: {cache}; choose a new data.cache directory')
    cache.mkdir(parents=True, exist_ok=True)
    partitions, val_groups = split_files(files, cfg['val_fraction'], config['seed'], cfg['split_by'])
    manifest = dict(identity=identity, splits={}, observation_schema=None, action_count=None,
                    validation_groups=val_groups, episodes=len(files))
    write_json(cache / 'sources.json', records)
    started, processed = time.monotonic(), 0
    stats = {str(p): (size, mtime) for p, (_, size, mtime) in zip(files, records)}
    for split, paths in partitions.items():
        shards, buffer = [], []

        def flush():
            if not buffer:
                return
            relative = f'{split}/{len(shards):05d}'
            folder = cache / relative
            folder.mkdir(parents=True)
            # Current frames are stored once. Repeated history is represented by
            # local indices; only a cross-shard previous frame needs an extra copy.
            count = len(buffer)
            previous_index = np.arange(count, dtype=np.int64) - 1
            for index, row in enumerate(buffer):
                if row['episode_start']:
                    previous_index[index] = count  # Shared zero frame.
            carry = not buffer[0]['episode_start']
            if carry:
                previous_index[0] = count + 1
            for key in ('rgb', 'depth'):
                frames = [row[key] for row in buffer] + [np.zeros_like(buffer[0][key])]
                if carry:
                    frames.append(buffer[0][f'prev_{key}'])
                np.save(folder / f'{key}.npy', np.stack(frames), allow_pickle=False)
            np.save(folder / 'previous_index.npy', previous_index, allow_pickle=False)
            for key in ('spectrogram', 'gm', 'am', 'action', 'action_mask', 'angle_label'):
                np.save(folder / f'{key}.npy', np.stack([row[key] for row in buffer]), allow_pickle=False)
            shards.append(dict(path=relative, samples=len(buffer),
                               files={p.name: p.stat().st_size for p in folder.glob('*.npy')}))
            buffer.clear()

        for path, (rows, schema, actions) in bounded_episodes(paths, inputs, cfg['convert_workers']):
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != stats[str(path)]:
                raise RuntimeError(f'Source changed during conversion: {path}')
            if manifest['observation_schema'] is None:
                manifest.update(observation_schema=schema, action_count=actions)
            if schema != manifest['observation_schema'] or actions != manifest['action_count']:
                raise ValueError(f'{path}: inconsistent observation schema/action space')
            # Split episodes across shards only AFTER assigning an entire scene/episode to a split.
            for row in rows:
                buffer.append(row)
                if len(buffer) == cfg['shard_size']:
                    flush()
            processed += 1
            if processed % 100 == 0 or processed == len(files):
                print(f'Convert {processed}/{len(files)} episodes, {time.monotonic()-started:.1f}s', flush=True)
        flush()
        manifest['splits'][split] = shards
    manifest['samples'] = {key: sum(s['samples'] for s in shards)
                           for key, shards in manifest['splits'].items()}
    manifest['bytes'] = sum(sum(s['files'].values()) for shards in manifest['splits'].values() for s in shards)
    write_json(manifest_path, manifest)  # Completion marker published last.
    print(f'Cache ready: {manifest["samples"]}, {manifest["bytes"]/2**30:.2f} GiB', flush=True)
    return manifest
