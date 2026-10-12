"""Align collector sound IDs with the frozen AVWAN foundation train shards."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from .data_pipeline.prepare import write_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build(cache, index, output):
    cache, index, output = map(lambda p: Path(p).resolve(), (cache, index, output))
    if output.exists():
        raise FileExistsError(f'Choose a new sound-label output: {output}')
    manifest = json.loads((cache / 'manifest.json').read_text())
    if manifest['identity']['purpose'] != 'avwan_foundation_spatial':
        raise ValueError('Expected a foundation observation cache')
    sources = json.loads((cache / 'sources.json').read_text())['train']
    paths = [row[0] for row in sources]
    random.Random(manifest['identity']['options']['seed']).shuffle(paths)
    requested = set(paths)
    records = {}
    with index.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('file') in requested:
                key = row['file']
                if key in records:
                    raise ValueError(f'Duplicate indexed episode: {key}')
                if not row.get('saved') or int(row['transitions']) < 1 or not row.get('sound'):
                    raise ValueError(f'Incomplete sound metadata: {key}')
                records[key] = (str(row['sound']), int(row['transitions']))
    if set(records) != requested:
        raise ValueError(f'Missing {len(requested - set(records))} frozen train episodes in index')
    classes = sorted({sound for sound, _ in records.values()})
    lookup = {sound: idx for idx, sound in enumerate(classes)}
    labels = np.concatenate([np.full(records[path][1], lookup[records[path][0]],
                                     dtype=np.int16) for path in paths])
    if len(labels) != manifest['samples']['train']:
        raise ValueError('Episode transition counts differ from frozen train cache')
    output.mkdir(parents=True)
    position, shards = 0, []
    for shard in manifest['splits']['train']:
        count = shard['samples']
        selected = labels[position:position + count]
        source = Path(shard['path'])
        folder = output / source.parent
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / (source.name + '.npy')
        # A sound change inside a shard must coincide with a new episode.
        previous = np.load(cache / shard['path'] / 'previous_index.npy', mmap_mode='r')
        changes = np.flatnonzero(selected[1:] != selected[:-1]) + 1
        if np.any(previous[changes] < count):
            raise ValueError(f'Sound labels misaligned with episode boundaries: {shard["path"]}')
        np.save(destination, selected, allow_pickle=False)
        shards.append(dict(path=str(destination.relative_to(output)), samples=count,
                           sha256=sha256(destination)))
        position += count
    if position != len(labels):
        raise ValueError('Shards do not cover all labels')
    counts = np.bincount(labels, minlength=len(classes)).tolist()
    report = dict(cache=str(cache), cache_fingerprint=manifest['identity']['fingerprint'],
                  sources_sha256=sha256(cache / 'sources.json'), episodes=len(paths),
                  samples=len(labels), classes=classes, class_samples=counts, shards=shards)
    write_json(output / 'manifest.json', report)
    print(json.dumps(dict(episodes=len(paths), samples=len(labels), sounds=len(classes),
                          shards=len(shards)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path(
        'result/2026-10-11-large-foundation-cache'))
    parser.add_argument('--index', type=Path, default=Path(
        '/data/Getuanhui/avwan_multiple_greedy_every_1000/episodes.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.cache, args.index, args.output)


if __name__ == '__main__':
    main()
