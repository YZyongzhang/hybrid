"""Freeze the saved AVWAN train episodes while collection continues.

The output contains symlinks to immutable, completed episode files.  Its index
is fixed at creation time, so feature extraction sees a stable population.
Run from the repository root with ``python -m experiment.avwan.snapshot_large_train``.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_records(split):
    path = ROOT / 'data/datasets/audionav/mp3d/v1' / split / f'{split}.json.gz'
    if not path.exists():
        path = path.with_suffix('')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return json.load(handle)['episodes']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path('/data/getuanhui/avwan_multiple_greedy_every_1000'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    marker = output / 'snapshot.json'
    if marker.exists():
        print(marker.read_text(), end='')
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Nonempty snapshot directory: {output}')
    index = source / 'episodes.jsonl'
    rows = [json.loads(line) for line in index.open() if line.strip()]
    rows = [row for row in rows if row.get('saved')]
    keys = {(row['scene'], str(row['episode_id'])) for row in rows}
    if len(keys) != len(rows):
        raise ValueError('Duplicate training episode keys')
    for split in ('test_multiple', 'test_multiple_unheard'):
        records = test_records(split)
        scenes = {Path(row['scene_id']).stem for row in records}
        if scenes & {key[0] for key in keys}:
            raise ValueError(f'Training/test scene overlap: {split}')
        test_keys = {(Path(row['scene_id']).stem, str(row['episode_id'])) for row in records}
        if keys & test_keys:
            raise ValueError(f'Training/test episode overlap: {split}')
    paths = []
    for row in rows:
        relative = Path(row['file'])
        if (relative.is_absolute() or '..' in relative.parts or
                relative.parts[:1] != ('episodes',)):
            raise ValueError(f'Unexpected episode path: {relative}')
        actual = source / relative
        if not actual.is_file():
            raise FileNotFoundError(actual)
        paths.append((relative, actual))
    output.mkdir(parents=True)
    for relative, actual in paths:
        link = output / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(actual)
    frozen = sorted((row['scene'], str(row['episode_id']), row['file']) for row in rows)
    digest = hashlib.sha256(json.dumps(frozen, separators=(',', ':')).encode()).hexdigest()
    report = dict(source=str(source), episodes=len(rows), scenes=len({x[0] for x in keys}),
                  sounds=len({row['sound'] for row in rows}), index_sha256=digest,
                  test_scene_overlap=0, test_episode_overlap=0,
                  note='Links freeze file membership; source episode files must remain immutable.')
    marker.write_text(json.dumps(report, indent=2) + '\n')
    print(marker.read_text(), end='')


if __name__ == '__main__':
    main()
