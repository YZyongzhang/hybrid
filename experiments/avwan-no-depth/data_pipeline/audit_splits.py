"""Audit episode provenance and split overlap using manifests only."""
import gzip
import hashlib
import json
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _scene_name(value):
    return Path(str(value)).stem


def _read_index(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing split episode index: {path}')
    keys, scenes = set(), set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get('saved', True):
                continue
            scene, episode_id = row.get('scene'), row.get('episode_id')
            if scene is None or episode_id is None:
                raise ValueError(f'{path}:{line_number}: missing scene or episode_id')
            key = (_scene_name(scene), str(episode_id))
            if key in keys:
                raise ValueError(f'{path}:{line_number}: duplicate episode key {key}')
            keys.add(key)
            scenes.add(key[0])
    if not keys:
        raise ValueError(f'No saved episodes in split index: {path}')
    return dict(path=str(path.resolve()), sha256=_sha256(path),
                episodes=len(keys), scenes=len(scenes), keys=keys, scene_ids=scenes)


def _read_test_manifest(root, split):
    root = Path(root)
    path = root / split / f'{split}.json.gz'
    compressed = True
    if not path.is_file():
        path = root / split / f'{split}.json'
        compressed = False
    if not path.is_file():
        raise FileNotFoundError(f'Missing test split manifest: {path}')
    opener = gzip.open if compressed else open
    with opener(path, 'rt') as handle:
        rows = json.load(handle)['episodes']
    keys, scenes = set(), set()
    for row in rows:
        # The audit reads only scene and episode IDs from test metadata.
        key = (_scene_name(row['scene_id']), str(row['episode_id']))
        if key in keys:
            raise ValueError(f'{path}: duplicate test episode key {key}')
        keys.add(key)
        scenes.add(key[0])
    if not keys:
        raise ValueError(f'No episodes in test split manifest: {path}')
    return dict(path=str(path.resolve()), sha256=_sha256(path),
                episodes=len(keys), scenes=len(scenes), keys=keys, scene_ids=scenes)


def _keys_hash(keys):
    text = '\n'.join(f'{scene}\t{episode}' for scene, episode in sorted(keys))
    return hashlib.sha256(text.encode()).hexdigest()


def audit_splits(config):
    """Fail closed on train/validation/test ID leakage before data conversion."""
    integrity = config['integrity']
    train = _read_index(integrity['train_index'])
    validation = _read_index(integrity['val_index'])
    if train['keys'] & validation['keys']:
        raise ValueError('Training and validation episode IDs overlap')

    train_test = {}
    validation_test = {}
    for split in integrity['test_splits']:
        test = _read_test_manifest(integrity['test_manifest_root'], split)
        train_episode_overlap = train['keys'] & test['keys']
        train_scene_overlap = train['scene_ids'] & test['scene_ids']
        val_episode_overlap = validation['keys'] & test['keys']
        if train_episode_overlap or train_scene_overlap:
            raise ValueError(
                f'Training/test overlap in {split}: '
                f'{len(train_episode_overlap)} episodes, {len(train_scene_overlap)} scenes')
        if val_episode_overlap:
            raise ValueError(
                f'Validation/test episode overlap in {split}: {len(val_episode_overlap)} episodes')
        train_test[split] = dict(test_manifest=test['path'],
                                 test_manifest_sha256=test['sha256'],
                                 test_episodes=test['episodes'], test_scenes=test['scenes'],
                                 test_episode_ids_sha256=_keys_hash(test['keys']),
                                 training_episode_overlap=0, training_scene_overlap=0,
                                 validation_episode_overlap=0,
                                 validation_scene_overlap=len(validation['scene_ids'] & test['scene_ids']))

    return dict(
        policy='Manifest-only overlap audit; no test observations, actions, rewards, labels, '
               'trajectories or metrics are loaded.',
        training=dict(manifest=train['path'], manifest_sha256=train['sha256'],
                      episodes=train['episodes'], scenes=train['scenes'],
                      episode_ids_sha256=_keys_hash(train['keys'])),
        validation=dict(manifest=validation['path'], manifest_sha256=validation['sha256'],
                        episodes=validation['episodes'], scenes=validation['scenes'],
                        episode_ids_sha256=_keys_hash(validation['keys']),
                        training_episode_overlap=0),
        test_splits=train_test,
    )
