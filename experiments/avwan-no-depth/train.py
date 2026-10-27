"""Train the AVWAN foundation architecture without the depth-image input."""
import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from data_pipeline.audit_splits import audit_splits
from data_pipeline.prepare import prepare


def training_config(node):
    data = {key.lower(): value for key, value in node['DATA'].items()}
    integrity = {key.lower(): value for key, value in node.get('INTEGRITY', {}).items()}
    integrity.setdefault('train_index', str(Path(data['source']) / 'episodes.jsonl'))
    if data.get('val_source'):
        integrity.setdefault('val_index', str(Path(data['val_source']) / 'episodes.jsonl'))
    integrity.setdefault('test_manifest_root', str(PROJECT_ROOT / 'data/datasets/audionav/mp3d/v1'))
    integrity.setdefault('test_splits', ['test_multiple', 'test_multiple_unheard'])
    config = dict(seed=node['SEED'],
                  data=data,
                  training={key.lower(): value for key, value in node.items()
                            if key not in ('OPEN', 'STAGE', 'RESUME', 'SEED', 'DATA',
                                           'INTEGRITY')},
                  integrity=integrity,
                  stage=node['STAGE'], resume=node['RESUME'])
    for section, key in ((config['data'], 'source'), (config['data'], 'cache'),
                         (config['training'], 'output')):
        section[key] = str((ROOT / Path(section[key]).expanduser()).resolve())
    if config['data'].get('val_source'):
        config['data']['val_source'] = str(
            (ROOT / Path(config['data']['val_source']).expanduser()).resolve())
    for key in ('train_index', 'val_index'):
        if config['integrity'].get(key):
            config['integrity'][key] = str(
                (PROJECT_ROOT / Path(config['integrity'][key]).expanduser()).resolve())
    config['integrity']['test_manifest_root'] = str(
        (PROJECT_ROOT / Path(config['integrity']['test_manifest_root']).expanduser()).resolve())
    return config


def load_config(path):
    return training_config(yaml.safe_load(Path(path).read_text())['TRAIN'])


def validate(config):
    data, train = config['data'], config['training']
    if data['inputs'] != ['rgb', 'spectrogram', 'gm', 'am']:
        raise ValueError('No-depth inputs must be [rgb, spectrogram, gm, am]')
    for section, keys in ((data, ('shard_size', 'convert_workers')),
                          (train, ('epochs', 'batch_size', 'prefetch_factor',
                                   'cpu_threads', 'log_every'))):
        for key in keys:
            if not isinstance(section[key], int) or section[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
    for value in (data['max_episodes'], train['workers'], train['max_batches']):
        if not isinstance(value, int) or value < 0:
            raise ValueError('max_episodes, workers, max_batches must be nonnegative integers')
    if data['split_by'] not in ('scene', 'episode') or not 0 < data['val_fraction'] < 1:
        raise ValueError('split_by must be scene/episode and val_fraction must be in (0,1)')
    if any(train[key] <= 0 for key in
           ('learning_rate', 'action_loss_weight', 'angle_loss_weight', 'grad_clip')):
        raise ValueError('Learning rate and both supervision weights must be positive')
    if train['weight_decay'] < 0:
        raise ValueError('weight_decay must be nonnegative')


def run_training(config, stage=None, resume=None):
    stage = stage or config['stage']
    resume = config['resume'] if resume is None else resume
    if stage not in ('all', 'prepare', 'train'):
        raise ValueError('TRAIN.STAGE must be all, prepare or train')
    validate(config)
    if stage == 'train' and not (Path(config['data']['cache']) / 'manifest.json').exists():
        raise FileNotFoundError('No completed no-depth cache; run with --stage prepare first')
    integrity_audit = audit_splits(config)
    manifest = prepare(config)
    manifest['integrity_audit'] = integrity_audit
    if stage != 'prepare':
        from training.trainer import train
        checkpoint = str((ROOT / Path(resume).expanduser()).resolve()) if resume else None
        train(config, manifest, checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config/foundation.yaml'))
    parser.add_argument('--stage', choices=['all', 'prepare', 'train'])
    parser.add_argument('--source')
    parser.add_argument('--val-source')
    parser.add_argument('--train-index')
    parser.add_argument('--val-index')
    parser.add_argument('--test-manifest-root')
    parser.add_argument('--cache')
    parser.add_argument('--output')
    parser.add_argument('--device')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--max-episodes', type=int)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--resume')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for key in ('source', 'val_source', 'cache', 'max_episodes'):
        value = getattr(args, key)
        if value is not None:
            config['data'][key] = value
    if args.source and not args.train_index:
        config['integrity']['train_index'] = str(Path(args.source).resolve() / 'episodes.jsonl')
    if args.val_source and not args.val_index:
        config['integrity']['val_index'] = str(Path(args.val_source).resolve() / 'episodes.jsonl')
    for key in ('train_index', 'val_index', 'test_manifest_root'):
        value = getattr(args, key)
        if value is not None:
            config['integrity'][key] = value
    for key in ('output', 'device', 'epochs', 'batch_size', 'workers', 'max_batches'):
        value = getattr(args, key)
        if value is not None:
            config['training'][key] = value
    for section, key in ((config['data'], 'source'), (config['data'], 'val_source'),
                         (config['data'], 'cache'), (config['training'], 'output')):
        if section.get(key):
            section[key] = str((ROOT / Path(section[key]).expanduser()).resolve())
    for key in ('train_index', 'val_index', 'test_manifest_root'):
        config['integrity'][key] = str(
            (PROJECT_ROOT / Path(config['integrity'][key]).expanduser()).resolve())
    run_training(config, args.stage, args.resume)


if __name__ == '__main__':
    main()
