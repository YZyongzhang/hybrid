"""Train the AVWAN hybrid model using the TRAIN section in audiogoal.yaml."""
import argparse
from pathlib import Path
import sys

import yaml

# Also support direct execution without importing the simulator-owning experiment package.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'avwan'

from .data_pipeline.prepare import prepare

ROOT = Path(__file__).resolve().parent


def training_config(node):
    """Accept a plain YAML mapping or the resolved Habitat/YACS TRAIN node."""
    config = dict(seed=node['SEED'], data={k.lower(): v for k, v in node['DATA'].items()},
                  training={k.lower(): v for k, v in node.items()
                            if k not in ('OPEN', 'STAGE', 'RESUME', 'SEED', 'DATA',
                                         'MODE', 'OFFLINE_RL')},
                  stage=node['STAGE'], resume=node['RESUME'])
    for section, key in ((config['data'], 'source'), (config['data'], 'cache'), (config['training'], 'output')):
        section[key] = str((ROOT / Path(section[key]).expanduser()).resolve())
    if config['data'].get('val_source'):
        config['data']['val_source'] = str((ROOT / Path(config['data']['val_source']).expanduser()).resolve())
    return config


def load_config(path):
    values = yaml.safe_load(Path(path).read_text())
    return training_config(values['TRAIN'])


def validate(config):
    data, train = config['data'], config['training']
    for section, keys in [(data, ['shard_size', 'convert_workers']),
                          (train, ['epochs', 'batch_size', 'prefetch_factor', 'cpu_threads', 'log_every'])]:
        for key in keys:
            if not isinstance(section[key], int) or section[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
    for value in (data['max_episodes'], train['workers'], train['max_batches']):
        if not isinstance(value, int) or value < 0:
            raise ValueError('max_episodes, workers, max_batches must be nonnegative integers')
    if data['split_by'] not in ('scene', 'episode') or not 0 < data['val_fraction'] < 1:
        raise ValueError('split_by must be scene/episode and val_fraction must be in (0,1)')
    if any(train[key] <= 0 for key in ('learning_rate', 'action_loss_weight', 'angle_loss_weight', 'grad_clip')):
        raise ValueError('Learning rate and both supervision weights must be positive')
    if train['weight_decay'] < 0:
        raise ValueError('weight_decay must be nonnegative')


def run_training(config, stage=None, resume=None):
    stage = stage or config['stage']
    resume = config['resume'] if resume is None else resume
    if stage not in ('all', 'prepare', 'train'):
        raise ValueError('TRAIN.FOUNDATION.STAGE must be all, prepare, or train')
    validate(config)
    if stage == 'train' and not (Path(config['data']['cache']) / 'manifest.json').exists():
        raise FileNotFoundError('No completed cache; set TRAIN.FOUNDATION.STAGE to all or prepare first')
    manifest = prepare(config)
    if stage != 'prepare':
        from .training.trainer import train
        checkpoint = str((ROOT / Path(resume).expanduser()).resolve()) if resume else None
        train(config, manifest, checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config/audiogoal.yaml'), help='Task YAML containing TRAIN')
    parser.add_argument('--stage', choices=['all', 'prepare', 'train'])
    parser.add_argument('--source')
    parser.add_argument('--cache')
    parser.add_argument('--output')
    parser.add_argument('--device')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--max-episodes', type=int)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--resume', help='Checkpoint path relative to experiment/avwan; epochs is TOTAL target')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for key in ('source', 'cache', 'max_episodes'):
        if getattr(args, key) is not None:
            config['data'][key] = getattr(args, key)
    for key in ('output', 'device', 'epochs', 'batch_size', 'workers', 'max_batches'):
        if getattr(args, key) is not None:
            config['training'][key] = getattr(args, key)
    for section, key in ((config['data'], 'source'), (config['data'], 'cache'), (config['training'], 'output')):
        section[key] = str((ROOT / Path(section[key]).expanduser()).resolve())
    run_training(config, args.stage, args.resume)


if __name__ == '__main__':
    main()
