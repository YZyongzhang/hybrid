"""Extract frozen foundation features, then train LSTM discrete SAC-CQL."""
import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'avwan'

from .offline.features import extract, validate_cache

ROOT = Path(__file__).resolve().parent


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    for section, keys in [('features', ['source', 'val_source', 'checkpoint', 'cache']),
                          ('training', ['output'])]:
        for key in keys:
            config[section][key] = str((ROOT / Path(config[section][key]).expanduser()).resolve())
    return config


def validate(config):
    for section, keys in [('features', ['batch_size', 'episode_batch_size', 'workers']),
                          ('training', ['epochs', 'batch_size', 'hidden_dim', 'lstm_layers', 'log_every'])]:
        for key in keys:
            if not isinstance(config[section][key], int) or config[section][key] < 1:
                raise ValueError(f'{section}.{key} must be a positive integer')
    if config['cpu_threads'] < 1 or config['features']['max_episodes'] < 0:
        raise ValueError('Invalid CPU thread count or max_episodes')
    cfg = config['training']
    if not 0 <= cfg['gamma'] <= 1:
        raise ValueError('training.gamma must be between 0 and 1')
    if any(cfg[k] <= 0 for k in ('learning_rate', 'initial_alpha', 'cql_temperature', 'grad_clip')):
        raise ValueError('Learning rate, alpha, temperature and clipping must be positive')
    if cfg['cql_weight'] < 0 or not 0 < cfg['tau'] <= 1 or not 0 <= cfg['entropy_ratio'] <= 1:
        raise ValueError('Invalid SAC-CQL hyperparameters')
    if cfg.get('bc_weight', 0.) < 0:
        raise ValueError('training.bc_weight must be nonnegative')
    if cfg.get('actor_rl_weight', 1.) < 0:
        raise ValueError('training.actor_rl_weight must be nonnegative')
    for key in ('terminal_mask', 'foundation_prior'):
        if key in cfg and not isinstance(cfg[key], bool):
            raise ValueError(f'training.{key} must be boolean')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config/offline_rl.yaml'))
    parser.add_argument('--stage', choices=['all', 'extract', 'train'], default='all')
    for key in ('source', 'val-source', 'checkpoint', 'cache', 'output', 'device', 'resume'):
        parser.add_argument('--' + key)
    for key in ('epochs', 'batch-size', 'extract-batch-size', 'max-episodes'):
        parser.add_argument('--' + key, type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for key in ('source', 'val_source', 'checkpoint', 'cache', 'max_episodes'):
        value = getattr(args, key)
        if value is not None:
            config['features'][key] = value if key == 'max_episodes' else str(Path(value).expanduser().resolve())
    for key in ('output', 'epochs', 'batch_size'):
        value = getattr(args, key)
        if value is not None:
            config['training'][key] = str(Path(value).expanduser().resolve()) if key == 'output' else value
    if args.extract_batch_size is not None:
        config['features']['batch_size'] = args.extract_batch_size
    if args.device:
        config['device'] = args.device
    validate(config)
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if config['device'] == 'auto' else config['device'])
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        device = torch.device('cuda', device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
    if args.stage != 'train':
        manifest = extract(config, device)
    else:
        cache = Path(config['features']['cache'])
        manifest = json.loads((cache / 'manifest.json').read_text())
        validate_cache(cache, manifest)
    if args.stage != 'extract':
        from .offline.trainer import train
        train(config, manifest, device, args.resume)


if __name__ == '__main__':
    main()
