"""Run a set or grid of offline RL training configurations in sequence."""
import argparse
from datetime import datetime, timezone
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
TRAINING_KEYS = {
    'epochs', 'batch_size', 'hidden_dim', 'lstm_layers', 'learning_rate', 'gamma',
    'initial_alpha', 'entropy_ratio', 'tau', 'cql_weight', 'cql_temperature',
    'grad_clip', 'log_every', 'tensorboard', 'terminal_mask', 'bc_weight',
    'actor_rl_weight', 'foundation_prior',
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def resolve_from(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def safe_name(value):
    if (not isinstance(value, str) or value in ('.', '..') or len(value) > 180 or
            not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value)):
        raise ValueError(f'Invalid experiment name {value!r}; use letters, digits, dot, dash or underscore')
    return value


def value_slug(value):
    if isinstance(value, float):
        value = f'{value:g}'
    text = str(value).replace('-', 'm').replace('.', 'p')
    return re.sub(r'[^A-Za-z0-9_]+', '', text)[:16] or 'value'


def make_experiments(spec):
    explicit = spec.get('experiments')
    grid = spec.get('grid')
    if bool(explicit) == bool(grid):
        raise ValueError('Provide exactly one nonempty `experiments` list or `grid` mapping')
    result = []
    if explicit:
        if not isinstance(explicit, list):
            raise ValueError('`experiments` must be a list')
        for row in explicit:
            if not isinstance(row, dict) or not isinstance(row.get('training', {}), dict):
                raise ValueError('Each experiment must be a mapping with optional `training` overrides')
            unknown = set(row) - {'name', 'training', 'resume'}
            if unknown:
                raise ValueError(f'Unknown experiment fields: {sorted(unknown)}')
            name = safe_name(row.get('name'))
            training = row.get('training', {})
            unknown_params = set(training) - TRAINING_KEYS
            if unknown_params:
                raise ValueError(f'{name}: unknown training parameters: {sorted(unknown_params)}')
            result.append(dict(name=name, training=training, resume=row.get('resume')))
    else:
        if not isinstance(grid, dict) or not grid:
            raise ValueError('`grid` must be a nonempty mapping of training.parameter to value lists')
        keys = list(grid)
        for key in keys:
            if not isinstance(key, str) or not key.startswith('training.'):
                raise ValueError(f'Grid key {key!r} must start with `training.`')
            param = key.split('.', 1)[1]
            if param not in TRAINING_KEYS:
                raise ValueError(f'Unknown training parameter in grid: {param}')
            if not isinstance(grid[key], list) or not grid[key]:
                raise ValueError(f'Grid values for {key} must be a nonempty list')
            if any(not isinstance(value, (str, int, float, bool)) or
                   isinstance(value, float) and not math.isfinite(value) for value in grid[key]):
                raise ValueError(f'Grid values for {key} must be finite YAML scalars')
        varying = [key for key in keys if len(grid[key]) > 1]
        for values in product(*(grid[key] for key in keys)):
            overrides = dict(zip((key.split('.', 1)[1] for key in keys), values))
            labels = [f'{key.split(".", 1)[1]}-{value_slug(value)}'
                      for key, value in zip(keys, values) if key in varying]
            name = '__'.join(labels) if labels else 'default'
            result.append(dict(name=safe_name(name),
                               training=overrides, resume=None))
    names = [row['name'] for row in result]
    if len(names) != len(set(names)):
        raise ValueError('Experiment names must be unique')
    return result


def prepare_config(base, run, run_dir):
    config = json.loads(json.dumps(base))
    config['training'].update(run['training'])
    config['training']['output'] = str((run_dir / 'training').resolve())
    config['tensorboard_dir'] = str((run_dir.parent / 'tensorboard' / run_dir.name).resolve())
    return config


def can_extend_run(previous, current, run_dir):
    """Allow changing the target epoch count when a resumable checkpoint exists."""
    if (not isinstance(previous, dict) or not isinstance(previous.get('training'), dict)
            or not isinstance(current.get('training'), dict)):
        return False
    old = json.loads(json.dumps(previous))
    new = json.loads(json.dumps(current))
    old_training, new_training = old.pop('training'), new.pop('training')
    if old != new or old_training.get('output') != new_training.get('output'):
        return False
    old_epochs = old_training.pop('epochs', None)
    new_epochs = new_training.pop('epochs', None)
    return (old_epochs != new_epochs and old_training == new_training
            and (run_dir / 'training/latest.pt').is_file())


def validate_config(config):
    features, training = config['features'], config['training']
    for key in ('source', 'checkpoint', 'cache'):
        if key not in features or not isinstance(features[key], str):
            raise ValueError(f'features.{key} must be a path')
    for section, keys in (('features', ('batch_size', 'episode_batch_size', 'workers')),
                          ('training', ('epochs', 'batch_size', 'hidden_dim', 'lstm_layers', 'log_every'))):
        values = config[section]
        for key in keys:
            if type(values.get(key)) is not int or values[key] < 1:
                raise ValueError(f'{section}.{key} must be a positive integer')
    if type(features.get('max_episodes')) is not int or features['max_episodes'] < 0:
        raise ValueError('features.max_episodes must be a nonnegative integer')
    if type(config.get('cpu_threads')) is not int or config['cpu_threads'] < 1:
        raise ValueError('cpu_threads must be a positive integer')
    if type(config.get('seed')) is not int or not isinstance(config.get('device'), str):
        raise ValueError('seed must be an integer and device must be a string')
    for key in ('gamma', 'learning_rate', 'initial_alpha', 'entropy_ratio', 'tau',
                'cql_weight', 'cql_temperature', 'grad_clip'):
        if key not in training or type(training[key]) not in (int, float) or not math.isfinite(training[key]):
            raise ValueError(f'training.{key} must be a finite number')
    if not 0 <= training['gamma'] <= 1:
        raise ValueError('training.gamma must be between 0 and 1')
    if any(training[key] <= 0 for key in ('learning_rate', 'initial_alpha', 'cql_temperature', 'grad_clip')):
        raise ValueError('Learning rate, alpha, temperature and clipping must be positive')
    if (training['cql_weight'] < 0 or not 0 < training['tau'] <= 1
            or not 0 <= training['entropy_ratio'] <= 1):
        raise ValueError('Invalid SAC-CQL hyperparameters')
    for key, default in (('bc_weight', 0.), ('actor_rl_weight', 1.)):
        value = training.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'training.{key} must be a nonnegative finite number')
    for key in ('terminal_mask', 'foundation_prior', 'tensorboard'):
        if key in training and not isinstance(training[key], bool):
            raise ValueError(f'training.{key} must be boolean')


def run_command(command, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    label = log_path.parent.name
    print(f'[{label}] starting', flush=True)
    with log_path.open('a') as log:
        log.write(f'\n[{timestamp()}] $ {json.dumps(command, ensure_ascii=False)}\n')
        log.flush()
        process = subprocess.Popen(command, cwd=cwd, env=dict(os.environ, PYTHONUNBUFFERED='1'),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            stripped = line.strip()
            if stripped.startswith('{'):
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    record = None
                if isinstance(record, dict) and 'epoch' in record and 'train' in record:
                    metrics = record['train']
                    print(f'[{label}] epoch={record["epoch"]} '
                          f'td={metrics["td_loss"]:.4f} cql={metrics["cql_loss"]:.4f} '
                          f'alpha={metrics["alpha"]:.4f}', flush=True)
                    continue
            if (stripped.startswith(('Extract ', 'Features complete:', 'Reusing complete feature cache:',
                                     'Offline RL training complete:'))
                    or 'Traceback' in stripped or 'Error:' in stripped or 'Error ' in stripped):
                print(line, end='', flush=True)
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config/offline_rl_batch.example.yaml'),
                        help='Batch YAML file (default: config/offline_rl_batch.example.yaml)')
    parser.add_argument('--dry-run', action='store_true', help='Print the expanded runs without writing files')
    parser.add_argument('--skip-extract', action='store_true', help='Use the existing feature cache')
    args = parser.parse_args(argv)

    batch_path = Path(args.config).expanduser().resolve()
    spec = yaml.safe_load(batch_path.read_text()) or {}
    if not isinstance(spec, dict):
        raise ValueError('Batch config must be a YAML mapping')
    unknown = set(spec) - {
        'base_config', 'output_root', 'extract', 'experiments', 'grid',
        'seed', 'device', 'cpu_threads', 'features', 'training',
    }
    if unknown:
        raise ValueError(f'Unknown batch config fields: {sorted(unknown)}')
    if 'extract' in spec and not isinstance(spec['extract'], bool):
        raise ValueError('`extract` must be true or false')
    base_path = resolve_from(batch_path.parent, spec.get('base_config', 'config/offline_rl.yaml'))
    base = yaml.safe_load(base_path.read_text())
    if not isinstance(base, dict) or not isinstance(base.get('features'), dict) or not isinstance(base.get('training'), dict):
        raise ValueError(f'Base config must contain `features` and `training` mappings: {base_path}')
    for key in ('seed', 'device', 'cpu_threads'):
        if key in spec:
            base[key] = spec[key]
    if 'features' in spec:
        if not isinstance(spec['features'], dict):
            raise ValueError('`features` must be a mapping')
        unknown_features = set(spec['features']) - {
            'source', 'checkpoint', 'cache', 'batch_size', 'episode_batch_size', 'workers', 'max_episodes',
        }
        if unknown_features:
            raise ValueError(f'Unknown feature settings: {sorted(unknown_features)}')
        base['features'].update(spec['features'])
    if 'training' in spec:
        if not isinstance(spec['training'], dict):
            raise ValueError('`training` must be a mapping')
        unknown_training = set(spec['training']) - TRAINING_KEYS
        if unknown_training:
            raise ValueError(f'Unknown batch training settings: {sorted(unknown_training)}')
        base['training'].update(spec['training'])
    # Match train_offline_rl.py: paths inside an offline RL config are relative to experiment/avwan.
    for key in ('source', 'checkpoint', 'cache'):
        base['features'][key] = str(resolve_from(ROOT, base['features'][key]))
    runs = make_experiments(spec)
    output_root = resolve_from(ROOT, spec.get('output_root', 'runs/offline_rl_batch'))
    should_extract = bool(spec.get('extract', True)) and not args.skip_extract

    plans = []
    for run in runs:
        run_dir = output_root / run['name']
        config = prepare_config(base, run, run_dir)
        validate_config(config)
        resume = resolve_from(REPO_ROOT, run['resume']) if run.get('resume') else None
        plans.append(dict(run=run, run_dir=run_dir, config=config, resume=resume))
    if args.dry_run:
        print(yaml.safe_dump(dict(base_config=str(base_path), output_root=str(output_root),
                                  extract=should_extract,
                                  runs=[dict(name=p['run']['name'], training=p['run']['training'],
                                             output=p['config']['training']['output'],
                                             resume=str(p['resume']) if p['resume'] else None)
                                         for p in plans]), sort_keys=False, allow_unicode=True))
        return

    # Check every existing config before creating or updating any run outputs.
    for plan in plans:
        config_path = plan['run_dir'] / 'config.yaml'
        if config_path.exists():
            previous = yaml.safe_load(config_path.read_text())
            if (previous != plan['config'] and
                    not can_extend_run(previous, plan['config'], plan['run_dir'])):
                raise FileExistsError(f'Existing experiment config differs: {config_path}; use a new run name')
    output_root.mkdir(parents=True, exist_ok=True)
    for plan in plans:
        plan['run_dir'].mkdir(parents=True, exist_ok=True)
        config_path = plan['run_dir'] / 'config.yaml'
        previous = yaml.safe_load(config_path.read_text()) if config_path.exists() else None
        if previous != plan['config']:
            config_path.write_text(yaml.safe_dump(plan['config'], sort_keys=False, allow_unicode=True))

    statuses = []

    def save_batch_status(status, **extra):
        write_json(output_root / 'batch_status.json', dict(
            status=status, config=str(batch_path), feature_cache=base['features']['cache'],
            runs=statuses + [dict(name=p['run']['name'], status='pending')
                              for p in plans[len(statuses):]], **extra))

    save_batch_status('running', started_at=timestamp())
    try:
        if should_extract:
            command = [sys.executable, '-m', 'experiment.avwan.train_offline_rl',
                       '--config', str(plans[0]['run_dir'] / 'config.yaml'), '--stage', 'extract']
            run_command(command, REPO_ROOT, output_root / 'extract.log')

        for plan in plans:
            name, run_dir = plan['run']['name'], plan['run_dir']
            training_dir = Path(plan['config']['training']['output'])
            completed = training_dir / 'completed.json'
            saved_config = training_dir / 'config.json'
            if (not plan['resume'] and completed.exists() and saved_config.exists()
                    and json.loads(saved_config.read_text()) == plan['config']):
                print(f'Skipping completed run: {name}', flush=True)
                statuses.append(dict(name=name, status='completed', skipped=True))
                save_batch_status('running', updated_at=timestamp())
                continue
            write_json(run_dir / 'status.json', dict(status='running', started_at=timestamp()))
            command = [sys.executable, '-m', 'experiment.avwan.train_offline_rl',
                       '--config', str(run_dir / 'config.yaml'), '--stage', 'train']
            resume = plan['resume'] or (training_dir / 'latest.pt' if (training_dir / 'latest.pt').exists() else None)
            if resume:
                command.extend(['--resume', str(resume)])
            try:
                run_command(command, REPO_ROOT, run_dir / 'training.log')
                write_json(run_dir / 'status.json', dict(status='completed', completed_at=timestamp()))
                statuses.append(dict(name=name, status='completed'))
                save_batch_status('running', updated_at=timestamp())
            except BaseException as error:
                write_json(run_dir / 'status.json', dict(status='failed', error=repr(error), time=timestamp()))
                statuses.append(dict(name=name, status='failed', error=repr(error)))
                raise
    except BaseException as error:
        save_batch_status('failed', error=repr(error), failed_at=timestamp())
        raise
    save_batch_status('completed', completed_at=timestamp())
    print(f'Offline RL batch complete: {output_root}', flush=True)


if __name__ == '__main__':
    main()
