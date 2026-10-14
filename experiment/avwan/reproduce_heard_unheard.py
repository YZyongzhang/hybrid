"""Reproduce the no-selector AVWAN offline policy and evaluate both full test splits.

Run from the repository root with the SoundSpaces ``ss`` Python environment.
The training command uses the immutable Oct 3 foundation/parent feature caches
and the Oct 4 executed-action STOP/recovery annotations. It creates a new run.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .stop_study import FOUNDATION, PARENT


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'result/2026-10-10-heard-unheard-reproduction'
EVALUATION_CONFIG = ROOT / 'result/10-11/total/2026-10-04-stop-negative-recovery-study/evaluation.yaml'
EXPECTED = {'test_multiple': 1000, 'test_multiple_unheard': 1000}
BASE_CACHE = ROOT / 'result/2026-10-03-unheard-generalization-study/ratio_memory_features'
ANNOTATIONS = Path('/data/Getuanhui/avwan_stop_negative_recovery_20261004/training_annotations.json')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def train(args):
    from . import train_stop_rl as runner

    args.output.mkdir(parents=True, exist_ok=True)
    runner.OUT = args.output
    runner.TRAIN_SEED = args.seed
    runner.train(args.name, 'recovery_expert_bc', args.failed_stop_penalty,
                 base_cache=args.base_cache, steps_per_epoch=args.steps_per_epoch,
                 epochs=args.epochs, sampling=args.sampling,
                 foundation_checkpoint=args.foundation,
                 parent_checkpoint=args.parent,
                 additional_cache=args.additional_cache,
                 validation_cache=args.validation_cache,
                  training_overrides=json.loads(args.training_overrides),
                 random_initialization=args.random_initialization,
                  device=args.device, cpu_threads=args.cpu_threads,
                  gpu_memory_fraction=args.gpu_memory_fraction,
                  max_rss_mib=args.max_rss_mb,
                  min_available_gib=args.min_available_gib)
    checkpoint = args.output / args.name / 'training/latest.pt'
    parent_path = args.parent or (None if args.random_initialization else PARENT)
    print(json.dumps({'checkpoint': str(checkpoint), 'sha256': digest(checkpoint),
                      'foundation': str(args.foundation), 'foundation_sha256': digest(args.foundation),
                      'parent': str(parent_path) if parent_path else None,
                      'parent_sha256': digest(parent_path) if parent_path else None}, indent=2))


def episode_key(path):
    path = Path(path)
    name = path.name.removesuffix('.pkl.gz')
    for prefix in ('correction_', 'policy_', 'recovery_'):
        name = name.removeprefix(prefix)
    return path.parent.name, name


def audit(args):
    import torch

    checkpoint = args.checkpoint or args.output / args.name / 'training/latest.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = state['config']
    if state['signature']['features']['checkpoint_sha256'] != digest(args.foundation):
        raise ValueError('Foundation hash does not match the policy')
    if cfg['stop_pipeline']['mode'] != 'recovery_expert_bc':
        raise ValueError('Unexpected training data/BC scope')
    training = cfg['training']
    if (not 0 <= training['gamma'] <= 1 or training['bc_weight'] < 0 or
            training['actor_rl_weight'] < 0 or training['cql_weight'] < 0 or
            training['cql_temperature'] <= 0 or training['learning_rate'] <= 0 or
            training['initial_alpha'] <= 0 or not isinstance(training['terminal_mask'], bool) or
            not isinstance(training['foundation_prior'], bool)):
        raise ValueError('Invalid SAC-CQL training configuration')
    if state['signature']['architecture'] != 'independent_lstm_discrete_sac_cql_v1':
        raise ValueError('Checkpoint is not the recurrent SAC-CQL actor')
    parent_value = cfg['stop_pipeline'].get('parent')
    parent_checkpoint = Path(parent_value) if parent_value else None
    if parent_checkpoint is not None:
        if state['initialization_checkpoint_sha256'] != digest(parent_checkpoint):
            raise ValueError('Parent actor hash does not match the training checkpoint')
    elif state.get('initialization_checkpoint_sha256') is not None:
        raise ValueError('Expected random actor initialization, found an external parent checkpoint')
    if state['signature']['stop_annotations_sha256'] != digest(ANNOTATIONS):
        raise ValueError('STOP/recovery annotation hash does not match the checkpoint')
    expert_annotations = Path('/data/Getuanhui/avwan_stop_negative_recovery_20261004/expert_bc_annotations.json')
    if cfg['stop_pipeline']['expert_annotation_sha256'] != digest(expert_annotations):
        raise ValueError('Executed-expert BC annotation hash does not match the checkpoint')
    base_cache = Path(cfg['stop_pipeline'].get('base_cache', BASE_CACHE))
    base_manifest = json.loads((base_cache / 'manifest.json').read_text())
    if base_manifest['identity']['checkpoint_sha256'] != digest(args.foundation):
        raise ValueError('Base feature cache has a different foundation')
    additional_cache = Path(cfg['stop_pipeline'].get('additional_cache',
                              '/data/Getuanhui/avwan_stop_negative_recovery_20261004/frozen_ratio_features'))
    validation_cache = Path(cfg['stop_pipeline'].get('validation_cache', additional_cache))
    for cache in (additional_cache, validation_cache):
        identity = json.loads((cache / 'manifest.json').read_text())['identity']
        if identity['checkpoint_sha256'] != digest(args.foundation):
            raise ValueError(f'Feature cache has a different foundation: {cache}')
    base = base_manifest['splits']['train']
    extra = json.loads(ANNOTATIONS.read_text())
    train_keys = {episode_key(row['source']) for row in base}
    train_keys.update(episode_key(row['path']) for row in extra)
    collection_index = Path('/data/Getuanhui/avwan_multiple_greedy_train/episodes.jsonl')
    collected = [row for line in collection_index.read_text().splitlines() if line.strip()
                 for row in [json.loads(line)]
                 if (row['scene'], str(row['episode_id'])) in train_keys]
    if base_cache.resolve() != BASE_CACHE.resolve():
        base_keys = {episode_key(row['source']) for row in base}
        source = base_manifest['identity']['source']
        source_roots = source if isinstance(source, list) else [source]
        frozen_snapshots = [Path(root) for root in source_roots
                            if (Path(root) / 'snapshot.json').is_file()]
        if frozen_snapshots:
            if len(source_roots) != 1 or len(frozen_snapshots) != 1:
                raise ValueError('Expected one frozen large-data source snapshot')
            snapshot = frozen_snapshots[0]
            marker = json.loads((snapshot / 'snapshot.json').read_text())
            if marker['episodes'] != len(base):
                raise ValueError('Frozen large-data snapshot/cache count mismatch')
            from .data_pipeline.prepare import snapshot as source_snapshot
            if source_snapshot(snapshot)[3] != base_manifest['identity']['fingerprint']:
                raise ValueError('Frozen large-data source files changed since feature extraction')
            index_paths = [Path(marker['source']) / 'episodes.jsonl']
        else:
            # A custom feature-cache path can still point directly at ordinary
            # collector directories. Only frozen snapshot directories carry a
            # snapshot.json marker; verify raw sources by their feature-cache
            # fingerprint instead of assuming every custom cache is a snapshot.
            from .offline.features import snapshot_sources
            max_episodes = base_manifest['identity'].get('max_episodes', 0)
            if snapshot_sources(source, max_episodes)[3] != base_manifest['identity']['fingerprint']:
                raise ValueError('Base feature source files changed since feature extraction')
            index_paths = [Path(root) / 'episodes.jsonl' for root in source_roots]

        source_rows = [json.loads(line) for index_path in index_paths
                       for line in index_path.open() if line.strip()]
        base_rows = [row for row in source_rows if row.get('saved') and
                     (row['scene'], str(row['episode_id'])) in base_keys]
        indexed_base_keys = {(row['scene'], str(row['episode_id'])) for row in base_rows}
        if indexed_base_keys != base_keys:
            raise ValueError('Base feature episodes differ from their source collection index')
        collected.extend(base_rows)
    indexed_keys = {(row['scene'], str(row['episode_id'])) for row in collected if row.get('saved')}
    if train_keys != indexed_keys:
        raise ValueError('Training features differ from the frozen train collection index')
    train_sounds = {row['sound'] for row in collected if row.get('saved')}
    populations = {}
    for split, expected in EXPECTED.items():
        path = ROOT / 'data/datasets/audionav/mp3d/v1' / split / f'{split}.json.gz'
        if not path.exists():
            path = path.with_suffix('')
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as f:
            records = json.load(f)['episodes']
        keys = {(Path(row['scene_id']).stem, str(row['episode_id'])) for row in records}
        if len(records) != expected or len(keys) != expected:
            raise ValueError(f'{split} does not have {expected} unique episodes')
        overlap = train_keys & keys
        if overlap:
            raise ValueError(f'Training/test episode overlap in {split}: {sorted(overlap)[:5]}')
        sounds = {row['info']['sound'] for row in records}
        shared = sounds & train_sounds
        if (split == 'test_multiple' and shared != sounds) or (split == 'test_multiple_unheard' and shared):
            raise ValueError(f'{split} has unexpected heard/unheard sound membership')
        populations[split] = {'episodes': expected, 'training_episode_overlap': 0,
                              'sounds': len(sounds), 'sounds_seen_in_training': len(shared)}
    audited_sources = ('experiment/avwan/evaluate.py', 'experiment/avwan/offline/policy.py',
                       'experiment/avwan/offline/sac_cql.py',
                       'experiment/avwan/environment/avwan_env.py',
                       'experiment/avwan/environment/Planner.py')
    report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
              'foundation_sha256': digest(args.foundation),
              'training_episode_keys': len(train_keys), 'base_cache': str(base_cache),
              'additional_cache': str(additional_cache), 'validation_cache': str(validation_cache),
              'parent_checkpoint_sha256': digest(parent_checkpoint) if parent_checkpoint is not None else None,
              'test_populations': populations,
              'actor': 'masked argmax over trained SAC actor logits',
              'revisit_selector_enabled': False, 'expert_action_at_evaluation': False,
              'source_sha256': {rel: digest(ROOT / rel) for rel in audited_sources},
              'note': 'The common AVWAN local waypoint planner executes selected actions.'}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'audit.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def complete(output, expected):
    path = output / 'summary.json'
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    return summary['episodes'] == summary['planned'] == expected


def evaluate_split(args, split, checkpoint):
    all_rows = []
    selected_shards = ([args.shard_index] if args.shard_index is not None
                       else range(args.shards))
    for shard in selected_shards:
        expected = len(range(shard, EXPECTED[split], args.shards))
        folder = args.output / 'full_test' / split / f'shard_{shard}'
        while not complete(folder, expected):
            cmd = [sys.executable, '-m', 'experiment.avwan.evaluate',
                   '--checkpoint', str(checkpoint), '--foundation-checkpoint', str(args.foundation),
                   '--split', split, '--prediction-interval', str(args.prediction_interval),
                   '--output', str(folder), '--seed', str(args.eval_seed), '--device', args.device,
                   '--shards', str(args.shards), '--shard-index', str(shard),
                   '--max-new-episodes', str(args.episodes_per_process),
                   '--max-rss-mb', str(args.max_rss_mb), '--resume']
            if args.gpu_memory_fraction:
                cmd.extend(['--gpu-memory-fraction', str(args.gpu_memory_fraction)])
            if args.evaluation_config:
                cmd.extend(['--config', str(args.evaluation_config)])
            print(' '.join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
        selection = [tuple(x) for x in json.loads((folder / 'selection.json').read_text())]
        run = json.loads((folder / 'run.json').read_text())
        if (run['checkpoint_sha256'] != digest(checkpoint)
                or run['foundation_sha256'] != digest(args.foundation)
                or run['policy'] != 'frozen_foundation_lstm_sac_masked_argmax'
                or run['prediction_interval'] != args.prediction_interval
                or 'revisit_penalty' in run or 'reset_memory_every_step' in run):
            raise ValueError(f'Unexpected policy or rule-based evaluation setting: {folder}')
        rows = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
        keys = [(row['scene'], row['episode_id']) for row in rows]
        if len(rows) != expected or set(keys) != set(selection) or len(set(keys)) != expected:
            raise ValueError(f'Incomplete or duplicate rollout in {folder}')
        all_rows.extend(rows)
    if args.shard_index is not None:
        print(f'{split} shard {args.shard_index}/{args.shards} complete: {len(all_rows)} episodes', flush=True)
        return None
    keys = [(row['scene'], row['episode_id']) for row in all_rows]
    if len(keys) != EXPECTED[split] or len(set(keys)) != EXPECTED[split]:
        raise ValueError(f'{split}: test coverage is not exactly {EXPECTED[split]} unique episodes')
    result = {'split': split, 'episodes': len(all_rows),
              'spl': sum(row['spl'] for row in all_rows) / len(all_rows),
              'success_rate': sum(bool(row['success']) for row in all_rows) / len(all_rows),
              'errors': sum(bool(row.get('error')) for row in all_rows),
              'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
              'foundation': str(args.foundation), 'foundation_sha256': digest(args.foundation),
              'revisit_penalty': 0.0, 'reset_memory': False,
              'prediction_interval': 5, 'policy': 'frozen_foundation_lstm_sac_masked_argmax'}
    if result['errors']:
        raise RuntimeError(f'{split}: {result["errors"]} evaluation errors')
    path = args.output / 'full_test' / f'{split}.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def evaluate(args):
    checkpoint = args.checkpoint or args.output / args.name / 'training/latest.pt'
    if not checkpoint.exists() or not args.foundation.exists():
        raise FileNotFoundError('Checkpoint or foundation is missing')
    for split in args.splits:
        evaluate_split(args, split, checkpoint)
    report(args)


def validate(args):
    checkpoint = args.checkpoint or args.output / args.name / 'training/latest.pt'
    split = args.validation_split
    folder = args.output / 'validation' / split
    evaluation_config = args.evaluation_config or EVALUATION_CONFIG
    while not complete(folder, args.validation_episodes):
        cmd = [sys.executable, '-m', 'experiment.avwan.evaluate',
               '--checkpoint', str(checkpoint), '--foundation-checkpoint', str(args.foundation),
               '--config', str(evaluation_config), '--split', split,
               '--prediction-interval', str(args.prediction_interval),
               '--max-episodes', str(args.validation_episodes), '--output', str(folder),
               '--seed', str(args.eval_seed), '--device', args.device,
               '--max-new-episodes', str(args.episodes_per_process),
               '--max-rss-mb', str(args.max_rss_mb), '--resume']
        if args.gpu_memory_fraction:
            cmd.extend(['--gpu-memory-fraction', str(args.gpu_memory_fraction)])
        subprocess.run(cmd, cwd=ROOT, check=True)
    run = json.loads((folder / 'run.json').read_text())
    rows = [json.loads(line) for line in (folder / 'episodes.jsonl').open()]
    selection = [tuple(key) for key in json.loads((folder / 'selection.json').read_text())]
    keys = [(row['scene'], row['episode_id']) for row in rows]
    if (run['checkpoint_sha256'] != digest(checkpoint)
            or run['foundation_sha256'] != digest(args.foundation)
            or run['policy'] != 'frozen_foundation_lstm_sac_masked_argmax'
            or 'revisit_penalty' in run or 'reset_memory_every_step' in run
            or len(rows) != args.validation_episodes or set(keys) != set(selection)
            or len(set(keys)) != len(rows) or any(row.get('error') for row in rows)):
        raise ValueError('Incomplete or rule-assisted validation results')
    result = dict(split=split, episodes=len(rows),
                  spl=sum(row['spl'] for row in rows)/len(rows),
                  success_rate=sum(bool(row['success']) for row in rows)/len(rows),
                  checkpoint_sha256=digest(checkpoint),
                  selection_sha256=run['selection_sha256'])
    path = args.output / 'validation' / ('result.json' if split == 'val_multiple_unheard' else f'{split}.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def report(args):
    checkpoint = args.checkpoint or args.output / args.name / 'training/latest.pt'
    result_paths = [args.output / 'full_test' / f'{split}.json' for split in EXPECTED]
    if not all(path.exists() for path in result_paths):
        if args.stage == 'report':
            raise FileNotFoundError('Both complete split results are required to finalize')
        return
    results = [json.loads(path.read_text()) for path in result_paths]
    if any(row['checkpoint_sha256'] != digest(checkpoint) or row['episodes'] != EXPECTED[row['split']]
           or row['errors'] or row['foundation_sha256'] != digest(args.foundation) for row in results):
        raise ValueError('Full-test results do not match this checkpoint, foundation, or population')
    summary = {'spl_target': 0.4, 'passed': all(x['spl'] > 0.4 for x in results),
               'results': results}
    path = args.output / 'full_test/report.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['train', 'audit', 'validate', 'evaluate', 'report'])
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--name', default='retrain_seed_20261010')
    p.add_argument('--seed', type=int, default=20261010)
    p.add_argument('--base-cache', type=Path,
                   help='Optional frozen large-data feature cache for the original trajectory pool')
    p.add_argument('--steps-per-epoch', type=int, default=186)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--sampling', type=int, nargs=3, metavar=('ORIGINAL','NEGATIVE','RECOVERY'),
                   help='Three trajectory counts per batch; their sum must match training batch_size')
    p.add_argument('--failed-stop-penalty', type=float, default=1.0)
    p.add_argument('--training-overrides', default='{}',
                   help='JSON object overriding SAC-CQL training parameters from the parent config')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--gpu-memory-fraction', type=float, default=.12)
    p.add_argument('--min-available-gib', type=float, default=12.)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--foundation', type=Path, default=FOUNDATION)
    p.add_argument('--parent', type=Path, help='Actor initialization checkpoint for training')
    p.add_argument('--random-initialization', action='store_true',
                   help='Initialize the SAC-CQL actor and critics randomly instead of loading PARENT')
    p.add_argument('--additional-cache', type=Path,
                   help='Frozen features for failed STOP and recovery trajectories')
    p.add_argument('--validation-cache', type=Path,
                   help='Frozen independent validation demonstration features')
    p.add_argument('--eval-seed', type=int, default=20260930)
    p.add_argument('--evaluation-config', type=Path,
                   help='Optional Habitat config override; defaults to audiogoal.yaml.')
    p.add_argument('--prediction-interval', type=int, default=5)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard-index', type=int,
                   help='Evaluate one shard; rerun without this option to aggregate all shards')
    p.add_argument('--episodes-per-process', type=int, default=20)
    p.add_argument('--validation-episodes', type=int, default=100)
    p.add_argument('--validation-split', choices=('val_multiple', 'val_multiple_unheard'),
                   default='val_multiple_unheard')
    p.add_argument('--max-rss-mb', type=int, default=12000)
    p.add_argument('--splits', nargs='+', choices=tuple(EXPECTED), default=tuple(EXPECTED),
                   help='Evaluate one split per process to run heard and unheard concurrently')
    args = p.parse_args()
    args.output = args.output.resolve()
    args.foundation = args.foundation.resolve()
    if args.evaluation_config:
        args.evaluation_config = args.evaluation_config.resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
    if args.base_cache:
        args.base_cache = args.base_cache.resolve()
    if args.parent:
        args.parent = args.parent.resolve()
    if args.additional_cache:
        args.additional_cache = args.additional_cache.resolve()
    if args.validation_cache:
        args.validation_cache = args.validation_cache.resolve()
    if min(args.shards, args.episodes_per_process, args.validation_episodes,
           args.steps_per_epoch, args.epochs) < 1:
        p.error('shards, process episodes, validation episodes, training steps, and epochs must be positive')
    if not 0 < args.gpu_memory_fraction <= 1:
        p.error('--gpu-memory-fraction must be in (0, 1]')
    if args.max_rss_mb < 1 or args.min_available_gib < 0:
        p.error('Memory thresholds must be positive/nonnegative')
    if not 0 <= args.gpu_memory_fraction <= 1:
        p.error('--gpu-memory-fraction must be between 0 and 1')
    if args.shard_index is not None and not 0 <= args.shard_index < args.shards:
        p.error('--shard-index must be between 0 and --shards - 1')
    {'train': train, 'audit': audit, 'validate': validate,
     'evaluate': evaluate, 'report': report}[args.stage](args)


if __name__ == '__main__':
    main()
