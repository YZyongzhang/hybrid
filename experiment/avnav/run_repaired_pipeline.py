"""Resume the declared repaired experiment after foundation training completes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment.avnav.train_codex_foundation import _atomic_json, _sha256
from experiment.avnav.benchmark_runtime import BENCHMARK_ROOT


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('val_multiple', 'val_multiple_unheard')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def checked_result(path, checkpoint):
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result['status'] != 'pass':
        return None
    if result['checkpoint_sha256'] != _sha256(checkpoint):
        raise RuntimeError(f'Checkpoint mismatch: {path}')
    if result.get('partial_validation_smoke') or result['evaluated_episodes'] != result['expected_episodes']:
        raise RuntimeError(f'Incomplete result cannot be selected: {path}')
    if any(result[key] for key in ('evaluation_errors', 'missing_expected_ids', 'unexpected_ids')):
        raise RuntimeError(f'Evaluation errors or identity mismatch: {path}')
    return result


def run(args):
    run_dir = Path(args.run_dir).resolve()
    protocol_path = run_dir / 'run_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    logs = run_dir / 'logs'
    logs.mkdir(exist_ok=True)
    # Prevent concurrent orchestrators from writing the same outputs.
    import fcntl
    lock = (run_dir / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'started_at': now(), 'status': 'running'}
    if not (run_dir / 'foundation/summary.json').exists():
        if not args.wait_foundation_pid:
            raise RuntimeError('Foundation must finish before launching this pipeline')
        while not (run_dir / 'foundation/summary.json').exists():
            # Verify the actual process every interval, not just a stale status file.
            os.kill(args.wait_foundation_pid, 0)
            state.update(stage='waiting_foundation', child_pid=args.wait_foundation_pid)
            _atomic_json(run_dir / 'pipeline_status.json', state)
            time.sleep(30)
    snapshot = run_dir / 'source_snapshot'
    source_manifest = {}
    for source in sorted((ROOT / 'experiment/avnav').rglob('*.py')) + [ROOT / 'Dataset/dataset_avnav.py']:
        destination = snapshot / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and _sha256(destination) != _sha256(source):
            raise RuntimeError(f'Source changed since pipeline start: {source}')
        shutil.copy2(source, destination)
        source_manifest[str(source.relative_to(ROOT))] = _sha256(source)
    _atomic_json(run_dir / 'source_snapshot_manifest.json', source_manifest)

    def command(stage, module, options):
        log = logs / f'{stage}.log'
        cmd = [sys.executable, '-m', module, *map(str, options)]
        state.update(stage=stage, command=cmd, log=str(log), stage_started_at=now())
        print(json.dumps({'stage': stage, 'status': 'started', 'time': now()}), flush=True)
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet', MPLCONFIGDIR='/tmp/avnav-repaired-mpl')
        with log.open('a') as handle:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            _atomic_json(run_dir / 'pipeline_status.json', state)
            code = child.wait()
        if code:
            state.update(status='failed', exit_code=code, finished_at=now())
            _atomic_json(run_dir / 'pipeline_status.json', state)
            raise RuntimeError(f'{stage} failed with {code}; see {log}')
        print(json.dumps({'stage': stage, 'status': 'completed', 'time': now()}), flush=True)

    features = run_dir / 'offline/features/manifest.json'
    if not features.exists():
        command('features', 'experiment.avnav.extract_codex_features', ['--run-dir', run_dir])
    cache = json.loads(features.read_text())
    if cache.get('feature_schema') != 'repaired-v2-sample-local-audio-spatial-tokens' or cache['test_data_loaded']:
        raise RuntimeError('Invalid feature cache schema or data split')
    if cache['foundation_sha256'] != _sha256(run_dir / 'foundation/action_stage/best.pth'):
        raise RuntimeError('Stale features')
    coverage = json.loads((run_dir / 'audit/transition_coverage.json').read_text())
    if coverage['collision_steps_followed_by_progress_within_16'] == 0:
        raise RuntimeError('No post-collision progress coverage; train data collection required')

    candidates = []
    for spec in protocol['candidates']:
        training = run_dir / 'offline/candidates' / spec['id']
        if not (training / 'summary.json').exists():
            options = ['--run-dir', run_dir, '--output-dir', training,
                       '--seed', protocol['seed'], '--epochs', protocol['offline_common']['epochs'],
                       '--updates-per-epoch', protocol['offline_common']['updates_per_epoch'],
                       '--batch-size', protocol['offline_common']['batch_size'],
                       '--collision-bc-weight', spec['collision_bc_weight'], '--sequence-length', spec['sequence_length'],
                       '--gamma', spec['gamma'], '--checkpoint-epochs', *protocol['offline_common']['checkpoint_epochs']]
            if spec['previous_action']:
                options.append('--previous-action')
            command(f'train_{spec["id"]}', 'experiment.avnav.train_codex_saccql_lstm', options)
        for epoch in protocol['offline_common']['checkpoint_epochs']:
            checkpoint = training / f'epoch_{epoch:03d}.pt'
            output = run_dir / 'evaluation/candidates' / spec['id'] / f'epoch_{epoch:03d}'
            results, artifacts = [], []
            for split in SPLITS:
                path = output / f'{split}.json'
                result = checked_result(path, checkpoint)
                if result is None:
                    command(f'val_{spec["id"]}_{epoch}_{split}', 'experiment.avnav.evaluate_repaired',
                            ['--run-dir', run_dir, '--checkpoint', checkpoint,
                             '--training-config', training / 'training_config.json', '--output-dir', output, '--split', split])
                    result = checked_result(path, checkpoint)
                results.append(result)
                artifacts.append({'path': str(path), 'sha256': _sha256(path)})
            candidates.append({'candidate': spec['id'], 'epoch': epoch, 'checkpoint': str(checkpoint),
                               'training_config': str(training / 'training_config.json'),
                               'mean_validation_SPL': sum(r['SPL'] for r in results) / 2,
                               'mean_validation_SR': sum(r['SR'] for r in results) / 2,
                               'validation_results': artifacts})
            _atomic_json(run_dir / 'validation_candidates.json', candidates)

    # Stable max retains the predeclared candidate/epoch order for exact ties.
    selected = max(candidates, key=lambda r: (r['mean_validation_SPL'], r['mean_validation_SR']))
    frozen_path = run_dir / 'frozen_selection.json'
    if not frozen_path.exists():
        final = run_dir / 'offline/training'
        final.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected['checkpoint'], final / 'best_policy.pt')
        shutil.copy2(selected['training_config'], final / 'training_config.json')
        source_files = sorted((ROOT / 'experiment/avnav').rglob('*.py'))
        source_files.extend([ROOT / 'Dataset/dataset_avnav.py', BENCHMARK_ROOT / 'configs/default.py',
                             BENCHMARK_ROOT / 'configs/audiogoal.yaml', protocol_path, features,
                             run_dir / 'audit/data_source_audit.json', run_dir / 'validation_candidates.json',
                             final / 'best_policy.pt', final / 'training_config.json',
                             run_dir / 'foundation/action_stage/best.pth'])
        for external in (Path('/home/getuanhui/project/sound-spaces/soundspaces'),
                         Path('/home/getuanhui/project/sound-spaces/ss_baselines/common'),
                         Path('/home/getuanhui/project/habitat-lab/habitat/core'),
                         Path('/home/getuanhui/project/habitat-lab/habitat/tasks/nav')):
            source_files.extend(sorted(external.rglob('*.py')))
        for split in ('val_multiple', 'val_multiple_unheard', 'test_multiple', 'test_multiple_unheard'):
            source_files.append(run_dir / f'audit/split_episode_ids/{split}.jsonl')
        frozen = {**selected, 'selected_source_checkpoint': selected['checkpoint'],
                  'checkpoint': str(final / 'best_policy.pt'), 'training_config': str(final / 'training_config.json'),
                  'frozen_at': now(), 'selection_rule': protocol['selection']['rule'],
                  'immutable_files': {str(p): _sha256(p) for p in source_files}}
        _atomic_json(frozen_path, frozen)
    else:
        frozen = json.loads(frozen_path.read_text())
        if frozen['selected_source_checkpoint'] != selected['checkpoint']:
            raise RuntimeError('Frozen selection differs from declared validation selection')
    for split in ('test_multiple', 'test_multiple_unheard'):
        if checked_result(run_dir / f'evaluation/test/{split}.json', frozen['checkpoint']) is None:
            command(split, 'experiment.avnav.evaluate_repaired', ['--run-dir', run_dir, '--split', split])
    tests = {split: json.loads((run_dir / f'evaluation/test/{split}.json').read_text())
             for split in ('test_multiple', 'test_multiple_unheard')}
    _atomic_json(run_dir / 'final_results.json', {
        'status': 'completed', 'selection': frozen,
        'test': {split: {k: result[k] for k in ['status', 'expected_episodes', 'evaluated_episodes',
                 'evaluation_errors', 'SR', 'SPL', 'forward_collision_rate', 'timeout_rate',
                 'mean_success_path_efficiency']} for split, result in tests.items()},
        'completed_at': now(), 'test_used_for_training_or_selection': False,
    })
    state.update(status='completed', finished_at=now())
    _atomic_json(run_dir / 'pipeline_status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--wait-foundation-pid', type=int)
    run(parser.parse_args())
