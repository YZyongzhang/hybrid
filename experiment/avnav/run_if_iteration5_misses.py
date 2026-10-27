"""Run a larger, disjoint AVNAV data/training cycle if iteration 5 misses targets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import yaml


ROOT = Path('/home/getuanhui/project/hybrid_offline_RL')
BASE = ROOT / 'result_avnav/2026-10-19-avnav-generalization'
PREVIOUS = BASE / 'iteration5'
OUT = BASE / 'iteration6'
PYTHON = '/home/getuanhui/anaconda3/envs/ss/bin/python'
LETTERS = 'abcdef'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def log(message):
    line = f'[{time.strftime("%Y-%m-%dT%H:%M:%S%z")}] {message}'
    with (OUT / 'conditional_pipeline.log').open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')
        handle.flush()
    print(line, flush=True)


def run(command, output):
    log(f'Running: {" ".join(map(str, command))}')
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    with (OUT / output).open('w', encoding='utf-8') as handle:
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                       check=True, env=env)


def read_keys(path):
    return {(str(row['scene']), str(row['episode_id']))
            for row in json.loads(Path(path).read_text(encoding='utf-8'))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archived_script = OUT / 'run_if_iteration5_misses.py'
    archived_script.write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
    if not (OUT / 'run_context.json').exists():
        write_json(OUT / 'run_context.json', {
            'status': 'waiting_iteration5_full_test',
            'condition': 'run only if iteration 5 test_multiple_unheard SPL is below 0.10',
            'used_avwan_evaluation_code': False,
        })
    summary_path = PREVIOUS / 'full_test_summary.json'
    while not summary_path.is_file():
        time.sleep(60)
    previous = json.loads(summary_path.read_text(encoding='utf-8'))
    previous_unheard_spl = float(
        previous.get('full_test', {}).get('test_multiple_unheard', {}).get('spl', 0.0)
    )
    if previous_unheard_spl >= 0.10:
        write_json(OUT / 'run_context.json', {
            'status': 'not_required_iteration5_met_targets',
            'reason': 'AVNAV iteration 5 test_multiple_unheard SPL reached 0.10.',
            'test_multiple_unheard_spl_target': 0.10,
            'spl_target_met': True,
            'iteration5_full_test': previous.get('full_test'),
        })
        log('Iteration 5 test_multiple_unheard SPL reached 0.10; iteration 6 is not needed.')
        return 0

    audit5_path = PREVIOUS / 'collection_audit.json'
    if not audit5_path.is_file():
        raise SystemExit('Iteration 5 full test missed its target but collection audit is missing')
    audit5 = json.loads(audit5_path.read_text(encoding='utf-8'))
    if audit5.get('status') != 'passed':
        raise SystemExit('Cannot extend AVNAV training: iteration 5 collection audit did not pass')
    log('Iteration 5 missed a target; starting 6,000 disjoint AVNAV train_multiple episodes.')

    data = OUT / 'data'
    data.mkdir(parents=True, exist_ok=True)
    selected_paths = [
        BASE / 'iteration4/data/selected_train_episodes.json',
        PREVIOUS / 'data/selected_train_episodes.json',
    ]
    excluded_path = data / 'excluded_prior_and_iterations3_to5.jsonl'
    old_exclusion = BASE / 'iteration4/data/excluded_prior_and_iteration3.jsonl'
    excluded_rows = [json.loads(line) for line in old_exclusion.read_text(encoding='utf-8').splitlines()
                     if line.strip()]
    excluded_keys = {(str(row['scene']), str(row['episode_id'])) for row in excluded_rows}
    for path in selected_paths:
        keys = read_keys(path)
        if len(keys) != 3000 or keys & excluded_keys:
            raise SystemExit(f'Invalid or overlapping AVNAV iteration keys: {path}')
        excluded_keys.update(keys)
        excluded_rows.extend(json.loads(Path(path).read_text(encoding='utf-8')))
    if len(excluded_keys) != len(excluded_rows):
        # Historical exclusions may overlap one another; selected round 4/5 keys
        # are checked separately below and must remain unique.
        prior_unique = {(str(row['scene']), str(row['episode_id'])) for row in excluded_rows}
        if len(prior_unique) < len(excluded_rows) - 6000:
            log('Prior exclusion index contains historical duplicate keys; de-duplicating it.')
            seen = set()
            excluded_rows = [row for row in excluded_rows
                             if not ((str(row['scene']), str(row['episode_id'])) in seen or
                                     seen.add((str(row['scene']), str(row['episode_id']))))]
    excluded_path.write_text(''.join(json.dumps(row) + '\n' for row in excluded_rows),
                              encoding='utf-8')

    selection_path = data / 'selected_train_episodes.json'
    run([PYTHON, '-m', 'experiment.avnav.collector.select_diverse_episodes',
         '--config', str(ROOT / 'experiment/avnav/config/collect_round3a_20261019.yaml'),
         '--count', '6000', '--seed', '20261070', '--output', str(selection_path),
         '--exclude-index', str(excluded_path),
         '--distance-fractions', 'long=0.34,medium=0.33,short=0.33'], 'selection.log')
    selected = json.loads(selection_path.read_text(encoding='utf-8'))
    selected_keys = {(str(row['scene']), str(row['episode_id'])) for row in selected}
    if len(selected) != 6000 or len(selected_keys) != 6000 or selected_keys & excluded_keys:
        raise SystemExit('Iteration 6 train_multiple selection is not unique and disjoint')
    selection_summary = json.loads((data / 'selected_train_episodes_summary.json').read_text(encoding='utf-8'))

    collect_base = yaml.safe_load(
        (ROOT / 'experiment/avnav/config/collect_round4a_20261019.yaml').read_text(encoding='utf-8'))
    checkpoint = PREVIOUS / 'foundation/best.pth'
    if not checkpoint.is_file():
        raise SystemExit(f'Missing AVNAV iteration 5 foundation checkpoint: {checkpoint}')
    shard_specs = [(letter, index * 1000, (index + 1) * 1000, 20261071 + index)
                   for index, letter in enumerate(LETTERS)]
    configs = []
    for letter, start, end, seed in shard_specs:
        source = f'/data/Getuanhui/avnav_diverse_round6{letter}_20261019'
        config = dict(collect_base)
        config.update({
            'COLLECTION_NAME': f'avnav_diverse_round6{letter}_20261019',
            'SELECTION_PATH': str(selection_path), 'OUTPUT': source,
            'EPISODE_START': start, 'EPISODE_END': end, 'CHECKPOINT': str(checkpoint),
            'SEED': seed, 'MODEL_PROBABILITY': 0.20,
            'ORACLE_PROBABILITY': 0.30, 'RANDOM_PROBABILITY': 0.50,
        })
        config_path = ROOT / f'experiment/avnav/config/collect_round6{letter}_20261019.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        configs.append((letter, start, end, seed, source, config_path))

    context = {
        'status': 'collection_running',
        'purpose': 'AVNAV-only continuation because iteration 5 full test missed a requested threshold.',
        'split': 'train_multiple', 'planned_episodes': 6000, 'seed': 20261070,
        'selection': {
            'scene_count': selection_summary['scene_count'],
            'sound_count': selection_summary['sound_count'],
            'distance_counts': selection_summary['distance_counts'],
            'orientation_counts': selection_summary['orientation_counts'],
            'selected_unique_keys': len(selected_keys), 'overlap_with_previous': 0,
            'excluded_keys_count': len(excluded_keys),
        },
        'collection': {
            'checkpoint': str(checkpoint.resolve()),
            'policy_mixture': {'foundation_model': 0.20, 'oracle': 0.30, 'explore': 0.50},
            'max_steps': 200, 'recovery_steps': 4,
            'shards': [{'name': f'avnav_diverse_round6{letter}_20261019',
                        'config': str(config_path), 'output': source,
                        'slice': [start, end], 'seed': seed}
                       for letter, start, end, seed, source, config_path in configs],
        },
        'iteration5_full_test': previous.get('full_test'),
        'used_avwan_evaluation_code': False,
    }
    write_json(OUT / 'run_context.json', context)
    (OUT / 'EXPERIMENT.md').write_text(
        '# AVNAV generalization iteration 6\n\n'
        'This conditional AVNAV-only cycle runs only if iteration 5 misses a full-test target. '
        'It selects 6,000 unique train_multiple episodes, excludes prior training keys, and '
        'collects with a 20% model, 30% oracle, and 50% exploration mixture. It keeps the '
        'official test split out of training and uses only the AVNAV evaluator.\n',
        encoding='utf-8')

    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    all_rows = []
    failed = False
    for batch_start in range(0, len(configs), 3):
        children = []
        for letter, start, end, seed, source, config_path in configs[batch_start:batch_start + 3]:
            handle = (OUT / f'collection_{letter}.log').open('w', encoding='utf-8')
            proc = subprocess.Popen(
                [PYTHON, '-u', '-m', 'experiment.avnav.collector.collect_data',
                 '--config', str(config_path.relative_to(ROOT))],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
            children.append((letter, start, end, source, proc, handle))
            log(f'Launched AVNAV round6{letter}, pid={proc.pid}, episodes={start}:{end}.')
        for letter, start, end, source, proc, handle in children:
            code = proc.wait()
            handle.close()
            if code:
                failed = True
                log(f'AVNAV round6{letter} collector exited with code {code}.')
        for letter, start, end, source, _, _ in children:
            directory = Path(source)
            manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
            shard_rows = [json.loads(line) for line in (directory / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()
                          if line.strip()]
            expected = {(str(row['scene']), str(row['episode_id'])) for row in selected[start:end]}
            got = {(str(row['scene']), str(row['episode_id'])) for row in shard_rows}
            if (manifest.get('planned') != 1000 or manifest.get('completed') != 1000 or
                    manifest.get('errors') != 0 or got != expected):
                failed = True
            all_rows.extend(shard_rows)
    all_keys = {(str(row['scene']), str(row['episode_id'])) for row in all_rows}
    if len(all_rows) != 6000 or len(all_keys) != 6000 or all_keys & excluded_keys:
        failed = True
    audit = {
        'status': 'failed' if failed else 'passed', 'planned': 6000,
        'completed': len(all_rows),
        'successful_episodes': sum(bool(row.get('success')) for row in all_rows),
        'errors': sum(row.get('status') == 'error' for row in all_rows),
        'unique_episode_keys': len(all_keys),
        'overlap_with_previous': len(all_keys & excluded_keys),
        'transitions': sum(int(row.get('transitions', 0)) for row in all_rows),
        'sources': [source for _, _, _, _, source, _ in configs],
    }
    write_json(OUT / 'collection_audit.json', audit)
    context['status'] = 'collection_completed' if not failed else 'collection_incomplete'
    context['collection']['status'] = context['status']
    context['collection']['audit'] = str((OUT / 'collection_audit.json').resolve())
    write_json(OUT / 'run_context.json', context)
    with (OUT / 'EXPERIMENT.md').open('a', encoding='utf-8') as handle:
        handle.write(f"\n## Collection audit\n\n- Status: `{audit['status']}`; {audit['completed']}/6,000 episodes, "
                     f"{audit['successful_episodes']} successful, {audit['errors']} errors, "
                     f"{audit['unique_episode_keys']} unique AVNAV keys.\n")
    if failed:
        raise SystemExit('AVNAV iteration 6 collection audit failed')

    sources = [f'/data/Getuanhui/avnav_diverse_round6{x}_20261019/episodes' for x in LETTERS]
    destinations = [Path(f'/data/Getuanhui/avnav_diverse_round6_pt_{x}_20261019') for x in LETTERS]
    for letter, source, destination, seed in zip(LETTERS, sources, destinations,
                                                 (20261081, 20261082, 20261083,
                                                  20261084, 20261085, 20261086)):
        run([PYTHON, '-m', 'experiment.avnav.prepare_foundation_data', '--source', source,
             '--destination', str(destination), '--shard-size', '256', '--seed', str(seed)],
            f'convert_round6{letter}.log')

    foundation = yaml.safe_load(
        (ROOT / 'experiment/avnav/config/foundation_round5_20261019.yaml').read_text(encoding='utf-8'))
    foundation['DATA']['TRAIN_PT'] = list(foundation['DATA']['TRAIN_PT']) + [str(path) for path in destinations]
    foundation['FOUNDATION'].update({
        'OUTPUT': str(OUT / 'foundation'), 'AUDIO_EPOCHS': 18, 'ACTION_EPOCHS': 16,
        'GRL_SCALE': 0.85, 'DOMAIN_LOSS_WEIGHT': 2.5,
        'AUDIO_FREQ_MASK_WIDTH': 10, 'AUDIO_TIME_MASK_WIDTH': 6, 'SEED': 20261087,
    })
    foundation_config = ROOT / 'experiment/avnav/config/foundation_round6_20261019.yaml'
    foundation_config.write_text(yaml.safe_dump(foundation, sort_keys=False), encoding='utf-8')
    offline = yaml.safe_load(
        (ROOT / 'experiment/avnav/config/offline_rl_round5_20261019.yaml').read_text(encoding='utf-8'))
    offline['SEED'] = 20261088
    offline['DATA']['TRAIN_EPISODES'] = list(offline['DATA']['TRAIN_EPISODES']) + sources
    offline['FOUNDATION']['CHECKPOINT'] = str(OUT / 'foundation/best.pth')
    offline['OFFLINE_RL']['OUTPUT'] = str(OUT / 'offline/candidates')
    offline['OFFLINE_RL']['FEATURE_CACHE_ROOT'] = str(OUT / 'offline/shared_features')
    offline_config = ROOT / 'experiment/avnav/config/offline_rl_round6_20261019.yaml'
    offline_config.write_text(yaml.safe_dump(offline, sort_keys=False), encoding='utf-8')
    context['status'] = 'foundation_training'
    context['foundation_plan'] = {
        'audio_epochs': 18, 'action_epochs': 16, 'grl_scale': 0.85,
        'domain_loss_weight': 2.5, 'audio_freq_mask_width': 10,
        'audio_time_mask_width': 6, 'uses_iterations3_to6_success_data': True,
        'processed_data_stored_under': '/data/Getuanhui to protect workspace capacity',
    }
    write_json(OUT / 'run_context.json', context)
    run([PYTHON, '-m', 'experiment.avnav.train_staged_foundation', '--config', str(foundation_config)],
        'foundation_training.log')
    run([PYTHON, '-m', 'experiment.avnav.sweep_offline_rl', '--config', str(offline_config),
         '--output-root', str(OUT / 'offline/candidates'), '--epochs', '24',
         '--updates-per-epoch', '250'], 'offline_sweep.log')
    run([PYTHON, '-m', 'experiment.avnav.screen_offline_rl', '--sweep',
         str(OUT / 'offline/candidates/sweep.json'), '--episodes', '150', '--top-k', '8'],
        'validation_screen.log')

    screen = json.loads((OUT / 'offline/candidates/validation_screen.json').read_text(encoding='utf-8'))
    ranking = []
    for candidate in screen['candidates']:
        if candidate.get('status') != 'completed':
            continue
        splits = {row['split']: row for row in candidate['splits']}
        if not {'val_multiple', 'val_multiple_unheard'} <= set(splits):
            continue
        multiple = splits['val_multiple']['success_rate']
        unheard = splits['val_multiple_unheard']['success_rate']
        multiple_spl = splits['val_multiple']['spl']
        unheard_spl = splits['val_multiple_unheard']['spl']
        ranking.append({'candidate': candidate['name'], 'output': candidate['output'],
                        'val_multiple_success_rate': multiple,
                        'val_multiple_unheard_success_rate': unheard,
                        'val_multiple_spl': multiple_spl,
                        'val_multiple_unheard_spl': unheard_spl,
                        'mean_validation_spl': (multiple_spl + unheard_spl) / 2,
                        'balanced_min_success_rate': min(multiple, unheard),
                        'mean_success_rate': (multiple + unheard) / 2})
    if not ranking:
        raise SystemExit('No completed AVNAV iteration 6 policy passed validation')
    ranking.sort(key=lambda row: (row['val_multiple_unheard_spl'],
                                  row['mean_validation_spl'],
                                  row['val_multiple_spl']), reverse=True)
    chosen = ranking[0]
    write_json(OUT / 'offline/full_test_selection.json', {
        'status': 'selected_for_full_test',
        'selection_rule': 'maximize AVNAV val_multiple_unheard SPL; tie-break by mean validation SPL, then val_multiple SPL',
        **chosen, 'ranking': ranking,
    })
    run([PYTHON, '-m', 'experiment.avnav.evaluate_online', '--config',
         str(Path(chosen['output']) / 'candidate_config.yaml'), '--checkpoint',
         str(Path(chosen['output']) / 'best_policy.pt'), '--output',
         str(Path(chosen['output']) / 'full_test'), '--episodes', '1000', '--splits',
         'test_multiple', 'test_multiple_unheard'], 'full_test.log')
    report_path = Path(chosen['output']) / 'full_test/report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    full = {row['split']: {'episodes': row['episodes'], 'errors': row['errors'],
                           'success_rate': row['success_rate'], 'spl': row['spl']}
            for row in report['splits']}
    spl_target_met = full['test_multiple_unheard']['spl'] >= .10
    success_targets_met = (full['test_multiple']['success_rate'] >= .20 and
                           full['test_multiple_unheard']['success_rate'] >= .15)
    summary = {'status': 'spl_target_met' if spl_target_met else 'completed_below_target',
               'iteration5_full_test': previous.get('full_test'), 'candidate': chosen['candidate'],
               'validation': chosen, 'full_test': full,
               'test_multiple_unheard_spl_target': 0.10,
               'spl_target_met': spl_target_met,
               'success_targets_met': success_targets_met,
               'targets_met': spl_target_met,
               'full_test_report': str(report_path.resolve())}
    write_json(OUT / 'full_test_summary.json', summary)
    context = json.loads((OUT / 'run_context.json').read_text(encoding='utf-8'))
    context['status'] = summary['status']
    context['full_test'] = {'status': summary['status'], 'selected_candidate': chosen['candidate'],
                            'results': full, 'targets_met': spl_target_met,
                            'spl_target_met': spl_target_met,
                            'success_targets_met': success_targets_met,
                            'evaluation_code': 'experiment.avnav.evaluate_online',
                            'used_avwan_evaluation_code': False}
    write_json(OUT / 'run_context.json', context)
    with (OUT / 'EXPERIMENT.md').open('a', encoding='utf-8') as handle:
        handle.write('\n## Full-test results\n\n')
        handle.write(f"- Selected AVNAV policy: `{chosen['candidate']}`.\n")
        for split, row in full.items():
            handle.write(f"- `{split}`: {row['success_rate']:.1%} success, SPL {row['spl']:.4f}, "
                         f"{row['episodes']} episodes, {row['errors']} errors.\n")
        handle.write(f"- `test_multiple_unheard` SPL target (>= 0.10) met: **{spl_target_met}**.\n")
        handle.write(f"- Existing success-rate thresholds met: **{success_targets_met}**.\n")
    log(f'Iteration 6 full test completed. Unheard SPL target met={spl_target_met}; results={full}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
