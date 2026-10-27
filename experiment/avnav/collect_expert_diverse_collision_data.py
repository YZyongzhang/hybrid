"""Collect train-only expert, diverse, and collision-focused AVNav trajectories.

All collision handling is observational and reward-only. Actions always come from
compute_oracle_actions or a predeclared train-only exploration branch; there is
no collision-triggered action override or recovery sequence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = Path('/data2/getuanhui/avnav-codex')
DEFAULT_PLAN = DEFAULT_OUTPUT / 'selection_manifest.jsonl'
TRAIN_CONFIG = ROOT / 'experiment/avnav/config/collision_reward_scaling_20261026.yaml'
TASK_CONFIG = ROOT / 'experiment/avnav/config/audiogoal.yaml'

# Frozen train-only behavior and reward settings.
GLOBAL_SEED = 20261110
MAX_STEPS = 200
DIVERSE_PERTURB_PROBABILITY = 0.20
COLLISION_PREFIX_STEPS = 8
COLLISION_PREFIX_PERTURB_PROBABILITY = 0.75
MAX_COLLISION_ATTEMPTS = 30
COLLISION_PENALTY = 1.0
FIRST_RECOVERY_BONUS = 0.5
RECOVERY_BONUS_DECAY = 0.5
UNRECOVERED_PENALTY = 0.5
RECOVERY_WINDOW_ORACLE_STEPS = 5
PROGRESS_TOLERANCE_METERS = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _task_seed(task_id: str, attempt: int) -> int:
    raw = f'{GLOBAL_SEED}:{task_id}:{attempt}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**32)


def _load_plan(path: Path, category: str, worker_id: int, workers: int):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]
    rows = [row for row in rows if row['category'] == category]
    if not rows:
        raise ValueError(f'No {category} tasks in {path}')
    if len({row['task_id'] for row in rows}) != len(rows):
        raise ValueError('Selection manifest has duplicate task IDs')
    return [row for i, row in enumerate(rows) if i % workers == worker_id]


def _short_info(info):
    keep = ('distance_to_goal', 'normalized_distance_to_goal', 'success', 'spl',
            'softspl', 'na', 'sna')
    result = {}
    for key in keep:
        value = (info or {}).get(key)
        if value is None:
            continue
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _collect_one(env, requested_episode, task, attempt):
    import numpy as np
    from experiment.avnav.evaluate_online import _distance_to_goal, _metrics

    scene = str(task['scene'])
    episode_id = str(task['episode_id'])
    expected = (scene, episode_id)
    env._env.current_episode = requested_episode
    observation = env.reset()
    active = env._env.current_episode
    actual = (Path(str(active.scene_id)).stem, str(active.episode_id))
    if actual != expected:
        raise RuntimeError(f'Episode reset changed identity: expected {expected}, got {actual}')

    rng = random.Random(_task_seed(task['task_id'], attempt))
    sim = env._env._sim
    obs_rows = [observation]
    actions, env_rewards, shaped_rewards, reward_adjustments = [], [], [], []
    dones, collisions, behavior_source, infos = [], [], [], []
    distances = [_distance_to_goal(env)]
    pending_recovery_age = None
    recovery_count = 0
    unresolved_count = 0
    collision_count = 0
    recovery_events = 0
    done, info = False, {}
    max_steps = MAX_STEPS

    for step_index in range(max_steps):
        oracle_actions = sim.compute_oracle_actions() or []
        if not oracle_actions:
            raise RuntimeError('Oracle returned an empty action sequence')
        oracle_action = int(oracle_actions[0])
        action = oracle_action
        source = 'expert_oracle'

        perturb_probability = 0.0
        if task['category'] == 'deverse':
            perturb_probability = DIVERSE_PERTURB_PROBABILITY
        elif task['category'] == 'collision' and step_index < COLLISION_PREFIX_STEPS:
            perturb_probability = COLLISION_PREFIX_PERTURB_PROBABILITY
        if oracle_action != 0 and perturb_probability and rng.random() < perturb_probability:
            alternatives = [candidate for candidate in (1, 2, 3)
                            if candidate != oracle_action]
            if alternatives:
                action = rng.choice(alternatives)
                source = 'train_exploration'

        distance_before = distances[-1]
        next_observation, reward, done, info = env.step(action=action)
        collided = bool(getattr(sim, 'previous_step_collided', False))
        distance_after = _distance_to_goal(env)
        progress = distance_before - distance_after
        env_reward = float(reward)
        adjustment = 0.0

        if pending_recovery_age is not None and source == 'expert_oracle':
            pending_recovery_age += 1
            if (not collided) and progress > PROGRESS_TOLERANCE_METERS:
                adjustment += FIRST_RECOVERY_BONUS * (RECOVERY_BONUS_DECAY ** recovery_count)
                recovery_count += 1
                recovery_events += 1
                pending_recovery_age = None
            elif pending_recovery_age >= RECOVERY_WINDOW_ORACLE_STEPS:
                adjustment -= UNRECOVERED_PENALTY
                unresolved_count += 1
                pending_recovery_age = None

        if collided:
            collision_count += 1
            adjustment -= COLLISION_PENALTY
            if pending_recovery_age is None:
                pending_recovery_age = 0

        shaped_reward = env_reward + adjustment
        actions.append(int(action))
        env_rewards.append(env_reward)
        shaped_rewards.append(shaped_reward)
        reward_adjustments.append(adjustment)
        dones.append(bool(done))
        collisions.append(collided)
        behavior_source.append(source)
        infos.append(_short_info(info))
        distances.append(distance_after)
        obs_rows.append(next_observation)
        if done:
            break

    if pending_recovery_age is not None and shaped_rewards:
        shaped_rewards[-1] -= UNRECOVERED_PENALTY
        reward_adjustments[-1] -= UNRECOVERED_PENALTY
        unresolved_count += 1

    spl, success = _metrics(env, info)
    trajectory = {
        'obs': obs_rows,
        'action_id': actions,
        'reward': shaped_rewards,
        'reward_env': env_rewards,
        'reward_adjustment': reward_adjustments,
        'done': dones,
        'collision': collisions,
        'behavior_source': behavior_source,
        'distance_to_goal': distances,
        'info': infos,
        'task_id': task['task_id'],
        'category': task['category'],
        'parent_scene': scene,
        'parent_episode_id': episode_id,
        'parent_replicate': int(task['parent_replicate']),
        'collection_attempt': int(attempt),
        'seed': _task_seed(task['task_id'], attempt),
        'episode_success': bool(success),
        'spl': float(spl),
        'collision_count': collision_count,
        'recovery_event_count': recovery_events,
        'unresolved_collision_count': unresolved_count,
        'action_handling': (
            'native shortest-path oracle output recomputed each state; fixed train-only '
            'exploration schedule by category; no collision-triggered action override or recovery'
        ),
    }
    return trajectory


def _setup_env(task_rows, split='train_multiple'):
    # Load LLVM before Habitat-Sim's bundled audio-propagation library. On this
    # node, the reverse order triggers a libz IFUNC relocation abort.
    import llvmlite.binding  # noqa: F401
    from experiment.avnav.config import resolve_path
    from experiment.avnav.environment.avnav_env import AudioNavRLEnv
    from experiment.avnav.environment.config import get_config
    from experiment.avnav.environment.evaluation_episodes import load_episodes

    keys = sorted({(str(row['scene']), str(row['episode_id'])) for row in task_rows})
    config = get_config(task_config_path=str(resolve_path(TASK_CONFIG)),
                        opts=['TASK_CONFIG.DATASET.SPLIT', split])
    config.defrost()
    config.TASK_CONFIG.TRAIN.OPEN = False
    config.TASK_CONFIG.COLLECT.OPEN = False
    config.TASK_CONFIG.TASK.SENSORS = ['SPECTROGRAM_SENSOR']
    config.TASK_CONFIG.TASK.MEASUREMENTS = [
        name for name in config.TASK_CONFIG.TASK.MEASUREMENTS if name != 'TOP_DOWN_MAP'
    ]
    config.freeze()
    dataset = load_episodes(config.TASK_CONFIG.DATASET, max_episodes=0,
                            seed=GLOBAL_SEED, episode_keys=keys)
    by_key = {(Path(str(ep.scene_id)).stem, str(ep.episode_id)): ep
              for ep in dataset.episodes}
    if set(by_key) != set(keys):
        raise ValueError('Loaded task episodes do not match the frozen train manifest')
    return AudioNavRLEnv(config, dataset), by_key


def _write_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def collect_worker(category, output_root, plan_path, worker_id=0, workers=1,
                   max_collision_attempts=MAX_COLLISION_ATTEMPTS):
    output_root = Path(output_root).expanduser().resolve()
    plan_path = Path(plan_path).expanduser().resolve()
    tasks = _load_plan(plan_path, category, worker_id, workers)
    env, episodes = _setup_env(tasks)
    category_root = output_root / category
    category_root.mkdir(parents=True, exist_ok=True)
    episode_root = category_root / 'episodes'
    manifest_path = category_root / f'manifest_worker_{worker_id:02d}.jsonl'
    attempts_path = category_root / f'attempts_worker_{worker_id:02d}.jsonl'
    completed = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get('status') == 'saved':
                    completed.add(row['task_id'])
    errors = 0
    saved = 0
    try:
        for i, task in enumerate(tasks, 1):
            if task['task_id'] in completed:
                continue
            episode = episodes[(str(task['scene']), str(task['episode_id']))]
            attempts = max_collision_attempts if category == 'collision' else 1
            final_row = None
            for attempt in range(attempts):
                try:
                    trajectory = _collect_one(env, episode, task, attempt)
                    row = {
                        'task_id': task['task_id'], 'category': category,
                        'scene': task['scene'], 'episode_id': task['episode_id'],
                        'attempt': attempt, 'steps': len(trajectory['action_id']),
                        'collisions': trajectory['collision_count'],
                        'recoveries': trajectory['recovery_event_count'],
                        'unresolved': trajectory['unresolved_collision_count'],
                        'success': trajectory['episode_success'], 'spl': trajectory['spl'],
                        'error': None,
                    }
                    with attempts_path.open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                        handle.flush()
                    if category == 'collision' and trajectory['collision_count'] == 0:
                        final_row = row
                        continue
                    out_path = episode_root / str(task['scene']) / f"{task['task_id']}.pkl"
                    _write_atomic(out_path, trajectory)
                    row.update(status='saved', file=str(out_path))
                    final_row = row
                    saved += 1
                    break
                except Exception as error:
                    errors += 1
                    final_row = {
                        'task_id': task['task_id'], 'category': category,
                        'scene': task['scene'], 'episode_id': task['episode_id'],
                        'attempt': attempt, 'status': 'error',
                        'error': f'{type(error).__name__}: {error}',
                    }
                    with attempts_path.open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(final_row, ensure_ascii=False) + '\n')
                        handle.flush()
            if final_row is None:
                final_row = {'task_id': task['task_id'], 'category': category,
                             'scene': task['scene'], 'episode_id': task['episode_id'],
                             'status': 'shortfall', 'reason': 'no attempt completed'}
            elif category == 'collision' and final_row.get('collisions', 0) == 0:
                final_row.update(status='shortfall', reason='no collision in retry budget')
            if final_row.get('status') != 'saved':
                if final_row.get('status') == 'error':
                    errors += 0
            with manifest_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(final_row, ensure_ascii=False) + '\n')
                handle.flush()
            if i % 25 == 0 or i == len(tasks):
                print(json.dumps({
                    'category': category, 'worker': worker_id,
                    'progress': f'{i}/{len(tasks)}', 'saved': saved, 'errors': errors,
                }), flush=True)
    finally:
        env.close()
    return {'worker': worker_id, 'tasks': len(tasks), 'saved': saved, 'errors': errors}


def _protocol_payload(output_root, plan_path):
    plan = json.loads((Path(output_root) / 'collection_plan.json').read_text())
    collector_script = Path(__file__).resolve()
    return {
        'status': 'frozen_before_collection',
        'output_root': str(Path(output_root).resolve()),
        'collector_script': str(collector_script),
        'collector_script_sha256': _sha256(collector_script),
        'task_config': str(TASK_CONFIG.resolve()),
        'task_config_sha256': _sha256(TASK_CONFIG),
        'plan': str((Path(output_root) / 'collection_plan.json').resolve()),
        'selection_manifest': str(Path(plan_path).resolve()),
        'selection_manifest_sha256': _sha256(Path(plan_path)),
        'source': {
            'split': 'train_multiple',
            'manifest': plan['source_training_manifest'],
            'manifest_sha256': plan['source_training_manifest_sha256'],
            'audit': plan['source_audit'],
            'validation_or_test_used': False,
        },
        'scene_count': plan['scene_count'],
        'category_counts': plan['task_counts'],
        'total_target_episodes': plan['tasks_total'],
        'max_steps': MAX_STEPS,
        'categories': {
            'greedy': 'native shortest-path oracle; recompute and execute the first raw oracle action every step',
            'deverse': {
                'parent_episodes': 'same selected parent episodes as greedy',
                'policy': 'native oracle with a fixed 20% per-step random alternative-action probability; applies irrespective of collision state',
            },
            'collision': {
                'parent_episodes': 'seeded 50-episode subset per scene from the greedy parents',
                'policy': f'fixed first-{COLLISION_PREFIX_STEPS}-step exploration window with {COLLISION_PREFIX_PERTURB_PROBABILITY:.2f} per-step random alternative-action probability; raw oracle actions thereafter, irrespective of collision state',
                'collision_selection': f'retry up to {MAX_COLLISION_ATTEMPTS} seeds and save only trajectories with at least one recorded collision; selection uses train-only outcomes',
            },
        },
        'action_handling': 'No collision-conditioned actions, no forced recovery sequence, no depth-based heuristic; random collection branches are fixed before collection and only used on training data.',
        'reward_shaping': {
            'applies_to': 'train-only collected transitions in all three categories',
            'collision_penalty': -COLLISION_PENALTY,
            'recovery_definition': f'an expert-oracle action with no collision and distance improvement > {PROGRESS_TOLERANCE_METERS} m after a collision',
            'first_recovery_bonus': FIRST_RECOVERY_BONUS,
            'later_recovery_decay': RECOVERY_BONUS_DECAY,
            'unrecovered_after_oracle_steps': RECOVERY_WINDOW_ORACLE_STEPS,
            'unrecovered_penalty': -UNRECOVERED_PENALTY,
            'raw_environment_reward_preserved_as': 'reward_env',
            'shaped_training_reward_field': 'reward',
        },
        'checkpoint_selection_rule': 'No checkpoint selection in this data collection run; all source scenes and parent episodes are train split only.',
    }


def _merge_manifests(output_root, category, workers):
    category_root = Path(output_root) / category
    rows = []
    for worker in range(workers):
        path = category_root / f'manifest_worker_{worker:02d}.jsonl'
        if path.exists():
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    rows.sort(key=lambda row: row['task_id'])
    out = category_root / 'manifest.jsonl'
    with out.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    counts = {}
    for row in rows:
        scene = row.get('scene', 'unknown')
        item = counts.setdefault(scene, {'targets': 0, 'saved': 0, 'collisions': 0,
                                         'recoveries': 0, 'errors': 0, 'shortfalls': 0})
        item['targets'] += 1
        item['saved'] += int(row.get('status') == 'saved')
        item['collisions'] += int(row.get('collisions', 0))
        item['recoveries'] += int(row.get('recoveries', 0))
        item['errors'] += int(row.get('status') == 'error')
        item['shortfalls'] += int(row.get('status') == 'shortfall')
    return {'category': category, 'target_tasks': len(rows),
            'saved_episodes': sum(x['saved'] for x in counts.values()),
            'actual_collision_steps': sum(x['collisions'] for x in counts.values()),
            'recovery_events': sum(x['recoveries'] for x in counts.values()),
            'errors': sum(x['errors'] for x in counts.values()),
            'shortfalls': sum(x['shortfalls'] for x in counts.values()),
            'manifest': str(out), 'per_scene': counts}


def run_all(output_root, plan_path, workers):
    output_root = Path(output_root).expanduser().resolve()
    plan_path = Path(plan_path).expanduser().resolve()
    protocol_path = output_root / 'collection_protocol.json'
    payload = _protocol_payload(output_root, plan_path)
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding='utf-8'))
        if existing != payload:
            raise FileExistsError('Existing output directory has a different frozen protocol')
    else:
        protocol_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                                 encoding='utf-8')
    summaries = []
    for category in ('greedy', 'deverse', 'collision'):
        category_root = output_root / category
        category_root.mkdir(parents=True, exist_ok=True)
        processes = []
        logs = []
        for worker in range(workers):
            log_path = category_root / f'worker_{worker:02d}.log'
            handle = log_path.open('w', encoding='utf-8')
            command = [sys.executable, '-m', 'experiment.avnav.collect_expert_diverse_collision_data',
                       '--worker-category', category, '--output-root', str(output_root),
                       '--plan', str(plan_path), '--worker-id', str(worker),
                       '--workers', str(workers)]
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                       stderr=subprocess.STDOUT, text=True)
            processes.append((process, handle, worker))
            logs.append(str(log_path))
        exit_codes = []
        for process, handle, worker in processes:
            exit_codes.append((worker, process.wait()))
            handle.close()
        if any(code != 0 for _, code in exit_codes):
            raise RuntimeError(f'{category} workers failed: {exit_codes}; logs={logs}')
        summaries.append(_merge_manifests(output_root, category, workers))
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    report = {
        'status': 'completed' if all(row['shortfalls'] == 0 and row['errors'] == 0
                                     for row in summaries) else 'completed_with_shortfalls_or_errors',
        'collection_protocol': str(protocol_path),
        'selection_manifest': str(plan_path),
        'category_summaries': summaries,
        'total_target_episodes': sum(row['target_tasks'] for row in summaries),
        'total_saved_episodes': sum(row['saved_episodes'] for row in summaries),
    }
    (output_root / 'collection_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--plan', default=str(DEFAULT_PLAN))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--worker-category', choices=('greedy','deverse','collision'))
    parser.add_argument('--worker-id', type=int, default=0)
    parser.add_argument('--max-collision-attempts', type=int, default=MAX_COLLISION_ATTEMPTS)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker_id < args.workers:
        parser.error('workers must be positive and worker-id must be in range')
    if args.worker_category:
        result = collect_worker(args.worker_category, args.output_root, args.plan,
                                args.worker_id, args.workers,
                                args.max_collision_attempts)
        print(json.dumps(result), flush=True)
    else:
        run_all(args.output_root, args.plan, args.workers)


if __name__ == '__main__':
    main()
