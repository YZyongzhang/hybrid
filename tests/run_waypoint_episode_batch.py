"""Reproducible multi-scene expert validation, including navigation failures.

Run with the ss environment and GPU/EGL access. Each episode has a separate
process; one blocked/failed episode does not prevent the remaining checks.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONTENT = ROOT / 'data/datasets/audionav/mp3d/v1/train_telephone/content'


def read_scene(scene):
    with gzip.open(CONTENT / f'{scene}.json.gz', 'rt') as handle:
        return json.load(handle)


def select_cases(seed, count):
    cases = [dict(scene=scene, episode_id=eid, seed=case_seed, selection='regression')
             for scene, eid, case_seed in [('r1Q1Z4BcV1o', '77250', 0),
                                         ('VVfe2KiqLaN', '548', 20260924),
                                         ('GdvgFV5R1Z5', '11', 20260926)]]
    rng = random.Random(seed)
    scenes = sorted(p.name.removesuffix('.json.gz') for p in CONTENT.glob('*.json.gz')
                    if p.name.removesuffix('.json.gz') not in {c['scene'] for c in cases})
    rng.shuffle(scenes)
    for index, scene in enumerate(scenes[:count - len(cases)]):
        # Cover short, medium and long routes in separate randomly drawn scenes.
        low, high = [(0, 5), (5, 15), (15, float('inf'))][index % 3]
        episodes = read_scene(scene)['episodes']
        pool = [ep for ep in episodes if low < ep['info']['geodesic_distance'] <= high]
        ep = rng.choice(pool or episodes)
        cases.append(dict(scene=scene, episode_id=str(ep['episode_id']),
                          seed=seed + index, selection='sample',
                          dataset_distance=ep['info']['geodesic_distance']))
    return cases


def run_case(case, out):
    import numpy as np
    from PIL import Image
    from experiment import AVWANRLEnv, get_avwan_config
    from experiment.avwan.environment.BFSExpert import NoReachableWaypointError
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from soundspaces.datasets.audionav_dataset import AudioNavDataset
    from utils.visualizations import plot_top_down_map_waypoint

    result = dict(case, status='error', steps=[], runtime_error=None, blocked=None,
                  success=False, reset_ok=False, forward_attempts=0, forward_collisions=0)
    env = None
    collision_edges = Counter()
    try:
        random.seed(case['seed'])
        np.random.seed(case['seed'])
        config = get_avwan_config(opts=['TASK_CONFIG.DATASET.CONTENT_SCENES', [case['scene']]])
        raw = read_scene(case['scene'])
        raw['episodes'] = [ep for ep in raw['episodes'] if str(ep['episode_id']) == case['episode_id']]
        assert len(raw['episodes']) == 1
        dataset = AudioNavDataset()
        dataset._config = config.TASK_CONFIG.DATASET
        dataset.from_json(json.dumps(raw), scenes_dir=config.TASK_CONFIG.DATASET.SCENES_DIR)
        env = AVWANRLEnv(config, dataset)
        update = env.planner.update_map_and_graph

        def record_update(observation):
            if env.planner._prev_action == HabitatSimActions.MOVE_FORWARD:
                result['forward_attempts'] += 1
                if bool(np.asarray(observation['collision']).flat[0]):
                    result['forward_collisions'] += 1
                    x, y = env.planner.mapper.get_maps_and_agent_pose()[2:4]
                    collision_edges[(int(x), int(y), env.planner._prev_next_node)] += 1
            return update(observation)

        env.planner.update_map_and_graph = record_update
        env.seed(case['seed'])
        obs = env.reset()
        info = env.get_info(obs)
        result['initial_distance'] = float(info['distance_to_goal'])
        result['min_distance'] = result['initial_distance']
        done = False
        while not done:
            assert all(np.isfinite(np.asarray(obs[k])).all()
                       for k in ('depth', 'spectrogram', 'action_map', 'gm'))
            try:
                action = env.get_expert_map_action()
            except NoReachableWaypointError as error:
                result['blocked'] = str(error)
                break
            assert env.action_mask(obs)[action]
            if action == env.stop_action:
                assert env.habitat_env.sim.reaching_goal
            obs, reward, done, info = env.step(action)
            assert np.isfinite(reward)
            assert 1 <= info['num_steps'] <= config.PREDICTION_INTERVAL
            assert len(info['low_level_actions']) == info['num_steps']
            assert len(info['waypoint_positions']) == len(result['steps']) + 1
            assert info['expert_decision']['action'] == action
            transition = env.last_transition
            assert transition['done'] == done
            assert not (transition['terminated'] and transition['truncated'])
            assert np.isclose(transition['reward'], sum(
                env.gamma ** i * r for i, r in enumerate(info['low_level_rewards'])))
            if info['reaching_waypoint'] or action == env.stop_action:
                np.testing.assert_allclose(info['waypoint_position'],
                                           env.habitat_env.sim.get_agent_state().position,
                                           atol=1e-4, rtol=0)
            result['steps'].append(dict(action=action, reward=reward,
                distance=float(info['distance_to_goal']), num_steps=info['num_steps'],
                reached=info['reaching_waypoint'], blocked=info['cant_reach_waypoint'],
                low_level_actions=info['low_level_actions'], expert_decision=info['expert_decision']))
            result['min_distance'] = min(result['min_distance'], float(info['distance_to_goal']))
        result.update(success=bool(info.get('success', False)), done=bool(done),
                      truncated=bool(info.get('truncated', False)),
                      final_distance=float(info['distance_to_goal']), spl=float(info['spl']),
                      primitive_steps=sum(s['num_steps'] for s in result['steps']),
                      most_collided_edges=collision_edges.most_common(3))
        if result['success']:
            assert result['final_distance'] == 0 and action == env.stop_action
        if done:
            try:
                env.get_expert_map_action()
            except RuntimeError:
                pass
            else:
                raise AssertionError('Expert must require reset after episode termination')
        Image.fromarray(plot_top_down_map_waypoint(info)).save(out.with_suffix('.png'))
        # Snapshot collision counts above; reset is also checked for failed cases.
        env.reset()
        assert len(env.bfs_expert.visit_counts) == 1 and not env.bfs_expert.failed_targets
        assert env.bfs_expert.last_decision is None and not env.waypoint_positions
        result['reset_ok'] = True
        result['status'] = ('success' if result['success'] else
                            'blocked' if result['blocked'] else 'timeout' if result['truncated'] else 'failure')
    except Exception:
        result['runtime_error'] = traceback.format_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                result['status'] = 'error'
                result['runtime_error'] = traceback.format_exc()
        out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'steps'}), flush=True)
    return result['status'] != 'error'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20260923)
    parser.add_argument('--case', type=int)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.case is not None:
        cases = json.loads((out / 'selection.json').read_text())
        case = cases[args.case]
        return 0 if run_case(case, out / f"{case['scene']}_{case['episode_id']}.json") else 1
    if not 3 <= args.count <= 59:
        parser.error('--count must be between 3 and 59')
    cases = select_cases(args.seed, args.count)
    (out / 'selection.json').write_text(json.dumps(cases, indent=2) + '\n')
    sources = list((ROOT / 'experiment/avwan/environment').glob('*.py'))
    (out / 'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2) + '\n')
    rows = []
    for index, case in enumerate(cases):
        name = f"{case['scene']}_{case['episode_id']}"
        with (out / f'{name}.log').open('w') as log:
            try:
                process = subprocess.run([sys.executable, __file__, '--output', str(out),
                    '--case', str(index)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=240)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = 'process_timeout'
        path = out / f'{name}.json'
        row = json.loads(path.read_text()) if path.exists() else dict(case, status='error', success=False)
        row = {k: v for k, v in row.items() if k != 'steps'}
        row['exit_code'] = code
        if code != 0:
            row['status'] = 'error'
        rows.append(row)
        summary = dict(total=len(rows), success=sum(r['status'] == 'success' for r in rows),
                       statuses=dict(Counter(r['status'] for r in rows)), results=rows)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f"{index + 1}/{len(cases)} {name}: {row['status']}, "
              f"distance={row.get('final_distance')}, collisions={row.get('forward_collisions')}", flush=True)
    return 0 if all(r['status'] == 'success' for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
