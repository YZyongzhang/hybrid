"""Run three fixed regression episodes, each in a separate GPU process.

Outputs: outputs/local_navigation_depth_fix/{scene}_{episode}.{json,png}.
"""
import sys
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/local_navigation_depth_fix'
OUT.mkdir(parents=True,exist_ok=True)
if len(sys.argv)==1:
    for case in range(3):
        with (OUT/f'case_{case}.log').open('w') as handle:
            subprocess.run([sys.executable,__file__,str(case)],cwd=ROOT,stdout=handle,
                           stderr=subprocess.STDOUT,check=True)
    summaries = []
    import json
    for result in sorted(OUT.glob('*.json')):
        if result.name not in ('summary.json', 'comparison.json'):
            summaries.append({k: v for k, v in json.loads(result.read_text()).items() if k != 'steps'})
    (OUT/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    sys.exit(0)
import gzip,json,random
import numpy as np
from PIL import Image
from experiment import AVWANRLEnv,get_avwan_config
from experiment.avwan.environment.BFSExpert import NoReachableWaypointError
from soundspaces.datasets.audionav_dataset import AudioNavDataset
from utils.visualizations import plot_top_down_map_waypoint
case=int(sys.argv[1])
scene,eid,seed=[('r1Q1Z4BcV1o','77250',0),('VVfe2KiqLaN','548',20260924),
                ('GdvgFV5R1Z5','11',20260926)][case]
random.seed(seed);np.random.seed(seed)
config=get_avwan_config(opts=['TASK_CONFIG.DATASET.CONTENT_SCENES',[scene]])
with gzip.open(ROOT/f'data/datasets/audionav/mp3d/v1/train_telephone/content/{scene}.json.gz','rt') as h:
    raw=json.load(h)
raw['episodes']=[ep for ep in raw['episodes'] if str(ep['episode_id'])==eid]
assert len(raw['episodes'])==1
dataset=AudioNavDataset();dataset._config=config.TASK_CONFIG.DATASET
dataset.from_json(json.dumps(raw),scenes_dir=config.TASK_CONFIG.DATASET.SCENES_DIR)
env=AVWANRLEnv(config,dataset)
steps=[];blocked=None
collision_stats = {'forward_attempts': 0, 'forward_collisions': 0}
original_update = env.planner.update_map_and_graph
def record_update(observation):
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    if env.planner._prev_action == HabitatSimActions.MOVE_FORWARD:
        collision_stats['forward_attempts'] += 1
        collision_stats['forward_collisions'] += int(bool(np.asarray(observation['collision']).flat[0]))
    return original_update(observation)
env.planner.update_map_and_graph = record_update
try:
    from experiment.avwan.environment.Planner import Planner
    from experiment.avwan.environment.Mapper import Mapper
    from experiment.avwan.environment.nav import LocalEgoMap
    assert type(env.planner) is Planner
    assert type(env.planner.mapper) is Mapper
    assert type(env.habitat_env.task.sensor_suite.sensors['ego_map']) is LocalEgoMap
    assert config.TASK_CONFIG.TASK.EGOMAP_SENSOR.TYPE == 'EgoMap'
    env.seed(seed);obs=env.reset()
    assert len(env.bfs_expert.visit_counts)==1
    assert not env.bfs_expert.failed_targets
    done=False;info=env.get_info(obs)
    restored_edges = 0
    while not done:
        try:action=env.get_expert_map_action()
        except NoReachableWaypointError as e:
            blocked=str(e);break
        if action==env.stop_action:assert env.habitat_env.sim.reaching_goal
        previously_removed = set(env.planner._removed_edges)
        obs,reward,done,info=env.step(action)
        restored_edges += len(previously_removed - env.planner._removed_edges)
        assert info['expert_decision']['action']==action
        assert env._pending_expert_decision is None
        assert len(info['waypoint_positions'])==len(steps)+1
        record=dict(step=len(steps)+1,action=action,mode=info['expert_decision']['mode'],
                    reward=reward,distance=info['distance_to_goal'],num_steps=info['num_steps'],
                    reached=info['reaching_waypoint'],blocked=info['cant_reach_waypoint'],
                    success=info['success'],done=done)
        steps.append(record);print(json.dumps(record),flush=True)
    summary=dict(scene=scene,episode_id=eid,success=info.get('success',False),
                 seed=seed,collision_edge_blocking=False,**collision_stats,
                 spl=info.get('spl'),final_distance=info['distance_to_goal'],done=done,
                 truncated=info.get('truncated',False),blocked=blocked,
                 restored_edges_between_waypoints=restored_edges,
                 primitive_steps=sum(s['num_steps'] for s in steps),steps=steps,
                 visited_cells=len(env.bfs_expert.visit_counts),failed_targets=len(env.bfs_expert.failed_targets))
    (OUT/f'{scene}_{eid}.json').write_text(json.dumps(summary,indent=2)+'\n')
    Image.fromarray(plot_top_down_map_waypoint(info)).save(OUT/f'{scene}_{eid}.png')
    print('SUMMARY '+json.dumps({k:v for k,v in summary.items() if k!='steps'}),flush=True)
    # Episode 11 used to lose its sound-source cell to obstacle closing and
    # biased depth quantization. Keep actual arrival as a regression check.
    if case in (0, 2):
        assert summary['success'], summary
        assert summary['final_distance'] == 0.0, summary
    if done:
        try:env.get_expert_map_action()
        except RuntimeError:pass
        else:raise AssertionError('Expert should require reset after done')
    env.reset()
    assert len(env.bfs_expert.visit_counts)==1 and not env.bfs_expert.failed_targets
    assert env.bfs_expert.last_decision is None
    assert env._pending_expert_decision is None
    print('RESET LIFECYCLE PASS',flush=True)
finally:env.close()
