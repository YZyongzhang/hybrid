"""Collect real terminal STOP failures and separately executed expert recoveries."""
import argparse
import json
import random
import resource
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .stop_study import OUT, DATA, PARENT, FOUNDATION, PYTHON, TRAIN_SEED, prepare
from .run_gamma_study import command_run, write, sha, now


def targets(kind,stage):
    rows=json.loads((DATA/'selection.json').read_text())['episodes']
    if kind=='policy':return [r for r in rows if r['collection_stage']==stage]
    ids={r['global_index'] for r in (json.loads(p.read_text()) for p in (DATA/'policy/metadata').glob('*.json')) if r['false_stop']}
    return [r for r in rows if r['global_index'] in ids]


def worker(kind,stage,shard,shards):
    import torch
    import numpy as np
    from .evaluate import load_episodes
    from .config.default import get_config
    from .offline.policy import OfflinePolicy
    from .collector.collect import save_episode
    from .environment.BFSExpert import NoReachableWaypointError
    from experiment import AVWANRLEnv
    random.seed(TRAIN_SEED);np.random.seed(TRAIN_SEED);torch.manual_seed(TRAIN_SEED);torch.set_num_threads(2)
    state=torch.load(PARENT,map_location='cpu',weights_only=False)
    policy=OfflinePolicy.from_checkpoint(state,FOUNDATION,torch.device('cuda:0'))
    cfg=get_config(str(OUT/'evaluation.yaml'),opts=['TASK_CONFIG.DATASET.SPLIT','train_multiple'])
    cfg.defrost();cfg.TASK_CONFIG.DATASET.DATA_PATH=str(DATA/'dataset.json')
    cfg.TASK_CONFIG.TRAIN.OPEN=False;cfg.TASK_CONFIG.COLLECT.OPEN=False
    cfg.TASK_CONFIG.TASK.MEASUREMENTS=[m for m in cfg.TASK_CONFIG.TASK.MEASUREMENTS if m!='TOP_DOWN_MAP'];cfg.freeze()
    ds=load_episodes(cfg);eps={(Path(e.scene_id).stem,str(e.episode_id)):e for e in ds.episodes}
    rows=targets(kind,stage)[shard::shards];env=AVWANRLEnv(cfg,ds);new=0
    try:
        for spec in rows:
            i=spec['global_index'];meta=DATA/kind/'metadata'/f'{i:05d}.json'
            if meta.exists():continue
            ep=eps[spec['scene'],str(spec['episode_id'])]
            env.habitat_env.current_episode=ep;env.seed(TRAIN_SEED+i);obs=env.reset();policy.reset()
            transitions=[];done=False;recovering=False;interventions=0;blocked=None
            initial=float(env.get_info(obs)['distance_to_goal']);info={}
            while not done:
                at_goal=bool(env.habitat_env.sim.reaching_goal)
                before=float(env.get_info(obs)['distance_to_goal'])
                proposed=int(policy.act(obs,env.action_mask(obs)));teacher=None
                if kind=='recovery':
                    if proposed==40 and not at_goal:recovering=True;interventions+=1
                    if recovering or at_goal:
                        try:teacher=int(env.get_expert_map_action())
                        except NoReachableWaypointError as error:
                            blocked=str(error)
                            if transitions:transitions[-1].update(done=True,truncated=True,collector_truncated=True)
                            break
                action=proposed if teacher is None else teacher
                obs,_,done,info=env.step(action);tr=env.last_transition
                failed=bool(action==40 and not info['success'])
                tr.update(proposed_action=proposed,teacher_action=teacher,expert_control=teacher is not None,
                    at_goal_before=at_goal,false_stop=failed,correct_stop=bool(action==40 and info['success']),
                    bc_allowed=not failed,distance_before=before,distance_after=float(info['distance_to_goal']),
                    reaching_waypoint=bool(info['reaching_waypoint']),cant_reach_waypoint=bool(info['cant_reach_waypoint']))
                if failed:assert done and tr['terminated'] and tr['discount']==0
                transitions.append(tr)
            if not transitions:raise RuntimeError(f'Empty {kind} episode {i}: {blocked}')
            path=DATA/kind/'episodes'/spec['scene']/(str(spec['episode_id'])+'.pkl.gz')
            record=dict(global_index=i,scene=spec['scene'],episode_id=str(spec['episode_id']),sound=spec['sound'],
                split='train_multiple',kind=kind,seed=TRAIN_SEED+i,policy_sha256=sha(PARENT),
                false_stop=any(t['false_stop'] for t in transitions),correct_stop=any(t['correct_stop'] for t in transitions),
                success=bool(info.get('success',False)),truncated=bool(transitions[-1]['truncated']),
                initial_distance=initial,final_distance=float(info.get('distance_to_goal',initial)),
                transitions=len(transitions),primitive_steps=sum(t['num_steps'] for t in transitions),
                interventions=interventions,expert_steps=sum(t['expert_control'] for t in transitions),blocked=blocked,
                file=str(path),raw_action_reward_next_state_preserved=True)
            save_episode(path,transitions,record);meta.parent.mkdir(parents=True,exist_ok=True);write(meta,record)
            print(json.dumps(record),flush=True);new+=1
            if new>=6 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>=6500:break
    finally:env.close()


def collect_group(kind,stage,shards=6):
    expected=targets(kind,stage)
    def run(shard):
        subset=expected[shard::shards];previous=-1
        while True:
            count=sum((DATA/kind/'metadata'/f"{r['global_index']:05d}.json").exists() for r in subset)
            if count==len(subset):return
            if count<=previous:raise RuntimeError(f'No collection progress: {kind}/{stage}/{shard}')
            previous=count
            command_run([PYTHON,'-m','experiment.avwan.collect_stop_data','--kind',kind,'--stage',str(stage),
                         '--shard',str(shard),'--shards',str(shards)],OUT/'collection',f'{kind}_stage{stage}_worker{shard}')
    with ThreadPoolExecutor(max_workers=shards) as pool:list(pool.map(run,range(shards)))


def collect():
    prepare();(OUT/'collection').mkdir(exist_ok=True)
    if (DATA/'collection_complete.json').exists() and (OUT/'collection_complete.json').exists():return
    collect_group('policy',0)
    stage0={r['global_index'] for r in targets('policy',0)}
    records=[r for r in (json.loads(p.read_text()) for p in (DATA/'policy/metadata').glob('*.json')) if r['global_index'] in stage0]
    count=sum(r['false_stop'] for r in records)
    write(OUT/'collection/stage_decision.json',dict(first_stage_episodes=len(records),negative_stops=count,
          threshold=100,expanded=count<100,decided_at=now()))
    if count<100:collect_group('policy',1)
    collect_group('recovery',0)
    totals={}
    for kind in ['policy','recovery']:
        records=[json.loads(p.read_text()) for p in (DATA/kind/'metadata').glob('*.json')]
        totals[kind]=dict(episodes=len(records),transitions=sum(r['transitions'] for r in records),
            negative_stops=sum(r['false_stop'] for r in records),positive_stops=sum(r['correct_stop'] for r in records),
            interventions=sum(r['interventions'] for r in records),expert_steps=sum(r['expert_steps'] for r in records))
    write(DATA/'collection_complete.json',dict(completed_at=now(),totals=totals))
    write(OUT/'collection_complete.json',dict(completed_at=now(),data=str(DATA),totals=totals))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['policy','recovery']);p.add_argument('--stage',type=int,default=0)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=6);args=p.parse_args()
    if args.kind:worker(args.kind,args.stage,args.shard,args.shards)
    else:collect()
