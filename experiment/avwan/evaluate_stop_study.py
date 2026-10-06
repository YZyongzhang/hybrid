"""Fresh-process, fixed-episode evaluation with passive STOP/collision diagnostics."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .stop_study import OUT, PARENT, FOUNDATION, PYTHON, EVAL_SEED, SPLITS
from .run_gamma_study import write, sha, now, command_run


def checkpoint_for(name):
    if name == 'exp001-parent-baseline':return PARENT
    selected=OUT/name/'evaluation_checkpoint.json'
    return Path(json.loads(selected.read_text())['checkpoint']) if selected.exists() else OUT/name/'training/best_policy.pt'


def worker(name, split, index):
    # Torch must precede the simulator's native libraries.
    import torch
    import numpy as np
    from .config.default import get_config
    from .evaluate import load_episodes
    from .offline.policy import OfflinePolicy
    from experiment import AVWANRLEnv
    random.seed(EVAL_SEED); np.random.seed(EVAL_SEED); torch.manual_seed(EVAL_SEED)
    torch.set_num_threads(2)
    path = checkpoint_for(name)
    state = torch.load(path, map_location='cpu', weights_only=False)
    policy = OfflinePolicy.from_checkpoint(state, FOUNDATION, torch.device('cuda:0'))
    cfg = get_config(str(OUT/'evaluation.yaml'), opts=['TASK_CONFIG.DATASET.SPLIT', split])
    cfg.defrost(); cfg.TASK_CONFIG.TRAIN.OPEN=False; cfg.TASK_CONFIG.COLLECT.OPEN=False
    cfg.TASK_CONFIG.TASK.MEASUREMENTS = [m for m in cfg.TASK_CONFIG.TASK.MEASUREMENTS if m != 'TOP_DOWN_MAP']
    cfg.freeze()
    assert cfg.PREDICTION_INTERVAL == 5 and cfg.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS == 200
    dataset = load_episodes(cfg, str(OUT/'selections'/split))
    planned = list(dataset.episodes); assert len(planned) == 100
    ep = planned[index]
    folder = OUT/name/split/f'episode_{index:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(name=name, split=split, global_index=index, scene=Path(ep.scene_id).stem,
                    episode_id=str(ep.episode_id), seed=EVAL_SEED+index, checkpoint=str(path),
                    checkpoint_sha256=sha(path), foundation_sha256=sha(FOUNDATION), epoch=state['epoch'],
                    selection_sha256=sha(OUT/'selections'/split/'episodes.jsonl'),
                    fresh_process=True, expert_actions=False, prediction_interval=5, max_primitive_steps=200)
    write(folder/'run.json', metadata)
    env = AVWANRLEnv(cfg, dataset)
    events=[]; macros=[]; macro=-1; previous=None
    original = env.planner.update_map_and_graph
    def capture(obs):
        nonlocal previous
        primitive = env.planner._prev_action
        original(obs)
        position = np.asarray(env.habitat_env.sim.get_agent_state().position, dtype=float)
        if macro >= 0:
            events.append(dict(step=len(events)+1, macro=macro, primitive=int(primitive),
                collision=bool(np.asarray(obs['collision']).reshape(-1)[0]), position=position.tolist(),
                displacement=float(np.linalg.norm(position-previous)),
                distance=float(env.get_info(obs)['distance_to_goal']), at_goal=bool(env.habitat_env.sim.reaching_goal)))
        previous = position.copy()
    env.planner.update_map_and_graph = capture
    try:
        env.habitat_env.current_episode=ep; env.seed(EVAL_SEED+index); policy.reset(); obs=env.reset()
        hashes={}
        for key, value in {**{k:obs[k] for k in ['rgb','depth','spectrogram','gm','am']}, 'action_mask':env.action_mask(obs)}.items():
            array=np.ascontiguousarray(value)
            hashes[key]=hashlib.sha256(str((array.shape,str(array.dtype))).encode()+array.tobytes()).hexdigest()
        initial=float(env.get_info(obs)['distance_to_goal']); done=False
        while not done:
            macro+=1
            before=float(env.get_info(obs)['distance_to_goal']); at_goal=bool(env.habitat_env.sim.reaching_goal)
            action=int(policy.act(obs,env.action_mask(obs)))
            obs,reward,done,info=env.step(action)
            macros.append(dict(macro=macro,action=action,distance_before=before,at_goal_before=at_goal,
                distance_after=float(info['distance_to_goal']),reward=float(reward),
                low_level_actions=info['low_level_actions'],num_steps=int(info['num_steps']),
                blocked=bool(info['cant_reach_waypoint']),reached=bool(info['reaching_waypoint'])))
        summary=dict(**metadata,initial_observation_hashes=hashes,initial_distance=initial,
            final_distance=float(info['distance_to_goal']),success=bool(info['success']),spl=float(info['spl']),
            false_stop=bool(action==40 and not info['success']),correct_stop=bool(action==40 and info['success']),
            timeout=bool(info['truncated']),terminated=bool(info['terminated']),
            primitive_steps=sum(m['num_steps'] for m in macros),policy_decisions=len(macros),
            collisions=sum(e['collision'] for e in events),minimum_distance=min([initial]+[e['distance'] for e in events]),
            visited_goal=any(e['at_goal'] for e in events),goal_at_decision=any(m['at_goal_before'] for m in macros),
            missed_goal_inside_macro=any(e['at_goal'] for e in events) and not any(m['at_goal_before'] for m in macros),
            completed_at=now())
        for filename, rows in [('primitive_steps.jsonl',events),('macro_steps.jsonl',macros)]:
            (folder/filename).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        write(folder/'summary.json',summary)
        print(json.dumps(summary),flush=True)
    finally:
        env.close()


def evaluate(name, split):
    folder=OUT/name/split; folder.mkdir(parents=True,exist_ok=True)
    identity=sha(checkpoint_for(name)); foundation_identity=sha(FOUNDATION)
    selection_identity=sha(OUT/'selections'/split/'episodes.jsonl')
    def run(index):
        target=folder/f'episode_{index:03d}'
        output=target/'summary.json'
        if output.exists():
            row=json.loads(output.read_text())
            assert row['checkpoint_sha256']==identity and row['foundation_sha256']==foundation_identity
            assert row['selection_sha256']==selection_identity and row['global_index']==index and row['split']==split
            return row
        target.mkdir(exist_ok=True)
        command_run([PYTHON,'-m','experiment.avwan.evaluate_stop_study','--name',name,'--split',split,
                     '--index',str(index),'--output-root',str(OUT)],target,'evaluation')
        return json.loads(output.read_text())
    with ThreadPoolExecutor(max_workers=int(os.environ.get('AVWAN_STOP_EVAL_WORKERS','6'))) as pool:
        rows=list(pool.map(run,range(100)))
    assert len({(r['scene'],r['episode_id']) for r in rows})==100
    summary=dict(name=name,split=split,episodes=len(rows),checkpoint_sha256=identity,
                 spl=sum(r['spl'] for r in rows)/len(rows),success_rate=sum(r['success'] for r in rows)/len(rows))
    for key in ['false_stop','correct_stop','timeout','collisions','visited_goal','goal_at_decision','missed_goal_inside_macro']:
        summary[key]=sum(r[key] for r in rows)
    write(folder/'results.json',summary)
    (folder/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--name',required=True); p.add_argument('--split',choices=SPLITS,required=True)
    p.add_argument('--index',type=int); p.add_argument('--output-root',type=Path); args=p.parse_args()
    if args.output_root is not None: OUT=args.output_root.resolve()
    if args.index is None: print(json.dumps(evaluate(args.name,args.split)))
    else: worker(args.name,args.split,args.index)
