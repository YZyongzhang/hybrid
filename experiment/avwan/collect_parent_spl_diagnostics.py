"""Train-only pure-parent diagnostics and expert-assisted corrections.

This collector deliberately separates the parent's unassisted rollout from
expert-assisted replay. Only episodes from the scene-disjoint train_multiple
source are selected; validation and held-out data are never read by training.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "result/2026-10-15-avwan-spl-improvement"
DIAG = OUT / "diagnostics"
CORR = OUT / "corrections"
TRAIN_INDEX = Path("/data/Getuanhui/avwan_multiple_greedy_every_1000/episodes.jsonl")
TRAIN_CONTENT = ROOT / "data/datasets/audionav/mp3d/v1/train_multiple/content"
STUDY = ROOT / "result/10-11/total/2026-10-03-unheard-generalization-study"
EVAL_CONFIG = STUDY / "evaluation.yaml"
PARENT = ROOT / "result/2026-10-14-avwan-diverse-retraining/offline/seed_20261014_deverse_hard_longrange/training/best_policy.pt"
FOUNDATION = ROOT / "result/2026-10-13-staged-grl-reproduction/foundation/action_stage/best.pt"
PYTHON = "/home/getuanhui/anaconda3/envs/ss/bin/python"
SEED = 20261016
PER_BIN = 100
STOP = 40


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def category(distance):
    if distance < 16:
        return "under_16m"
    if distance < 24:
        return "16_to_24m"
    if distance < 32:
        return "24_to_32m"
    return "32m_plus"


def balanced_sample(rows, count, scene_counts, sound_counts, rng, scene_cap=16):
    remaining = list(rows)
    tie = {(r["scene"], str(r["episode_id"])): rng.random() for r in remaining}
    chosen = []
    while len(chosen) < count:
        eligible = [r for r in remaining if scene_counts[r["scene"]] < scene_cap]
        if not eligible:
            raise ValueError(f"Insufficient candidates under scene cap {scene_cap}")
        row = min(eligible, key=lambda r: (scene_counts[r["scene"]], sound_counts[r["sound"]],
                                           tie[(r["scene"], str(r["episode_id"]))],
                                           str(r["episode_id"])))
        chosen.append(row)
        remaining.remove(row)
        scene_counts[row["scene"]] += 1
        sound_counts[row["sound"]] += 1
    return chosen


def load_split_filters():
    audit = json.loads((STUDY / "split_audit.json").read_text())
    heldout = set(audit["test_multiple"]["scenes"]) | set(audit["test_multiple_unheard"]["scenes"])
    validation = set(audit["val_multiple"]["scenes"]) | set(audit["val_multiple_unheard"]["scenes"])
    unheard_sounds = set(audit["test_multiple_unheard"]["sound_classes"])
    return heldout, validation, unheard_sounds


def select_diagnostic_episodes():
    if (DIAG / "selection.json").is_file():
        return json.loads((DIAG / "selection.json").read_text())
    if not all(p.is_file() for p in (TRAIN_INDEX, EVAL_CONFIG, PARENT, FOUNDATION)):
        raise FileNotFoundError("Missing train source, evaluation config, or checkpoint")
    source_rows = [json.loads(line) for line in TRAIN_INDEX.read_text().splitlines() if line.strip()]
    heldout, validation, unheard_sounds = load_split_filters()
    candidates = [row for row in source_rows if row.get("saved") and
                  row["scene"] not in heldout | validation and row["sound"] not in unheard_sounds]
    pools = {name: [r for r in candidates if category(float(r["initial_distance"])) == name]
             for name in ("under_16m", "16_to_24m", "24_to_32m", "32m_plus")}
    if min(map(len, pools.values())) < PER_BIN:
        raise ValueError(f"Insufficient scene-disjoint diagnostic candidates: {Counter({k:len(v) for k,v in pools.items()})}")
    rng = random.Random(SEED)
    scene_counts, sound_counts = Counter(), Counter()
    selected = []
    for name in pools:
        selected.extend(balanced_sample(pools[name], PER_BIN, scene_counts, sound_counts, rng))
    selected.sort(key=lambda r: (category(float(r["initial_distance"])), r["scene"], str(r["episode_id"])))
    keys = {(r["scene"], str(r["episode_id"])) for r in selected}
    if len(keys) != 4 * PER_BIN:
        raise ValueError("Diagnostic selection contains duplicate episodes")
    records = []
    for scene in sorted({r["scene"] for r in selected}):
        paths = [p for p in TRAIN_CONTENT.glob(scene + ".json*") if p.name.endswith((".json", ".json.gz"))]
        if len(paths) != 1:
            raise ValueError(f"Expected one train_multiple content file for {scene}: {paths}")
        opener = gzip.open if paths[0].suffix == ".gz" else open
        with opener(paths[0], "rt") as handle:
            records.extend(ep for ep in json.load(handle)["episodes"]
                           if (Path(ep["scene_id"]).stem, str(ep["episode_id"])) in keys)
    if len(records) != len(keys):
        raise ValueError(f"Dataset records missing for selected train episodes: {len(records)}/{len(keys)}")
    DIAG.mkdir(parents=True, exist_ok=True)
    (DIAG / "dataset.json").write_text(json.dumps({"episodes": records}))
    selection = {
        "episodes": [{k: row[k] for k in ("scene", "episode_id", "sound", "initial_distance", "status")} for row in selected],
        "seed": SEED, "episodes_per_distance_bin": PER_BIN,
        "distance_bins": dict(Counter(category(float(r["initial_distance"])) for r in selected)),
        "scene_count": len({r["scene"] for r in selected}), "sound_count": len({r["sound"] for r in selected}),
        "source": str(TRAIN_INDEX), "source_sha256": sha(TRAIN_INDEX),
        "selection_split": "train_multiple", "heldout_and_validation_scenes_excluded": True,
        "unheard_test_sounds_excluded": True, "train_episode_overlap_with_heldout": 0,
        "policy_checkpoint": str(PARENT), "policy_sha256": sha(PARENT),
        "foundation_checkpoint": str(FOUNDATION), "foundation_sha256": sha(FOUNDATION),
        "evaluation_config": str(EVAL_CONFIG), "evaluation_config_sha256": sha(EVAL_CONFIG),
        "executed_policy": "pure_parent_only; expert queried for labels but never executed",
        "heldout_used_for_training": False,
    }
    atomic_json(DIAG / "selection.json", selection)
    shutil.copy2(EVAL_CONFIG, DIAG / "evaluation.yaml")
    return selection


def setup_env(data_dir):
    from .evaluate import load_episodes
    from .config.default import get_config
    from experiment import AVWANRLEnv
    config = get_config(str(data_dir / "evaluation.yaml"), opts=["TASK_CONFIG.DATASET.SPLIT", "train_multiple"])
    config.defrost()
    config.PREDICTION_INTERVAL = 5
    config.OFFLINE_RL.GAMMA = 0.99
    config.TASK_CONFIG.DATASET.DATA_PATH = str(data_dir / "dataset.json")
    config.TASK_CONFIG.TRAIN.OPEN = False
    config.TASK_CONFIG.COLLECT.OPEN = False
    config.TASK_CONFIG.TASK.MEASUREMENTS = [m for m in config.TASK_CONFIG.TASK.MEASUREMENTS if m != "TOP_DOWN_MAP"]
    config.freeze()
    dataset = load_episodes(config)
    episode_map = {(Path(ep.scene_id).stem, str(ep.episode_id)): ep for ep in dataset.episodes}
    return config, dataset, episode_map, AVWANRLEnv


def worker(kind, shard, shards):
    import numpy as np
    import torch
    from .offline.policy import OfflinePolicy
    from .environment.BFSExpert import NoReachableWaypointError
    from .collector.collect import save_episode

    torch.set_num_threads(2)
    torch.manual_seed(SEED + shard)
    state = torch.load(PARENT, map_location="cpu", weights_only=False)
    policy = OfflinePolicy.from_checkpoint(state, FOUNDATION, torch.device("cuda:0"))
    data_dir = DIAG
    config, dataset, episode_map, env_class = setup_env(data_dir)
    selected = json.loads((DIAG / "selection.json").read_text())["episodes"]
    if kind == "assist":
        selected = json.loads((CORR / "selection.json").read_text())["episodes"]
        data_dir = CORR
        config, dataset, episode_map, env_class = setup_env(data_dir)
    env = env_class(config=config, dataset=dataset)
    completed_this_process = 0
    try:
        for spec in selected[shard::shards]:
            scene, eid = spec["scene"], str(spec["episode_id"])
            metadata_path = data_dir / "metadata" / scene / f"{eid}.json"
            if metadata_path.is_file():
                continue
            episode = episode_map[(scene, eid)]
            seed = (SEED + int(hashlib.sha256(f"{scene}/{eid}".encode()).hexdigest()[:8], 16)) % (2**32 - 1)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.habitat_env.current_episode = episode
            env.seed(seed)
            obs = env.reset()
            policy.reset()
            actual_initial = float(env.get_info(obs)["distance_to_goal"])
            current_distance = actual_initial
            traces, done, blocked = [], False, None
            parent_positions = []
            while not done:
                before = current_distance
                policy_action = int(policy.act(obs, env.action_mask(obs)))
                try:
                    expert_action = int(env.get_expert_map_action())
                except NoReachableWaypointError:
                    expert_action = None
                if kind == "diagnostic":
                    action = policy_action
                    intervention, reason = False, "pure_parent_action"
                else:
                    if expert_action is None:
                        action, intervention, reason = policy_action, False, "expert_unavailable"
                    elif policy_action != expert_action:
                        action, intervention, reason = expert_action, True, "expert_policy_disagreement"
                    else:
                        action, intervention, reason = policy_action, False, "policy_agreement"
                obs, _, done, info = env.step(action)
                transition = env.last_transition
                after = float(info["distance_to_goal"])
                current_distance = after
                collisions = int(transition.get("collision_count", 0))
                position = np.asarray(env.habitat_env.sim.get_agent_state().position, dtype=float).tolist()
                traces.append(dict(action=int(action), policy_action=policy_action, expert_action=expert_action,
                    expert_control=intervention, intervention_reason=reason, distance_before=before,
                    distance_after=after, distance_progress=before-after,
                    collision_count=collisions, num_steps=int(info["num_steps"]),
                    reaching_waypoint=bool(info.get("reaching_waypoint", False)),
                    cant_reach_waypoint=bool(info.get("cant_reach_waypoint", False)),
                    waypoint_position=np.asarray(info.get("waypoint_position", position)).tolist(),
                    agent_position_after=position, done=bool(done), terminated=bool(info.get("terminated", False)),
                    truncated=bool(info.get("truncated", False))))
                parent_positions.append(position)
                if kind == "assist":
                    transition.update(policy_action=policy_action, expert_action=expert_action,
                        expert_control=bool(intervention),
                        expert_disagreement=expert_action is not None and policy_action != expert_action,
                        intervention_reason=reason, distance_before=before, distance_after=after,
                        distance_progress=before-after,
                        reaching_waypoint=bool(info.get("reaching_waypoint", False)),
                        cant_reach_waypoint=bool(info.get("cant_reach_waypoint", False)),
                        correction_progress=bool(info.get("reaching_waypoint", False) or after < before-1e-6),
                        correction_collision_count=collisions)
                    traces[-1]["raw_transition"] = transition
                if done:
                    final_info = info
            success = bool(final_info.get("success", False))
            spl = float(final_info.get("spl", 0.0))
            positions = [tuple(round(float(x), 2) for x in p) for p in parent_positions]
            unique_positions = len(set(positions))
            tcount = len(traces)
            primitive_steps = sum(t["num_steps"] for t in traces)
            meta = {
                "scene": scene, "episode_id": eid, "sound": spec.get("sound"),
                "split": "train_multiple", "distance_bin": category(actual_initial),
                "selection_initial_distance": float(spec["initial_distance"]),
                "initial_distance": actual_initial, "distance_source_delta": actual_initial-float(spec["initial_distance"]),
                "final_distance": float(final_info.get("distance_to_goal", traces[-1]["distance_after"])),
                "success": success, "spl": spl, "truncated": bool(final_info.get("truncated", False)),
                "blocked": blocked, "status": "success" if success else "timeout" if final_info.get("truncated") else "failure",
                "waypoint_transitions": tcount, "primitive_steps": primitive_steps,
                "false_stop_actions": sum(t["policy_action"] == STOP and not (success and t["done"]) for t in traces),
                "correct_stop_actions": sum(t["policy_action"] == STOP and success and t["done"] for t in traces),
                "policy_expert_disagreements": sum(t["expert_action"] is not None and t["policy_action"] != t["expert_action"] for t in traces),
                "primitive_collision_events": sum(t["collision_count"] for t in traces),
                "waypoint_actions_with_collision": sum(t["collision_count"] > 0 for t in traces),
                "unreachable_waypoints": sum(t["cant_reach_waypoint"] for t in traces),
                "nonpositive_progress_actions": sum(t["distance_progress"] <= 0 and not t["reaching_waypoint"] for t in traces),
                "positive_distance_progress_m": sum(max(t["distance_progress"], 0.) for t in traces),
                "repeated_endpoint_fraction": 1.-unique_positions/max(len(positions), 1),
                "unique_endpoint_positions": unique_positions,
                "policy_checkpoint": str(PARENT), "policy_sha256": sha(PARENT),
                "foundation_sha256": sha(FOUNDATION), "seed": seed,
                "executed_action_is_parent_only": kind == "diagnostic",
                "heldout_used_for_training": False,
                "interventions": sum(t["expert_control"] for t in traces),
                "useful_interventions": sum(t["expert_control"] and t["collision_count"] == 0 and
                    (t["distance_progress"] > 1e-6 or t["reaching_waypoint"] or
                     (success and t["action"] == STOP and t["done"])) for t in traces),
            }
            if kind == "diagnostic":
                meta["actions"] = [t["policy_action"] for t in traces]
                meta["trace"] = [{k:v for k,v in t.items() if k != "raw_transition"} for t in traces]
            else:
                raw_transitions = [t["raw_transition"] for t in traces]
                meta["return"] = sum(float(t["reward"]) for t in raw_transitions)
                meta["diagnostic_parent_spl"] = float(spec["parent_spl"])
                meta["diagnostic_parent_success"] = bool(spec["parent_success"])
                meta["diagnostic_parent_primitive_steps"] = int(spec["parent_primitive_steps"])
                path = CORR / "attempts" / "episodes" / scene / f"{eid}.pkl.gz"
                save_episode(path, raw_transitions, meta)
                meta.pop("trace", None)
            atomic_json(metadata_path, meta)
            print(json.dumps({k:v for k,v in meta.items() if k not in ("trace", "actions")}, ensure_ascii=False), flush=True)
            completed_this_process += 1
            if completed_this_process >= 16 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 >= 6500:
                break
    finally:
        env.close()


def finalize_diagnostic():
    selection = json.loads((DIAG / "selection.json").read_text())
    rows = [json.loads((DIAG / "metadata" / r["scene"] / f"{r['episode_id']}.json").read_text())
            for r in selection["episodes"]]
    def summarize(group):
        n = len(group)
        nonpositive = sum(t["distance_progress"] <= 0 and not t["reaching_waypoint"] and
                          not (r["success"] and t["policy_action"] == STOP and t["done"])
                          for r in group for t in r.get("trace", []))
        return {"episodes": n, "success_rate": sum(r["success"] for r in group)/max(n,1),
            "spl": sum(r["spl"] for r in group)/max(n,1), "mean_primitive_steps": sum(r["primitive_steps"] for r in group)/max(n,1),
            "false_stop_episodes": sum(r["false_stop_actions"] > 0 for r in group),
            "timeouts": sum(r["status"] == "timeout" for r in group),
            "primitive_collision_events": sum(r["primitive_collision_events"] for r in group),
            "unreachable_waypoints": sum(r["unreachable_waypoints"] for r in group),
            "nonpositive_progress_actions": nonpositive,
            "mean_repeated_endpoint_fraction": sum(r["repeated_endpoint_fraction"] for r in group)/max(n,1),
            "policy_expert_disagreements": sum(r["policy_expert_disagreements"] for r in group)}
    result = {"status":"completed", "episodes":len(rows), "selection_sha256":sha(DIAG/"selection.json"),
        "policy_sha256":selection["policy_sha256"], "training_only":True,
        "executed_actions_parent_only":all(r["executed_action_is_parent_only"] for r in rows),
        "distance_selection_mismatch_episodes":sum(abs(r["distance_source_delta"])>0.5 for r in rows),
        "overall":summarize(rows), "by_distance_bin":{k:summarize([r for r in rows if r["distance_bin"]==k])
            for k in ("under_16m","16_to_24m","24_to_32m","32m_plus")}}
    atomic_json(DIAG/"manifest.json", result)
    with (DIAG/"episodes.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False)+"\n")
    return rows, result


def select_corrections(rows):
    # A failure means unsuccessful execution, or very inefficient long-range
    # success where the parent's SPL is below the predeclared 0.20 threshold.
    candidates = [r for r in rows if (not r["success"] or
                  (r["initial_distance"] >= 24 and r["spl"] < 0.20))]
    if not candidates:
        raise ValueError("Pure-parent diagnostic found no eligible correction candidates")
    CORR.mkdir(parents=True, exist_ok=True)
    (CORR / "dataset.json").write_text((DIAG / "dataset.json").read_text())
    shutil.copy2(DIAG / "evaluation.yaml", CORR / "evaluation.yaml")
    specs = []
    for row in candidates:
        source = next(x for x in json.loads((DIAG/"selection.json").read_text())["episodes"]
                      if x["scene"] == row["scene"] and str(x["episode_id"]) == str(row["episode_id"]))
        specs.append(dict(source, parent_success=row["success"], parent_spl=row["spl"],
                          parent_primitive_steps=row["primitive_steps"], distance_bin=row["distance_bin"]))
    specs.sort(key=lambda r:(r["distance_bin"],r["scene"],str(r["episode_id"])))
    selection = {"episodes":specs,"count":len(specs),"seed":SEED,
        "selection_rule":"pure-parent failure OR initial_distance>=24m and parent SPL<0.20",
        "source_diagnostic_manifest_sha256":sha(DIAG/"manifest.json"),
        "parent_checkpoint":str(PARENT),"policy_sha256":sha(PARENT),
        "training_split_only":True,"heldout_used_for_training":False,
        "assist_policy":"on-policy DAgger replay: execute BFS expert action iff it disagrees with parent; retain actual transitions"}
    atomic_json(CORR/"selection.json",selection)
    return selection


def finalize_assist():
    selection=json.loads((CORR/"selection.json").read_text())
    rows=[]
    for spec in selection["episodes"]:
        path=CORR/"metadata"/spec["scene"]/f"{spec['episode_id']}.json"
        if not path.is_file(): raise FileNotFoundError(path)
        row=json.loads(path.read_text())
        # Worker metadata uses the shorter names; normalize the report schema.
        row["expert_interventions"]=int(row.get("expert_interventions",row["interventions"]))
        rows.append(row)
    improved=[r for r in rows if r["success"] and r["spl"] >= r["diagnostic_parent_spl"] + 0.02]
    productive=[r for r in improved if float(r["initial_distance"]) >= 24.0]
    productive_keys={(r["scene"],str(r["episode_id"])) for r in productive}
    reason_counts, productive_reason_counts = Counter(), Counter()
    bc_eligible, productive_bc_eligible = 0, 0
    for row in rows:
        raw_path=CORR/"attempts"/"episodes"/row["scene"]/f"{row['episode_id']}.pkl.gz"
        with gzip.open(raw_path,"rb") as handle:
            raw=pickle.load(handle)
        for t in raw["transitions"]:
            if not bool(t.get("expert_control")):
                continue
            reason=t.get("intervention_reason","unknown")
            reason_counts[reason]+=1
            useful=bool(t.get("correction_progress")) or (
                bool(row["success"]) and int(t["action"])==STOP and bool(t["done"]))
            allowed=(useful and int(t.get("collision_count",0))==0 and
                     int(t.get("correction_collision_count",0))==0)
            if allowed:
                bc_eligible+=1
            if (row["scene"],str(row["episode_id"])) in productive_keys:
                productive_reason_counts[reason]+=1
                if allowed:
                    productive_bc_eligible+=1
    train_root=CORR/"training_source"
    train_root.mkdir(parents=True,exist_ok=True)
    for row in productive:
        source=CORR/"attempts"/"episodes"/row["scene"]/f"{row['episode_id']}.pkl.gz"
        target=train_root/"episodes"/row["scene"]/source.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copy2(source,target)
    training_rows=[]
    for row in productive:
        training_rows.append({k:row[k] for k in ("scene","episode_id","sound","split","initial_distance","success","spl","primitive_steps","diagnostic_parent_spl","diagnostic_parent_success","diagnostic_parent_primitive_steps","expert_interventions","useful_interventions")})
    with (train_root/"episodes.jsonl").open("w") as handle:
        for row in training_rows: handle.write(json.dumps(row,ensure_ascii=False)+"\n")
    diag_selection=json.loads((DIAG/"selection.json").read_text())
    productive_keys={(r["scene"],str(r["episode_id"])) for r in productive}
    train_selection={"episodes":[r for r in selection["episodes"]
                      if (r["scene"],str(r["episode_id"])) in productive_keys],
        "count":len(productive),"seed":SEED,
        "selection_split":"train_multiple","source":diag_selection["source"],
        "source_sha256":diag_selection["source_sha256"],
        "policy_checkpoint":str(PARENT),"policy_sha256":sha(PARENT),
        "training_only":True,"heldout_used_for_training":False,
        "selection_rule":"productive subset of pure-parent failure correction attempts at actual reset distance >=24m; assisted success and SPL gain >=0.02"}
    atomic_json(train_root/"selection.json",train_selection)
    def summarize(group):
        n=len(group)
        return {"episodes":n,"successes":sum(r["success"] for r in group),
            "success_rate":sum(r["success"] for r in group)/max(n,1),
            "assisted_spl":sum(r["spl"] for r in group)/max(n,1),
            "parent_spl":sum(r["diagnostic_parent_spl"] for r in group)/max(n,1),
            "mean_spl_gain":sum(r["spl"]-r["diagnostic_parent_spl"] for r in group)/max(n,1),
            "mean_primitive_steps":sum(r["primitive_steps"] for r in group)/max(n,1),
            "expert_interventions":sum(r["expert_interventions"] for r in group),
            "useful_executed_expert_actions":sum(r["useful_interventions"] for r in group)}
    result={"status":"completed","attempted_episodes":len(rows),"productive_episodes":len(productive),
        "improved_assisted_episodes":len(improved),
        "productive_short_distance_episodes_excluded_from_training":len(improved)-len(productive),
        "productivity_rule":"assisted success and SPL >= pure-parent SPL + 0.02",
        "training_pool_rule":"productive assisted episode with reset-measured initial distance >=24m",
        "by_distance_bin":{k:summarize([r for r in rows if r["distance_bin"]==k])
            for k in ("under_16m","16_to_24m","24_to_32m","32m_plus")},
        "overall_attempts":summarize(rows),"productive_training_pool":summarize(productive),
        "executed_expert_action_reasons":dict(reason_counts),
        "productive_expert_action_reasons":dict(productive_reason_counts),
        "bc_eligible_executed_expert_transitions_all_attempts":bc_eligible,
        "bc_eligible_executed_expert_transitions_productive_pool":productive_bc_eligible,
        "training_source":str(train_root),"training_only":True,
        "actual_action_reward_next_state_preserved":True,
        "heldout_used_for_training":False}
    atomic_json(CORR/"manifest.json",result)
    atomic_json(train_root/"manifest.json",{
        "status":"completed","episodes":len(productive),"training_only":True,
        "actual_action_reward_next_state_preserved":True,
        "source_sha256":diag_selection["source_sha256"],
        "selection_sha256":sha(train_root/"selection.json"),
        "policy_sha256":sha(PARENT),"foundation_sha256":sha(FOUNDATION),
        "productive_rule":train_selection["selection_rule"],
        "transitions":sum(r["waypoint_transitions"] for r in productive)})
    with (CORR/"episodes.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row,ensure_ascii=False)+"\n")
    return result


def run_collect(kind):
    if kind=="diagnostic":
        selection=select_diagnostic_episodes(); data_dir=DIAG; selected=selection["episodes"]
    else:
        rows,_=finalize_diagnostic(); selection=select_corrections(rows); data_dir=CORR; selected=selection["episodes"]
    shards=2
    def run_shard(shard):
        subset=selected[shard::shards]
        while True:
            count=sum((data_dir/"metadata"/r["scene"]/f"{r['episode_id']}.json").exists() for r in subset)
            if count==len(subset): return
            cmd=[PYTHON,"-m","experiment.avwan.collect_parent_spl_diagnostics","--worker",kind,str(shard),str(shards)]
            log=data_dir/f"worker_{shard}.log"
            with log.open("a") as handle:
                subprocess.run(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True,
                    env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",MAGNUM_LOG="quiet",HABITAT_SIM_LOG="quiet",PYTHONUNBUFFERED="1"))
            now=sum((data_dir/"metadata"/r["scene"]/f"{r['episode_id']}.json").exists() for r in subset)
            if now==count: raise RuntimeError(f"No progress in {kind} worker {shard}; see {log}")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=shards) as pool:
        list(pool.map(run_shard,range(shards)))
    return finalize_diagnostic() if kind=="diagnostic" else finalize_assist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect",choices=("diagnostic","assist"))
    parser.add_argument("--worker",nargs=3,metavar=("KIND","SHARD","SHARDS"))
    parser.add_argument("--finalize",choices=("diagnostic","assist"))
    args=parser.parse_args()
    if args.worker:
        kind,shard,shards=args.worker; worker(kind,int(shard),int(shards))
    elif args.collect:
        print(json.dumps(run_collect(args.collect),ensure_ascii=False,indent=2))
    elif args.finalize:
        result=finalize_diagnostic()[1] if args.finalize=="diagnostic" else finalize_assist()
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:
        parser.error("choose --collect, --worker, or --finalize")


if __name__=="__main__": main()
