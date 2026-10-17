"""Collect train-only long-distance DAgger rollouts with executable expert corrections."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "result/2026-10-15-avwan-hard-corrections"
DATA = OUT / "collection"
TRAIN_INDEX = Path("/data/Getuanhui/avwan_multiple_greedy_every_1000/episodes.jsonl")
TRAIN_CONTENT = ROOT / "data/datasets/audionav/mp3d/v1/train_multiple/content"
STUDY = ROOT / "result/10-11/total/2026-10-03-unheard-generalization-study"
EVALUATION_CONFIG = STUDY / "evaluation.yaml"
PARENT = ROOT / "result/2026-10-14-avwan-diverse-retraining/offline/seed_20261014_deverse_hard_longrange/training/best_policy.pt"
FOUNDATION = ROOT / "result/2026-10-13-staged-grl-reproduction/foundation/action_stage/best.pt"
PYTHON = "/home/getuanhui/anaconda3/envs/ss/bin/python"
SEED = 20261015
COUNT = 250
MIN_DISTANCE = 24.0
STOP = 40


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Collector workers update status concurrently; give each atomic writer its
    # own temporary file so one thread cannot replace another thread's temp.
    temp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def select_balanced(pool, wanted, scene_counts, sound_counts, rng, scene_cap=16):
    remaining = list(pool)
    selected = []
    tie = {(r["scene"], str(r["episode_id"])): rng.random() for r in remaining}
    while len(selected) < wanted:
        eligible = [r for r in remaining if scene_counts[r["scene"]] < scene_cap]
        if not eligible:
            raise ValueError(f"Insufficient episodes under per-scene cap {scene_cap}")
        row = min(eligible, key=lambda r: (scene_counts[r["scene"]], sound_counts[r["sound"]],
                                           tie[(r["scene"], str(r["episode_id"]))],
                                           str(r["episode_id"])))
        selected.append(row)
        remaining.remove(row)
        scene_counts[row["scene"]] += 1
        sound_counts[row["sound"]] += 1
    return selected


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    if (DATA / "selection.json").exists() and (DATA / "dataset.json").exists():
        return json.loads((DATA / "selection.json").read_text())
    if not all(path.is_file() for path in (TRAIN_INDEX, EVALUATION_CONFIG, PARENT, FOUNDATION)):
        raise FileNotFoundError("Training index, evaluation config, or parent checkpoints are missing")
    source_rows = [json.loads(line) for line in TRAIN_INDEX.read_text().splitlines() if line.strip()]
    train_keys = {(r["scene"], str(r["episode_id"])) for r in source_rows if r.get("saved")}
    candidates = [r for r in source_rows if r.get("saved") and r.get("initial_distance", -1) >= MIN_DISTANCE]
    bands = {
        "24_to_32m": [r for r in candidates if 24 <= r["initial_distance"] < 32],
        "32m_plus": [r for r in candidates if r["initial_distance"] >= 32],
    }
    if min(map(len, bands.values())) < COUNT // 2:
        raise ValueError(f"Not enough train-only long-distance candidates: {Counter({k: len(v) for k, v in bands.items()})}")
    audit_path = STUDY / "split_audit.json"
    audit = json.loads(audit_path.read_text())
    test_scenes = set(audit["test_multiple"]["scenes"]) | set(audit["test_multiple_unheard"]["scenes"])
    validation_scenes = set(audit["val_multiple"]["scenes"]) | set(audit["val_multiple_unheard"]["scenes"])
    if ({r["scene"] for r in candidates} & test_scenes) or ({r["scene"] for r in candidates} & validation_scenes):
        # Candidate pool may include other training episodes in a held-out scene;
        # keep the experiment scene-disjoint, matching the existing split audit.
        candidates = [r for r in candidates if r["scene"] not in test_scenes | validation_scenes]
        bands = {"24_to_32m": [r for r in candidates if 24 <= r["initial_distance"] < 32],
                 "32m_plus": [r for r in candidates if r["initial_distance"] >= 32]}
    if not candidates:
        raise ValueError("No long-distance train candidates remain after the split audit")
    unheard_sounds = set(audit["test_multiple_unheard"]["sound_classes"])
    if any(r["sound"] in unheard_sounds for r in candidates):
        raise ValueError("Unheard test sound class appeared in the training candidate pool")

    rng = random.Random(SEED)
    scene_counts, sound_counts = Counter(), Counter()
    low_count = COUNT // 2
    high_count = COUNT - low_count
    selected = select_balanced(bands["24_to_32m"], low_count, scene_counts, sound_counts, rng)
    selected += select_balanced(bands["32m_plus"], high_count, scene_counts, sound_counts, rng)
    selected = sorted(selected, key=lambda r: (r["scene"], str(r["episode_id"])))
    selected_keys = {(r["scene"], str(r["episode_id"])) for r in selected}
    if len(selected_keys) != COUNT or not selected_keys <= train_keys:
        raise ValueError("Selection is not a unique subset of the train-only source")

    records = []
    for scene in sorted({r["scene"] for r in selected}):
        paths = [p for p in TRAIN_CONTENT.glob(scene + ".json*") if p.name.endswith((".json", ".json.gz"))]
        if len(paths) != 1:
            raise ValueError(f"Expected one train dataset file for {scene}: {paths}")
        path = paths[0]
        with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
            episodes = json.load(f)["episodes"]
        records.extend(ep for ep in episodes
                       if (Path(ep["scene_id"]).stem, str(ep["episode_id"])) in selected_keys)
    if len(records) != COUNT:
        raise ValueError(f"Selected training episodes missing from train_multiple: {len(records)}/{COUNT}")
    write(DATA / "dataset.json", {"episodes": records})
    selection = {
        "episodes": [{key: row[key] for key in ("scene", "episode_id", "sound", "initial_distance", "status")}
                     for row in selected],
        "seed": SEED,
        "count": COUNT,
        "distance_threshold_m": MIN_DISTANCE,
        "source": str(TRAIN_INDEX),
        "source_sha256": sha(TRAIN_INDEX),
        "method": "250 train_multiple episodes at >=24m; 125 from [24,32), 125 from [32,+); greedy balancing by scene and sound, deterministic seed",
        "selection_split": "train_multiple",
        "train_episode_overlap_with_heldout": 0,
        "unheard_test_sounds_in_selection": 0,
        "scene_count": len({r["scene"] for r in selected}),
        "sound_count": len({r["sound"] for r in selected}),
        "distance_bins": dict(Counter("24_to_32m" if r["initial_distance"] < 32 else "32m_plus" for r in selected)),
        "scenes": dict(Counter(r["scene"] for r in selected)),
        "sounds": dict(Counter(r["sound"] for r in selected)),
        "base_outcomes_for_selected_episodes": dict(Counter(r["status"] for r in selected)),
        "collection_policy": str(PARENT),
        "collection_policy_sha256": sha(PARENT),
        "foundation": str(FOUNDATION),
        "foundation_sha256": sha(FOUNDATION),
        "evaluation_config": str(EVALUATION_CONFIG),
        "evaluation_config_sha256": sha(EVALUATION_CONFIG),
        "expert_intervention_probability_on_disagreement": 0.5,
        "expert_for_predicted_false_stop": True,
        "expert_for_correct_stop": True,
        "expert_after_collision_or_unreachable_waypoint": True,
        "heldout_used_for_training": False,
    }
    write(DATA / "selection.json", selection)
    shutil.copy2(EVALUATION_CONFIG, DATA / "evaluation.yaml")
    write(DATA / "status.json", {"status": "prepared", "selected": COUNT})
    return selection


def worker(shard, shards):
    import numpy as np
    import torch
    from .evaluate import load_episodes
    from .config.default import get_config
    from .collector.collect import save_episode
    from .offline.policy import OfflinePolicy
    from .environment.BFSExpert import NoReachableWaypointError
    from experiment import AVWANRLEnv

    torch.set_num_threads(2)
    torch.manual_seed(SEED + shard)
    state = torch.load(PARENT, map_location="cpu", weights_only=False)
    policy = OfflinePolicy.from_checkpoint(state, FOUNDATION, torch.device("cuda:0"))
    config = get_config(str(DATA / "evaluation.yaml"), opts=["TASK_CONFIG.DATASET.SPLIT", "train_multiple"])
    config.defrost()
    config.PREDICTION_INTERVAL = 5
    config.OFFLINE_RL.GAMMA = 0.99
    config.TASK_CONFIG.DATASET.DATA_PATH = str(DATA / "dataset.json")
    config.TASK_CONFIG.TRAIN.OPEN = False
    config.TASK_CONFIG.COLLECT.OPEN = False
    config.TASK_CONFIG.TASK.MEASUREMENTS = [m for m in config.TASK_CONFIG.TASK.MEASUREMENTS if m != "TOP_DOWN_MAP"]
    config.freeze()
    dataset = load_episodes(config)
    episode_map = {(Path(ep.scene_id).stem, str(ep.episode_id)): ep for ep in dataset.episodes}
    selected = json.loads((DATA / "selection.json").read_text())["episodes"]
    selection = selected[shard::shards]
    env = AVWANRLEnv(config, dataset)
    completed_this_process = 0
    try:
        for index, spec in enumerate(selection):
            scene, episode_id = spec["scene"], str(spec["episode_id"])
            output = DATA / "episodes" / scene / (episode_id + ".pkl.gz")
            metadata_path = DATA / "metadata" / scene / (episode_id + ".json")
            if output.exists() and metadata_path.exists():
                continue
            episode = episode_map[(scene, episode_id)]
            episode_seed = (SEED + int(hashlib.sha256(f"{scene}/{episode_id}".encode()).hexdigest()[:8], 16)) % (2 ** 32 - 1)
            rng = np.random.default_rng(episode_seed)
            torch.manual_seed(episode_seed)
            env.habitat_env.current_episode = episode
            env.seed(episode_seed)
            observation = env.reset()
            policy.reset()
            initial_distance = float(env.get_info(observation)["distance_to_goal"])
            transitions, done, blocked, info = [], False, None, {}
            counters = Counter()
            previous_collision, previous_unreachable = False, False
            while not done:
                before = float(env.get_info(observation)["distance_to_goal"])
                mask = env.action_mask(observation)
                policy_action = int(policy.act(observation, mask))
                try:
                    expert_action = int(env.get_expert_map_action())
                except NoReachableWaypointError as error:
                    blocked = str(error)
                    if transitions:
                        transitions[-1].update(done=True, truncated=True, collector_truncated=True)
                    break
                disagreement = policy_action != expert_action
                if expert_action == STOP:
                    use_expert, reason = True, "expert_correct_stop"
                elif policy_action == STOP:
                    use_expert, reason = True, "correct_false_stop"
                elif previous_collision:
                    use_expert, reason = True, "recovery_after_collision"
                elif previous_unreachable:
                    use_expert, reason = True, "recovery_after_unreachable_waypoint"
                elif disagreement and rng.random() < 0.5:
                    use_expert, reason = True, "sampled_expert_disagreement"
                else:
                    use_expert, reason = False, "policy_action"
                action = expert_action if use_expert else policy_action
                observation, _, done, info = env.step(action)
                transition = env.last_transition
                after = float(info["distance_to_goal"])
                collisions = int(transition.get("collision_count", 0))
                useful = bool(info.get("reaching_waypoint", False) or after < before - 1e-6)
                transition.update(
                    policy_action=policy_action,
                    expert_action=expert_action,
                    expert_control=bool(use_expert),
                    expert_disagreement=bool(disagreement),
                    intervention_reason=reason,
                    distance_before=before,
                    distance_after=after,
                    distance_progress=before - after,
                    reaching_waypoint=bool(info.get("reaching_waypoint", False)),
                    cant_reach_waypoint=bool(info.get("cant_reach_waypoint", False)),
                    correction_progress=useful,
                    correction_collision_count=collisions,
                )
                counters["transitions"] += 1
                counters["policy_expert_disagreements"] += int(disagreement)
                counters["expert_interventions"] += int(use_expert)
                counters["useful_interventions"] += int(use_expert and useful and collisions == 0)
                counters["collision_events"] += collisions
                counters["unreachable_waypoints"] += int(info.get("cant_reach_waypoint", False))
                counters["negative_distance_progress"] += int(after > before + 1e-6)
                transitions.append(transition)
                previous_collision = collisions > 0
                previous_unreachable = bool(info.get("cant_reach_waypoint", False))
            if not transitions:
                raise RuntimeError(f"No transitions for {scene}/{episode_id}: {blocked}")
            final_distance = float(info.get("distance_to_goal", transitions[-1].get("distance_after", initial_distance)))
            success = bool(info.get("success", False))
            truncated = bool(transitions[-1].get("truncated", False))
            status = "success" if success else "blocked" if blocked else "timeout" if truncated else "failure"
            metadata = {
                "scene": scene, "episode_id": episode_id, "sound": spec["sound"],
                "split": "train_multiple", "selection_initial_distance": spec["initial_distance"],
                "initial_distance": initial_distance, "final_distance": final_distance,
                "success": success, "truncated": truncated, "status": status, "blocked": blocked,
                "policy_checkpoint": str(PARENT), "policy_sha256": sha(PARENT),
                "foundation_sha256": sha(FOUNDATION), "seed": episode_seed,
                "prediction_interval": 5, **dict(counters),
                "actual_action_reward_next_state_preserved": True,
                "heldout_used_for_training": False,
            }
            save_episode(output, transitions, metadata)
            write(metadata_path, metadata)
            print(json.dumps(metadata, ensure_ascii=False), flush=True)
            completed_this_process += 1
            if completed_this_process >= 8 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 >= 6500:
                break
    finally:
        env.close()


def finalize(selection):
    expected = selection["episodes"]
    rows = []
    for spec in expected:
        path = DATA / "metadata" / spec["scene"] / (str(spec["episode_id"]) + ".json")
        if not path.is_file():
            raise FileNotFoundError(f"Missing collected episode metadata: {path}")
        row = json.loads(path.read_text())
        if (row.get("scene"), str(row.get("episode_id"))) != (spec["scene"], str(spec["episode_id"])):
            raise ValueError(f"Collected metadata key mismatch: {path}")
        episode_path = DATA / "episodes" / spec["scene"] / (str(spec["episode_id"]) + ".pkl.gz")
        if not episode_path.is_file():
            raise FileNotFoundError(f"Missing collected episode: {episode_path}")
        rows.append(row)
    keys = {(row["scene"], str(row["episode_id"])) for row in rows}
    if len(rows) != len(expected) or len(keys) != len(rows):
        raise ValueError(f"Collection output count mismatch: {len(rows)}/{len(expected)} unique episodes")
    (DATA / "episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    transitions = sum(row["transitions"] for row in rows)
    totals = {
        "status": "completed", "episodes": len(rows), "transitions": transitions,
        "successes": sum(row["success"] for row in rows),
        "timeouts": sum(row["status"] == "timeout" for row in rows),
        "blocked": sum(row["status"] == "blocked" for row in rows),
        "policy_expert_disagreements": sum(row["policy_expert_disagreements"] for row in rows),
        "expert_interventions": sum(row["expert_interventions"] for row in rows),
        "useful_interventions": sum(row["useful_interventions"] for row in rows),
        "collision_events": sum(row["collision_events"] for row in rows),
        "unreachable_waypoints": sum(row["unreachable_waypoints"] for row in rows),
        "negative_distance_progress": sum(row["negative_distance_progress"] for row in rows),
        "source_sha256": selection["source_sha256"],
        "selection_sha256": sha(DATA / "selection.json"),
        "policy_sha256": selection["collection_policy_sha256"],
        "foundation_sha256": selection["foundation_sha256"],
        "training_only": True,
        "actual_action_reward_next_state_preserved": all(
            row["actual_action_reward_next_state_preserved"] for row in rows),
    }
    write(DATA / "manifest.json", totals)
    write(DATA / "status.json", totals)
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    return totals


def collect():
    selection = prepare()
    shards = 2
    expected = selection["episodes"]
    DATA.mkdir(parents=True, exist_ok=True)
    write(DATA / "status.json", {"status": "collecting", "completed": 0, "planned": expected,
                                  "selection_sha256": sha(DATA / "selection.json")})

    def run_shard(shard):
        subset = selection["episodes"][shard::shards]
        previous = -1
        while True:
            count = sum((DATA / "metadata" / r["scene"] / (str(r["episode_id"]) + ".json")).exists()
                        for r in subset)
            total = sum(1 for _ in (DATA / "metadata").rglob("*.json"))
            write(DATA / "status.json", {"status": "collecting", "completed": total,
                                          "planned": expected,
                                          "selection_sha256": sha(DATA / "selection.json")})
            if count == len(subset):
                return
            if count <= previous:
                raise RuntimeError(f"Collection worker {shard} made no progress")
            previous = count
            cmd = [PYTHON, "-m", "experiment.avwan.collect_hard_corrections", "--worker",
                   str(shard), "--shards", str(shards)]
            log = DATA / f"worker_{shard}.log"
            with log.open("a") as f:
                subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True,
                               env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                                        MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet", PYTHONUNBUFFERED="1"))

    with ThreadPoolExecutor(max_workers=shards) as pool:
        list(pool.map(run_shard, range(shards)))
    return finalize(selection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--finalize-only", action="store_true",
                        help="Build the collection index and manifest from completed episode files")
    args = parser.parse_args()
    if args.finalize_only:
        finalize(prepare())
    elif args.worker is None:
        collect()
    else:
        worker(args.worker, args.shards)
