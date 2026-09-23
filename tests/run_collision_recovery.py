"""Replay one expert episode and verify collision edges are never retried.

Run with the ss environment and GPU/EGL access. Defaults to VVfe2KiqLaN/548.
Saves per-primitive collision evidence and Habitat's standard TopDownMap image.
"""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-avwan-test")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="VVfe2KiqLaN")
    parser.add_argument("--episode", default="548")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--prediction-interval", type=int)
    parser.add_argument("--require-success", action="store_true",
                        help="Also require arrival; otherwise only enforce collision recovery invariants")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/results/collision_recovery")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    import numpy as np
    from PIL import Image
    from habitat.utils.visualizations import maps
    from experiment import AVWANRLEnv, get_avwan_config
    from experiment.avwan.environment.BFSExpert import NoReachableWaypointError
    from soundspaces.datasets.audionav_dataset import AudioNavDataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    config = get_avwan_config(opts=["TASK_CONFIG.DATASET.CONTENT_SCENES", [args.scene]])
    config.defrost()
    if args.prediction_interval is not None:
        if args.prediction_interval < 1:
            parser.error('--prediction-interval must be positive')
        config.PREDICTION_INTERVAL = args.prediction_interval
    config.TASK_CONFIG.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = False
    config.freeze()
    source = ROOT / f"data/datasets/audionav/mp3d/v1/train_telephone/content/{args.scene}.json.gz"
    with gzip.open(source, "rt") as handle:
        raw = json.load(handle)
    raw["episodes"] = [ep for ep in raw["episodes"] if str(ep["episode_id"]) == args.episode]
    assert len(raw["episodes"]) == 1
    dataset = AudioNavDataset()
    dataset._config = config.TASK_CONFIG.DATASET
    dataset.from_json(json.dumps(raw), scenes_dir=config.TASK_CONFIG.DATASET.SCENES_DIR)
    env = AVWANRLEnv(config, dataset)
    primitive_steps, waypoints = [], []
    collisions = Counter()
    try:
        env.seed(args.seed)
        obs = env.reset()
        planner = env.planner
        sim = env.habitat_env.sim
        topdown = env.habitat_env.task.measurements.measures["top_down_map"]
        if not hasattr(env.current_episode, "goal"):
            topdown._draw_goals_positions(env.current_episode)
            topdown._draw_shortest_path(env.current_episode, sim.get_agent_state().position)
        original_plan = planner.plan
        original_update = planner.update_map_and_graph
        pending = None

        def plan(observation, goal, stop, distribution=None):
            nonlocal pending
            action = int(original_plan(observation, goal, stop, distribution))
            _, _, x, y, _ = planner.mapper.get_maps_and_agent_pose()
            a, b = planner._map_index_to_graph_nodes(
                [(x, y), planner.mapper.get_adjacent_point_coordinates()])
            pending = dict(waypoint=len(waypoints) + 1, action=action,
                           edge=planner._edge_key(a, b), depth_probe=planner.is_depth_probe(goal),
                           position=sim.get_agent_state().position.tolist())
            return action

        def update(observation):
            nonlocal pending
            original_update(observation)
            if pending is None:
                return
            row = pending
            row["collision"] = bool(np.asarray(observation["collision"]).flat[0])
            row["position_after"] = sim.get_agent_state().position.tolist()
            row["edge_present_after"] = planner._graph.has_edge(*row["edge"])
            if row["collision"] and row["action"] == 1:
                collisions[row["edge"]] += 1
            primitive_steps.append(row)
            pending = None

        planner.plan, planner.update_map_and_graph = plan, update
        done, blocked = False, None
        info = env.get_info(obs)
        while not done:
            try:
                action = env.get_expert_map_action()
            except NoReachableWaypointError as error:
                blocked = str(error)
                break
            assert env.action_mask(obs)[action]
            obs, reward, done, info = env.step(action)
            waypoints.append(dict(action=action, reward=float(reward),
                                  distance=float(info["distance_to_goal"]),
                                  reached=bool(info["reaching_waypoint"]),
                                  blocked=bool(info["cant_reach_waypoint"]),
                                  expert_decision=info.get("expert_decision"),
                                  low_level_actions=info["low_level_actions"]))
        collision_rows = [r for r in primitive_steps if r["collision"] and r["action"] == 1]
        source_map_state = None
        state = sim.get_agent_state()
        target = np.asarray(env.current_episode.goals[0].position)
        for action in range(config.TASK_CONFIG.TASK.ACTION_MAP.MAP_SIZE ** 2):
            world = np.asarray(env.bfs_expert.action_to_world_position(action, state))
            if np.linalg.norm(world[[0, 2]] - target[[0, 2]]) < 1e-4:
                x, y = planner.get_map_coordinates(action)
                gm = planner.mapper.get_maps_and_agent_pose()[0]
                node = planner._map_index_to_graph_nodes([(x, y)])[0]
                source_map_state = dict(cell=[int(x), int(y)], occupancy=float(gm[y, x, 0]),
                                        explored=float(gm[y, x, 1]), node_present=node in planner._graph,
                                        reachable=bool(planner.check_navigability((x, y))))
                break
        summary = dict(scene=args.scene, episode_id=args.episode, seed=args.seed,
                       prediction_interval=config.PREDICTION_INTERVAL,
                       success=bool(info["success"]), final_distance=float(info["distance_to_goal"]),
                       spl=float(info["spl"]), primitive_steps=len(primitive_steps), blocked=blocked,
                       forward_attempts=sum(r["action"] == 1 for r in primitive_steps),
                       depth_probe_attempts=sum(r['action'] == 1 and r['depth_probe'] for r in primitive_steps),
                       collisions=len(collision_rows), max_collisions_per_edge=max(collisions.values(), default=0),
                       collision_edges_retained=sum(r["edge_present_after"] for r in collision_rows),
                       source_map_state=source_map_state)
        stem = output / f"{args.scene}_{args.episode}"
        stem.with_suffix(".json").write_text(json.dumps(
            dict(summary=summary, waypoints=waypoints, steps=primitive_steps), indent=2) + "\n")
        Image.fromarray(maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], output_height=800)).save(stem.with_suffix(".png"))
        print(json.dumps(summary, indent=2), flush=True)
        assert summary["collision_edges_retained"] == 0, summary
        assert summary["max_collisions_per_edge"] <= 1, summary
        if args.require_success:
            assert summary["success"] and summary["final_distance"] == 0, summary
        env.reset()
        assert not planner._collision_edges
        assert not planner._verified_edges and not planner._verified_nodes
        print("PASS: collision edge persistence and episode reset", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
