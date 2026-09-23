"""Replay ground-truth routes for VVfe2KiqLaN/548, without a learned planner.

Run: conda run -n ss python tests/run_shortest_path_548.py

SoundSpaces uses its discrete graph oracle. Habitat-Sim uses
make_greedy_follower().find_path(goal) on the NavMesh, with 0.25 m forward
steps and 10 degree turns. The latter greedily follows a geodesic; it is
not a proof of a minimum-action sequence. Both action lists are replayed.
Images use Habitat's TopDownMap measure and its standard colorize/agent renderer.
"""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-avwan-test")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
SCENE, EPISODE_ID, SEED = "VVfe2KiqLaN", "548", 20260924


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def summarize(actions, positions):
    import numpy as np

    return dict(action_count=len(actions), action_counts=dict(Counter(actions)),
                executed_length_m=float(np.linalg.norm(
                    np.diff(np.asarray(positions), axis=0), axis=1).sum()))


def run_soundspaces(config, dataset):
    from copy import deepcopy
    import habitat
    import numpy as np
    from habitat.sims.habitat_simulator.actions import HabitatSimActions

    # Primitive Habitat Env, so no local Planner/BFSExpert or waypoint wrapper.
    task_config = config.TASK_CONFIG.clone()
    task_config.defrost()
    task_config.TASK.SENSORS = ["COLLISION"]
    task_config.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "TOP_DOWN_MAP"]
    task_config.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = False
    task_config.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = task_config.ENVIRONMENT.MAX_EPISODE_STEPS
    task_config.freeze()
    names = {int(getattr(HabitatSimActions, name)): name for name in
             ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")}
    with habitat.Env(config=task_config, dataset=dataset) as env:
        env.seed(SEED)
        env.reset()
        sim = env.sim
        start = sim.get_agent_state().position.copy()
        goal = np.asarray(env.current_episode.goals[0].position)
        topdown = env.task.measurements.measures["top_down_map"]
        # This Habitat checkout's reset checks `episode.goal` (singular),
        # although AudioNav episodes expose `goals`. Use its own drawing methods.
        if not hasattr(env.current_episode, "goal"):
            topdown._draw_goals_positions(env.current_episode)
            topdown._draw_shortest_path(env.current_episode, start)
        reference = sim.get_straight_shortest_path_points(start, goal)
        initial_distance = float(env.get_metrics()["distance_to_goal"])
        action_ids = [int(a) for a in sim.compute_oracle_actions()]
        assert action_ids[-1] == HabitatSimActions.STOP
        positions, steps = [start.tolist()], []
        for index, action in enumerate(action_ids):
            assert not env.episode_over, "Episode ended before oracle sequence finished"
            env.step({"action": action})
            state = sim.get_agent_state()
            metrics = env.get_metrics()
            positions.append(state.position.tolist())
            steps.append(dict(step=index + 1, action=action, name=names[action],
                              position=positions[-1], collided=bool(sim.previous_step_collided),
                              distance_to_goal=float(metrics["distance_to_goal"])))
        result = dict(api="SoundSpacesSim.compute_oracle_actions",
                      start=start.tolist(), goal=goal.tolist(),
                      initial_distance_m=initial_distance,
                      reference_points=np.asarray(reference).tolist(),
                      actions=action_ids, action_names=[names[a] for a in action_ids],
                      positions=positions, steps=steps, success=bool(metrics["success"]),
                      spl=float(metrics["spl"]), done=bool(env.episode_over),
                      final_distance_m=float(metrics["distance_to_goal"]),
                      collisions=sum(s["collided"] for s in steps),
                      **summarize([names[a] for a in action_ids], positions))
        assert result["success"] and result["done"], result
        assert result["final_distance_m"] == 0 and result["collisions"] == 0, result
        np.testing.assert_allclose(positions[-1], goal, atol=1e-4, rtol=0)
        moved = [positions[0]] + [s["position"] for s in steps if s["name"] == "MOVE_FORWARD"]
        np.testing.assert_allclose(moved, reference, atol=1e-4, rtol=0)
        assert result["action_counts"]["MOVE_FORWARD"] == len(reference) - 1
        # Graph distance counts 1 m edges; SPL measures 3D displacement, so
        # small floor-height changes can make even this oracle's SPL < 1.
        assert np.isclose(initial_distance, (len(reference) - 1) * sim.config.GRID_SIZE)
        expected_spl = initial_distance / max(initial_distance, result["executed_length_m"])
        assert np.isclose(result["spl"], expected_spl), result
        return result, deepcopy(metrics["top_down_map"])


def run_habitat_sim(episode, config):
    from copy import deepcopy
    from types import SimpleNamespace
    import habitat_sim
    import numpy as np
    from habitat.tasks.nav.nav import TopDownMap
    from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum

    sim_config = habitat_sim.SimulatorConfiguration()
    sim_config.scene_id = episode.scene_id
    sim_config.scene_dataset_config_file = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
    sim_config.enable_physics = False
    sim_config.allow_sliding = False
    agent_config = habitat_sim.AgentConfiguration()
    agent_config.height = config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
    agent_config.radius = config.TASK_CONFIG.SIMULATOR.AGENT_0.RADIUS
    agent_config.sensor_specifications = []
    agent_config.action_space = {
        name: habitat_sim.agent.ActionSpec(name, habitat_sim.agent.ActuationSpec(amount=amount))
        for name, amount in [("move_forward", 0.25), ("turn_left", 10.0), ("turn_right", 10.0)]
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_config, [agent_config]))
    try:
        sim.seed(SEED)
        pf = sim.pathfinder
        assert pf.is_loaded, "Scene NavMesh was not loaded"
        original_start = np.asarray(episode.start_position)
        original_goal = np.asarray(episode.goals[0].position)
        start = np.asarray(pf.snap_point(original_start), dtype=np.float32)
        goal = np.asarray(pf.snap_point(original_goal), dtype=np.float32)
        assert np.isfinite(start).all() and np.isfinite(goal).all()
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = start, goal
        assert pf.find_path(path), "No geodesic route between start and source"
        state = habitat_sim.AgentState()
        state.position = start
        state.rotation = quat_from_coeffs(episode.start_rotation)
        sim.initialize_agent(0, state)
        goal_radius = 0.15
        follower = sim.make_greedy_follower(goal_radius=goal_radius, stop_key="STOP")
        binding_workaround = None
        try:
            actions = follower.find_path(goal)
        except SystemError as error:
            if "__next__" not in str(error) or "PyCapsule" not in str(error):
                raise
            # This installation's pybind vector iterator errors at iteration.
            # Use indexing on the same C++ find_path result, not another planner.
            binding_workaround = str(error)
            follower.reset()
            sim.get_agent(0).set_state(state)
            codes = follower.impl.find_path(quat_to_magnum(state.rotation), state.position, goal)
            actions = [follower.action_mapping[codes[i]] for i in range(len(codes))]
        assert actions and actions[-1] == "STOP"
        # Replay the complete returned sequence from the original pose.
        sim.get_agent(0).set_state(state)

        def shortest_path_points(source, target):
            reference = habitat_sim.ShortestPath()
            reference.requested_start, reference.requested_end = source, target
            assert pf.find_path(reference)
            return reference.points

        # TopDownMap expects the Habitat-Lab simulator interface. Adapt only
        # accessors; map generation, trajectory drawing and coloring are Habitat's.
        map_sim = SimpleNamespace(pathfinder=pf, get_agent=sim.get_agent,
                                  get_agent_state=sim.get_agent(0).get_state,
                                  get_straight_shortest_path_points=shortest_path_points)
        map_episode = SimpleNamespace(start_position=start,
                                      goals=[SimpleNamespace(position=goal)], goal=goal)
        map_config = config.TASK_CONFIG.TASK.TOP_DOWN_MAP.clone()
        map_config.defrost()
        map_config.FOG_OF_WAR.DRAW = False
        map_config.MAX_EPISODE_STEPS = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        map_config.freeze()
        topdown = TopDownMap(sim=map_sim, config=map_config)
        topdown.reset_metric(map_episode)
        positions, steps = [start.tolist()], []
        for index, action in enumerate(actions):
            collided = False
            if action != "STOP":
                observation = sim.step(action)
                collided = bool(observation.get("collided", False))
            topdown.update_metric(map_episode, action)
            position = sim.get_agent(0).get_state().position.copy()
            positions.append(position.tolist())
            remaining = habitat_sim.ShortestPath()
            remaining.requested_start, remaining.requested_end = position, goal
            assert pf.find_path(remaining)
            steps.append(dict(step=index + 1, name=action, position=position.tolist(),
                              collided=collided, distance_to_goal=float(remaining.geodesic_distance)))
        result = dict(api="habitat_sim.Simulator.make_greedy_follower().find_path(goal)",
                      binding_workaround=binding_workaround,
                      forward_step_m=0.25, turn_degrees=10, goal_radius_m=goal_radius,
                      start=start.tolist(), goal=goal.tolist(),
                      dataset_start=original_start.tolist(), dataset_goal=original_goal.tolist(),
                      start_snap_offset_m=float(np.linalg.norm(start - original_start)),
                      goal_snap_offset_m=float(np.linalg.norm(goal - original_goal)),
                      geodesic_length_m=float(path.geodesic_distance),
                      reference_points=np.asarray(path.points).tolist(),
                      actions=actions, positions=positions, steps=steps,
                      final_distance_m=steps[-1]["distance_to_goal"],
                      reached_goal_radius=steps[-1]["distance_to_goal"] <= goal_radius + 1e-5,
                      collisions=sum(s["collided"] for s in steps), **summarize(actions, positions))
        assert result["reached_goal_radius"], result
        return result, deepcopy(topdown.get_metric())
    finally:
        sim.close()


def save_topdown(path, map_info):
    from habitat.utils.visualizations import maps
    from PIL import Image

    Image.fromarray(maps.colorize_draw_agent_and_fit_to_height(
        map_info, output_height=800)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tests/results/VVfe2KiqLaN_548_shortest_path")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    from experiment import get_avwan_config
    from soundspaces.datasets.audionav_dataset import AudioNavDataset

    config = get_avwan_config(opts=["TASK_CONFIG.DATASET.CONTENT_SCENES", [SCENE]])
    content = ROOT / f"data/datasets/audionav/mp3d/v1/train_telephone/content/{SCENE}.json.gz"
    with gzip.open(content, "rt") as handle:
        raw = json.load(handle)
    raw["episodes"] = [ep for ep in raw["episodes"] if str(ep["episode_id"]) == EPISODE_ID]
    assert len(raw["episodes"]) == 1
    dataset = AudioNavDataset()
    dataset._config = config.TASK_CONFIG.DATASET
    dataset.from_json(json.dumps(raw), scenes_dir=config.TASK_CONFIG.DATASET.SCENES_DIR)
    assert len(dataset.episodes[0].goals) == 1
    save_json(output / "episode.json", raw["episodes"][0])
    # Run the sensorless simulator before SoundSpaces creates an EGL context.
    # Habitat-Sim 0.2.2 rejects create_renderer=False with an existing context.
    print("Replaying Habitat-Sim greedy geodesic follower...", flush=True)
    native, native_map = run_habitat_sim(dataset.episodes[0], config)
    save_json(output / "habitat_sim_follower.json", native)
    save_topdown(output / "habitat_sim_topdown.png", native_map)
    print("Replaying SoundSpaces graph oracle...", flush=True)
    discrete, discrete_map = run_soundspaces(config, dataset)
    save_json(output / "soundspaces_oracle.json", discrete)
    save_topdown(output / "soundspaces_topdown.png", discrete_map)
    save_topdown(output / "trajectory.png", discrete_map)
    summary = {"scene": SCENE, "episode_id": EPISODE_ID, "seed": SEED}
    for name, result in [("soundspaces", discrete), ("habitat_sim", native)]:
        summary[name] = {k: v for k, v in result.items() if k not in
                         ("positions", "reference_points", "steps", "actions", "action_names")}
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
