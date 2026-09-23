import sys
sys.path.append("./")
from experiment import AVWANRLEnv , get_avwan_config

def test_expert_action():
    """Run one expert episode and save its complete outputs and final map."""
    from copy import deepcopy
    from pathlib import Path
    import pickle

    import numpy as np
    from PIL import Image
    from utils.visualizations import plot_top_down_map_waypoint

    output_dir = Path(__file__).resolve().parent
    pickle_path = output_dir / "avwan_expert_episode.pkl"
    map_path = output_dir / "avwan_expert_topdown.png"
    config = get_avwan_config()
    env = AVWANRLEnv(config=config)
    try:
        env.seed(config.SEED)
        observation = env.reset()
        episode = env.current_episode
        data = {
            "episode_id": str(episode.episode_id),
            "scene_id": str(episode.scene_id),
            "initial_observation": deepcopy(observation),
            "steps": [],
        }
        expert_action = getattr(env, "get_expert_amp_action", None)
        if expert_action is None:
            expert_action = env.get_expert_map_action

        done = False
        while not done:
            action = expert_action()
            observation, reward, done, info = env.step(action)
            assert len(info["waypoint_positions"]) == len(data["steps"]) + 1
            if info["reaching_waypoint"] or action == env.stop_action:
                np.testing.assert_allclose(
                    info["waypoint_position"], env.habitat_env.sim.get_agent_state().position,
                    atol=1e-4, rtol=0)
            # Snapshot arrays/maps before the next simulator step updates them.
            data["steps"].append(deepcopy({
                "action": action,
                "observation": observation,
                "reward": reward,
                "done": done,
                "info": info,
            }))
            print(f"step={len(data['steps'])}, action={action}, "
                  f"reward={reward:.4f}, done={done}")

        final_info = data["steps"][-1]["info"]
        for key in ("start_position", "sound_source_positions", "waypoint_positions"):
            data[key] = deepcopy(final_info[key])
        assert len(data["waypoint_positions"]) == len(data["steps"])
        with pickle_path.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        top_down_map = plot_top_down_map_waypoint(
            final_info,
            dataset=config.TASK_CONFIG.SIMULATOR.SCENE_DATASET,
        )
        Image.fromarray(top_down_map).save(map_path)
        print(f"Episode {data['episode_id']} saved to {pickle_path}")
        print(f"Top-down map saved to {map_path}")
    finally:
        env.close()
def test_env(config):
    env = AVWANRLEnv(config= config)
    

if __name__ == "__main__":
    
    config = get_avwan_config()
    test_env(config)
