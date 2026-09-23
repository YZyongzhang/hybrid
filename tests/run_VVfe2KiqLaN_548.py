"""单独运行 VVfe2KiqLaN 场景的 episode 548，使用 BFS 专家导航。

运行：conda activate ss && python tests/run_VVfe2KiqLaN_548.py
"""
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

SCENE = "VVfe2KiqLaN"
EPISODE_ID = "548"
SEED = 20260924  # 与批量脚本中的该案例保持一致。
OUTPUT = ROOT / "tests/results/VVfe2KiqLaN_548"


def main():
    import numpy as np
    from PIL import Image
    from experiment import AVWANRLEnv, get_avwan_config
    from experiment.avwan.environment.BFSExpert import NoReachableWaypointError
    from soundspaces.datasets.audionav_dataset import AudioNavDataset
    from utils.visualizations import plot_top_down_map_waypoint

    os.chdir(ROOT)  # 配置中的数据路径相对于项目根目录。
    random.seed(SEED)
    np.random.seed(SEED)
    config = get_avwan_config(opts=["TASK_CONFIG.DATASET.CONTENT_SCENES", [SCENE]])
    episode_file = ROOT / f"data/datasets/audionav/mp3d/v1/train_telephone/content/{SCENE}.json.gz"
    with gzip.open(episode_file, "rt") as handle:
        raw = json.load(handle)
    raw["episodes"] = [ep for ep in raw["episodes"] if str(ep["episode_id"]) == EPISODE_ID]
    if len(raw["episodes"]) != 1:
        raise ValueError(f"应找到唯一的 episode {EPISODE_ID}，实际找到 {len(raw['episodes'])} 个")
    dataset = AudioNavDataset()
    dataset._config = config.TASK_CONFIG.DATASET
    dataset.from_json(json.dumps(raw), scenes_dir=config.TASK_CONFIG.DATASET.SCENES_DIR)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = AVWANRLEnv(config, dataset)
    try:
        env.seed(SEED)
        observation = env.reset()
        info = env.get_info(observation)
        done, blocked = False, None
        steps = []
        print(f"{SCENE}/{EPISODE_ID}: initial_distance={info['distance_to_goal']:.2f}", flush=True)
        while not done:
            try:
                action = env.get_expert_map_action()
            except NoReachableWaypointError as error:
                blocked = str(error)
                print(f"导航阻塞：{blocked}", flush=True)
                break
            observation, reward, done, info = env.step(action)
            steps.append(dict(action=int(action), reward=float(reward),
                              distance=float(info["distance_to_goal"]),
                              num_steps=int(info["num_steps"]),
                              reached=bool(info["reaching_waypoint"])))
            print(f"step={len(steps)}, action={action}, reward={reward:.3f}, "
                  f"distance={info['distance_to_goal']:.2f}, "
                  f"reached={info['reaching_waypoint']}, done={done}", flush=True)

        result = dict(scene=SCENE, episode_id=EPISODE_ID, seed=SEED,
                      success=bool(info.get("success", False)), done=bool(done),
                      truncated=bool(info.get("truncated", False)), blocked=blocked,
                      final_distance=float(info["distance_to_goal"]), spl=float(info["spl"]),
                      primitive_steps=sum(step["num_steps"] for step in steps))
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        (OUTPUT / "result.json").write_text(
            json.dumps(dict(result, steps=steps), indent=2, ensure_ascii=False) + "\n")
        Image.fromarray(plot_top_down_map_waypoint(info)).save(OUTPUT / "trajectory.png")
        print(f"结果和轨迹图已保存到：{OUTPUT}", flush=True)
        return 0 if result["success"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
