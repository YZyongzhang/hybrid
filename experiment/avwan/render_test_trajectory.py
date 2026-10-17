"""Replay one saved test episode and render its AV-WaN waypoint map.

Example:
python -m experiment.avwan.render_test_trajectory \
  --split test_multiple --scene gxdoqLR6rwA --episode-id 253578 \
  --output result/2026-10-16-avwan-offline-final/trajectories
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from experiment import AVWANRLEnv
from experiment.avwan.config.default import get_config
from experiment.avwan.evaluate import load_episodes
from experiment.avwan.offline.policy import OfflinePolicy
from utils.visualizations import plot_top_down_map_waypoint


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / (
    "result/2026-10-15-avwan-spl-improvement/offline/"
    "seed_20261016_actor_only_bc/training/latest.pt"
)
DEFAULT_FOUNDATION = ROOT / (
    "result/2026-10-13-staged-grl-reproduction/foundation/action_stage/best.pt"
)
DEFAULT_CONFIG = ROOT / (
    "result/10-11/total/2026-10-04-stop-negative-recovery-study/evaluation.yaml"
)


def render(args):
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)

    config = get_config(
        str(args.config),
        opts=["TASK_CONFIG.DATASET.SPLIT", args.split,
              "PREDICTION_INTERVAL", str(args.prediction_interval)],
    )
    config.defrost()
    config.TASK_CONFIG.TRAIN.OPEN = False
    config.TASK_CONFIG.COLLECT.OPEN = False
    measurements = list(config.TASK_CONFIG.TASK.MEASUREMENTS)
    if "TOP_DOWN_MAP" not in measurements:
        measurements.append("TOP_DOWN_MAP")
    config.TASK_CONFIG.TASK.MEASUREMENTS = measurements
    config.freeze()

    dataset = load_episodes(config)
    planned = list(dataset.episodes)
    matches = [
        (index, episode) for index, episode in enumerate(planned)
        if Path(episode.scene_id).stem == args.scene
        and str(episode.episode_id) == str(args.episode_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {args.scene}/{args.episode_id} episode in {args.split}; "
            f"found {len(matches)}"
        )
    global_index, episode = matches[0]

    state = torch.load(args.checkpoint, map_location="cpu")
    policy = OfflinePolicy.from_checkpoint(state, str(args.foundation), device)
    env = AVWANRLEnv(config, dataset)
    try:
        env.habitat_env.current_episode = episode
        env.seed(args.seed + global_index)
        policy.reset()
        observation = env.reset()
        initial_info = env.get_info(observation)
        done = False
        info = initial_info
        primitive_steps = 0
        while not done:
            action = policy.act(observation, env.action_mask(observation))
            observation, _, done, info = env.step(action)
            primitive_steps += int(info.get("num_steps", 0))
        info = env.get_info(observation)

        image = plot_top_down_map_waypoint(info)
        args.output.mkdir(parents=True, exist_ok=True)
        stem = f"{args.split}_{args.scene}_{args.episode_id}"
        output_image = args.output / f"{stem}.png"
        figure, axis = plt.subplots(figsize=(10, 9), dpi=180)
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"{args.split} · {args.scene}/{args.episode_id}\n"
            f"{info.get('success', False) and 'Success' or 'Failure'} · "
            f"SPL {float(info.get('spl', 0.0)):.3f} · "
            f"start distance {float(initial_info['distance_to_goal']):.1f} m · "
            f"{len(info['waypoint_positions'])} waypoints",
            fontsize=12,
        )
        legend = [
            Line2D([0], [0], color=(1.0, 190/255, 0), lw=2,
                   label="Waypoint order (start → targets)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#00e600",
                   markeredgecolor="white", markersize=9, label="Start"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="red",
                   markeredgecolor="white", markersize=8, label="Waypoints (numbered)"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="blue",
                   markeredgecolor="white", markersize=8, label="Sound source"),
        ]
        figure.legend(handles=legend, loc="lower center", ncol=2,
                      frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.02))
        figure.tight_layout(rect=(0, 0.07, 1, 0.94))
        figure.savefig(output_image, bbox_inches="tight", pad_inches=0.12)
        plt.close(figure)

        record = {
            "split": args.split,
            "scene": args.scene,
            "episode_id": str(args.episode_id),
            "seed": args.seed + global_index,
            "success": bool(info.get("success", False)),
            "spl": float(info.get("spl", 0.0)),
            "initial_distance": float(initial_info["distance_to_goal"]),
            "final_distance": float(info["distance_to_goal"]),
            "primitive_steps": primitive_steps,
            "waypoint_count": len(info["waypoint_positions"]),
            "start_position": info["start_position"],
            "sound_source_positions": info["sound_source_positions"],
            "waypoint_positions": info["waypoint_positions"],
            "checkpoint": str(args.checkpoint),
            "image": str(output_image),
        }
        (args.output / f"{stem}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(json.dumps(record, indent=2), flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--foundation", type=Path, default=DEFAULT_FOUNDATION)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--prediction-interval", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.12)
    args = parser.parse_args()
    for attr in ("output", "checkpoint", "foundation", "config"):
        path = getattr(args, attr).expanduser()
        setattr(args, attr, path if path.is_absolute() else (ROOT / path).resolve())
    render(args)


if __name__ == "__main__":
    main()
