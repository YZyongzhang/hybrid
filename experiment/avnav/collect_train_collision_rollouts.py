"""Collect raw learned-policy collision transitions on AVNav training episodes."""
import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.avnav.config import resolve_path
from experiment.avnav.environment.avnav_env import AudioNavRLEnv
from experiment.avnav.environment.config import get_config
from experiment.avnav.environment.evaluation_episodes import load_episodes
from experiment.avnav.evaluate_online import OnlinePolicy, _metrics
from experiment.avnav.model.hybrid import Network
from experiment.avnav.offline_rl import _initial_policy_head


def collect(config_path, checkpoint_path, output_dir, episodes=100, split="train_multiple",
            max_steps=200, successful_only=False, sample_actions=False,
            sample_temperature=1.0, episode_manifest=None, episode_role=None,
            save_full_trajectories=False, save_collision_clips=True):
    config_path = Path(config_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty collection output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device(run_config["EVALUATION"]["DEVICE"])
    torch.set_num_threads(2)
    seed = int(run_config["EVALUATION"]["SEED"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    foundation_path = resolve_path(run_config["FOUNDATION"]["CHECKPOINT"])
    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(foundation_path, map_location=device,
                                          weights_only=True))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state["foundation_sha256"] != __import__("hashlib").sha256(
            foundation_path.read_bytes()).hexdigest():
        raise ValueError("Offline policy was trained with a different foundation checkpoint")
    policy_config = state.get("config") or {}
    action_history = bool(policy_config.get("action_history", False))
    collision_history = bool(policy_config.get("collision_history", False))
    feature_dim = 256 + (4 if action_history else 0) + (1 if collision_history else 0)
    actor = _initial_policy_head(foundation, feature_dim).to(device)
    actor.load_state_dict(state["actor"])
    policy = OnlinePolicy(foundation, actor, device, action_history=action_history,
                          collision_history=collision_history)

    env_config = get_config(
        task_config_path=str(resolve_path(run_config["EVALUATION"]["CONFIG"])),
        opts=["TASK_CONFIG.DATASET.SPLIT", split],
    )
    env_config.defrost()
    env_config.TASK_CONFIG.TRAIN.OPEN = False
    env_config.TASK_CONFIG.COLLECT.OPEN = False
    env_config.TASK_CONFIG.TASK.SENSORS = ["SPECTROGRAM_SENSOR"]
    env_config.TASK_CONFIG.TASK.MEASUREMENTS = [
        name for name in env_config.TASK_CONFIG.TASK.MEASUREMENTS
        if name != "TOP_DOWN_MAP"
    ]
    env_config.freeze()
    selected_keys = None
    if episode_manifest:
        rows = [json.loads(line) for line in Path(episode_manifest).read_text().splitlines()
                if line.strip()]
        if episode_role:
            rows = [row for row in rows if row.get("role") == episode_role]
        selected_keys = [(str(row["scene"]), str(row["episode_id"])) for row in rows]
        if len(selected_keys) != len(set(selected_keys)):
            raise ValueError("Episode selection manifest contains duplicate keys")
        if episodes not in (0, len(selected_keys)):
            raise ValueError("episodes must be 0 or match the selected manifest row count")
    dataset = load_episodes(
        env_config.TASK_CONFIG.DATASET,
        max_episodes=0 if selected_keys else episodes,
        seed=seed,
        episode_keys=selected_keys,
    )
    chosen_episodes = list(dataset.episodes)
    keys = [(Path(str(row.scene_id)).stem, str(row.episode_id)) for row in chosen_episodes]
    env = AudioNavRLEnv(env_config, dataset)
    provenance = []
    totals = {"episodes": len(chosen_episodes), "transitions": 0,
              "collisions": 0, "collision_clips": 0, "success_episodes": 0,
              "successful_collision_episodes": 0, "errors": 0}
    try:
        for episode_index, requested_episode in enumerate(chosen_episodes):
            scene, episode_id = keys[episode_index]
            observations, actions, rewards, dones, collisions = [], [], [], [], []
            error = None
            try:
                env._env.current_episode = requested_episode
                observation = env.reset()
                active = env._env.current_episode
                active_key = (Path(str(active.scene_id)).stem, str(active.episode_id))
                if active_key != keys[episode_index]:
                    raise RuntimeError(f"Reset changed episode identity to {active_key}")
                policy.reset()
                observations.append(observation)
                done = False
                info = {}
                while not done and len(actions) < max_steps:
                    action = policy.act(observation, sample=sample_actions,
                                        temperature=sample_temperature)
                    observation, reward, done, info = env.step(action=action)
                    sim = getattr(env._env, "_sim", None)
                    collided = bool(getattr(sim, "previous_step_collided", False))
                    policy.record_collision(collided)
                    actions.append(int(action))
                    rewards.append(float(reward))
                    dones.append(bool(done))
                    collisions.append(collided)
                    observations.append(observation)
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
                totals["errors"] += 1
            totals["transitions"] += len(actions)
            totals["collisions"] += sum(collisions)
            episode_success = False
            if error is None:
                _, episode_success = _metrics(env, info)
                totals["success_episodes"] += int(episode_success)
                totals["successful_collision_episodes"] += int(
                    episode_success and any(collisions))
            provenance.append({"scene": scene, "episode_id": episode_id,
                               "transitions": len(actions), "collision_steps": sum(collisions),
                               "success": bool(episode_success), "error": error})

            if error is None:
                if save_full_trajectories:
                    trajectory = {
                        "obs": observations,
                        "action_id": actions,
                        "reward": rewards,
                        "done": dones,
                        "collision": collisions,
                        "parent_scene": scene,
                        "parent_episode_id": episode_id,
                        "episode_success": bool(episode_success),
                        "policy_checkpoint": str(checkpoint_path),
                        "action_handling": "raw actor output; no action override or recovery",
                    }
                    trajectory_dir = output_dir / "trajectories" / scene
                    trajectory_dir.mkdir(parents=True, exist_ok=True)
                    trajectory_path = trajectory_dir / f"{episode_id}.pkl"
                    with trajectory_path.open("wb") as handle:
                        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    provenance[-1]["trajectory_file"] = str(trajectory_path.resolve())
                clip_count = 0
                for collision_step, collided in enumerate(collisions):
                    if not save_collision_clips:
                        break
                    if not collided:
                        continue
                    if successful_only and not episode_success:
                        continue
                    start = max(0, collision_step - 1)
                    stop = min(len(actions), collision_step + 2)
                    clip = {
                        "obs": observations[start:stop + 1],
                        "action_id": actions[start:stop],
                        "reward": rewards[start:stop],
                        "done": dones[start:stop],
                        "collision": collisions[start:stop],
                        "parent_scene": scene,
                        "parent_episode_id": episode_id,
                        "parent_collision_step": collision_step,
                        "clip_start_step": start,
                        "episode_success": bool(episode_success),
                        "policy_checkpoint": str(checkpoint_path),
                    }
                    clip_dir = output_dir / scene
                    clip_dir.mkdir(parents=True, exist_ok=True)
                    clip_path = clip_dir / f"{episode_id}_collision_{collision_step:04d}.pkl"
                    with clip_path.open("wb") as handle:
                        pickle.dump(clip, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    clip_count += 1
                totals["collision_clips"] += clip_count
                provenance[-1]["collision_clips"] = clip_count
            if (episode_index + 1) % 10 == 0 or episode_index + 1 == len(chosen_episodes):
                print(f"{split}: {episode_index + 1}/{len(chosen_episodes)} episodes; "
                      f"train-only collisions={totals['collisions']}; "
                      f"clips={totals['collision_clips']}", flush=True)
    finally:
        env.close()

    with (output_dir / "parent_episode_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in provenance:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "status": "completed",
        "split": split,
        "episodes_requested": len(selected_keys) if selected_keys is not None else episodes,
        "episodes_loaded": len(chosen_episodes),
        "seed": seed,
        "max_steps": max_steps,
        "checkpoint": str(checkpoint_path),
        "foundation_checkpoint": str(foundation_path),
        "action_handling": (
            f"train-only learned actor categorical sample at temperature {sample_temperature}; "
            "no action override or recovery" if sample_actions else
            "raw learned actor argmax; no action override or recovery"),
        "successful_only_collision_clips": bool(successful_only),
        "episode_selection_manifest": str(Path(episode_manifest).resolve()) if episode_manifest else None,
        "episode_selection_role": episode_role,
        "full_trajectories_saved": bool(save_full_trajectories),
        "collision_clips_saved": bool(save_collision_clips),
        "rollout_source": "canonical train_multiple episodes",
        "totals": totals,
        "parent_episode_manifest": str((output_dir / "parent_episode_manifest.jsonl").resolve()),
    }
    (output_dir / "collection_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--split", default="train_multiple")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--successful-only", action="store_true",
                        help="Save collision clips only from episodes that eventually succeed")
    parser.add_argument("--sample-actions", action="store_true",
                        help="Collect train-only exploration by sampling the learned actor")
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--episode-manifest", default="",
                        help="JSONL list of preselected train split episode keys")
    parser.add_argument("--episode-role", choices=("train", "heldout"), default="",
                        help="Optional role filter in the episode selection manifest")
    parser.add_argument("--save-full-trajectories", action="store_true",
                        help="Save every observation/action/reward/collision for each train episode")
    parser.add_argument("--no-collision-clips", action="store_true",
                        help="Skip per-collision clips when saving full trajectories")
    args = parser.parse_args()
    print(json.dumps(collect(args.config, args.checkpoint, args.output,
                             args.episodes, args.split, args.max_steps,
                             args.successful_only, args.sample_actions,
                             args.sample_temperature,
                             args.episode_manifest or None,
                             args.episode_role or None,
                             args.save_full_trajectories,
                             not args.no_collision_clips),
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
