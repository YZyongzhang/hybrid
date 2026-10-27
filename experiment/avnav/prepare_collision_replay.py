"""Extract collision and immediate recovery transitions from AVNav training episodes."""
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.dataset_avnav import get_files
from experiment.avnav.config import resolve_path
from experiment.avnav.offline_rl import _read_observation
from experiment.avnav.model.hybrid import Network


def _episode_collision_features(data, model, device, batch_size):
    actions = np.asarray(data["action_id"]).reshape(-1).astype(np.int64)
    rewards = np.asarray(data["reward"], dtype=np.float32).reshape(-1)
    dones = np.asarray(data["done"], dtype=np.bool_).reshape(-1)
    count = len(actions)
    if count < 1 or len(data.get("obs", [])) != count + 1:
        raise ValueError("Episode observations and actions have inconsistent lengths")
    collision = data.get("collision")
    if collision is None:
        path_points = data.get("path_point")
        if path_points is None:
            return None
        if (isinstance(path_points, (tuple, list)) and len(path_points) == 1 and
                len(path_points[0]) == count + 1):
            path_points = path_points[0]
        try:
            positions = np.asarray(path_points, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if positions.shape != (count + 1, 3):
            return None
        displacement = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        # This is a training-data label: a recorded FORWARD command that made
        # effectively no progress in the stored simulator path. It never
        # changes an action during data collection or evaluation.
        collision = (actions == 1) & (displacement <= 1e-3)
        collision_label_method = "train_path_stationary_forward_proxy"
    else:
        collision_label_method = "recorded_training_collision_flag"
    collision = np.asarray(collision, dtype=np.bool_).reshape(-1)
    if len(collision) != count:
        raise ValueError("Collision labels and actions have inconsistent lengths")
    collision_indices = np.flatnonzero(collision)
    if "parent_collision_step" in data:
        # Collection writes one short clip per source collision. Keep only its
        # center collision so neighboring clips cannot duplicate transitions.
        if int(data["parent_collision_step"]) == 0:
            center = 0
        elif "clip_start_step" in data:
            center = int(data["parent_collision_step"]) - int(data["clip_start_step"])
        else:
            center = 1
        if not 0 <= center < count or not collision[center]:
            raise ValueError("Collision clip center does not match its collision metadata")
        collision_indices = np.asarray([center], dtype=np.int64)
    if not len(collision_indices):
        return None

    # Include the colliding action and the next action selected from the
    # post-collision observation. Those are the two decisions the augmentation
    # is intended to teach.
    selected = set()
    for index in collision_indices.tolist():
        selected.add(index)
        if index + 1 < count and not collision[index + 1]:
            selected.add(index + 1)
    state_indices = sorted(selected | {index + 1 for index in selected})
    audio_rows, rgb_rows, depth_rows = [], [], []
    for index in state_indices:
        audio, rgb, depth = _read_observation(data["obs"][index])
        if index == 0:
            previous_rgb, previous_depth = torch.zeros_like(rgb), torch.zeros_like(depth)
        else:
            _, previous_rgb, previous_depth = _read_observation(data["obs"][index - 1])
        audio_rows.append(audio)
        rgb_rows.append(torch.cat((previous_rgb, rgb), dim=-1))
        depth_rows.append(torch.cat((previous_depth, depth), dim=-1))

    encoded = []
    with torch.inference_mode():
        for start in range(0, len(state_indices), batch_size):
            stop = min(len(state_indices), start + batch_size)
            encoded.append(model.embedding_forward(
                torch.stack(audio_rows[start:stop]).to(device),
                torch.stack(rgb_rows[start:stop]).to(device),
                torch.stack(depth_rows[start:stop]).to(device),
            ).float().cpu())
    encoded = torch.cat(encoded)
    state_map = {index: encoded[offset] for offset, index in enumerate(state_indices)}

    rows = {key: [] for key in ("states", "next_states", "actions", "rewards", "dones",
                                "collision", "post_collision", "previous_actions")}
    for index in sorted(selected):
        is_collision = index in set(collision_indices.tolist())
        post_collision = index > 0 and (index - 1) in set(collision_indices.tolist())
        rows["states"].append(state_map[index])
        rows["next_states"].append(state_map[index + 1])
        rows["actions"].append(int(actions[index]))
        rows["rewards"].append(float(rewards[index]))
        rows["dones"].append(float(dones[index]))
        rows["collision"].append(is_collision)
        rows["post_collision"].append(post_collision)
        rows["previous_actions"].append(int(actions[index - 1]) if index > 0 else -1)
    return {"states": torch.stack(rows["states"]),
            "next_states": torch.stack(rows["next_states"]),
            "actions": torch.tensor(rows["actions"], dtype=torch.long),
            "rewards": torch.tensor(rows["rewards"], dtype=torch.float32),
            "dones": torch.tensor(rows["dones"], dtype=torch.float32),
            "collision": torch.tensor(rows["collision"], dtype=torch.bool),
            "post_collision": torch.tensor(rows["post_collision"], dtype=torch.bool),
            "previous_actions": torch.tensor(rows["previous_actions"], dtype=torch.long),
            "collision_label_method": collision_label_method}


def prepare(sources, foundation_path, output, device="cuda:0", batch_size=32):
    device = torch.device(device)
    foundation_path = resolve_path(foundation_path)
    output = resolve_path(output)
    model = Network().to(device)
    model.load_state_dict(torch.load(foundation_path, map_location=device, weights_only=True))
    model.eval().requires_grad_(False)
    paths = [resolve_path(item) for item in sources]
    files = []
    for path in paths:
        files.extend(get_files(path, (".pkl", ".pickle")))
    files = sorted({Path(item).resolve() for item in files})
    accumulated = {key: [] for key in ("states", "next_states", "actions", "rewards", "dones",
                                      "collision", "post_collision", "previous_actions")}
    label_methods = set()
    eligible_files = collision_episodes = collision_steps = 0
    for path in tqdm(files, desc="Extract collision transitions", mininterval=10):
        with path.open("rb") as handle:
            episode = pickle.load(handle)
        transitions = _episode_collision_features(episode, model, device, batch_size)
        if transitions is None:
            continue
        label_methods.add(transitions.pop("collision_label_method"))
        eligible_files += 1
        collision_episodes += 1
        collision_steps += int(transitions["collision"].sum().item())
        for key in accumulated:
            accumulated[key].append(transitions[key])
    if not accumulated["actions"]:
        raise ValueError("No collision-labelled transitions were found in the source data")
    payload = {key: torch.cat(values) for key, values in accumulated.items()}
    payload["source_files"] = [str(item.resolve()) for item in paths]
    payload["eligible_files"] = eligible_files
    payload["collision_steps"] = collision_steps
    payload["transition_count"] = len(payload["actions"])
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "collision_replay.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(output / "collision_replay.pt")
    action_context = torch.zeros((len(payload["actions"]), 4), dtype=payload["states"].dtype)
    known_previous = payload["previous_actions"] >= 0
    action_context[known_previous] = torch.nn.functional.one_hot(
        payload["previous_actions"][known_previous], num_classes=4
    ).to(dtype=payload["states"].dtype)
    next_action_context = torch.nn.functional.one_hot(
        payload["actions"], num_classes=4
    ).to(dtype=payload["states"].dtype)
    history_payload = dict(payload)
    history_payload["states"] = torch.cat((payload["states"], action_context), dim=-1)
    history_payload["next_states"] = torch.cat((payload["next_states"], next_action_context), dim=-1)
    history_path = output / "collision_replay_action_history.pt"
    torch.save(history_payload, history_path)
    summary = {"source_directories": payload["source_files"],
               "source_files_scanned": len(files),
               "episodes_with_collisions": eligible_files,
               "collision_steps": collision_steps,
               "collision_label_methods": sorted(label_methods),
               "action_history_features": True,
               "action_override_used": False,
               "collision_and_post_collision_transitions": len(payload["actions"]),
               "colliding_transitions": int(payload["collision"].sum().item()),
               "post_collision_transitions": int(payload["post_collision"].sum().item()),
               "output": str((output / "collision_replay.pt").resolve()),
               "action_history_output": str(history_path.resolve())}
    (output / "collision_replay_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--foundation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source, args.foundation, args.output,
                             args.device, args.batch_size), indent=2), flush=True)


if __name__ == "__main__":
    main()
