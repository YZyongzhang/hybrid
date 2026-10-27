"""Train a discrete CQL policy from AVNav's recorded SoundSpaces episodes."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import yaml

from Dataset.dataset_avnav import get_files
from experiment.avnav.config import resolve_path
from experiment.avnav.model.hybrid import Network


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_clean_data_audit(config):
    audit_path = config.get("DATA", {}).get("SOURCE_AUDIT")
    if not audit_path:
        return None
    audit_path = resolve_path(audit_path)
    if not audit_path.is_file():
        raise FileNotFoundError(f"Configured source audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise ValueError(f"Training source audit did not pass: {audit_path}")
    return str(audit_path.resolve())


def _read_observation(obs):
    rgb = torch.as_tensor(np.asarray(obs["rgb"])).float() / 255.0
    depth = torch.as_tensor(np.asarray(obs["depth"])).float()
    if depth.ndim == 2:
        depth = depth.unsqueeze(-1)
    spectrogram = obs["spectrogram"]
    if isinstance(spectrogram, (tuple, list)):
        spectrogram = spectrogram[0]
    audio = torch.as_tensor(np.asarray(spectrogram)).float()
    if audio.ndim == 4 and audio.shape[0] == 1:
        audio = audio[0]
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or depth.shape != (*rgb.shape[:2], 1):
        raise ValueError("Expected matching HWC RGB and one-channel depth observations")
    if audio.shape != (65, 26, 2):
        raise ValueError(f"Expected audio [65, 26, 2], got {tuple(audio.shape)}")
    return audio, rgb, depth


def _collision_labels(data, actions):
    """Return recorded collision flags, or the documented train-path proxy."""
    collision = data.get("collision")
    if collision is not None:
        collision = np.asarray(collision, dtype=np.bool_).reshape(-1)
        if len(collision) != len(actions):
            raise ValueError("Collision labels and actions have inconsistent lengths")
        return collision
    path_points = data.get("path_point")
    if path_points is None:
        return np.zeros(len(actions), dtype=np.bool_)
    try:
        positions = np.asarray(path_points, dtype=np.float32)
    except (TypeError, ValueError):
        return np.zeros(len(actions), dtype=np.bool_)
    if positions.ndim == 3 and positions.shape[0] == 1:
        positions = positions[0]
    if positions.shape != (len(actions) + 1, 3):
        return np.zeros(len(actions), dtype=np.bool_)
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return (actions == 1) & (displacement <= 1e-3)


def _episode_features(data, model, device, batch_size, action_history=False,
                      collision_history=False):
    observations = data["obs"]
    actions = np.asarray(data["action_id"]).reshape(-1)
    rewards = np.asarray(data["reward"], dtype=np.float32).reshape(-1)
    dones = np.asarray(data["done"], dtype=np.bool_).reshape(-1)
    count = len(observations) - 1
    if count < 1 or len(actions) != count or len(rewards) != count or len(dones) != count:
        raise ValueError("Episode obs/action_id/reward/done lengths do not describe the same transitions")
    if np.any(actions != actions.astype(np.int64)) or np.any((actions < 0) | (actions > 3)):
        raise ValueError("AVNav offline actions must be integer IDs in [0, 3]")
    collision = _collision_labels(data, actions) if collision_history else None

    audios, rgbs, depths = [], [], []
    previous_rgb = previous_depth = None
    for observation in observations:
        audio, rgb, depth = _read_observation(observation)
        if previous_rgb is None:
            previous_rgb, previous_depth = torch.zeros_like(rgb), torch.zeros_like(depth)
        audios.append(audio)
        rgbs.append(torch.cat((previous_rgb, rgb), dim=-1))
        depths.append(torch.cat((previous_depth, depth), dim=-1))
        previous_rgb, previous_depth = rgb, depth

    embeddings = []
    with torch.inference_mode():
        for start in range(0, count + 1, batch_size):
            stop = min(count + 1, start + batch_size)
            features = model.embedding_forward(
                torch.stack(audios[start:stop]).to(device),
                torch.stack(rgbs[start:stop]).to(device),
                torch.stack(depths[start:stop]).to(device),
            )
            embeddings.append(features.float().cpu())
    states = torch.cat(embeddings)
    current_states, next_states = states[:-1], states[1:]
    if action_history:
        previous_actions = torch.zeros(count, dtype=torch.long)
        previous_actions[1:] = torch.as_tensor(actions[:-1], dtype=torch.long)
        current_states = torch.cat((current_states, F.one_hot(previous_actions, 4).float()), dim=-1)
        next_states = torch.cat((next_states,
                                 F.one_hot(torch.as_tensor(actions, dtype=torch.long), 4).float()),
                                dim=-1)
    if collision_history:
        previous_collision = torch.zeros(count, dtype=torch.float32)
        previous_collision[1:] = torch.as_tensor(collision[:-1], dtype=torch.float32)
        current_collision = torch.as_tensor(collision, dtype=torch.float32)
        current_states = torch.cat((current_states, previous_collision[:, None]), dim=-1)
        next_states = torch.cat((next_states, current_collision[:, None]), dim=-1)
    return dict(
        states=current_states, next_states=next_states,
        actions=torch.as_tensor(actions, dtype=torch.long),
        rewards=torch.as_tensor(rewards, dtype=torch.float32),
        dones=torch.as_tensor(dones, dtype=torch.float32),
    )


def build_feature_dataset(source, destination, model, foundation_path, device,
                          batch_size=32, shard_size=8192, seed=42,
                          action_history=False, collision_history=False):
    """Extract frozen foundation features while retaining the recorded RL reward."""
    destination = Path(destination)
    manifest_path = destination / "manifest.json"
    source_paths = source if isinstance(source, (list, tuple)) else [source]
    resolved_sources = [Path(item).expanduser().resolve() for item in source_paths]
    feature_dim = 256 + (4 if action_history else 0) + (1 if collision_history else 0)
    signature = dict(source=[str(path) for path in resolved_sources],
                     foundation_sha256=_sha256(foundation_path),
                     feature_dim=feature_dim, action_count=4,
                     action_history=bool(action_history))
    if collision_history:
        signature["collision_history"] = True
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in signature.items()):
            return manifest
        raise FileExistsError(f"Feature directory does not match this run: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Feature directory already contains partial output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for source_path in resolved_sources:
        files.extend(get_files(source_path, (".pkl", ".pickle")))
    resolved_files = [path.resolve() for path in files]
    if len(set(resolved_files)) != len(resolved_files):
        raise ValueError("Offline RL source directories contain duplicate episode files")
    files = [Path(path) for path in resolved_files]
    random.Random(seed).shuffle(files)
    buffers = {"states": [], "next_states": [], "actions": [], "rewards": [], "dones": []}
    shards, samples = [], 0

    def flush():
        nonlocal samples
        if not buffers["actions"]:
            return
        name = f"transitions_{len(shards):05d}.pt"
        payload = {key: torch.cat(value, dim=0) for key, value in buffers.items()}
        torch.save(payload, destination / name)
        count = len(payload["actions"])
        shards.append({"file": name, "samples": count})
        samples += count
        for values in buffers.values():
            values.clear()

    episodes = 0
    source_label = "+".join(path.name for path in resolved_sources)
    for path in tqdm(files, desc=f"Features {source_label}", mininterval=10):
        with Path(path).open("rb") as handle:
            episode = pickle.load(handle)
        transitions = _episode_features(episode, model, device, batch_size,
                                        action_history=action_history,
                                        collision_history=collision_history)
        for key in buffers:
            buffers[key].append(transitions[key])
        episodes += 1
        if sum(len(item) for item in buffers["actions"]) >= shard_size:
            flush()
    flush()
    if not samples:
        raise ValueError(f"No transitions found at {source}")
    manifest = dict(**signature, episodes=episodes, samples=samples, shards=shards)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class FeatureDataset:
    def __init__(self, path):
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        tensors = {key: [] for key in ("states", "next_states", "actions", "rewards", "dones")}
        for entry in manifest["shards"]:
            shard = torch.load(path / entry["file"], map_location="cpu", weights_only=True)
            for key in tensors:
                if len(shard[key]) != entry["samples"]:
                    raise ValueError(f"{entry['file']}: manifest transition count mismatch")
                tensors[key].append(shard[key])
        self.data = {key: torch.cat(value) for key, value in tensors.items()}
        self.samples = manifest["samples"]


class QNetwork(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=256, action_count=4):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, action_count))

    def forward(self, states):
        return self.layers(states)


class ValueNetwork(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=256):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, 1))

    def forward(self, states):
        return self.layers(states).squeeze(-1)


def _initial_policy_head(foundation, feature_dim):
    """Initialize a policy head with optional learned temporal context."""
    source = foundation.final.action_head
    if feature_dim == 256:
        return deepcopy(source)
    context_dim = feature_dim - 256
    if context_dim not in (1, 4, 5, 8):
        raise ValueError(f"Unsupported policy feature dimension: {feature_dim}")
    target = nn.Sequential(
        nn.Linear(feature_dim, 128), nn.ReLU(), nn.LayerNorm(128), nn.Linear(128, 4)
    )
    with torch.no_grad():
        target[0].weight[:, :256].copy_(source[0].weight)
        target[0].weight[:, 256:].zero_()
        target[0].bias.copy_(source[0].bias)
        target[2].weight.copy_(source[2].weight)
        target[2].bias.copy_(source[2].bias)
        target[3].weight.copy_(source[3].weight)
        target[3].bias.copy_(source[3].bias)
    return target


def _batch(data, indices, device):
    return {key: value[indices].to(device, non_blocking=True) for key, value in data.items()}


def evaluate_actor(actor, data, device, batch_size=1024):
    actor.eval()
    total_loss, correct, seen = 0.0, 0, 0
    with torch.inference_mode():
        for start in range(0, len(data["actions"]), batch_size):
            end = min(len(data["actions"]), start + batch_size)
            states = data["states"][start:end].to(device)
            actions = data["actions"][start:end].to(device)
            logits = actor(states)
            total_loss += F.cross_entropy(logits, actions, reduction="sum").item()
            correct += (logits.argmax(-1) == actions).sum().item()
            seen += len(actions)
    actor.train()
    return dict(action_nll=total_loss / max(seen, 1), action_accuracy=correct / max(seen, 1),
                samples=seen)


def evaluate_validation(actor, validation_data, device):
    """Use the worst heard/unheard validation NLL when both are available."""
    if not isinstance(validation_data, dict):
        return evaluate_actor(actor, validation_data.data, device)
    splits = {name: evaluate_actor(actor, data.data, device)
              for name, data in validation_data.items()}
    return dict(action_nll=max(row["action_nll"] for row in splits.values()),
                splits=splits)


def train_offline_rl(train_data, validation_data, foundation, config, output):
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])

    feature_dim = int(train_data.data["states"].shape[-1])
    actor = _initial_policy_head(foundation, feature_dim).to(device).requires_grad_(True)
    q1 = QNetwork(feature_dim=feature_dim, hidden_dim=config["hidden_dim"]).to(device)
    q2 = QNetwork(feature_dim=feature_dim, hidden_dim=config["hidden_dim"]).to(device)
    target_q1, target_q2 = deepcopy(q1).eval(), deepcopy(q2).eval()
    for target in (target_q1, target_q2):
        target.requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["learning_rate"])
    critic_optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                                        lr=config["learning_rate"])

    train_tensors = train_data.data
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history, best_nll = [], float("inf")
    tau, gamma = config["target_tau"], config["gamma"]
    alpha, cql_weight, bc_weight = config["alpha"], config["cql_weight"], config["bc_weight"]
    for epoch in range(1, config["epochs"] + 1):
        actor.train(); q1.train(); q2.train()
        sums = {key: 0.0 for key in (
            "critic_loss", "actor_loss", "sac_loss", "td_loss", "cql_gap", "bc_loss",
            "policy_entropy", "q_mean", "target_mean", "sample_reward_mean",
            "sample_terminal_fraction",
        )}
        for _ in tqdm(range(config["updates_per_epoch"]), desc=f"CQL {epoch}/{config['epochs']}", mininterval=10):
            indices = torch.randint(len(train_tensors["actions"]), (config["batch_size"],))
            data = _batch(train_tensors, indices, device)
            with torch.no_grad():
                next_logits = actor(data["next_states"])
                next_log_probs = F.log_softmax(next_logits, dim=-1)
                next_probs = next_log_probs.exp()
                next_q = torch.minimum(target_q1(data["next_states"]), target_q2(data["next_states"]))
                next_value = (next_probs * (next_q - alpha * next_log_probs)).sum(-1)
                target = data["rewards"] + gamma * (1.0 - data["dones"]) * next_value

            q_values_1, q_values_2 = q1(data["states"]), q2(data["states"])
            chosen_1 = q_values_1.gather(1, data["actions"].unsqueeze(1)).squeeze(1)
            chosen_2 = q_values_2.gather(1, data["actions"].unsqueeze(1)).squeeze(1)
            td_loss = F.mse_loss(chosen_1, target) + F.mse_loss(chosen_2, target)
            cql_gap = (torch.logsumexp(q_values_1, dim=-1) - chosen_1).mean() + (
                torch.logsumexp(q_values_2, dim=-1) - chosen_2).mean()
            critic_loss = td_loss + cql_weight * cql_gap
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 10.0)
            critic_optimizer.step()

            logits = actor(data["states"])
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            with torch.no_grad():
                policy_q = torch.minimum(q1(data["states"]), q2(data["states"]))
            policy_entropy = -(probs * log_probs).sum(-1).mean()
            sac_loss = (probs * (alpha * log_probs - policy_q)).sum(-1).mean()
            bc_loss = F.cross_entropy(logits, data["actions"])
            actor_loss = sac_loss + bc_weight * bc_loss
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            actor_optimizer.step()

            with torch.no_grad():
                for target_net, source_net in ((target_q1, q1), (target_q2, q2)):
                    for target_parameter, source_parameter in zip(target_net.parameters(), source_net.parameters()):
                        target_parameter.lerp_(source_parameter, tau)
            for key, value in (("critic_loss", critic_loss), ("actor_loss", actor_loss),
                               ("sac_loss", sac_loss), ("td_loss", td_loss),
                               ("cql_gap", cql_gap), ("bc_loss", bc_loss),
                               ("policy_entropy", policy_entropy),
                               ("q_mean", policy_q.mean()), ("target_mean", target.mean()),
                               ("sample_reward_mean", data["rewards"].mean()),
                               ("sample_terminal_fraction", data["dones"].mean())):
                sums[key] += float(value.detach().item())

        metrics = {key: value / config["updates_per_epoch"] for key, value in sums.items()}
        metrics["epoch"] = epoch
        metrics["validation"] = evaluate_validation(actor, validation_data, device)
        history.append(metrics)
        (output / "metrics.json.tmp").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (output / "metrics.json.tmp").replace(output / "metrics.json")
        checkpoint = dict(epoch=epoch, actor=actor.state_dict(), q1=q1.state_dict(), q2=q2.state_dict(),
                          target_q1=target_q1.state_dict(), target_q2=target_q2.state_dict(),
                          foundation_sha256=_sha256(config["foundation_checkpoint"]),
                          config=config, metrics=metrics)
        torch.save(checkpoint, output / f"epoch_{epoch:03d}.pt")
        if metrics["validation"]["action_nll"] < best_nll:
            best_nll = metrics["validation"]["action_nll"]
            torch.save(checkpoint, output / "best_policy.pt")
        torch.save(checkpoint, output / "latest.pt")
        print(json.dumps(metrics), flush=True)
    return history


def train_offline_iql(train_data, validation_data, foundation, config, output):
    """Train IQL with expectile value learning and advantage-weighted BC."""
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    feature_dim = int(train_data.data["states"].shape[-1])
    actor = _initial_policy_head(foundation, feature_dim).to(device).requires_grad_(True)
    q1 = QNetwork(feature_dim=feature_dim, hidden_dim=config["hidden_dim"]).to(device)
    q2 = QNetwork(feature_dim=feature_dim, hidden_dim=config["hidden_dim"]).to(device)
    value = ValueNetwork(feature_dim=feature_dim, hidden_dim=config["hidden_dim"]).to(device)
    target_value = deepcopy(value).eval().requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["learning_rate"])
    critic_optimizer = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()), lr=config["learning_rate"]
    )
    value_optimizer = torch.optim.Adam(value.parameters(), lr=config["learning_rate"])
    train_tensors = train_data.data
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history, best_nll = [], float("inf")
    tau, gamma = config["target_tau"], config["gamma"]
    expectile, beta = config["expectile"], config["beta"]
    for epoch in range(1, config["epochs"] + 1):
        actor.train(); q1.train(); q2.train(); value.train()
        sums = {key: 0.0 for key in ("critic_loss", "value_loss", "actor_loss",
                                     "advantage", "awr_weight")}
        for _ in tqdm(range(config["updates_per_epoch"]), desc=f"IQL {epoch}/{config['epochs']}",
                      mininterval=10):
            indices = torch.randint(len(train_tensors["actions"]), (config["batch_size"],))
            data = _batch(train_tensors, indices, device)
            with torch.no_grad():
                target = data["rewards"] + gamma * (1.0 - data["dones"]) * target_value(data["next_states"])
            q_values_1, q_values_2 = q1(data["states"]), q2(data["states"])
            chosen_1 = q_values_1.gather(1, data["actions"].unsqueeze(1)).squeeze(1)
            chosen_2 = q_values_2.gather(1, data["actions"].unsqueeze(1)).squeeze(1)
            critic_loss = F.mse_loss(chosen_1, target) + F.mse_loss(chosen_2, target)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 10.0)
            critic_optimizer.step()

            with torch.no_grad():
                q_min = torch.minimum(q1(data["states"]), q2(data["states"])).gather(
                    1, data["actions"].unsqueeze(1)
                ).squeeze(1)
            value_prediction = value(data["states"])
            difference = q_min - value_prediction
            expectile_weight = torch.where(difference > 0, expectile, 1.0 - expectile)
            value_loss = (expectile_weight * difference.square()).mean()
            value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            nn.utils.clip_grad_norm_(value.parameters(), 10.0)
            value_optimizer.step()

            advantage = q_min - value(data["states"]).detach()
            awr_weight = torch.exp(advantage / beta).clamp(max=100.0)
            logits = actor(data["states"])
            actor_loss = (awr_weight * F.cross_entropy(
                logits, data["actions"], reduction="none"
            )).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            actor_optimizer.step()

            with torch.no_grad():
                for target_parameter, source_parameter in zip(
                        target_value.parameters(), value.parameters()):
                    target_parameter.lerp_(source_parameter, tau)
            for key, item in (("critic_loss", critic_loss), ("value_loss", value_loss),
                              ("actor_loss", actor_loss), ("advantage", advantage.mean()),
                              ("awr_weight", awr_weight.mean())):
                sums[key] += float(item.detach().item())

        metrics = {key: val / config["updates_per_epoch"] for key, val in sums.items()}
        metrics["epoch"] = epoch
        metrics["validation"] = evaluate_validation(actor, validation_data, device)
        history.append(metrics)
        (output / "metrics.json.tmp").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (output / "metrics.json.tmp").replace(output / "metrics.json")
        checkpoint = dict(
            epoch=epoch, actor=actor.state_dict(), q1=q1.state_dict(), q2=q2.state_dict(),
            value=value.state_dict(), target_value=target_value.state_dict(),
            foundation_sha256=_sha256(config["foundation_checkpoint"]),
            config=config, metrics=metrics, algorithm="IQL",
        )
        torch.save(checkpoint, output / f"epoch_{epoch:03d}.pt")
        if metrics["validation"]["action_nll"] < best_nll:
            best_nll = metrics["validation"]["action_nll"]
            torch.save(checkpoint, output / "best_policy.pt")
        torch.save(checkpoint, output / "latest.pt")
        print(json.dumps(metrics), flush=True)
    return history


def run(config_path, gamma_override=None, output_override=None,
        bc_weight_override=None, cql_weight_override=None, algorithm_override=None,
        device_override=None, prepare_features_only=False):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    source_audit = _require_clean_data_audit(config)
    seed = int(config.get("SEED", 42))
    device = torch.device(config["FOUNDATION"]["DEVICE"])
    torch.set_num_threads(2)
    foundation_path = resolve_path(config["FOUNDATION"]["CHECKPOINT"])
    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(foundation_path, map_location=device, weights_only=True))
    foundation.eval().requires_grad_(False)
    rl = config["OFFLINE_RL"]
    if gamma_override is not None:
        rl["GAMMA"] = float(gamma_override)
    if bc_weight_override is not None:
        rl["BC_WEIGHT"] = float(bc_weight_override)
    if cql_weight_override is not None:
        rl["CQL_WEIGHT"] = float(cql_weight_override)
    algorithm = (algorithm_override or rl.get("ALGORITHM", "cql")).lower()
    if algorithm not in ("cql", "iql"):
        raise ValueError(f"Unsupported offline RL algorithm: {algorithm}")
    if device_override is not None:
        rl["DEVICE"] = device_override
    output = (resolve_path(output_override) if output_override
              else resolve_path(rl["OUTPUT"]))
    feature_cache_root = resolve_path(rl.get("FEATURE_CACHE_ROOT", str(output / "features")))
    feature_root = feature_cache_root / _sha256(foundation_path)[:16]
    data_sources = [("train", config["DATA"]["TRAIN_EPISODES"]),
                    ("validation", config["DATA"]["VALIDATION_EPISODES"])]
    if config["DATA"].get("VALIDATION_UNHEARD_EPISODES"):
        data_sources.append(("validation_unheard",
                             config["DATA"]["VALIDATION_UNHEARD_EPISODES"]))
    for split, source in data_sources:
        sources = source if isinstance(source, (list, tuple)) else [source]
        resolved_sources = [resolve_path(item) for item in sources]
        if len(resolved_sources) == 1:
            resolved_sources = resolved_sources[0]
        build_feature_dataset(
            resolved_sources, feature_root / split, foundation, foundation_path, device,
            config["FOUNDATION"]["EMBED_BATCH_SIZE"], config["FOUNDATION"]["FEATURE_SHARD_SIZE"],
            seed, action_history=bool(rl.get("ACTION_HISTORY", False)),
            collision_history=bool(rl.get("COLLISION_HISTORY", False)),
        )
    if prepare_features_only:
        return dict(status="features_prepared", feature_root=str(feature_root.resolve()),
                    source_audit=source_audit)
    train_data = FeatureDataset(feature_root / "train")
    validation_data = {split: FeatureDataset(feature_root / split)
                       for split, _ in data_sources if split.startswith("validation")}
    if len(validation_data) == 1:
        validation_data = next(iter(validation_data.values()))
    if isinstance(validation_data, dict):
        validation_sample_counts = {split: data.samples for split, data in validation_data.items()}
    else:
        validation_sample_counts = validation_data.samples
    print(f"Frozen foundation features: train={train_data.samples}, "
          f"validation={validation_sample_counts}", flush=True)
    training_config = dict(seed=seed, epochs=int(rl["EPOCHS"]),
                           updates_per_epoch=int(rl["UPDATES_PER_EPOCH"]),
                           batch_size=int(rl["BATCH_SIZE"]), learning_rate=float(rl["LEARNING_RATE"]),
                           gamma=float(rl["GAMMA"]),
                           target_tau=float(rl["TARGET_TAU"]), hidden_dim=int(rl["HIDDEN_DIM"]),
                           device=rl["DEVICE"], foundation_checkpoint=str(foundation_path),
                           action_history=bool(rl.get("ACTION_HISTORY", False)),
                           collision_history=bool(rl.get("COLLISION_HISTORY", False)))
    training_config["algorithm"] = algorithm
    if algorithm == "cql":
        training_config.update(alpha=float(rl["ALPHA"]),
                               cql_weight=float(rl["CQL_WEIGHT"]),
                               bc_weight=float(rl["BC_WEIGHT"]))
    else:
        training_config.update(expectile=float(rl.get("EXPECTILE", 0.7)),
                               beta=float(rl.get("IQL_BETA", 3.0)))
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    trainer = train_offline_iql if algorithm == "iql" else train_offline_rl
    history = trainer(train_data, validation_data, foundation, training_config, output)
    best = min(history, key=lambda row: row["validation"]["action_nll"])
    summary = dict(status="completed", algorithm=algorithm, training_config=training_config,
                   epochs=len(history), best_epoch=best["epoch"],
                   best_validation=best["validation"], train_samples=train_data.samples,
                   validation_samples=validation_sample_counts,
                   foundation_checkpoint=str(foundation_path),
                   data_source_audit=source_audit,
                   best_policy=str(output / "best_policy.pt"))
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiment/avnav/config/offline_rl.yaml")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--bc-weight", type=float, default=None)
    parser.add_argument("--cql-weight", type=float, default=None)
    parser.add_argument("--algorithm", choices=("cql", "iql"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prepare-features-only", action="store_true",
                        help="Extract configured train/validation features and exit")
    args = parser.parse_args(argv)
    started = time.time()
    history = run(args.config, args.gamma, args.output or None,
                  args.bc_weight, args.cql_weight, args.algorithm, args.device,
                  args.prepare_features_only)
    if isinstance(history, dict):
        print(json.dumps(history, indent=2), flush=True)
        return
    print(json.dumps(dict(status="completed", epochs=len(history), seconds=time.time() - started,
                          best_epoch=min(history, key=lambda row: row["validation"]["action_nll"])["epoch"]),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
