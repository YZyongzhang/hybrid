"""Train a train-split action-conditioned collision predictor and fine-tune AVNav."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from Dataset.dataset_avnav import get_files
from experiment.avnav.collision_predictor import ActionCollisionPredictor
from experiment.avnav.config import resolve_path
from experiment.avnav.model.hybrid import Network
from experiment.avnav.offline_rl import _collision_labels, _episode_features, _read_observation


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(directory):
    result = sorted(Path(path).resolve() for path in get_files(directory, (".pkl", ".pickle")))
    if not result:
        raise FileNotFoundError(f"No trajectory pickles found under {directory}")
    return result


def _episode_key(path):
    return path.parent.name, path.stem


def _read_trajectory_features(paths, foundation, device, batch_size, output_path,
                              foundation_sha256):
    states, actions, labels, keys, ranges = [], [], [], [], []
    foundation.eval()
    for path in paths:
        with path.open("rb") as handle:
            episode = pickle.load(handle)
        action_ids = np.asarray(episode["action_id"]).reshape(-1).astype(np.int64)
        collision = _collision_labels(episode, action_ids)
        features = _episode_features(
            episode, foundation, device, batch_size,
            action_history=True, collision_history=False,
        )
        # Match online reset semantics: no previous action is an all-zero vector.
        features["states"][0, 256:260] = 0.0
        if len(features["actions"]) != len(collision):
            raise ValueError(f"Collision labels do not align with {path}")
        begin = sum(len(row) for row in actions)
        states.append(features["states"].float())
        actions.append(features["actions"].long())
        labels.append(torch.from_numpy(collision.astype(np.float32)))
        keys.append(_episode_key(path))
        ranges.append({"scene": keys[-1][0], "episode_id": keys[-1][1],
                       "start": begin, "count": len(collision),
                       "collisions": int(collision.sum())})
    payload = {
        "states": torch.cat(states),
        "actions": torch.cat(actions),
        "collision": torch.cat(labels),
        "episodes": ranges,
        "foundation_sha256": foundation_sha256,
    }
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)
    return payload


def _predictor_logits(model, states, actions, device, batch_size=8192):
    logits = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(actions), batch_size):
            stop = min(len(actions), start + batch_size)
            batch_states = states[start:stop].to(device)
            action = actions[start:stop].to(device)
            candidate = F.one_hot(action, 4).to(batch_states.dtype)
            logits.append(model(batch_states[:, :256], batch_states[:, 256:260], candidate).cpu())
    return torch.cat(logits) if logits else torch.empty(0)


def _classification_metrics(logits, labels):
    labels = labels.float().cpu()
    probabilities = torch.sigmoid(logits.float().cpu())
    prediction = probabilities >= 0.5
    positive = labels >= 0.5
    tp = int((prediction & positive).sum())
    tn = int((~prediction & ~positive).sum())
    fp = int((prediction & ~positive).sum())
    fn = int((~prediction & positive).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        auc = float(roc_auc_score(labels.numpy(), probabilities.numpy()))
        average_precision = float(average_precision_score(labels.numpy(), probabilities.numpy()))
    except (ImportError, ValueError):
        auc = average_precision = float("nan")
    collision_rate = float(positive.float().mean()) if len(labels) else 0.0
    return {
        "samples": len(labels),
        "collision_rate": collision_rate,
        "majority_class_accuracy": max(collision_rate, 1.0 - collision_rate),
        "accuracy_at_0_5": (tp + tn) / max(len(labels), 1),
        "balanced_accuracy_at_0_5": (recall + specificity) / 2,
        "precision_at_0_5": tp / max(tp + fp, 1),
        "recall_at_0_5": recall,
        "specificity_at_0_5": specificity,
        "f1_at_0_5": 2 * tp / max(2 * tp + fp + fn, 1),
        "roc_auc": auc,
        "average_precision": average_precision,
        "bce": float(F.binary_cross_entropy_with_logits(logits.float().cpu(), labels)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _per_action_metrics(logits, labels, actions):
    names = ("STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT")
    result = {}
    actions = actions.long().cpu()
    for action_id, name in enumerate(names):
        mask = actions == action_id
        if mask.any():
            result[name] = _classification_metrics(logits[mask], labels[mask])
        else:
            result[name] = {"samples": 0}
    return result


def _train_predictor(train, heldout, device, output, seed, epochs=30, batch_size=512):
    torch.manual_seed(seed)
    random.seed(seed)
    model = ActionCollisionPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_positive = max(float(train["collision"].sum()), 1.0)
    n_negative = max(float(len(train["collision"]) - n_positive), 1.0)
    pos_weight = torch.tensor([n_negative / n_positive], device=device).clamp(0.25, 4.0)
    best_selection = (-float("inf"), float("inf"))
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train["actions"]))
        loss_sum = seen = 0
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            states = train["states"][ids].to(device)
            actions = train["actions"][ids].to(device)
            labels = train["collision"][ids].to(device)
            candidate = F.one_hot(actions, 4).to(states.dtype)
            logits = model(states[:, :256], states[:, 256:260], candidate)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            seen += len(ids)
        heldout_logits = _predictor_logits(model, heldout["states"], heldout["actions"], device)
        metrics = _classification_metrics(heldout_logits, heldout["collision"])
        metrics["per_action"] = _per_action_metrics(
            heldout_logits, heldout["collision"], heldout["actions"]
        )
        row = {"epoch": epoch, "train_bce": loss_sum / max(seen, 1), "heldout": metrics}
        history.append(row)
        print(json.dumps({"stage": "collision_head", **row}), flush=True)
        selection_key = (metrics["balanced_accuracy_at_0_5"], -metrics["bce"])
        if selection_key > best_selection:
            best_selection = selection_key
            stale = 0
            torch.save({"predictor": model.state_dict(), "epoch": epoch,
                        "metrics": metrics, "pos_weight": float(pos_weight.item()),
                        "input_schema": "embedding256+previous_action_onehot4+candidate_action_onehot4",
                        "candidate_action_order": ["STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT"]},
                       output / "collision_head_pretrained.pt")
        else:
            stale += 1
            if stale >= 6:
                break
    checkpoint = torch.load(output / "collision_head_pretrained.pt", map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint["predictor"])
    (output / "collision_head_pretrain_metrics.json").write_text(
        json.dumps({"history": history, "selected_epoch": checkpoint["epoch"],
                    "selected_heldout": checkpoint["metrics"]}, indent=2) + "\n")
    return model, checkpoint


def _raw_batches(path, batch_size):
    with Path(path).open("rb") as handle:
        episode = pickle.load(handle)
    observations = episode["obs"]
    actions = np.asarray(episode["action_id"]).reshape(-1).astype(np.int64)
    collisions = _collision_labels(episode, actions).astype(np.float32)
    if len(observations) != len(actions) + 1:
        raise ValueError(f"Observation/action mismatch in {path}")
    prepared = [_read_observation(row) for row in observations]
    zero_rgb, zero_depth = torch.zeros_like(prepared[0][1]), torch.zeros_like(prepared[0][2])
    audio_rows, rgb_rows, depth_rows = [], [], []
    previous_ids = np.full(len(actions), -1, dtype=np.int64)
    if len(actions) > 1:
        previous_ids[1:] = actions[:-1]
    for index in range(len(actions)):
        previous_rgb, previous_depth = (zero_rgb, zero_depth) if index == 0 else (
            prepared[index - 1][1], prepared[index - 1][2]
        )
        audio_rows.append(prepared[index][0])
        rgb_rows.append(torch.cat((previous_rgb, prepared[index][1]), dim=-1))
        depth_rows.append(torch.cat((previous_depth, prepared[index][2]), dim=-1))
    for start in range(0, len(actions), batch_size):
        stop = min(len(actions), start + batch_size)
        yield (
            torch.stack(audio_rows[start:stop]),
            torch.stack(rgb_rows[start:stop]),
            torch.stack(depth_rows[start:stop]),
            torch.from_numpy(actions[start:stop]),
            torch.from_numpy(collisions[start:stop]),
            torch.from_numpy(previous_ids[start:stop]),
        )


def _evaluate_foundation(model, predictor, paths, device, batch_size=32):
    model.eval(); predictor.eval()
    logits_all, labels_all, actions_all = [], [], []
    action_correct = action_seen = 0
    with torch.inference_mode():
        for path in paths:
            for audio, rgb, depth, actions, labels, previous_ids in _raw_batches(path, batch_size):
                audio, rgb, depth = audio.to(device), rgb.to(device), depth.to(device)
                actions, labels = actions.to(device), labels.to(device)
                previous_ids = previous_ids.to(device)
                previous = torch.zeros((len(previous_ids), 4), dtype=torch.float32,
                                       device=device)
                valid_previous = previous_ids >= 0
                if valid_previous.any():
                    previous[valid_previous] = F.one_hot(
                        previous_ids[valid_previous], 4).float()
                embedding = model.embedding_forward(audio, rgb, depth)
                action_logits = model.final.action_head(embedding)
                candidate = F.one_hot(actions, 4).float()
                logits = predictor(embedding, previous, candidate)
                logits_all.append(logits.cpu()); labels_all.append(labels.cpu())
                actions_all.append(actions.cpu())
                action_correct += int((action_logits.argmax(-1) == actions).sum())
                action_seen += len(actions)
    collision_metrics = _classification_metrics(torch.cat(logits_all), torch.cat(labels_all))
    collision_metrics["per_action"] = _per_action_metrics(
        torch.cat(logits_all), torch.cat(labels_all), torch.cat(actions_all)
    )
    collision_metrics["behavior_action_accuracy"] = action_correct / max(action_seen, 1)
    return collision_metrics


def _fine_tune_foundation(train_paths, heldout_paths, base_foundation, predictor,
                          device, output, seed, epochs=4, batch_size=32):
    torch.manual_seed(seed)
    teacher = Network().to(device)
    teacher.load_state_dict(base_foundation.state_dict())
    teacher.eval().requires_grad_(False)
    model = deepcopy(base_foundation).to(device).requires_grad_(True)
    predictor = deepcopy(predictor).to(device)
    collision_total = 0
    transition_total = 0
    for path in train_paths:
        with path.open("rb") as handle:
            episode = pickle.load(handle)
        actions = np.asarray(episode["action_id"]).reshape(-1).astype(np.int64)
        labels = _collision_labels(episode, actions)
        transition_total += len(labels); collision_total += int(labels.sum())
    pos_weight = torch.tensor(
        [(transition_total - collision_total) / max(collision_total, 1)], device=device
    ).clamp(0.25, 4.0)
    foundation_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=1e-4
    )
    predictor_optimizer = torch.optim.AdamW(predictor.parameters(), lr=2e-4,
                                            weight_decay=1e-4)
    history = []
    best_selection = (-float("inf"), float("inf"))
    for epoch in range(1, epochs + 1):
        model.train(); predictor.train()
        order = list(train_paths); random.Random(seed + epoch).shuffle(order)
        totals = {"collision_loss": 0.0, "action_loss": 0.0, "distill_loss": 0.0,
                  "samples": 0}
        for path in order:
            for audio, rgb, depth, actions, labels, previous_ids in _raw_batches(path, batch_size):
                audio, rgb, depth = audio.to(device), rgb.to(device), depth.to(device)
                actions, labels = actions.to(device), labels.to(device)
                previous_ids = previous_ids.to(device)
                previous = torch.zeros((len(previous_ids), 4), dtype=torch.float32,
                                       device=device)
                valid_previous = previous_ids >= 0
                if valid_previous.any():
                    previous[valid_previous] = F.one_hot(
                        previous_ids[valid_previous], 4).float()
                candidate = F.one_hot(actions, 4).float()
                with torch.no_grad():
                    target_embedding = teacher.embedding_forward(audio, rgb, depth)
                embedding = model.embedding_forward(audio, rgb, depth)
                action_logits = model.final.action_head(embedding)
                collision_logits = predictor(embedding, previous, candidate)
                collision_loss = F.binary_cross_entropy_with_logits(
                    collision_logits, labels, pos_weight=pos_weight
                )
                action_loss = F.cross_entropy(action_logits, actions)
                distill_loss = F.mse_loss(embedding, target_embedding)
                loss = collision_loss + 0.5 * action_loss + 0.25 * distill_loss
                foundation_optimizer.zero_grad(set_to_none=True)
                predictor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)
                foundation_optimizer.step(); predictor_optimizer.step()
                count = len(actions)
                totals["collision_loss"] += float(collision_loss.detach()) * count
                totals["action_loss"] += float(action_loss.detach()) * count
                totals["distill_loss"] += float(distill_loss.detach()) * count
                totals["samples"] += count
        for key in ("collision_loss", "action_loss", "distill_loss"):
            totals[key] /= max(totals["samples"], 1)
        heldout = _evaluate_foundation(model, predictor, heldout_paths, device, batch_size)
        row = {"epoch": epoch, "train": totals, "heldout": heldout}
        history.append(row)
        print(json.dumps({"stage": "foundation_finetune", **row}), flush=True)
        selection_key = (heldout["balanced_accuracy_at_0_5"], -heldout["bce"])
        if selection_key > best_selection:
            best_selection = selection_key
            torch.save({
                "foundation": model.state_dict(),
                "predictor": predictor.state_dict(),
                "epoch": epoch,
                "heldout_metrics": heldout,
                "foundation_sha256": None,
                "predictor_input_schema": "embedding256+previous_action_onehot4+candidate_action_onehot4",
            }, output / "collision_foundation_best.pt")
    (output / "foundation_finetune_metrics.json").write_text(
        json.dumps({"history": history, "selected_epoch": history[
            max(range(len(history)), key=lambda i: (
                history[i]["heldout"]["balanced_accuracy_at_0_5"],
                -history[i]["heldout"]["bce"],
            ))
        ]["epoch"]}, indent=2) + "\n"
    )
    return torch.load(output / "collision_foundation_best.pt", map_location="cpu",
                      weights_only=True)


def train(train_directory, heldout_directory, foundation_path, output, device="cuda:0",
          seed=20261025, head_epochs=30, foundation_epochs=4, batch_size=32):
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    train_paths, heldout_paths = _paths(train_directory), _paths(heldout_directory)
    train_keys = {_episode_key(path) for path in train_paths}
    heldout_keys = {_episode_key(path) for path in heldout_paths}
    if train_keys & heldout_keys:
        raise ValueError("Predictor train and holdout episodes overlap")
    base_path = resolve_path(foundation_path)
    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(base_path, map_location=device, weights_only=True))
    foundation.eval().requires_grad_(False)
    train_features_path = output / "predictor_train_features.pt"
    heldout_features_path = output / "predictor_holdout_features.pt"
    train_features = _read_trajectory_features(
        train_paths, foundation, device, batch_size, train_features_path,
        _sha256(base_path),
    )
    heldout_features = _read_trajectory_features(
        heldout_paths, foundation, device, batch_size, heldout_features_path,
        _sha256(base_path),
    )
    head, head_record = _train_predictor(
        train_features, heldout_features, device, output, seed, head_epochs
    )
    foundation_record = _fine_tune_foundation(
        train_paths, heldout_paths, foundation, head, device, output, seed,
        foundation_epochs, batch_size
    )
    tuned_path = output / "collision_foundation_best.pt"
    base_sha = _sha256(base_path)
    tuned_record = torch.load(tuned_path, map_location="cpu", weights_only=True)
    foundation_out = output / "foundation.pt"
    predictor_out = output / "collision_predictor.pt"
    torch.save(tuned_record["foundation"], foundation_out)
    torch.save({
        "predictor": tuned_record["predictor"],
        "foundation_sha256": _sha256(foundation_out),
        "base_foundation_sha256": base_sha,
        "selected_foundation_epoch": tuned_record["epoch"],
        "input_schema": tuned_record["predictor_input_schema"],
        "candidate_action_order": ["STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT"],
        "heldout_metrics": tuned_record["heldout_metrics"],
    }, predictor_out)
    source_manifest = {
        "status": "completed",
        "algorithm": "action-conditioned collision prediction",
        "base_foundation": str(base_path),
        "base_foundation_sha256": base_sha,
        "train_trajectory_directory": str(Path(train_directory).resolve()),
        "heldout_trajectory_directory": str(Path(heldout_directory).resolve()),
        "train_episodes": len(train_paths),
        "heldout_episodes": len(heldout_paths),
        "train_transitions": int(len(train_features["actions"])),
        "train_collisions": int(train_features["collision"].sum()),
        "heldout_transitions": int(len(heldout_features["actions"])),
        "heldout_collisions": int(heldout_features["collision"].sum()),
        "train_episode_keys_sha256": hashlib.sha256(
            "\n".join(f"{s}/{e}" for s, e in sorted(train_keys)).encode()).hexdigest(),
        "heldout_episode_keys_sha256": hashlib.sha256(
            "\n".join(f"{s}/{e}" for s, e in sorted(heldout_keys)).encode()).hexdigest(),
        "selection": "collision-head and fine-tuned foundation selected by maximum scene-held-out train-split balanced accuracy at fixed threshold 0.5, tie-break by minimum BCE; no validation/test examples used to fit the predictor",
        "collision_head_pretraining": {
            "selected_epoch": int(head_record["epoch"]),
            "pos_weight": float(head_record["pos_weight"]),
            "heldout_metrics": head_record["metrics"],
        },
        "fine_tuned_foundation": {
            "selected_epoch": tuned_record["epoch"],
            "heldout_metrics": tuned_record["heldout_metrics"],
        },
        "artifacts": {"foundation": str(foundation_out),
                      "collision_predictor": str(predictor_out)},
    }
    (output / "training_summary.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n"
    )
    return source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trajectories", required=True)
    parser.add_argument("--heldout-trajectories", required=True)
    parser.add_argument("--foundation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--foundation-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(train(args.train_trajectories, args.heldout_trajectories,
                           args.foundation, args.output, args.device, args.seed,
                           args.head_epochs, args.foundation_epochs, args.batch_size),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
