"""Fit immediate and short-horizon collision risks from train-only branches."""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
import yaml

from Dataset.dataset_avnav import get_files
from experiment.avnav.collision_predictor import MultiHorizonActionCollisionPredictor
from experiment.avnav.config import resolve_path
from experiment.avnav.model.hybrid import Network
from experiment.avnav.offline_rl import (
    _collision_labels,
    _read_observation,
    _require_clean_data_audit,
    _sha256,
)


def _paths(sources):
    sources = sources if isinstance(sources, (list, tuple)) else [sources]
    result = []
    for source in sources:
        result.extend(Path(path).resolve()
                      for path in get_files(resolve_path(source), (".pkl", ".pickle")))
    return sorted(set(result))


def _embed_episode(episode, foundation, device, batch_size):
    observations = episode["obs"]
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
    foundation.eval()
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            stop = min(len(observations), start + batch_size)
            embeddings.append(foundation.embedding_forward(
                torch.stack(audios[start:stop]).to(device),
                torch.stack(rgbs[start:stop]).to(device),
                torch.stack(depths[start:stop]).to(device),
            ).float().cpu())
    return torch.cat(embeddings)


def _previous_actions(actions):
    previous = torch.zeros(len(actions), dtype=torch.long)
    if len(actions) > 1:
        previous[1:] = torch.as_tensor(actions[:-1], dtype=torch.long)
    one_hot = F.one_hot(previous, 4).float()
    one_hot[0].zero_()
    return one_hot


def _future_any(collision, index, horizon):
    stop = min(len(collision), index + horizon)
    return float(bool(np.asarray(collision[index + 1:stop]).any()))


def _collect_rows(paths, foundation, device, batch_size, horizon,
                  counterfactual_label_root=None, source_name="data"):
    embeddings, previous_actions, actions, immediate, future = [], [], [], [], []
    labels_by_key = {}
    if counterfactual_label_root:
        roots = (list(counterfactual_label_root)
                 if isinstance(counterfactual_label_root, (list, tuple))
                 else [counterfactual_label_root])
        for label_root in roots:
            for path in sorted(Path(label_root).rglob("*.jsonl")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (str(row["scene"]), str(row["episode_id"]))
                    labels_by_key.setdefault(key, []).append(row)
    used_counterfactual_files = set()
    episode_count = 0
    logged_count = 0
    for path in tqdm(paths, desc=f"Predictor rows {source_name}", mininterval=10):
        with path.open("rb") as handle:
            episode = pickle.load(handle)
        action_ids = np.asarray(episode["action_id"]).reshape(-1).astype(np.int64)
        count = len(action_ids)
        if len(episode["obs"]) != count + 1:
            raise ValueError(f"Observation/action mismatch in {path}")
        collision = _collision_labels(episode, action_ids).astype(np.float32)
        state_embeddings = _embed_episode(episode, foundation, device, batch_size)
        if len(state_embeddings) != count + 1:
            raise ValueError(f"Feature/action mismatch in {path}")
        previous = _previous_actions(action_ids)
        embeddings.append(state_embeddings[:count])
        previous_actions.append(previous[:count])
        actions.append(torch.from_numpy(action_ids).long())
        immediate.append(torch.from_numpy(collision))
        future.append(torch.tensor([_future_any(collision, index, horizon)
                                    for index in range(count)], dtype=torch.float32))
        episode_count += 1
        logged_count += count

        scene = str(episode.get("parent_scene", path.parent.name))
        episode_id = str(episode.get("parent_episode_id", path.stem))
        key = (scene, episode_id)
        rows = labels_by_key.get(key, [])
        if rows:
            used_counterfactual_files.add(key)
            for row in rows:
                anchor = int(row["anchor_step"])
                candidate = int(row["candidate_action_id"])
                if not 0 <= anchor < count:
                    raise ValueError(f"Counterfactual anchor out of range: {key}/{anchor}")
                embeddings.append(state_embeddings[anchor:anchor + 1])
                previous_actions.append(previous[anchor:anchor + 1])
                actions.append(torch.tensor([candidate], dtype=torch.long))
                immediate.append(torch.tensor([float(row["immediate_collision"])],
                                              dtype=torch.float32))
                future.append(torch.tensor([
                    float(row["future_any_collision_after_first"])
                ], dtype=torch.float32))
    if not embeddings:
        raise ValueError(f"No predictor rows found for {source_name}")
    data = {"embedding": torch.cat(embeddings),
            "previous_action": torch.cat(previous_actions),
            "actions": torch.cat(actions),
            "immediate": torch.cat(immediate),
            "future": torch.cat(future)}
    return data, {"episodes": episode_count, "logged_samples": logged_count,
        "counterfactual_anchor_episodes": len(used_counterfactual_files),
        "samples": len(data["actions"])}


def _batch_metrics(logits, labels):
    labels = labels.float().cpu()
    probabilities = torch.sigmoid(logits.float().cpu())
    predicted = probabilities >= 0.5
    positive = labels >= 0.5
    tp = int((predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "samples": len(labels), "collision_rate": float(positive.float().mean()),
        "balanced_accuracy_at_0_5": (recall + specificity) / 2,
        "precision_at_0_5": tp / max(tp + fp, 1),
        "recall_at_0_5": recall, "specificity_at_0_5": specificity,
        "bce": float(F.binary_cross_entropy_with_logits(logits.float().cpu(), labels)),
        "brier_score": float(((probabilities - labels) ** 2).mean()),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _predict(model, data, device, batch_size=8192):
    immediate, future = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data["actions"]), batch_size):
            stop = min(len(data["actions"]), start + batch_size)
            embedding = data["embedding"][start:stop].to(device)
            previous = data["previous_action"][start:stop].to(device)
            candidate = F.one_hot(data["actions"][start:stop].to(device), 4).float()
            immediate.append(model(embedding, previous, candidate).cpu())
            future.append(model.forward_future(embedding, previous, candidate).cpu())
    return torch.cat(immediate), torch.cat(future)


def train(config_path, counterfactual_parent_manifest, output_dir,
          epochs=30, batch_size=512, horizon=5,
          validation_counterfactual_label_root=None):
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty predictor output: {output}")
    audit_path = _require_clean_data_audit(config)
    data_cfg = config["DATA"]
    if not counterfactual_parent_manifest:
        raise ValueError("Counterfactual parent manifest is required")
    parent_manifest = Path(counterfactual_parent_manifest).expanduser().resolve()
    labels_root = parent_manifest.parent / "candidate_labels"
    validation_label_root = (Path(validation_counterfactual_label_root).expanduser().resolve()
                             if validation_counterfactual_label_root else None)
    foundation_path = resolve_path(config["FOUNDATION"]["CHECKPOINT"])
    device = torch.device(config["FOUNDATION"]["DEVICE"])
    seed = int(config.get("SEED", 20261025))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    foundation = Network().to(device)
    foundation.load_state_dict(torch.load(foundation_path, map_location=device,
                                          weights_only=True))
    foundation.eval().requires_grad_(False)

    train_paths = _paths(data_cfg["TRAIN_EPISODES"])
    validation_paths = _paths([data_cfg["VALIDATION_EPISODES"],
                               data_cfg["VALIDATION_UNHEARD_EPISODES"]])
    validation_protocols = []
    validation_keys = set()
    if validation_label_root is not None:
        for protocol_path in sorted(validation_label_root.rglob("collection_protocol.json")):
            validation_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            audit_path = protocol_path.parent / "collection_quality_audit.json"
            if (not audit_path.is_file() or
                    json.loads(audit_path.read_text(encoding="utf-8")).get("status") != "pass"):
                raise ValueError(f"Validation counterfactual audit did not pass: {audit_path}")
            if (validation_protocol.get("episode_role") != "validation" or
                    validation_protocol.get("split") not in
                    ("val_multiple", "val_multiple_unheard") or
                    validation_protocol.get("validation_labels_used_for_training") is not False):
                raise ValueError(f"Non-selection data found under validation label root: {protocol_path}")
            validation_protocols.append({
                "protocol": str(protocol_path.resolve()),
                "protocol_sha256": _sha256(protocol_path),
                "quality_audit": str(audit_path.resolve()),
                "quality_audit_sha256": _sha256(audit_path),
                "split": validation_protocol["split"],
                "parent_manifest": validation_protocol["episode_manifest"],
                "parent_manifest_sha256": validation_protocol["episode_manifest_sha256"],
            })
            parent_path = protocol_path.parent / "parent_episode_manifest.jsonl"
            for line in parent_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("error") or row.get("role") != "validation":
                    raise ValueError(f"Invalid validation-only candidate labels: {parent_path}")
                validation_keys.add((str(row["scene"]), str(row["episode_id"])))
        if not validation_protocols or not validation_keys:
            raise ValueError("Validation counterfactual label root contains no selection labels")
        validation_file_keys = {(path.parent.name, path.stem) for path in validation_paths}
        if not validation_keys <= validation_file_keys:
            raise ValueError("Validation candidate-label episodes do not match validation sources")
        train_parent_rows = [json.loads(line) for line in parent_manifest.read_text().splitlines()
                             if line.strip()]
        train_keys = {(str(row["scene"]), str(row["episode_id"]))
                      for row in train_parent_rows}
        if train_keys & validation_keys:
            raise ValueError("Train and validation counterfactual episode keys overlap")
    train_data, train_source = _collect_rows(
        train_paths, foundation, device,
        int(config["FOUNDATION"].get("EMBED_BATCH_SIZE", 32)), horizon,
        labels_root, "train",
    )
    validation_rollout_paths = []
    validation_data_parts = []
    validation_source_parts = []
    for validation_protocol in validation_protocols:
        protocol_path = Path(validation_protocol["protocol"])
        rollout_paths = _paths([protocol_path.parent / "trajectories"])
        label_dir = protocol_path.parent / "candidate_labels"
        split_data, split_source = _collect_rows(
            rollout_paths, foundation, device,
            int(config["FOUNDATION"].get("EMBED_BATCH_SIZE", 32)), horizon,
            [label_dir], f"validation_counterfactual_{validation_protocol['split']}",
        )
        validation_rollout_paths.extend(rollout_paths)
        validation_data_parts.append(split_data)
        validation_source_parts.append({"split": validation_protocol["split"],
                                        **split_source})
    if not validation_data_parts:
        raise ValueError("No validation counterfactual rollout sources are available")
    validation_data = {
        key: torch.cat([part[key] for part in validation_data_parts], dim=0)
        for key in validation_data_parts[0]
    }
    validation_source = {
        "episodes": sum(row["episodes"] for row in validation_source_parts),
        "logged_samples": sum(row["logged_samples"] for row in validation_source_parts),
        "counterfactual_anchor_episodes": sum(
            row["counterfactual_anchor_episodes"] for row in validation_source_parts),
        "samples": sum(row["samples"] for row in validation_source_parts),
        "splits": validation_source_parts,
    }
    model = MultiHorizonActionCollisionPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    class_weights = {}
    for name in ("immediate", "future"):
        positive = max(float(train_data[name].sum()), 1.0)
        negative = max(float(len(train_data[name]) - positive), 1.0)
        class_weights[name] = torch.tensor([negative / positive], device=device).clamp(0.25, 4.0)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "frozen_before_predictor_training",
        "config": str(config_path), "config_sha256": _sha256(config_path),
        "data_source_audit": audit_path,
        "data_source_audit_sha256": _sha256(audit_path) if audit_path else None,
        "counterfactual_parent_manifest": str(parent_manifest),
        "counterfactual_parent_manifest_sha256": _sha256(parent_manifest),
        "counterfactual_label_root": str(labels_root.resolve()),
        "validation_counterfactual_label_root": (
            str(validation_label_root) if validation_label_root else None),
        "validation_counterfactual_protocols": validation_protocols,
        "foundation_checkpoint": str(foundation_path),
        "foundation_sha256": _sha256(foundation_path),
        "train_sources": [str(path) for path in train_paths],
        "validation_behavior_sources": [str(path) for path in validation_paths],
        "validation_counterfactual_rollout_sources": [str(path) for path in validation_rollout_paths],
        "train_source_counts": train_source,
        "validation_source_counts": validation_source,
        "horizon": horizon,
        "training_labels": "immediate collision and any collision during continuation after candidate action, with raw frozen actor continuation",
        "model_selection": "maximize mean immediate/future balanced accuracy on simulator counterfactual validation labels at threshold 0.5; tie-break by lower mean BCE",
        "validation_used_for_training": False,
    }
    (output / "predictor_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")

    best_key = (-float("inf"), float("inf"))
    history, stale = [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_data["actions"]))
        loss_sum, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            embedding = train_data["embedding"][ids].to(device)
            previous = train_data["previous_action"][ids].to(device)
            actions = train_data["actions"][ids].to(device)
            candidates = F.one_hot(actions, 4).to(embedding.dtype)
            immediate_target = train_data["immediate"][ids].to(device)
            future_target = train_data["future"][ids].to(device)
            immediate_logits = model(embedding, previous, candidates)
            future_logits = model.forward_future(embedding, previous, candidates)
            immediate_loss = F.binary_cross_entropy_with_logits(
                immediate_logits, immediate_target, pos_weight=class_weights["immediate"]
            )
            future_loss = F.binary_cross_entropy_with_logits(
                future_logits, future_target, pos_weight=class_weights["future"]
            )
            loss = 0.5 * (immediate_loss + future_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            seen += len(ids)

        immediate_logits, future_logits = _predict(model, validation_data, device)
        immediate_metrics = _batch_metrics(immediate_logits, validation_data["immediate"])
        future_metrics = _batch_metrics(future_logits, validation_data["future"])
        mean_balanced = (immediate_metrics["balanced_accuracy_at_0_5"] +
                         future_metrics["balanced_accuracy_at_0_5"]) / 2
        mean_bce = (immediate_metrics["bce"] + future_metrics["bce"]) / 2
        row = {"epoch": epoch, "train_bce": loss_sum / max(seen, 1),
               "validation_immediate": immediate_metrics,
               "validation_future": future_metrics,
               "validation_mean_balanced_accuracy": mean_balanced,
               "validation_mean_bce": mean_bce}
        history.append(row)
        print(json.dumps(row), flush=True)
        current_key = (mean_balanced, -mean_bce)
        if current_key > best_key:
            best_key, stale = current_key, 0
            checkpoint = {
                "predictor": model.state_dict(), "epoch": epoch,
                "foundation_sha256": _sha256(foundation_path),
                "input_schema": "embedding256+previous_action_onehot4+candidate_action_onehot4",
                "outputs": ["immediate_collision", f"collision_after_candidate_within_{horizon-1}_continuation_steps"],
                "candidate_action_order": ["STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT"],
                "horizon": horizon, "metrics": row,
                "selection_rule": protocol["model_selection"],
            }
            torch.save(checkpoint, output / "best_collision_predictor.pt")
        else:
            stale += 1
            if stale >= 6:
                break
    (output / "predictor_training_metrics.json").write_text(
        json.dumps({"history": history, "selected_epoch": min(
            history, key=lambda row: (-row["validation_mean_balanced_accuracy"],
                                      row["validation_mean_bce"])
        )["epoch"]}, indent=2) + "\n", encoding="utf-8")
    return {"status": "completed", "selected_epoch": torch.load(
        output / "best_collision_predictor.pt", map_location="cpu",
        weights_only=True)["epoch"], "training_protocol": str(output / "predictor_protocol.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--counterfactual-parent-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--validation-counterfactual-label-root", default="")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.counterfactual_parent_manifest,
                           args.output, args.epochs, args.batch_size, args.horizon,
                           args.validation_counterfactual_label_root or None),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
