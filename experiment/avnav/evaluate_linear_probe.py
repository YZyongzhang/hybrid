"""Evaluate a linear action probe on frozen AVNav foundation embeddings."""
import argparse
import json
import pickle
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset.dataset_avnav import AVNavDataset, get_files
from experiment.avnav.model.hybrid import Network
from experiment.avnav.offline_rl import _episode_features


def _extract_train(model, train_pt, device, batch_size):
    dataset = AVNavDataset(get_files(train_pt, (".pt",)), cache_size=2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feature_parts, action_parts = [], []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extract train embeddings"):
            audio, rgb, depth = (value.to(device, non_blocking=True) for value in batch[:3])
            features = model.embedding_forward(audio, rgb, depth)
            feature_parts.append(features.float().cpu())
            action_parts.append(batch[4].long().cpu())
    return torch.cat(feature_parts), torch.cat(action_parts)


def _extract_raw_split(model, source, device, batch_size):
    files = get_files(source, (".pkl", ".pickle"))
    features, actions, scenes, sounds = [], [], [], []
    model.eval()
    for path in tqdm(files, desc=f"Extract {Path(source).name}", mininterval=10):
        with Path(path).open("rb") as handle:
            episode = pickle.load(handle)
        transitions = _episode_features(episode, model, device, batch_size)
        episode_actions = transitions["actions"]
        sound_ids = np.asarray(episode.get("sound_id", "unknown"), dtype=object).reshape(-1)
        if len(sound_ids) == 1:
            episode_sounds = [str(sound_ids[0])] * len(episode_actions)
        elif len(sound_ids) in (len(episode_actions), len(episode_actions) + 1):
            episode_sounds = [str(value) for value in sound_ids[:len(episode_actions)]]
        else:
            raise ValueError(f"{path}: sound_id count does not match transitions")
        features.append(transitions["states"])
        actions.append(episode_actions)
        scenes.extend([path.parent.name] * len(episode_actions))
        sounds.extend(episode_sounds)
    return {
        "features": torch.cat(features),
        "actions": torch.cat(actions),
        "scenes": scenes,
        "sounds": sounds,
        "episodes": len(files),
    }


def _fit_linear_probe(features, actions, device, seed, epochs, batch_size):
    torch.manual_seed(seed)
    model = nn.Linear(features.shape[1], 4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    inputs = features.to(device)
    targets = actions.to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(epochs):
        indices = torch.randperm(len(targets), generator=generator)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size].to(device)
            logits = model(inputs[batch_indices])
            loss = F.cross_entropy(logits, targets[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _metric_row(logits, labels):
    labels = labels.long().cpu()
    predictions = logits.argmax(dim=1).long().cpu()
    confusion = torch.bincount(labels * 4 + predictions, minlength=16).reshape(4, 4).float()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    recall = confusion.diag() / support.clamp_min(1)
    precision = confusion.diag() / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    present = support > 0
    return {
        "samples": int(len(labels)),
        "accuracy": float((predictions == labels).float().mean()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "nll": float(F.cross_entropy(logits.float().cpu(), labels)),
        "class_support": [int(value) for value in support.tolist()],
        "confusion_matrix": [[int(value) for value in row] for row in confusion.tolist()],
    }


def _evaluate(logits, labels, groups=None):
    report = {"overall": _metric_row(logits, labels)}
    if groups is None:
        return report
    grouped = {}
    for group_type, values in groups.items():
        rows = {}
        for group in sorted(set(values)):
            indices = torch.tensor([index for index, value in enumerate(values) if value == group])
            rows[group] = _metric_row(logits[indices], labels[indices])
        mean_metrics = {
            key: float(np.mean([row[key] for row in rows.values()]))
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "nll")
        }
        grouped[group_type] = {"macro_mean": mean_metrics, "groups": rows}
    report["by_group"] = grouped
    return report


def _predict_heads(model, probe, features, device, batch_size):
    original_logits, probe_logits = [], []
    model.eval()
    probe.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = features[start:start + batch_size].to(device)
            original_logits.append(model.final.action_head(batch).float().cpu())
            probe_logits.append(probe(batch).float().cpu())
    return torch.cat(original_logits), torch.cat(probe_logits)


def _scene_names(root):
    return sorted(path.name for path in Path(root).iterdir() if path.is_dir())


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    allowed_cache_files = {"heard_features.pt", "unheard_features.pt"}
    unexpected = {path.name for path in output.iterdir()} - allowed_cache_files
    if unexpected:
        raise FileExistsError(f"Choose an empty output directory; found: {sorted(unexpected)}")

    model = Network().to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    train_manifest = json.loads((Path(args.train_pt) / "manifest.json").read_text(encoding="utf-8"))
    train_episode_root = Path(train_manifest["source"])
    train_scenes = set(_scene_names(train_episode_root))
    heard_scenes = set(_scene_names(args.heard_episodes))
    unheard_scenes = set(_scene_names(args.unheard_episodes))
    if train_scenes & (heard_scenes | unheard_scenes):
        raise ValueError("Training and validation episode sets share scene IDs")

    train_features, train_actions = _extract_train(
        model, args.train_pt, device, args.extract_batch_size
    )
    probe = _fit_linear_probe(
        train_features, train_actions, device, args.seed, args.epochs, args.probe_batch_size
    )
    train_original, train_probe = _predict_heads(
        model, probe, train_features, device, args.extract_batch_size
    )
    report = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "train_features": str(Path(args.train_pt).resolve()),
            "validation_episodes": {
                "heard": str(Path(args.heard_episodes).resolve()),
                "unheard": str(Path(args.unheard_episodes).resolve()),
            },
            "frozen_feature": "Network.embedding_forward, 256 dimensions",
            "probe": "single nn.Linear(256, 4), cross entropy, AdamW",
            "probe_epochs": args.epochs,
            "probe_seed": args.seed,
            "scene_group_split": "training scenes and both validation scene sets are disjoint",
            "scene_counts": {
                "train": len(train_scenes),
                "heard": len(heard_scenes),
                "unheard": len(unheard_scenes),
                "heard_unheard_overlap": len(heard_scenes & unheard_scenes),
            },
            "validation_scenes_overlap": sorted(heard_scenes & unheard_scenes),
            "validation_used_for_probe_selection": False,
        },
        "train": {
            "samples": len(train_actions),
            "original_action_head": _metric_row(train_original, train_actions),
            "linear_probe": _metric_row(train_probe, train_actions),
        },
        "validation": {},
    }

    for split, source in (("heard", args.heard_episodes), ("unheard", args.unheard_episodes)):
        feature_cache = output / f"{split}_features.pt"
        if feature_cache.exists():
            data = torch.load(feature_cache, map_location="cpu", weights_only=False)
            data["episodes"] = len(get_files(source, (".pkl", ".pickle")))
        else:
            data = _extract_raw_split(model, source, device, args.extract_batch_size)
        original_logits, probe_logits = _predict_heads(
            model, probe, data["features"], device, args.extract_batch_size
        )
        groups = {"scene": data["scenes"], "sound": data["sounds"]}
        report["validation"][split] = {
            "episodes": data["episodes"],
            "original_action_head": _evaluate(original_logits, data["actions"], groups),
            "linear_probe": _evaluate(probe_logits, data["actions"], groups),
        }
        if not feature_cache.exists():
            torch.save({"features": data["features"], "actions": data["actions"],
                        "scenes": data["scenes"], "sounds": data["sounds"]}, feature_cache)
        del data

    torch.save(probe.cpu().state_dict(), output / "linear_probe.pt")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "train": report["train"],
        "validation": {
            split: {
                head: {
                    key: value["overall"][key]
                    for key in ("samples", "accuracy", "balanced_accuracy", "macro_f1", "nll")
                }
                for head in ("original_action_head", "linear_probe")
                for value in (metrics[head],)
            }
            for split, metrics in report["validation"].items()
        },
    }, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="result/2026-10-13-avnav-staged-grl-gamma/foundation/best.pth")
    parser.add_argument("--train-pt", default="/data/Getuanhui/avnav-hybrid-mutiple/pt/train")
    parser.add_argument("--heard-episodes", default="/data/Getuanhui/avnav-hybrid-mutiple/pickle/hybird(val_multiple)")
    parser.add_argument("--unheard-episodes", default="/data/Getuanhui/avnav-hybrid-mutiple/pickle/hybird(val_multiple_unheard)")
    parser.add_argument("--output", default="result/2026-10-14-avnav-linear-probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--probe-batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--cpu-threads", type=int, default=2)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
