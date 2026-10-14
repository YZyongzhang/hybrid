"""AVNAV two-stage foundation training with AV-WaN's audio GRL strategy."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from Dataset.dataset_avnav import AVNavDataset, ShardBatchSampler, get_files
from experiment.avnav.config import resolve_path
from experiment.avnav.model.hybrid import Network


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def angle_to_sector(angle_degrees):
    labels = torch.floor(torch.remainder(angle_degrees.float() + 180.0, 360.0) / 45.0)
    return labels.long().clamp_(0, 7)


def build_data(config, seed, batch_size, workers):
    train_dir = resolve_path(config["DATA"]["TRAIN_PT"])
    heard_dir = resolve_path(config["DATA"]["VALIDATION_PT"])
    unheard_dir = resolve_path(config["DATA"]["UNHEARD_PT"])
    for path in (train_dir, heard_dir, unheard_dir):
        if not (path / "manifest.json").is_file():
            raise FileNotFoundError(f"Expected converted AVNAV dataset manifest: {path}")
    train_manifest = json.loads((train_dir / "manifest.json").read_text(encoding="utf-8"))
    classes = train_manifest.get("sound_label_to_id", {})
    if not classes:
        raise ValueError("Training manifest has no sound_label_to_id map")
    class_names = sorted(classes, key=lambda name: classes[name])
    if sorted(classes.values()) != list(range(len(class_names))):
        raise ValueError("Training sound IDs must be contiguous and start at zero")
    shuffled = list(class_names)
    random.Random(seed).shuffle(shuffled)
    pseudo_unheard = set(shuffled[:len(shuffled) // 2])
    label_to_domain = torch.tensor(
        [int(name in pseudo_unheard) for name in class_names], dtype=torch.long
    )
    dirs = {"train": train_dir, "val_heard": heard_dir, "val_unheard": unheard_dir}
    datasets = {}
    for split, directory in dirs.items():
        files = get_files(directory, (".pt",))
        datasets[split] = AVNavDataset(
            files, cache_size=2, include_sound_ids=(split == "train")
        )
    train_loader = DataLoader(
        datasets["train"],
        batch_sampler=ShardBatchSampler(datasets["train"], batch_size, seed),
        num_workers=workers, pin_memory=False, persistent_workers=workers > 0,
    )
    loaders = {"train": train_loader}
    loaders.update({
        split: DataLoader(data, batch_size=batch_size, shuffle=False,
                          num_workers=workers, pin_memory=False,
                          persistent_workers=workers > 0)
        for split, data in datasets.items() if split != "train"
    })
    split_details = {
        "seed": seed,
        "method": "deterministic half-class pseudo-unheard split of training sound labels",
        "pseudo_unheard_sound_classes": sorted(pseudo_unheard),
        "pseudo_heard_sound_classes": sorted(set(class_names) - pseudo_unheard),
        "training_sound_classes": class_names,
        "independent_validation_sources": {
            "heard": train_manifest_source(heard_dir),
            "unheard": train_manifest_source(unheard_dir),
            "used_for_audio_and_action_checkpoint_selection": True,
            "used_for_training": False,
        },
    }
    return loaders, datasets, label_to_domain, split_details


def train_audio_epoch(model, classifier, loader, device, label_to_domain,
                      grl_scale, domain_weight, optimizer=None, scaler=None):
    update = optimizer is not None
    model.train(update)
    model.audio_encoder.train(update)
    classifier.train(update)
    totals = {"loss": 0., "angle_loss": 0., "angle_accuracy": 0.,
              "domain_loss": 0., "domain_accuracy": 0.}
    seen = valid_seen = 0
    start_time = time.monotonic()
    grad_context = torch.enable_grad if update else torch.inference_mode
    with grad_context():
        for batch in loader:
            if update and len(batch) != 6:
                raise ValueError("Audio training batches must include sound_ids for GRL supervision")
            audio, _, _, angle, action = batch[:5]
            audio = audio.to(device, non_blocking=True)
            angle = angle.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            valid = action != 0  # AVNAV action 0 is STOP and has no target direction.
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                token = model.audio_encoder.encoder_forward(audio).squeeze(1).float()
                angle_logits = model.audio_encoder.fnn(token)
                angle_labels = angle_to_sector(angle)
                angle_loss = (F.cross_entropy(angle_logits[valid], angle_labels[valid])
                              if valid.any() else angle_logits.sum() * 0.)
                if update:
                    sound_id = batch[5].to(device, non_blocking=True)
                    domain_logits = classifier(ReverseGradient.apply(token, grl_scale))
                    domain_labels = label_to_domain[sound_id]
                    domain_loss = F.cross_entropy(domain_logits, domain_labels)
                    loss = angle_loss + domain_weight * domain_loss
                else:
                    domain_logits = domain_labels = None
                    domain_loss = angle_loss.detach() * 0.
                    loss = angle_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite AVNAV audio stage loss")
            if update:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.audio_encoder.parameters()) + list(classifier.parameters()), 10.
                )
                scaler.step(optimizer)
                scaler.update()
            count = len(action)
            seen += count
            valid_seen += int(valid.sum())
            totals["loss"] += float(loss.detach()) * count
            totals["angle_loss"] += float(angle_loss.detach()) * count
            totals["angle_accuracy"] += float(((angle_logits.argmax(-1) == angle_labels) & valid).sum())
            if update:
                totals["domain_loss"] += float(domain_loss.detach()) * count
                totals["domain_accuracy"] += float((domain_logits.argmax(-1) == domain_labels).sum())
    return {
        "loss": totals["loss"] / max(seen, 1),
        "angle_loss": totals["angle_loss"] / max(seen, 1),
        "angle_accuracy": totals["angle_accuracy"] / max(valid_seen, 1),
        "samples": seen, "direction_samples": valid_seen,
        "seconds": time.monotonic() - start_time,
    }


def forward_action(model, batch, device):
    audio, rgb, depth = (value.to(device, non_blocking=True) for value in batch[:3])
    actions = batch[4].to(device, non_blocking=True)
    features = model.embedding_forward(audio, rgb, depth)
    return model.final.action_head(features), actions


def train_action_epoch(model, loader, device, optimizer=None, scaler=None):
    update = optimizer is not None
    model.train(update)
    model.audio_encoder.eval()
    loss_sum = correct = seen = 0
    started = time.monotonic()
    grad_context = torch.enable_grad if update else torch.inference_mode
    with grad_context():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits, actions = forward_action(model, batch, device)
                loss = F.cross_entropy(logits.float(), actions)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite AVNAV action stage loss")
            if update:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 10.
                )
                scaler.step(optimizer)
                scaler.update()
            count = len(actions)
            loss_sum += float(loss.detach()) * count
            correct += int((logits.argmax(-1) == actions).sum())
            seen += count
    return {"action_loss": loss_sum / max(seen, 1),
            "action_accuracy": correct / max(seen, 1), "samples": seen,
            "seconds": time.monotonic() - started}


def train_manifest_source(directory):
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8")).get("source", str(directory))


def run(config_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    training = config["FOUNDATION"]
    seed = int(training["SEED"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(training["DEVICE"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(float(training["GPU_MEMORY_FRACTION"]), device)
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(training["CPU_THREADS"]))
    output = resolve_path(training["OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Choose an empty staged foundation output directory: {output}")
    loaders, datasets, label_to_domain, split_details = build_data(
        config, seed, int(training["BATCH_SIZE"]), int(training["WORKERS"])
    )
    label_to_domain = label_to_domain.to(device)
    model = Network().to(device)
    classifier = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2)).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    audio_dir = output / "audio_stage"
    action_dir = output / "action_stage"
    audio_dir.mkdir()
    action_dir.mkdir()
    experiment = {
        "method": "AV-WaN two-stage foundation: audio angle + GRL sound-domain adaptation, then frozen audio with visual/fusion/action training",
        "seed": seed, "device": str(device), "data": config["DATA"],
        "training": training, "pseudo_unheard_split": split_details,
        "selection": {
            "audio": "minimum worst angle cross entropy across heard and independent unheard validation",
            "action": "minimum worst action cross entropy across heard and independent unheard validation",
        },
        "stop_action_id_masked_from_angle_loss": 0,
    }
    write_json(output / "experiment_config.json", experiment)
    write_json(output / "dataset_counts.json", {key: len(value) for key, value in datasets.items()})
    write_json(output / "status.json", {"status": "audio_training"})
    audio_parameters = list(model.audio_encoder.parameters()) + list(classifier.parameters())
    audio_optimizer = torch.optim.AdamW(
        audio_parameters, lr=float(training["AUDIO_LEARNING_RATE"]), weight_decay=1e-4
    )
    validation_splits = ("val_heard", "val_unheard")
    best_audio = float("inf")
    for epoch in range(1, int(training["AUDIO_EPOCHS"]) + 1):
        row = {"epoch": epoch}
        row["train"] = train_audio_epoch(
            model, classifier, loaders["train"], device, label_to_domain,
            float(training["GRL_SCALE"]), float(training["DOMAIN_LOSS_WEIGHT"]),
            audio_optimizer, scaler,
        )
        for split in validation_splits:
            row[split] = train_audio_epoch(
                model, classifier, loaders[split], device, label_to_domain,
                float(training["GRL_SCALE"]), float(training["DOMAIN_LOSS_WEIGHT"]),
            )
        row["worst_validation_angle_loss"] = max(row[s]["angle_loss"] for s in validation_splits)
        with (audio_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        payload = {"model": model.state_dict(), "sound_classifier": classifier.state_dict(),
                   "epoch": epoch, "metrics": row, "config": experiment}
        torch.save(payload, audio_dir / "latest.pt")
        if row["worst_validation_angle_loss"] < best_audio:
            best_audio = row["worst_validation_angle_loss"]
            torch.save(payload, audio_dir / "best.pt")
            write_json(audio_dir / "best.json", {
                "epoch": epoch, "worst_validation_angle_loss": best_audio,
                "checkpoint_sha256": sha256(audio_dir / "best.pt"),
            })
        print(json.dumps({"stage": "audio", **row}), flush=True)

    write_json(output / "status.json", {"status": "action_training"})
    audio_state = torch.load(audio_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(audio_state["model"], strict=True)
    model.requires_grad_(True)
    model.audio_encoder.requires_grad_(False)
    model.audio_encoder.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    write_json(action_dir / "trainable_parameters.json", {
        "trainable_count": sum(parameter.numel() for parameter in trainable),
        "frozen_audio_count": sum(parameter.numel() for parameter in model.audio_encoder.parameters()),
        "trainable_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
    })
    action_optimizer = torch.optim.AdamW(
        trainable, lr=float(training["ACTION_LEARNING_RATE"]), weight_decay=1e-4
    )
    action_scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_action = float("inf")
    for epoch in range(1, int(training["ACTION_EPOCHS"]) + 1):
        row = {"epoch": epoch,
               "train": train_action_epoch(model, loaders["train"], device,
                                            action_optimizer, action_scaler)}
        for split in validation_splits:
            row[split] = train_action_epoch(model, loaders[split], device)
        row["worst_validation_action_loss"] = max(
            row[split]["action_loss"] for split in validation_splits
        )
        with (action_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        payload = {"model": model.state_dict(), "epoch": epoch, "metrics": row,
                   "audio_checkpoint": str(audio_dir / "best.pt"),
                   "audio_checkpoint_sha256": sha256(audio_dir / "best.pt"),
                   "config": experiment}
        torch.save(payload, action_dir / "latest.pt")
        if row["worst_validation_action_loss"] < best_action:
            best_action = row["worst_validation_action_loss"]
            torch.save(payload, action_dir / "best.pt")
            write_json(action_dir / "best.json", {
                "epoch": epoch, "worst_validation_action_loss": best_action,
                "checkpoint_sha256": sha256(action_dir / "best.pt"),
            })
        print(json.dumps({"stage": "action", **row}), flush=True)

    best_state = torch.load(action_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model"], strict=True)
    model.requires_grad_(True)
    model.audio_encoder.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frozen_audio_matches = all(
        torch.equal(model.state_dict()[name].cpu(), audio_state["model"][name].cpu())
        for name in audio_state["model"] if name.startswith("audio_encoder.")
    )
    if not frozen_audio_matches:
        raise AssertionError("Frozen audio weights changed during the action stage")
    write_json(action_dir / "frozen_audio_audit.json", {
        "passed": frozen_audio_matches,
        "audio_checkpoint_sha256": sha256(audio_dir / "best.pt"),
        "checked_parameters": sum(name.startswith("audio_encoder.") for name in model.state_dict()),
    })
    checkpoint = output / "best.pth"
    torch.save(model.state_dict(), checkpoint)
    write_json(output / "status.json", {
        "status": "foundation_trained", "best_audio_validation_angle_loss": best_audio,
        "best_action_validation_action_loss": best_action,
        "foundation_checkpoint": str(checkpoint.resolve()),
        "foundation_checkpoint_sha256": sha256(checkpoint),
    })
    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiment/avnav/config/foundation_staged.yaml")
    args = parser.parse_args(argv)
    checkpoint = run(args.config)
    print(json.dumps({"status": "foundation_trained", "checkpoint": str(checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
