"""Two-stage audio/foundation training for the AVNav codex experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Dataset.dataset_avnav import AVNavDataset, ShardBatchSampler, get_files
from experiment.avnav.hybrid.foundation_model import Network


TRAIN_PT = Path("/data/Getuanhui/avnav-hybrid-mutiple/pt/train")
VAL_PT = Path("/data/Getuanhui/avnav-hybrid-mutiple/pt/val")
UNHEARD_VAL_PT = Path("/data/Getuanhui/avnav-hybrid-mutiple/pt/val_unheard")
VAL_PICKLES = Path("/data/Getuanhui/avnav-hybrid-mutiple/pickle/hybird(val_multiple)")
UNHEARD_VAL_PICKLES = Path(
    "/data/Getuanhui/avnav-hybrid-mutiple/pickle/hybird(val_multiple_unheard)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _save_state(path: Path, state_dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save({key: value.detach().cpu() for key, value in state_dict.items()}, temp)
    temp.replace(path)
    return _sha256(path)


def _parameter_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _angle_labels(angle_degrees: torch.Tensor) -> torch.Tensor:
    return torch.floor(torch.remainder(angle_degrees.float() + 180.0, 360.0) / 45.0).long().clamp_(0, 7)


def _make_loader(path: Path, batch_size: int, train: bool, seed: int) -> DataLoader:
    data = AVNavDataset(get_files(path, (".pt",)), cache_size=2)
    if train:
        return DataLoader(
            data,
            batch_sampler=ShardBatchSampler(data, batch_size, seed),
            num_workers=0,
            pin_memory=True,
        )
    return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


def _batch_to_device(batch, device):
    return tuple(value.to(device, non_blocking=True) for value in batch)


@torch.inference_mode()
def _evaluate_audio(model, loader, device):
    model.audio_encoder.eval()
    loss_sum = correct = direction_count = sample_count = 0
    for batch in loader:
        audio, _, _, angle, action = _batch_to_device(batch, device)
        valid = action != 0
        if not valid.any():
            continue
        logits = model.audio_encoder(audio)
        labels = _angle_labels(angle)
        loss_sum += F.cross_entropy(logits[valid], labels[valid], reduction="sum").item()
        correct += int((logits[valid].argmax(-1) == labels[valid]).sum())
        direction_count += int(valid.sum())
        sample_count += len(action)
    return {
        "angle_nll": loss_sum / max(direction_count, 1),
        "angle_accuracy": correct / max(direction_count, 1),
        "direction_samples": direction_count,
        "samples": sample_count,
    }


def _train_audio_epoch(model, loader, device, optimizer):
    model.audio_encoder.train()
    loss_sum = correct = direction_count = sample_count = 0
    for batch in loader:
        audio, _, _, angle, action = _batch_to_device(batch, device)
        valid = action != 0
        if not valid.any():
            continue
        labels = _angle_labels(angle)
        logits = model.audio_encoder(audio)
        loss = F.cross_entropy(logits[valid], labels[valid])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite audio-stage loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.audio_encoder.parameters(), 10.0)
        optimizer.step()
        loss_sum += float(loss.detach()) * int(valid.sum())
        correct += int((logits.detach()[valid].argmax(-1) == labels[valid]).sum())
        direction_count += int(valid.sum())
        sample_count += len(action)
    return {
        "angle_nll": loss_sum / max(direction_count, 1),
        "angle_accuracy": correct / max(direction_count, 1),
        "direction_samples": direction_count,
        "samples": sample_count,
    }


@torch.inference_mode()
def _evaluate_action(model, loader, device):
    model.eval()
    model.audio_encoder.eval()
    action_loss_sum = angle_loss_sum = action_correct = angle_correct = seen = 0
    for batch in loader:
        audio, rgb, depth, angle, action = _batch_to_device(batch, device)
        _, angle_logits, action_logits = model(audio, rgb, depth)
        angle_labels = _angle_labels(angle)
        action_loss_sum += F.cross_entropy(action_logits, action, reduction="sum").item()
        angle_loss_sum += F.cross_entropy(angle_logits, angle_labels, reduction="sum").item()
        action_correct += int((action_logits.argmax(-1) == action).sum())
        angle_correct += int((angle_logits.argmax(-1) == angle_labels).sum())
        seen += len(action)
    return {
        "action_nll": action_loss_sum / max(seen, 1),
        "action_accuracy": action_correct / max(seen, 1),
        "fused_angle_nll": angle_loss_sum / max(seen, 1),
        "fused_angle_accuracy": angle_correct / max(seen, 1),
        "samples": seen,
    }


def _train_action_epoch(model, loader, device, optimizer, angle_weight):
    model.train()
    model.audio_encoder.eval()
    action_loss_sum = angle_loss_sum = action_correct = angle_correct = seen = 0
    for batch in loader:
        audio, rgb, depth, angle, action = _batch_to_device(batch, device)
        _, angle_logits, action_logits = model(audio, rgb, depth)
        angle_labels = _angle_labels(angle)
        action_loss = F.cross_entropy(action_logits, action)
        angle_loss = F.cross_entropy(angle_logits, angle_labels)
        loss = action_loss + angle_weight * angle_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite foundation-stage loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        optimizer.step()
        action_loss_sum += float(action_loss.detach()) * len(action)
        angle_loss_sum += float(angle_loss.detach()) * len(action)
        action_correct += int((action_logits.detach().argmax(-1) == action).sum())
        angle_correct += int((angle_logits.detach().argmax(-1) == angle_labels).sum())
        seen += len(action)
    return {
        "action_nll": action_loss_sum / max(seen, 1),
        "action_accuracy": action_correct / max(seen, 1),
        "fused_angle_nll": angle_loss_sum / max(seen, 1),
        "fused_angle_accuracy": angle_correct / max(seen, 1),
        "samples": seen,
    }


def run(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    audit_path = run_dir / "audit/data_source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Training blocked: data-source audit is not pass")
    train_manifest = TRAIN_PT / "manifest.json"
    val_manifest = VAL_PT / "manifest.json"
    unheard_manifest = UNHEARD_VAL_PT / "manifest.json"
    for path, expected_source in (
        (train_manifest, "/data/Getuanhui/avnav-hybrid-mutiple/pickle/hybird(heard_train_multiple)"),
        (val_manifest, str(VAL_PICKLES)),
        (unheard_manifest, str(UNHEARD_VAL_PICKLES)),
    ):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if Path(manifest["source"]).resolve() != Path(expected_source).resolve():
            raise ValueError(f"Unexpected PT source in {path}: {manifest.get('source')}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    train_loader = _make_loader(TRAIN_PT, args.batch_size, True, args.seed)
    val_loaders = {
        "heard": _make_loader(VAL_PT, args.batch_size, False, args.seed),
        "unheard": _make_loader(UNHEARD_VAL_PT, args.batch_size, False, args.seed),
    }
    out = run_dir / "foundation"
    audio_out = out / "audio_stage"
    action_out = out / "action_stage"
    if (audio_out / "metrics.jsonl").exists() or (action_out / "metrics.jsonl").exists():
        raise FileExistsError('Refusing to overwrite an existing foundation training run')
    audio_out.mkdir(parents=True, exist_ok=True)
    action_out.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "experiment.avnav.hybrid.foundation_model.Network",
        "data_source": str(TRAIN_PT),
        "train_manifest_sha256": _sha256(train_manifest),
        "validation_manifests": {
            "heard": {"path": str(val_manifest), "sha256": _sha256(val_manifest)},
            "unheard": {"path": str(unheard_manifest), "sha256": _sha256(unheard_manifest)},
        },
        "source_audit": str(audit_path),
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "audio_epochs": args.audio_epochs,
        "action_epochs": args.action_epochs,
        "audio_learning_rate": args.audio_lr,
        "action_learning_rate": args.action_lr,
        "action_angle_auxiliary_weight": args.angle_weight,
        "selection_rule": "minimum worst-split validation action NLL; audio stage uses worst-split validation angle NLL",
        "validation_used_for_gradient_updates": False,
        "audio_frozen_during_action_stage": True,
        "audio_normalization": "per-sample over frequency/time/both ears; population std; epsilon 1e-6",
        "visual_tokens": "flatten spatial dimensions then transpose channels",
        "position_encoding": "sinusoidal 10000 inverse frequencies; device-following buffer",
    }
    _atomic_json(out / "training_config.json", config)

    model = Network().to(device)
    audio_optimizer = torch.optim.Adam(model.audio_encoder.parameters(), lr=args.audio_lr)
    audio_history = []
    best_audio_score = float("inf")
    for epoch in range(1, args.audio_epochs + 1):
        started = time.monotonic()
        train_metrics = _train_audio_epoch(model, train_loader, device, audio_optimizer)
        val_metrics = {name: _evaluate_audio(model, loader, device)
                       for name, loader in val_loaders.items()}
        score = max(value["angle_nll"] for value in val_metrics.values())
        row = {"stage": "audio", "epoch": epoch, "train": train_metrics,
               "validation": val_metrics, "selection_score": score,
               "seconds": time.monotonic() - started}
        audio_history.append(row)
        with (audio_out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        state_hash = _save_state(audio_out / "latest.pth", model.audio_encoder.state_dict())
        if score < best_audio_score:
            best_audio_score = score
            best_hash = _save_state(audio_out / "best.pth", model.audio_encoder.state_dict())
            _atomic_json(audio_out / "best.json", {
                "epoch": epoch, "selection_score_worst_validation_angle_nll": score,
                "validation": val_metrics, "checkpoint": str(audio_out / "best.pth"),
                "checkpoint_sha256": best_hash,
            })
        _atomic_json(audio_out / "latest.json", {"epoch": epoch, "checkpoint_sha256": state_hash})

    audio_best = audio_out / "best.pth"
    model.audio_encoder.load_state_dict(torch.load(audio_best, map_location=device, weights_only=True))
    for parameter in model.audio_encoder.parameters():
        parameter.requires_grad_(False)
    model.audio_encoder.eval()
    frozen_before = _parameter_digest(model.audio_encoder)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.action_lr,
    )
    action_history = []
    best_action_score = float("inf")
    for epoch in range(1, args.action_epochs + 1):
        started = time.monotonic()
        train_metrics = _train_action_epoch(
            model, train_loader, device, optimizer, args.angle_weight
        )
        val_metrics = {name: _evaluate_action(model, loader, device)
                       for name, loader in val_loaders.items()}
        score = max(value["action_nll"] for value in val_metrics.values())
        row = {"stage": "action", "epoch": epoch, "train": train_metrics,
               "validation": val_metrics, "selection_score": score,
               "seconds": time.monotonic() - started}
        action_history.append(row)
        with (action_out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        latest_hash = _save_state(action_out / "latest.pth", model.state_dict())
        if score < best_action_score:
            best_action_score = score
            best_hash = _save_state(action_out / "best.pth", model.state_dict())
            _atomic_json(action_out / "best.json", {
                "epoch": epoch, "selection_score_worst_validation_action_nll": score,
                "validation": val_metrics, "checkpoint": str(action_out / "best.pth"),
                "checkpoint_sha256": best_hash,
            })
        _atomic_json(action_out / "latest.json", {"epoch": epoch, "checkpoint_sha256": latest_hash})

    frozen_after = _parameter_digest(model.audio_encoder)
    if frozen_before != frozen_after:
        raise RuntimeError("Frozen audio parameters changed during foundation action training")
    _atomic_json(action_out / "frozen_audio_audit.json", {
        "status": "pass", "parameters_sha256_before": frozen_before,
        "parameters_sha256_after": frozen_after,
        "audio_checkpoint": str(audio_best), "audio_checkpoint_sha256": _sha256(audio_best),
        "action_stage_checkpoint": str(action_out / "best.pth"),
        "audio_frozen": True,
    })
    _atomic_json(out / "summary.json", {
        "status": "completed", "audio_epochs": args.audio_epochs,
        "action_epochs": args.action_epochs,
        "best_audio": json.loads((audio_out / "best.json").read_text()),
        "best_foundation": json.loads((action_out / "best.json").read_text()),
        "frozen_audio_audit": str(action_out / "frozen_audio_audit.json"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="result/2026-10-27-avnav-codex-full-training")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--audio-epochs", type=int, default=8)
    parser.add_argument("--action-epochs", type=int, default=15)
    parser.add_argument("--audio-lr", type=float, default=1e-4)
    parser.add_argument("--action-lr", type=float, default=3e-5)
    parser.add_argument("--angle-weight", type=float, default=0.2)
    parser.add_argument("--cpu-threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
