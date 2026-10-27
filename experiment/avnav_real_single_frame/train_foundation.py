"""Train a one-frame RGB AVNav foundation; audio and split protocol match avnav_real."""

import argparse
import bisect
import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader, Sampler

from Dataset.dataset_avnav import get_files
from experiment.avnav_real_single_frame.model import NetworkSingleFrameNoDepth


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SingleFrameActionDataset(Dataset):
    """Read current RGB from cached [previous RGB, current RGB] tensors."""

    def __init__(self, root, cache_size=2):
        root = Path(root)
        self.files = [Path(path) for path in get_files(root, (".pt",))]
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        counts = {row["file"]: int(row["samples"])
                  for row in manifest.get("shards", [])}
        self.ends = []
        total = 0
        for path in self.files:
            count = counts.get(path.name)
            if count is None:
                count = len(torch.load(path, map_location="cpu",
                                       weights_only=True)["actions"])
            total += count
            self.ends.append(total)
        if not total:
            raise ValueError(f"No action samples in {root}")
        self.samples = total
        self.cache_size = int(cache_size)
        self.cache = OrderedDict()

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        shard_index = bisect.bisect_right(self.ends, index)
        offset = index - (self.ends[shard_index - 1] if shard_index else 0)
        if shard_index not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
            self.cache[shard_index] = torch.load(
                self.files[shard_index], map_location="cpu", weights_only=True
            )
        payload = self.cache[shard_index]
        self.cache.move_to_end(shard_index)
        stacked = payload["rgb"][offset]
        if stacked.ndim != 3 or stacked.shape[-1] != 6:
            raise ValueError(f"Expected cached two-frame RGB, got {tuple(stacked.shape)}")
        # Dataset.dataset_avnav stores torch.cat((previous_rgb, current_rgb), -1).
        current_rgb = stacked[..., 3:6].contiguous()
        return (payload["audios"][offset].float(), current_rgb.float(),
                payload["actions"][offset].long())


class ShardBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed):
        self.ends = dataset.ends
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        starts = [0] + self.ends[:-1]
        shards = list(range(len(self.ends)))
        rng.shuffle(shards)
        for shard in shards:
            ids = list(range(starts[shard], self.ends[shard]))
            rng.shuffle(ids)
            for start in range(0, len(ids), self.batch_size):
                yield ids[start:start + self.batch_size]

    def __len__(self):
        starts = [0] + self.ends[:-1]
        return sum((end - start + self.batch_size - 1) // self.batch_size
                   for start, end in zip(starts, self.ends))


def _evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = samples = 0
    predicted = torch.zeros(4, dtype=torch.long)
    labels_count = torch.zeros(4, dtype=torch.long)
    confusion = torch.zeros((4, 4), dtype=torch.long)
    with torch.inference_mode():
        for audio, rgb, actions in loader:
            audio = audio.to(device, non_blocking=True)
            rgb = rgb.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            logits = model.final.action_head(model.embedding_forward(audio, rgb))
            guesses = logits.argmax(-1)
            loss_sum += F.cross_entropy(logits.float(), actions,
                                        reduction="sum").item()
            correct += int((guesses == actions).sum())
            samples += len(actions)
            predicted += torch.bincount(guesses.cpu(), minlength=4)
            labels_count += torch.bincount(actions.cpu(), minlength=4)
            confusion += torch.bincount(
                (actions * 4 + guesses).cpu(), minlength=16
            ).reshape(4, 4)
    recalls = confusion.diag().float() / confusion.sum(1).clamp_min(1)
    return {
        "action_nll": loss_sum / max(samples, 1),
        "action_accuracy": correct / max(samples, 1),
        "samples": samples,
        "predicted_action_counts": predicted.tolist(),
        "demonstrated_action_counts": labels_count.tolist(),
        "per_action_recall": recalls.tolist(),
        "confusion_rows_true_cols_predicted": confusion.tolist(),
    }


def run(config_path):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    audit_path = Path(cfg["DATA"]["SOURCE_AUDIT"]).resolve()
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "pass":
        raise ValueError(f"Training source audit did not pass: {audit_path}")
    out = Path(cfg["FOUNDATION"]["OUTPUT"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run artifacts: {out}")
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["SEED"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(cfg["FOUNDATION"]["DEVICE"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Configured device {device} is unavailable")
    torch.set_num_threads(2)

    train = SingleFrameActionDataset(cfg["DATA"]["TRAIN_PT"])
    batch_size = int(cfg["FOUNDATION"]["BATCH_SIZE"])
    train_loader = DataLoader(
        train, batch_sampler=ShardBatchSampler(train, batch_size, seed),
        num_workers=int(cfg["FOUNDATION"].get("WORKERS", 0)), pin_memory=False,
    )
    validation = {}
    for name, key in (("val_multiple", "VALIDATION_PT"),
                      ("val_multiple_unheard", "VALIDATION_UNHEARD_PT")):
        dataset = SingleFrameActionDataset(cfg["DATA"][key])
        validation[name] = (dataset, DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        ))

    model = NetworkSingleFrameNoDepth().to(device)
    audio_init = Path(cfg["FOUNDATION"]["AUDIO_INIT_CHECKPOINT"]).resolve()
    audio_payload = torch.load(audio_init, map_location="cpu", weights_only=False)
    audio_state = audio_payload.get("model", audio_payload)
    audio_state = {key[len("audio_encoder."):]: value
                   for key, value in audio_state.items()
                   if key.startswith("audio_encoder.")}
    if not audio_state:
        raise ValueError(f"No audio encoder weights found in {audio_init}")
    model.audio_encoder.load_state_dict(audio_state, strict=True)
    model.audio_encoder.requires_grad_(False).eval()

    cfg_record = {
        "architecture": NetworkSingleFrameNoDepth.architecture,
        "visual_input": "single current RGB frame only; depth and previous RGB are not passed",
        "rgb_cache_slice": "channels 3:6 from [previous_rgb,current_rgb]",
        "visual_input_channels": 3,
        "train_split_source": train.files[0].parent.as_posix(),
        "train_samples": train.samples,
        "validation_samples": {
            name: dataset.samples for name, (dataset, _) in validation.items()
        },
        "seed": seed,
        "epochs": int(cfg["FOUNDATION"]["EPOCHS"]),
        "batch_size": batch_size,
        "learning_rate": float(cfg["FOUNDATION"]["LEARNING_RATE"]),
        "weight_decay": float(cfg["FOUNDATION"]["WEIGHT_DECAY"]),
        "frozen_audio": True,
        "audio_init_checkpoint": str(audio_init),
        "audio_init_sha256": _sha256(audio_init),
        "source_audit": str(audit_path),
        "source_audit_sha256": _sha256(audit_path),
        "train_validation_test_overlap": {
            split: audit["split_audit"][split]["episode_key_overlap_count"]
            for split in ("val_multiple", "val_multiple_unheard",
                          "test_multiple", "test_multiple_unheard")
        },
        "validation_only_checkpoint_selection": True,
        "test_used": False,
    }
    (out / "training_config.json").write_text(json.dumps(cfg_record, indent=2) + "\n")
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=cfg_record["learning_rate"], weight_decay=cfg_record["weight_decay"],
    )
    history = []
    best_nll = float("inf")
    for epoch in range(1, cfg_record["epochs"] + 1):
        model.train()
        model.audio_encoder.eval()
        total_loss = correct = samples = 0
        for audio, rgb, actions in train_loader:
            audio = audio.to(device, non_blocking=True)
            rgb = rgb.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model.final.action_head(model.embedding_forward(audio, rgb))
                loss = F.cross_entropy(logits.float(), actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad], 10.0
            )
            optimizer.step()
            n = len(actions)
            total_loss += float(loss.detach()) * n
            correct += int((logits.argmax(-1) == actions).sum())
            samples += n
        row = {"epoch": epoch, "train_action_nll": total_loss / max(samples, 1),
               "train_action_accuracy": correct / max(samples, 1),
               "train_samples": samples}
        row["validation"] = {
            name: _evaluate(model, dataset_loader, device)
            for name, (_, dataset_loader) in validation.items()
        }
        row["selection_nll"] = max(
            metrics["action_nll"] for metrics in row["validation"].values()
        )
        history.append(row)
        (out / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
        checkpoint = {
            "model": model.state_dict(), "epoch": epoch,
            "training_config": cfg_record, "metrics": row,
            "architecture": NetworkSingleFrameNoDepth.architecture,
        }
        torch.save(checkpoint, out / f"epoch_{epoch:03d}.pt")
        if row["selection_nll"] < best_nll:
            best_nll = row["selection_nll"]
            torch.save(model.state_dict(), out / "foundation.pt")
            checkpoint["checkpoint_sha256"] = _sha256(out / "foundation.pt")
            torch.save(checkpoint, out / "best_action_validation.pt")
        print(json.dumps(row), flush=True)
    best_epoch = min(history, key=lambda item: item["selection_nll"])["epoch"]
    summary = {
        "status": "completed", **cfg_record, "best_epoch": best_epoch,
        "best_validation_action_nll": best_nll,
        "checkpoint": str((out / "foundation.pt").resolve()),
        "checkpoint_sha256": _sha256(out / "foundation.pt"),
        "per_epoch_metrics": history,
        "test_used_for_training_or_selection": False,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=(
        "experiment/avnav_real_single_frame/config/foundation_single_frame.yaml"
    ))
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2), flush=True)


if __name__ == "__main__":
    main()
