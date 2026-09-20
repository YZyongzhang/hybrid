import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

NUM_SECTORS = 8


def angle_to_sector_label(angle_deg: torch.Tensor, num_sectors: int = NUM_SECTORS):
    angle_deg = angle_deg.float()
    sector_size = 360.0 / float(num_sectors)
    normalized = torch.remainder(angle_deg + 180.0, 360.0)
    labels = torch.floor(normalized / sector_size).long()
    return torch.clamp(labels, min=0, max=num_sectors - 1)


class Train:
    def __init__(
        self,
        model,
        Adam,
        train_loader,
        val_loader,
        epoch,
        writer,
        save_dir,
        train_size,
        val_size,
        device="cuda",
    ):
        self.train_model = model.to(device)
        self.epoch = int(epoch)
        self.writer = writer
        self.device = device
        self.save_dir = save_dir
        self.train_size = int(train_size)
        self.val_size = int(val_size)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = Adam

        self.action_loss_weight = 1.0

    @staticmethod
    def semantic_audio(angle_logits, angle_deg):
        labels = angle_to_sector_label(angle_deg)
        loss = F.cross_entropy(angle_logits, labels)
        return {"labels": labels, "preds": angle_logits.argmax(dim=1),
                "classifier_loss": loss, "total_loss": loss}

    def _forward(self, batch_audio, batch_rgb, batch_depth):
        return self.train_model.forward_joint(batch_audio, batch_rgb, batch_depth)

    def _maybe_reload_dataset(self):
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "on_epoch_end"):
            dataset.on_epoch_end()

    def train(self):
        global_step = 0
        best_val_loss = float("inf")
        history = []

        for ep in range(self.epoch):
            started = time.monotonic()
            self.train_model.train()
            local_step = 0
            epoch_total_loss = 0.0
            epoch_action_loss = 0.0
            epoch_angle_loss = 0.0
            correct_action = 0
            correct_angle = 0
            seen = 0

            for batch in tqdm(self.train_loader, desc=f"Train {ep + 1}/{self.epoch}", mininterval=30):
                batch_audio, batch_rgb, batch_depth, batch_angle, batch_action = batch
                batch_audio = batch_audio.to(self.device)
                batch_rgb = batch_rgb.to(self.device)
                batch_depth = batch_depth.to(self.device)
                batch_action = batch_action.long().to(self.device)
                batch_angle = batch_angle.float().to(self.device)

                action_logits, angle_logits = self._forward(batch_audio, batch_rgb, batch_depth)
                loss_action = F.cross_entropy(action_logits, batch_action)
                semantic_out = self.semantic_audio(angle_logits, batch_angle)
                angle_labels = semantic_out["labels"]
                loss_angle = semantic_out["classifier_loss"]
                loss = self.action_loss_weight * loss_action + semantic_out["total_loss"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                action_pred = action_logits.argmax(dim=1)
                angle_pred = semantic_out["preds"]
                batch_size = int(batch_action.shape[0])
                seen += batch_size
                correct_action += (action_pred == batch_action).sum().item()
                correct_angle += (angle_pred == angle_labels).sum().item()

                epoch_total_loss += loss.item() * batch_size
                epoch_action_loss += loss_action.item() * batch_size
                epoch_angle_loss += loss_angle.item() * batch_size
                local_step += 1

                if self.writer:
                    self.writer.add_scalar("Loss/step_total", loss.item(), global_step)
                    self.writer.add_scalar("Loss/step_action", loss_action.item(), global_step)
                    self.writer.add_scalar("Loss/step_angle", loss_angle.item(), global_step)
                global_step += 1

            validation = self.validate(ep)
            self._maybe_reload_dataset()

            avg_total = epoch_total_loss / max(1, seen)
            avg_action = epoch_action_loss / max(1, seen)
            avg_angle = epoch_angle_loss / max(1, seen)
            acc_action = correct_action / max(1, seen)
            acc_angle = correct_angle / max(1, seen)
            if self.writer:
                self.writer.add_scalar("Loss/epoch_total", avg_total, ep)
                self.writer.add_scalar("Loss/epoch_action", avg_action, ep)
                self.writer.add_scalar("Loss/epoch_angle", avg_angle, ep)
                self.writer.add_scalar("Acc/epoch_action", acc_action, ep)
                self.writer.add_scalar("Acc/epoch_angle", acc_angle, ep)

            print(
                f"Epoch {ep + 1} finished, total={avg_total:.4f}, "
                f"action={avg_action:.4f}, angle={avg_angle:.4f}, "
                f"action_acc={acc_action:.4f}, angle_acc={acc_angle:.4f}"
            )

            save_path = f"{self.save_dir}/model_epoch_{ep + 1}.pth"
            torch.save(self.train_model.state_dict(), save_path)
            tqdm.write(f"Saved model checkpoint to {save_path}")
            if validation["total_loss"] < best_val_loss:
                best_val_loss = validation["total_loss"]
                torch.save(self.train_model.state_dict(), Path(self.save_dir) / "best.pth")
            history.append(dict(epoch=ep + 1, seconds=time.monotonic() - started,
                                train=dict(total_loss=avg_total, action_loss=avg_action,
                                           angle_loss=avg_angle, action_acc=acc_action,
                                           angle_acc=acc_angle, samples=seen), val=validation))
            metrics_path = Path(self.save_dir) / "metrics.json"
            temporary = metrics_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
            temporary.replace(metrics_path)

    def validate(self, step):
        self.train_model.eval()

        val_total = 0.0
        val_action = 0.0
        val_angle = 0.0
        correct_action = 0
        correct_angle = 0
        seen = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validate {step + 1}", mininterval=30):
                batch_audio, batch_rgb, batch_depth, batch_angle, batch_action = batch
                batch_audio = batch_audio.to(self.device)
                batch_rgb = batch_rgb.to(self.device)
                batch_depth = batch_depth.to(self.device)
                batch_action = batch_action.long().to(self.device)
                batch_angle = batch_angle.float().to(self.device)

                action_logits, angle_logits = self._forward(batch_audio, batch_rgb, batch_depth)
                loss_action = F.cross_entropy(action_logits, batch_action)
                semantic_out = self.semantic_audio(angle_logits, batch_angle)
                angle_labels = semantic_out["labels"]
                loss_angle = semantic_out["classifier_loss"]
                loss = self.action_loss_weight * loss_action + semantic_out["total_loss"]

                batch_size = int(batch_action.shape[0])
                val_total += loss.item() * batch_size
                val_action += loss_action.item() * batch_size
                val_angle += loss_angle.item() * batch_size

                action_pred = action_logits.argmax(dim=1)
                angle_pred = semantic_out["preds"]
                batch_size = int(batch_action.shape[0])
                seen += batch_size
                correct_action += (action_pred == batch_action).sum().item()
                correct_angle += (angle_pred == angle_labels).sum().item()

        avg_total = val_total / max(1, seen)
        avg_action = val_action / max(1, seen)
        avg_angle = val_angle / max(1, seen)
        action_acc = correct_action / max(1, seen)
        angle_acc = correct_angle / max(1, seen)

        if self.writer:
            self.writer.add_scalar("Val/total_loss", avg_total, step)
            self.writer.add_scalar("Val/action_loss", avg_action, step)
            self.writer.add_scalar("Val/angle_loss", avg_angle, step)
            self.writer.add_scalar("Val/action_acc", action_acc, step)
            self.writer.add_scalar("Val/angle_acc", angle_acc, step)

        self.train_model.train()
        metrics = dict(total_loss=avg_total, action_loss=avg_action, angle_loss=avg_angle,
                       action_acc=action_acc, angle_acc=angle_acc, samples=seen)
        print(f"Validation {step + 1}: {json.dumps(metrics)}", flush=True)
        return metrics
