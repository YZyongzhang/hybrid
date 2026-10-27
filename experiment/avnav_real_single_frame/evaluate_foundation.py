"""Validation-only closed-loop evaluation of the single-frame foundation head."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from experiment.avnav.evaluate_online import evaluate_split
from experiment.avnav_real_single_frame.model import NetworkSingleFrameNoDepth


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio(observation):
    value = observation["spectrogram"]
    if isinstance(value, (tuple, list)):
        value = value[0]
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.shape != (65, 26, 2):
        raise ValueError(f"Expected spectrogram [65,26,2], got {value.shape}")
    return torch.from_numpy(np.ascontiguousarray(value))


class SingleFrameFoundationPolicy:
    def __init__(self, model, device):
        self.model = model.eval()
        self.device = device
        self.last_collision_probability = None
        self.last_future_collision_probability = None

    def reset(self):
        pass

    def record_collision(self, collided):
        pass

    @torch.inference_mode()
    def act(self, observation, sample=False, temperature=1.0,
            current_distance=None):
        rgb = torch.as_tensor(np.asarray(observation["rgb"])).float().div_(255.0)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected single HWC RGB frame, got {tuple(rgb.shape)}")
        audio = _audio(observation).unsqueeze(0).to(self.device)
        rgb = rgb.unsqueeze(0).to(self.device)
        embedding = self.model.embedding_forward(audio, rgb)
        logits = self.model.final.action_head(embedding)
        return int(logits.argmax(-1).item())


def run(config_path, episodes_per_split=None, diagnostic_episodes=None,
        splits=None, output_dir=None):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    ev = cfg["EVALUATION"]
    device = torch.device(ev["DEVICE"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Configured device {device} is unavailable")
    torch.set_num_threads(2)
    model_path = Path(cfg["FOUNDATION"]["OUTPUT"]) / "foundation.pt"
    model = NetworkSingleFrameNoDepth().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True), strict=True)
    model.eval().requires_grad_(False)

    selected_pool = int(episodes_per_split or ev["EPISODES_PER_SPLIT"])
    count = int(diagnostic_episodes or ev["DIAGNOSTIC_EPISODES"])
    if not 1 <= count <= selected_pool:
        raise ValueError("Diagnostic count must be between 1 and the validation pool size")
    selected_splits = list(splits or ev["SPLITS"])
    root = Path(output_dir or ev["OUTPUT"])
    root.mkdir(parents=True, exist_ok=True)
    settings = {
        "env_config": str(Path(ev["CONFIG"]).resolve()),
        "episodes_per_split": selected_pool,
        "max_steps": int(ev["MAX_STEPS"]),
    }
    policy = SingleFrameFoundationPolicy(model, device)
    reports = {}
    for split in selected_splits:
        if not split.startswith("val_"):
            raise ValueError("This ablation runner evaluates validation splits only")
        split_root = root / split
        split_root.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_split(
            split, policy, settings, int(ev["SEED"]), split_root,
            index_range=(0, count),
        )
        reports[split] = metrics
        (split_root / "report.json").write_text(json.dumps({
            "status": "completed", "split": split,
            "validation_only": True, "episodes_evaluated": count,
            "validation_pool_episodes": selected_pool,
            "checkpoint": str(model_path.resolve()),
            "checkpoint_sha256": _sha256(model_path),
            "architecture": NetworkSingleFrameNoDepth.architecture,
            "visual_input": "single current RGB frame; no depth or previous RGB",
            "action_handling": ev["ACTION_HANDLING"],
            "test_used_for_training_or_selection": False,
            "metrics": metrics,
        }, indent=2) + "\n")
    summary = {
        "status": "completed", "checkpoint": str(model_path.resolve()),
        "checkpoint_sha256": _sha256(model_path),
        "evaluation": "validation-only diagnostic; no test episodes loaded",
        "episodes_per_split": count,
        "validation_pool_episodes": selected_pool,
        "max_steps": int(ev["MAX_STEPS"]),
        "action_handling": ev["ACTION_HANDLING"],
        "reports": reports,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=(
        "experiment/avnav_real_single_frame/config/foundation_single_frame.yaml"
    ))
    parser.add_argument("--episodes-per-split", type=int, default=None)
    parser.add_argument("--diagnostic-episodes", type=int, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.episodes_per_split,
                         args.diagnostic_episodes, args.splits, args.output),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
