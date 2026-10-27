"""Export the best staged action checkpoint as an AVNav foundation model."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from experiment.avnav.config import resolve_path
from experiment.avnav.model.hybrid import Network
from experiment.avnav.train_staged_foundation import sha256, write_json


def export(config_path, early_stop_reason=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = resolve_path(config["FOUNDATION"]["OUTPUT"])
    audio_dir = output / "audio_stage"
    action_dir = output / "action_stage"
    audio_path = audio_dir / "best.pt"
    action_path = action_dir / "best.pt"
    audio_summary = json.loads((audio_dir / "best.json").read_text(encoding="utf-8"))
    action_summary = json.loads((action_dir / "best.json").read_text(encoding="utf-8"))
    if sha256(audio_path) != audio_summary.get("checkpoint_sha256"):
        raise ValueError("Audio checkpoint SHA256 does not match audio_stage/best.json")
    if sha256(action_path) != action_summary.get("checkpoint_sha256"):
        raise ValueError("Action checkpoint SHA256 does not match action_stage/best.json")

    audio_state = torch.load(audio_path, map_location="cpu", weights_only=False)
    action_state = torch.load(action_path, map_location="cpu", weights_only=False)
    if action_state.get("audio_checkpoint_sha256") != sha256(audio_path):
        raise ValueError("Action checkpoint was trained from a different audio checkpoint")
    model = Network()
    model.load_state_dict(action_state["model"], strict=True)
    frozen_audio_matches = all(
        torch.equal(model.state_dict()[name], audio_state["model"][name])
        for name in audio_state["model"] if name.startswith("audio_encoder.")
    )
    if not frozen_audio_matches:
        raise AssertionError("Action checkpoint changed the frozen audio encoder")

    write_json(action_dir / "frozen_audio_audit.json", {
        "passed": True,
        "audio_checkpoint_sha256": sha256(audio_path),
        "checked_parameters": sum(name.startswith("audio_encoder.")
                                   for name in model.state_dict()),
        "action_checkpoint_sha256": sha256(action_path),
    })
    checkpoint = output / "best.pth"
    torch.save(model.state_dict(), checkpoint)
    completed_rows = [json.loads(line) for line in
                      (action_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                      if line.strip()]
    status = {
        "status": "foundation_trained",
        "best_audio_validation_angle_loss": float(audio_summary["worst_validation_angle_loss"]),
        "best_action_validation_action_loss": float(action_summary["worst_validation_action_loss"]),
        "action_best_epoch": int(action_summary["epoch"]),
        "action_epochs_completed": len(completed_rows),
        "action_epoch_limit": int(config["FOUNDATION"]["ACTION_EPOCHS"]),
        "action_stopped_early": early_stop_reason is not None,
        "action_stop_reason": early_stop_reason,
        "foundation_checkpoint": str(checkpoint.resolve()),
        "foundation_checkpoint_sha256": sha256(checkpoint),
        "audio_checkpoint_sha256": sha256(audio_path),
        "action_checkpoint_sha256": sha256(action_path),
    }
    write_json(output / "status.json", status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--early-stop-reason", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(export(args.config, args.early_stop_reason), indent=2), flush=True)


if __name__ == "__main__":
    main()
