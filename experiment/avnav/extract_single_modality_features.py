"""Extract train-only frozen features from one AVNav encoder branch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiment.avnav.extract_codex_features import _atomic_json, _extract_category, _save_split
from experiment.avnav.hybrid.foundation_model import Network


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    run_dir = Path(args.run_dir).resolve()
    audit_path = run_dir / "audit/data_source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Train-only feature extraction requires a passing provenance audit")
    for category, declared in audit["categories"].items():
        manifest = Path(declared["manifest"])
        if _sha256(manifest) != declared["manifest_sha256"]:
            raise RuntimeError(f"Training manifest changed since audit: {category}")
    for split in ("val_multiple", "val_multiple_unheard"):
        overlap = audit["selected_parent_episode_overlap"][split]["count"]
        if overlap:
            raise RuntimeError(f"Training source overlaps {split}: {overlap} episodes")

    foundation_path = run_dir / "foundation/action_stage/best.pth"
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        torch.cuda.manual_seed_all(args.seed)
    model = Network().to(device)
    model.load_state_dict(torch.load(foundation_path, map_location=device, weights_only=True))
    model.eval().requires_grad_(False)

    output_dir = run_dir / "offline/single_modality" / args.modality / "features"
    train_episodes = []
    sources = {}
    for category in ("greedy", "deverse", "collision"):
        declared = audit["categories"][category]
        episodes = _extract_category(
            model, category, device, args.batch_size, modality=args.modality
        )
        sources[category] = {
            "manifest": declared["manifest"],
            "manifest_sha256": declared["manifest_sha256"],
            "episodes": len(episodes),
            "transitions": sum(len(row["actions"]) for row in episodes),
            "recorded_collisions": sum(int(row["collisions"].sum()) for row in episodes),
        }
        train_episodes.extend(episodes)
        print(json.dumps({
            "stage": "single_modality_features", "modality": args.modality,
            "source": category, "episodes": len(episodes),
            "transitions": sources[category]["transitions"],
        }), flush=True)

    train_path = output_dir / "train.pt"
    _save_split(train_path, train_episodes)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "status": "completed",
        "modality": args.modality,
        "feature_extraction": (
            "frozen audio encoder token only" if args.modality == "audio_only"
            else "frozen RGB-D visual encoder and visual transformer only; spatial tokens mean-pooled"
        ),
        "feature_dim": 128,
        "foundation_checkpoint": str(foundation_path),
        "foundation_sha256": _sha256(foundation_path),
        "foundation_frozen": True,
        "data_source_audit": str(audit_path),
        "data_source_audit_sha256": _sha256(audit_path),
        "train_sources": sources,
        "output": {
            "path": str(train_path), "sha256": _sha256(train_path),
            "episodes": len(train_episodes),
            "transitions": sum(len(row["actions"]) for row in train_episodes),
            "recorded_collisions": sum(int(row["collisions"].sum()) for row in train_episodes),
        },
        "validation_data_loaded": False,
        "test_data_loaded": False,
        "other_modality_encoder_called": False,
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--modality", choices=["vision_only", "audio_only"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261027)
    run(parser.parse_args())
