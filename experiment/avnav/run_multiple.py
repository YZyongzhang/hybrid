"""Convert all multiple-source splits, train, and evaluate the best model on unheard audio."""
import argparse
import json
from pathlib import Path
import platform
import shutil

import torch
from torch.utils.data import DataLoader

from configs.default import get_config
from Dataset.dataset_avnav import AVNavDataset, convert_pickle_to_pt
from experiment.avnav.config import resolve_path
from experiment.avnav.run import run_conversion, run_training
from experiment.avnav.model.hybrid import Network
from experiment.avnav.train_class.hybrid_train import Train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config/experiment.yaml"))
    parser.add_argument("--stage", choices=("convert", "train", "evaluate", "all"), default="train")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = get_config(str(config_path))
    result = resolve_path(config.TASK_CONFIG.TRAIN.SAVE_DIR)
    result.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, result / "config.yaml")
    unheard_raw = Path(config.TASK_CONFIG.PT.FROM_PATH).parent / "hybird(val_multiple_unheard)"
    unheard_pt = Path(config.TASK_CONFIG.PT.TO_PATH).parent / "val_unheard"
    if args.stage in ("all", "convert"):
        for source, destination in (
            (config.TASK_CONFIG.PT.FROM_PATH, config.TASK_CONFIG.PT.TO_PATH),
            (config.TASK_CONFIG.PT.VAL_FROM_PATH, config.TASK_CONFIG.PT.VAL_TO_PATH),
            (unheard_raw, unheard_pt),
        ):
            manifest = Path(destination) / "manifest.json"
            if manifest.exists():
                print(f"Reusing existing converted split: {destination}", flush=True)
                continue
            convert_pickle_to_pt(resolve_path(source), resolve_path(destination),
                                 config.TASK_CONFIG.PT.SHARD_SIZE,
                                 config.TASK_CONFIG.PT.SHUFFLE, config.SEED)
        print("All three splits converted successfully.", flush=True)
    if args.stage in ("all", "train"):
        metadata = dict(python=platform.python_version(), pytorch=str(torch.__version__),
                        cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        (result / "environment.json").write_text(json.dumps(metadata, indent=2))
        run_training(config)
    if args.stage in ("all", "train", "evaluate"):
        model = Network()
        model.load_state_dict(torch.load(result / "best.pth", map_location="cpu", weights_only=True))
        dataset = AVNavDataset(unheard_pt, cache_size=2)
        loader = DataLoader(dataset, batch_size=config.TASK_CONFIG.TRAIN.BATCH_SIZE,
                            num_workers=2, pin_memory=True)
        trainer = Train(model, None, None, loader, 0, None, str(result), 0, len(dataset), "cuda")
        metrics = trainer.validate(0)
        metrics["checkpoint"] = "best.pth"
        metrics["split"] = "val_unheard"
        (result / "unheard_metrics.json").write_text(json.dumps(metrics, indent=2))
        print("Training and unheard validation completed successfully.", flush=True)


if __name__ == "__main__":
    main()
