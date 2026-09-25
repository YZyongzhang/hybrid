"""AVNav pickle conversion and Hybrid Network supervised training."""
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.default import get_config
from experiment.avnav.config import resolve_path
from Dataset.dataset_avnav import AVNavDataset, ShardBatchSampler, convert_pickle_to_pt, get_files


def run_conversion(config):
    cfg = config.TASK_CONFIG.PT
    outputs = convert_pickle_to_pt(resolve_path(cfg.FROM_PATH), resolve_path(cfg.TO_PATH),
                                   cfg.SHARD_SIZE, cfg.SHUFFLE, config.SEED)
    if cfg.VAL_FROM_PATH:
        convert_pickle_to_pt(resolve_path(cfg.VAL_FROM_PATH), resolve_path(cfg.VAL_TO_PATH),
                             cfg.SHARD_SIZE, cfg.SHUFFLE, config.SEED)
    return outputs


def build_datasets(cfg, seed):
    files = get_files(resolve_path(cfg.DATA_PATH), (".pt",))
    if cfg.VAL_DATA_PATH:
        val_files = get_files(resolve_path(cfg.VAL_DATA_PATH), (".pt",))
        if set(map(lambda p: p.resolve(), files)) & set(map(lambda p: p.resolve(), val_files)):
            raise ValueError("Training and validation PT files must be disjoint")
    else:
        if not 0 < cfg.VAL_RATIO < 1:
            raise ValueError("VAL_RATIO must be between 0 and 1")
        if len(files) < 2:
            raise ValueError("Automatic validation split needs at least two PT shards; "
                             "provide VAL_DATA_PATH or convert multiple episodes with a smaller SHARD_SIZE")
        random.Random(seed).shuffle(files)
        count = max(1, min(len(files) - 1, round(len(files) * cfg.VAL_RATIO)))
        val_files, files = files[:count], files[count:]
    return AVNavDataset(files, cfg.CACHE_SIZE), AVNavDataset(val_files, cfg.CACHE_SIZE)


def run_training(config):
    cfg = config.TASK_CONFIG.TRAIN
    if cfg.EPOCHS < 1 or cfg.BATCH_SIZE < 1 or cfg.NUM_WORKERS < 0:
        raise ValueError("EPOCHS/BATCH_SIZE must be positive and NUM_WORKERS nonnegative")
    if cfg.LEARNING_RATE <= 0 or cfg.WEIGHT_DECAY < 0:
        raise ValueError("LEARNING_RATE must be positive and WEIGHT_DECAY nonnegative")
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if cfg.DEVICE == "auto" else cfg.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; set TRAIN.DEVICE to cpu or auto")
    train_data, val_data = build_datasets(cfg, config.SEED)
    loader_options = dict(num_workers=cfg.NUM_WORKERS, pin_memory=device.type == "cuda",
                          persistent_workers=cfg.NUM_WORKERS > 0)
    train_loader = DataLoader(train_data,
                              batch_sampler=ShardBatchSampler(train_data, cfg.BATCH_SIZE, config.SEED),
                              **loader_options)
    val_loader = DataLoader(val_data, batch_size=cfg.BATCH_SIZE, shuffle=False, **loader_options)

    from experiment.avwan.model.hybrid import Network
    from experiment.avnav.train_class.hybrid_train import Train

    model = Network().to(device)
    if cfg.MODEL_PATH:
        model.load_state_dict(torch.load(resolve_path(cfg.MODEL_PATH), map_location="gpu", weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    save_dir = resolve_path(cfg.SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if cfg.TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(resolve_path(cfg.LOG_DIR)))
    print(f"Training Network on {device}: train={len(train_data)}, val={len(val_data)}")
    try:
        trainer = Train(model=model, Adam=optimizer, train_loader=train_loader, val_loader=val_loader,
                        epoch=cfg.EPOCHS, writer=writer, save_dir=str(save_dir),
                        train_size=len(train_data), val_size=len(val_data), device=device)
        trainer.train()
    finally:
        if writer is not None:
            writer.close()
    return model


def run(config):
    pt, train = config.TASK_CONFIG.PT, config.TASK_CONFIG.TRAIN
    if not pt.OPEN and not train.OPEN:
        raise ValueError("Enable TASK_CONFIG.PT.OPEN or TASK_CONFIG.TRAIN.OPEN")
    if pt.OPEN:
        run_conversion(config)
    if train.OPEN:
        return run_training(config)


def main(argv=None):
    # import pdb;pdb.set_trace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiment/avnav/config/av_nav_multiple.yaml" , help="YAML overrides for experiment/avnav/config/default.yaml")
    parser.add_argument("--stage", default="train" , choices=("convert", "train", "all"), help="Override workflow OPEN flags")
    args = parser.parse_args(argv)
    config = get_config(args.config)
    if args.stage:
        config.TASK_CONFIG.PT.OPEN = args.stage in ("convert", "all")
        config.TASK_CONFIG.TRAIN.OPEN = args.stage in ("train", "all")
    run(config)


if __name__ == "__main__":
    main()
