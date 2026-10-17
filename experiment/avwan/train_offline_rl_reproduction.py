"""Run the refined heard/unheard SAC-CQL recipe from one YAML configuration."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from .offline.features import extract


ROOT = Path(__file__).resolve().parents[2]


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def plain_config(value):
    if hasattr(value, "items"):
        return {str(key).lower(): plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_config(item) for item in value]
    return value


def prepare_config(cfg):
    cfg = plain_config(cfg)
    cfg.pop("stage", None)
    for section, key in (("run", "output"), ("caches", "base"), ("caches", "additional"),
                         ("caches", "validation")):
        cfg[section][key] = resolve(cfg[section][key])
    if cfg["caches"].get("hard_correction"):
        cfg["caches"]["hard_correction"] = resolve(cfg["caches"]["hard_correction"])
    augment = cfg["sources"].get("base_augment", [])
    augment = augment if isinstance(augment, list) else ([augment] if augment else [])
    if augment:
        cfg["sources"]["base"] = [cfg["sources"]["base"], *augment]
    longrange = augment if cfg.get("longrange_pool", False) else []
    cfg["sources"]["longrange"] = [resolve(path) for path in longrange]
    for key in ("base", "additional", "validation"):
        value = cfg["sources"][key]
        cfg["sources"][key] = ([resolve(path) for path in value]
                               if isinstance(value, list) else resolve(value))
    if cfg["sources"].get("hard_correction"):
        cfg["sources"]["hard_correction"] = resolve(cfg["sources"]["hard_correction"])
    cfg["foundation"] = resolve(cfg["foundation"])
    cfg["parent"] = resolve(cfg["parent"]) if cfg.get("parent") else None
    cfg["run"]["output"].mkdir(parents=True, exist_ok=True)
    schedule = cfg["training"].get("curriculum_schedule")
    separate_corrections = bool(cfg.get("separate_hard_correction_pool",
                                       cfg["training"].get("separate_hard_correction_pool", False)))
    expected_pools = 6 if schedule else None
    if schedule:
        if len(cfg["training"]["sampling"]) != expected_pools:
            raise ValueError("Curriculum training.sampling must contain six pool counts")
        if sum(int(stage["epochs"]) for stage in schedule) != int(cfg["training"]["epochs"]):
            raise ValueError("Curriculum stage epochs must sum to training.epochs")
        for stage in schedule:
            counts = stage["sampling"]
            if len(counts) != expected_pools or sum(counts) != 32 or any(int(n) < 0 for n in counts):
                raise ValueError(f"Invalid curriculum sampling in stage {stage.get('name')}")
    else:
        allowed = ((5,) if cfg.get("longrange_pool", False) else (4,)) if separate_corrections else (3, 4)
        if len(cfg["training"]["sampling"]) not in allowed or sum(cfg["training"]["sampling"]) != 32:
            raise ValueError(f"training.sampling must contain {allowed} counts totaling 32")
    if separate_corrections and not cfg["sources"].get("hard_correction"):
        raise ValueError("A separate hard correction pool requires sources.hard_correction")
    for label, key in (("foundation", "foundation"), ("parent", "parent")):
        if cfg[key] is None and key == "parent":
            continue
        if not cfg[key].is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {cfg[key]}")
    for key in ("base", "additional", "validation"):
        sources = cfg["sources"][key]
        for source in (sources if isinstance(sources, list) else [sources]):
            if not source.is_dir():
                raise FileNotFoundError(f"{key} source not found: {source}")
    if cfg["sources"].get("hard_correction") and not cfg["sources"]["hard_correction"].is_dir():
        raise FileNotFoundError(f"hard_correction source not found: {cfg['sources']['hard_correction']}")
    return cfg


def extract_caches(cfg, fresh=False):
    settings = cfg["extraction"]
    if settings["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; set extraction.device to cpu in the YAML")
    device = torch.device(settings["device"])
    torch.set_num_threads(settings["cpu_threads"])
    if device.type == "cuda":
        fraction = settings.get("gpu_memory_fraction", 0.15)
        if not 0 < fraction <= 1:
            raise ValueError("extraction.gpu_memory_fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    available_gib = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))) / (1024 * 1024)
    if available_gib < settings.get("min_available_gib", 12):
        raise MemoryError(f"Stopping feature extraction with only {available_gib:.1f} GiB RAM available")
    keys = ["base", "additional", "validation"]
    if cfg["sources"].get("hard_correction"):
        keys.append("hard_correction")
    for key in keys:
        cache = cfg["caches"][key]
        if fresh:
            try:
                cache.relative_to(cfg["run"]["output"])
            except ValueError as error:
                raise ValueError(
                    f"Fresh feature output must be inside this run directory: {cache}") from error
            if cache.exists():
                shutil.rmtree(cache)
        target = cache
        source = cfg["sources"][key]
        feature_cfg = {
            "features": {
                "source": ([str(value) for value in source]
                           if isinstance(source, list) else str(source)),
                "checkpoint": str(cfg["foundation"]),
                "cache": str(target),
                "batch_size": settings["batch_size"],
                "episode_batch_size": settings["episode_batch_size"],
                "workers": settings["workers"],
                "max_episodes": settings["max_episodes"],
            }
        }
        print(f"Extracting {key} features -> {target}", flush=True)
        extract(feature_cfg, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_module(module, *args):
    command = [sys.executable, "-m", module, *(str(arg) for arg in args)]
    subprocess.run(command, cwd=ROOT, check=True)


def train(cfg):
    run, training = cfg["run"], cfg["training"]
    overrides = {key: value for key, value in training.items()
                 if key not in {"device", "cpu_threads", "epochs", "steps_per_epoch",
                                "sampling", "failed_stop_penalty", "gpu_memory_fraction",
                                "max_rss_mib", "min_available_gib", "curriculum_schedule",
                                "method", "separate_hard_correction_pool", "reward_shaping"}}
    command = [
        "experiment.avwan.reproduce_heard_unheard", "train",
        "--output", run["output"], "--name", run["name"], "--seed", run["seed"],
        "--device", training["device"], "--cpu-threads", training["cpu_threads"],
        "--foundation", cfg["foundation"],
        "--base-cache", cfg["caches"]["base"],
        "--additional-cache", cfg["caches"]["additional"],
        "--validation-cache", cfg["caches"]["validation"],
        "--steps-per-epoch", training["steps_per_epoch"],
        "--epochs", training["epochs"], "--sampling", *training["sampling"],
        "--failed-stop-penalty", training["failed_stop_penalty"],
        "--gpu-memory-fraction", training.get("gpu_memory_fraction", 0.12),
        "--max-rss-mb", training.get("max_rss_mib", 12000),
        "--min-available-gib", training.get("min_available_gib", 12),
        "--training-overrides", json.dumps(overrides, separators=(",", ":")),
        "--training-method", training.get("method", "sac_cql"),
    ]
    if training.get("reward_shaping"):
        command.extend(["--reward-shaping", json.dumps(training["reward_shaping"], separators=(",", ":"))])
    if cfg.get("separate_hard_correction_pool", training.get("separate_hard_correction_pool", False)):
        command.append("--separate-hard-correction-pool")
    if cfg["sources"].get("longrange"):
        command.extend(["--longrange-source", *cfg["sources"]["longrange"]])
    if cfg["sources"].get("hard_correction"):
        command.extend(["--hard-correction-source", cfg["sources"]["hard_correction"]])
        command.extend(["--hard-correction-cache", cfg["caches"]["hard_correction"]])
    if training.get("curriculum_schedule"):
        command.extend(["--curriculum-schedule",
                        json.dumps(training["curriculum_schedule"], separators=(",", ":"))])
    if cfg["parent"] is None:
        command.append("--random-initialization")
    else:
        command.extend(["--parent", cfg["parent"]])
    run_module(*command)


def audit(cfg):
    command = [
        "experiment.avwan.reproduce_heard_unheard", "audit",
        "--output", cfg["run"]["output"], "--name", cfg["run"]["name"],
        "--checkpoint", cfg["run"]["output"] / cfg["run"]["name"] / "training" / "best_policy.pt",
        "--foundation", cfg["foundation"],
    ]
    if cfg["sources"].get("hard_correction"):
        command.extend(["--hard-correction-source", cfg["sources"]["hard_correction"]])
        command.extend(["--hard-correction-cache", cfg["caches"]["hard_correction"]])
    run_module(*command)


def validate(cfg):
    for split in ("val_multiple", "val_multiple_unheard"):
        run_module(
            "experiment.avwan.reproduce_heard_unheard", "validate",
            "--output", cfg["run"]["output"], "--name", cfg["run"]["name"],
            "--checkpoint", cfg["run"]["output"] / cfg["run"]["name"] / "training" / "best_policy.pt",
            "--foundation", cfg["foundation"], "--validation-split", split,
            "--validation-episodes", cfg["validation"]["episodes"],
            "--device", cfg["training"]["device"],
        )


def run_config(config, stage="all", fresh_features=False):
    cfg = prepare_config(config)
    if stage not in ("all", "extract", "train", "audit", "validate"):
        raise ValueError("OFFLINE_RL.STAGE must be all, extract, train, audit, or validate")
    if stage in ("all", "extract"):
        extract_caches(cfg, fresh=fresh_features)
    if stage in ("all", "train"):
        train(cfg)
    if stage in ("all", "audit"):
        audit(cfg)
    if stage in ("all", "validate"):
        validate(cfg)


def run_from_node(node):
    stage = str(node.STAGE).lower()
    run_config(node, stage, fresh_features=bool(node.get("FRESH_FEATURES", False)))


def load_config(config_path):
    value = yaml.safe_load(config_path.read_text())
    if "TASK_CONFIG" in value:
        value = value["TASK_CONFIG"]
    value = value.get("TRAIN", value)
    value = value.get("OFFLINE_RL", value)
    return plain_config(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiment/avwan/config/audiogoal.yaml"))
    parser.add_argument("--stage", choices=("all", "extract", "train", "audit", "validate"), default="all")
    parser.add_argument("--fresh-features", action="store_true",
                        help="Delete this run's derived features and extract them again from raw episodes")
    args = parser.parse_args(argv)
    run_config(load_config(Path(args.config).expanduser().resolve()), args.stage,
               args.fresh_features)


if __name__ == "__main__":
    main()
