"""Run the refined heard/unheard SAC-CQL recipe from one YAML configuration."""
import argparse
from datetime import datetime
import json
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
    for key in ("base", "additional", "validation"):
        value = cfg["sources"][key]
        cfg["sources"][key] = ([resolve(path) for path in value]
                               if isinstance(value, list) else resolve(value))
    cfg["foundation"] = resolve(cfg["foundation"])
    cfg["parent"] = resolve(cfg["parent"])
    cfg["run"]["output"].mkdir(parents=True, exist_ok=True)
    if not isinstance(cfg["extraction"].get("rebuild_caches", False), bool):
        raise ValueError("extraction.rebuild_caches must be true or false")
    if len(cfg["training"]["sampling"]) != 3 or sum(cfg["training"]["sampling"]) != 32:
        raise ValueError("training.sampling must contain three counts totaling 32")
    for label, key in (("foundation", "foundation"), ("parent", "parent")):
        if not cfg[key].is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {cfg[key]}")
    for key in ("base", "additional", "validation"):
        sources = cfg["sources"][key]
        for source in (sources if isinstance(sources, list) else [sources]):
            if not source.is_dir():
                raise FileNotFoundError(f"{key} source not found: {source}")
    return cfg


def backup_name(path, label="previous"):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.name}.{label}-{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{label}-{stamp}-{suffix}")
        suffix += 1
    return candidate


def install_rebuilt_cache(staging, cache):
    backup = None
    if cache.exists():
        backup = backup_name(cache)
        cache.replace(backup)
    try:
        staging.replace(cache)
    except Exception:
        if backup is not None and not cache.exists():
            backup.replace(cache)
        raise
    if backup is not None:
        print(f"Previous cache preserved at {backup}", flush=True)


def extract_caches(cfg):
    settings = cfg["extraction"]
    if settings["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; set extraction.device to cpu in the YAML")
    device = torch.device(settings["device"])
    torch.set_num_threads(settings["cpu_threads"])
    rebuild = settings.get("rebuild_caches", False)
    staged_caches = []
    for key in ("base", "additional", "validation"):
        cache = cfg["caches"][key]
        target = cache.with_name(cache.name + ".rebuild") if rebuild else cache
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
        try:
            extract(feature_cfg, device)
        except ValueError as error:
            # A previous interrupted rebuild may have used an older source or
            # foundation. Preserve it and retry in a clean staging directory.
            if not (rebuild and "Feature cache checkpoint/source changed" in str(error)):
                raise
            stale = backup_name(target, "stale")
            target.replace(stale)
            print(f"Stale rebuild cache preserved at {stale}", flush=True)
            extract(feature_cfg, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if rebuild:
            staged_caches.append((target, cache))
    for staging, cache in staged_caches:
        install_rebuilt_cache(staging, cache)


def run_module(module, *args):
    command = [sys.executable, "-m", module, *(str(arg) for arg in args)]
    subprocess.run(command, cwd=ROOT, check=True)


def train(cfg):
    run, training = cfg["run"], cfg["training"]
    overrides = {key: value for key, value in training.items()
                 if key not in {"device", "cpu_threads", "epochs", "steps_per_epoch",
                                "sampling", "failed_stop_penalty"}}
    run_module(
        "experiment.avwan.reproduce_heard_unheard", "train",
        "--output", run["output"], "--name", run["name"], "--seed", run["seed"],
        "--device", training["device"], "--cpu-threads", training["cpu_threads"],
        "--foundation", cfg["foundation"], "--parent", cfg["parent"],
        "--base-cache", cfg["caches"]["base"],
        "--additional-cache", cfg["caches"]["additional"],
        "--validation-cache", cfg["caches"]["validation"],
        "--steps-per-epoch", training["steps_per_epoch"],
        "--epochs", training["epochs"], "--sampling", *training["sampling"],
        "--failed-stop-penalty", training["failed_stop_penalty"],
        "--training-overrides", json.dumps(overrides, separators=(",", ":")),
    )


def audit(cfg):
    run_module(
        "experiment.avwan.reproduce_heard_unheard", "audit",
        "--output", cfg["run"]["output"], "--name", cfg["run"]["name"],
        "--foundation", cfg["foundation"],
    )


def validate(cfg):
    for split in ("val_multiple", "val_multiple_unheard"):
        run_module(
            "experiment.avwan.reproduce_heard_unheard", "validate",
            "--output", cfg["run"]["output"], "--name", cfg["run"]["name"],
            "--foundation", cfg["foundation"], "--validation-split", split,
            "--validation-episodes", cfg["validation"]["episodes"],
            "--device", cfg["training"]["device"],
        )


def run_config(config, stage="all"):
    cfg = prepare_config(config)
    if stage not in ("all", "extract", "train", "audit", "validate"):
        raise ValueError("OFFLINE_RL.STAGE must be all, extract, train, audit, or validate")
    if stage in ("all", "extract"):
        extract_caches(cfg)
    if stage in ("all", "train"):
        train(cfg)
    if stage in ("all", "audit"):
        audit(cfg)
    if stage in ("all", "validate"):
        validate(cfg)


def run_from_node(node):
    stage = str(node.STAGE).lower()
    run_config(node, stage)


def load_config(config_path):
    value = yaml.safe_load(config_path.read_text())
    if "TASK_CONFIG" in value:
        value = value["TASK_CONFIG"]
    value = value.get("TRAIN", value)
    value = value["OFFLINE_RL"]
    return plain_config(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiment/avwan/config/audiogoal.yaml"))
    parser.add_argument("--stage", choices=("all", "extract", "train", "audit", "validate"), default="all")
    args = parser.parse_args(argv)
    run_config(load_config(Path(args.config).expanduser().resolve()), args.stage)


if __name__ == "__main__":
    main()
