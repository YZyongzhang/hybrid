"""Resolve and hash canonical AVNav episode manifests for a frozen run."""

import hashlib
from pathlib import Path

import yaml

from experiment.avnav.config import resolve_path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_manifest_hashes(task_config_path, splits):
    task_config_path = resolve_path(task_config_path)
    config = yaml.safe_load(task_config_path.read_text(encoding="utf-8"))
    dataset = config["DATASET"]
    result = {}
    for split in splits:
        path = resolve_path(dataset["DATA_PATH"].format(
            version=dataset["VERSION"], split=split
        ))
        if not path.exists() and path.suffix == ".gz":
            uncompressed = path.with_suffix("")
            if uncompressed.exists():
                path = uncompressed
        if not path.is_file():
            raise FileNotFoundError(f"Canonical {split} episode manifest is missing: {path}")
        result[split] = {"path": str(path.resolve()), "sha256": sha256(path)}
    return result
