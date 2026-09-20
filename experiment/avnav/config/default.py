"""AVNav workflow configuration; relative paths are rooted at the repository."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("default.yaml")


def _merge(defaults, updates, prefix=""):
    for key, value in updates.items():
        if key not in defaults:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} must be a mapping")
            _merge(defaults[key], value, f"{prefix}{key}.")
        else:
            defaults[key] = value


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    return value


def get_config(config_path=None):
    with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
        values = deepcopy(yaml.safe_load(handle))
    if config_path is not None:
        with Path(config_path).expanduser().open(encoding="utf-8") as handle:
            updates = yaml.safe_load(handle) or {}
        if not isinstance(updates, dict):
            raise ValueError("Configuration must be a YAML mapping")
        _merge(values, updates)
    return _namespace(values)


def resolve_path(path):
    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value
