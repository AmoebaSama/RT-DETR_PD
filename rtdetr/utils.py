from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import yaml


def _resource_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def _app_root() -> Path:
    configured_root = os.environ.get("RTDETR_ROOT")
    if configured_root:
        return Path(configured_root)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return _resource_root()


RESOURCE_ROOT = _resource_root()
ROOT = _app_root()
CONFIG_DIR = RESOURCE_ROOT / "rtdetr" / "configs"
RUNS_DIR = ROOT / "rtdetr" / "runs"
GENERATED_DIR = ROOT / "rtdetr" / ".generated"
DEFAULT_DATASET = CONFIG_DIR / "solder_defects_dataset2.yaml"
DEFAULT_WEIGHTS = RUNS_DIR / "solder_defects_rtdetr" / "weights" / "best.pt"


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    app_path = ROOT / path
    if app_path.exists():
        return app_path

    return RESOURCE_ROOT / path


def resolve_device(device: str | None) -> str | int:
    if device:
        return int(device) if device.isdigit() else device
    return 0 if torch.cuda.is_available() else "cpu"


def require_file(path_value: str | Path, description: str) -> Path:
    path = resolve_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def active_device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def _resolve_dataset_entry(base_path: Path, entry: str | list[str]) -> str | list[str]:
    if isinstance(entry, list):
        return [str((base_path / Path(item)).resolve()).replace("\\", "/") for item in entry]

    entry_path = Path(entry)
    if entry_path.is_absolute():
        return str(entry_path).replace("\\", "/")
    return str((base_path / entry_path).resolve()).replace("\\", "/")


def prepare_dataset_config(path_value: str | Path) -> Path:
    source_path = require_file(path_value, "Dataset config")

    with source_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Dataset config must be a mapping: {source_path}")

    yaml_dir = source_path.parent
    base_path_value = data.get("path")
    if base_path_value:
        base_path = Path(base_path_value)
        if not base_path.is_absolute():
            candidate_from_yaml = (yaml_dir / base_path).resolve()
            candidate_from_root = (ROOT / base_path).resolve()
            base_path = candidate_from_yaml if candidate_from_yaml.exists() else candidate_from_root
    else:
        base_path = yaml_dir

    for split_name in ("train", "val", "test"):
        split_value = data.get(split_name)
        if split_value:
            data[split_name] = _resolve_dataset_entry(base_path, split_value)

    data.pop("path", None)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    resolved_path = GENERATED_DIR / f"{source_path.stem}.resolved.yaml"
    with resolved_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return resolved_path