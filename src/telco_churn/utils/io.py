"""Input/output helper utilities for persisting and loading artifacts."""

import json
from pathlib import Path
from typing import Any
import joblib
import yaml


def save_joblib(obj: Any, file_path: str | Path) -> Path:
    """Save an object using joblib."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, str(path))
    return path


def load_joblib(file_path: str | Path) -> Any:
    """Load an object using joblib."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at: {path}")
    return joblib.load(str(path))


def save_json(data: dict[str, Any] | list[Any], file_path: str | Path, indent: int = 2) -> Path:
    """Save a dictionary or list to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    return path


def load_json(file_path: str | Path) -> Any:
    """Load data from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data: dict[str, Any], file_path: str | Path) -> Path:
    """Save a dictionary to a YAML file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
