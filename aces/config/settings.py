"""
Centralized project settings and path resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import os


DEFAULT_DATASETS_DIR = "datasets_unified"
DEFAULT_SCREENSHOTS_DIR = "screenshots"

_DATASET_CANDIDATES = [
    "datasets_unified",
    "datasets",
    "../ACES/datasets",
]


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_datasets_dir(override: Optional[str] = None) -> Path:
    """
    Resolve datasets directory with sane defaults.
    Order: override -> ACES_DATASETS_DIR -> fallbacks.
    """
    candidates = []
    if override:
        candidates.append(Path(override))

    env_path = os.getenv("ACES_DATASETS_DIR") or os.getenv("DATASETS_DIR")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(Path(p) for p in _DATASET_CANDIDATES)

    resolved = _first_existing(candidates)
    return resolved or Path(override or env_path or DEFAULT_DATASETS_DIR)


def resolve_screenshots_dir(override: Optional[str] = None) -> Path:
    """
    Resolve screenshots directory with sane defaults.
    """
    if override:
        return Path(override)
    env_path = os.getenv("ACES_SCREENSHOTS_DIR")
    return Path(env_path or DEFAULT_SCREENSHOTS_DIR)
