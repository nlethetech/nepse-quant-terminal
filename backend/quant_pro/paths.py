"""Shared path helpers for locating the project root after package moves."""

from __future__ import annotations

import shutil
from pathlib import Path


def get_project_root(anchor: str | Path | None = None) -> Path:
    """Resolve the repository root from any module path inside the project."""
    start = Path(anchor).resolve() if anchor else Path(__file__).resolve()
    candidates = [start] + list(start.parents)
    for candidate in candidates:
        if (candidate / "requirements.txt").exists() and (candidate / "data").exists():
            return candidate
    for candidate in candidates:
        if (candidate / "README.md").exists() and (candidate / "backend").exists():
            return candidate
    return start.parent if start.is_file() else start


def get_data_dir(anchor: str | Path | None = None) -> Path:
    return get_project_root(anchor) / "data"


def get_runtime_dir(anchor: str | Path | None = None) -> Path:
    return get_data_dir(anchor) / "runtime"


def get_trading_runtime_dir(anchor: str | Path | None = None) -> Path:
    return get_runtime_dir(anchor) / "trading"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def migrate_legacy_path(target: Path, legacy_paths: list[str | Path]) -> Path:
    """Move the first legacy file into the new location when needed."""
    target = Path(target).expanduser().resolve()
    ensure_dir(target.parent)
    if target.exists():
        return target
    for legacy in legacy_paths:
        source = Path(legacy).expanduser().resolve()
        if source == target or not source.exists():
            continue
        try:
            shutil.move(str(source), str(target))
        except shutil.Error:
            shutil.copy2(str(source), str(target))
        if target.exists():
            break
    return target
