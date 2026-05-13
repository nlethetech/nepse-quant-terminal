#!/usr/bin/env python3
"""Create/update the local environment and launch the Textual dashboard."""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Sequence

MIN_PYTHON = (3, 10)
MAX_PYTHON_EXCLUSIVE = (3, 14)
VENV_DIRNAME = ".venv"
REQUIREMENTS_STAMP = ".nepse_quant_requirements.sha256"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_supported_python(version: Sequence[int] | None = None) -> bool:
    parts = tuple(version or sys.version_info[:3])
    return MIN_PYTHON <= parts[:2] < MAX_PYTHON_EXCLUSIVE


def requirements_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / VENV_DIRNAME / "Scripts" / "python.exe"
    return root / VENV_DIRNAME / "bin" / "python"


def run(cmd: Sequence[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(str(part) for part in cmd)}", flush=True)
    return subprocess.run([str(part) for part in cmd], cwd=str(cwd), check=check)


def ensure_supported_python() -> None:
    if is_supported_python():
        return
    current = ".".join(map(str, sys.version_info[:3]))
    raise SystemExit(
        f"Python {current} is not supported. Install Python 3.10 through 3.13, "
        "then launch again. Python 3.12 is recommended."
    )


def ensure_virtualenv(root: Path) -> Path:
    python = venv_python(root)
    if python.exists():
        return python
    print(f"Creating local Python environment at {root / VENV_DIRNAME}...")
    run([sys.executable, "-m", "venv", root / VENV_DIRNAME], cwd=root)
    return python


def ensure_requirements(root: Path, python: Path) -> None:
    requirements = root / "requirements.txt"
    stamp = root / VENV_DIRNAME / REQUIREMENTS_STAMP
    current_hash = requirements_hash(requirements)
    if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == current_hash:
        print("Python dependencies are already installed.")
        return

    print("Installing Python dependencies. First launch can take a few minutes...")
    run([python, "-m", "pip", "install", "-U", "pip"], cwd=root)
    try:
        run([python, "-m", "pip", "install", "-r", requirements], cwd=root)
    except subprocess.CalledProcessError as exc:
        print(
            "\nDependency install failed. Make sure Git is installed and available on PATH, "
            "then launch again."
        )
        raise SystemExit(exc.returncode) from exc
    stamp.write_text(current_hash + "\n", encoding="utf-8")


def ensure_database(root: Path, python: Path) -> None:
    probe = (
        "from pathlib import Path\n"
        "from backend.quant_pro.database import get_db_path\n"
        "path = Path(get_db_path())\n"
        "print(path)\n"
        "raise SystemExit(0 if path.exists() else 2)\n"
    )
    result = run([python, "-c", probe], cwd=root, check=False)
    if result.returncode == 0:
        print("Market database found.")
        return

    print("Market database is missing. Downloading the bundled database...")
    run([python, "setup_data.py"], cwd=root)


def run_preflight(root: Path, python: Path) -> None:
    print("Running launch preflight...")
    try:
        run([python, "-m", "scripts.ops.windows_preflight"], cwd=root)
    except subprocess.CalledProcessError as exc:
        print("\nPreflight failed. Fix the issue above, then launch again.")
        raise SystemExit(exc.returncode) from exc


def launch_tui(root: Path, python: Path) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("NEPSE_LAUNCHED_FROM_HELPER", platform.system())
    print("\nLaunching NEPSE Quant Terminal...\n")
    return subprocess.run(
        [str(python), "-m", "apps.tui.dashboard_tui"],
        cwd=str(root),
        env=env,
        check=False,
    ).returncode


def main() -> int:
    root = repo_root()
    os.chdir(root)
    ensure_supported_python()
    python = ensure_virtualenv(root)
    ensure_requirements(root, python)
    ensure_database(root, python)
    run_preflight(root, python)
    return launch_tui(root, python)


if __name__ == "__main__":
    raise SystemExit(main())
