"""Hard guards for every test in this repository.

Tests must never touch a real person's keys, settings or accounts, and must
never reach the internet (nepsequant.com, OpenAI, Anthropic, NEPSE, ...).
These guards are installed when pytest loads this file — before any test
module is imported — so even code that runs at import time is covered:

1. HOME and XDG_CONFIG_HOME point at a throwaway folder, so
   ~/.config/nepse-quant-terminal (credentials.json, ai.json) is never read
   or written. Every test also gets its own fresh folder.
2. Every API key and endpoint override is removed from the environment.
3. Network connections are blocked except to this computer (127.0.0.1,
   ::1, localhost, Unix sockets). A test that tries is failed with a clear
   message instead of quietly calling a real service.
4. At the end of the run, the real ~/.config/nepse-quant-terminal folder is
   compared with its state at the start; any change fails the whole run.
"""

from __future__ import annotations

import hashlib
import os
import socket
import sys
import tempfile
from pathlib import Path

import pytest

# The real folder, captured before anything below changes HOME.
_REAL_CONFIG = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "nepse-quant-terminal"

_SECRET_ENV = (
    "NEPSE_QUANT_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_BASE_URL",
    "OPENAI_BASE_URL",
    "OLLAMA_HOST",
    "NEPSE_AGENT_BACKEND",
    "NEPSE_AGENT_MODEL",
    "NQT_BASE_URL",
    "NQT_SHARED_BOOK",
)

_SANDBOX = Path(tempfile.mkdtemp(prefix="nqt-test-home-"))
os.environ["HOME"] = str(_SANDBOX)
os.environ["USERPROFILE"] = str(_SANDBOX)  # Windows: Path.home() reads USERPROFILE, not HOME
os.environ["XDG_CONFIG_HOME"] = str(_SANDBOX / ".config")
for _name in _SECRET_ENV:
    os.environ.pop(_name, None)

# ── network guard ────────────────────────────────────────────────────────────
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


class NetworkBlocked(RuntimeError):
    pass


def _check(address) -> None:
    if isinstance(address, (str, bytes)):  # a Unix socket path
        return
    host = str(address[0]) if address else ""
    if host in _LOCAL_HOSTS or host.startswith("127."):
        return
    raise NetworkBlocked(
        f"Tests may not use the network (tried {host}). Use a fake request function or a local server."
    )


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _guarded_connect(self, address):
    _check(address)
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    _check(address)
    return _real_connect_ex(self, address)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex


# ── the real config folder must not change ───────────────────────────────────
def _fingerprint(folder: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            try:
                out[str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                out[str(path.relative_to(folder))] = "unreadable"
    return out


_REAL_CONFIG_AT_START = _fingerprint(_REAL_CONFIG)


def pytest_sessionfinish(session, exitstatus):
    if _fingerprint(_REAL_CONFIG) != _REAL_CONFIG_AT_START:
        sys.stderr.write(
            f"\nSAFETY FAILURE: a test changed the real {_REAL_CONFIG} folder (keys/settings). "
            "Tests must only use their own throwaway folder.\n"
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


# ── per test: a fresh, empty config folder and clean caches ──────────────────
@pytest.fixture(autouse=True)
def _isolate_user_config(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    for name in _SECRET_ENV:
        monkeypatch.delenv(name, raising=False)
    briefing = sys.modules.get("backend.agents.nepse_briefing")
    if briefing is not None:
        briefing._CACHE.clear()
    analyst = sys.modules.get("backend.agents.agent_analyst")
    if analyst is not None:
        analyst._BASE_CONTEXT_CACHE.clear()
        analyst._MACRO_CACHE.clear()
    yield
