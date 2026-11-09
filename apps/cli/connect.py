"""Connect this terminal to your NEPSE Quant account.

    ./nqt-connect login      paste a personal key from nepsequant.com
    ./nqt-connect status     show which account is connected
    ./nqt-connect paper      show your paper portfolio
    ./nqt-connect real       show your real portfolio
    ./nqt-connect logout     remove the saved key

The connection is read-only. The key is saved on this computer with
owner-only permissions and is only ever sent to nepsequant.com.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SITE = "https://nepsequant.com"
KEYS_PAGE = f"{SITE}/desk/settings"
KEY_PREFIX = "nqk_"
USER_AGENT = "nepse-quant-terminal/connect"
TIMEOUT_S = 30


class ConnectError(Exception):
    """A problem the user can act on; shown without a traceback."""


def base_url() -> str:
    url = os.environ.get("NQT_BASE_URL", SITE).rstrip("/")
    parsed = urllib.parse.urlparse(url)
    local = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
    if parsed.scheme != "https" and not local:
        raise ConnectError("NQT_BASE_URL must use https, so your key is never sent unencrypted.")
    return url


def credentials_path() -> Path:
    root = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(root) / "nepse-quant-terminal" / "credentials.json"


def save_key(key: str) -> Path:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"api_key": key}, handle)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path


def load_key() -> str:
    from_env = os.environ.get("NEPSE_QUANT_API_KEY", "").strip()
    if from_env:
        return from_env
    try:
        key = str(json.loads(credentials_path().read_text()).get("api_key") or "")
    except (OSError, ValueError):
        key = ""
    if not key:
        raise ConnectError(f"Not connected. Run: ./nqt-connect login\nCreate a free key at {KEYS_PAGE}")
    return key


def request_json(path: str, key: str) -> dict[str, Any]:
    req = urllib.request.Request(
        base_url() + path,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as response:  # noqa: S310 - scheme checked in base_url
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise ConnectError(f"That key was not accepted. Create a new one at {KEYS_PAGE}") from exc
        if exc.code == 429:
            raise ConnectError("Too many requests. Wait a minute and try again.") from exc
        raise ConnectError(f"NEPSE Quant answered with an error ({exc.code}). Try again shortly.") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise ConnectError(f"Could not reach {base_url()}. Check your connection and try again.") from exc


def money(value: Any) -> str:
    return "n/a" if value is None else f"Rs {float(value):,.2f}"


def percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.2f}%"


def render_account(data: dict[str, Any]) -> str:
    book = str(data.get("book") or "")
    title = "Paper portfolio (virtual money)" if book == "paper" else "Real portfolio (read-only)"
    lines = [title, f"Value  {money(data.get('value'))}"]
    if data.get("cash") is not None:
        lines.append(f"Cash   {money(data.get('cash'))}")
    positions = data.get("positions") or []
    if not positions:
        lines.append("No holdings.")
    else:
        lines.append("")
        lines.append(f"{'SYMBOL':<12}{'SHARES':>10}{'AVG COST':>12}{'LAST':>12}{'VALUE':>16}{'RETURN':>10}")
        for row in positions:
            avg = row.get("avg_cost")
            last = row.get("last")
            lines.append(
                f"{str(row.get('symbol')):<12}{int(row.get('quantity') or 0):>10,}"
                f"{'n/a' if avg is None else format(float(avg), ',.2f'):>12}"
                f"{'n/a' if last is None else format(float(last), ',.2f'):>12}"
                f"{0.0 if row.get('value') is None else float(row['value']):>16,.2f}"
                f"{percent(row.get('return_pct')):>10}"
            )
    orders = data.get("open_orders") or []
    if orders:
        lines.append("")
        lines.append("Open orders")
        lines += [f"  {o.get('action')} {int(o.get('quantity') or 0):,} {o.get('symbol')} at {money(o.get('limit_price'))}" for o in orders]
    lines += ["", f"Full view: {SITE}/desk/portfolio"]
    return "\n".join(lines)


def login(key: str | None) -> str:
    if key is None:
        print(f"Create a free key at {KEYS_PAGE}")
        key = getpass.getpass("Paste your key (hidden): ")
    key = key.strip()
    if not key.startswith(KEY_PREFIX):
        raise ConnectError(f"That does not look like a NEPSE Quant key. Keys start with {KEY_PREFIX}")
    request_json("/api/keys/verify", key)
    path = save_key(key)
    return f"Connected to NEPSE Quant. Key saved to {path}\nTry: ./nqt-connect paper"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nqt-connect", description="Connect this terminal to nepsequant.com (read-only).")
    commands = parser.add_subparsers(dest="command", required=True)
    login_parser = commands.add_parser("login", help="save a personal key from nepsequant.com")
    login_parser.add_argument("--key", help="the key; omit to be asked without it showing on screen")
    commands.add_parser("status", help="check the saved key")
    commands.add_parser("logout", help="remove the saved key")
    for name, text in (("paper", "show your paper portfolio"), ("real", "show your real portfolio")):
        book = commands.add_parser(name, help=text)
        book.add_argument("--json", action="store_true", help="print the raw data")
    args = parser.parse_args(argv)

    try:
        if args.command == "login":
            print(login(args.key))
        elif args.command == "logout":
            credentials_path().unlink(missing_ok=True)
            print("Disconnected. To cancel the key itself, revoke it at " + KEYS_PAGE)
        elif args.command == "status":
            info = request_json("/api/keys/verify", load_key())
            print(f"Connected to {base_url()} as {info.get('email') or 'your account'} ({info.get('scopes') or 'read'} access)")
        else:
            data = request_json(f"/api/cli/account?book={args.command}", load_key())
            print(json.dumps(data, indent=2) if args.json else render_account(data))
    except ConnectError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
