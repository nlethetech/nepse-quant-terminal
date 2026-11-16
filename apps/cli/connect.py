"""Connect this terminal to your NEPSE Quant account.

    ./nqt-connect login      paste a personal key from nepsequant.com
    ./nqt-connect status     show which account is connected
    ./nqt-connect paper      show your paper portfolio
    ./nqt-connect real       show your real portfolio
    ./nqt-connect logout     remove the saved key

    ./nqt-connect ai setup   choose the AI analyst: Ollama (local), OpenAI or Claude (your own API key)
    ./nqt-connect ai status  show the AI provider, model and whether a key is saved (never the key)
    ./nqt-connect ai off     delete saved AI keys and go back to local Ollama

With a key made with "Paper trading" ticked, the terminal also uses your
nepsequant.com paper portfolio as its paper book (apps/tui/io/website_book.py):
one book on the website and here. This command line itself only reads. The key
is saved on this computer with owner-only permissions and is only ever sent to
nepsequant.com.
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
import zlib
from pathlib import Path
from typing import Any

SITE = "https://nepsequant.com"
KEYS_PAGE = f"{SITE}/desk/settings"
KEY_PREFIX = "nqk_"
USER_AGENT = "nepse-quant-terminal/connect"
TIMEOUT_S = 30
MAX_BODY_BYTES = 256 * 1024 * 1024  # a gzip reply is never unpacked past this


class ConnectError(Exception):
    """A problem the user can act on; shown without a traceback."""


# Where a NEPSE Quant key may ever be sent: the website itself, or this
# computer (a local copy of the website, for testing). Nothing else, whatever
# NQT_BASE_URL says — so a wrong or planted setting cannot leak the key.
ALLOWED_HOSTS = {"nepsequant.com", "www.nepsequant.com"}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def base_url() -> str:
    url = os.environ.get("NQT_BASE_URL", SITE).strip().rstrip("/")
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in LOCAL_HOSTS and parsed.scheme in {"http", "https"}:
        return url
    if parsed.scheme != "https":
        raise ConnectError("NQT_BASE_URL must use https, so your key is never sent unencrypted.")
    if host not in ALLOWED_HOSTS or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ConnectError(
            "NQT_BASE_URL may only point at https://nepsequant.com (or this computer, for testing), "
            "so your key is never sent anywhere else."
        )
    return url


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow redirects. urllib would otherwise re-send the
    Authorization header (the key) to wherever a server points it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


_KEYED_OPENER = urllib.request.build_opener(_NoRedirect)


def open_keyed(request: urllib.request.Request, timeout: float):
    """Open a request that carries a key: never follows a redirect."""
    return _KEYED_OPENER.open(request, timeout=timeout)  # noqa: S310 - callers check the scheme


def _redacted(message: str, key: str) -> str:
    """Never show the key in a message, even if a server echoed it back."""
    return message.replace(key, "[key removed]") if key else message


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


def _detail(exc: urllib.error.HTTPError) -> str:
    """The website's own one-line explanation of a refusal, when it sent one."""
    try:
        detail = json.loads(exc.read().decode("utf-8")).get("detail")
    except Exception:
        return ""
    return detail.strip() if isinstance(detail, str) else ""


def _gunzip(raw: bytes) -> bytes:
    unpack = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out = unpack.decompress(raw, MAX_BODY_BYTES)
    except zlib.error as exc:
        raise ValueError(f"damaged gzip reply: {exc}") from exc
    if unpack.unconsumed_tail:
        raise ValueError("reply too large")
    return out


def request_json(
    path: str,
    key: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: float = TIMEOUT_S,
    compressed: bool = False,
) -> dict[str, Any]:
    """``compressed=True`` asks for a gzip reply (large market data) and unpacks it."""
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json", "User-Agent": USER_AGENT}
    if compressed:
        headers["Accept-Encoding"] = "gzip"
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url() + path, data=data, headers=headers, method=method)
    try:
        with open_keyed(req, timeout) as response:
            raw = response.read()
            encoding = str(getattr(response, "headers", {}).get("Content-Encoding", "") or "").strip().lower()
            if compressed and encoding == "gzip":
                raw = _gunzip(raw)
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise ConnectError(f"That key was not accepted. Create a new one at {KEYS_PAGE}") from exc
        if 300 <= exc.code < 400:
            raise ConnectError(
                f"{base_url()} answered with a redirect ({exc.code}); not following it, so your key stays private."
            ) from exc
        if exc.code == 429:
            raise ConnectError(_redacted(_detail(exc), key) or "Too many requests. Wait a minute and try again.") from exc
        if exc.code in (403, 404, 409, 422, 428):
            raise ConnectError(_redacted(_detail(exc), key) or f"NEPSE Quant refused the request ({exc.code}).") from exc
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


AI_KEY_PAGES = {
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "the Claude Console (platform.claude.com) under API keys",
}
AI_NAMES = {"ollama": "Ollama", "openai": "OpenAI", "anthropic": "Anthropic (Claude)"}


def _ai_default_model(provider: str) -> str:
    from backend.agents import ai_settings
    from backend.agents.runtime_config import AGENT_BACKEND_PRESETS

    default = ai_settings.remembered_model(provider) or str(AGENT_BACKEND_PRESETS[provider].get("model") or "")
    if provider != "ollama":
        return default
    from backend.agents import llm_ollama

    try:  # offer a model that is actually installed
        installed = llm_ollama.installed_models(llm_ollama.ollama_host(), use_cache=False)
    except llm_ollama.OllamaUnavailable:
        return default
    return llm_ollama.resolve_model(default, installed)[0] or llm_ollama.SUGGESTED_MODEL


def ai_setup(
    provider: str | None = None,
    model: str | None = None,
    *,
    ask: Any = None,
    secret: Any = None,
) -> str:
    """Pick the analyst's provider, save a key (hidden prompt) and an optional model."""
    from backend.agents import ai_settings
    from backend.agents.runtime_config import set_active_agent

    ask = ask or input
    secret = secret or getpass.getpass

    if provider is None:
        print("Which AI should the terminal's analyst use?")
        print("  1) ollama     free, runs on this computer (install from ollama.com)")
        print("  2) openai     your own OpenAI API key (default model gpt-5-mini)")
        print("  3) anthropic  your own Anthropic API key for Claude (default claude-opus-5-5; low-cost claude-haiku-5-5)")
        answer = str(ask("Choose 1, 2 or 3 [1]: ")).strip().lower() or "1"
        provider = {"1": "ollama", "2": "openai", "3": "anthropic"}.get(answer, answer)
    provider = str(provider).strip().lower()
    if provider not in ai_settings.PROVIDERS:
        raise ConnectError("Choose one of: ollama, openai, anthropic.")

    key = ""
    notes: list[str] = []
    if provider in ai_settings.CLOUD_PROVIDERS:
        _existing, source = ai_settings.get_api_key(provider)
        print(f"Get a key from {AI_KEY_PAGES[provider]}. It is saved only on this computer and sent only to {AI_NAMES[provider]}.")
        keep = " (press Enter to keep the saved one)" if source == "file" else ""
        key = str(secret(f"Paste your {AI_NAMES[provider]} API key (hidden){keep}: ")).strip()
        if key.startswith(KEY_PREFIX):
            raise ConnectError("That is a NEPSE Quant key. Save it with ./nqt-connect login instead.")
        if not key and not source:
            raise ConnectError("No key entered, so nothing was changed.")
        if key and provider == "anthropic" and not key.startswith("sk-ant-"):
            notes.append("Note: Anthropic keys usually start with sk-ant-. Saved anyway.")
        if key and provider == "openai" and not key.startswith("sk-") and "api.openai.com" in os.environ.get("OPENAI_BASE_URL", "api.openai.com"):
            notes.append("Note: OpenAI keys usually start with sk-. Saved anyway.")
        if source == "env":
            notes.append(f"{ai_settings.ENV_KEYS[provider]} is set in your environment, so that key is used instead of the saved one.")

    if model is None:
        default = _ai_default_model(provider)
        model = str(ask(f"Model [{default}]: ")).strip() or default
    if ai_settings.looks_like_api_key(model):
        raise ConnectError("That looks like a key, not a model name. Nothing was changed.")
    path = ai_settings.update_ai_settings(provider=provider, model=model, key=key or None)
    set_active_agent(provider, model=model)

    lines = [f"AI analyst set to {AI_NAMES[provider]} · {model}. Settings saved to {path} (only you can read it)."]
    if provider == "ollama":
        lines.append(f"Make sure Ollama is running and the model is installed: ollama pull {model}")
    else:
        lines.append(f"Each chat answer and picks analysis is billed to your own {AI_NAMES[provider]} key.")
    lines += notes
    lines.append("Open the terminal's Agents tab (key 5) to use it.")
    return "\n".join(lines)


def ai_status(*, ollama_models: Any = None) -> str:
    """Provider, model and key presence - never the key itself."""
    from backend.agents import ai_settings, llm_ollama
    from backend.agents.runtime_config import load_active_agent_config

    cfg = load_active_agent_config()
    backend = str(cfg.get("backend") or "ollama")
    name = {"claude": "Claude CLI", "gemma4_mlx": "Gemma 4 MLX"}.get(backend) or AI_NAMES.get(backend, backend)
    lines = [f"AI analyst: {name} · {cfg.get('model') or 'default'}"]
    for provider in ai_settings.CLOUD_PROVIDERS:
        _key, source = ai_settings.get_api_key(provider)
        where = {"env": f"yes (from {ai_settings.ENV_KEYS[provider]})", "file": "yes (saved on this computer)"}.get(source, "no")
        lines.append(f"{AI_NAMES[provider]} key: {where}")
    host = llm_ollama.ollama_host(str(cfg.get("ollama_host") or ""))
    try:
        names = list(ollama_models) if ollama_models is not None else llm_ollama.installed_models(host, use_cache=False)
        installed = ", ".join(names[:5]) + (" ..." if len(names) > 5 else "") if names else f"no models yet (ollama pull {llm_ollama.SUGGESTED_MODEL})"
        lines.append(f"Ollama at {host}: running, {installed}")
    except llm_ollama.OllamaUnavailable:
        lines.append(f"Ollama at {host}: not running")
    path = ai_settings.ai_config_path()
    lines.append(f"Settings: {path}" + ("" if path.exists() else " (not created yet)"))
    lines.append("Cloud models bill your own API key. Keys stay on this computer. Change with ./nqt-connect ai setup")
    return "\n".join(lines)


def ai_off() -> str:
    from backend.agents import ai_settings
    from backend.agents.runtime_config import load_active_agent_config, set_active_agent

    removed = ai_settings.clear_ai_settings()
    lines = ["Removed the saved AI keys and settings." if removed else "No saved AI keys to remove."]
    if str(load_active_agent_config().get("backend") or "") in ai_settings.CLOUD_PROVIDERS:
        set_active_agent("ollama")
        lines.append("The analyst is back on Ollama (local, free).")
    for provider, env_name in ai_settings.ENV_KEYS.items():
        if os.environ.get(env_name, "").strip():
            lines.append(f"{env_name} is still set in your environment; unset it to stop using that key.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nqt-connect", description="Connect this terminal to your nepsequant.com account.")
    commands = parser.add_subparsers(dest="command", required=True)
    login_parser = commands.add_parser("login", help="save a personal key from nepsequant.com")
    login_parser.add_argument("--key", help="the key; omit to be asked without it showing on screen")
    commands.add_parser("status", help="check the saved key")
    commands.add_parser("logout", help="remove the saved key")
    for name, text in (("paper", "show your paper portfolio"), ("real", "show your real portfolio")):
        book = commands.add_parser(name, help=text)
        book.add_argument("--json", action="store_true", help="print the raw data")
    ai = commands.add_parser("ai", help="choose the AI analyst: ollama, openai or anthropic (Claude)")
    ai_commands = ai.add_subparsers(dest="ai_command", required=True)
    setup = ai_commands.add_parser("setup", help="pick a provider, paste a key (hidden) and a model")
    setup.add_argument("--provider", choices=["ollama", "openai", "anthropic"], help="skip the question")
    setup.add_argument("--model", help="model name; omit to be asked (Enter keeps the default)")
    ai_commands.add_parser("status", help="show provider, model and whether keys are saved")
    ai_commands.add_parser("off", help="delete saved AI keys and go back to local Ollama")
    args = parser.parse_args(argv)

    try:
        if args.command == "ai":
            if args.ai_command == "setup":
                print(ai_setup(args.provider, args.model))
            elif args.ai_command == "status":
                print(ai_status())
            else:
                print(ai_off())
        elif args.command == "login":
            print(login(args.key))
        elif args.command == "logout":
            credentials_path().unlink(missing_ok=True)
            print("Disconnected. To cancel the key itself, revoke it at " + KEYS_PAGE)
        elif args.command == "status":
            info = request_json("/api/keys/verify", load_key())
            scopes = {s.strip() for s in str(info.get("scopes") or "read").split(",")}
            access = "read + paper trading" if "paper" in scopes else "read only"
            who = info.get("user") or info.get("email") or "your account"
            print(f"Connected to {base_url()} as {who} ({access})")
            if "paper" not in scopes:
                print(f"To trade your paper book from the terminal, create a key with Paper trading ticked at {KEYS_PAGE}")
        else:
            data = request_json(f"/api/cli/account?book={args.command}", load_key())
            print(json.dumps(data, indent=2) if args.json else render_account(data))
    except ConnectError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
