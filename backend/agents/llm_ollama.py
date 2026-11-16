"""Local models served by Ollama (https://ollama.com) for the AI analyst.

Uses ``POST /api/chat`` with system + user messages. Before calling, it checks
``GET /api/tags``: when the configured model is not installed it uses the first
installed chat model and says so; with no models at all it says which one to
pull. Host: ``OLLAMA_HOST``, else the terminal's agent config, else
http://localhost:11434.

``call_ollama_tools`` offers the analyst's read-only lookups to models that
support tool calling. A model that does not (Ollama answers 400 "does not
support tools") is remembered and gets ``NO_TOOLS`` back, so the analyst can
fall back to pre-fetched data in the prompt.
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from backend.agents.ai_settings import setting

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
SUGGESTED_MODEL = "llama3.2"
TAGS_TTL_S = 60.0
TIMEOUTS_S = {"chat": 120.0, "analysis": 300.0}
# Ollama's default context window silently cuts long prompts; ask for one that
# fits. Fixed steps keep the model from reloading on every small size change.
CONTEXT_STEPS = (4096, 8192, 16384, 32768)

_TAGS: dict[str, tuple[float, list[str]]] = {}
_TAGS_LOCK = threading.Lock()
NO_TOOLS = "NO_TOOLS: this model cannot call tools"
_NO_TOOL_MODELS: set[str] = set()
_LAST: dict[str, Any] = {}
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


class OllamaUnavailable(Exception):
    pass


def ollama_host(configured: str = "") -> str:
    raw = (os.environ.get("OLLAMA_HOST") or configured or setting("ollama_host") or DEFAULT_OLLAMA_HOST).strip()
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    return raw.rstrip("/")


def _canonical(name: str) -> str:
    clean = str(name or "").strip().lower()
    return clean if ":" in clean else f"{clean}:latest"


def installed_models(
    host: str,
    *,
    urlopen: Optional[Callable[..., Any]] = None,
    timeout: float = 3.0,
    use_cache: bool = True,
) -> list[str]:
    """Names of the installed models (newest first, as Ollama lists them)."""
    now = time.monotonic()
    if use_cache:
        with _TAGS_LOCK:
            hit = _TAGS.get(host)
        if hit and now - hit[0] < TAGS_TTL_S:
            return list(hit[1])
    opener = urlopen or urllib.request.urlopen
    try:
        with opener(urllib.request.Request(f"{host}/api/tags", method="GET"), timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, ValueError) as exc:
        raise OllamaUnavailable(str(exc)) from exc
    names = [
        str(row.get("name") or row.get("model") or "").strip()
        for row in list((payload or {}).get("models") or [])
        if isinstance(row, dict)
    ]
    names = [name for name in names if name]
    with _TAGS_LOCK:
        _TAGS[host] = (now, names)
    return names


def resolve_model(configured: str, installed: list[str]) -> tuple[str, str]:
    """(model to use, notice for the status line). Model is "" when none is installed."""
    if not installed:
        return "", ""
    wanted = _canonical(configured)
    for name in installed:
        if _canonical(name) == wanted:
            return name, ""
    chat_models = [name for name in installed if "embed" not in name.lower()] or installed
    pick = chat_models[0]
    shown = configured or "(none)"
    return pick, f"Ollama has no model '{shown}', so the analyst used '{pick}'. Pick one with /model <name>."


def no_models_message() -> str:
    return f"ERROR: Ollama has no models installed. Run: ollama pull {SUGGESTED_MODEL}   then /model {SUGGESTED_MODEL}"


def not_running_message(host: str) -> str:
    return f"ERROR: Ollama is not running at {host}. Open the Ollama app (or run: ollama serve), or switch with /agent openai or /agent anthropic."


def context_window(*texts: str, max_tokens: int = 0) -> int:
    estimate = sum(len(text or "") for text in texts) // 3 + int(max_tokens or 0) + 256
    for step in CONTEXT_STEPS:
        if estimate <= step:
            return step
    return CONTEXT_STEPS[-1]


def last_call() -> dict[str, Any]:
    return dict(_LAST)


def build_request(
    prompt: str,
    *,
    system: str,
    model: str,
    max_tokens: int,
    history: Optional[list[dict]] = None,
    temperature: float = 0.15,
) -> dict[str, Any]:
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m["role"], "content": m["content"]} for m in list(history or [])]
    messages.append({"role": "user", "content": prompt})
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": int(max_tokens),
            "num_ctx": context_window(*(m["content"] for m in messages), max_tokens=max_tokens),
            "temperature": float(temperature),
        },
    }


def _chat(
    host: str,
    body: dict[str, Any],
    *,
    chosen: str,
    purpose: str,
    urlopen: Optional[Callable[..., Any]],
) -> tuple[Optional[dict], str, str]:
    """(reply, "", "") or (None, error line, raw error detail)."""
    request = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    wait = TIMEOUTS_S.get(purpose, TIMEOUTS_S["chat"])
    opener = urlopen or urllib.request.urlopen
    try:
        with opener(request, timeout=wait) as response:
            return json.loads(response.read().decode("utf-8")), "", ""
    except urllib.error.HTTPError as exc:
        try:
            detail = str(json.loads(exc.read().decode("utf-8")).get("error") or "")
        except Exception:
            detail = ""
        if exc.code == 404:
            with _TAGS_LOCK:
                _TAGS.pop(host, None)
            return None, f"ERROR: Ollama has no model '{chosen}'. Run: ollama pull {chosen}   or pick another with /model <name>.", detail
        return None, f"ERROR: Ollama answered with an error ({exc.code}){': ' + ' '.join(detail.split())[:160] if detail else ''}.", detail
    except (TimeoutError, socket.timeout):
        return None, f"ERROR: Ollama did not answer within {int(wait)}s. Try a smaller model (/model {SUGGESTED_MODEL}) or a shorter question.", ""
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            return None, f"ERROR: Ollama did not answer within {int(wait)}s. Try a smaller model (/model {SUGGESTED_MODEL}) or a shorter question.", ""
        return None, not_running_message(host), ""
    except ValueError:
        return None, "ERROR: Ollama sent a reply that is not JSON.", ""


def _resolve(host: str, model: str, urlopen: Optional[Callable[..., Any]]) -> tuple[str, str]:
    """(chosen model, "") or ("", error line)."""
    _LAST.clear()
    _LAST.update({"host": host, "configured_model": model, "model": model, "notice": ""})
    try:
        installed = installed_models(host, urlopen=urlopen)
    except OllamaUnavailable:
        return "", not_running_message(host)
    chosen, notice = resolve_model(model, installed)
    if not chosen:
        return "", no_models_message()
    _LAST.update({"model": chosen, "notice": notice})
    return chosen, ""


def _text(data: dict) -> str:
    text = str(dict(data.get("message") or {}).get("content") or "")
    return _THINK_BLOCK.sub("", text).strip()


def call_ollama(
    prompt: str,
    *,
    system: str,
    max_tokens: int,
    model: str,
    host: str = "",
    purpose: str = "chat",
    history: Optional[list[dict]] = None,
    temperature: float = 0.15,
    urlopen: Optional[Callable[..., Any]] = None,
) -> str:
    """One answer from a local model, or a one-line ``ERROR: ...`` saying what to do."""
    host = ollama_host(host)
    chosen, error = _resolve(host, model, urlopen)
    if error:
        return error
    body = build_request(prompt, system=system, model=chosen, max_tokens=max_tokens, history=history, temperature=temperature)
    started = time.monotonic()
    data, error, _detail = _chat(host, body, chosen=chosen, purpose=purpose, urlopen=urlopen)
    if error:
        return error
    _LAST.update(
        {
            "seconds": round(time.monotonic() - started, 2),
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "num_ctx": body["options"]["num_ctx"],
        }
    )
    return _text(data) or "ERROR: Ollama returned an empty answer. Try again, or another model with /model <name>."


def supports_tools(model: str) -> bool:
    """False once Ollama has said this model cannot call tools."""
    return _canonical(model) not in _NO_TOOL_MODELS


def call_ollama_tools(
    prompt: str,
    *,
    system: str,
    max_tokens: int,
    model: str,
    session: Any,
    host: str = "",
    purpose: str = "chat",
    history: Optional[list[dict]] = None,
    temperature: float = 0.15,
    urlopen: Optional[Callable[..., Any]] = None,
    max_rounds: int = 3,
    deadline_s: float = 90.0,
) -> str:
    """An answer that may look things up first, for models with tool calling.
    Returns ``NO_TOOLS`` when the model cannot call tools (nothing was answered)."""
    host = ollama_host(host)
    chosen, error = _resolve(host, model, urlopen)
    if error:
        return error
    if not supports_tools(chosen):
        return NO_TOOLS
    body = build_request(prompt, system=system, model=chosen, max_tokens=max_tokens, history=history, temperature=temperature)
    tools = [{"type": "function", "function": {"name": t["function"]["name"], "description": t["function"]["description"],
                                               "parameters": t["function"]["parameters"]}} for t in session.openai_tools()]
    started = time.monotonic()
    rounds = 0
    totals = {"input_tokens": 0, "output_tokens": 0}
    data: dict = {}
    while True:
        final = rounds >= max_rounds or session.exhausted or (time.monotonic() - started) > deadline_s
        if final:
            body.pop("tools", None)  # Ollama has no tool_choice: the last request offers none
        else:
            body["tools"] = tools
        body["options"]["num_ctx"] = context_window(*(str(m.get("content") or "") for m in body["messages"]), max_tokens=max_tokens)
        data, error, detail = _chat(host, body, chosen=chosen, purpose=purpose, urlopen=urlopen)
        if error:
            if rounds == 0 and "support tools" in detail.lower():
                _NO_TOOL_MODELS.add(_canonical(chosen))
                return NO_TOOLS
            return error
        totals["input_tokens"] += int(data.get("prompt_eval_count") or 0)
        totals["output_tokens"] += int(data.get("eval_count") or 0)
        message = dict(data.get("message") or {})
        calls = [c for c in list(message.get("tool_calls") or []) if isinstance(c, dict)]
        if not calls or final:
            break
        rounds += 1
        body["messages"].append({"role": "assistant", "content": str(message.get("content") or ""), "tool_calls": calls})
        for call in calls:
            fn = dict(call.get("function") or {})
            text, _is_error = session.run(fn.get("name"), fn.get("arguments"))
            body["messages"].append({"role": "tool", "content": text, "tool_name": str(fn.get("name") or "")})
    _LAST.update(
        {
            **totals,
            "seconds": round(time.monotonic() - started, 2),
            "num_ctx": body["options"]["num_ctx"],
            "requests": rounds + 1,
        }
    )
    return _text(data) or "ERROR: Ollama returned an empty answer. Try again, or another model with /model <name>."
