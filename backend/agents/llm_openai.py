"""OpenAI (or any OpenAI-compatible server) for the AI analyst, on the user's own key.

Plain Chat Completions over urllib, so no extra package is needed:

    POST {base}/chat/completions      base default https://api.openai.com/v1

``OPENAI_BASE_URL`` (or ``openai_base_url`` in ai.json) points it at another
OpenAI-compatible server; https is required unless the server is on this
computer. The key comes from ``OPENAI_API_KEY`` or ``./nqt-connect ai setup``.

Reasoning models (gpt-5 family, o-series) take ``max_completion_tokens`` (which
also covers their hidden reasoning), a ``reasoning_effort`` and no temperature.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from backend.agents.ai_settings import get_api_key, setting, redact_keys

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
USER_AGENT = "nepse-quant-terminal/agent"
TIMEOUTS_S = {"chat": 60.0, "analysis": 300.0}
EFFORT = {"chat": "low", "analysis": "medium"}

_LAST_USAGE: dict[str, Any] = {}


def base_url() -> str:
    url = (os.environ.get("OPENAI_BASE_URL") or setting("openai_base_url") or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
    parsed = urllib.parse.urlparse(url)
    local = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "https" and not local:
        raise ValueError("OPENAI_BASE_URL must use https (or http://localhost), so your key is never sent unencrypted.")
    return url


def is_reasoning_model(model: str) -> bool:
    name = str(model or "").strip().lower().rsplit("/", 1)[-1]
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


def build_request(
    prompt: str,
    *,
    system: str,
    model: str,
    max_tokens: int,
    purpose: str = "chat",
    history: Optional[list[dict]] = None,
) -> dict[str, Any]:
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m["role"], "content": m["content"]} for m in list(history or [])]
    messages.append({"role": "user", "content": prompt})
    body: dict[str, Any] = {"model": model, "messages": messages, "max_completion_tokens": int(max_tokens)}
    if is_reasoning_model(model):
        body["reasoning_effort"] = EFFORT.get(purpose, "low")
    else:
        body["temperature"] = 0.15
    return body


def _who(url: str) -> str:
    host = urllib.parse.urlparse(url).hostname or url
    return "OpenAI" if host == "api.openai.com" else f"the OpenAI-compatible server at {host}"


def _error_detail(exc: urllib.error.HTTPError) -> tuple[str, str]:
    """(error code, message) from an OpenAI error body, when it has one."""
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except Exception:
        return "", ""
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return "", ""
    code = str(error.get("code") or error.get("type") or "")
    message = " ".join(str(error.get("message") or "").split())[:200]
    return code, message


def http_error_message(exc: urllib.error.HTTPError, *, model: str, who: str = "OpenAI") -> str:
    code, message = _error_detail(exc)
    status = int(getattr(exc, "code", 0) or 0)
    if status == 401:
        return f"ERROR: {who} did not accept the API key (401). Save a new one with ./nqt-connect ai setup."
    if status == 404 or code == "model_not_found":
        return f"ERROR: {who} has no model \"{model}\" for this key. Choose another with /model <name>, e.g. /model {DEFAULT_OPENAI_MODEL}."
    if status == 429 and code == "insufficient_quota":
        return f"ERROR: {who} says this key has no credit left. Add billing on platform.openai.com, or switch with /agent ollama."
    if status == 429:
        return f"ERROR: {who} rate limit reached (429). Wait a minute and try again."
    if status == 403:
        return f"ERROR: {who} refused this key for \"{model}\" (403){': ' + message if message else ''}. Try /model <name>."
    if status >= 500:
        return f"ERROR: {who} is having problems ({status}). Try again shortly."
    return f"ERROR: {who} refused the request ({status}){': ' + message if message else ''}."


def last_usage() -> dict[str, Any]:
    return dict(_LAST_USAGE)


def call_openai(
    prompt: str,
    *,
    system: str,
    max_tokens: int,
    purpose: str = "chat",
    history: Optional[list[dict]] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    urlopen: Optional[Callable[..., Any]] = None,
) -> str:
    """One answer from the model, or a one-line ``ERROR: ...`` saying what to do."""
    model = str(model or DEFAULT_OPENAI_MODEL).strip()
    key, _source = get_api_key("openai")
    if not key:
        return "ERROR: No OpenAI API key. Run ./nqt-connect ai setup (or set OPENAI_API_KEY), or switch with /agent ollama."
    try:
        url = base_url()
    except ValueError as exc:
        return f"ERROR: {exc}"
    who = _who(url)
    body = build_request(prompt, system=system, model=model, max_tokens=max_tokens, purpose=purpose, history=history)
    request = urllib.request.Request(
        url + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    wait = float(timeout or TIMEOUTS_S.get(purpose, 60.0))
    if urlopen is None:
        from apps.cli.connect import open_keyed  # never follows a redirect with the key

        opener = open_keyed
    else:
        opener = urlopen
    started = time.monotonic()
    try:
        with opener(request, timeout=wait) as response:  # noqa: S310 - scheme checked in base_url
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return redact_keys(http_error_message(exc, model=model, who=who))
    except (TimeoutError, socket.timeout):
        return f"ERROR: {who} did not answer within {int(wait)}s. Try again, or ask a shorter question."
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            return f"ERROR: {who} did not answer within {int(wait)}s. Try again, or ask a shorter question."
        return f"ERROR: Could not reach {who}. Check your internet connection."
    except ValueError:
        return f"ERROR: {who} sent a reply that is not JSON."

    choices = data.get("choices") if isinstance(data, dict) else None
    choice = dict((choices or [{}])[0] or {})
    message = dict(choice.get("message") or {})
    content = message.get("content") or ""
    if isinstance(content, list):  # some compatible servers send content parts
        content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    text = str(content).strip()
    usage = dict(data.get("usage") or {}) if isinstance(data, dict) else {}
    _LAST_USAGE.clear()
    _LAST_USAGE.update(
        {
            "model": str(data.get("model") or model) if isinstance(data, dict) else model,
            "input_tokens": usage.get("prompt_tokens"),
            "cached_input_tokens": dict(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "seconds": round(time.monotonic() - started, 2),
        }
    )
    if text:
        return text
    if message.get("refusal"):
        return f"ERROR: The model declined: {' '.join(str(message['refusal']).split())[:200]}"
    if choice.get("finish_reason") == "length":
        return "ERROR: The model used its whole token allowance before answering. Ask a narrower question, or try another model with /model."
    return f"ERROR: {who} returned an empty answer."
