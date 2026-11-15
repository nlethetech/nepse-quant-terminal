"""Claude through the Anthropic API, on the user's own key, for the AI analyst.

Uses the official ``anthropic`` Python SDK (``pip install anthropic``). The key
comes from ``ANTHROPIC_API_KEY`` or ``./nqt-connect ai setup``; with neither,
the SDK's own credential lookup (e.g. ``ant auth login``) still applies.

Per request:
- the analyst's fixed system prompt is marked for prompt caching, so repeat
  calls within a few minutes read it from cache instead of paying full price;
- ``output_config.effort`` is "low" for chat and "medium" for the picks analysis;
- on models that support it, server-side ``fallbacks="default"`` (beta
  ``server-side-fallback-2026-07-01``) re-runs a declined request on the model
  Anthropic recommends, inside the same call;
- a ``refusal`` stop reason becomes a clear one-line message.

Default model ``claude-opus-5-5``; ``claude-haiku-5-5`` is the low-cost option.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Optional

from backend.agents.ai_settings import get_api_key

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
CHEAP_ANTHROPIC_MODEL = "claude-haiku-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept the server-side ``fallbacks`` parameter. Others (e.g. the
# Haiku line) have no server-side fallback, so the parameter is left out.
FALLBACK_MODELS = frozenset(
    {
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
        "claude-fable-5-2",
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-mythos-5-2",
        "claude-mythos-5-1",
    }
)
EFFORT = {"chat": "low", "analysis": "medium"}
# (seconds per attempt, retries) - chat stays inside the terminal's 90 s chat timeout.
REQUEST_LIMITS = {"chat": (40.0, 1), "analysis": (240.0, 2)}
INSTALL_HINT = "ERROR: The Claude API needs the anthropic package. Run: pip install anthropic  (or pip install -r requirements.txt)"
NO_KEY_HINT = "ERROR: No Anthropic API key. Run ./nqt-connect ai setup (or set ANTHROPIC_API_KEY), or switch with /agent ollama."

_CLIENTS: dict[str, Any] = {}
_CLIENT_LOCK = threading.Lock()
_LAST_USAGE: dict[str, Any] = {}


def supports_effort(model: str) -> bool:
    """Older models reject ``output_config.effort``; everything from Opus 4.5 on takes it."""
    name = str(model or "").strip().lower()
    if name.startswith(("claude-3", "claude-2", "claude-instant")):
        return False
    if name.startswith(("claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1", "claude-opus-4-0", "claude-sonnet-4-0")):
        return False
    return not re.fullmatch(r"claude-(opus|sonnet)-4(-\d{8})?", name)


def supports_fallbacks(model: str) -> bool:
    return str(model or "").strip().lower() in FALLBACK_MODELS


def build_params(
    prompt: str,
    *,
    system: str,
    model: str,
    max_tokens: int,
    purpose: str = "chat",
    history: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """The request body. The system prompt is fixed text, so it is the cached prefix;
    everything that changes per call (data, history, the question) comes after it."""
    messages = [{"role": m["role"], "content": m["content"]} for m in list(history or [])]
    messages.append({"role": "user", "content": prompt})
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": messages,
    }
    if supports_effort(model):
        params["output_config"] = {"effort": EFFORT.get(purpose, "low")}
    return params


def _get_client(anthropic_module: Any, key: str) -> Any:
    with _CLIENT_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = anthropic_module.Anthropic(api_key=key) if key else anthropic_module.Anthropic()
            _CLIENTS[key] = client
        return client


def last_usage() -> dict[str, Any]:
    return dict(_LAST_USAGE)


def _read_response(response: Any, model: str) -> str:
    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details is not None else None
        why = f" ({category})" if category else ""
        return f"ERROR: Claude declined this request{why}. Rephrase it, or try another model with /model <name>."
    text = "".join(
        str(getattr(block, "text", "") or "")
        for block in list(getattr(response, "content", None) or [])
        if getattr(block, "type", "") == "text"
    ).strip()
    usage = getattr(response, "usage", None)
    _LAST_USAGE.clear()
    _LAST_USAGE.update(
        {
            "model": str(getattr(response, "model", "") or model),
            "input_tokens": getattr(usage, "input_tokens", None),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }
    )
    if text:
        return text
    if getattr(response, "stop_reason", None) == "max_tokens":
        return "ERROR: Claude used its whole token allowance before answering. Ask a narrower question."
    return "ERROR: Claude returned an empty answer. Try again."


def call_anthropic(
    prompt: str,
    *,
    system: str,
    max_tokens: int,
    purpose: str = "chat",
    history: Optional[list[dict]] = None,
    model: Optional[str] = None,
    client: Any = None,
) -> str:
    """One answer from Claude, or a one-line ``ERROR: ...`` saying what to do."""
    try:
        import anthropic
    except ImportError:
        return INSTALL_HINT

    model = str(model or DEFAULT_ANTHROPIC_MODEL).strip()
    key, _source = get_api_key("anthropic")
    if client is None:
        client = _get_client(anthropic, key)
    seconds, retries = REQUEST_LIMITS.get(purpose, REQUEST_LIMITS["chat"])
    api = client.with_options(timeout=seconds, max_retries=retries)
    params = build_params(prompt, system=system, model=model, max_tokens=max_tokens, purpose=purpose, history=history)
    started = time.monotonic()
    try:
        if supports_fallbacks(model):
            response = api.beta.messages.create(**params, betas=[FALLBACK_BETA], fallbacks="default")
        else:
            response = api.messages.create(**params)
    except anthropic.AuthenticationError:
        if not key:
            return NO_KEY_HINT
        return "ERROR: Anthropic did not accept the API key (401). Save a new one with ./nqt-connect ai setup."
    except anthropic.PermissionDeniedError:
        return f"ERROR: This Anthropic key may not use {model} (403). Try /model {CHEAP_ANTHROPIC_MODEL} or check the key in the Claude Console."
    except anthropic.NotFoundError:
        return f"ERROR: Anthropic has no model \"{model}\". Choose another with /model <name>, e.g. /model {DEFAULT_ANTHROPIC_MODEL}."
    except anthropic.RateLimitError:
        return "ERROR: Anthropic rate limit reached (429). Wait a minute and try again."
    except anthropic.BadRequestError as exc:
        detail = " ".join(str(getattr(exc, "message", "") or exc).split())[:200]
        return f"ERROR: Anthropic refused the request (400): {detail}"
    except anthropic.APIStatusError as exc:
        status = int(getattr(exc, "status_code", 0) or 0)
        if status >= 500:
            return f"ERROR: Anthropic is busy or having problems ({status}). Try again shortly."
        detail = " ".join(str(getattr(exc, "message", "") or exc).split())[:200]
        return f"ERROR: Anthropic refused the request ({status}): {detail}"
    except anthropic.APITimeoutError:
        return f"ERROR: Claude did not answer within {int(seconds)}s. Try again, or ask a shorter question."
    except anthropic.APIConnectionError:
        return "ERROR: Could not reach the Anthropic API. Check your internet connection."
    except anthropic.AnthropicError as exc:  # e.g. no credentials found at all
        if not key:
            return NO_KEY_HINT
        return f"ERROR: Claude API call failed: {' '.join(str(exc).split())[:200]}"
    text = _read_response(response, model)
    _LAST_USAGE["seconds"] = round(time.monotonic() - started, 2)
    return text
