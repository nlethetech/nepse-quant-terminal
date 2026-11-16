"""Safety guards around keys: tests, where keys may go, and what AI sees.

* tests run in a throwaway home folder with no keys and no internet
  (tests/conftest.py);
* a NEPSE Quant key is only ever sent to nepsequant.com (or this computer,
  for testing), never over plain http, never through a redirect;
* a key never appears in an error message;
* anything that looks like a key is removed before an AI model sees a prompt.
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import threading
from pathlib import Path

import pytest

from apps.cli import connect

KEY = "nqk_" + "A" * 40


# ── the test sandbox itself ─────────────────────────────────────────────────


def test_tests_cannot_reach_the_internet():
    with pytest.raises(RuntimeError, match="Tests may not use the network"):
        socket.create_connection(("93.184.216.34", 443), timeout=1)
    with pytest.raises(RuntimeError, match="Tests may not use the network"):
        connect.urllib.request.urlopen("https://nepsequant.com/api/healthz", timeout=1)


def test_tests_run_in_a_throwaway_home_without_keys():
    home = Path.home()
    assert "nqt-test-home" in str(home) or "pytest" in str(home)
    assert str(connect.credentials_path()).startswith(os.environ["XDG_CONFIG_HOME"])
    for name in ("NEPSE_QUANT_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NQT_BASE_URL"):
        assert name not in os.environ
    with pytest.raises(connect.ConnectError):
        connect.load_key()


# ── where a NEPSE Quant key may be sent ─────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.com",
        "https://nepsequant.com.evil.example",
        "https://nepsequant.com@evil.example",
        "https://evil.example/nepsequant.com",
        "https://nepsequant.com:8443",
        "http://nepsequant.com",
        "ftp://nepsequant.com",
    ],
)
def test_key_is_only_sent_to_nepsequant(monkeypatch, url):
    monkeypatch.setenv("NQT_BASE_URL", url)
    with pytest.raises(connect.ConnectError):
        connect.base_url()


@pytest.mark.parametrize("url", ["https://nepsequant.com", "https://www.nepsequant.com/", "http://127.0.0.1:8299", "http://localhost:8200"])
def test_allowed_destinations(monkeypatch, url):
    monkeypatch.setenv("NQT_BASE_URL", url)
    assert connect.base_url() == url.rstrip("/")


class _Recorder(http.server.BaseHTTPRequestHandler):
    seen: list = []
    reply = (200, {}, b"{}")

    def do_GET(self):  # noqa: N802
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        status, headers, body = type(self).reply
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _serve(reply):
    handler = type("H", (_Recorder,), {"seen": [], "reply": reply})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler


def test_a_redirect_is_never_followed_with_the_key(monkeypatch):
    target, target_seen = _serve((200, {}, b'{"ok": true}'))
    hop, _ = _serve((302, {"Location": f"http://127.0.0.1:{target.server_port}/steal"}, b""))
    try:
        monkeypatch.setenv("NQT_BASE_URL", f"http://127.0.0.1:{hop.server_port}")
        with pytest.raises(connect.ConnectError, match="redirect"):
            connect.request_json("/api/keys/verify", KEY)
        assert target_seen.seen == []  # the key never reached the redirect target
    finally:
        target.shutdown()
        hop.shutdown()


def test_the_key_never_appears_in_an_error_message(monkeypatch):
    body = json.dumps({"detail": f"Bad key {KEY} for this route"}).encode()
    server, _ = _serve((403, {"Content-Type": "application/json"}, body))
    try:
        monkeypatch.setenv("NQT_BASE_URL", f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(connect.ConnectError) as err:
            connect.request_json("/api/cli/paper/account", KEY)
        assert KEY not in str(err.value) and "[key removed]" in str(err.value)
    finally:
        server.shutdown()


def test_openai_requests_do_not_follow_redirects(monkeypatch):
    from backend.agents import llm_openai

    target, target_seen = _serve((200, {}, b"{}"))
    hop, _ = _serve((307, {"Location": f"http://127.0.0.1:{target.server_port}/v1/chat/completions"}, b""))

    class _Post(_Recorder):
        pass

    try:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-" + "B" * 40)
        monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{hop.server_port}/v1")
        out = llm_openai.call_openai("hi", system="s", max_tokens=10, purpose="chat")
        assert out.startswith("ERROR:") and target_seen.seen == []
        assert "sk-proj-" not in out
    finally:
        target.shutdown()
        hop.shutdown()


# ── what an AI model may see ────────────────────────────────────────────────


def test_keys_are_removed_before_any_model_sees_a_prompt(monkeypatch):
    from backend.agents import agent_analyst

    seen = {}

    def fake_openai(prompt, *, system, max_tokens, purpose, history):
        seen.update(prompt=prompt, system=system, history=history)
        return "ok"

    monkeypatch.setattr(agent_analyst, "_agent_backend", lambda: "openai")
    monkeypatch.setattr(agent_analyst, "_call_openai", fake_openai)
    secrets = ["sk-ant-" + "C" * 30, "sk-proj-" + "D" * 30, "nqk_" + "E" * 30, "nq_live_" + "F" * 30]
    agent_analyst._call_primary_agent(
        f"news mentions {secrets[0]} and {secrets[2]}",
        system=f"system {secrets[1]}",
        purpose="chat",
        history=[{"role": "user", "content": f"my key is {secrets[3]}"}],
    )
    blob = json.dumps(seen)
    assert not any(s in blob for s in secrets)
    assert blob.count("[key removed]") == 4
