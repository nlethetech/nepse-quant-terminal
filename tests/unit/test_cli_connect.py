"""The website connection: key handling, safety rules and display."""

from __future__ import annotations

import json
import stat

import pytest

from apps.cli import connect


@pytest.fixture(autouse=True)
def config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NEPSE_QUANT_API_KEY", raising=False)
    monkeypatch.delenv("NQT_BASE_URL", raising=False)


def test_saved_key_is_private():
    path = connect.save_key("nqk_abc")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert connect.load_key() == "nqk_abc"


def test_key_is_never_sent_over_plain_http(monkeypatch):
    monkeypatch.setenv("NQT_BASE_URL", "http://example.com")
    with pytest.raises(connect.ConnectError):
        connect.base_url()


def test_local_testing_address_is_allowed(monkeypatch):
    monkeypatch.setenv("NQT_BASE_URL", "http://127.0.0.1:8200/")
    assert connect.base_url() == "http://127.0.0.1:8200"


def test_default_site():
    assert connect.base_url() == "https://nepsequant.com"


def test_bad_key_is_refused_before_any_request(monkeypatch):
    monkeypatch.setattr(connect, "request_json", lambda *_: pytest.fail("no request expected"))
    with pytest.raises(connect.ConnectError):
        connect.login("not-a-key")


def test_login_saves_only_after_the_site_accepts(monkeypatch):
    monkeypatch.setattr(connect, "request_json", lambda *_: (_ for _ in ()).throw(connect.ConnectError("no")))
    with pytest.raises(connect.ConnectError):
        connect.login("nqk_bad")
    assert not connect.credentials_path().exists()


def test_not_connected_points_to_the_site(capsys):
    assert connect.main(["paper"]) == 1
    assert "nepsequant.com" in capsys.readouterr().err


def test_portfolio_display_and_json(monkeypatch, capsys):
    connect.save_key("nqk_x")
    data = {
        "book": "paper", "value": 1000.0, "cash": 100.0,
        "positions": [{"symbol": "NABIL", "quantity": 2, "avg_cost": 500.0, "last": 450.0, "value": 900.0, "return_pct": -10.0}],
        "open_orders": [{"action": "SELL", "quantity": 2, "symbol": "NABIL", "limit_price": 600.0}],
    }
    monkeypatch.setattr(connect, "request_json", lambda *_: data)
    assert connect.main(["paper"]) == 0
    out = capsys.readouterr().out
    assert "virtual money" in out and "NABIL" in out and "-10.00%" in out and "nepsequant.com" in out
    assert connect.main(["paper", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["value"] == 1000.0
