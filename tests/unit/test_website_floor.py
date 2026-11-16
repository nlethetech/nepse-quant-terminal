"""Broker panels on a terminal linked to nepsequant.com.

The website is faked (a request function answering like GET /api/cli/floorsheet),
so these pin the terminal's side: what it asks for, how often, what it shows,
and that a linked terminal never reads the local scraped database for them.
"""

from __future__ import annotations

import gzip
import io
import json

import pytest

from apps.cli import connect
from apps.tui import dashboard_tui
from apps.tui.io import floor_metrics as fm
from apps.tui.io import website_floor as wf
from apps.tui.render import broker_floor as bf

DAY = "2026-11-13"


def payload(version: str = f"{DAY}:6:stamp", day: str = DAY, complete=True) -> dict:
    # broker 5 and 6 trade back and forth (circular); broker 7 trades with itself
    trades = [(0, 5, 6, 300), (0, 6, 5, 300), (0, 5, 6, 200), (0, 6, 5, 200), (1, 7, 7, 900), (1, 1, 7, 100)]
    return {
        "date": day, "version": version, "available_dates": [day], "complete": complete, "notice": "n",
        "symbols": ["CIRC", "SELF"], "brokers": [{"id": c, "name": None} for c in (1, 5, 6, 7)],
        "symbol_idx": [t[0] for t in trades], "buyer": [t[1] for t in trades],
        "seller": [t[2] for t in trades], "qty": [t[3] for t in trades],
        "rate": [100.0] * len(trades), "count": len(trades),
    }


class FakeSite:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, path, key, **kw):
        self.calls.append((path, kw))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _floor(*replies):
    site, clock = FakeSite(replies), Clock()
    return wf.WebsiteFloor("nqk_test", request=site, clock=clock, engine="python"), site, clock


# ── the client ───────────────────────────────────────────────────────────────
def test_first_refresh_downloads_and_computes():
    floor, site, _ = _floor(payload())
    assert floor.view is None and floor.due()
    assert floor.refresh() is True
    path, kw = site.calls[0]
    assert path == "/api/cli/floorsheet" and kw["compressed"] is True
    view = floor.view
    assert view.day.date == DAY and view.day.engine == "python"
    assert {r["symbol"] for r in view.signals} == {"CIRC", "SELF"}
    assert view.top_brokers[0]["broker_code"] in (1, 7)


def test_at_most_every_few_minutes_and_a_held_day_is_not_downloaded_again():
    unchanged = {"date": DAY, "version": f"{DAY}:6:stamp", "unchanged": True}
    floor, site, clock = _floor(payload(), unchanged)
    floor.refresh()
    first = floor.view
    assert not floor.due() and floor.refresh() is False and len(site.calls) == 1
    clock.t += wf.TTL_S - 1
    assert not floor.due()
    clock.t += 2
    assert floor.refresh() is False  # nothing new
    assert site.calls[1][0] == f"/api/cli/floorsheet?have={DAY}%3A6%3Astamp"
    assert floor.view is first


def test_a_new_day_replaces_the_old_one():
    floor, site, clock = _floor(payload(), payload(version="2026-11-14:6:s", day="2026-11-14"))
    floor.refresh()
    clock.t += wf.TTL_S + 1
    assert floor.refresh() is True and floor.view.day.date == "2026-11-14"


def test_a_failed_refresh_keeps_the_last_day_and_retries_sooner():
    floor, site, clock = _floor(payload(), connect.ConnectError("Could not reach https://nepsequant.com."), payload())
    floor.refresh()
    clock.t += wf.TTL_S + 1
    assert floor.refresh() is True and "Could not reach" in floor.error
    assert floor.view.day.date == DAY  # last good day still shown
    clock.t += wf.RETRY_S + 1
    assert floor.due() and floor.refresh() is True and floor.error == ""


def test_a_malformed_reply_is_reported_not_raised():
    floor, _, _ = _floor({"date": DAY, "symbols": ["A"], "symbol_idx": [0, 0], "buyer": [1], "seller": [1], "qty": [1]})
    assert floor.refresh() is True and floor.view is None and "Could not read the floorsheet" in floor.error


def test_linked_only_with_a_key():
    assert wf.WebsiteFloor.from_book(None) is None

    class Book:
        _key = "nqk_x"

    assert isinstance(wf.WebsiteFloor.from_book(Book()), wf.WebsiteFloor)


# ── gzip on the wire ─────────────────────────────────────────────────────────
class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, encoding: str = ""):
        super().__init__(body)
        self.headers = {"Content-Encoding": encoding} if encoding else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_compressed_requests_ask_for_gzip_and_unpack_it(monkeypatch):
    seen = []
    body = json.dumps({"count": 3}).encode()

    def fake_open(req, timeout):
        seen.append(dict(req.header_items()))
        return FakeResponse(gzip.compress(body), "gzip") if req.get_header("Accept-encoding") else FakeResponse(body)

    monkeypatch.setattr(connect, "open_keyed", fake_open)
    assert connect.request_json("/api/cli/floorsheet", "nqk_x", compressed=True) == {"count": 3}
    assert seen[-1].get("Accept-encoding") == "gzip"
    assert connect.request_json("/api/cli/account", "nqk_x") == {"count": 3}
    assert "Accept-encoding" not in seen[-1]  # other calls unchanged


def test_a_damaged_gzip_reply_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(connect, "open_keyed", lambda req, timeout: FakeResponse(b"\x1f\x8bnot gzip", "gzip"))
    with pytest.raises(connect.ConnectError):
        connect.request_json("/api/cli/floorsheet", "nqk_x", compressed=True)


# ── what the panels show ─────────────────────────────────────────────────────
def test_panel_title_shows_the_day_the_source_and_the_engine():
    floor, _, _ = _floor(payload(complete=False))
    floor.refresh()
    title, rows, top_cols, top_rows = bf.website_floor_panels(floor.view, floor.error)
    assert title.startswith(f"BROKER FLOOR  {DAY}  |") and "nepsequant.com · python" in title
    assert "still being captured" in title
    assert "floorsheet_ingestion" not in title and "scraper" not in title.lower()
    labels = {str(r[0]): str(r[1]) for r in rows}
    assert "CIRCULAR" in labels["CIRC"] and "SELF" in labels["SELF"]
    assert top_cols == bf.TOP_BROKER_COLS and len(top_rows) == 4
    assert not any("BUY" in str(c) for r in rows for c in r[1:2])  # no new buy wording


def test_panel_before_and_after_a_failure():
    title, rows, cols, _ = bf.website_floor_panels(None, "")
    assert "loading the floorsheet from nepsequant.com" in title and cols == []
    title, rows, _, _ = bf.website_floor_panels(None, "Could not reach https://nepsequant.com.")
    assert "not loaded: Could not reach" in title
    assert all("scraper" not in str(c).lower() and "ingestion" not in str(c).lower() for c in rows[0])
    empty = wf.FloorView(fm.FloorDay(date=None, engine="python"))
    assert "no floorsheet on nepsequant.com yet" in bf.website_floor_panels(empty, "")[0]


def test_local_and_linked_rows_read_the_same():
    record = {"symbol": "ABC", "hhi_buy": 0.4, "hhi_sell": 0.1, "circular_score": 0.0, "pump_score": 0.0,
              "self_trade_pct": 0.0, "smart_money_score": 0.5, "n_trades": 30, "micro_score": None}
    (row,) = bf.floor_signal_rows([record])
    assert [str(c) for c in row] == ["ABC", "[+] SMART $   ", " 0.40", " 0.10", "   0%", "   0%", "  0.375", "    30"]


class FakeTable:
    def __init__(self):
        self.columns, self.rows = [], []

    def clear(self, columns=False):
        self.rows = []
        if columns:
            self.columns = []

    def add_column(self, label, key=None, width=None):
        self.columns.append(label)

    def add_row(self, *row):
        self.rows.append([str(c) for c in row])


class FakeStatic:
    def __init__(self):
        self.text = ""

    def update(self, text):
        self.text = str(text)


def test_a_linked_terminal_fills_both_panels_from_the_website(monkeypatch):
    floor, _, _ = _floor(payload())
    floor.refresh()
    widgets = {
        "#dt-broker-floor": FakeTable(), "#dt-broker-top": FakeTable(),
        "#broker-floor-title": FakeStatic(), "#broker-top-title": FakeStatic(),
    }
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    app._website_floor = floor
    app._broker_floor_filter = "all"
    app.query_one = lambda selector, *_: widgets[selector]
    # a stale payload from before the floorsheet arrived
    app._signals_workspace_cache_payload = {"broker_cols": [], "broker_rows": [], "broker_title": "BROKER FLOOR  —  loading"}
    dashboard_tui.NepseDashboard._apply_website_floor(app)
    assert widgets["#broker-floor-title"].text.startswith(f"BROKER FLOOR  {DAY}  |")
    assert widgets["#broker-top-title"].text == f"TOP BROKERS BY VOLUME  {DAY}"
    assert {r[0] for r in widgets["#dt-broker-floor"].rows} == {"CIRC", "SELF"}
    assert len(widgets["#dt-broker-top"].rows) == 4
    assert widgets["#dt-broker-floor"].columns[0] == "SYMBOL"


def test_the_refresh_worker_only_starts_when_due(monkeypatch):
    floor, site, _ = _floor(payload())
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    app._website_floor = floor
    started = []
    app._refresh_website_floor = lambda force=False: started.append(force)
    dashboard_tui.NepseDashboard._maybe_refresh_website_floor(app)
    floor.refresh()
    dashboard_tui.NepseDashboard._maybe_refresh_website_floor(app)
    assert started == [False]
    app._website_floor = None
    dashboard_tui.NepseDashboard._maybe_refresh_website_floor(app)  # unlinked: nothing
    assert started == [False]
