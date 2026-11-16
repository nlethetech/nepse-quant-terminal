"""One paper book: the terminal uses the nepsequant.com paper book when linked.

The website is faked here (a request function that answers like the
/api/cli/paper routes), so these tests pin the terminal's side of the
contract: what it reads, what it sends, and that a linked terminal never
touches the local paper book or trades automatically.
"""

from __future__ import annotations

import pandas as pd
import pytest

from apps.cli import connect
from apps.tui import dashboard_tui
from apps.tui.io import stats as stats_io
from apps.tui.io import website_book as wb
from apps.tui.state.mixins import order_book as order_book_mixin
from backend.quant_pro.control_plane.command_service import build_tui_control_plane

ACCOUNT = {
    "book": "paper",
    "value": 1_005_000.0,
    "cash": 950_000.0,
    "starting_cash": 1_000_000.0,
    "invested": 50_000.0,
    "market_value": 55_000.0,
    "unrealized_pnl": 5_000.0,
    "realized_pnl": 0.0,
    "day_pnl": 0.0,
    "day_pnl_pct": 0.0,
    "positions": [
        {"symbol": "NABIL", "quantity": 100, "avg_cost": 500.0, "last": 550.0, "value": 55_000.0,
         "return_pct": 10.0, "cost_basis": 50_000.0, "unrealized_pnl": 5_000.0, "acquired_date": "2026-11-10"},
    ],
    "market": {"open": True, "phase": "OPEN"},
    "can_trade": True,
}
ORDERS = {
    "open": [{"order_id": "abc123def456", "action": "BUY", "symbol": "SBL", "order_type": "limit", "quantity": 20,
              "filled_qty": 0, "limit_price": 400.0, "fill_price": 0.0, "status": "OPEN",
              "created_at": "2026-11-09 11:05:00", "updated_at": "2026-11-09 11:05:00", "reason": ""}],
    "history": [
        {"order_id": "new", "action": "BUY", "symbol": "NABIL", "order_type": "market", "quantity": 100, "filled_qty": 100,
         "limit_price": 500.0, "fill_price": 500.0, "status": "FILLED", "created_at": "2026-11-10 11:00:00",
         "updated_at": "2026-11-10 11:00:01", "reason": ""},
        {"order_id": "old", "action": "BUY", "symbol": "HBL", "order_type": "limit", "quantity": 5, "filled_qty": 0,
         "limit_price": 200.0, "fill_price": 0.0, "status": "CANCELLED", "created_at": "2026-11-03 11:00:00",
         "updated_at": "2026-11-03 12:00:00", "reason": ""},
    ],
}
TRADES = {"trades": [
    {"date": "2026-11-10", "action": "BUY", "symbol": "NABIL", "shares": 100, "price": 500.0, "fees": 250.0,
     "pnl": 0.0, "pnl_pct": 0.0, "reason": "manual"},
    {"date": "2026-11-01", "action": "SELL", "symbol": "HBL", "shares": 10, "price": 220.0, "fees": 30.0,
     "pnl": 150.0, "pnl_pct": 7.5, "reason": "manual"},
]}
NAV = {"nav": [{"date": "2026-11-09", "nav": 1_000_000.0}, {"date": "2026-11-10", "nav": 1_005_000.0}]}


class FakeSite:
    """Answers like the website's /api/cli/paper routes and records what was sent."""

    def __init__(self, *, can_trade: bool = True, fail: bool = False):
        self.account = {**ACCOUNT, "can_trade": can_trade}
        self.fail = fail
        self.sent: list[tuple[str, str, dict | None]] = []
        self.order_reply = {"ok": True, "status": "FILLED", "order_id": "x1", "fill_price": 551.0, "resting": False,
                            "message": "filled"}

    def __call__(self, path, key, *, method="GET", body=None, timeout=30):
        assert key == "nqk_test"
        if self.fail:
            raise connect.ConnectError("Could not reach https://nepsequant.com. Check your connection and try again.")
        self.sent.append((method, path, body))
        if method == "GET":
            return {
                "/api/cli/paper/account": self.account,
                "/api/cli/paper/orders": ORDERS,
                "/api/cli/paper/trades?limit=500": TRADES,
                "/api/cli/paper/nav": NAV,
            }[path]
        if method == "POST" and path == "/api/cli/paper/orders":
            return self.order_reply
        if method == "DELETE":
            return {"ok": True, "message": "cancelled"}
        raise AssertionError((method, path))


def _book(**kw) -> tuple[wb.WebsiteBook, FakeSite]:
    site = FakeSite(**kw)
    book = wb.WebsiteBook("nqk_test", request=site)
    assert book.refresh() is True
    return book, site


# ── linking ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NEPSE_QUANT_API_KEY", raising=False)
    monkeypatch.delenv("NQT_SHARED_BOOK", raising=False)


def test_no_saved_key_means_the_local_book(config_home):
    assert wb.WebsiteBook.from_saved() is None


def test_a_saved_key_links_the_website_book(config_home):
    connect.save_key("nqk_saved")
    book = wb.WebsiteBook.from_saved()
    assert book is not None and book._key == "nqk_saved"


def test_the_shared_book_can_be_switched_off(config_home, monkeypatch):
    connect.save_key("nqk_saved")
    monkeypatch.setenv("NQT_SHARED_BOOK", "0")
    assert wb.WebsiteBook.from_saved() is None


# ── reading ──────────────────────────────────────────────────────────────────


def test_portfolio_frame_carries_the_website_numbers():
    book, _ = _book()
    port = book.portfolio_frame()
    assert list(port.columns) == wb.PORT_COLS
    row = port.iloc[0]
    assert row.Symbol == "NABIL" and row.Quantity == 100 and row.Buy_Price == 500.0
    assert row.Total_Cost_Basis == 50_000.0 and row.Last_LTP == 550.0 and row.Entry_Date == "2026-11-10"
    assert book.cash() == 950_000.0 and book.starting_cash() == 1_000_000.0 and book.can_trade


def test_trade_log_is_oldest_first_with_fractional_return():
    book, _ = _book()
    tl = book.trade_log_frame()
    assert list(tl.columns) == wb.TRADE_LOG_COLS
    assert list(tl.Symbol) == ["HBL", "NABIL"]
    assert tl.iloc[0].PnL_Pct == pytest.approx(0.075)


def test_orders_in_the_terminal_shape():
    book, _ = _book()
    open_rows, history = book.order_records()
    o = open_rows[0]
    assert o["id"] == "abc123def456" and o["qty"] == 20 and o["price"] == 400.0 and o["status"] == "OPEN"
    assert o["day"] == "2026-11-09" and o["slippage_pct"] == 0.0
    assert [h["id"] for h in history] == ["old", "new"]  # oldest first, like the local history file
    assert history[1]["filled_qty"] == 100 and history[1]["fill_price"] == 500.0


def test_stats_match_the_website_value():
    book, _ = _book()

    class _MD:
        quotes = pd.DataFrame()

        @staticmethod
        def ltps():
            return {"NABIL": 999.0}  # the terminal's own price must not override the book's

    stats = stats_io._stats_from_frames(
        _MD(), book.portfolio_frame(), book.nav_frame(), book.trade_log_frame(),
        account_capital=book.starting_cash(), cash=book.cash(), prefer_book_price=True,
    )
    assert stats["nav"] == pytest.approx(ACCOUNT["value"])
    assert stats["cash"] == pytest.approx(950_000.0)
    assert stats["positions"][0]["sym"] == "NABIL" and stats["positions"][0]["qty"] == 100
    assert stats["total_return"] == pytest.approx(0.5)
    assert stats["realized"] == pytest.approx(150.0)


def test_a_failed_refresh_keeps_the_last_book_and_says_why():
    book, site = _book()
    site.fail = True
    assert book.refresh() is False
    assert "Could not reach" in book.error
    assert book.cash() == 950_000.0  # last good snapshot still on screen


# ── trading ──────────────────────────────────────────────────────────────────


def test_no_price_is_a_market_order():
    book, site = _book()
    result = book.submit("buy", "nabil", 10)
    assert result["ok"] is True
    method, path, body = [s for s in site.sent if s[0] == "POST"][0]
    assert body == {"action": "BUY", "symbol": "NABIL", "quantity": 10, "order_type": "market"}


def test_a_price_is_a_limit_order_on_the_tick():
    book, site = _book()
    book.submit("SELL", "NABIL", 10, limit_price=551.04)
    body = [s for s in site.sent if s[0] == "POST"][0][2]
    assert body["order_type"] == "limit" and body["limit_price"] == 551.0


def test_a_refused_order_comes_back_with_the_websites_reason():
    book, site = _book()

    def refuse(path, key, *, method="GET", body=None, timeout=30):
        if method == "POST":
            raise connect.ConnectError("This key can only read.")
        return site(path, key, method=method, body=body)

    book._request = refuse
    result = book.submit("BUY", "NABIL", 10)
    assert result == {"ok": False, "status": "REFUSED", "message": "This key can only read.", "transient": False}
    assert "Rejected" in wb.describe_result(result, "BUY", "NABIL", 10, None)


def test_cancel_all_cancels_each_open_order():
    book, site = _book()
    done, problems = book.cancel_all()
    assert done == 1 and problems == []
    assert ("DELETE", "/api/cli/paper/orders/abc123def456", None) in site.sent


def test_status_lines():
    filled = wb.describe_result({"ok": True, "status": "FILLED", "fill_price": 551.0}, "BUY", "NABIL", 10, None)
    assert filled.startswith("Filled: BUY NABIL x10 at market @ 551.0")
    resting = wb.describe_result({"ok": True, "status": "OPEN", "resting": True, "order_id": "abcdef123"}, "BUY", "NABIL", 10, 500)
    assert "OPEN on your nepsequant.com paper book" in resting and "limit 500.0" in resting


# ── the terminal on the shared book ──────────────────────────────────────────


def _app(book) -> dashboard_tui.NepseDashboard:
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    app._website_book = book
    app._paper_orders = []
    app._paper_order_history = []
    app._paper_trades_today = []
    return app


def test_linked_terminal_reads_orders_from_the_website(monkeypatch):
    book, _ = _book()
    app = _app(book)
    monkeypatch.setattr(order_book_mixin, "_account_dir", lambda *_: pytest.fail("local book read"))
    dashboard_tui.NepseDashboard._load_paper_orders(app)
    assert [o["id"] for o in app._paper_orders] == ["abc123def456"]
    assert [o["id"] for o in app._paper_order_history] == ["old", "new"]


def test_linked_terminal_sends_orders_to_the_website(monkeypatch):
    book, _ = _book()
    app = _app(book)
    sent = []
    app._send_website_order = lambda *a: sent.append(a)
    monkeypatch.setattr(order_book_mixin, "PaperExecutionService", lambda *a, **k: pytest.fail("local book written"))

    msg = dashboard_tui.NepseDashboard._submit_paper_order(app, "BUY", "NABIL", 10, 550.0, 2.0, market=True)
    assert sent[-1] == ("BUY", "NABIL", 10, None) and "at market" in msg
    msg = dashboard_tui.NepseDashboard._submit_paper_order(app, "SELL", "NABIL", 10, 560.0, 2.0)
    assert sent[-1] == ("SELL", "NABIL", 10, 560.0) and "limit 560.0" in msg


def test_read_only_key_explains_how_to_trade():
    book, _ = _book(can_trade=False)
    app = _app(book)
    app._send_website_order = lambda *a: pytest.fail("a read-only key must not send")
    msg = dashboard_tui.NepseDashboard._submit_paper_order(app, "BUY", "NABIL", 10, 550.0, market=True)
    assert "Paper trading" in msg and "nqt-connect login" in msg


def test_website_applies_the_same_day_rule_not_the_local_file(monkeypatch):
    book, _ = _book()
    app = _app(book)
    monkeypatch.setattr(order_book_mixin, "load_port", lambda: pytest.fail("local book read"))
    assert dashboard_tui.NepseDashboard._paper_same_day_trade_block(app, "SELL", "NABIL") is None


def test_linked_terminal_never_fills_locally(monkeypatch):
    book, _ = _book()
    app = _app(book)
    monkeypatch.setattr(order_book_mixin, "PaperExecutionService", lambda *a, **k: pytest.fail("local fill"))
    monkeypatch.setattr(order_book_mixin, "exec_buy", lambda *a, **k: pytest.fail("local fill"))
    dashboard_tui.NepseDashboard._match_paper_orders(app)
    assert app._paper_orders and app._paper_orders[0]["id"] == "abc123def456"


def test_cancels_go_to_the_website():
    book, _ = _book()
    app = _app(book)
    asked = []
    app._cancel_website_orders = lambda oid: asked.append(oid)
    assert "Cancelling order abc123de" in dashboard_tui.NepseDashboard._cancel_paper_order(app, "abc123def456")
    assert "Cancelling all" in dashboard_tui.NepseDashboard._cancel_all_paper_orders(app)
    assert asked == ["abc123def456", None]


def test_automatic_orders_never_reach_the_shared_book():
    book, _ = _book()
    app = _app(book)
    app._submit_paper_order = lambda *a, **k: pytest.fail("automatic order sent")
    plane = build_tui_control_plane(app)
    submitter = plane.paper_submitter if hasattr(plane, "paper_submitter") else plane._paper_submitter
    ok, msg, _ = submitter("BUY", "NABIL", 10, 550.0)
    assert ok is False and "Automatic orders are off" in msg


def test_local_book_untouched_when_not_linked(tmp_path):
    """Not linked: the account files still drive the stats exactly as before."""
    (tmp_path / "paper_portfolio.csv").write_text(
        "Entry_Date,Symbol,Quantity,Buy_Price,Total_Cost_Basis,Last_LTP,Signal_Type\n2026-11-10,NABIL,10,500,5025,510,\n"
    )
    (tmp_path / "paper_state.json").write_text('{"cash": 994975.0, "initial_capital": 1000000}')

    class _MD:
        quotes = pd.DataFrame()

        @staticmethod
        def ltps():
            return {"NABIL": 520.0}

    stats = stats_io._compute_account_portfolio_stats(_MD(), tmp_path)
    assert stats["cash"] == pytest.approx(994_975.0)
    assert stats["positions"][0]["cur"] == 520.0  # local book: the terminal's own live price
    assert stats["nav"] == pytest.approx(994_975.0 + 5_200.0)

    (tmp_path / "paper_state.json").unlink()
    stats = stats_io._compute_account_portfolio_stats(_MD(), tmp_path)
    assert stats["cash"] == pytest.approx(stats_io._account_initial_capital_from_files(tmp_path) - 5_025.0)


def test_account_panel_describes_the_shared_book(monkeypatch):
    book, _ = _book()
    app = _app(book)
    app._stats = {"nav": 1_005_000.0, "cash": 950_000.0}
    from apps.tui.state.mixins import tab_refresh as tab_refresh_mixin
    monkeypatch.setattr(tab_refresh_mixin, "load_port", lambda: pytest.fail("local book read"))
    snap = dashboard_tui.NepseDashboard._profile_runtime_snapshot(app)
    assert (snap["holdings"], snap["trades"], snap["nav_rows"]) == (1, 2, 2)
    assert snap["cash"] == 950_000.0 and snap["nav"] == 1_005_000.0
