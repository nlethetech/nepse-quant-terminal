import sqlite3

from apps.tui import dashboard_tui
import pandas as pd
from rich.text import Text


def _seed_market_quotes(db_path, rows):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE market_quotes (
            raw_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            security_id TEXT,
            security_name TEXT,
            last_traded_price REAL,
            close_price REAL,
            previous_close REAL,
            percentage_change REAL,
            total_trade_quantity REAL,
            source TEXT NOT NULL,
            fetched_at_utc TEXT NOT NULL,
            PRIMARY KEY (raw_id, symbol)
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO market_quotes (
            raw_id, symbol, security_id, security_name, last_traded_price, close_price,
            previous_close, percentage_change, total_trade_quantity, source, fetched_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def test_load_intraday_ohlcv_builds_session_bars(tmp_path, monkeypatch):
    db_path = tmp_path / "quotes.db"
    _seed_market_quotes(
        db_path,
        [
            (1, "NABIL", None, None, 100.0, 100.0, None, None, 100.0, "test", "2026-04-06T04:15:00+00:00"),
            (2, "NABIL", None, None, 105.0, 105.0, None, None, 130.0, "test", "2026-04-06T04:17:00+00:00"),
            (3, "NABIL", None, None, 103.0, 103.0, None, None, 160.0, "test", "2026-04-06T04:21:00+00:00"),
        ],
    )
    monkeypatch.setattr(dashboard_tui, "_db", lambda: sqlite3.connect(str(db_path)))

    bars, session_date, snapshot_count = dashboard_tui._load_intraday_ohlcv(
        "NABIL",
        preferred_session_date="2026-04-06",
    )

    assert session_date == "2026-04-06"
    assert snapshot_count == 3
    assert len(bars) == 2
    assert bars.iloc[0]["date"].strftime("%H:%M") == "10:00"
    assert bars.iloc[0]["open"] == 100.0
    assert bars.iloc[0]["high"] == 105.0
    assert bars.iloc[0]["low"] == 100.0
    assert bars.iloc[0]["close"] == 105.0
    assert bars.iloc[0]["volume"] == 30.0
    assert bars.iloc[1]["date"].strftime("%H:%M") == "10:05"
    assert bars.iloc[1]["close"] == 103.0
    assert bars.iloc[1]["volume"] == 30.0


def test_load_intraday_ohlcv_falls_back_to_latest_available_session(tmp_path, monkeypatch):
    db_path = tmp_path / "quotes.db"
    _seed_market_quotes(
        db_path,
        [
            (1, "NABIL", None, None, 98.0, 98.0, None, None, 50.0, "test", "2026-04-04T18:30:00+00:00"),
            (2, "NABIL", None, None, 101.0, 101.0, None, None, 80.0, "test", "2026-04-05T04:20:00+00:00"),
        ],
    )
    monkeypatch.setattr(dashboard_tui, "_db", lambda: sqlite3.connect(str(db_path)))

    bars, session_date, snapshot_count = dashboard_tui._load_intraday_ohlcv(
        "NABIL",
        preferred_session_date="2026-04-06",
    )

    assert session_date == "2026-04-05"
    assert snapshot_count == 2
    assert len(bars) == 2


def test_refresh_active_tab_view_only_updates_visible_tab():
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    calls = []

    app.active_tab = "watchlist"
    app.trade_mode = "live"
    app.tms_service = object()
    app.lookup_sym = ""

    app._populate_market = lambda: calls.append("market")
    app._populate_portfolio_and_risk = lambda: calls.append("portfolio")
    app._populate_trades_full = lambda: calls.append("trades")
    app._populate_calendar = lambda: calls.append("calendar")
    app._populate_lookup = lambda: calls.append("lookup")
    app._populate_news = lambda: calls.append("news")
    app._populate_agent_tab = lambda: calls.append("agents")
    app._populate_orders_tab = lambda: calls.append("orders")
    app._populate_watchlist = lambda: calls.append("watchlist")
    app._refresh_watchlist_live = lambda force=False: calls.append(f"watchlist_sync:{force}")
    app._populate_screener = lambda: calls.append("screener")
    app._populate_kalimati = lambda: calls.append("kalimati")

    dashboard_tui.NepseDashboard._refresh_active_tab_view(app, force_watchlist_sync=True)

    assert calls == ["watchlist", "watchlist_sync:True"]


def test_paper_filled_orders_for_day_filters_history_rows():
    rows = dashboard_tui._paper_filled_orders_for_day(
        [
            {"status": "FILLED", "symbol": "NABIL", "created_at": "2026-04-09 11:00:01", "updated_at": "2026-04-09 11:00:05"},
            {"status": "CANCELLED", "symbol": "SBL", "created_at": "2026-04-09 11:05:00", "updated_at": "2026-04-09 11:05:02"},
            {"status": "FILLED", "symbol": "SHIVM", "created_at": "2026-04-08 14:00:00", "updated_at": "2026-04-08 14:00:03"},
        ],
        "2026-04-09",
    )

    assert [row["symbol"] for row in rows] == ["NABIL"]


def test_submit_paper_order_rejects_same_day_buy_after_sell(monkeypatch):
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    app._paper_orders = []
    app._paper_order_history = [
        {
            "symbol": "AAA",
            "action": "SELL",
            "status": "FILLED",
            "day": "2026-04-09",
            "created_at": "2026-04-09 09:31:44",
        }
    ]
    app._load_paper_orders = lambda: None
    app._save_paper_orders = lambda: None
    app._populate_orders_tab = lambda: None

    monkeypatch.setattr(dashboard_tui, "load_port", lambda: pd.DataFrame())
    monkeypatch.setattr("backend.trading.live_trader.now_nst", lambda: dashboard_tui.datetime(2026, 4, 9, 10, 0, 0))

    msg = dashboard_tui.NepseDashboard._submit_paper_order(app, "BUY", "AAA", 100, 100.0, 2.0)

    assert "same-day rule" in msg.lower()
    assert app._paper_orders == []


def test_submit_paper_order_rejects_same_day_sell_after_buy(monkeypatch):
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    app._paper_orders = []
    app._paper_order_history = [
        {
            "symbol": "AAA",
            "action": "BUY",
            "status": "FILLED",
            "day": "2026-04-09",
            "created_at": "2026-04-09 09:31:44",
        }
    ]
    app._load_paper_orders = lambda: None
    app._save_paper_orders = lambda: None
    app._populate_orders_tab = lambda: None

    monkeypatch.setattr(
        dashboard_tui,
        "load_port",
        lambda: pd.DataFrame(
            [
                {
                    "Symbol": "AAA",
                    "Entry_Date": "2026-04-09",
                    "Quantity": 100,
                    "Buy_Price": 100.3,
                }
            ]
        ),
    )
    monkeypatch.setattr("backend.trading.live_trader.now_nst", lambda: dashboard_tui.datetime(2026, 4, 9, 10, 0, 0))

    msg = dashboard_tui.NepseDashboard._submit_paper_order(app, "SELL", "AAA", 100, 114.7, 2.0)

    assert "same-day rule" in msg.lower()
    assert app._paper_orders == []


def test_match_paper_orders_sets_status_when_same_day_rule_cancels(monkeypatch):
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    app._paper_orders = [
        {
            "id": "ord1",
            "action": "SELL",
            "symbol": "AAA",
            "qty": 100,
            "price": 114.7,
            "slippage_pct": 2.0,
            "status": "OPEN",
            "filled_qty": 0,
            "fill_price": 0.0,
            "trigger_price": 114.7,
            "created_at": "2026-04-09 10:00:00",
            "updated_at": "2026-04-09 10:00:00",
            "day": "2026-04-09",
            "source": "dashboard_tui",
            "reason": "",
        }
    ]
    app._paper_order_history = [
        {
            "id": "hist1",
            "action": "BUY",
            "symbol": "AAA",
            "qty": 100,
            "status": "FILLED",
            "created_at": "2026-04-09 09:31:44",
            "updated_at": "2026-04-09 09:31:44",
            "day": "2026-04-09",
        }
    ]
    app._paper_trades_today = []
    app.md = type("MD", (), {"quotes": pd.DataFrame([{"symbol": "AAA", "close": 114.7}]), "gainers": pd.DataFrame(), "losers": pd.DataFrame()})()
    app._append_activity = lambda msg: events.append(("activity", msg))
    app._set_status = lambda msg: events.append(("status", msg))
    app._save_paper_orders = lambda: None
    app._populate_orders_tab = lambda: None
    app._populate_portfolio_and_risk = lambda: None
    app._populate_trades_full = lambda: None

    events = []
    monkeypatch.setattr(dashboard_tui, "load_port", lambda: pd.DataFrame([{"Symbol": "AAA", "Entry_Date": "2026-04-09"}]))
    monkeypatch.setattr("backend.trading.live_trader.now_nst", lambda: dashboard_tui.datetime(2026, 4, 9, 10, 1, 0))

    dashboard_tui.NepseDashboard._match_paper_orders(app)

    assert any(kind == "status" and "same-day rule" in msg.lower() for kind, msg in events)
    assert app._paper_orders == []
    assert any(str(row.get("status")) == "CANCELLED" for row in app._paper_order_history)


def test_set_status_colors_rejections_red():
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    captured = {}

    class DummyStatus:
        def update(self, value):
            captured["value"] = value

    app.query_one = lambda selector, cls=None: DummyStatus()

    dashboard_tui.NepseDashboard._set_status(app, "Rejected: NEPSE same-day rule blocks selling AAA on the same day as a buy.")

    value = captured["value"]
    assert isinstance(value, Text)
    assert any(span.style == dashboard_tui.LOSS_HI for span in value.spans)


def test_display_live_override_keeps_paper_mode_labels(monkeypatch):
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    app.trade_mode = "paper"
    app._trading_engine = None

    monkeypatch.setenv("NEPSE_TUI_SCREENSHOT_LIVE", "1")

    assert dashboard_tui.NepseDashboard._display_live_badge(app) is False
    assert dashboard_tui.NepseDashboard._display_mode_label(app) == "PAPER AUTO"
    assert "PAPER" in dashboard_tui.NepseDashboard._display_nav_mode_tag(app)


def test_populate_trades_full_scales_fractional_pnl_pct_to_percent(monkeypatch):
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    app.trade_mode = "paper"
    app._trading_engine = None
    app.tms_service = None

    class DummyTable:
        def __init__(self):
            self.rows = []
        def clear(self, columns=False):
            self.rows = []
        def add_column(self, *args, **kwargs):
            return None
        def add_row(self, *cells):
            self.rows.append(cells)

    class DummyStatic:
        def update(self, value):
            return None

    trades_table = DummyTable()
    title = DummyStatic()

    def query_one(selector, cls=None):
        if selector == "#dt-trades-full":
            return trades_table
        if selector == "#trades-title":
            return title
        raise AssertionError(f"Unexpected selector {selector}")

    app.query_one = query_one

    monkeypatch.setattr(
        dashboard_tui,
        "_load_trade_log",
        lambda: pd.DataFrame(
            [
                {
                    "Date": "2026-04-09",
                    "Action": "SELL",
                    "Symbol": "SBL",
                    "Shares": 375,
                    "Price": 399.7,
                    "Fees": 646.0,
                    "PnL": 6224.84,
                    "PnL_Pct": 0.0435,
                    "Reason": "holding_period",
                }
            ]
        ),
    )

    dashboard_tui.NepseDashboard._populate_trades_full(app)

    rtn_cell = trades_table.rows[0][7]
    assert isinstance(rtn_cell, Text)
    assert rtn_cell.plain == "+4.35%"


def test_resolve_sell_qty_accepts_all_and_rejects_overflow():
    holdings = {"NABIL": 120, "SBL": 55}

    assert dashboard_tui._resolve_sell_qty("NABIL", "all", holdings) == 120
    assert dashboard_tui._resolve_sell_qty("SBL", "25", holdings) == 25

    try:
        dashboard_tui._resolve_sell_qty("SBL", "90", holdings)
    except ValueError as exc:
        assert "holding 55" in str(exc)
    else:
        raise AssertionError("Expected overflow sell quantity to fail")


def test_compute_portfolio_stats_includes_daily_change(monkeypatch):
    class FakeMD:
        def __init__(self):
            self.quotes = pd.DataFrame(
                [
                    {
                        "symbol": "SHIVM",
                        "ltp": 654.0,
                        "prev_close": 646.0,
                        "pc": 1.24,
                        "vol": 1000,
                        "ts": "2026-04-06T05:24:04+00:00",
                    }
                ]
            )
            self.nepse = pd.DataFrame()

        def ltps(self):
            return {"SHIVM": 654.0}

    monkeypatch.setattr(
        dashboard_tui,
        "load_port",
        lambda: pd.DataFrame(
            [
                {
                    "Symbol": "SHIVM",
                    "Quantity": 211,
                    "Buy_Price": 676.0,
                    "Total_Cost_Basis": 142636.0,
                    "Entry_Date": "2026-04-01",
                    "Signal_Type": "fundamental",
                }
            ]
        ),
    )
    monkeypatch.setattr(dashboard_tui, "_load_nav_log", lambda: pd.DataFrame())
    monkeypatch.setattr(dashboard_tui, "_load_trade_log", lambda: pd.DataFrame())
    monkeypatch.setattr(dashboard_tui, "_load_manual_paper_cash", lambda total_cost, nav_log=None: 857364.0)

    stats = dashboard_tui._compute_portfolio_stats(FakeMD())

    assert round(stats["day_pnl"], 2) == 1688.0
    assert round(stats["cash"], 2) == 857364.0
    assert round(stats["day_ret"], 4) == round((1688.0 / (857364.0 + 646.0 * 211)) * 100, 4)
    assert round(stats["positions"][0]["day_pnl"], 2) == 1688.0
    assert round(stats["positions"][0]["day_ret"], 2) == 1.24


def test_compute_portfolio_stats_exposes_gross_return_above_net_when_fees_paid(monkeypatch):
    class FakeMD:
        def __init__(self):
            self.quotes = pd.DataFrame(
                [
                    {
                        "symbol": "SHIVM",
                        "ltp": 654.0,
                        "prev_close": 646.0,
                        "pc": 1.24,
                        "vol": 1000,
                        "ts": "2026-04-06T05:24:04+00:00",
                    }
                ]
            )
            self.nepse = pd.DataFrame()

        def ltps(self):
            return {"SHIVM": 654.0}

    monkeypatch.setattr(
        dashboard_tui,
        "load_port",
        lambda: pd.DataFrame(
            [
                {
                    "Symbol": "SHIVM",
                    "Quantity": 211,
                    "Buy_Price": 676.0,
                    "Buy_Fees": 536.0,
                    "Total_Cost_Basis": 142636.0,
                    "Entry_Date": "2026-04-01",
                    "Signal_Type": "fundamental",
                }
            ]
        ),
    )
    monkeypatch.setattr(dashboard_tui, "_load_nav_log", lambda: pd.DataFrame())
    monkeypatch.setattr(
        dashboard_tui,
        "_load_trade_log",
        lambda: pd.DataFrame(
            [
                {"Date": "2026-04-01", "Action": "BUY", "Symbol": "SHIVM", "Shares": 211, "Price": 676.0, "Fees": 536.0, "Reason": "fundamental", "PnL": 0.0, "PnL_Pct": 0.0}
            ]
        ),
    )
    monkeypatch.setattr(dashboard_tui, "_load_manual_paper_cash", lambda total_cost, nav_log=None: 857364.0)

    stats = dashboard_tui._compute_portfolio_stats(FakeMD())

    assert round(stats["fees_paid"], 2) == 536.0
    assert stats["gross_return"] > stats["total_return"]
    assert round(stats["gross_nav"] - stats["nav"], 2) == 536.0


def test_normalize_import_portfolio_accepts_generic_columns():
    df = pd.DataFrame(
        [
            {
                "Ticker": "nabil",
                "Shares": 20,
                "Avg Price": 550.5,
                "Date": "2026-04-01",
                "Signal": "fundamental",
                "LTP": 565.0,
            }
        ]
    )

    normalized = dashboard_tui._normalize_import_portfolio(df)

    assert list(normalized.columns) == dashboard_tui.PORTFOLIO_COLS
    assert normalized.iloc[0]["Symbol"] == "NABIL"
    assert normalized.iloc[0]["Quantity"] == 20
    assert normalized.iloc[0]["Buy_Price"] == 550.5
    assert normalized.iloc[0]["Last_LTP"] == 565.0


def test_merge_watchlist_entries_prioritizes_holdings_without_duplicates():
    merged = dashboard_tui._merge_watchlist_entries(
        [dashboard_tui._stock_watchlist_entry("NABIL"), dashboard_tui._stock_watchlist_entry("SBL")],
        [dashboard_tui._stock_watchlist_entry("SBL"), dashboard_tui._stock_watchlist_entry("NICA")],
    )

    assert [row["symbol"] for row in merged] == ["NABIL", "SBL", "NICA"]


def test_build_account_seed_state_uses_mark_value_for_cash():
    portfolio = pd.DataFrame(
        [
            {"Symbol": "NABIL", "Quantity": 10, "Buy_Price": 500.0, "Last_LTP": 550.0},
            {"Symbol": "SBL", "Quantity": 20, "Buy_Price": 300.0, "Last_LTP": 290.0},
        ]
    )

    state, nav_log = dashboard_tui._build_account_seed_state(portfolio, 20_000.0)

    assert state["cash"] == 8_700.0
    assert state["daily_start_nav"] == 20_000.0
    assert float(nav_log.iloc[0]["Positions_Value"]) == 11_300.0
    assert float(nav_log.iloc[0]["NAV"]) == 20_000.0


def test_apply_indicator_history_change_computes_delta_and_updates_history():
    history = {"gold_per_tola": {"value": 290000.0, "timestamp": "2026-04-08T00:00:00"}}

    change, change_pct = dashboard_tui._apply_indicator_history_change(
        history,
        key="gold_per_tola",
        value=294700.0,
        timestamp="2026-04-09T00:00:00",
    )

    assert round(change, 2) == 4700.0
    assert round(change_pct, 4) == round((4700.0 / 290000.0) * 100, 4)
    assert history["gold_per_tola"]["value"] == 294700.0


def test_account_hotkeys_route_existing_actions_contextually():
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    events = []

    app.active_tab = "account"
    app._account_activate_selected = lambda: events.append("activate") or "ok"
    app._account_sync_watchlist = lambda: events.append("watchlist") or "ok"
    app._set_status = lambda msg: events.append(f"status:{msg}")
    app.action_tab = lambda name: events.append(f"tab:{name}")
    app._run_agent_analysis = lambda force=False: events.append(f"agent:{force}")

    dashboard_tui.NepseDashboard.action_run_agent(app)
    dashboard_tui.NepseDashboard.action_tf(app, "W")

    assert events == ["activate", "status:ok", "watchlist", "status:ok"]


def test_agent_chat_stop_command_invokes_stop_handler():
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    events = []

    app._stop_active_agent_chat = lambda announce=True: events.append(("stop", announce)) or True
    app._set_status = lambda message: events.append(("status", message))

    handled = dashboard_tui.NepseDashboard._handle_agent_chat_command(app, "/stop")

    assert handled is True
    assert events == [("stop", True)]


def test_load_agent_runtime_state_clears_mismatched_account_analysis(monkeypatch):
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    app._current_account_id = "account_1"
    app._agent_show_archived = False
    app._agent_visible_since = 0.0

    monkeypatch.setattr(
        dashboard_tui,
        "load_agent_analysis",
        lambda: {"stocks": [{"symbol": "NABIL"}], "account_id": "account_2"},
    )
    monkeypatch.setattr(dashboard_tui, "load_agent_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(dashboard_tui, "load_agent_archive_history", lambda *args, **kwargs: [])

    dashboard_tui.NepseDashboard._load_agent_runtime_state(app)

    assert app._agent_analysis == {}


def test_headline_fallback_from_url_rejects_generic_merolagani_detail_url():
    assert dashboard_tui._headline_fallback_from_url(
        "https://merolagani.com/NewsDetail.aspx?newsID=124946"
    ) == ""


def test_headline_fallback_from_url_keeps_readable_slug_urls():
    assert dashboard_tui._headline_fallback_from_url(
        "https://example.com/news/nepse-turnover-jumps-after-policy-shift"
    ) == "Nepse Turnover Jumps After Policy Shift"


def test_signal_code_maps_long_signal_names_to_short_labels():
    assert dashboard_tui._signal_code("fundamental") == "F"
    assert dashboard_tui._signal_code("mean_reversion") == "MR"
    assert dashboard_tui._signal_code("xsec_momentum") == "XS"
    assert dashboard_tui._signal_code("accumulation") == "A"


def test_ensure_lookup_history_backfills_sparse_daily_history(tmp_path, monkeypatch):
    db_path = tmp_path / "quotes.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE stock_prices (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO stock_prices (symbol, date, open, high, low, close, volume)
        VALUES ('RURU', '2026-04-07', 654.0, 654.0, 654.0, 654.0, 15000.0)
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(dashboard_tui, "_db", lambda: sqlite3.connect(str(db_path)))
    monkeypatch.setenv("NEPSE_DB_FILE", str(db_path))
    monkeypatch.setattr(
        "backend.quant_pro.vendor_api.fetch_ohlcv_chunk",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {"Date": "2026-04-03", "Open": 640.0, "High": 645.0, "Low": 638.0, "Close": 644.0, "Volume": 1100.0},
                {"Date": "2026-04-06", "Open": 648.0, "High": 652.0, "Low": 647.0, "Close": 650.0, "Volume": 1200.0},
                {"Date": "2026-04-07", "Open": 654.0, "High": 654.0, "Low": 654.0, "Close": 654.0, "Volume": 15000.0},
            ]
        ),
    )

    loaded = dashboard_tui._ensure_lookup_history("RURU", min_sessions=2, history_days=10)

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM stock_prices WHERE symbol='RURU'").fetchone()[0]
    conn.close()

    assert loaded >= 3
    assert count == 3


def test_build_agent_auto_order_spec_selects_only_super_signal():
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)

    class FakeMD:
        def ltps(self):
            return {"NABIL": 500.0, "SBL": 600.0}

    app.trade_mode = "paper"
    app.md = FakeMD()
    app._stats = {
        "cash": 500_000.0,
        "nav": 1_000_000.0,
        "positions": [{"sym": "SHIVM"}],
    }
    app._paper_orders = []
    app._last_agent_auto_order_key = None

    spec = dashboard_tui.NepseDashboard._build_agent_auto_order_spec(
        app,
        {
            "trade_today": True,
            "regime": "bull",
            "context_date": "2026-04-06",
            "timestamp": 12345,
            "stocks": [
                {
                    "symbol": "SBL",
                    "auto_entry_candidate": False,
                    "signal_score": 0.50,
                    "conviction": 0.60,
                    "last_price": 600.0,
                },
                {
                    "symbol": "NABIL",
                    "auto_entry_candidate": True,
                    "signal_score": 1.08,
                    "conviction": 0.92,
                    "last_price": 500.0,
                },
            ],
        },
    )

    assert spec is not None
    assert spec["symbol"] == "NABIL"
    assert spec["quantity"] == 400
    assert spec["order_key"] == "2026-04-06:NABIL:12345"


def test_split_agent_messages_by_cutoff_hides_pre_session_history():
    visible, hidden = dashboard_tui._split_agent_messages_by_cutoff(
        [
            {"role": "AGENT", "message": "old", "ts": 10},
            {"role": "YOU", "message": "new", "ts": 25},
        ],
        20,
    )

    assert [item["message"] for item in visible] == ["new"]
    assert [item["message"] for item in hidden] == ["old"]


def test_agent_focus_row_updates_focus_panel():
    app = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    app._agent_analysis = {
        "stocks": [
            {
                "symbol": "SAHAS",
                "verdict": "APPROVE",
                "action_label": "BUY",
                "signal_score": 0.90,
                "conviction": 0.90,
                "what_matters": "Strong recent profit growth anchors the thesis.",
                "bull_case": "Profit growth is accelerating.",
                "bear_case": "Execution still depends on open-session liquidity.",
            }
        ]
    }

    class FakeStatic:
        def __init__(self):
            self.value = None

        def update(self, value):
            self.value = value

    title = FakeStatic()
    detail = FakeStatic()
    widgets = {
        "#agent-detail-title": title,
        "#agent-detail": detail,
    }
    app.query_one = lambda selector, *_args, **_kwargs: widgets[selector]

    dashboard_tui.NepseDashboard._show_agent_focus_row(app, 0)

    assert title.value == "FOCUS · SAHAS"
    assert "SAHAS" in detail.value.plain
    assert "BUY" in detail.value.plain
