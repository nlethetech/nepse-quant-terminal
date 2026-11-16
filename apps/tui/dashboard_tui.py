#!/usr/bin/env python3
"""
NEPSE Bloomberg-Style Terminal Dashboard — Textual TUI

Run:  python3 dashboard_tui.py
Keys: 1-9 tabs │ / search │ L lookup │ R refresh │ Q quit
"""
from __future__ import annotations

import copy
import json as _json
import logging
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests as _requests
from rich.markup import escape as _escape_markup
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import ContentSwitcher, DataTable, Input, Static, Button, Label, OptionList
from textual.widgets.option_list import Option


from apps.tui.widgets.signal_defs import (  # re-export
    _silence_tui_noisy_loggers,
    _sig_btn_id,
    _sig_type_from_id,
    _SIGNAL_DEFS,
    _SIG_ID_PREFIX,
)


_silence_tui_noisy_loggers()





# ── Data layer + theme ────────────────────────────────────────────────────────
from backend.quant_pro.dashboard_data import (
    MD, _db, load_port, save_port, exec_buy, exec_sell,
)
from apps.tui.theme import (
    _vol, _pct, _npr,
    AMBER, WHITE, DIM, LABEL, GAIN_HI, GAIN, LOSS_HI, LOSS, CYAN, YELLOW, PURPLE, BLUE,
)
from configs.long_term import LONG_TERM_CONFIG

from apps.tui.render import charts as _charts
from apps.tui.render.cells import (  # re-export
    _chg_text,
    _sym_text,
    _dim_text,
    _vol_text,
    _price_text,
    _pnl_color,
    _npr_k,
    _format_compact_npr,
    _contains_non_ascii,
    _truncate_text,
    _news_display_headline,
)
from apps.tui.render.text_blocks import (  # re-export
    _render_stock_report,
    _render_lookup_intelligence,
    _headline_fallback_from_url,
)
from apps.tui.render.charts import _render_candlestick_chart, _render_sparkline  # re-export

from apps.tui.io.watchlist_io import (  # re-export
    _stock_watchlist_entry,
    _normalize_watchlist_entry,
    _watchlist_entry_key,
    _dedupe_watchlist_entries,
)
from apps.tui.io.csv_import import (  # re-export
    _normalize_import_portfolio,
    _merge_watchlist_entries,
    _dedupe_symbol_rows,
    _coerce_dragdrop_path,
    _normalize_frame_columns,
    _pick_column,
    _is_meroshare_csv,
    _normalize_meroshare_portfolio,
    _normalize_import_trade_log,
    _normalize_import_nav_log,
    _build_holdings_watchlist_entries,
    _positive_float,
)
from apps.tui.io.agent_io import _split_agent_messages_by_cutoff  # re-export
from apps.tui.io.orders_io import (  # re-export
    _paper_filled_orders_for_day,
    _build_sell_holdings_map,
    _format_sell_holdings_summary,
    _resolve_sell_qty,
)
from apps.tui.io import stats as _stats_io
from apps.tui.io.stats import (  # re-export
    _compute_portfolio_stats,
    _compute_account_portfolio_stats,
    _get_regime,
    _signal_label,
    _signal_code,
)
from apps.tui.io import intraday as _intraday
from apps.tui.io.intraday import (  # re-export
    _load_intraday_ohlcv,
    _ensure_lookup_history,
    _resample_ohlcv,
    _nst_today_str,
    _format_nst_hm,
)
from apps.tui.io.fetchers import (  # re-export
    _fetch_nrb_forex_rates,
    _fetch_gold_silver_prices,
    _fetch_yahoo_futures_price,
    _fetch_noc_fuel_prices,
    _extract_decimal_price,
    _translate_nepali,
    _translate_batch,
)
from apps.tui.io import persistence as _persistence
from apps.tui.io.persistence import (  # re-export
    INITIAL_CAPITAL,
    PROJECT_ROOT,
    RUNTIME_DIR,
    TRADING_RUNTIME_DIR,
    HEDGE_TRADE_LOG_FILE,
    WATCHLIST_FILE,
    PAPER_NAV_LOG_FILE,
    PAPER_TRADE_LOG_FILE,
    PAPER_STATE_FILE,
    PAPER_PORTFOLIO_FILE,
    TUI_PAPER_PORTFOLIO_FILE,
    TUI_PAPER_NAV_LOG_FILE,
    TUI_PAPER_TRADE_LOG_FILE,
    TUI_PAPER_STATE_FILE,
    PAPER_PROFILE_FILE,
    PAPER_IMPORT_BACKUP_DIR,
    PAPER_ACCOUNTS_DIR,
    PAPER_ACCOUNTS_REGISTRY_FILE,
    MACRO_INDICATOR_HISTORY_FILE,
    TUI_PAPER_ORDERS_FILE,
    TUI_PAPER_ORDER_HISTORY_FILE,
    ACTIVE_ACCOUNT_FILES,
    DEFAULT_WATCHLIST,
    _load_watchlist,
    _save_watchlist,
    _ensure_csv_file,
    _ensure_paper_runtime_files,
    _lock_file_exclusive,
    _unlock_file,
    _write_json_locked,
    _load_macro_indicator_history,
    _save_macro_indicator_history,
    _apply_indicator_history_change,
    _load_accounts_registry,
    _save_accounts_registry,
    _account_dir,
    _copy_file_if_exists,
    _blank_account_files,
    _next_account_id,
    _portfolio_mark_value,
    _build_account_seed_state,
    _load_profile_config,
    _save_profile_config,
    _bootstrap_paper_accounts,
    _load_manual_paper_cash,
    _account_initial_capital_from_files,
    _tms_health_flag,
    _load_cached_tms_bundle,
    _merge_tms_bundle_with_cache,
    _load_nav_log,
    _load_trade_log,
    _load_hedge_trade_log,
    _save_hedge_trade_log,
)

from backend.quant_pro.paths import (
    ensure_dir,
)
from backend.agents.agent_analyst import (
    analyze as agent_analyze,
    append_external_agent_chat_message,
    build_algo_shortlist_snapshot,
    publish_agent_signal_snapshot,
    check_trade_approval,
    load_agent_analysis,
    load_agent_archive_history,
    load_agent_history,
)
from backend.agents.runtime_config import (
    ACTIVE_AGENT_FILE,
    list_agent_backends,
    load_active_agent_config,
    set_active_agent,
)
from backend.quant_pro.stock_report import build_stock_report
from backend.trading.tui_trading_engine import TUITradingEngine
from backend.trading.paper_execution import PaperExecutionService
from backend.market.kalimati_market import init_kalimati_db, refresh_kalimati, get_kalimati_display_rows
from backend.trading.live_trader import (
    NAV_LOG_COLS,
    PORTFOLIO_COLS,
    TRADE_LOG_COLS,
    calculate_cash_from_trade_log,
    load_runtime_state,
    save_runtime_state,
)
from backend.trading import strategy_registry

MAX_ACCOUNTS = 5
AGENT_ARCHIVE_RENDER_LIMIT = 60
AGENT_CHAT_TIMEOUT_SECS = 90

STRATEGY_DISPLAY_NAMES = {
    "default_c5": "C5 Baseline",
    "temp_forward_winner": "TFW",
    "strat_3_p2r25": "P2R25",
    "strat_4_p3r32": "P3R32",
    "strat_6_r83": "R83",
    "strat_7_h53": "H53",
    
    
    
}

_TMS_AUDIT_SNAPSHOT_MAP = {
    "health": "tms_health",
    "account": "tms_account",
    "watchlist": "tms_watchlist",
    "funds": "tms_funds",
    "holdings": "tms_holdings",
    "orders_daily": "tms_orders_daily",
    "orders_historic": "tms_orders_historic",
    "trades_daily": "tms_trades_daily",
    "trades_historic": "tms_trades_historic",
}


# ── Ticker scroll speed ─────────────────────────────────────────────────────
TICKER_SPEED = 0.15  # seconds between scroll steps

# ── OSINT API ────────────────────────────────────────────────────────────────
# Optional, self-hosted OSINT enrichment service. Disabled by default in the
# public build — set NEPSE_OSINT_BASE to your own endpoint to enable it.
OSINT_BASE = os.environ.get("NEPSE_OSINT_BASE", "").rstrip("/")
OSINT_TIMEOUT = 8

def _fetch_osint_stories(limit: int = 40) -> list[dict]:
    """Fetch latest stories from Nepal OSINT API (disabled unless configured)."""
    if not OSINT_BASE:
        return []
    try:
        r = _requests.get(f"{OSINT_BASE}/analytics/consolidated-stories",
                          params={"limit": limit}, timeout=OSINT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

def _fetch_osint_brief() -> dict:
    """Fetch latest intelligence brief (disabled unless configured)."""
    if not OSINT_BASE:
        return {}
    try:
        r = _requests.get(f"{OSINT_BASE}/briefs/latest", timeout=OSINT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

# _render_candlestick_chart resolves the resampler from its own module namespace.
_charts._resample_ohlcv = _resample_ohlcv


# Bind the persistence helpers onto the stats module so its moved compute
# functions resolve them in their own namespace.
_stats_io._load_nav_log = _persistence._load_nav_log
_stats_io._load_trade_log = _persistence._load_trade_log
_stats_io._load_manual_paper_cash = _persistence._load_manual_paper_cash
_stats_io._account_initial_capital_from_files = _persistence._account_initial_capital_from_files


# ═══════════════════════════════════════════════════════════════════════════════
# WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

from apps.tui.widgets.market_panel import MarketPanel  # re-export


# ═══════════════════════════════════════════════════════════════════════════════
# MODAL DIALOGS
# ═══════════════════════════════════════════════════════════════════════════════

from apps.tui.screens.dialog import ModalDialog  # re-export


# ═══════════════════════════════════════════════════════════════════════════════
# MODE SELECT SCREEN — shown on startup
# ═══════════════════════════════════════════════════════════════════════════════

from apps.tui.screens.mode_select import ModeSelectScreen  # re-export
from apps.tui.screens.tms_login import TMSLoginScreen, load_tms_settings  # re-export
from apps.tui.screens.command_palette import CommandPalette  # re-export
from apps.tui.screens.watchlist_add import WatchlistAddScreen  # re-export
from apps.tui.screens.lookup import LookupScreen  # re-export

from apps.tui.state.mixins.lifecycle import LifecycleMixin  # re-export
from apps.tui.state.mixins.header_status import HeaderStatusMixin  # re-export
from apps.tui.state.mixins.trading_engine import TradingEngineMixin  # re-export
from apps.tui.state.mixins.order_book import OrderBookMixin  # re-export
from apps.tui.state.mixins.tab_refresh import TabRefreshMixin  # re-export
from apps.tui.state.mixins.agent_chat import AgentChatMixin  # re-export
from apps.tui.state.mixins.accounts_strategies import AccountsStrategiesMixin  # re-export
from apps.tui.state.mixins.events_actions import EventsActionsMixin  # re-export
from apps.tui.state.mixins.live_tms import LiveTMSMixin  # re-export
from apps.tui.state.mixins.website_strategies_ui import WebsiteStrategiesMixin  # re-export


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

TAB_NAMES = {
    "market": "1", "portfolio": "2", "signals": "3",
    "lookup": "4", "agents": "5", "orders": "6",
    "watchlist": "7", "kalimati": "8", "account": "9", "strategies": "0",
}

TAB_LABELS = {
    "market": "MARKET",
    "portfolio": "PORTFOLIO",
    "signals": "SIGNALS",
    "lookup": "LOOKUP",
    "agents": "AGENTS",
    "orders": "ORDERS",
    "watchlist": "WATCHLIST",
    "kalimati": "RATES & COMMODITIES",
    "account": "ACCOUNT",
    "strategies": "STRATEGIES",
}


class NepseDashboard(
    LifecycleMixin,
    HeaderStatusMixin,
    TradingEngineMixin,
    OrderBookMixin,
    TabRefreshMixin,
    AgentChatMixin,
    AccountsStrategiesMixin,
    EventsActionsMixin,
    LiveTMSMixin,
    WebsiteStrategiesMixin,
    App,
):
    CSS_PATH = Path(__file__).with_name("dashboard_tui.tcss")

    BINDINGS = [
        Binding("1", "tab('market')", "Market", show=False),
        Binding("2", "tab('portfolio')", "Portfolio", show=False),
        Binding("3", "tab('signals')", "Signals", show=False),
        Binding("4", "tab('lookup')", "Lookup", show=False),
        Binding("5", "tab('agents')", "Agents", show=False),
        Binding("6", "tab('orders')", "Orders", show=False),
        Binding("7", "tab('watchlist')", "Watchlist", show=False),
        Binding("8", "tab('kalimati')", "Rates & Commodities", show=False),
        Binding("9", "tab('account')", "Account", show=False),
        Binding("0", "tab('strategies')", "Strategies", show=False),
        # B/S hotkeys removed — orders placed via order book only
        Binding("l", "lookup", "Lookup", show=False),
        Binding("a", "run_agent", "Agent", show=False),
        Binding("r", "refresh", "Refresh", show=False),
        Binding("f", "force_signals_reload", "Force Signals", show=False),
        Binding("q", "quit", "Quit", show=False),
        Binding("slash", "command_palette", "GO", show=False),
        Binding("ctrl+p", "command_palette", "GO", show=False),
        Binding("equals_sign", "watchlist_add", "=Watch", show=False),
        Binding("minus", "watchlist_remove", "-Watch", show=False),
        Binding("d", "tf('D')", "Daily", show=False),
        Binding("w", "tf('W')", "Weekly", show=False),
        Binding("m", "tf('M')", "Monthly", show=False),
        Binding("y", "tf('Y')", "Yearly", show=False),
        Binding("i", "tf('I')", "Intraday", show=False),
    ]

    active_tab: str = "market"
    lookup_sym: str = ""
    lookup_tf: str = "D"  # D=Daily, W=Weekly, M=Monthly, I=Intraday
    trade_mode: str = "paper"
    tms_service = None
    _tms_bundle = None   # cached fetch_monitor_bundle result
    _last_tms_watchlist_fetch_at: float = 0.0
    _tms_watchlist_refresh_inflight: bool = False
    _trading_engine: Optional[TUITradingEngine] = None
    # The nepsequant.com paper book, when the terminal is linked (./nqt-connect
    # login): then it is the terminal's ONLY paper book (apps/tui/io/website_book.py).
    _website_book = None
    # Linked: the user's nepsequant.com strategies (apps/tui/io/website_strategies.py)
    # and the auto-trader for the assigned one (apps/tui/io/strategy_autotrader.py).
    _website_strategies = None
    _autotrader = None
    # Linked: the broker panels' floorsheet from nepsequant.com (apps/tui/io/website_floor.py).
    _website_floor = None
    _webstrat_index: int = 0
    _account_engines: dict = {}  # account_id -> TUITradingEngine
    _news_search_query: str = ""  # current news search filter
    _vector_search_results: list = []  # semantic search results from OSINT API
    _order_action: str = "BUY"   # current selected action in order form
    _paper_orders: list = []     # paper mode order book
    _paper_order_history: list = []  # filled/cancelled orders
    _paper_trades_today: list = []   # today's filled trades
    _paper_watchlist: list[dict] = []
    _live_watchlist: list[dict] = []
    _watchlist: list[dict] = []   # user watchlist entries
    _paper_accounts: list[dict] = []
    _current_account_id: str = "account_1"
    _selected_account_id: str = "account_1"
    _account_help_visible: bool = False
    _strategies: list[dict] = []
    _selected_strategy_id: str = "default_c5"
    _strategy_backtest_result: dict = {}
    _strategy_backtest_statuses: dict = {}
    _agent_chat_process: Optional[subprocess.Popen] = None
    _agent_chat_request_id: int = 0
    _agent_chat_stop_requested: bool = False
    _agent_chat_busy: bool = False
    _agent_analysis_running: bool = False
    _agent_brief_running: bool = False
    _agent_brief_skip_until: float = 0.0
    _screener_sort: str = "chg"  # screener sort column
    _screener_filter: str = ""   # screener sector filter
    _refresh_inflight: bool = False

    PAPER_ORDERS_FILE = TUI_PAPER_ORDERS_FILE
    PAPER_ORDER_HISTORY_FILE = TUI_PAPER_ORDER_HISTORY_FILE

    def compose(self) -> ComposeResult:
        with Vertical(id="top-bars"):
            yield Static(id="header-bar")
            yield Static(id="ticker-bar")
            yield Static(id="index-bar")

        with ContentSwitcher(id="content", initial="market"):
            # ── 1 MARKET: 3-panel top + 2-panel bottom ──
            with Vertical(id="market", classes="tab-pane"):
                with Horizontal(id="market-top"):
                    yield MarketPanel(panel_title="1) GAINERS", title_color=GAIN, id="p-gainers")
                    yield MarketPanel(panel_title="2) LOSERS", title_color=LOSS, id="p-losers")
                    yield MarketPanel(panel_title="3) VOLUME LEADERS", title_color=CYAN, id="p-volume")
                with Horizontal(id="market-bot"):
                    yield MarketPanel(panel_title="4) 52-WEEK EXTREMES", title_color=YELLOW, id="p-52wk")
                    yield MarketPanel(panel_title="5) LIVE QUOTES", title_color=PURPLE, id="p-quotes")

            # ── 2 PORTFOLIO: NAV + risk bar + holdings/risk split + trades ──
            with Vertical(id="portfolio", classes="tab-pane"):
                yield Static("", id="nav-summary", classes="panel-title")
                yield Static("", id="risk-summary", classes="panel-title")
                with Horizontal(id="port-split"):
                    with Vertical(id="port-left", classes="full-pane"):
                        yield Static("HOLDINGS", classes="panel-title")
                        yield DataTable(id="dt-portfolio", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="port-right", classes="full-pane"):
                        yield Static("CONCENTRATION & SECTOR", classes="panel-title")
                        yield DataTable(id="dt-concentration", zebra_stripes=True, cursor_type="row")
                        yield Static("WINNERS / LOSERS", classes="panel-title")
                        yield DataTable(id="dt-winloss", zebra_stripes=True, cursor_type="row")
                with Horizontal(id="port-bottom"):
                    with Vertical(id="port-hedge", classes="full-pane"):
                        with Horizontal(id="hedge-header"):
                            yield Static("GOLD / SILVER HEDGE", classes="panel-title", id="hedge-title")
                            yield Button("● HEDGE ON", id="hedge-toggle-btn", classes="hedge-btn-on")
                        yield Static("", id="hedge-info-bar")
                        yield Static("", id="hedge-rec-bar")
                        yield Static("", id="hedge-trade-bar")

            # ── 3 SIGNALS WORKSPACE ──
            with Vertical(id="signals", classes="tab-pane"):
                yield Static("", id="screener-status-bar")
                with Horizontal(id="signals-main"):
                    with Vertical(id="signals-table-pane", classes="full-pane"):
                        with Horizontal(id="signals-header-row"):
                            yield Static("SIGNALS", classes="panel-title", id="signals-table-title")
                            yield Input(value="0.0", placeholder="min score", id="signal-min-score", classes="signal-threshold-input")
                        yield DataTable(id="dt-signals", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="signals-screener-pane", classes="full-pane"):
                        yield Static("ACTIVE STOCKS", classes="panel-title", id="screener-list-title")
                        yield DataTable(id="dt-screener", zebra_stripes=True, cursor_type="row")
                with Horizontal(id="signals-broker"):
                    with Vertical(id="broker-floor-left"):
                        with Horizontal(id="broker-floor-header"):
                            yield Static("BROKER FLOOR SIGNALS", classes="panel-title", id="broker-floor-title")
                            yield Button("ALL",      id="bf-filter-all",   classes="bf-filter-btn bf-active")
                            yield Button("CIRCULAR", id="bf-filter-circ",  classes="bf-filter-btn")
                            yield Button("PUMP",     id="bf-filter-pump",  classes="bf-filter-btn")
                            yield Button("SMART $",  id="bf-filter-smart", classes="bf-filter-btn")
                        yield DataTable(id="dt-broker-floor", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="broker-floor-right"):
                        yield Static("TOP BROKERS", classes="panel-title", id="broker-top-title")
                        yield DataTable(id="dt-broker-top", zebra_stripes=True, cursor_type="row")
                with Horizontal(id="signals-bottom"):
                    with Vertical(id="signals-calendar", classes="full-pane"):
                        yield Static("CORPORATE ACTIONS — Next 30 Days", classes="panel-title")
                        yield DataTable(id="dt-calendar", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="screener-heatmap-pane", classes="full-pane"):
                        yield Static("SECTOR PERFORMANCE", classes="panel-title")
                        yield Static("", id="heatmap-content")

            # ── 4 LOOKUP ──

            with Vertical(id="lookup", classes="tab-pane"):
                yield Static("Press L to look up any stock", classes="panel-title", id="lookup-title")
                yield Static("", id="lookup-header")
                yield Static("", id="lookup-chart")
                with Horizontal(id="lookup-split"):
                    with Vertical(id="lookup-table-pane"):
                        yield Static("OHLCV", classes="panel-title", id="lookup-ohlcv-title")
                        yield DataTable(id="dt-lookup", zebra_stripes=True, cursor_type="row")
                    with VerticalScroll(id="lookup-summary-pane"):
                        yield Static("", id="lookup-stats")
                        yield Static("", id="lookup-report")
                    with VerticalScroll(id="lookup-side-pane"):
                        yield Static("", id="lookup-fin-title")
                        yield DataTable(id="dt-lookup-fin", zebra_stripes=True, cursor_type="row")
                        yield Static("", id="lookup-ca-title")
                        yield DataTable(id="dt-lookup-ca", zebra_stripes=True, cursor_type="row")
                        yield Static("", id="lookup-intel-title")
                        yield Static("", id="lookup-intel")

            # ── 5 AGENTS ──
            with Vertical(id="agents", classes="tab-pane"):
                yield Static("", id="agent-status-bar")
                yield Static("", id="agent-market-view")
                with Horizontal(id="agent-split"):
                    with Vertical(id="agent-left", classes="full-pane"):
                        yield Static("TOP 10 AGENT PICKS", classes="panel-title")
                        yield Static("", id="agent-picks-subtitle")
                        yield DataTable(id="dt-agent-verdicts", zebra_stripes=True, cursor_type="row")
                        with Vertical(id="agent-left-bottom"):
                            yield Static("FOCUS", classes="panel-title", id="agent-detail-title")
                            with VerticalScroll(id="agent-detail-pane"):
                                yield Static("", id="agent-detail")
                    with Vertical(id="agent-right", classes="full-pane"):
                        yield Static("CHAT", classes="panel-title")
                        yield Static("", id="agent-chat-subtitle")
                        yield VerticalScroll(id="agent-chat-scroll")
                        with Vertical(id="agent-right-bottom"):
                            yield Static("COMPOSER", classes="panel-title", id="agent-compose-title")
                            with Vertical(id="agent-chat-footer"):
                                yield Static("", id="agent-chat-hint")
                                yield Input(id="agent-input", placeholder="Ask the agent...  (/agent list, /agent openai, /model <name>, /stop)")

            # ── 6 ORDERS (Order Management) ──
            with Vertical(id="orders", classes="tab-pane"):
                yield Static("", id="order-status-bar")
                with Horizontal(id="order-form-bar"):
                    yield Static("SIDE", classes="order-chip order-chip-neutral")
                    yield Button("BUY", id="order-btn-buy")
                    yield Button("SELL", id="order-btn-sell")
                    yield Static("•", classes="order-sep")
                    yield Static("TICKET", classes="order-chip order-chip-neutral")
                    yield Static("SYM", classes="order-field-label")
                    yield Input(id="order-inp-symbol", placeholder="NABIL")
                    yield Static("QTY", classes="order-field-label")
                    yield Input(id="order-inp-qty", placeholder="10")
                    yield Static("LIMIT", classes="order-field-label")
                    yield Input(id="order-inp-price", placeholder="LTP")
                    yield Static("SLIP%", classes="order-field-label")
                    yield Input(id="order-inp-slippage", placeholder="2.0")
                    yield Static("•", classes="order-sep")
                    yield Static("EXEC", classes="order-chip order-chip-neutral")
                    yield Button("SUBMIT", id="order-btn-submit")
                    yield Button("CANCEL", id="order-btn-cancel-all")
                with Horizontal(id="order-books-split"):
                    with Vertical(id="order-daily-pane", classes="full-pane"):
                        yield Static("DAILY ORDER BOOK", classes="panel-title", id="order-daily-title")
                        yield DataTable(id="dt-orders-daily", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="order-historic-pane", classes="full-pane"):
                        yield Static("HISTORIC ORDER BOOK", classes="panel-title", id="order-historic-title")
                        yield DataTable(id="dt-orders-historic", zebra_stripes=True, cursor_type="row")
                with Vertical(id="order-trades-pane", classes="full-pane"):
                    yield Static("TODAY'S TRADES (FILLED)", classes="panel-title", id="order-trades-title")
                    yield DataTable(id="dt-orders-trades", zebra_stripes=True, cursor_type="row")

            # ── 7 WATCHLIST ──
            with Vertical(id="watchlist", classes="tab-pane"):
                yield Static("", id="wl-status-bar")
                with Horizontal(id="wl-split"):
                    with Vertical(id="wl-main", classes="full-pane"):
                        yield Static("STOCK WATCHLIST", classes="panel-title", id="wl-main-title")
                        yield DataTable(id="dt-watchlist", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="wl-side", classes="full-pane"):
                        with Vertical(id="wl-rates-pane", classes="full-pane"):
                            yield Static("FOREX & MACRO", classes="panel-title", id="wl-rates-title")
                            yield DataTable(id="dt-watchlist-rates", zebra_stripes=True, cursor_type="row")
                        with Vertical(id="wl-commodities-pane", classes="full-pane"):
                            yield Static("COMMODITIES", classes="panel-title", id="wl-commodities-title")
                            yield DataTable(id="dt-watchlist-commodities", zebra_stripes=True, cursor_type="row")

            # ── 8 RATES & COMMODITIES ───────────────────────────────────────
            with Vertical(id="kalimati", classes="tab-pane"):
                yield Static("", id="kalimati-status-bar")
                with Horizontal(id="kalimati-search-bar"):
                    yield Static("SEARCH", id="kalimati-search-label")
                    yield Input(
                        id="kalimati-search-input",
                        placeholder="Search commodities, gold, silver, crude, petrol, forex..."
                    )
                    yield Button("CLEAR", id="kalimati-search-clear")
                with Horizontal(id="kalimati-split"):
                    with Vertical(id="kalimati-left-pane", classes="full-pane"):
                        yield Static("KALIMATI COMMODITIES", classes="panel-title", id="kalimati-left-title")
                        yield Static("", id="kalimati-movers-bar")
                        with Vertical(id="kalimati-main", classes="full-pane"):
                            yield DataTable(id="dt-kalimati", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="kalimati-right-pane", classes="full-pane"):
                        yield Static("GLOBAL RATES & PRICES", classes="panel-title", id="kalimati-right-title")
                        with Vertical(id="kalimati-macro-pane", classes="full-pane"):
                            yield Static("METALS, ENERGY & NOC", classes="panel-title", id="macro-top-title")
                            yield DataTable(id="dt-macro", zebra_stripes=True, cursor_type="row")
                        with Vertical(id="kalimati-forex-pane", classes="full-pane"):
                            yield Static("FOREX RATES", classes="panel-title", id="macro-forex-title")
                            yield DataTable(id="dt-forex", zebra_stripes=True, cursor_type="row")

            with Vertical(id="account", classes="tab-pane"):
                with Horizontal(id="account-split"):
                  with Vertical(id="account-left", classes="full-pane"):
                    with Vertical(id="account-main", classes="full-pane"):
                      with Vertical(id="profile-pane"):
                        yield Static("PAPER ACCOUNT", classes="panel-title")
                        yield Static("", id="profile-summary")
                        yield Static("ACCOUNTS", classes="panel-title")
                        yield DataTable(id="dt-account-list", zebra_stripes=True, cursor_type="row")
                        yield Static("ACCOUNT SETUP", classes="panel-title")
                        yield Static("", id="profile-shortcuts")
                        with Horizontal(id="profile-primary-row"):
                            yield Static("NAME", classes="profile-inline-label")
                            yield Input(id="profile-inp-account-name", placeholder="Account 2", classes="profile-inline-input profile-name-input")
                            yield Static("NAV", classes="profile-inline-label")
                            yield Input(id="profile-inp-target-nav", placeholder="1000000", classes="profile-inline-input profile-nav-input")
                            with Horizontal(id="profile-primary-actions"):
                                yield Button("N NEW", id="profile-btn-create-account", classes="profile-btn profile-btn-primary")
                                yield Button("A ACTIVATE", id="profile-btn-activate-account", classes="profile-btn")
                        with Horizontal(id="profile-seed-row"):
                            yield Static("SEED", classes="profile-inline-label")
                            yield Input(id="profile-inp-portfolio", placeholder="paper_portfolio.csv or MeroShare CSV  —  or press B BROWSE")
                            yield Button("B BROWSE", id="profile-btn-browse-seed", classes="profile-btn profile-btn-browse")
                        with Horizontal(id="profile-actions"):
                            yield Button("W WATCHLIST", id="profile-btn-sync-watchlist", classes="profile-btn profile-action-button")
                            yield Button("V SET NAV", id="profile-btn-set-nav", classes="profile-btn profile-action-button")
                            yield Button("S SNAPSHOT", id="profile-btn-save-account", classes="profile-btn profile-action-button")
                            yield Button("DEL ACCOUNT", id="profile-btn-delete-account", classes="profile-btn profile-action-button profile-btn-delete-account")
                        yield Static("", id="account-help")
                        yield Static(
                            "Create a blank account with target NAV, or press B BROWSE to pick a SEED file. "
                            "Accepts paper_portfolio.csv or MeroShare 'My Shares Values.csv' (auto-detected). "
                            "Selecting and activating an account swaps the full paper runtime.",
                            id="profile-note",
                        )
                  with Vertical(id="account-right", classes="full-pane"):
                      with Vertical(id="port-trades", classes="full-pane"):
                          yield Static("TRADE HISTORY", classes="panel-title", id="trades-title")
                          yield DataTable(id="dt-trades-full", zebra_stripes=True, cursor_type="row")
                      with Vertical(id="port-activity", classes="full-pane"):
                          yield Static("ENGINE LOG", classes="panel-title", id="activity-title")
                          yield VerticalScroll(id="activity-scroll")

            with Vertical(id="strategies", classes="tab-pane"):
                with Vertical(id="strategies-main", classes="full-pane"):
                    yield Static("STRATEGY REGISTRY", classes="panel-title")
                    yield Static("", id="strategy-summary")
                    with Horizontal(id="strategies-split"):
                        with Vertical(id="strategies-left", classes="full-pane"):
                            yield Static("SAVED STRATEGIES", classes="panel-title", id="strategies-left-title")
                            yield DataTable(id="dt-strategy-list", zebra_stripes=True, cursor_type="row")
                            yield Static("", id="strategy-accounts-note")
                        with Vertical(id="strategies-right", classes="full-pane"):
                            yield Static("STRATEGY BUILDER", classes="panel-title", id="strategy-builder-title")
                            # Linked to nepsequant.com: strategies come from the website (no builder)
                            with Vertical(id="webstrat-panel"):
                                yield Static("", id="webstrat-detail")
                                with Horizontal(id="webstrat-actions", classes="strategy-row"):
                                    yield Button("ASSIGN", id="webstrat-btn-assign", classes="profile-btn profile-btn-primary")
                                    yield Button("AUTO OFF", id="webstrat-btn-auto", classes="profile-btn")
                                    yield Button("REFRESH", id="webstrat-btn-refresh", classes="profile-btn")
                                yield Static("AUTO-TRADE LOG", classes="panel-title")
                                yield Static("", id="webstrat-log")
                            with Horizontal(id="strategy-name-row", classes="strategy-row"):
                                yield Static("NAME", classes="profile-inline-label")
                                yield Input(id="strategy-inp-name", placeholder="My Strategy", classes="profile-inline-input profile-name-input")
                                yield Static("DESC", classes="profile-inline-label")
                                yield Input(id="strategy-inp-description", placeholder="Notes", classes="profile-inline-input")
                            yield Static("SIGNALS", classes="panel-title", id="signal-picker-label")
                            with Vertical(id="signal-picker-area"):
                                with Horizontal(classes="signal-picker-row"):
                                    for _lbl, _sig in _SIGNAL_DEFS[:5]:
                                        yield Button(_lbl, id=_sig_btn_id(_sig), classes="signal-btn")
                                with Horizontal(classes="signal-picker-row"):
                                    for _lbl, _sig in _SIGNAL_DEFS[5:10]:
                                        yield Button(_lbl, id=_sig_btn_id(_sig), classes="signal-btn")
                                with Horizontal(classes="signal-picker-row"):
                                    for _lbl, _sig in _SIGNAL_DEFS[10:]:
                                        yield Button(_lbl, id=_sig_btn_id(_sig), classes="signal-btn")
                                yield Static("", id="signal-picker-active", classes="signal-active-display")
                            yield Static("PARAMETERS", classes="panel-title", id="strategy-params-label")
                            with Horizontal(id="strategy-config-row-a", classes="strategy-row"):
                                yield Static("HOLD", classes="profile-inline-label")
                                yield Input(id="strategy-inp-holding-days", placeholder="40", classes="profile-inline-input strategy-small-input")
                                yield Static("REBAL", classes="profile-inline-label")
                                yield Input(id="strategy-inp-rebalance", placeholder="5", classes="profile-inline-input strategy-small-input")
                                yield Static("MAX POS", classes="profile-inline-label")
                                yield Input(id="strategy-inp-max-positions", placeholder="5", classes="profile-inline-input strategy-small-input")
                            with Horizontal(id="strategy-config-row-b", classes="strategy-row"):
                                yield Static("STOP", classes="profile-inline-label")
                                yield Input(id="strategy-inp-stop-loss", placeholder="0.08", classes="profile-inline-input strategy-small-input")
                                yield Static("TRAIL", classes="profile-inline-label")
                                yield Input(id="strategy-inp-trailing-stop", placeholder="0.10", classes="profile-inline-input strategy-small-input")
                                yield Static("SECTOR", classes="profile-inline-label")
                                yield Input(id="strategy-inp-sector-limit", placeholder="0.35", classes="profile-inline-input strategy-small-input")
                            with Horizontal(id="strategy-actions-row", classes="strategy-row"):
                                yield Button("NEW", id="strategy-btn-new", classes="profile-btn")
                                yield Button("SAVE", id="strategy-btn-save", classes="profile-btn profile-btn-primary")
                                yield Button("→ ACTIVE ACCT", id="strategy-btn-assign-current", classes="profile-btn")
                                yield Button("→ SELECTED ACCT", id="strategy-btn-assign-selected", classes="profile-btn")
                                yield Button("DELETE", id="strategy-btn-delete", classes="profile-btn strategy-btn-delete")
                            yield Static("BACKTEST", classes="panel-title", id="strategy-backtest-label")
                            with Horizontal(id="strategy-backtest-row", classes="strategy-row"):
                                yield Static("START", classes="profile-inline-label")
                                yield Input(id="strategy-inp-backtest-start", placeholder="2025-01-01", classes="profile-inline-input strategy-date-input")
                                yield Static("END", classes="profile-inline-label")
                                yield Input(id="strategy-inp-backtest-end", placeholder="2026-04-11", classes="profile-inline-input strategy-date-input")
                                yield Static("CAP", classes="profile-inline-label")
                                yield Input(id="strategy-inp-backtest-capital", placeholder="1000000", classes="profile-inline-input strategy-capital-input")
                                yield Button("RUN", id="strategy-btn-backtest", classes="profile-btn profile-action-button")
                                yield Button("CHART", id="strategy-btn-chart", classes="profile-btn strategy-btn-chart")
                            yield Static("", id="strategy-backtest-summary")

        yield Static(id="status-bar")


# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    NepseDashboard().run()


if __name__ == "__main__":
    main()
