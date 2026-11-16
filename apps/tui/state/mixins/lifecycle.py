"""App boot/teardown + the instance-state birth site + mode/credential callbacks.

Owns the canonical 47-attr state birth (_init_dashboard) so every other mixin
reads attrs that exist. Migrated last so the state contract is frozen against the
now-final mixin set — no attr is referenced before birth.

State ownership seam: self._stats is born here as {} (seed only). The AUTHORITATIVE
producer is _populate_portfolio_and_risk in TabRefreshMixin; TradingEngineMixin
overwrites _stats only for the active account. This mixin never overwrites the seed.
"""

import time
from typing import Optional

from rich.text import Text
from textual import work

from backend.quant_pro.dashboard_data import MD
from backend.market.kalimati_market import init_kalimati_db
from backend.trading import strategy_registry
from apps.tui.screens.mode_select import ModeSelectScreen
from apps.tui.io.persistence import (
    _bootstrap_paper_accounts,
    _ensure_paper_runtime_files,
    _load_accounts_registry,
    _load_hedge_trade_log,
    _load_profile_config,
    _load_watchlist,
)

# Ticker scroll speed (seconds between scroll steps) — consumed only by _init_dashboard.
TICKER_SPEED = 0.15


class LifecycleMixin:
    def on_mount(self) -> None:
        self.push_screen(ModeSelectScreen(), callback=self._on_mode_selected)

    def _on_mode_selected(self, mode: str | None) -> None:
        if not mode:
            self.exit()
            return
        self.trade_mode = "paper"
        self._init_dashboard()

    def _on_tms_credentials(self, result: dict | None) -> None:
        if not result:
            # User pressed ESC — fall back to mode select
            self.push_screen(ModeSelectScreen(), callback=self._on_mode_selected)
            return
        self._tms_credentials = result
        self._init_dashboard()
        self._init_tms()

    def _init_dashboard(self) -> None:
        _ensure_paper_runtime_files()
        self._paper_accounts, self._current_account_id = _bootstrap_paper_accounts()
        self._selected_account_id = self._current_account_id
        self._load_strategies_registry()
        self._apply_active_strategy_runtime()
        self._sync_agent_account_context_env()
        try:
            from backend.quant_pro.database import init_db
            init_db()
        except Exception:
            pass
        # Pull live quotes into DB before loading market data
        try:
            from backend.quant_pro.realtime_market import get_market_data_provider
            snap = get_market_data_provider().fetch_snapshot(force=True)
            if snap and snap.quotes:
                self._upsert_live_prices(snap)
        except Exception:
            pass
        self.md = MD(top_n=25)
        self._regime = "loading..."
        self._stats: dict = {}
        self._lookup_cache: dict[tuple[str, str, str], dict] = {}
        self._lookup_request_key: Optional[tuple[str, str, str]] = None
        self._signals_table_cache_key: str = ""
        self._signals_table_cache_payload: Optional[tuple[list[tuple[str, str]], list[list[Text]], int]] = None
        self._signals_workspace_cache_key: str = ""
        self._signals_workspace_cache_payload: Optional[dict] = None
        self._signals_last_loaded_at: str = ""
        self._signals_last_strategy_name: str = ""
        self._signals_last_count: int = 0
        self._signal_min_score: float = 0.0
        self._broker_floor_filter: str = "all"  # all | circ | pump | smart
        self._ticker_text = ""
        self._ticker_offset = 0
        self._rates_search_query: str = ""
        self._kalimati_rows: list[dict] = []
        self._kalimati_status: str = "Loading..."
        self._macro_rates: dict = {}
        self._hedge_enabled: bool = True    # ON/OFF toggle for gold/silver hedge overlay
        self._hedge_trade_log: list = _load_hedge_trade_log()
        self._active_signals: set[str] = set()  # signal picker state
        self._new_strategy_mode: bool = False   # True after NEW pressed, False after list selection or SAVE
        self._watchlist_stock_rows: list[dict] = []
        self._watchlist_rates_rows: list[dict] = []
        self._watchlist_commodity_rows: list[dict] = []
        self._last_macro_rates_fetch_at: float = 0.0
        # Linked to nepsequant.com (./nqt-connect login)? Then the website's paper
        # book is the terminal's paper book; it is fetched in the background.
        from apps.tui.io.website_book import WebsiteBook
        self._website_book = WebsiteBook.from_saved() if self.trade_mode == "paper" else None
        # Linked: the broker panels come from nepsequant.com's floorsheet too.
        from apps.tui.io.website_floor import WebsiteFloor
        self._website_floor = WebsiteFloor.from_book(self._website_book)
        self._init_website_strategies()
        self._build_ticker()
        self._update_header()
        self._update_index()
        self._load_profile_inputs()
        self._populate_strategies_tab()
        self._populate_market()
        self._populate_portfolio_and_risk()
        self._populate_trades_full()
        self._render_hedge_panel()
        self._osint_stories: list[dict] = []
        self._agent_analysis: dict = {}
        self._agent_history: list[dict] = []
        self._agent_archived_history: list[dict] = []
        self._agent_archive_count: int = 0
        self._agent_show_archived = False
        self._agent_hidden_recent_history: list[dict] = []
        self._agent_preview_override: Optional[dict] = None
        self._last_agent_auto_order_key: Optional[str] = None
        self._agent_typing_visible = False
        self._agent_typing_frame = 0
        self._agent_session_started_at = time.time()
        self._agent_visible_since = self._agent_session_started_at
        self._agent_chat_loaded = False
        self._load_agent_runtime_state()
        self._populate_agent_tab()
        if not list((self._agent_analysis or {}).get("stocks") or []):
            self._run_agent_analysis(force=False)
        self._load_paper_orders()
        self._populate_orders_tab()
        self._refresh_order_action_buttons()
        self._paper_watchlist = _load_watchlist()
        self._watchlist = list(self._paper_watchlist)
        self._populate_watchlist()
        self._populate_signals_workspace()
        self._load_signals_async()
        self._load_regime_async()
        self._load_osint_async()
        init_kalimati_db()
        self._load_kalimati_async()
        self._load_macro_rates_async(force=True)
        mode_label = self._display_mode_label()
        self._set_status(
            f"Mode: {mode_label}  │  Session: {self.md.latest}  │  "
            f"▲{self.md.adv} ▼{self.md.dec}  │  Auto-refresh 30s"
            + ("  │  Paper book: nepsequant.com (loading…)" if self._website_book is not None else "")
        )
        if self._website_book is not None:
            self._load_website_book()
        self.set_interval(30, self._auto_refresh)
        self.set_interval(TICKER_SPEED, self._scroll_ticker)
        self.set_interval(0.45, self._animate_agent_typing)

        # Start one auto-trading engine per account in background threads.
        # Not with the shared website book: automatic trading would trade a
        # local book you no longer see.
        if self.trade_mode == "paper" and self._website_book is None:
            self._start_all_account_engines()

    @work(thread=True, group="website-book")
    def _load_website_book(self) -> None:
        book = self._website_book
        ok = book.refresh()
        self.call_from_thread(self._on_website_book_loaded, ok)

    def _on_website_book_loaded(self, ok: bool) -> None:
        book = self._website_book
        if book is None:
            return
        if ok:
            note = "" if book.can_trade else "  (read-only key: create a key with Paper trading ticked to trade)"
            self._set_status(f"Paper book: your nepsequant.com paper portfolio, shared with the website{note}")
        else:
            self._set_status(f"Paper book: could not load nepsequant.com — {book.error}")
        self._load_paper_orders()
        self._populate_orders_tab()
        self._populate_portfolio_and_risk()
        self._populate_trades_full()
        self._refresh_active_tab_view()
        self._load_website_strategies(True)

    def on_unmount(self) -> None:
        """Graceful shutdown — stop trading engine."""
        try:
            if self.trade_mode == "paper":
                self._persist_active_account_snapshot()
        except Exception:
            pass
        for _eng in list(getattr(self, "_account_engines", {}).values()):
            try:
                _eng.stop()
            except Exception:
                pass
        try:
            src = getattr(self, "_tms_live_src", None)
            if src is not None:
                src.stop()
        except Exception:
            pass

    def _reload_account_bindings_from_disk(self) -> None:
        registry = _load_accounts_registry()
        accounts = strategy_registry.ensure_account_strategy_ids(list(registry.get("accounts") or []))
        if not accounts:
            return
        self._paper_accounts = accounts
        known_ids = {str(account.get("id") or "") for account in accounts}
        profile = _load_profile_config()
        current_account_id = str(
            profile.get("current_account_id")
            or getattr(self, "_current_account_id", "account_1")
            or "account_1"
        ).strip()
        if current_account_id not in known_ids:
            current_account_id = str(accounts[0].get("id") or "account_1")
        self._current_account_id = current_account_id
        selected_account_id = str(getattr(self, "_selected_account_id", current_account_id) or current_account_id).strip()
        if selected_account_id not in known_ids:
            selected_account_id = current_account_id
        self._selected_account_id = selected_account_id
        binding = self._strategy_account_binding(current_account_id) or {}
        self._selected_strategy_id = str(
            binding.get("strategy_id")
            or getattr(self, "_selected_strategy_id", "")
            or strategy_registry.default_strategy_for_account(current_account_id)
        )
        self._sync_agent_account_context_env()
