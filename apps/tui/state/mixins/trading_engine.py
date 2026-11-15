"""Per-account TUITradingEngine fleet + the 4 engine->UI callbacks.

Start/stop engine workers, the 4 engine->UI callback factories, the
engine-thread callback READERS (co-located with their producers so the
worker write-site and the call_from_thread reader it feeds live together),
account CRUD that constructs/destroys engines, and active-strategy runtime
application.

The self._stats single-owner / source-priority guard lives in
_make_engine_portfolio_cb: the portfolio callback only marshals to
_on_engine_portfolio_changed when the firing account == _current_account_id,
so a non-active account never clobbers the active account's _stats.
"""

import json as _json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from textual import work
from textual.containers import VerticalScroll
from textual.widgets import Input, Static

from apps.tui.io.csv_import import (
    _build_holdings_watchlist_entries,
    _coerce_dragdrop_path,
    _merge_watchlist_entries,
    _normalize_import_portfolio,
)
from apps.tui.io.persistence import (
    INITIAL_CAPITAL,
    PAPER_IMPORT_BACKUP_DIR,
    ACTIVE_ACCOUNT_FILES,
    _account_dir,
    _account_initial_capital_from_files,
    _blank_account_files,
    _build_account_seed_state,
    _load_accounts_registry,
    _load_profile_config,
    _load_watchlist,
    _next_account_id,
    _save_accounts_registry,
    _save_profile_config,
)
from configs.long_term import LONG_TERM_CONFIG
from backend.quant_pro.paths import ensure_dir
from backend.trading import strategy_registry
from backend.trading.tui_trading_engine import TUITradingEngine
from backend.trading.live_trader import (
    NAV_LOG_COLS,
    PORTFOLIO_COLS,
    TRADE_LOG_COLS,
    save_runtime_state,
)

MAX_ACCOUNTS = 5


class TradingEngineMixin:
    @work(thread=True)
    def _init_tms(self) -> None:
        """Live brokerage not available in public release."""
        pass

    def _attach_tms_live_source(self) -> None:
        """Live brokerage not available in public release."""
        pass

    @work(thread=True)
    def _start_trading_loop(self) -> None:
        """Run the auto-trading engine in a background thread (legacy single-account)."""
        if self._trading_engine:
            self._trading_engine.run_loop()

    def _start_all_account_engines(self) -> None:
        """Create and start one TUITradingEngine per account."""
        self._account_engines = {}
        for account in list(getattr(self, "_paper_accounts", []) or []):
            account_id = str(account.get("id") or "")
            if not account_id:
                continue
            strategy_id = str(account.get("strategy_id") or "")
            strategy = strategy_registry.load_strategy(strategy_id) if strategy_id else None
            config = dict((strategy or {}).get("config") or {}) or dict(LONG_TERM_CONFIG)
            account_dir = _account_dir(account_id)
            account_capital = _account_initial_capital_from_files(
                account_dir,
                float(config.get("initial_capital") or INITIAL_CAPITAL),
            )
            engine = TUITradingEngine(
                capital=account_capital,
                signal_types=list(config.get("signal_types") or list(LONG_TERM_CONFIG.get("signal_types") or [])),
                max_positions=int(config.get("max_positions") or LONG_TERM_CONFIG.get("max_positions") or 5),
                holding_days=int(config.get("holding_days") or LONG_TERM_CONFIG.get("holding_days") or 40),
                sector_limit=float(config.get("sector_limit") or LONG_TERM_CONFIG.get("sector_limit") or 0.35),
                stop_loss_pct=float(config.get("stop_loss_pct") or LONG_TERM_CONFIG.get("stop_loss_pct") or 0.08),
                trailing_stop_pct=float(config.get("trailing_stop_pct") or LONG_TERM_CONFIG.get("trailing_stop_pct") or 0.10),
                hedge_enabled=bool(getattr(self, "_hedge_enabled", True)),
                portfolio_file=account_dir / "paper_portfolio.csv",
                trade_log_file=account_dir / "paper_trade_log.csv",
                nav_log_file=account_dir / "paper_nav_log.csv",
                state_file=account_dir / "paper_state.json",
                account_id=account_id,
                strategy_id=strategy_id,
                strategy_config=config,
                on_status=self._make_engine_status_cb(account_id),
                on_activity=self._make_engine_activity_cb(account_id),
                on_portfolio_changed=self._make_engine_portfolio_cb(account_id),
                on_agent_updated=self._make_engine_agent_cb(account_id),
            )
            self._account_engines[account_id] = engine
            if account_id == self._current_account_id:
                self._trading_engine = engine
            self._start_account_engine_worker(account_id)

    @work(thread=True)
    def _start_account_engine_worker(self, account_id: str) -> None:
        """Run one account's trading engine in a dedicated background worker thread."""
        engine = self._account_engines.get(account_id)
        if engine:
            engine.run_loop()

    def _make_engine_status_cb(self, account_id: str):
        """Status callback — only updates UI when this account is active."""
        def cb(msg: str) -> None:
            if account_id == getattr(self, "_current_account_id", ""):
                self.call_from_thread(self._set_status, msg)
        return cb

    def _make_engine_activity_cb(self, account_id: str):
        """Activity log callback — only updates UI when this account is active."""
        def cb(msg: str) -> None:
            if account_id == getattr(self, "_current_account_id", ""):
                self.call_from_thread(self._append_activity, msg)
        return cb

    def _make_engine_portfolio_cb(self, account_id: str):
        """Portfolio-changed callback — only refreshes UI when this account is active."""
        def cb() -> None:
            if account_id == getattr(self, "_current_account_id", ""):
                self.call_from_thread(self._on_engine_portfolio_changed)
        return cb

    def _make_engine_agent_cb(self, account_id: str):
        """Agent-updated callback — only refreshes UI when this account is active."""
        def cb() -> None:
            if account_id == getattr(self, "_current_account_id", ""):
                self.call_from_thread(self._on_engine_agent_updated)
        return cb

    def _append_activity(self, msg: str) -> None:
        """Append a message to the activity log in the portfolio tab."""
        try:
            scroll = self.query_one("#activity-scroll", VerticalScroll)
            label = Static(msg, classes="activity-line")
            scroll.mount(label)
            scroll.scroll_end(animate=False)
            # Keep max 100 lines
            children = list(scroll.children)
            if len(children) > 100:
                for child in children[:len(children) - 100]:
                    child.remove()
        except Exception:
            pass

    def _on_engine_portfolio_changed(self) -> None:
        """Called by engine when positions change — refresh portfolio display."""
        if self._website_book is not None:
            return  # the shared website book is the one on screen
        if self._trading_engine:
            self._stats = self._trading_engine.get_portfolio_stats()
            self._populate_portfolio_tab(self._stats)
            self._populate_risk_tab(self._stats)
            self._populate_trades_full()
        self._render_hedge_panel()

    def _on_engine_agent_updated(self) -> None:
        """Called by engine after agent analysis — refresh agents tab."""
        try:
            self._load_agent_runtime_state()
            self._populate_agent_tab()
        except Exception:
            pass

    def _display_nav_mode_tag(self) -> str:
        if self._trading_engine:
            phase = self._trading_engine.phase.upper().replace("_", " ")
            return f"[bold #00cfff]AUTO[/] [dim]{phase}[/]"
        return "[bold #00ff7f]PAPER[/]"

    def _apply_active_strategy_runtime(self) -> None:
        config = self._active_strategy_config()
        self.signal_types = list(config.get("signal_types") or list(LONG_TERM_CONFIG.get("signal_types") or []))
        self.max_positions = int(config.get("max_positions") or LONG_TERM_CONFIG.get("max_positions") or 0)
        self.holding_days = int(config.get("holding_days") or LONG_TERM_CONFIG.get("holding_days") or 0)
        self.sector_limit = float(config.get("sector_limit") or LONG_TERM_CONFIG.get("sector_limit") or 0.0)
        self.use_regime_filter = bool(config.get("use_regime_filter", True))

    def _activate_paper_account(self, account_id: str) -> str:
        target_id = str(account_id or "").strip()
        if not target_id:
            raise ValueError("Select an account to activate")
        if target_id == self._current_account_id:
            return f"{target_id} is already active"
        self._persist_active_account_snapshot()
        source_dir = _account_dir(target_id)
        _blank_account_files(source_dir)
        backup_dir = self._backup_profile_targets(list(ACTIVE_ACCOUNT_FILES.values()))
        for name, active_path in ACTIVE_ACCOUNT_FILES.items():
            source = source_dir / name
            target = Path(active_path)
            if source.exists():
                shutil.copy2(source, target)
        self._current_account_id = target_id
        self._selected_account_id = target_id
        # Switch active engine reference to the newly activated account
        self._trading_engine = self._account_engines.get(target_id)
        target_account = self._strategy_account_binding(target_id) or {}
        self._selected_strategy_id = str(target_account.get("strategy_id") or self._selected_strategy_id)
        self._apply_active_strategy_runtime()
        self._signals_table_cache_key = ""
        self._signals_table_cache_payload = None
        self._sync_agent_account_context_env()
        profile = _load_profile_config()
        profile["current_account_id"] = target_id
        _save_profile_config(profile)
        self._paper_watchlist = _load_watchlist()
        self._watchlist = list(self._paper_watchlist)
        self._load_paper_orders()
        self._populate_portfolio_and_risk()
        self._populate_trades_full()
        self._render_hedge_panel()
        self._populate_orders_tab()
        self._populate_watchlist()
        self._populate_paper_profile_panel(self._stats)
        self._populate_strategies_tab()
        self._load_signals_async(force=True)
        backup_note = f" | backup {backup_dir.name}" if backup_dir else ""
        active_name = next((str(a.get("name") or target_id) for a in self._paper_accounts if str(a.get("id") or "") == target_id), target_id)
        return f"Activated {active_name}{backup_note}"

    def _create_paper_account(self, raw_name: str, raw_portfolio_path: str, raw_target_nav: str) -> str:
        name = str(raw_name or "").strip()
        if not name:
            raise ValueError("Enter an account name")
        portfolio_path = _coerce_dragdrop_path(raw_portfolio_path)
        token = str(raw_target_nav or "").strip().replace(",", "")
        if not token:
            raise ValueError("Enter a target NAV")
        target_nav = float(token)
        if target_nav <= 0:
            raise ValueError("Target NAV must be positive")
        if portfolio_path:
            if not portfolio_path.exists():
                raise ValueError("Portfolio CSV path not found")
            df = pd.read_csv(portfolio_path)
            portfolio_df = _normalize_import_portfolio(df)
        else:
            portfolio_df = pd.DataFrame(columns=PORTFOLIO_COLS)
        state, nav_log = _build_account_seed_state(portfolio_df, target_nav)
        registry = _load_accounts_registry()
        accounts = list(registry.get("accounts") or [])
        if len(accounts) >= MAX_ACCOUNTS:
            raise ValueError(f"Maximum {MAX_ACCOUNTS} accounts allowed")
        if any(str(account.get("name") or "").strip().lower() == name.lower() for account in accounts):
            raise ValueError(f"{name} already exists")
        account_id = _next_account_id(accounts)
        target_dir = _account_dir(account_id)
        _blank_account_files(target_dir)
        portfolio_df.reindex(columns=PORTFOLIO_COLS).to_csv(target_dir / "paper_portfolio.csv", index=False)
        portfolio_df.reindex(columns=PORTFOLIO_COLS).to_csv(target_dir / "tui_paper_portfolio.csv", index=False)
        pd.DataFrame(columns=TRADE_LOG_COLS).to_csv(target_dir / "paper_trade_log.csv", index=False)
        pd.DataFrame(columns=TRADE_LOG_COLS).to_csv(target_dir / "tui_paper_trade_log.csv", index=False)
        nav_log.reindex(columns=NAV_LOG_COLS).to_csv(target_dir / "paper_nav_log.csv", index=False)
        nav_log.reindex(columns=NAV_LOG_COLS).to_csv(target_dir / "tui_paper_nav_log.csv", index=False)
        save_runtime_state(str(target_dir / "paper_state.json"), dict(state))
        save_runtime_state(str(target_dir / "tui_paper_state.json"), dict(state))
        watchlist_entries = _build_holdings_watchlist_entries(portfolio_df)
        (target_dir / "watchlist.json").write_text(_json.dumps(_merge_watchlist_entries(watchlist_entries), indent=2))
        (target_dir / "tui_paper_orders.json").write_text("[]")
        (target_dir / "tui_paper_order_history.json").write_text("[]")
        now_stamp = datetime.now().isoformat(timespec="seconds")
        accounts.append({
            "id": account_id,
            "name": name,
            "strategy_id": strategy_registry.default_strategy_for_account(account_id),
            "created_at": now_stamp,
            "updated_at": now_stamp,
        })
        registry["accounts"] = accounts
        _save_accounts_registry(registry)
        self._paper_accounts = strategy_registry.ensure_account_strategy_ids(accounts)
        self._persist_profile_paths(portfolio_path=str(portfolio_path) if portfolio_path else "", target_nav=target_nav)
        self.query_one("#profile-inp-account-name", Input).value = f"Account {len(accounts) + 1}"
        self._selected_account_id = account_id
        # Start engine for the new account if multi-engine is running
        if self.trade_mode == "paper" and account_id not in getattr(self, "_account_engines", {}):
            _new_strategy = strategy_registry.load_strategy(strategy_registry.default_strategy_for_account(account_id))
            _new_config = dict((_new_strategy or {}).get("config") or {}) or dict(LONG_TERM_CONFIG)
            _new_adir = _account_dir(account_id)
            _new_capital = _account_initial_capital_from_files(
                _new_adir,
                float(_new_config.get("initial_capital") or target_nav or INITIAL_CAPITAL),
            )
            _new_engine = TUITradingEngine(
                capital=_new_capital,
                signal_types=list(_new_config.get("signal_types") or list(LONG_TERM_CONFIG.get("signal_types") or [])),
                max_positions=int(_new_config.get("max_positions") or LONG_TERM_CONFIG.get("max_positions") or 5),
                holding_days=int(_new_config.get("holding_days") or LONG_TERM_CONFIG.get("holding_days") or 40),
                sector_limit=float(_new_config.get("sector_limit") or LONG_TERM_CONFIG.get("sector_limit") or 0.35),
                stop_loss_pct=float(_new_config.get("stop_loss_pct") or LONG_TERM_CONFIG.get("stop_loss_pct") or 0.08),
                trailing_stop_pct=float(_new_config.get("trailing_stop_pct") or LONG_TERM_CONFIG.get("trailing_stop_pct") or 0.10),
                hedge_enabled=bool(getattr(self, "_hedge_enabled", True)),
                portfolio_file=_new_adir / "paper_portfolio.csv",
                trade_log_file=_new_adir / "paper_trade_log.csv",
                nav_log_file=_new_adir / "paper_nav_log.csv",
                state_file=_new_adir / "paper_state.json",
                account_id=account_id,
                strategy_id=strategy_registry.default_strategy_for_account(account_id),
                strategy_config=_new_config,
                on_status=self._make_engine_status_cb(account_id),
                on_activity=self._make_engine_activity_cb(account_id),
                on_portfolio_changed=self._make_engine_portfolio_cb(account_id),
                on_agent_updated=self._make_engine_agent_cb(account_id),
            )
            self._account_engines[account_id] = _new_engine
            self._start_account_engine_worker(account_id)
        message = self._activate_paper_account(account_id)
        seed_note = f"{portfolio_df.shape[0]} seeded holdings" if not portfolio_df.empty else "cash-only account"
        return f"Created {name} | {seed_note} | {message}"

    def _delete_paper_account(self, account_id: str) -> str:
        """Stop engine, backup files, remove account from registry."""
        target_id = str(account_id or "").strip()
        if not target_id:
            raise ValueError("No account selected")
        registry = _load_accounts_registry()
        accounts = list(registry.get("accounts") or [])
        if len(accounts) <= 1:
            raise ValueError("Cannot delete the only account")
        target = next((a for a in accounts if str(a.get("id") or "") == target_id), None)
        if not target:
            raise ValueError(f"Account {target_id} not found")
        # Stop and remove the account's engine
        engine = self._account_engines.pop(target_id, None)
        if engine:
            try:
                engine.stop()
            except Exception:
                pass
        if self._trading_engine is engine:
            self._trading_engine = None
        # Backup account directory before deleting
        account_dir = _account_dir(target_id)
        if account_dir.exists():
            try:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = ensure_dir(PAPER_IMPORT_BACKUP_DIR / f"deleted_{target_id}_{stamp}")
                shutil.copytree(account_dir, backup / target_id)
                shutil.rmtree(account_dir, ignore_errors=True)
            except Exception:
                pass
        # Remove from registry
        remaining = [a for a in accounts if str(a.get("id") or "") != target_id]
        registry["accounts"] = remaining
        _save_accounts_registry(registry)
        self._paper_accounts = strategy_registry.ensure_account_strategy_ids(remaining)
        name = str(target.get("name") or target_id)
        # If deleting the active account, activate the first remaining one
        if target_id == self._current_account_id and remaining:
            try:
                self._activate_paper_account(str(remaining[0].get("id") or ""))
            except Exception:
                pass
        else:
            self._populate_account_list()
        return f"Deleted '{name}' (backup saved)"
