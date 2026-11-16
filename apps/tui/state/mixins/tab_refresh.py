"""Data-refresh engine: master worker + per-tab dispatch + all populate readers.

The largest mixin. Owns the master _do_refresh worker, every async fetch worker,
the per-tab dispatch hub (_refresh_active_tab_view), and every populate_*/render_*
payload reader for market / portfolio / risk / signals / kalimati / news /
strategies / lookup / screener / watchlist. Each @work worker is co-located with
the call_from_thread payload reader it marshals to.

State ownership seams:
  - self._stats single-owner: _populate_portfolio_and_risk is the AUTHORITATIVE
    producer of _stats here. LifecycleMixin only seeds {} at init; TradingEngineMixin
    overwrites it only for the active account. That invariant must hold.
  - _load_regime_async writes self._regime; the reader is HeaderStatusMixin._update_index
    (marshaled via call_from_thread) — a safe cross-mixin marshal.
  - _do_refresh writes self.md / self._osint_stories / self._kalimati_rows; readers
    (_match_paper_orders in OrderBookMixin, the populate_* here) resolve via the MRO.

OSINT seam: _fetch_osint_stories / OSINT_BASE / OSINT_TIMEOUT / _requests stay in
dashboard_tui.py (patch-target home). The three workers that need them
(_do_refresh / _load_osint_async / _do_vector_search) pull them via an in-function
import from dashboard_tui at call time — preserving the dashboard_tui patch seam and
avoiding a circular import at module load.
"""

import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.text import Text
from textual import work
from textual.containers import VerticalScroll
from textual.widgets import Button, DataTable, Input, OptionList, Static
from textual.widgets.option_list import Option

from apps.tui.theme import (
    AMBER, BLUE, CYAN, DIM, GAIN, GAIN_HI, LABEL, LOSS, LOSS_HI, PURPLE, WHITE, YELLOW,
    _npr, _vol,
)
from apps.tui.widgets.market_panel import MarketPanel
from apps.tui.widgets.signal_defs import _SIGNAL_DEFS, _sig_btn_id
from apps.tui.render.cells import (
    _chg_text,
    _contains_non_ascii,
    _dim_text,
    _format_compact_npr,
    _news_display_headline,
    _npr_k,
    _pnl_color,
    _price_text,
    _sym_text,
    _truncate_text,
    _vol_text,
)
from apps.tui.io.floor_metrics import local_floor_records
from apps.tui.render.broker_floor import (
    BROKER_COLS,
    TOP_BROKER_COLS,
    floor_signal_rows,
    floor_title,
    message_row,
    top_broker_rows as top_broker_rows_for,
    website_floor_panels,
)
from apps.tui.render.text_blocks import (
    _headline_fallback_from_url,
    _render_lookup_intelligence,
    _render_stock_report,
)
from apps.tui.render.charts import _render_candlestick_chart, _render_sparkline
from apps.tui.io.stats import (
    _compute_account_portfolio_stats,
    _compute_portfolio_stats,
    _get_regime,
    _signal_label,
    _stats_from_frames,
)
from apps.tui.io.intraday import (
    _ensure_lookup_history,
    _load_intraday_ohlcv,
    _nst_today_str,
    _resample_ohlcv,
)
from apps.tui.io.fetchers import (
    _fetch_gold_silver_prices,
    _fetch_noc_fuel_prices,
    _fetch_nrb_forex_rates,
    _fetch_yahoo_futures_price,
    _translate_batch,
)
from apps.tui.io.csv_import import _dedupe_symbol_rows, _merge_watchlist_entries
from apps.tui.io.watchlist_io import _dedupe_watchlist_entries
from apps.tui.io.persistence import (
    PAPER_STATE_FILE,
    TRADING_RUNTIME_DIR,
    _account_dir,
    _apply_indicator_history_change,
    _load_cached_tms_bundle,
    _load_macro_indicator_history,
    _load_manual_paper_cash,
    _load_nav_log,
    _load_profile_config,
    _load_trade_log,
    _merge_tms_bundle_with_cache,
    _portfolio_mark_value,
    _save_macro_indicator_history,
    _tms_health_flag,
)
from configs.long_term import LONG_TERM_CONFIG
from backend.quant_pro.dashboard_data import _db, load_port
from backend.quant_pro.stock_report import build_stock_report
from backend.agents.agent_analyst import publish_agent_signal_snapshot
from backend.market.kalimati_market import get_kalimati_display_rows, refresh_kalimati
from backend.trading.live_trader import (
    NAV_LOG_COLS,
    PORTFOLIO_COLS,
    TRADE_LOG_COLS,
    load_runtime_state,
)
from backend.trading import strategy_registry


SEVERITY_STYLE = {
    "critical": f"bold {LOSS_HI}", "high": LOSS, "medium": YELLOW,
    "low": LABEL, "": LABEL,
}
TYPE_STYLE = {
    "political": PURPLE, "security": LOSS, "economic": CYAN,
    "disaster": LOSS_HI, "social": BLUE, "": LABEL,
}


class TabRefreshMixin:
    def _build_ticker(self) -> None:
        """Build the ticker string from market data."""
        items = []
        # Top gainers
        if not self.md.gainers.empty:
            for _, r in self.md.gainers.head(5).iterrows():
                chg = r["chg"]
                arrow = "▲" if chg >= 0 else "▼"
                items.append(f"{r['symbol']} {r['close']:.1f} {arrow}{chg:+.2f}%")
        # Top losers
        if not self.md.losers.empty:
            for _, r in self.md.losers.head(5).iterrows():
                items.append(f"{r['symbol']} {r['close']:.1f} ▼{r['chg']:+.2f}%")
        # Volume leaders
        if not self.md.vol_top.empty:
            for _, r in self.md.vol_top.head(3).iterrows():
                items.append(f"{r['symbol']} Vol:{_vol(r['volume'])}")
        # NEPSE index
        if len(self.md.nepse) >= 2:
            ni = self.md.nepse.iloc[0]["close"]
            np_ = self.md.nepse.iloc[1]["close"]
            chg = (ni - np_) / np_ * 100
            items.insert(0, f"NEPSE {ni:,.1f} {chg:+.2f}%")
        # Corporate actions
        if not self.md.corp.empty:
            for _, r in self.md.corp.head(3).iterrows():
                bc = r["bookclose_date"]
                days = (bc - datetime.now()).days
                cash = float(r.get("cash_dividend_pct") or 0)
                bonus = float(r.get("bonus_share_pct") or 0)
                parts = []
                if cash > 0: parts.append(f"Div {cash:.0f}%")
                if bonus > 0: parts.append(f"Bonus {bonus:.0f}%")
                if parts:
                    items.append(f"{r['symbol']} {' '.join(parts)} BookClose:{days}d")
        # OSINT headlines in ticker (prefer translated English display text)
        news_count = 0
        for s in getattr(self, "_osint_stories", []):
            sev = str(s.get("severity") or "")
            if sev not in ("high", "medium", "critical"):
                continue
            headline = _news_display_headline(s)
            if not headline or _contains_non_ascii(headline):
                continue
            tag = sev.upper()[:4]
            items.append(f"[{tag}] {_truncate_text(headline, 65)}")
            news_count += 1
            if news_count >= 10:
                break

        # Build the full ticker line with separators
        sep = "  ◆  "
        self._ticker_text = sep + sep.join(items) + sep
        self._ticker_offset = 0
    def _scroll_ticker(self) -> None:
        """Advance the ticker by one character."""
        text = self._ticker_text
        if not text:
            return
        # Get terminal width
        try:
            w = self.size.width
        except Exception:
            w = 120
        # Build the visible window
        doubled = text + text  # loop seamlessly
        start = self._ticker_offset % len(text)
        visible = doubled[start:start + w]
        # Color the ticker: symbols in amber, numbers in white, arrows colored
        from rich.text import Text as RichText
        styled = RichText()
        i = 0
        while i < len(visible):
            ch = visible[i]
            if ch == "▲":
                styled.append(ch, style=f"bold {GAIN_HI}")
            elif ch == "▼":
                styled.append(ch, style=f"bold {LOSS_HI}")
            elif ch == "◆":
                styled.append(ch, style="#555555")
            elif ch == "+" or (ch == "-" and i + 1 < len(visible) and visible[i + 1].isdigit()):
                # Collect the number
                j = i + 1
                while j < len(visible) and (visible[j].isdigit() or visible[j] in ".%"):
                    j += 1
                num_str = visible[i:j]
                if ch == "+":
                    styled.append(num_str, style=GAIN)
                else:
                    styled.append(num_str, style=LOSS)
                i = j
                continue
            elif ch == "[":
                # Severity tag like [HIGH] or [MED ]
                end = visible.find("]", i)
                if end != -1 and end - i <= 6:
                    tag = visible[i+1:end].strip()
                    tag_style = {"CRIT": f"bold {LOSS_HI}", "HIGH": LOSS,
                                 "MEDI": YELLOW, "LOW": LABEL}.get(tag, LABEL)
                    styled.append(visible[i:end+1], style=tag_style)
                    i = end + 1
                    continue
                styled.append(ch, style="#888888")
            elif ch.isalpha() or ord(ch) > 127:
                # Collect word (ASCII or Nepali unicode)
                j = i
                while j < len(visible) and (visible[j].isalpha() or visible[j] == ":" or ord(visible[j]) > 127):
                    j += 1
                word = visible[i:j]
                if word in ("NEPSE", "Vol", "Div", "Bonus", "BookClose"):
                    styled.append(word, style=f"bold {AMBER}")
                else:
                    styled.append(word, style=WHITE)
                i = j
                continue
            else:
                styled.append(ch, style="#888888")
            i += 1
        self.query_one("#ticker-bar", Static).update(styled)
        self._ticker_offset += 1
    @staticmethod
    def _upsert_live_prices(snap) -> None:
        """Write live quote LTPs + NEPSE index into stock_prices."""
        import sqlite3
        from backend.quant_pro.database import get_db_path
        from backend.trading.live_trader import now_nst
        today = now_nst().strftime("%Y-%m-%d")
        rows = []
        for sym, q in snap.quotes.items():
            ltp = q.get("last_traded_price") or q.get("close_price")
            if not ltp or float(ltp) <= 0:
                continue
            ltp = float(ltp)
            vol = int(float(q.get("total_trade_quantity") or 0))
            rows.append((sym, today, ltp, ltp, ltp, ltp, vol))

        # Fetch NEPSE index from API + compute total market volume
        try:
            from nepse import Nepse
            client = Nepse()
            client.setTLSVerification(False)
            indices = client.getNepseIndex()
            for idx in (indices or []):
                if idx.get("index") == "NEPSE Index":
                    cur = float(idx.get("currentValue") or idx.get("close") or 0)
                    prev = float(idx.get("close") or idx.get("previousClose") or 0)
                    high = float(idx.get("high") or cur)
                    low = float(idx.get("low") or cur)
                    # Total market volume = sum of all stock volumes today
                    market_vol = sum(r[6] for r in rows)
                    if cur > 0:
                        rows.append(("NEPSE", today, prev, high, low, cur, market_vol))
                    break
        except Exception:
            pass

        if not rows:
            return
        try:
            conn = sqlite3.connect(str(get_db_path()))
            conn.executemany(
                "INSERT OR REPLACE INTO stock_prices (symbol, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
    def _auto_refresh(self) -> None:
        if self._refresh_inflight:
            return
        self._do_refresh()
    def _data_version(self) -> str:
        ts = getattr(self, "md", None)
        if ts is None:
            return "unknown"
        return f"{self.md.latest}:{int(self.md.ts.timestamp())}"
    def _signals_cache_key(self) -> str:
        payload = self._active_strategy_payload() or {}
        config = dict(payload.get("config") or {})
        signal_types = ",".join(str(item).strip() for item in list(config.get("signal_types") or []))
        return ":".join([
            self._data_version(),
            str(payload.get("id") or "default_c5"),
            signal_types,
            "regime" if bool(config.get("use_regime_filter", True)) else "no_regime",
        ])
    def _refresh_active_tab_view(self, *, force_watchlist_sync: bool = False) -> None:
        """Refresh only the currently visible tab to keep redraws stable."""
        tab = str(self.active_tab or "market")
        if tab == "market":
            self._populate_market()
            return
        if tab == "portfolio":
            self._populate_portfolio_and_risk()
            self._populate_trades_full()
            self._render_hedge_panel()
            return
        if tab == "signals":
            self._populate_signals_workspace()
            self._load_signals_async()
            return
        if tab == "lookup":
            if not str(self.lookup_sym or "").strip():
                self.lookup_sym = "NEPSE"
            self._populate_lookup()
            return
        if tab == "agents":
            self._populate_agent_tab()
            return
        if tab == "orders":
            self._populate_orders_tab()
            return
        if tab == "watchlist":
            self._populate_watchlist()
            if self.trade_mode == "live" and self.tms_service:
                self._refresh_watchlist_live(force=force_watchlist_sync)
            return
        if tab == "kalimati":
            self._load_macro_rates_async()
            self._populate_kalimati()
            return
        if tab == "account":
            self._populate_portfolio_and_risk()
            self._populate_paper_profile_panel(self._stats)
            return
        if tab == "strategies":
            self._populate_strategies_tab()
            return
    @work(thread=True)
    def _do_refresh(self) -> None:
        if self._refresh_inflight:
            return
        self._refresh_inflight = True
        try:
            # Pull live quotes from API + upsert today's prices into stock_prices
            try:
                from backend.quant_pro.realtime_market import get_market_data_provider
                provider = get_market_data_provider()
                snap = provider.fetch_snapshot(force=True)
                if snap and snap.quotes:
                    self._upsert_live_prices(snap)
            except Exception:
                pass

            self.md.refresh()
            if self._website_book is not None:
                self._website_book.refresh()  # the shared nepsequant.com paper book
                self._run_autotrader_pass()  # the assigned strategy, when auto-trade is on
            from apps.tui.dashboard_tui import _fetch_osint_stories
            self._osint_stories = _fetch_osint_stories(80)

            # Keep hidden tabs stable; only repaint the visible view.
            self.call_from_thread(self._update_header)
            self.call_from_thread(self._update_index)
            self.call_from_thread(self._build_ticker)
            self.call_from_thread(self._match_paper_orders)

            # Refresh kalimati data in the background, but repaint only if visible.
            rows, status = refresh_kalimati()
            self._kalimati_rows = rows
            self._kalimati_status = status

            self.call_from_thread(self._refresh_active_tab_view)
            ts = self.md.ts.strftime("%H:%M:%S")
            self.call_from_thread(
                self._set_status,
                f"Refreshed {ts}  │  ▲{self.md.adv} ▼{self.md.dec}  ={self.md.unch}",
            )
        finally:
            self._refresh_inflight = False
    @work(thread=True)
    def _load_regime_async(self) -> None:
        self._regime = _get_regime(self.md)
        self.call_from_thread(self._update_index)
    @work(thread=True)
    def _load_signals_async(self, force: bool = False) -> None:
        strategies = self._website_strategies
        if strategies is not None:
            # Linked: the assigned nepsequant.com strategy's picks for today.
            if not strategies.loaded:
                strategies.refresh()
            if strategies.assigned:
                strategies.picks(strategies.assigned, force=force)
            self.call_from_thread(self._render_website_signals)
            return
        cache_key = self._signals_cache_key()
        if (
            not force
            and cache_key == self._signals_table_cache_key
            and self._signals_table_cache_payload is not None
        ):
            cols, rows, count = self._signals_table_cache_payload
            self.call_from_thread(self._set_signals_table, cols, rows)
            return
        payload = self._active_strategy_payload() or {}
        config = dict(payload.get("config") or {})
        strategy_name = str(payload.get("name") or "Active Strategy")
        signal_types = list(config.get("signal_types") or list(LONG_TERM_CONFIG.get("signal_types") or []))
        use_regime_filter = bool(config.get("use_regime_filter", True))
        self.call_from_thread(self._set_signals_table_loading, strategy_name)
        self.call_from_thread(self._set_status, f"Loading signals for {strategy_name}...")
        try:
            from backend.backtesting.simple_backtest import load_all_prices
            from backend.trading.live_trader import generate_signals
            conn = _db()
            prices_df = load_all_prices(conn)
            conn.close()
            sigs, _regime = generate_signals(
                prices_df,
                signal_types,
                use_regime_filter=use_regime_filter,
            )
            min_score = float(getattr(self, "_signal_min_score", 0.0))
            if min_score > 0:
                sigs = [s for s in sigs if float(s.get("score") or 0.0) >= min_score]
            sigs = list(sorted(sigs, key=lambda x: float(x.get("score") or 0.0), reverse=True)[:50])

            cols = [("  #", "n"), ("SYMBOL", "sym"), ("SCORE", "score"),
                    ("TYPE", "type"), ("STR", "str"), ("CONF", "conf"), ("DIR", "dir")]
            rows = []
            for i, s in enumerate(sigs, 1):
                score = float(s.get("score") or 0.0)
                strength = float(s.get("strength") or 0.0)
                confidence = float(s.get("confidence") or 0.0)
                signal_type = str(s.get("signal_type") or "")
                score_style = GAIN_HI if score > 0 else LOSS if score < 0 else LABEL
                rows.append([
                    _dim_text(str(i)), _sym_text(str(s.get("symbol") or "")),
                    Text(f"{score:.3f}", style=score_style),
                    Text(signal_type, style=CYAN),
                    Text(f"{strength:.2f}", style=WHITE),
                    Text(f"{confidence:.2f}", style=WHITE),
                    Text("▲ LONG", style=f"bold {GAIN_HI}"),
                ])
            self._signals_last_strategy_name = strategy_name
            self._signals_last_count = len(sigs)
            self._signals_last_loaded_at = datetime.now().strftime("%H:%M:%S")
            publish_agent_signal_snapshot(
                {
                    "account_id": str(getattr(self, "_current_account_id", "account_1") or "account_1"),
                    "strategy_id": str((self._strategy_account_binding(getattr(self, "_current_account_id", "account_1")) or {}).get("strategy_id") or strategy_registry.default_strategy_for_account(getattr(self, "_current_account_id", "account_1"))),
                    "strategy_name": strategy_name,
                    "context_date": str(getattr(self.md, "latest", datetime.now().strftime("%Y-%m-%d")) or datetime.now().strftime("%Y-%m-%d")),
                    "regime": _regime,
                    "signals": list(sigs),
                }
            )
            self.call_from_thread(self._render_signals_table_payload, cache_key, cols, rows, len(sigs))
            self.call_from_thread(self._update_signals_table_title)
            self.call_from_thread(self._set_status,
                f"Signals loaded: {len(sigs)} │ {strategy_name} │ Session: {self.md.latest}")
        except Exception as e:
            self.call_from_thread(self._set_status, f"Signal error: {e}")
    def _render_signals_table_payload(
        self,
        cache_key: str,
        cols: list[tuple[str, str]],
        rows: list[list[Text]],
        count: int,
    ) -> None:
        self._signals_table_cache_key = cache_key
        self._signals_table_cache_payload = (cols, rows, count)
        self._set_signals_table(cols, rows)
    def _set_signals_table_loading(self, strategy_name: str = "") -> None:
        label = str(strategy_name or self._signals_last_strategy_name or self._strategy_name_for_account(getattr(self, "_current_account_id", "account_1"))).strip()
        suffix = f" — {label}" if label else ""
        try:
            self.query_one("#signals-table-title", Static).update(f"SIGNALS{suffix} — Loading...  |  R/F force reload")
        except Exception:
            pass
    def _update_signals_table_title(self) -> None:
        strategy_name = str(self._signals_last_strategy_name or self._strategy_name_for_account(getattr(self, "_current_account_id", "account_1"))).strip()
        loaded_at = str(self._signals_last_loaded_at or "not loaded")
        count = int(self._signals_last_count or 0)
        title = f"SIGNALS — {strategy_name}  |  {count} loaded  |  {loaded_at}  |  R/F force reload"
        try:
            self.query_one("#signals-table-title", Static).update(title)
        except Exception:
            pass
    def _set_signals_table(self, cols, rows):
        dt = self.query_one("#dt-signals", DataTable)
        dt.clear(columns=True)
        for label, key in cols:
            dt.add_column(label, key=key)
        for row in rows:
            dt.add_row(*row)
    @work(thread=True)
    def _load_kalimati_async(self) -> None:
        self.call_from_thread(self._set_status, "Loading commodity prices...")
        try:
            rows, status = refresh_kalimati()
            self._kalimati_rows = rows
            self._kalimati_status = status
        except Exception as e:
            self._kalimati_rows = get_kalimati_display_rows()
            self._kalimati_status = f"Rates & Commodities: {e}"
        self.call_from_thread(self._populate_kalimati)
    @work(thread=True)
    def _load_macro_rates_async(self, force: bool = False) -> None:
        now_ts = time.monotonic()
        if not force and (now_ts - self._last_macro_rates_fetch_at) < 1800:
            return
        metals = _fetch_gold_silver_prices()
        noc = _fetch_noc_fuel_prices()
        forex_rows = _fetch_nrb_forex_rates(("USD", "EUR", "GBP", "INR", "CNY", "JPY", "AUD", "CAD"))
        indicator_rows: list[dict] = []
        indicator_history = _load_macro_indicator_history()
        snapshot_ts = datetime.utcnow().isoformat()

        if metals:
            gold = float(metals.get("gold_per_tola") or 0)
            silver = float(metals.get("silver_per_tola") or 0)
            if gold > 0:
                # Prefer DB-stored CHG% (cross-session, today vs yesterday)
                # Fall back to in-session memory tracker if DB has no previous day yet
                db_gold_chg_pct = metals.get("gold_chg_pct")
                db_gold_chg_abs = metals.get("gold_chg_abs")
                if db_gold_chg_pct is not None:
                    gold_change = db_gold_chg_abs
                    gold_change_pct = db_gold_chg_pct
                else:
                    gold_change, gold_change_pct = _apply_indicator_history_change(
                        indicator_history, key="gold_per_tola",
                        value=gold, timestamp=snapshot_ts,
                    )
                indicator_rows.append(
                    {
                        "item": "Gold / Tola",
                        "group": "Metals",
                        "value": gold,
                        "unit": "NPR/tola",
                        "change": gold_change,
                        "change_pct": gold_change_pct,
                        "source": "FENEGOSIDA",
                    }
                )
            if silver > 0:
                db_silver_chg_pct = metals.get("silver_chg_pct")
                db_silver_chg_abs = metals.get("silver_chg_abs")
                if db_silver_chg_pct is not None:
                    silver_change = db_silver_chg_abs
                    silver_change_pct = db_silver_chg_pct
                else:
                    silver_change, silver_change_pct = _apply_indicator_history_change(
                        indicator_history, key="silver_per_tola",
                        value=silver, timestamp=snapshot_ts,
                    )
                indicator_rows.append(
                    {
                        "item": "Silver / Tola",
                        "group": "Metals",
                        "value": silver,
                        "unit": "NPR/tola",
                        "change": silver_change,
                        "change_pct": silver_change_pct,
                        "source": "FENEGOSIDA",
                    }
                )

        for symbol, label in (("CL=F", "WTI Crude"), ("BZ=F", "Brent Crude")):
            quote = _fetch_yahoo_futures_price(symbol, label)
            if quote:
                indicator_rows.append(
                    {
                        "item": label,
                        "group": "Energy",
                        "value": float(quote.get("value") or 0),
                        "unit": quote.get("unit") or "USD",
                        "change_pct": quote.get("change_pct"),
                        "source": quote.get("source") or "Yahoo Finance",
                    }
                )

        if noc:
            for label, key, unit in (
                ("Petrol", "petrol", "NPR/L"),
                ("Diesel", "diesel", "NPR/L"),
                ("Kerosene", "kerosene", "NPR/L"),
                ("LPG", "lpg", "NPR/cylinder"),
            ):
                value = float(noc.get(key) or 0)
                if value <= 0:
                    continue
                change, change_pct = _apply_indicator_history_change(
                    indicator_history,
                    key=f"noc_{key}",
                    value=value,
                    timestamp=snapshot_ts,
                )
                indicator_rows.append(
                    {
                        "item": label,
                        "group": "NOC",
                        "value": value,
                        "unit": unit,
                        "change": change,
                        "change_pct": change_pct,
                        "source": f"NOC {noc.get('date_bs') or ''}".strip(),
                    }
                )

        _save_macro_indicator_history(indicator_history)
        rates = {
            "metals": metals,
            "noc": noc,
            "indicators": indicator_rows,
            "forex_rows": forex_rows,
        }
        self._last_macro_rates_fetch_at = now_ts
        self._macro_rates = rates
        self.call_from_thread(self._populate_kalimati)
    def _populate_kalimati(self) -> None:
        rows = self._kalimati_rows
        status = self._kalimati_status
        filtered_rows, filtered_indicators, filtered_forex = self._get_filtered_rates_payload()

        # ── Status bar ────────────────────────────────────────────────────────
        with_chg = [r for r in filtered_rows if r.get("change_pct") is not None]
        gainers = sum(1 for r in with_chg if r["change_pct"] > 0)
        losers  = sum(1 for r in with_chg if r["change_pct"] < 0)
        date_str = rows[0]["date"] if rows else "—"
        bar_text = (
            f"[bold #ffaf00] RATES & COMMODITIES [/]"
            f"[#444444] │ [/]"
            f"[#888888]{len(filtered_rows)}/{len(rows)} commodities[/]  "
            f"[#444444] │ [/]"
            f"[#888888]{len(filtered_indicators)} macro[/]  "
            f"[#888888]{len(filtered_forex)} forex[/]"
            f"[#444444] │ [/]"
            f"[#00C853]▲ {gainers}[/]  [#ff4444]▼ {losers}[/]"
            f"[#444444] │ [/][#888888]{date_str}[/]"
            f"[#444444] │ [/][#555555]{status}[/]"
        )
        self.query_one("#kalimati-status-bar", Static).update(bar_text)

        self.query_one("#kalimati-left-title", Static).update(
            f"KALIMATI COMMODITIES [{len(filtered_rows)}]"
        )
        self.query_one("#kalimati-right-title", Static).update(
            f"GLOBAL RATES & PRICES [{len(filtered_indicators) + len(filtered_forex)}]"
        )
        self.query_one("#macro-top-title", Static).update(
            f"METALS, ENERGY & NOC [{len(filtered_indicators)}]"
        )
        self.query_one("#macro-forex-title", Static).update(
            f"FOREX RATES [{len(filtered_forex)}]"
        )

        # ── Top movers bar (only shown if we have change data) ────────────────
        if with_chg:
            top_gain = sorted(with_chg, key=lambda r: r["change_pct"], reverse=True)[:5]
            top_lose = sorted(with_chg, key=lambda r: r["change_pct"])[:5]
            parts: list[str] = ["[#555555] MOVERS [/]"]
            for r in top_gain:
                nm = r["name_english"].split("(")[0].strip()[:14]
                parts.append(f"[bold #00C853]▲[/][#00C853]{nm} +{r['change_pct']:.1f}%[/]")
            parts.append("[#333333] ┃ [/]")
            for r in top_lose:
                nm = r["name_english"].split("(")[0].strip()[:14]
                parts.append(f"[bold #ff4444]▼[/][#ff4444]{nm} {r['change_pct']:.1f}%[/]")
            movers_markup = "  ".join(parts)
        else:
            # No prev-day data yet — show price stats instead
            if rows:
                most_exp = max(rows, key=lambda r: r["avg"])
                cheapest = min(rows, key=lambda r: r["avg"])
                avg_price = sum(r["avg"] for r in rows) / len(rows)
                movers_markup = (
                    f"[#555555] MARKET STATS [/]  "
                    f"[#888888]Avg Price:[/] [#ffaf00]{avg_price:.2f}[/]  "
                    f"[#555555]│[/]  "
                    f"[#888888]Most Expensive:[/] [#ffaf00]{most_exp['name_english'].split('(')[0].strip()[:20]}[/] "
                    f"[bold #ffaf00]{most_exp['avg']:,.0f}[/]  "
                    f"[#555555]│[/]  "
                    f"[#888888]Cheapest:[/] [#00C853]{cheapest['name_english'].split('(')[0].strip()[:20]}[/] "
                    f"[bold #00C853]{cheapest['avg']:,.0f}[/]  "
                    f"[#444444]│  Price change available after 2nd fetch[/]"
                )
            else:
                movers_markup = "[#444444] Fetching commodities data...[/]"
        self.query_one("#kalimati-movers-bar", Static).update(movers_markup)

        # ── Kalimati table ────────────────────────────────────────────────────
        dt = self.query_one("#dt-kalimati", DataTable)
        dt.clear(columns=True)

        dt.add_column("COMMODITY", key="name", width=26)
        dt.add_column("U",         key="unit", width=4)
        dt.add_column("AVG",    key="avg",  width=9)
        dt.add_column("CHANGE", key="chg",  width=9)
        dt.add_column("CHG %",  key="pct",  width=7)
        dt.add_column("MIN",    key="min",  width=9)
        dt.add_column("MAX",    key="max",  width=9)
        dt.add_column("▕  RANGE  ▏", key="rng", width=12)

        def _range_bar(mn: float, mx: float, avg: float, width: int = 10) -> Text:
            """Show where avg sits within the [min, max] band."""
            if mx <= mn:
                return Text("▕" + "─" * width + "▏", style=LABEL)
            pos = int((avg - mn) / (mx - mn) * width)
            pos = max(0, min(width - 1, pos))
            bar = "░" * pos + "█" + "░" * (width - pos - 1)
            # Colour: green if avg > midpoint, red if below
            mid = (mn + mx) / 2
            color = GAIN if avg >= mid else LOSS
            return Text(f"▕{bar}▏", style=color)

        def _chg_text(chg: float | None, pct: float | None) -> tuple[Text, Text]:
            """Return (change_text, pct_text) coloured appropriately."""
            if chg is None or pct is None:
                return Text("   ─", style=LABEL), Text("  ─", style=LABEL)
            if pct > 0:
                style, arrow = GAIN_HI if pct > 5 else GAIN, "▲"
            elif pct < 0:
                style, arrow = LOSS_HI if pct < -5 else LOSS, "▼"
            else:
                style, arrow = LABEL, "─"
            chg_str = f"{arrow}{abs(chg):,.1f}"
            pct_str = f"{'+' if pct > 0 else ''}{pct:.1f}%"
            return Text(chg_str, style=style), Text(pct_str, style=style)

        for r in filtered_rows:
            pct = r.get("change_pct")
            chg_t, pct_t = _chg_text(r.get("change"), pct)

            name = r["name_english"]
            is_nepali = any(ord(c) > 0x900 for c in name[:6])
            if is_nepali:
                name_text = Text(name[:25], style=f"italic {LABEL}")
            elif pct is not None and abs(pct) >= 5:
                name_text = Text(name[:25], style=f"bold {WHITE}")
            else:
                name_text = Text(name[:25], style=WHITE)

            dt.add_row(
                name_text,
                Text(r["unit"][:4], style=LABEL),
                Text(f"{r['avg']:>8,.1f}", style=f"bold {AMBER}"),
                chg_t,
                pct_t,
                Text(f"{r['min']:>8,.1f}", style=DIM),
                Text(f"{r['max']:>8,.1f}", style=DIM),
                _range_bar(r["min"], r["max"], r["avg"], width=10),
            )

        # ── Macro indicators ──────────────────────────────────────────────────
        dt_macro = self.query_one("#dt-macro", DataTable)
        dt_macro.clear(columns=True)
        dt_macro.add_column("ITEM", key="item", width=16)
        dt_macro.add_column("GROUP", key="group", width=9)
        dt_macro.add_column("VALUE", key="value", width=14)
        dt_macro.add_column("CHG %", key="pct", width=8)
        dt_macro.add_column("SOURCE", key="source", width=16)

        for row in filtered_indicators:
            value = float(row.get("value") or 0)
            unit = str(row.get("unit") or "")
            if unit.startswith("NPR"):
                value_text = Text(_format_compact_npr(value), style=f"bold {AMBER}")
            else:
                value_text = Text(f"{value:,.2f} {unit}".strip(), style=f"bold {AMBER}")

            pct = row.get("change_pct")
            if pct is None:
                pct_text = Text("—", style=LABEL)
            elif float(pct) > 0:
                pct_text = Text(f"+{float(pct):.2f}%", style=GAIN)
            elif float(pct) < 0:
                pct_text = Text(f"{float(pct):.2f}%", style=LOSS)
            else:
                pct_text = Text("0.00%", style=LABEL)

            dt_macro.add_row(
                Text(str(row.get("item") or "")[:16], style=WHITE),
                Text(str(row.get("group") or "")[:9], style=CYAN),
                value_text,
                pct_text,
                Text(str(row.get("source") or "")[:16], style=DIM),
            )

        # ── Forex table ───────────────────────────────────────────────────────
        dt_forex = self.query_one("#dt-forex", DataTable)
        dt_forex.clear(columns=True)
        dt_forex.add_column("CCY", key="ccy", width=6)
        dt_forex.add_column("NAME", key="name", width=18)
        dt_forex.add_column("BUY", key="buy", width=10)
        dt_forex.add_column("SELL", key="sell", width=10)
        dt_forex.add_column("CHG %", key="pct", width=8)
        dt_forex.add_column("UNIT", key="unit", width=6)
        dt_forex.add_column("SOURCE", key="source", width=8)

        for row in filtered_forex:
            buy = float(row.get("buy_rate") or 0)
            sell = float(row.get("sell_rate") or 0)
            pct = row.get("change_pct")
            if pct is None:
                pct_text = Text("—", style=LABEL)
            elif float(pct) > 0:
                pct_text = Text(f"+{float(pct):.2f}%", style=GAIN)
            elif float(pct) < 0:
                pct_text = Text(f"{float(pct):.2f}%", style=LOSS)
            else:
                pct_text = Text("0.00%", style=LABEL)
            dt_forex.add_row(
                Text(str(row.get("currency_code") or ""), style=f"bold {CYAN}"),
                Text(str(row.get("currency_name") or "")[:18], style=WHITE),
                Text(f"{buy:,.2f}", style=WHITE),
                Text(f"{sell:,.2f}", style=WHITE),
                pct_text,
                Text(str(row.get("unit") or 1), style=LABEL),
                Text(str(row.get("source") or "")[:8], style=DIM),
            )
    def _get_filtered_rates_payload(self) -> tuple[list[dict], list[dict], list[dict]]:
        query = self._rates_search_query.strip().lower()
        filtered_rows = [
            r for r in self._kalimati_rows
            if not query
            or query in str(r.get("name_english") or "").lower()
            or query in str(r.get("unit") or "").lower()
        ]
        macro_data = self._macro_rates or {}
        indicator_rows = list(macro_data.get("indicators") or [])
        forex_rows = list(macro_data.get("forex_rows") or [])
        filtered_indicators = [
            row for row in indicator_rows
            if not query
            or query in str(row.get("item") or "").lower()
            or query in str(row.get("group") or "").lower()
            or query in str(row.get("source") or "").lower()
        ]
        filtered_forex = [
            row for row in forex_rows
            if not query
            or query in str(row.get("currency_code") or "").lower()
            or query in str(row.get("currency_name") or "").lower()
            or query in str(row.get("source") or "").lower()
        ]
        return filtered_rows, filtered_indicators, filtered_forex
    @work(thread=True)
    def _load_osint_async(self) -> None:
        from apps.tui.dashboard_tui import _fetch_osint_stories
        self.call_from_thread(self._set_status, "Loading OSINT headlines...")
        self._osint_stories = _fetch_osint_stories(80)
        # Translate enough OSINT text to keep the ticker English-first.
        nepali_headlines = []
        nepali_indices = []
        nepali_summaries = []
        nepali_summary_indices = []
        for i, s in enumerate(self._osint_stories):
            hl = s.get("canonical_headline", "")
            summary = s.get("summary", "") or ""
            url = s.get("url", "") or ""
            is_nepali = hl and _contains_non_ascii(hl)
            has_english_summary = summary and not _contains_non_ascii(summary)
            has_slug = bool(_headline_fallback_from_url(url))
            if is_nepali and not has_english_summary and not has_slug:
                nepali_headlines.append(hl)
                nepali_indices.append(i)
            if summary and _contains_non_ascii(summary):
                nepali_summaries.append(summary)
                nepali_summary_indices.append(i)
        if nepali_headlines:
            self.call_from_thread(self._set_status,
                f"Translating {len(nepali_headlines)} Nepali headlines...")
            translated = _translate_batch(nepali_headlines)
            for idx, trans in zip(nepali_indices, translated):
                self._osint_stories[idx]["_translated"] = trans
        if nepali_summaries:
            translated_summaries = _translate_batch(nepali_summaries)
            for idx, trans in zip(nepali_summary_indices, translated_summaries):
                self._osint_stories[idx]["_translated_summary"] = trans
        self.call_from_thread(self._build_ticker)
        n = len(self._osint_stories)
        self.call_from_thread(
            self._set_status,
            f"OSINT headlines loaded: {n} stories │ Session: {self.md.latest}",
        )
    def _populate_news(self) -> None:
        stories = self._osint_stories
        query = self._news_search_query.strip()
        is_vector = bool(self._vector_search_results)

        # If we have vector search results, show those instead
        if query and is_vector:
            stories = self._vector_search_results

        self._news_visible_stories = list(stories)
        if not stories:
            self._news_selected_index = 0
        else:
            self._news_selected_index = max(0, min(self._news_selected_index, len(stories) - 1))

        # News list (OptionList — avoids DataTable cell-level positioning
        # which garbles Devanagari combining characters)
        ol = self.query_one("#news-list", OptionList)
        ol.clear_options()

        if stories:
            options: list[Option] = []
            for i, s in enumerate(stories, 1):
                ts_raw = s.get("first_reported_at", "")
                ts = ts_raw[11:16] if ts_raw else ""
                sev = s.get("severity", "")
                stype = str(s.get("story_type", "") or "").strip()
                src = s.get("source_name", "")
                src = src.replace(" (Nepali)", "").replace("(Nepal", "")[:14]
                headline = unicodedata.normalize('NFC', _news_display_headline(s))

                sev_style = SEVERITY_STYLE.get(sev, LABEL)
                sim = s.get("_similarity")
                hl_display = headline
                if sim is not None:
                    hl_display = f"[{sim:.0%}] {headline}"
                hl_display = _truncate_text(hl_display, 64)

                row = Text()
                row.append(f"{hl_display}\n", style=f"bold {WHITE}")
                row.append(f"{ts:5s}", style=LABEL)
                row.append("  ")
                row.append(sev.upper()[:4], style=sev_style)
                row.append("  ")
                row.append(src, style=DIM)
                if stype:
                    row.append("  •  ", style=LABEL)
                    row.append(_truncate_text(stype, 10), style=TYPE_STYLE.get(stype, CYAN))
                options.append(Option(row, id=f"news-{i}"))
            ol.add_options(options)
            if self._news_selected_index < len(stories):
                ol.highlighted = self._news_selected_index
        else:
            ol.add_option(Option(Text("No stories available right now.", style=LABEL)))

        # Summary bar
        brief_bar = self.query_one("#news-brief-bar", Static)
        n = len(stories)
        high_n = sum(1 for s in stories if s.get("severity") in ("high", "critical"))
        med_n = sum(1 for s in stories if s.get("severity") == "medium")
        types = {}
        for s in stories:
            t = s.get("story_type", "other")
            types[t] = types.get(t, 0) + 1
        type_parts = "  ".join(
            f"[{TYPE_STYLE.get(t, LABEL)}]{t}:{c}[/]"
            for t, c in sorted(types.items(), key=lambda x: -x[1])
        )
        search_info = ""
        if query and is_vector:
            search_info = f"[bold #ffaf00]  ⌕ VECTOR \"{query}\"  {n} results[/]   "
        elif query:
            search_info = f"[bold #ffaf00]  ⌕ \"{query}\"[/]   "
        brief_bar.update(Text.from_markup(
            f"[bold {AMBER}]◆ LIVE NEWS FEED[/]   "
            f"{search_info}"
            f"[#888888]Total:[/] [bold {WHITE}]{n}[/]   "
            f"[{LOSS_HI}]▲{high_n} high[/]   "
            f"[{YELLOW}]{med_n} medium[/]   "
            f"{type_parts}"
        ))
    @work(thread=True)
    def _do_vector_search(self, query: str) -> None:
        """Semantic search via OSINT embeddings API, with local text fallback."""
        from apps.tui.dashboard_tui import OSINT_BASE, OSINT_TIMEOUT, _requests
        self.call_from_thread(self._set_status, f"Vector search: \"{query}\"...")
        vector_ok = False
        try:
            if not OSINT_BASE:
                raise RuntimeError("OSINT disabled")  # fall through to local search
            r = _requests.post(
                f"{OSINT_BASE}/embeddings/search",
                json={"query": query, "top_k": 40, "hours": 8760, "min_similarity": 0.3},
                timeout=OSINT_TIMEOUT + 4,
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if results:
                stories = []
                for item in results:
                    stories.append({
                        "canonical_headline": item.get("title", ""),
                        "source_name": item.get("source_name", ""),
                        "first_reported_at": item.get("published_at", ""),
                        "severity": item.get("severity", ""),
                        "story_type": item.get("story_type", item.get("category", "")),
                        "summary": "",
                        "display_location": "",
                        "_similarity": item.get("similarity", 0),
                        "_story_id": item.get("story_id", ""),
                    })
                self._vector_search_results = stories
                vector_ok = True
                self.call_from_thread(self._populate_news)
                self.call_from_thread(
                    self._set_status,
                    f"Vector: {len(stories)} results for \"{query}\""
                )
        except Exception:
            pass

        # Fallback: local text filter on loaded stories
        if not vector_ok:
            terms = query.lower().split()
            filtered = []
            for s in self._osint_stories:
                haystack = " ".join([
                    s.get("canonical_headline", ""),
                    s.get("summary", "") or "",
                    s.get("_translated", "") or "",
                    s.get("source_name", "") or "",
                    s.get("story_type", "") or "",
                ]).lower()
                if all(t in haystack for t in terms):
                    filtered.append(s)
            self._vector_search_results = filtered
            self.call_from_thread(self._populate_news)
            self.call_from_thread(
                self._set_status,
                f"Text search: {len(filtered)} results for \"{query}\" (vector empty, local fallback)"
            )
    def _populate_market(self):
        self.query_one("#p-gainers", MarketPanel).set_data(
            [("SYMBOL", "sym"), ("PRICE", "price"), ("CHG%", "chg"), ("VOL", "vol")],
            [[_sym_text(r["symbol"]), _price_text(r["close"]),
              _chg_text(r["chg"]), _vol_text(r["volume"])]
             for _, r in self.md.gainers.iterrows()])
        self.query_one("#p-losers", MarketPanel).set_data(
            [("SYMBOL", "sym"), ("PRICE", "price"), ("CHG%", "chg"), ("VOL", "vol")],
            [[_sym_text(r["symbol"]), _price_text(r["close"]),
              _chg_text(r["chg"]), _vol_text(r["volume"])]
             for _, r in self.md.losers.iterrows()])
        self.query_one("#p-volume", MarketPanel).set_data(
            [("SYMBOL", "sym"), ("PRICE", "price"), ("CHG%", "chg"), ("VOL", "vol")],
            [[_sym_text(r["symbol"]), _price_text(r["close"]),
              _chg_text(r["chg"]), Text(_vol(r["volume"]), style=CYAN)]
             for _, r in self.md.vol_top.iterrows()])
        # 52-week
        rows_52 = []
        for _, r in self.md.near_hi.iterrows():
            rows_52.append([_sym_text(r["symbol"]), _price_text(r["close"]),
                            Text(f"{r['h']:.1f}", style=WHITE),
                            Text("▲ NEAR HIGH", style=f"bold {GAIN_HI}")])
        for _, r in self.md.near_lo.iterrows():
            rows_52.append([_sym_text(r["symbol"]), _price_text(r["close"]),
                            Text(f"{r['l']:.1f}", style=WHITE),
                            Text("▼ NEAR LOW", style=f"bold {LOSS_HI}")])
        if not rows_52:
            rows_52.append([_dim_text("—"), _dim_text("None today"), _dim_text(""), _dim_text("")])
        self.query_one("#p-52wk", MarketPanel).set_data(
            [("SYMBOL", "sym"), ("PRICE", "price"), ("REF", "ref"), ("SIGNAL", "sig")], rows_52)
        # Quotes
        quote_df = self.md.quotes if isinstance(self.md.quotes, pd.DataFrame) else pd.DataFrame()
        if not quote_df.empty and {"symbol", "ltp", "pc", "vol"}.issubset(quote_df.columns):
            qq = quote_df[quote_df["vol"] > 0].nlargest(60, "vol")
        else:
            qq = pd.DataFrame(columns=["symbol", "ltp", "pc", "vol"])
        ts_q = ""
        if not quote_df.empty and "ts" in quote_df.columns:
            ts_q = str(quote_df["ts"].iloc[0])[:16]
        self.query_one("#p-quotes", MarketPanel).update_title(f"5) LIVE QUOTES  {ts_q}")
        quote_rows = [
            [_sym_text(str(r["symbol"])), _price_text(r["ltp"]),
             _chg_text(r["pc"]), _vol_text(r["vol"])]
            for _, r in qq.iterrows()
        ]
        if not quote_rows:
            quote_rows = [[_dim_text("—"), _dim_text("No live quotes"), _dim_text(""), _dim_text("")]]
        self.query_one("#p-quotes", MarketPanel).set_data(
            [("SYMBOL", "sym"), ("LTP", "ltp"), ("CHG%", "chg"), ("VOL", "vol")],
            quote_rows)
    def _populate_portfolio_and_risk(self):
        book = self._website_book
        if book is not None and self.trade_mode == "paper":
            # Linked to nepsequant.com: the website's paper book is the book.
            self._stats = _stats_from_frames(
                self.md,
                book.portfolio_frame(),
                book.nav_frame(),
                book.trade_log_frame(),
                account_capital=book.starting_cash(),
                cash=book.cash(),
                prefer_book_price=True,
            )
        elif self._trading_engine or self.trade_mode == "paper":
            # Always use account-specific paper_portfolio.csv as the authoritative view.
            # This captures both manual trades and any engine trades persisted there.
            # The engine's tui_paper_portfolio.csv is used only as a fallback.
            _acc_dir = _account_dir(self._current_account_id)
            _paper_port = _acc_dir / "paper_portfolio.csv"
            if _paper_port.exists():
                self._stats = _compute_account_portfolio_stats(self.md, _acc_dir)
            elif self._trading_engine:
                self._stats = self._trading_engine.get_portfolio_stats()
            else:
                self._stats = _compute_portfolio_stats(self.md)
        elif self.trade_mode == "live":
            # Fetch TMS bundle once, cache for _populate_trades_full
            if self.tms_service:
                try:
                    self._set_status("TMS monitor  |  Scraping TMS pages...")
                    live_bundle = self.tms_service.executor.fetch_monitor_bundle()
                    self._tms_bundle = _merge_tms_bundle_with_cache(live_bundle)
                    h = self._tms_bundle.get("holdings", {})
                    items = h.get("items", []) if isinstance(h, dict) else h
                    health = self._tms_bundle.get("health", {})
                    login_req = _tms_health_flag(health, "login_required")
                    status_note = "using cached snapshots" if login_req else "live snapshots"
                    self._set_status(
                        f"TMS monitor  |  Bundle: {len(items)} holdings, "
                        f"login_required={login_req}  |  {status_note}"
                    )
                except Exception as e:
                    self._tms_bundle = _load_cached_tms_bundle()
                    cached_items = ((self._tms_bundle.get("holdings") or {}).get("items") or [])
                    self._set_status(
                        f"TMS fetch failed: {e}  |  cached holdings={len(cached_items)}"
                    )
            self._stats = self._compute_tms_portfolio_stats()
        else:
            self._stats = _compute_portfolio_stats(self.md)
        s = self._stats
        self._populate_portfolio_tab(s)
        self._populate_risk_tab(s)
    def _compute_tms_portfolio_stats(self) -> dict:
        """Compute portfolio stats from TMS broker data."""
        if not self.tms_service:
            # TMS not ready yet — show empty portfolio with message
            return {
                "positions": [], "total_cost": 0, "total_value": 0,
                "cash": 0, "nav": 0, "total_return": 0,
                "day_pnl": 0, "day_ret": 0,
                "realized": 0, "unrealized": 0,
                "max_dd": 0, "dd_date": "", "peak_nav": 0,
                "nepse_ret": 0, "alpha": 0,
                "n_positions": 0, "sector_exposure": {},
                "top3_conc": 0, "winners": [], "losers": [],
                "age_0_5": 0, "age_6_15": 0, "age_16": 0,
                "trade_log": pd.DataFrame(), "nav_log": pd.DataFrame(),
            }
        try:
            bundle = getattr(self, '_tms_bundle', None)
            if not bundle:
                bundle = _merge_tms_bundle_with_cache(
                    self.tms_service.executor.fetch_monitor_bundle()
                )
                self._tms_bundle = bundle

            # Check if login is required (session expired)
            health = bundle.get("health", {})
            if _tms_health_flag(health, "login_required"):
                self._set_status("TMS session expired — showing last cached broker snapshot")

            holdings_data = bundle.get("holdings", {})
            holdings = holdings_data.get("items", []) if isinstance(holdings_data, dict) else holdings_data
            funds = bundle.get("funds", {})
            ltps = self.md.ltps()
            quote_map = {}
            if not self.md.quotes.empty:
                quote_map = {
                    str(r.symbol): {
                        "ltp": float(getattr(r, "ltp", 0) or 0),
                        "prev_close": float(getattr(r, "prev_close", 0) or 0),
                        "pc": float(getattr(r, "pc", 0) or 0),
                    }
                    for r in self.md.quotes.itertuples()
                }

            positions = []
            total_cost = total_value = 0.0
            total_prev_value = 0.0
            sector_exposure = {}

            for h in holdings:
                sym = str(h.get("symbol", "")).strip().upper()
                if not sym:
                    continue
                qty = int(h.get("tms_balance") or h.get("cds_total_balance") or h.get("quantity", 0))
                if qty <= 0:
                    continue
                # TMS doesn't expose avg cost; use close_price as proxy for entry
                entry = float(h.get("close_price") or h.get("avg_cost", 0))
                cur = ltps.get(sym) or float(h.get("ltp") or h.get("close_price") or entry)
                quote = quote_map.get(sym, {})
                prev_close = float(quote.get("prev_close") or h.get("close_price") or 0) or cur
                cost = entry * qty
                val = cur * qty
                pnl = val - cost
                ret = pnl / cost * 100 if cost else 0
                day_pnl = (cur - prev_close) * qty if prev_close > 0 else 0.0
                day_ret = ((cur - prev_close) / prev_close * 100) if prev_close > 0 else float(quote.get("pc") or 0)
                total_cost += cost
                total_value += val
                total_prev_value += prev_close * qty

                try:
                    from backend.backtesting.simple_backtest import get_symbol_sector
                    sec = get_symbol_sector(sym) or "Other"
                except Exception:
                    sec = "Other"
                sector_exposure[sec] = sector_exposure.get(sec, 0) + val

                positions.append({
                    "sym": sym, "qty": qty, "entry": entry, "cur": cur,
                    "cost": cost, "val": val, "pnl": pnl, "ret": ret,
                    "prev_close": prev_close, "day_pnl": day_pnl, "day_ret": day_ret,
                    "signal": _signal_label("TMS"), "date": "", "days": 0, "sector": sec,
                })

            cash = float(
                funds.get("collateral_available")
                or funds.get("available_trading_limit")
                or funds.get("collateral_total")
                or funds.get("available_cash")
                or funds.get("cash_collateral_amount")
                or 0
            )
            nav = cash + total_value
            prev_nav = cash + total_prev_value
            day_pnl_total = nav - prev_nav
            day_ret_total = (day_pnl_total / prev_nav * 100) if prev_nav > 0 else 0.0
            total_return = (nav - total_cost - cash) / (total_cost + cash) * 100 if (total_cost + cash) > 0 else 0

            positions.sort(key=lambda x: x["val"], reverse=True)
            top3_conc = sum(p["val"] for p in positions[:3]) / total_value * 100 if total_value > 0 else 0
            winners = [p for p in positions if p["pnl"] > 0]
            losers = [p for p in positions if p["pnl"] < 0]

            nepse_ret = 0.0
            if len(self.md.nepse) >= 2:
                nepse_ret = (self.md.nepse.iloc[0]["close"] - self.md.nepse.iloc[1]["close"]) / self.md.nepse.iloc[1]["close"] * 100

            return {
                "positions": positions,
                "total_cost": total_cost, "total_value": total_value,
                "cash": cash, "nav": nav, "total_return": total_return,
                "day_pnl": day_pnl_total, "day_ret": day_ret_total,
                "realized": 0, "unrealized": total_value - total_cost,
                "max_dd": 0, "dd_date": "", "peak_nav": nav,
                "nepse_ret": nepse_ret, "alpha": total_return - nepse_ret,
                "n_positions": len(positions),
                "sector_exposure": sector_exposure,
                "top3_conc": top3_conc,
                "winners": winners, "losers": losers,
                "age_0_5": 0, "age_6_15": 0, "age_16": len(positions),
                "trade_log": pd.DataFrame(), "nav_log": pd.DataFrame(),
            }
        except Exception as e:
            self._set_status(f"TMS portfolio fetch failed: {e}")
            return _compute_portfolio_stats(self.md)
    def _load_profile_inputs(self) -> None:
        profile = _load_profile_config()
        try:
            self.query_one("#profile-inp-portfolio", Input).value = str(profile.get("portfolio_path") or "")
        except Exception:
            pass
        try:
            target_nav = profile.get("target_nav")
            self.query_one("#profile-inp-target-nav", Input).value = (
                f"{float(target_nav):,.2f}".replace(",", "") if isinstance(target_nav, (int, float)) else ""
            )
        except Exception:
            pass
        try:
            accounts = list(getattr(self, "_paper_accounts", []) or [])
            next_name = f"Account {len(accounts) + 1}"
            self.query_one("#profile-inp-account-name", Input).value = next_name
        except Exception:
            pass
    def _sync_signal_buttons(self) -> None:
        """Refresh signal picker button styles and active-display label."""
        active = getattr(self, "_active_signals", set())
        for _, sig_type in _SIGNAL_DEFS:
            try:
                btn = self.query_one(f"#{_sig_btn_id(sig_type)}", Button)
                if sig_type in active:
                    btn.add_class("signal-active")
                else:
                    btn.remove_class("signal-active")
            except Exception:
                pass
        try:
            label = ", ".join(s for _, s in _SIGNAL_DEFS if s in active) or "No signals selected"
            self.query_one("#signal-picker-active", Static).update(
                f"  [#555555]Active:[/]  [bold #d8e4f2]{label}[/]"
            )
        except Exception:
            pass
    def _set_strategy_form_from_payload(self, payload: Optional[dict]) -> None:
        if not payload:
            return
        config = dict(payload.get("config") or {})
        # Sync signal picker buttons
        active = set(str(s) for s in list(config.get("signal_types") or []))
        self._active_signals = active
        self._sync_signal_buttons()

        values = {
            "strategy-inp-name": str(payload.get("name") or ""),
            "strategy-inp-description": str(payload.get("description") or ""),
            "strategy-inp-holding-days": str(int(config.get("holding_days") or 0) or ""),
            "strategy-inp-rebalance": str(int(config.get("rebalance_frequency") or 0) or ""),
            "strategy-inp-max-positions": str(int(config.get("max_positions") or 0) or ""),
            "strategy-inp-stop-loss": str(config.get("stop_loss_pct") if config.get("stop_loss_pct") is not None else ""),
            "strategy-inp-trailing-stop": str(config.get("trailing_stop_pct") if config.get("trailing_stop_pct") is not None else ""),
            "strategy-inp-sector-limit": str(config.get("sector_limit") if config.get("sector_limit") is not None else ""),
        }
        for widget_id, value in values.items():
            try:
                self.query_one(f"#{widget_id}", Input).value = value
            except Exception:
                pass
        try:
            start_widget = self.query_one("#strategy-inp-backtest-start", Input)
            if not start_widget.value:
                start_widget.value = "2025-01-01"
        except Exception:
            pass
        try:
            end_widget = self.query_one("#strategy-inp-backtest-end", Input)
            if not end_widget.value:
                end_widget.value = str(getattr(self.md, "latest", datetime.now().strftime("%Y-%m-%d")) or datetime.now().strftime("%Y-%m-%d"))
        except Exception:
            pass
        try:
            cap_widget = self.query_one("#strategy-inp-backtest-capital", Input)
            if not cap_widget.value:
                cap_widget.value = "1000000"
        except Exception:
            pass
    def _render_strategy_backtest_summary(self) -> None:
        try:
            widget = self.query_one("#strategy-backtest-summary", Static)
        except Exception:
            return
        selected_id = str(getattr(self, "_selected_strategy_id", "") or "").strip()
        result = self._selected_strategy_chart_result(selected_id)
        if not result:
            widget.update(
                Text.from_markup(
                    f"[#8aa0b5]Backtest artifacts[/] data/strategy_registry/backtests/\n"
                    f"[#708091]Run a backtest from this tab, or select a strategy with a saved chartable artifact.[/]"
                )
            )
            return
        summary = dict(result.get("summary") or {})
        nepse = dict(result.get("nepse") or {})
        widget.update(
            Text.from_markup(
                f"[bold {WHITE}]Window[/] {result.get('window', {}).get('start')} → {result.get('window', {}).get('end')}   "
                f"[bold {WHITE}]Capital[/] {_npr_k(float(result.get('window', {}).get('capital') or 0.0))}\n"
                f"[bold {WHITE}]Strategy[/] {summary.get('total_return_pct', 0.0):+.2f}%   "
                f"[bold {WHITE}]NEPSE[/] {float(nepse.get('return_pct') or 0.0):+.2f}%   "
                f"[bold {WHITE}]Alpha[/] {float(summary.get('vs_nepse_pct_points') or 0.0):+.2f}pp   "
                f"[bold {WHITE}]Sharpe[/] {float(summary.get('sharpe_ratio', summary.get('sharpe')) or 0.0):.2f}   "
                f"[bold {WHITE}]MaxDD[/] {float(summary.get('max_drawdown_pct') or 0.0):+.2f}%   "
                f"[bold {WHITE}]Trades[/] {int(summary.get('trade_count', summary.get('total_trades')) or 0)}\n"
                f"[#8aa0b5]Saved[/] data/strategy_registry/backtests/{str(result.get('strategy', {}).get('id') or 'strategy')}_latest.json"
            )
        )
    def _strategy_backtest_status_cell(self, strategy_id: str, metrics: dict) -> Text:
        sid = str(strategy_id or "").strip()
        state = dict(((getattr(self, "_strategy_backtest_statuses", {}) or {}).get(sid) or {}))
        status = str(state.get("status") or "").upper()
        if status == "RUN":
            progress = state.get("progress_pct")
            if progress is not None:
                return Text(f"{int(progress):>3d}%", style=YELLOW)
            return Text("RUN", style=YELLOW)
        if status == "CHART":
            return Text("CHART", style=YELLOW)
        if status == "FAIL":
            return Text("FAIL", style=LOSS_HI)
        if metrics:
            return Text("OK", style=GAIN_HI)
        return Text("—", style=LABEL)
    def _populate_strategy_list(self) -> None:
        try:
            dt = self.query_one("#dt-strategy-list", DataTable)
        except Exception:
            return
        dt.clear(columns=True)
        for label, key, width in [
            ("NAME", "name", 8),
            ("SIG", "sig", 4),
            ("HOLD", "hold", 5),
            ("RET", "ret", 9),
            ("VS NP", "vs_np", 10),
            ("SHRP", "sharpe", 6),
            ("DD", "dd", 9),
            ("TRD", "trades", 4),
            ("WR", "win_rate", 6),
            ("BT", "backtest_status", 7),
        ]:
            dt.add_column(label, key=key, width=width)
        selected_index = 0
        for idx, strategy in enumerate(list(getattr(self, "_strategies", []) or [])):
            sid = str(strategy.get("id") or "")
            config = dict(strategy.get("config") or {})
            signals = list(config.get("signal_types") or [])
            metrics = self._strategy_saved_metrics(sid) or {}
            if sid == str(getattr(self, "_selected_strategy_id", "") or ""):
                selected_index = idx
            dt.add_row(
                Text(self._strategy_display_name(sid, str(strategy.get("name") or sid)), style=WHITE),
                Text(str(len(signals)), style=WHITE),
                Text(str(config.get("holding_days") or ""), style=WHITE),
                Text(f"{float(metrics.get('total_return_pct') or 0.0):+.2f}%" if metrics else "—", style=GAIN_HI if float(metrics.get("total_return_pct") or 0.0) >= 0 else LOSS_HI),
                Text(f"{float(metrics.get('total_return_pct') or 0.0) - float((metrics.get('nepse') or {}).get('return_pct') or 0.0):+.2f}pp" if metrics else "—", style=GAIN_HI if (float(metrics.get("total_return_pct") or 0.0) - float((metrics.get("nepse") or {}).get("return_pct") or 0.0)) >= 0 else LOSS_HI),
                Text(f"{float(metrics.get('sharpe_ratio') or 0.0):.2f}" if metrics else "—", style=WHITE),
                Text(f"{float(metrics.get('max_drawdown_pct') or 0.0):+.2f}%" if metrics else "—", style=WHITE),
                Text(str(int(metrics.get("trade_count") or 0)) if metrics else "—", style=WHITE),
                Text(f"{float(metrics.get('win_rate_pct') or 0.0):.0f}%" if metrics else "—", style=WHITE),
                self._strategy_backtest_status_cell(sid, metrics),
            )
        if self._strategies:
            try:
                dt.move_cursor(row=selected_index)
            except Exception:
                pass
        else:
            dt.add_row(_dim_text("No strategies"), Text(""), Text(""), Text(""), Text(""), Text(""), Text(""), Text(""), Text(""), Text(""))
        try:
            row_count = max(1, len(list(getattr(self, "_strategies", []) or [])))
            dt.styles.height = min(max(row_count + 3, 7), 10)
        except Exception:
            pass
    def _populate_strategies_tab(self) -> None:
        if self._website_strategies is not None:
            self._populate_website_strategies_tab()
            return
        if not getattr(self, "_strategies", None):
            self._load_strategies_registry()
        self._populate_strategy_list()
        selected = self._selected_strategy_payload()
        self._set_strategy_form_from_payload(selected)
        current_strategy = self._strategy_name_for_account(getattr(self, "_current_account_id", "account_1"))
        selected_account_strategy = self._strategy_name_for_account(getattr(self, "_selected_account_id", self._current_account_id))
        try:
            metrics = self._strategy_saved_metrics(str((selected or {}).get("id") or ""))
            metrics_line = ""
            if metrics:
                window = dict(metrics.get("window") or {})
                nepse = dict(metrics.get("nepse") or {})
                alpha = float(metrics.get("total_return_pct") or 0.0) - float(nepse.get("return_pct") or 0.0)
                metrics_line = (
                    f"\n[bold {WHITE}]Saved baseline[/] {window.get('start')} → {window.get('end')}   "
                    f"[bold {WHITE}]Growth[/] {float(metrics.get('total_return_pct') or 0.0):+.2f}%   "
                    f"[bold {WHITE}]vs NEPSE[/] {alpha:+.2f}pp   "
                    f"[bold {WHITE}]Sharpe[/] {float(metrics.get('sharpe_ratio') or 0.0):.2f}   "
                    f"[bold {WHITE}]MaxDD[/] {float(metrics.get('max_drawdown_pct') or 0.0):+.2f}%   "
                    f"[bold {WHITE}]Trades[/] {int(metrics.get('trade_count') or 0)}   "
                    f"[bold {WHITE}]Win rate[/] {float(metrics.get('win_rate_pct') or 0.0):.0f}%"
                )
            self.query_one("#strategy-summary", Static).update(
                Text.from_markup(
                    f"[bold {WHITE}]Selected strategy[/] {str((selected or {}).get('name') or self._selected_strategy_id)}   "
                    f"[bold {WHITE}]Current account[/] {current_strategy}   "
                    f"[bold {WHITE}]Selected account[/] {selected_account_strategy}\n"
                    f"[#8aa0b5]Storage[/] data/strategy_registry/{metrics_line}"
                )
            )
        except Exception:
            pass
        try:
            account_lines = []
            for account in list(getattr(self, "_paper_accounts", []) or []):
                aid = str(account.get("id") or "")
                account_lines.append(
                    f"{str(account.get('name') or aid)} → {strategy_registry.strategy_name(str(account.get('strategy_id') or ''))}"
                )
            note = "\n".join(account_lines) if account_lines else "No account bindings"
            self.query_one("#strategy-accounts-note", Static).update(
                Text.from_markup(f"[#8aa0b5]Bindings[/]\n{note}")
            )
        except Exception:
            pass
        self._render_strategy_backtest_summary()
    def _profile_runtime_snapshot(self, *, account_dir: Optional[Path] = None) -> dict:
        if account_dir:
            portfolio_path = account_dir / "paper_portfolio.csv"
            trade_log_path = account_dir / "paper_trade_log.csv"
            nav_log_path = account_dir / "paper_nav_log.csv"
            state_path = account_dir / "paper_state.json"
            portfolio = pd.read_csv(portfolio_path) if portfolio_path.exists() else pd.DataFrame(columns=PORTFOLIO_COLS)
            trade_log = pd.read_csv(trade_log_path) if trade_log_path.exists() else pd.DataFrame(columns=TRADE_LOG_COLS)
            nav_log = pd.read_csv(nav_log_path) if nav_log_path.exists() else pd.DataFrame(columns=NAV_LOG_COLS)
            state = load_runtime_state(str(state_path))
        elif self._website_book is not None:
            # linked: the account panel describes the shared nepsequant.com book
            book = self._website_book
            portfolio = book.portfolio_frame()
            nav_log = book.nav_frame()
            trade_log = book.trade_log_frame()
            state = {"cash": book.cash()}
        else:
            portfolio = load_port()
            nav_log = _load_nav_log()
            trade_log = _load_trade_log()
            state = load_runtime_state(str(PAPER_STATE_FILE))
        holdings = len(portfolio.index) if isinstance(portfolio, pd.DataFrame) else 0
        trade_count = len(trade_log.index) if isinstance(trade_log, pd.DataFrame) else 0
        nav_rows = len(nav_log.index) if isinstance(nav_log, pd.DataFrame) else 0
        cash = state.get("cash")
        if not isinstance(cash, (int, float)):
            cash = _load_manual_paper_cash(0.0, nav_log)
        nav_value = None
        if account_dir and hasattr(self, "md"):
            try:
                stats = _compute_account_portfolio_stats(self.md, account_dir)
                nav_value = float(stats.get("nav") or 0.0)
                cash = float(stats.get("cash") or cash)
            except Exception:
                nav_value = None
        elif not account_dir and isinstance(getattr(self, "_stats", None), dict) and self._stats:
            try:
                nav_value = float(self._stats.get("nav") or 0.0)
                cash = float(self._stats.get("cash") or cash)
            except Exception:
                nav_value = None
        if nav_value is None and isinstance(nav_log, pd.DataFrame) and not nav_log.empty and "NAV" in nav_log.columns:
            try:
                nav_value = float(nav_log.iloc[-1]["NAV"])
            except Exception:
                nav_value = None
        return {
            "holdings": holdings,
            "trades": trade_count,
            "nav_rows": nav_rows,
            "cash": float(cash),
            "nav": float(nav_value) if isinstance(nav_value, (int, float)) else (float(cash) + _portfolio_mark_value(portfolio)),
            "runtime_dir": str(TRADING_RUNTIME_DIR),
        }
    def _populate_paper_profile_panel(self, stats: Optional[dict] = None) -> None:
        snapshot = self._profile_runtime_snapshot()
        profile = _load_profile_config()
        current_id = str(getattr(self, "_current_account_id", "account_1") or "account_1")
        current_strategy = self._strategy_name_for_account(current_id)
        current_name = next(
            (str(account.get("name") or current_id) for account in getattr(self, "_paper_accounts", []) if str(account.get("id") or "") == current_id),
            current_id,
        )
        positions = list((stats or {}).get("positions") or [])
        held_syms = ", ".join(p.get("sym", "") for p in positions[:6] if p.get("sym"))
        if len(positions) > 6:
            held_syms += ", ..."
        if not held_syms:
            held_syms = "No holdings loaded"
        selected_id = str(getattr(self, "_selected_account_id", current_id) or current_id)
        selected_strategy = self._strategy_name_for_account(selected_id)
        selected_name = next(
            (str(account.get("name") or selected_id) for account in getattr(self, "_paper_accounts", []) if str(account.get("id") or "") == selected_id),
            selected_id,
        )
        mode_note = "Creating or activating an account swaps the full paper runtime"
        summary = Text.from_markup(
            f"[bold {WHITE}]Active[/] {current_name}   "
            f"[bold {WHITE}]Strategy[/] {current_strategy}   "
            f"[bold {WHITE}]Selected[/] {selected_name}   "
            f"[bold {WHITE}]Selected strategy[/] {selected_strategy}   "
            f"[bold {WHITE}]Holdings[/] {snapshot['holdings']}   "
            f"[bold {WHITE}]Trades[/] {snapshot['trades']}   "
            f"[bold {WHITE}]NAV rows[/] {snapshot['nav_rows']}   "
            f"[bold {WHITE}]NAV[/] {_npr_k(snapshot['nav'])}   "
            f"[bold {WHITE}]Cash[/] {_npr_k(snapshot['cash'])}\n"
            f"[#8aa0b5]Held symbols:[/] {held_syms}\n"
            f"[#708091]{mode_note}[/]"
        )
        try:
            self.query_one("#profile-summary", Static).update(summary)
        except Exception:
            return
        try:
            widget = self.query_one("#profile-inp-portfolio", Input)
            if not widget.value and profile.get("portfolio_path"):
                widget.value = str(profile.get("portfolio_path") or "")
        except Exception:
            pass
        try:
            target_nav = profile.get("target_nav")
            if target_nav:
                self.query_one("#profile-inp-target-nav", Input).value = f"{float(target_nav):,.2f}".replace(",", "")
        except Exception:
            pass
        try:
            self.query_one("#profile-shortcuts", Static).update(
                Text.from_markup(
                    f"[#708091]ACCOUNT KEYS[/]   "
                    f"[bold {AMBER}]N[/] NEW   "
                    f"[bold {AMBER}]A[/] ACTIVATE   "
                    f"[bold {AMBER}]V[/] SET NAV   "
                    f"[bold {AMBER}]W[/] WATCHLIST   "
                    f"[bold {AMBER}]H[/] HELP"
                )
            )
        except Exception:
            pass
        help_text = ""
        if self._account_help_visible:
            help_text = (
                "N  Create account from NAME + NAV + optional SEED file\n"
                "A  Activate selected account row\n"
                "V  Apply NAV field to active account\n"
                "W  Sync watchlist from active holdings\n"
                "H  Toggle command help\n"
                "Tip: select an account row first, then activate"
            )
        try:
            self.query_one("#account-help", Static).update(Text(help_text, style=LABEL))
        except Exception:
            pass
        self._populate_account_list()
    def _populate_account_list(self) -> None:
        try:
            dt = self.query_one("#dt-account-list", DataTable)
        except Exception:
            return
        dt.clear(columns=True)
        for label, key in [("ID", "id"), ("NAME", "name"), ("STRAT", "strategy"), ("HOLD", "hold"), ("NAV", "nav"), ("CASH", "cash"), ("STATUS", "status")]:
            dt.add_column(label, key=key)
        rows = []
        selected_index = 0
        for idx, account in enumerate(getattr(self, "_paper_accounts", []) or []):
            snap = self._account_snapshot(account)
            account_id = str(account.get("id") or "")
            status = "ACTIVE" if account_id == self._current_account_id else ("SELECTED" if account_id == self._selected_account_id else "")
            if account_id == self._selected_account_id:
                selected_index = idx
            rows.append(snap)
            dt.add_row(
                Text(account_id.replace("account_", "A"), style=DIM),
                Text(snap["name"], style=WHITE),
                Text(snap["strategy"], style=CYAN),
                Text(str(snap["holdings"]), style=WHITE),
                Text(_npr_k(snap["nav"]), style=AMBER),
                Text(_npr_k(snap["cash"]), style=WHITE),
                Text(status, style=f"bold {GAIN_HI}" if status == "ACTIVE" else CYAN if status == "SELECTED" else DIM),
            )
        if not rows:
            dt.add_row(_dim_text("—"), _dim_text("No accounts"), Text(""), Text(""), Text(""), Text(""), Text(""))
        else:
            try:
                dt.move_cursor(row=selected_index)
            except Exception:
                pass
    def _populate_portfolio_tab(self, s: dict):
        # NAV summary bar
        rc = _pnl_color(s["total_return"])
        gross_rc = _pnl_color(s.get("gross_return", s["total_return"]))
        alpha_c = _pnl_color(s["alpha"])
        mode_tag = self._display_nav_mode_tag()
        nav_parts = [
            mode_tag,
            f"[bold {AMBER}]NAV[/] [bold {WHITE}]{_npr_k(s['nav'])}[/]",
            f"[#888888]Cash[/] [bold {WHITE}]{_npr_k(s['cash'])}[/]",
            f"[#888888]Invested[/] [bold {WHITE}]{_npr_k(s['total_cost'])}[/]",
            f"[#888888]Day[/] [bold {_pnl_color(s.get('day_pnl', 0.0))}]{_npr_k(s.get('day_pnl', 0.0))}[/] "
            f"[{_pnl_color(s.get('day_ret', 0.0))}]{s.get('day_ret', 0.0):+.2f}%[/]",
            f"[#888888]Net[/] [bold {rc}]{s['total_return']:+.2f}%[/]",
            f"[#888888]Gross[/] [bold {gross_rc}]{s.get('gross_return', s['total_return']):+.2f}%[/]",
            f"[#888888]NEPSE[/] [{_pnl_color(s['nepse_ret'])}]{s['nepse_ret']:+.2f}%[/]",
            f"[#888888]Alpha[/] [bold {alpha_c}]{s['alpha']:+.2f}pts[/]",
            f"[#888888]MaxDD[/] [bold {LOSS_HI}]{s['max_dd']:.1f}%[/]",
            f"[#555555]Sig[/] [#888888]Showing full signal names[/]",
        ]
        self.query_one("#nav-summary", Static).update(
            Text.from_markup("   ".join(nav_parts)))

        # Holdings table
        dt = self.query_one("#dt-portfolio", DataTable)
        dt.clear(columns=True)
        for label, key in [("SYMBOL", "sym"), ("QTY", "qty"), ("ENTRY", "entry"),
                           ("LTP", "ltp"), ("DAY", "day"), ("DAY%", "day_rtn"),
                           ("P&L", "pnl"), ("RTN%", "rtn"),
                           ("DAYS", "days"), ("SIGNAL", "sig"), ("SECTOR", "sec")]:
            dt.add_column(label, key=key)
        for p in s["positions"]:
            dt.add_row(
                _sym_text(p["sym"]), Text(str(p["qty"]), style=WHITE),
                _dim_text(f"{p['entry']:.1f}"), _price_text(p["cur"]),
                _npr(p.get("day_pnl", 0.0)),
                _chg_text(p.get("day_ret", 0.0)),
                _npr(p["pnl"]), _chg_text(p["ret"]),
                Text(str(p["days"]) + "d", style=YELLOW if p["days"] > 30 else LABEL),
                _dim_text(p["signal"]), _dim_text(p["sector"]),
            )
        if not s["positions"]:
            dt.add_row(_dim_text("No positions"), *[Text("")] * 10)
        self._populate_paper_profile_panel(s)
    def _populate_risk_tab(self, s: dict):
        # Risk summary bar
        parts = [
            f"[bold {AMBER}]RISK DASHBOARD[/]",
            f"[#888888]Positions[/] [bold {WHITE}]{s['n_positions']}[/]",
            f"[#888888]MaxDD[/] [bold {LOSS_HI}]{s['max_dd']:.1f}%[/]",
            f"[#888888]Peak NAV[/] [bold {WHITE}]{_npr_k(s['peak_nav'])}[/]",
            f"[#888888]Top3 Conc[/] [bold {YELLOW}]{s['top3_conc']:.1f}%[/]",
            f"[#888888]Realized[/] [{_pnl_color(s['realized'])}]{_npr_k(s['realized'])}[/]",
            f"[#888888]Unrealized[/] [{_pnl_color(s['unrealized'])}]{_npr_k(s['unrealized'])}[/]",
            f"[#888888]Age[/] [bold {WHITE}]{s['age_0_5']}≤5d {s['age_6_15']}≤15d {s['age_16']}>15d[/]",
        ]
        self.query_one("#risk-summary", Static).update(
            Text.from_markup("   ".join(parts)))

        # Concentration table (sectors + positions)
        dt = self.query_one("#dt-concentration", DataTable)
        dt.clear(columns=True)
        for label, key in [("TYPE", "type"), ("NAME", "name"),
                           ("VALUE", "val"), ("WEIGHT%", "wt")]:
            dt.add_column(label, key=key)

        tv = s["total_value"] if s["total_value"] > 0 else 1
        # Top positions by weight
        for p in s["positions"][:5]:
            wt = p["val"] / tv * 100
            dt.add_row(
                Text("POSITION", style=CYAN),
                _sym_text(p["sym"]),
                Text(_npr_k(p["val"]), style=WHITE),
                Text(f"{wt:.1f}%", style=f"bold {YELLOW}" if wt > 25 else WHITE),
            )
        # Sectors
        for sec, val in sorted(s["sector_exposure"].items(), key=lambda x: -x[1]):
            wt = val / tv * 100
            dt.add_row(
                Text("SECTOR", style=PURPLE),
                Text(sec, style=WHITE),
                Text(_npr_k(val), style=WHITE),
                Text(f"{wt:.1f}%", style=f"bold {LOSS_HI}" if wt > 35 else WHITE),
            )

        # Winners / Losers table
        dt2 = self.query_one("#dt-winloss", DataTable)
        dt2.clear(columns=True)
        for label, key in [("", "tag"), ("SYMBOL", "sym"), ("P&L", "pnl"),
                           ("RTN%", "rtn"), ("DAYS", "days")]:
            dt2.add_column(label, key=key)

        for p in sorted(s["winners"], key=lambda x: -x["pnl"])[:5]:
            dt2.add_row(
                Text("▲ WIN", style=f"bold {GAIN_HI}"), _sym_text(p["sym"]),
                _npr(p["pnl"]), _chg_text(p["ret"]),
                _dim_text(f"{p['days']}d"),
            )
        for p in sorted(s["losers"], key=lambda x: x["pnl"])[:5]:
            dt2.add_row(
                Text("▼ LOSS", style=f"bold {LOSS_HI}"), _sym_text(p["sym"]),
                _npr(p["pnl"]), _chg_text(p["ret"]),
                _dim_text(f"{p['days']}d"),
            )
        if not s["winners"] and not s["losers"]:
            dt2.add_row(_dim_text("—"), _dim_text("No positions"), *[Text("")] * 3)
    def _populate_signals_workspace(self, force: bool = False) -> None:
        self._maybe_refresh_website_floor()
        cache_key = self._data_version()
        if (
            not force
            and cache_key == self._signals_workspace_cache_key
            and self._signals_workspace_cache_payload is not None
        ):
            self._render_signals_workspace_payload(cache_key, self._signals_workspace_cache_payload)
            return
        self._set_signals_workspace_loading()
        self._load_signals_workspace_async(cache_key)
    def _set_signals_workspace_loading(self) -> None:
        bar = self.query_one("#screener-status-bar", Static)
        bar.update(Text.from_markup(
            f"[bold {AMBER}]◆ SIGNALS WORKSPACE[/]   [#888888]Loading screener, calendar and sector view...[/]"
        ))
        self.query_one("#screener-list-title", Static).update("ACTIVE STOCKS — Loading...")
        self.query_one("#heatmap-content", Static).update(Text("  Loading sector data...", style=LABEL))
    @work(thread=True)
    def _load_signals_workspace_async(self, cache_key: str) -> None:
        try:
            calendar_cols = [
                ("SYMBOL", "sym"), ("BOOK CLOSE", "bc"), ("DAYS", "days"),
                ("CASH%", "cash"), ("BONUS%", "bonus"), ("BUY BY", "buy"),
            ]
            calendar_rows: list[list[Text]] = []
            now = datetime.now()
            if self.md.corp.empty:
                calendar_rows.append([_dim_text("—"), _dim_text("No upcoming events"), *[Text("")] * 4])
            else:
                for _, r in self.md.corp.iterrows():
                    bc = r["bookclose_date"]
                    days = (bc - now).days
                    cash = float(r.get("cash_dividend_pct") or 0)
                    bonus = float(r.get("bonus_share_pct") or 0)
                    buy_by = (bc - timedelta(days=5)).strftime("%Y-%m-%d")
                    uc = f"bold {YELLOW}" if days <= 7 else (YELLOW if days <= 14 else WHITE)
                    calendar_rows.append([
                        _sym_text(str(r["symbol"])),
                        Text(bc.strftime("%Y-%m-%d"), style=WHITE),
                        Text(f"{days}d", style=uc),
                        Text(f"{cash:.1f}%", style=f"bold {GAIN_HI}") if cash >= 5 else _dim_text("—"),
                        Text(f"{bonus:.1f}%", style=f"bold {GAIN_HI}") if bonus >= 10 else _dim_text("—"),
                        Text(buy_by, style=CYAN),
                    ])

            screener_cols = [
                ("SYMBOL", "sym"), ("SECTOR", "sec"), ("LTP", "ltp"),
                ("CHG%", "chg"), ("VOL", "vol"), ("VRAT", "vrat"),
                ("RANGE", "range"), ("TREND", "spark"),
            ]
            screener_rows: list[list[Text]] = []
            heatmap = Text("  Loading sector data...", style=LABEL)
            status_markup = (
                f"[bold {AMBER}]◆ SIGNALS WORKSPACE[/]   [#888888]Loading screener, calendar and sector view...[/]"
            )
            list_title = "ACTIVE STOCKS — Loading..."

            conn = None
            all_stocks: list[dict] = []
            sector_data: dict[str, dict] = {}
            MIN_VOL = 1000
            try:
                conn = _db()
                latest_date = conn.execute(
                    "SELECT MAX(date) FROM stock_prices WHERE symbol != 'NEPSE'"
                ).fetchone()[0]
                prev_date = None
                if latest_date:
                    prev_date = conn.execute(
                        "SELECT MAX(date) FROM stock_prices WHERE date < ? AND symbol != 'NEPSE'",
                        (latest_date,)
                    ).fetchone()[0]

                if latest_date and prev_date:
                    today_rows = conn.execute(
                        "SELECT symbol, open, high, low, close, volume FROM stock_prices "
                        "WHERE date=? AND symbol != 'NEPSE'",
                        (latest_date,)
                    ).fetchall()
                    today_rows = _dedupe_symbol_rows(today_rows)
                    prev_map = {
                        r[0]: float(r[1])
                        for r in _dedupe_symbol_rows(conn.execute(
                            "SELECT symbol, close FROM stock_prices WHERE date=?",
                            (prev_date,)
                        ).fetchall())
                    }
                    avg_vol_map = {
                        r[0]: float(r[1]) if r[1] else 0.0
                        for r in conn.execute(
                            "SELECT symbol, AVG(volume) FROM stock_prices "
                            "WHERE date > date(?, '-30 days') AND symbol != 'NEPSE' "
                            "GROUP BY symbol",
                            (latest_date,)
                        ).fetchall()
                    }

                    from backend.backtesting.simple_backtest import get_symbol_sector

                    for r in today_rows:
                        sym = r[0]
                        vol = int(r[5])
                        close = float(r[4])
                        prev = prev_map.get(sym, 0.0)
                        chg = (close - prev) / prev * 100 if prev > 0 else 0.0
                        sector = get_symbol_sector(sym) or "Other"
                        avg_v = avg_vol_map.get(sym, 0.0)
                        vol_ratio = vol / avg_v if avg_v > 0 else 0.0
                        stock = {
                            "sym": sym,
                            "sector": sector,
                            "ltp": close,
                            "chg": chg,
                            "vol": vol,
                            "vol_ratio": vol_ratio,
                            "open": float(r[1]),
                            "high": float(r[2]),
                            "low": float(r[3]),
                        }
                        if vol > 0:
                            sector_bucket = sector_data.setdefault(
                                sector,
                                {"total_chg": 0.0, "count": 0, "total_vol": 0, "stocks": []},
                            )
                            sector_bucket["total_chg"] += chg
                            sector_bucket["count"] += 1
                            sector_bucket["total_vol"] += vol
                            sector_bucket["stocks"].append(stock)
                        if vol >= MIN_VOL:
                            all_stocks.append(stock)

                    spark_data: dict[str, list[float]] = {}
                    top_syms = [s["sym"] for s in sorted(all_stocks, key=lambda x: -x["vol"])[:120]]
                    for sym in top_syms:
                        hist = conn.execute(
                            "SELECT close FROM stock_prices WHERE symbol=? "
                            "ORDER BY date DESC LIMIT 20",
                            (sym,),
                        ).fetchall()
                        if len(hist) >= 3:
                            spark_data[sym] = [float(r[0]) for r in reversed(hist)]

                    if sector_data:
                        sector_perf = []
                        total_vol_all = sum(d["total_vol"] for d in sector_data.values())
                        for sec, data in sector_data.items():
                            avg_chg = data["total_chg"] / data["count"] if data["count"] > 0 else 0.0
                            vol_wt = data["total_vol"] / total_vol_all * 100 if total_vol_all > 0 else 0.0
                            sector_perf.append((sec, avg_chg, data["count"], data["total_vol"], vol_wt))
                        sector_perf.sort(key=lambda x: -x[1])

                        heatmap = Text()
                        heatmap.append("\n", style="")
                        max_abs = max(abs(s[1]) for s in sector_perf) if sector_perf else 1
                        if max_abs == 0:
                            max_abs = 1
                        for sec, avg_chg, count, total_vol, vol_wt in sector_perf:
                            if avg_chg > 2:
                                fg = "#00ff7f"
                            elif avg_chg > 1:
                                fg = "#00cc60"
                            elif avg_chg > 0:
                                fg = "#66cc66"
                            elif avg_chg > -1:
                                fg = "#cc9933"
                            elif avg_chg > -3:
                                fg = "#cc4444"
                            else:
                                fg = "#ff4545"
                            n_filled = max(1, int(abs(avg_chg) / max_abs * 8))
                            blocks = "█" * n_filled + "░" * (8 - n_filled)
                            heatmap.append(f"  {blocks} ", style=fg)
                            heatmap.append(f"{avg_chg:+5.1f}%  ", style=f"bold {fg}")
                            heatmap.append(f"{sec}", style=WHITE)
                            heatmap.append(f"  {count}\n", style=DIM)
                        total_stocks = sum(d["count"] for d in sector_data.values())
                        n_up = sum(1 for s in all_stocks if s["chg"] > 0)
                        n_dn = sum(1 for s in all_stocks if s["chg"] < 0)
                        heatmap.append(f"\n  {total_stocks} stocks  {_vol(total_vol_all)} vol  ", style=LABEL)
                        heatmap.append(f"▲{n_up}", style=GAIN_HI)
                        heatmap.append(f" ▼{n_dn}\n", style=LOSS_HI)

                    all_stocks.sort(key=lambda x: -x["vol"])
                    for s in all_stocks[:100]:
                        spark = (
                            _render_sparkline(spark_data.get(s["sym"], []), width=12)
                            if s["sym"] in spark_data
                            else Text("", style=DIM)
                        )
                        vr = s["vol_ratio"]
                        if vr > 2.5:
                            vr_text = Text(f"{vr:.1f}x", style=f"bold {GAIN_HI}")
                        elif vr > 1.5:
                            vr_text = Text(f"{vr:.1f}x", style=YELLOW)
                        elif vr > 0:
                            vr_text = Text(f"{vr:.1f}x", style=DIM)
                        else:
                            vr_text = Text("—", style=DIM)
                        rng = s["high"] - s["low"]
                        rng_pct = rng / s["low"] * 100 if s["low"] > 0 else 0
                        rng_text = (
                            Text(f"{s['low']:.0f}-{s['high']:.0f}", style=DIM)
                            if rng_pct > 0
                            else Text("—", style=DIM)
                        )
                        sec_display = "—" if s["sector"] == "Other" else s["sector"][:14]
                        screener_rows.append([
                            _sym_text(s["sym"]),
                            _dim_text(sec_display),
                            _price_text(s["ltp"]),
                            _chg_text(s["chg"]),
                            _vol_text(s["vol"]),
                            vr_text,
                            rng_text,
                            spark,
                        ])

                    n_stocks = len(all_stocks)
                    n_up = sum(1 for s in all_stocks if s["chg"] > 0)
                    n_down = sum(1 for s in all_stocks if s["chg"] < 0)
                    n_sectors = len([s for s in sector_data if s != "Other"])
                    status_markup = (
                        f"[bold {AMBER}]◆ SIGNALS WORKSPACE[/]   "
                        f"[#888888]Active[/] [bold {WHITE}]{n_stocks}[/] [#555555](vol≥1K)[/]   "
                        f"[{GAIN_HI}]▲{n_up}[/]  [{LOSS_HI}]▼{n_down}[/]   "
                        f"[#888888]{n_sectors} sectors[/]   "
                        f"[#555555]Signals + screener merged  │  ENTER Lookup  │  / Command[/]"
                    )
                    list_title = (
                        f"ACTIVE STOCKS — {n_stocks}  │  Vol≥1K  │  Sorted by volume  │  VRAT=vol/20d avg"
                    )
            finally:
                if conn:
                    conn.close()

            # ── Broker floor signals ─────────────────────────────────────────
            broker_cols = BROKER_COLS
            broker_rows: list[list[Text]] = []
            top_broker_cols: list = []
            top_broker_rows: list[list[Text]] = []
            website_floor = getattr(self, "_website_floor", None)
            if website_floor is not None:
                # Linked: computed from nepsequant.com's floorsheet (no network here;
                # _refresh_website_floor downloads it in its own worker).
                broker_title, broker_rows, top_broker_cols, top_broker_rows = website_floor_panels(
                    website_floor.view, website_floor.error
                )
            else:
                broker_title = "BROKER FLOOR SIGNALS — python3 -m backend.quant_pro.data_scrapers.floorsheet_ingestion"
                try:
                    import sqlite3 as _sqlite3
                    from backend.quant_pro.database import get_db_path as _get_db_path
                    _bconn = _sqlite3.connect(str(_get_db_path()))
                    _bconn.row_factory = _sqlite3.Row
                    _latest_bdate, _brows, _top_rows = local_floor_records(_bconn)
                    if _latest_bdate:
                        broker_title = floor_title(_latest_bdate, _brows)
                        broker_rows = floor_signal_rows(_brows)
                        top_broker_cols = TOP_BROKER_COLS
                        top_broker_rows = top_broker_rows_for(_top_rows)
                    else:
                        broker_rows.append(message_row("Run floorsheet ingestion to populate broker data"))
                    _bconn.close()
                except Exception as _be:
                    broker_rows.append(message_row("Run floorsheet scraper to populate broker data"))
                    top_broker_cols = []
                    top_broker_rows = []

            payload = {
                "calendar_cols": calendar_cols,
                "calendar_rows": calendar_rows,
                "screener_cols": screener_cols,
                "screener_rows": screener_rows,
                "heatmap": heatmap,
                "status_markup": status_markup,
                "list_title": list_title,
                "broker_cols": broker_cols,
                "broker_rows": broker_rows,
                "broker_title": broker_title,
                "top_broker_cols": top_broker_cols,
                "top_broker_rows": top_broker_rows,
            }
            self.call_from_thread(self._render_signals_workspace_payload, cache_key, payload)
        except Exception as e:
            self.call_from_thread(self._set_status, f"Signals workspace error: {e}")
    def _render_signals_workspace_payload(self, cache_key: str, payload: dict) -> None:
        self._signals_workspace_cache_key = cache_key
        self._signals_workspace_cache_payload = payload

        dt_calendar = self.query_one("#dt-calendar", DataTable)
        dt_calendar.clear(columns=True)
        for label, key in payload["calendar_cols"]:
            dt_calendar.add_column(label, key=key)
        for row in payload["calendar_rows"]:
            dt_calendar.add_row(*row)

        dt_screener = self.query_one("#dt-screener", DataTable)
        dt_screener.clear(columns=True)
        for label, key in payload["screener_cols"]:
            dt_screener.add_column(label, key=key)
        for row in payload["screener_rows"]:
            dt_screener.add_row(*row)

        self.query_one("#heatmap-content", Static).update(payload["heatmap"])
        self.query_one("#screener-status-bar", Static).update(Text.from_markup(payload["status_markup"]))
        self.query_one("#screener-list-title", Static).update(payload["list_title"])

        self._render_broker_panels(payload)

    def _render_broker_panels(self, payload: dict) -> None:
        if getattr(self, "_website_floor", None) is not None:
            # Linked: always the newest nepsequant.com floorsheet the terminal holds.
            payload.update(self._website_floor_payload())
        self._signals_workspace_cache_payload = payload
        try:
            dt_broker = self.query_one("#dt-broker-floor", DataTable)
            dt_broker.clear(columns=True)
            for label, key, width in payload.get("broker_cols") or []:
                dt_broker.add_column(label, key=key, width=width)
            self._apply_broker_floor_filter()
            broker_title = payload.get("broker_title") or "BROKER FLOOR SIGNALS"
            self.query_one("#broker-floor-title", Static).update(broker_title)
        except Exception:
            pass

        try:
            dt_top = self.query_one("#dt-broker-top", DataTable)
            dt_top.clear(columns=True)
            top_cols = payload.get("top_broker_cols") or []
            top_rows = payload.get("top_broker_rows") or []
            for label, key, width in top_cols:
                dt_top.add_column(label, key=key, width=width)
            for row in top_rows:
                dt_top.add_row(*row)
            # Update title with latest date
            broker_title = payload.get("broker_title") or ""
            date_part = broker_title.split("|")[0].replace("BROKER FLOOR", "").strip() if "|" in broker_title else ""
            self.query_one("#broker-top-title", Static).update(
                f"TOP BROKERS BY VOLUME{('  ' + date_part) if date_part else ''}"
            )
        except Exception:
            pass
    # ── broker panels from nepsequant.com (linked terminals) ────────────────
    def _website_floor_payload(self) -> dict:
        floor = self._website_floor
        title, rows, top_cols, top_rows = website_floor_panels(floor.view, floor.error)
        return {
            "broker_cols": BROKER_COLS,
            "broker_rows": rows,
            "broker_title": title,
            "top_broker_cols": top_cols,
            "top_broker_rows": top_rows,
        }

    def _maybe_refresh_website_floor(self) -> None:
        floor = getattr(self, "_website_floor", None)
        if floor is not None and floor.due():
            self._refresh_website_floor()

    @work(thread=True, group="website-floor")
    def _refresh_website_floor(self, force: bool = False) -> None:
        """Download + compute in this worker; the UI thread only re-renders."""
        floor = getattr(self, "_website_floor", None)
        if floor is not None and floor.refresh(force=force):
            self.call_from_thread(self._apply_website_floor)

    def _apply_website_floor(self) -> None:
        payload = getattr(self, "_signals_workspace_cache_payload", None)
        if payload is not None:  # else the workspace render picks it up
            self._render_broker_panels(payload)

    def _apply_broker_floor_filter(self) -> None:
        """Re-populate broker floor table from cached payload using current filter."""
        payload = getattr(self, "_signals_workspace_cache_payload", None) or {}
        all_rows = list(payload.get("broker_rows") or [])
        f = getattr(self, "_broker_floor_filter", "all")
        if f == "circ":
            rows = [r for r in all_rows if str(r[1]) and ("CIRC" in str(r[1]) or "WASH" in str(r[1]))]
        elif f == "pump":
            rows = [r for r in all_rows if str(r[1]) and ("SELF" in str(r[1]) or "WASH" in str(r[1]))]
        elif f == "smart":
            rows = [r for r in all_rows if str(r[1]) and ("SMART" in str(r[1]) or "ACCUM" in str(r[1]))]
        else:
            rows = all_rows
        try:
            dt_broker = self.query_one("#dt-broker-floor", DataTable)
            dt_broker.clear()
            for row in rows:
                dt_broker.add_row(*row)
        except Exception:
            pass
    def _populate_calendar(self):
        dt = self.query_one("#dt-calendar", DataTable)
        dt.clear(columns=True)
        for label, key in [("SYMBOL", "sym"), ("BOOK CLOSE", "bc"), ("DAYS", "days"),
                           ("CASH%", "cash"), ("BONUS%", "bonus"), ("BUY BY", "buy")]:
            dt.add_column(label, key=key)
        now = datetime.now()
        if self.md.corp.empty:
            dt.add_row(_dim_text("—"), _dim_text("No upcoming events"), *[Text("")] * 4)
        else:
            for _, r in self.md.corp.iterrows():
                bc = r["bookclose_date"]; days = (bc - now).days
                cash = float(r.get("cash_dividend_pct") or 0)
                bonus = float(r.get("bonus_share_pct") or 0)
                buy_by = (bc - timedelta(days=5)).strftime("%Y-%m-%d")
                uc = f"bold {YELLOW}" if days <= 7 else (YELLOW if days <= 14 else WHITE)
                dt.add_row(
                    _sym_text(str(r["symbol"])),
                    Text(bc.strftime("%Y-%m-%d"), style=WHITE),
                    Text(f"{days}d", style=uc),
                    Text(f"{cash:.1f}%", style=f"bold {GAIN_HI}") if cash >= 5 else _dim_text("—"),
                    Text(f"{bonus:.1f}%", style=f"bold {GAIN_HI}") if bonus >= 10 else _dim_text("—"),
                    Text(buy_by, style=CYAN))
    def _render_hedge_panel(self) -> None:
        """Render the compact 4-line hedge strip in the Portfolio tab."""
        from backend.quant_pro.gold_hedge import get_gold_regime
        from backend.quant_pro.data_scrapers.gold_silver_ingestion import get_latest_nepal_metals
        from backend.quant_pro.database import get_db_path as _get_db_path
        import math as _math

        SEP   = "  [#2a3038]│[/]  "
        GOLD  = "#FFCA28"
        SILV  = "#90CAF9"

        def _ok(v):
            return v is not None and not (isinstance(v, float) and _math.isnan(v))

        def _chg(pct, abs_v):
            if not _ok(pct):
                return "[#3a3f45]—[/]"
            col  = "#2ebd6e" if pct >= 0 else "#e05050"
            sign = "+" if pct >= 0 else ""
            ab   = f" {sign}{abs_v:,.0f}" if _ok(abs_v) else ""
            return f"[{col}]{sign}{pct:.2f}%{ab}[/]"

        try:
            db      = str(_get_db_path())
            regime  = get_gold_regime(db)
            metals  = get_latest_nepal_metals(db)
            g_tola  = metals.get("gold_npr_tola") or 0
            s_tola  = metals.get("silver_npr_tola") or 0
            g_chg   = metals.get("gold_chg_pct")
            s_chg   = metals.get("silver_chg_pct")
            g_abs   = metals.get("gold_chg_abs")
            s_abs   = metals.get("silver_chg_abs")

            rname   = regime.get("regime", "no_data")
            mom     = regime.get("momentum_20d", 0) * 100
            rcol    = {"risk_off": "#E05050", "neutral": "#D1980B", "risk_on": "#2ebd6e"}.get(rname, "#606870")
            ricon   = {"risk_off": "▲ RISK-OFF", "neutral": "◉ NEUTRAL", "risk_on": "▼ RISK-ON"}.get(rname, "NO DATA")

            # ── Line 2: prices ────────────────────────────────────────────────
            info = (
                f"  [{rcol}]{ricon}[/]  [{rcol}]{mom:+.1f}%[/]"
                f"{SEP}[bold {GOLD}]GOLD[/]  NPR [bold]{g_tola:,.0f}[/]/tola  {_chg(g_chg, g_abs)}"
                f"{SEP}[bold {SILV}]SILVER[/]  NPR [bold]{s_tola:,.0f}[/]/tola  {_chg(s_chg, s_abs)}"
            )
            self.query_one("#hedge-info-bar", Static).update(info)

            # ── Line 3: recommendation ────────────────────────────────────────
            capital = getattr(self._trading_engine, "capital", None) if self._trading_engine else None
            hedge_on = self._hedge_enabled
            if not hedge_on:
                rec = f"  [#3a3f45]Hedge monitoring disabled — click [bold]● HEDGE ON[/] to enable[/]"
            elif rname == "risk_off":
                cap_str = f"  NPR {capital * 0.10:,.0f}" if (capital and capital > 0) else ""
                g_t = (capital * 0.10 * 0.70 / g_tola) if (capital and g_tola) else 0
                s_t = (capital * 0.10 * 0.30 / s_tola) if (capital and s_tola) else 0
                t_str = f"  →  [{GOLD}]{g_t:.2f}t[/] gold + [{SILV}]{s_t:.2f}t[/] silver" if (capital and capital > 0) else ""
                rec = f"  [bold #E05050]▲ HEDGE RECOMMENDED[/]  Withhold 10%{cap_str}{t_str}"
            elif rname == "neutral":
                rec = f"  [{rcol}]◉ Monitor — neutral regime, consider partial gold buffer[/]"
            else:
                rec = f"  [{rcol}]▼ Risk-on — full capital deployable, no hedge needed[/]"
            self.query_one("#hedge-rec-bar", Static).update(rec)

            # ── Line 4: hedge trade history summary ───────────────────────────
            trades = list(getattr(self, "_hedge_trade_log", []) or [])
            if trades:
                last = trades[-1]
                metal = str(last.get("metal", "GOLD"))
                col   = GOLD if metal == "GOLD" else SILV
                total_g = sum(float(t.get("total", 0)) for t in trades if t.get("metal") == "GOLD")
                total_s = sum(float(t.get("total", 0)) for t in trades if t.get("metal") == "SILVER")
                g_part  = f"[{GOLD}]GOLD NPR {total_g:,.0f}[/]" if total_g else ""
                s_part  = f"[{SILV}]SILVER NPR {total_s:,.0f}[/]" if total_s else ""
                invested = "  ".join(p for p in [g_part, s_part] if p)
                trade_bar = (
                    f"  [#3a3f45]HEDGE TRADES[/]  {len(trades)} recorded{SEP}"
                    f"Last  [{col}]{last.get('date','')}  {metal}  {float(last.get('tola',0)):.2f}t[/]"
                    f"{SEP}Total invested  {invested}"
                )
            else:
                trade_bar = (
                    f"  [#2a3038]HEDGE TRADES[/]  "
                    f"[#3a3f45]None recorded — physical gold/silver purchases logged here when executed[/]"
                )
            self.query_one("#hedge-trade-bar", Static).update(trade_bar)

        except Exception as e:
            try:
                self.query_one("#hedge-info-bar", Static).update(
                    f"  [#555555]Hedge data unavailable — {e}[/]"
                )
                self.query_one("#hedge-rec-bar", Static).update("")
                self.query_one("#hedge-trade-bar", Static).update("")
            except Exception:
                pass
    def _populate_trades_full(self):
        dt = self.query_one("#dt-trades-full", DataTable)
        dt.clear(columns=True)

        if self.trade_mode == "live" and not self.tms_service:
            # Live mode but TMS not connected yet
            for label, key in [("STATUS", "st")]:
                dt.add_column(label, key=key)
            dt.add_row(Text("Connecting to TMS — solve captcha in browser...", style=AMBER))
            self.query_one("#trades-title", Static).update("TMS TRADE BOOK  |  Connecting...")
            return

        if self.trade_mode == "live" and self.tms_service:
            # Live mode: show TMS broker trades
            for label, key in [("DATE", "dt"), ("ACTION", "act"), ("SYMBOL", "sym"),
                               ("QTY", "qty"), ("PRICE", "pr"), ("AMOUNT", "amt"),
                               ("STATUS", "st")]:
                dt.add_column(label, key=key)
            try:
                bundle = getattr(self, '_tms_bundle', None)
                if not bundle:
                    bundle = _merge_tms_bundle_with_cache(
                        self.tms_service.executor.fetch_monitor_bundle()
                    )
                    self._tms_bundle = bundle
                trades = bundle.get("trades_daily", {}).get("records", [])
                trades += bundle.get("trades_historic", {}).get("records", [])
                if trades:
                    for t in trades[:50]:
                        action = str(t.get("type", t.get("buy_sell", t.get("side", "")))).upper()
                        ac = GAIN_HI if "BUY" in action else (LOSS_HI if "SELL" in action else WHITE)
                        dt.add_row(
                            _dim_text(str(t.get("date", t.get("trade_date", "")))[:10]),
                            Text(action[:4], style=f"bold {ac}"),
                            _sym_text(str(t.get("symbol", t.get("script", "")))),
                            Text(str(t.get("quantity", t.get("qty", ""))), style=WHITE),
                            _price_text(float(t.get("rate", t.get("price", 0)) or 0)),
                            Text(str(t.get("amount", t.get("total", ""))), style=WHITE),
                            _dim_text(str(t.get("status", ""))[:12]),
                        )
                    self.query_one("#trades-title", Static).update(
                        f"TMS TRADE BOOK  |  {len(trades)} trades")
                else:
                    dt.add_row(_dim_text("No broker trades"), *[Text("")] * 6)
                    self.query_one("#trades-title", Static).update("TMS TRADE BOOK")
            except Exception as e:
                dt.add_row(_dim_text(f"TMS error: {e}"), *[Text("")] * 6)
            return

        # Paper mode: show local trade log
        for label, key in [("DATE", "dt"), ("ACTION", "act"), ("SYMBOL", "sym"),
                           ("SHARES", "sh"), ("PRICE", "pr"), ("FEES", "fees"),
                           ("P&L", "pnl"), ("RTN%", "rtn"), ("REASON", "rsn")]:
            dt.add_column(label, key=key)
        # Merge all trade log sources for the current account: engine (auto), paper (manual), tui (engine csv)
        _tl_sources: list[pd.DataFrame] = []
        if self._website_book is not None:
            _tl_sources.append(self._website_book.trade_log_frame())
        elif self._trading_engine:
            _etl = self._trading_engine.get_trade_log()
            if not _etl.empty:
                _tl_sources.append(_etl)
        _acc_tl_dir = _account_dir(self._current_account_id)
        for _fname in (() if self._website_book is not None else ("paper_trade_log.csv", "tui_paper_trade_log.csv")):
            _tl_path = _acc_tl_dir / _fname
            if _tl_path.exists():
                try:
                    _df = pd.read_csv(_tl_path)
                    if not _df.empty:
                        _tl_sources.append(_df)
                except Exception:
                    pass
        if self._website_book is not None:
            tl = _tl_sources[0]
        elif _tl_sources:
            tl = pd.concat(_tl_sources, ignore_index=True)
            _dedup_keys = [c for c in ("Date", "Action", "Symbol", "Shares", "Price") if c in tl.columns]
            if _dedup_keys:
                tl = tl.drop_duplicates(subset=_dedup_keys, keep="first")
            if "Date" in tl.columns:
                tl = tl.sort_values("Date").reset_index(drop=True)
        else:
            tl = _load_trade_log()
        if not tl.empty:
            for _, r in tl.iloc[::-1].iterrows():
                action = str(r.get("Action", ""))
                ac = GAIN_HI if action == "BUY" else (LOSS_HI if action == "SELL" else WHITE)
                pnl_v = float(r.get("PnL", 0) or 0)
                pnl_pct = float(r.get("PnL_Pct", 0) or 0)
                dt.add_row(
                    _dim_text(str(r.get("Date", ""))[:10]),
                    Text(action, style=f"bold {ac}"),
                    _sym_text(str(r.get("Symbol", ""))),
                    Text(str(int(r.get("Shares", 0))), style=WHITE),
                    _price_text(float(r.get("Price", 0))),
                    _dim_text(f"{float(r.get('Fees', 0)):.0f}"),
                    Text(f"{_npr_k(pnl_v)}", style=_pnl_color(pnl_v)) if pnl_v else _dim_text("—"),
                    _chg_text(pnl_pct * 100.0) if pnl_pct else _dim_text("—"),
                    _dim_text(str(r.get("Reason", ""))[:16]),
                )
        else:
            dt.add_row(_dim_text("No trades yet"), *[Text("")] * 8)
        # Append hedge trades (gold/silver physical purchases) at the top
        hedge_trades = list(reversed(getattr(self, "_hedge_trade_log", []) or []))
        GOLD_COL   = "#FFCA28"
        SILVER_COL = "#90CAF9"
        for ht in hedge_trades:
            metal = str(ht.get("metal", "GOLD"))
            col = GOLD_COL if metal == "GOLD" else SILVER_COL
            tola = float(ht.get("tola", 0) or 0)
            price = float(ht.get("price", 0) or 0)
            total = float(ht.get("total", 0) or 0)
            dt.add_row(
                _dim_text(str(ht.get("date", ""))[:10]),
                Text("HEDGE", style=f"bold {col}"),
                Text(metal, style=f"bold {col}"),
                Text(f"{tola:.2f}t", style=WHITE),
                _price_text(price),
                _dim_text("—"),
                Text(f"{_npr_k(total)}", style=col),
                _dim_text("—"),
                _dim_text(str(ht.get("reason", ""))[:16]),
            )
        if not tl.empty:
            buys = (tl["Action"] == "BUY").sum()
            sells = (tl["Action"] == "SELL").sum()
            total_pnl = tl["PnL"].sum() if "PnL" in tl.columns else 0
            wins = ((tl["PnL"] > 0) & (tl["Action"] == "SELL")).sum() if "PnL" in tl.columns else 0
            losses = ((tl["PnL"] < 0) & (tl["Action"] == "SELL")).sum() if "PnL" in tl.columns else 0
            wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            hedge_count = len(hedge_trades)
            hedge_note = f"  +{hedge_count} hedge" if hedge_count else ""
            self.query_one("#trades-title", Static).update(
                f"TRADE HISTORY  |  {buys} buys  {sells} sells{hedge_note}  |  "
                f"Win rate: {wr:.0f}%  |  Total P&L: {_npr_k(total_pnl)}")
    def _populate_lookup(self):
        sym = str(self.lookup_sym or "").strip().upper()
        if not sym:
            return
        tf = self.lookup_tf
        cache_key = (sym, tf, self._data_version())
        self._lookup_request_key = cache_key
        if cache_key in self._lookup_cache:
            self._render_lookup_payload(cache_key, self._lookup_cache[cache_key])
            return

        chart_w = self.query_one("#lookup-chart", Static)
        pane_width = max(60, int(getattr(chart_w.size, "width", 0) or (self.size.width - 6)))
        pane_height = max(14, int(getattr(chart_w.size, "height", 0) or 15))
        self._set_lookup_loading(sym, tf)
        self._load_lookup_async(sym, tf, cache_key[2], pane_width, pane_height)
    def _set_lookup_loading(self, sym: str, tf: str) -> None:
        tf_labels = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly", "I": "Intraday"}
        self.query_one("#lookup-title", Static).update(
            f"LOOKUP: {sym}  —  {tf_labels.get(tf, 'Daily')}  —  Loading..."
        )
        self.query_one("#lookup-header", Static).update(Text("  Loading price and report data...", style=LABEL))
        self.query_one("#lookup-chart", Static).update(Text("  Rendering chart...", style=LABEL))
        self.query_one("#lookup-stats", Static).update(Text("  Loading statistics...", style=LABEL))
        self.query_one("#lookup-report", Static).update(Text("  Building financial report...", style=LABEL))
        self.query_one("#lookup-intel-title", Static).update(f"CORPORATE INTELLIGENCE — {sym}")
        self.query_one("#lookup-intel", Static).update(Text("  Loading intelligence...", style=LABEL))
        self.query_one("#lookup-fin-title", Static).update(f"QUARTERLY FINANCIALS — {sym}")
        self.query_one("#lookup-ca-title", Static).update(f"CORPORATE ACTIONS — {sym}")
        for table_id in ("#dt-lookup", "#dt-lookup-fin", "#dt-lookup-ca"):
            self.query_one(table_id, DataTable).clear(columns=True)
    @work(thread=True)
    def _load_lookup_async(
        self,
        sym: str,
        tf: str,
        data_version: str,
        pane_width: int,
        pane_height: int,
    ) -> None:
        cache_key = (sym, tf, data_version)
        try:
            min_sessions = 2 if tf in {"D", "I"} else {"W": 10, "M": 18, "Y": 24}.get(tf, 2)
            _ensure_lookup_history(sym, min_sessions=min_sessions)
            limits = {"D": 120, "W": 365, "M": 730, "Y": 2500, "I": 60}
            det = self.md.detail(sym, limit=limits.get(tf, 120))
            if not det:
                payload = {"found": False, "sym": sym}
                self.call_from_thread(self._render_lookup_payload, cache_key, payload)
                return

            lat = det["lat"]
            chg = det["chg"]
            h = det["h"]
            ltp = self.md.ltps().get(sym, float(lat["close"]))
            tf_labels = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly", "I": "Intraday"}
            chart_source = h
            stats_source = h
            count_label = "sessions"
            intraday_note = ""

            if tf == "I":
                try:
                    from backend.trading.live_trader import is_market_open, now_nst
                    market_open = bool(is_market_open())
                    nst_now = now_nst()
                except Exception:
                    market_open = True
                    nst_now = datetime.utcnow() + timedelta(hours=5, minutes=45)

                if market_open:
                    intraday_rows, intraday_session_date, intraday_snapshots = _load_intraday_ohlcv(
                        sym,
                        preferred_session_date=_nst_today_str(),
                    )
                    if not intraday_rows.empty:
                        chart_source = intraday_rows
                        stats_source = intraday_rows
                        count_label = "bars"
                        if intraday_session_date:
                            intraday_note = f"  ({intraday_session_date} · {intraday_snapshots} snapshots)"
                    else:
                        intraday_note = "  (no intraday snapshots — showing daily)"
                else:
                    intraday_note = (
                        f"  (market closed at {nst_now.strftime('%H:%M')} NST — showing daily)"
                    )

            n_sessions = len(chart_source)
            title = (
                f"LOOKUP: {sym}  —  {tf_labels.get(tf, 'Daily')}  "
                f"({n_sessions} {count_label})  —  "
                f"D/W/M/Y/I to switch  —  Press L to change{intraday_note}"
            )
            header = Text.assemble(
                (f"  {sym}  ", f"bold {CYAN}"),
                ("LTP ", LABEL), (f"{ltp:.1f} ", f"bold {WHITE}"),
                _chg_text(chg),
                (f"   O {lat['open']:.1f}  H {lat['high']:.1f}  "
                 f"L {lat['low']:.1f}  Vol {_vol(lat['volume'])}", LABEL),
            )

            try:
                chart_rows = _resample_ohlcv(chart_source, tf).sort_values("date").reset_index(drop=True)
                ideal_width = 18 + max(12, len(chart_rows)) * 3
                cw = max(60, min(pane_width - 2, ideal_width))
                ch_h = max(12, pane_height - 1)
                chart = _render_candlestick_chart(chart_source, width=cw, height=ch_h, timeframe=tf)
            except Exception as e:
                chart = Text(f"  Chart error: {e}", style=LABEL)

            h_display = _resample_ohlcv(stats_source, tf) if tf in ("W", "M", "Y") else stats_source
            h_asc = h_display.sort_values("date")
            closes = h_asc["close"].tolist()
            vols = h_asc["volume"].tolist()
            avg_vol = sum(vols) / len(vols) if vols else 0
            hi_30 = float(h_display["high"].max())
            lo_30 = float(h_display["low"].min())
            avg_close = float(h_display["close"].mean())
            volatility = float(h_display["close"].pct_change().std() * 100) if len(h_display) > 2 else 0
            rng_pct = (hi_30 - lo_30) / lo_30 * 100 if lo_30 else 0

            stats = Text()
            stats.append("  MARKET SNAPSHOT\n", style=f"bold {AMBER}")
            stats.append("  HIGH ", style=LABEL); stats.append(f"{hi_30:.1f}   ", style=GAIN_HI)
            stats.append("LOW ", style=LABEL); stats.append(f"{lo_30:.1f}   ", style=LOSS_HI)
            stats.append("AVG ", style=LABEL); stats.append(f"{avg_close:.1f}\n", style=WHITE)
            stats.append("  AVG VOL ", style=LABEL); stats.append(f"{_vol(avg_vol)}   ", style=CYAN)
            stats.append("VOL ", style=LABEL); stats.append(f"{volatility:.2f}%   ", style=YELLOW)
            stats.append("RANGE ", style=LABEL); stats.append(f"{rng_pct:.1f}%\n", style=WHITE)
            stats.append("  PRICE ", style=LABEL)
            stats.append_text(_render_sparkline(closes, width=26))
            stats.append("\n", style=WHITE)
            stats.append("  VOLUME", style=LABEL)
            stats.append_text(_render_sparkline(vols, width=26))

            fin_title = f"QUARTERLY FINANCIALS — {sym}"
            fin_cols: list[tuple[str, str]] = []
            fin_rows: list[list[Text]] = []
            try:
                fin_report = build_stock_report(sym, current_price=ltp)
                report = _render_stock_report(fin_report)
                fin_cols = [
                    ("PERIOD", "period"), ("REVENUE", "revenue"), ("NET PROFIT", "net_profit"),
                    ("EPS", "eps"), ("BVPS", "book_value"),
                ]
                rows = fin_report.get("financial_rows") or []
                if rows:
                    for row in rows:
                        fin_rows.append([
                            _dim_text(row.get("period", "—")),
                            Text(str(row.get("revenue", "—")), style=CYAN),
                            Text(str(row.get("net_profit", "—")), style=WHITE),
                            Text(str(row.get("eps", "—")), style=GAIN_HI),
                            Text(str(row.get("book_value", "—")), style=WHITE),
                        ])
                else:
                    fin_rows.append([
                        _dim_text("—"),
                        _dim_text("No cached financials"),
                        _dim_text("Run scraper"),
                        _dim_text("—"),
                        _dim_text("—"),
                    ])
            except Exception as e:
                fin_report = {}
                report = Text(f"  Financial report error: {e}", style=LABEL)
                fin_cols = [("STATUS", "status")]
                fin_rows = [[_dim_text("Financial report unavailable")]]

            intel_title = f"CORPORATE INTELLIGENCE — {sym}"
            intel = _render_lookup_intelligence(fin_report, sym)

            tf_col_label = {"D": "DATE", "W": "WEEK", "M": "MONTH", "Y": "YEAR", "I": "TIME"}
            ohlcv_title = f"OHLCV — {sym} ({tf_labels.get(tf, 'Daily')})"
            ohlcv_cols = [
                (tf_col_label.get(tf, "DATE"), "dt"), ("OPEN", "o"), ("HIGH", "h"),
                ("LOW", "l"), ("CLOSE", "c"), ("VOL", "vol"), ("CHG%", "chg"),
            ]
            ohlcv_rows: list[list[Text]] = []
            h_sorted = h_display.sort_values("date").reset_index(drop=True)
            chg_map: dict[str, Optional[float]] = {}
            for i in range(len(h_sorted)):
                date_key = str(h_sorted.iloc[i]["date"])[:10]
                if i == 0:
                    chg_map[date_key] = None
                else:
                    prev_close = float(h_sorted.iloc[i - 1]["close"])
                    curr_close = float(h_sorted.iloc[i]["close"])
                    chg_map[date_key] = (curr_close - prev_close) / prev_close * 100 if prev_close > 0 else None
            for _, r in h_display.iterrows():
                date_key = str(r["date"])[:10]
                ct = _chg_text(chg_map.get(date_key)) if chg_map.get(date_key) is not None else _dim_text("—")
                ohlcv_rows.append([
                    _dim_text(date_key),
                    _price_text(r["open"]),
                    _price_text(r["high"]),
                    _price_text(r["low"]),
                    _price_text(r["close"]),
                    _vol_text(r["volume"]),
                    ct,
                ])

            ca = det["ca"]
            ca_title = f"CORPORATE ACTIONS — {sym}" if not ca.empty else f"CORPORATE ACTIONS — {sym} — None"
            ca_cols = [("BOOK CLOSE", "bc"), ("CASH DIV%", "cash"), ("BONUS%", "bonus")]
            ca_rows: list[list[Text]] = []
            if not ca.empty:
                for _, r in ca.iterrows():
                    ca_rows.append([
                        Text(str(r["bookclose_date"])[:10], style=WHITE),
                        Text(f"{r['cash_dividend_pct']:.1f}%", style=GAIN_HI)
                        if pd.notna(r.get("cash_dividend_pct")) and r["cash_dividend_pct"]
                        else _dim_text("—"),
                        Text(f"{r['bonus_share_pct']:.1f}%", style=GAIN_HI)
                        if pd.notna(r.get("bonus_share_pct")) and r["bonus_share_pct"]
                        else _dim_text("—"),
                    ])

            payload = {
                "found": True,
                "title": title,
                "header": header,
                "chart": chart,
                "stats": stats,
                "report": report,
                "fin_title": fin_title,
                "fin_cols": fin_cols,
                "fin_rows": fin_rows,
                "intel_title": intel_title,
                "intel": intel,
                "ohlcv_title": ohlcv_title,
                "ohlcv_cols": ohlcv_cols,
                "ohlcv_rows": ohlcv_rows,
                "ca_title": ca_title,
                "ca_cols": ca_cols,
                "ca_rows": ca_rows,
            }
            self.call_from_thread(self._render_lookup_payload, cache_key, payload)
        except Exception as e:
            self.call_from_thread(
                self._render_lookup_payload,
                cache_key,
                {"found": False, "sym": sym, "error": str(e)},
            )
    def _render_lookup_payload(self, cache_key: tuple[str, str, str], payload: dict) -> None:
        if cache_key != self._lookup_request_key:
            return
        self._lookup_cache[cache_key] = payload
        while len(self._lookup_cache) > 12:
            oldest_key = next(iter(self._lookup_cache))
            self._lookup_cache.pop(oldest_key, None)

        title_w = self.query_one("#lookup-title", Static)
        header_w = self.query_one("#lookup-header", Static)
        chart_w = self.query_one("#lookup-chart", Static)
        stats_w = self.query_one("#lookup-stats", Static)
        report_w = self.query_one("#lookup-report", Static)
        summary_scroll = self.query_one("#lookup-summary-pane", VerticalScroll)
        intel_title = self.query_one("#lookup-intel-title", Static)
        intel_w = self.query_one("#lookup-intel", Static)
        dt = self.query_one("#dt-lookup", DataTable)
        fin_title = self.query_one("#lookup-fin-title", Static)
        fin_dt = self.query_one("#dt-lookup-fin", DataTable)
        ca_title = self.query_one("#lookup-ca-title", Static)
        ca_dt = self.query_one("#dt-lookup-ca", DataTable)

        if not payload.get("found"):
            sym = payload.get("sym", self.lookup_sym)
            title_w.update(f"LOOKUP: {sym} — NOT FOUND")
            header_w.update("")
            chart_w.update("")
            stats_w.update("")
            report_w.update(Text(f"  {payload.get('error', '')}", style=LABEL) if payload.get("error") else "")
            dt.clear(columns=True)
            fin_title.update("")
            fin_dt.clear(columns=True)
            ca_title.update("")
            ca_dt.clear(columns=True)
            intel_title.update("")
            intel_w.update("")
            return

        title_w.update(payload["title"])
        header_w.update(payload["header"])
        chart_w.update(payload["chart"])
        stats_w.update(payload["stats"])
        report_w.update(payload["report"])
        intel_title.update(payload["intel_title"])
        intel_w.update(payload["intel"])

        self.query_one("#lookup-ohlcv-title", Static).update(payload["ohlcv_title"])
        dt.clear(columns=True)
        for label, key in payload["ohlcv_cols"]:
            dt.add_column(label, key=key)
        for row in payload["ohlcv_rows"]:
            dt.add_row(*row)

        fin_title.update(payload["fin_title"])
        fin_dt.clear(columns=True)
        for label, key in payload["fin_cols"]:
            fin_dt.add_column(label, key=key)
        for row in payload["fin_rows"]:
            fin_dt.add_row(*row)

        ca_title.update(payload["ca_title"])
        ca_dt.clear(columns=True)
        if payload["ca_rows"]:
            for label, key in payload["ca_cols"]:
                ca_dt.add_column(label, key=key)
            for row in payload["ca_rows"]:
                ca_dt.add_row(*row)

        try:
            summary_scroll.scroll_home(animate=False)
            summary_scroll.focus()
        except Exception:
            pass
    def _populate_watchlist(self) -> None:
        """Populate watchlist DataTable with watched stocks and macro items."""
        dt = self.query_one("#dt-watchlist", DataTable)
        dt.clear(columns=True)
        for label, key in [("  #", "n"), ("ITEM", "sym"), ("VALUE", "ltp"),
                           ("CHG%", "chg"), ("OPEN", "open"), ("HIGH", "high"),
                           ("LOW", "low"), ("VOLUME", "vol"), ("TREND", "spark"),
                           ("SIGNAL", "sig")]:
            dt.add_column(label, key=key)

        watchlist_source = "LOCAL"
        held_count = 0
        if self.trade_mode == "live" and self.tms_service:
            if self._set_watchlist_from_tms_snapshot((self._tms_bundle or {}).get("watchlist")):
                watchlist_source = "TMS"
            elif self._live_watchlist:
                local_extras = [item for item in self._paper_watchlist if str(item.get("kind") or "stock") != "stock"]
                self._watchlist = _merge_watchlist_entries(self._live_watchlist, local_extras)
                watchlist_source = "CACHE"
            else:
                self._watchlist = _dedupe_watchlist_entries(
                    [item for item in self._paper_watchlist if str(item.get("kind") or "stock") != "stock"]
                )
                watchlist_source = "TMS"
        else:
            self._watchlist, held_count = self._effective_paper_watchlist()
            if held_count:
                watchlist_source = "LOCAL+HELD"

        if not self._watchlist:
            dt.add_row(_dim_text("—"), _dim_text("Press = to add stocks"),
                       *[Text("")] * 8)
            bar = self.query_one("#wl-status-bar", Static)
            bar.update(Text.from_markup(
                f"[bold {AMBER}]◆ WATCHLIST[/]   "
                f"[#888888]Tracking[/] [bold {WHITE}]0[/] [#888888]items[/]   "
                f"[#888888]Source[/] [bold {WHITE}]{watchlist_source}[/]   "
                f"[#555555]= Add  │  Sync holdings from Portfolio profile  │  - Remove  │  ENTER Open[/]"
            ))
            self._populate_watchlist_side_panels([], [], {}, {})
            return

        kalimati_rows = list(getattr(self, "_kalimati_rows", []) or [])
        macro_rates = dict(getattr(self, "_macro_rates", {}) or {})

        if any(str(item.get("kind") or "stock") != "stock" for item in self._watchlist):
            if not kalimati_rows:
                self._load_kalimati_async()
            if not macro_rates:
                self._load_macro_rates_async()

        ltps = self.md.ltps() if hasattr(self, 'md') else {}
        kalimati_by_name = {str(row.get("name_english") or ""): row for row in kalimati_rows}
        macro_by_label = {str(row.get("item") or ""): row for row in macro_rates.get("indicators", [])}
        forex_by_code = {str(row.get("currency_code") or "").upper(): row for row in macro_rates.get("forex_rows", [])}
        port = load_port()
        stock_entries = [item for item in self._watchlist if str(item.get("kind") or "stock") == "stock"]
        rates_entries = [item for item in self._watchlist if str(item.get("kind") or "") in ("forex", "macro")]
        commodity_entries = [item for item in self._watchlist if str(item.get("kind") or "") == "commodity"]
        self._watchlist_stock_rows = list(stock_entries)
        self.query_one("#wl-main-title", Static).update(f"STOCK WATCHLIST [{len(stock_entries)}]")

        # Build price data for each watchlist symbol
        conn = None
        try:
            conn = _db()
        except Exception:
            pass

        gainers = 0
        losers = 0

        def _watch_pct_text(pct: Optional[float]) -> Text:
            if pct is None:
                return _dim_text("—")
            return _chg_text(float(pct))

        def _watch_range_bar(mn: float, mx: float, avg: float, width: int = 15) -> Text:
            if mx <= mn:
                return Text("—", style=LABEL)
            pos = int((avg - mn) / (mx - mn) * width)
            pos = max(0, min(width - 1, pos))
            bar = "░" * pos + "█" + "░" * (width - pos - 1)
            color = GAIN if avg >= ((mn + mx) / 2) else LOSS
            return Text(bar, style=color)

        for i, entry in enumerate(stock_entries, 1):
            kind = str(entry.get("kind") or "stock")
            sym = str(entry.get("symbol") or entry.get("label") or "").upper()
            ltp = ltps.get(sym, 0) if kind == "stock" else 0
            chg_val = 0.0
            open_p = high_p = low_p = 0.0
            vol = 0
            sparkline = Text("—", style=LABEL)
            signal_text = Text("", style=DIM)

            if kind == "stock" and conn:
                try:
                    row = conn.execute(
                        "SELECT open, high, low, close, volume FROM stock_prices "
                        "WHERE symbol=? ORDER BY date DESC LIMIT 1", (sym,)).fetchone()
                    if row:
                        open_p, high_p, low_p = float(row[0]), float(row[1]), float(row[2])
                        if ltp <= 0:
                            ltp = float(row[3])
                        vol = int(row[4])
                    # Previous close for change %
                    rows2 = conn.execute(
                        "SELECT close FROM stock_prices WHERE symbol=? "
                        "ORDER BY date DESC LIMIT 2", (sym,)).fetchall()
                    if len(rows2) >= 2:
                        prev = float(rows2[1][0])
                        if prev > 0:
                            chg_val = (ltp - prev) / prev * 100

                    # Sparkline from last 30 days
                    hist = conn.execute(
                        "SELECT close FROM stock_prices WHERE symbol=? "
                        "ORDER BY date DESC LIMIT 30", (sym,)).fetchall()
                    if len(hist) >= 3:
                        closes = [float(r[0]) for r in reversed(hist)]
                        sparkline = _render_sparkline(closes, width=15)
                except Exception:
                    pass

            if kind == "stock":
                if not port.empty and sym in port["Symbol"].values:
                    signal_text = Text("● HELD", style=f"bold {CYAN}")
                if chg_val > 0:
                    gainers += 1
                elif chg_val < 0:
                    losers += 1
                dt.add_row(
                    _dim_text(f"{i:2d}"),
                    _sym_text(sym),
                    _price_text(ltp) if ltp > 0 else _dim_text("—"),
                    _chg_text(chg_val),
                    _price_text(open_p) if open_p > 0 else _dim_text("—"),
                    _price_text(high_p) if high_p > 0 else _dim_text("—"),
                    _price_text(low_p) if low_p > 0 else _dim_text("—"),
                    _vol_text(vol) if vol > 0 else _dim_text("—"),
                    sparkline,
                    signal_text,
                )
                continue

            label = str(entry.get("label") or "")
            if kind == "commodity":
                row = kalimati_by_name.get(label) or {}
                pct = row.get("change_pct")
                if pct is not None and float(pct) > 0:
                    gainers += 1
                elif pct is not None and float(pct) < 0:
                    losers += 1
                dt.add_row(
                    _dim_text(f"{i:2d}"),
                    Text(label[:18], style=WHITE),
                    Text(f"{float(row.get('avg') or 0):,.1f}", style=f"bold {AMBER}") if row else _dim_text("—"),
                    _watch_pct_text(float(pct)) if pct is not None else _dim_text("—"),
                    Text(f"{float(row.get('min') or 0):,.1f}", style=DIM) if row else _dim_text("—"),
                    Text(f"{float(row.get('max') or 0):,.1f}", style=DIM) if row else _dim_text("—"),
                    Text(str(row.get("unit") or entry.get("unit") or "")[:8], style=LABEL),
                    Text("KALIMATI", style=DIM),
                    _watch_range_bar(float(row.get("min") or 0), float(row.get("max") or 0), float(row.get("avg") or 0), width=15) if row else Text("—", style=LABEL),
                    Text("● COMMODITY", style=f"bold {CYAN}"),
                )
                continue

            if kind == "macro":
                row = macro_by_label.get(label) or {}
                pct = row.get("change_pct")
                if pct is not None and float(pct) > 0:
                    gainers += 1
                elif pct is not None and float(pct) < 0:
                    losers += 1
                value = float(row.get("value") or 0)
                unit = str(row.get("unit") or "")
                if unit.startswith("NPR"):
                    value_text = Text(_format_compact_npr(value), style=f"bold {AMBER}") if value > 0 else _dim_text("—")
                else:
                    value_text = Text(f"{value:,.2f} {unit}".strip(), style=f"bold {AMBER}") if value > 0 else _dim_text("—")
                dt.add_row(
                    _dim_text(f"{i:2d}"),
                    Text(label[:18], style=WHITE),
                    value_text,
                    _watch_pct_text(float(pct)) if pct is not None else _dim_text("—"),
                    Text(unit[:10], style=LABEL) if unit else _dim_text("—"),
                    Text(str(row.get("group") or entry.get("group") or "")[:10], style=CYAN),
                    _dim_text("—"),
                    Text(str(row.get("source") or "")[:10], style=DIM),
                    Text("—", style=LABEL),
                    Text("● MACRO", style=f"bold {CYAN}"),
                )
                continue

            if kind == "forex":
                row = forex_by_code.get(label.upper()) or {}
                pct = row.get("change_pct")
                if pct is not None and float(pct) > 0:
                    gainers += 1
                elif pct is not None and float(pct) < 0:
                    losers += 1
                dt.add_row(
                    _dim_text(f"{i:2d}"),
                    Text(label[:18], style=f"bold {CYAN}"),
                    Text(f"{float(row.get('buy_rate') or 0):,.2f}", style=WHITE) if row else _dim_text("—"),
                    _watch_pct_text(float(pct)) if pct is not None else _dim_text("—"),
                    Text(f"{float(row.get('sell_rate') or 0):,.2f}", style=WHITE) if row else _dim_text("—"),
                    Text(str(row.get("currency_name") or entry.get("currency_name") or "")[:10], style=WHITE),
                    Text(str(row.get("unit") or 1), style=LABEL) if row else _dim_text("—"),
                    Text(str(row.get("source") or "NRB")[:10], style=DIM),
                    Text("—", style=LABEL),
                    Text("● FOREX", style=f"bold {CYAN}"),
                )

        if not stock_entries:
            dt.add_row(
                _dim_text("—"),
                _dim_text("No stock symbols"),
                *[Text("")] * 8
            )

        if conn:
            conn.close()

        self._populate_watchlist_side_panels(
            rates_entries,
            commodity_entries,
            macro_rates,
            {
                "kalimati_by_name": kalimati_by_name,
                "macro_by_label": macro_by_label,
                "forex_by_code": forex_by_code,
            },
        )

        # Status bar
        n = len(self._watchlist)
        bar = self.query_one("#wl-status-bar", Static)
        bar.update(Text.from_markup(
            f"[bold {AMBER}]◆ WATCHLIST[/]   "
            f"[#888888]Tracking[/] [bold {WHITE}]{n}[/] [#888888]items[/]   "
            f"[#888888]Stocks[/] [bold {WHITE}]{len(stock_entries)}[/]   "
            f"[#888888]Rates[/] [bold {CYAN}]{len(rates_entries)}[/]   "
            f"[#888888]Commodities[/] [bold {AMBER}]{len(commodity_entries)}[/]   "
            f"[#888888]Up[/] [bold {GAIN_HI}]{gainers}[/]   "
            f"[#888888]Down[/] [bold {LOSS_HI}]{losers}[/]   "
            f"[#888888]Source[/] [bold {WHITE}]{watchlist_source}[/]   "
            f"[#888888]Held[/] [bold {CYAN}]{held_count}[/]   "
            f"[#555555]= Add  │  Sync holdings from Portfolio profile  │  - Remove  │  ENTER Open[/]"
        ))
    def _populate_watchlist_side_panels(
        self,
        rates_entries: list[dict],
        commodity_entries: list[dict],
        macro_rates: dict,
        resolved_maps: dict,
    ) -> None:
        forex_by_code = dict(resolved_maps.get("forex_by_code") or {})
        macro_by_label = dict(resolved_maps.get("macro_by_label") or {})
        kalimati_by_name = dict(resolved_maps.get("kalimati_by_name") or {})

        if not forex_by_code and not macro_by_label and not macro_rates:
            self._load_macro_rates_async()
        if not kalimati_by_name:
            self._load_kalimati_async()

        dt_rates = self.query_one("#dt-watchlist-rates", DataTable)
        dt_rates.clear(columns=True)
        dt_rates.add_column("ITEM", key="item", width=16)
        dt_rates.add_column("VALUE", key="value", width=13)
        dt_rates.add_column("CHG%", key="chg", width=8)
        dt_rates.add_column("UNIT", key="unit", width=10)

        rate_rows: list[dict] = []
        if rates_entries:
            for entry in rates_entries:
                kind = str(entry.get("kind") or "")
                if kind == "forex":
                    row = forex_by_code.get(str(entry.get("label") or "").upper())
                    rate_rows.append({"kind": "forex", "tracked": True, "label": str(entry.get("label") or ""), "row": row})
                elif kind == "macro":
                    row = macro_by_label.get(str(entry.get("label") or ""))
                    rate_rows.append({"kind": "macro", "tracked": True, "label": str(entry.get("label") or ""), "row": row})
        self._watchlist_rates_rows = rate_rows

        def _pct_cell(pct: Optional[float]) -> Text:
            if pct is None:
                return _dim_text("—")
            return _chg_text(float(pct))

        if rate_rows:
            for item in rate_rows:
                row = item.get("row") or {}
                if item["kind"] == "forex":
                    value = float(row.get('buy_rate') or 0)
                    value_text = Text(f"{value:,.2f}", style=WHITE) if value > 0 else Text("Loading", style=DIM)
                    unit_text = Text(str(row.get("unit") or 1)[:10], style=LABEL)
                else:
                    value = float(row.get("value") or 0)
                    unit = str(row.get("unit") or "")
                    if value <= 0:
                        value_text = Text("Loading", style=DIM)
                    elif unit.startswith("NPR"):
                        value_text = Text(_format_compact_npr(value), style=f"bold {AMBER}")
                    else:
                        value_text = Text(f"{value:,.2f} {unit}".strip(), style=f"bold {AMBER}")
                    unit_text = Text(unit[:10], style=LABEL) if unit else _dim_text("—")
                label_style = f"bold {CYAN}" if item["kind"] == "forex" else WHITE
                dt_rates.add_row(
                    Text(str(item["label"])[:16], style=label_style),
                    value_text,
                    _pct_cell(row.get("change_pct")),
                    unit_text,
                )
        else:
            dt_rates.add_row(_dim_text("—"), _dim_text("No tracked rates"), Text(""), Text(""))
        self.query_one("#wl-rates-title", Static).update(f"FOREX & MACRO [{len(rate_rows)}]")

        dt_commodities = self.query_one("#dt-watchlist-commodities", DataTable)
        dt_commodities.clear(columns=True)
        dt_commodities.add_column("ITEM", key="item", width=18)
        dt_commodities.add_column("AVG", key="avg", width=10)
        dt_commodities.add_column("CHG%", key="chg", width=8)
        dt_commodities.add_column("UNIT", key="unit", width=8)

        commodity_rows: list[dict] = []
        if commodity_entries:
            for entry in commodity_entries:
                row = kalimati_by_name.get(str(entry.get("label") or ""))
                commodity_rows.append({"tracked": True, "label": str(entry.get("label") or ""), "row": row})
        self._watchlist_commodity_rows = commodity_rows

        if commodity_rows:
            for item in commodity_rows:
                row = item.get("row") or {}
                pct = row.get("change_pct")
                dt_commodities.add_row(
                    Text(str(item["label"])[:18], style=WHITE),
                    Text(f"{float(row.get('avg') or 0):,.1f}", style=f"bold {AMBER}") if float(row.get('avg') or 0) > 0 else Text("Loading", style=DIM),
                    _pct_cell(float(pct)) if pct is not None else _dim_text("—"),
                    Text(str(row.get("unit") or "")[:8], style=LABEL) if row else _dim_text("—"),
                )
        else:
            dt_commodities.add_row(_dim_text("—"), _dim_text("No tracked commodities"), Text(""), Text(""))
        self.query_one("#wl-commodities-title", Static).update(f"COMMODITIES [{len(commodity_rows)}]")
    def _populate_screener(self) -> None:
        """Populate sector treemap and stock screener."""
        heatmap_w = self.query_one("#heatmap-content", Static)
        dt = self.query_one("#dt-screener", DataTable)
        dt.clear(columns=True)

        for label, key in [("SYMBOL", "sym"), ("SECTOR", "sec"), ("LTP", "ltp"),
                           ("CHG%", "chg"), ("VOL", "vol"), ("VRAT", "vrat"),
                           ("RANGE", "range"), ("TREND", "spark")]:
            dt.add_column(label, key=key)

        # Gather all stock data
        conn = None
        all_stocks = []
        sector_data = {}  # sector -> {total_chg, count, symbols, total_vol, stocks}
        MIN_VOL = 1000  # filter out illiquid noise
        try:
            conn = _db()
            latest_date = conn.execute(
                "SELECT MAX(date) FROM stock_prices WHERE symbol != 'NEPSE'"
            ).fetchone()[0]
            if not latest_date:
                heatmap_w.update(Text("No data", style=LABEL))
                return
            prev_date = conn.execute(
                "SELECT MAX(date) FROM stock_prices WHERE date < ? AND symbol != 'NEPSE'",
                (latest_date,)
            ).fetchone()[0]

            if latest_date and prev_date:
                today_rows = conn.execute(
                    "SELECT symbol, open, high, low, close, volume FROM stock_prices "
                    "WHERE date=? AND symbol != 'NEPSE'",
                    (latest_date,)
                ).fetchall()
                today_rows = _dedupe_symbol_rows(today_rows)
                prev_map = {}
                for r in _dedupe_symbol_rows(conn.execute(
                    "SELECT symbol, close FROM stock_prices WHERE date=?",
                    (prev_date,)
                ).fetchall()):
                    prev_map[r[0]] = float(r[1])

                # Get 20-day avg volume for vol ratio
                avg_vol_map = {}
                for r in conn.execute(
                    "SELECT symbol, AVG(volume) FROM stock_prices "
                    "WHERE date > date(?, '-30 days') AND symbol != 'NEPSE' "
                    "GROUP BY symbol", (latest_date,)
                ).fetchall():
                    avg_vol_map[r[0]] = float(r[1]) if r[1] else 0

                from backend.backtesting.simple_backtest import get_symbol_sector

                for r in today_rows:
                    sym = r[0]
                    vol = int(r[5])
                    close = float(r[4])
                    prev = prev_map.get(sym, 0)
                    chg = (close - prev) / prev * 100 if prev > 0 else 0
                    sector = get_symbol_sector(sym) or "Other"
                    avg_v = avg_vol_map.get(sym, 0)
                    vol_ratio = vol / avg_v if avg_v > 0 else 0

                    stock = {
                        "sym": sym, "sector": sector, "ltp": close,
                        "chg": chg, "vol": vol, "vol_ratio": vol_ratio,
                        "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
                    }
                    # Include all for sector stats, but only liquid ones in table
                    if vol > 0:
                        if sector not in sector_data:
                            sector_data[sector] = {"total_chg": 0, "count": 0,
                                                   "total_vol": 0, "stocks": []}
                        sector_data[sector]["total_chg"] += chg
                        sector_data[sector]["count"] += 1
                        sector_data[sector]["total_vol"] += vol
                        sector_data[sector]["stocks"].append(stock)
                    if vol >= MIN_VOL:
                        all_stocks.append(stock)

                # Get sparklines for all displayed stocks
                spark_data = {}
                top_syms = [s["sym"] for s in sorted(all_stocks, key=lambda x: -x["vol"])[:120]]
                for sym in top_syms:
                    hist = conn.execute(
                        "SELECT close FROM stock_prices WHERE symbol=? "
                        "ORDER BY date DESC LIMIT 20", (sym,)
                    ).fetchall()
                    if len(hist) >= 3:
                        spark_data[sym] = [float(r[0]) for r in reversed(hist)]
        except Exception:
            pass
        finally:
            if conn:
                conn.close()

        # ── Sector Performance ──
        if sector_data:
            sector_perf = []
            total_vol_all = sum(d["total_vol"] for d in sector_data.values())
            for sec, data in sector_data.items():
                avg_chg = data["total_chg"] / data["count"] if data["count"] > 0 else 0
                vol_wt = data["total_vol"] / total_vol_all * 100 if total_vol_all > 0 else 0
                sector_perf.append((sec, avg_chg, data["count"], data["total_vol"], vol_wt))
            sector_perf.sort(key=lambda x: -x[1])  # best performing first

            heatmap = Text()
            heatmap.append("\n", style="")

            # Colored block indicators: ■ = sector health at a glance
            max_abs = max(abs(s[1]) for s in sector_perf) if sector_perf else 1
            if max_abs == 0:
                max_abs = 1

            for sec, avg_chg, count, total_vol, vol_wt in sector_perf:
                # Pick color from smooth gradient
                if avg_chg > 2:
                    fg = "#00ff7f"
                elif avg_chg > 1:
                    fg = "#00cc60"
                elif avg_chg > 0:
                    fg = "#66cc66"
                elif avg_chg > -1:
                    fg = "#cc9933"
                elif avg_chg > -3:
                    fg = "#cc4444"
                else:
                    fg = "#ff4545"

                # Filled/empty blocks proportional to magnitude
                n_filled = max(1, int(abs(avg_chg) / max_abs * 8))
                blocks = "█" * n_filled + "░" * (8 - n_filled)

                heatmap.append(f"  {blocks} ", style=fg)
                heatmap.append(f"{avg_chg:+5.1f}%  ", style=f"bold {fg}")
                heatmap.append(f"{sec}", style=WHITE)
                heatmap.append(f"  {count}\n", style=DIM)

            # Summary line
            total_stocks = sum(d["count"] for d in sector_data.values())
            n_up = sum(1 for s in all_stocks if s["chg"] > 0)
            n_dn = sum(1 for s in all_stocks if s["chg"] < 0)
            heatmap.append(f"\n  {total_stocks} stocks  {_vol(total_vol_all)} vol  ", style=LABEL)
            heatmap.append(f"▲{n_up}", style=GAIN_HI)
            heatmap.append(f" ▼{n_dn}\n", style=LOSS_HI)
            heatmap_w.update(heatmap)
        else:
            heatmap_w.update(Text("  Loading sector data...", style=LABEL))

        # ── Stock list (liquid stocks only, sorted by volume) ──
        all_stocks.sort(key=lambda x: -x["vol"])
        for s in all_stocks[:100]:
            spark = _render_sparkline(spark_data.get(s["sym"], []), width=12) \
                if s["sym"] in spark_data else Text("", style=DIM)
            # Vol ratio indicator
            vr = s["vol_ratio"]
            if vr > 2.5:
                vr_text = Text(f"{vr:.1f}x", style=f"bold {GAIN_HI}")
            elif vr > 1.5:
                vr_text = Text(f"{vr:.1f}x", style=YELLOW)
            elif vr > 0:
                vr_text = Text(f"{vr:.1f}x", style=DIM)
            else:
                vr_text = Text("—", style=DIM)
            # Day range as compact string
            rng = s["high"] - s["low"]
            rng_pct = rng / s["low"] * 100 if s["low"] > 0 else 0
            rng_text = Text(f"{s['low']:.0f}-{s['high']:.0f}", style=DIM) if rng_pct > 0 \
                else Text("—", style=DIM)
            sec_display = s["sector"]
            if sec_display == "Other":
                sec_display = "—"
            dt.add_row(
                _sym_text(s["sym"]),
                _dim_text(sec_display[:14]),
                _price_text(s["ltp"]),
                _chg_text(s["chg"]),
                _vol_text(s["vol"]),
                vr_text,
                rng_text,
                spark,
            )

        # Status bar
        n_stocks = len(all_stocks)
        n_up = sum(1 for s in all_stocks if s["chg"] > 0)
        n_down = sum(1 for s in all_stocks if s["chg"] < 0)
        n_sectors = len([s for s in sector_data if s != "Other"])
        bar = self.query_one("#screener-status-bar", Static)
        bar.update(Text.from_markup(
            f"[bold {AMBER}]◆ SIGNALS WORKSPACE[/]   "
            f"[#888888]Active[/] [bold {WHITE}]{n_stocks}[/] [#555555](vol≥1K)[/]   "
            f"[{GAIN_HI}]▲{n_up}[/]  [{LOSS_HI}]▼{n_down}[/]   "
            f"[#888888]{n_sectors} sectors[/]   "
            f"[#555555]Signals + screener merged  │  ENTER Lookup  │  / Command[/]"
        ))
        self.query_one("#screener-list-title", Static).update(
            f"ACTIVE STOCKS — {n_stocks}  │  Vol≥1K  │  Sorted by volume  │  VRAT=vol/20d avg")
