"""Strategies and Signals tabs on a terminal linked to nepsequant.com.

Linked, the terminal has no strategy builder: strategies are the ones the
user built and backtested on the website (Lab rule sets, signal blends, saved
Screener screens). The Strategies tab lists them with their backtest numbers,
ASSIGN picks one, AUTO-TRADE trades it on the shared paper book while the
terminal is open, and the Signals tab shows its picks for today.

Network calls run in @work threads; the UI only reads cached copies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from rich.text import Text
from textual import work
from textual.widgets import Button, DataTable, Static

from apps.tui.io import website_strategies as ws
from apps.tui.theme import AMBER, CYAN, DIM, GAIN, GAIN_HI, LABEL, LOSS, WHITE

# The builder widgets hidden while linked (the website panel takes their place).
BUILDER_IDS = (
    "strategy-name-row", "signal-picker-label", "signal-picker-area", "strategy-params-label",
    "strategy-config-row-a", "strategy-config-row-b", "strategy-actions-row",
    "strategy-backtest-label", "strategy-backtest-row", "strategy-backtest-summary",
)


def _nst_today():
    from backend.trading.live_trader import now_nst

    return now_nst().date()


class WebsiteStrategiesMixin:
    # ── setup ───────────────────────────────────────────────────────────────
    def _init_website_strategies(self) -> None:
        """Called once the shared website book exists."""
        from apps.tui.io.strategy_autotrader import AutoTrader

        book = self._website_book
        self._website_strategies = ws.WebsiteStrategies.from_book(book) if book is not None else None
        self._autotrader = AutoTrader(book._key) if book is not None else None
        self._apply_strategies_mode()

    def _strategies_linked(self) -> bool:
        return self._website_strategies is not None

    def _apply_strategies_mode(self) -> None:
        linked = self._strategies_linked()
        for wid in BUILDER_IDS:
            try:
                self.query_one(f"#{wid}").display = not linked
            except Exception:
                pass
        try:
            self.query_one("#webstrat-panel").display = linked
            self.query_one("#strategy-builder-title", Static).update(
                "STRATEGY" if linked else "STRATEGY BUILDER"
            )
            self.query_one("#strategies-left-title", Static).update(
                "YOUR STRATEGIES ON NEPSEQUANT.COM" if linked else "SAVED STRATEGIES"
            )
        except Exception:
            pass

    # ── loading ─────────────────────────────────────────────────────────────
    @work(thread=True, group="website-strategies", exclusive=True)
    def _load_website_strategies(self, then_picks: bool = True) -> None:
        strategies = self._website_strategies
        if strategies is None:
            return
        strategies.refresh()
        if then_picks and strategies.assigned:
            strategies.picks(strategies.assigned)
        self.call_from_thread(self._populate_website_strategies_tab)
        if then_picks:
            self.call_from_thread(self._render_website_signals)

    # ── Strategies tab ──────────────────────────────────────────────────────
    def _selected_website_strategy(self) -> Optional[dict]:
        items = list(getattr(self._website_strategies, "strategies", []) or [])
        if not items:
            return None
        self._webstrat_index = max(0, min(int(self._webstrat_index or 0), len(items) - 1))
        return items[self._webstrat_index]

    def _populate_website_strategies_tab(self) -> None:
        strategies = self._website_strategies
        if strategies is None:
            return
        self._apply_strategies_mode()
        trader = self._autotrader
        auto_on = bool(trader and trader.enabled)
        try:
            dt = self.query_one("#dt-strategy-list", DataTable)
            dt.clear(columns=True)
            for label, key in (("", "on"), ("NAME", "name"), ("TYPE", "type"), ("RETURN", "ret"),
                               ("VS NEPSE", "vs"), ("SHARPE", "shrp"), ("MAX DD", "dd"), ("TRADES", "trd")):
                dt.add_column(label, key=key)
            for s in strategies.strategies:
                assigned = s.get("id") == strategies.assigned
                mark = Text("AUTO" if assigned and auto_on else ("●" if assigned else ""), style=f"bold {GAIN_HI}")

                def num(key, fmt, style_by_sign=False):
                    v = ws.metric(s, key)
                    if v is None:
                        return Text("—", style=DIM)
                    style = (GAIN if v > 0 else LOSS if v < 0 else WHITE) if style_by_sign else WHITE
                    return Text(fmt.format(v), style=style)

                dt.add_row(
                    mark,
                    Text(str(s.get("name") or s.get("id")), style=WHITE),
                    Text(ws.KIND_LABEL.get(str(s.get("kind")), str(s.get("kind") or "").upper()), style=CYAN),
                    num("total_return_pct", "{:+.1f}%", True),
                    num("vs_nepse_pct_points", "{:+.1f}pp", True),
                    num("sharpe", "{:.2f}"),
                    num("max_drawdown_pct", "{:.1f}%"),
                    num("trade_count", "{:.0f}"),
                )
            if not strategies.strategies:
                dt.add_row(Text(""), Text("No strategies yet" if strategies.loaded else "Loading…", style=DIM),
                           *[Text("")] * 6)
        except Exception:
            pass
        assigned = strategies.assigned_strategy()
        note = (
            f"Assigned: {assigned.get('name')}   Auto-trade: {'ON' if auto_on else 'OFF'}"
            if assigned else "No strategy assigned. Select one and press ASSIGN."
        )
        if strategies.error:
            note += f"\n{strategies.error}"
        try:
            self.query_one("#strategy-accounts-note", Static).update(Text(note, style=LABEL))
            self.query_one("#strategy-summary", Static).update(Text.from_markup(
                f"[bold {WHITE}]Strategies come from nepsequant.com[/]  [#8a8a8a]Build and backtest them in the "
                f"website's Lab or Screener; assign one here to see its picks and auto-trade your paper book.[/]"
            ))
        except Exception:
            pass
        self._render_website_strategy_detail()

    def _render_website_strategy_detail(self) -> None:
        strategies = self._website_strategies
        selected = self._selected_website_strategy()
        trader = self._autotrader
        try:
            detail = self.query_one("#webstrat-detail", Static)
            if selected is None:
                detail.update(Text(
                    "You have no strategies on nepsequant.com yet. Build one in the Lab (rules or a signal blend) "
                    "or save a Screener screen, backtest it, then press REFRESH.", style=LABEL))
            else:
                lines = ws.describe(selected)
                is_assigned = selected.get("id") == strategies.assigned
                body = Text()
                body.append(lines[0] + "\n", style=f"bold {AMBER}")
                for line in lines[1:]:
                    body.append(line + "\n", style=WHITE)
                if is_assigned:
                    picks = strategies.cached_picks(selected.get("id"))
                    if picks and picks.rows:
                        body.append(f"Today's picks ({picks.as_of}): " + ", ".join(r["symbol"] for r in picks.rows[:8]) + "\n", style=CYAN)
                if strategies.notice:
                    body.append(strategies.notice, style=DIM)
                detail.update(body)
            assign = self.query_one("#webstrat-btn-assign", Button)
            is_assigned = bool(selected and selected.get("id") == strategies.assigned)
            assign.label = "UNASSIGN" if is_assigned else "ASSIGN"
            auto = self.query_one("#webstrat-btn-auto", Button)
            auto.label = f"AUTO {'ON' if trader and trader.enabled else 'OFF'}"
            auto.disabled = not strategies.assigned
            log = list((trader.state.get("log") if trader else []) or [])[-12:]
            self.query_one("#webstrat-log", Static).update(
                Text("\n".join(reversed(log)) if log else "Nothing yet.", style=DIM)
            )
        except Exception:
            pass

    def _webstrat_select_row(self, idx: int) -> None:
        self._webstrat_index = int(idx)
        self._render_website_strategy_detail()

    # ── buttons ─────────────────────────────────────────────────────────────
    def _on_webstrat_button(self, bid: str) -> bool:
        if bid == "webstrat-btn-refresh":
            self._set_status("Refreshing your nepsequant.com strategies…")
            self._load_website_strategies(True)
            return True
        if bid == "webstrat-btn-assign":
            selected = self._selected_website_strategy()
            if selected is None:
                self._set_status("Select a strategy first")
                return True
            target = None if selected.get("id") == self._website_strategies.assigned else selected.get("id")
            self._assign_website_strategy(target)
            return True
        if bid == "webstrat-btn-auto":
            self._toggle_autotrade()
            return True
        return False

    @work(thread=True, group="website-strategies")
    def _assign_website_strategy(self, strategy_id: Optional[str]) -> None:
        strategies = self._website_strategies
        ok, msg = strategies.assign(strategy_id)
        if ok and self._autotrader and self._autotrader.enabled:
            self._autotrader.set_enabled(False, strategy_id)
            msg += " · auto-trade switched off (turn it on for the new strategy)"
        if ok and strategy_id:
            strategies.picks(strategy_id, force=True)
        self.call_from_thread(self._set_status, msg)
        self.call_from_thread(self._populate_website_strategies_tab)
        self.call_from_thread(self._render_website_signals)

    def _toggle_autotrade(self) -> None:
        strategies, trader, book = self._website_strategies, self._autotrader, self._website_book
        if trader is None or strategies is None:
            return
        if trader.enabled:
            trader.set_enabled(False, strategies.assigned)
            self._set_status("Auto-trade OFF")
        elif not strategies.assigned:
            self._set_status("Assign a strategy first")
            return
        elif not (book and book.can_trade):
            self._set_status("Auto-trade needs a key with Paper trading ticked (nepsequant.com Settings → AI agents)")
            return
        else:
            name = (strategies.assigned_strategy() or {}).get("name") or strategies.assigned
            trader.set_enabled(True, strategies.assigned)
            self._set_status(
                f"Auto-trade ON for {name}: paper orders on your nepsequant.com book while this terminal is open "
                f"and NEPSE is trading. Stocks you bought yourself are never sold by it."
            )
        self._populate_website_strategies_tab()

    # ── the auto-trader pass (from the background refresh) ─────────────────
    def _run_autotrader_pass(self) -> None:
        """Runs in the refresh worker thread, after the book refreshed."""
        trader, strategies, book = self._autotrader, self._website_strategies, self._website_book
        if not (trader and strategies and book) or not trader.enabled:
            return
        lines = trader.run_once(book=book, strategies=strategies, today=_nst_today())
        if lines:
            book.refresh()
            self.call_from_thread(self._after_autotrade, lines)

    def _after_autotrade(self, lines: list[str]) -> None:
        for line in lines:
            try:
                self._append_activity(line)
            except Exception:
                pass
        self._set_status(lines[-1])
        self._load_paper_orders()
        self._populate_orders_tab()
        self._populate_portfolio_and_risk()
        self._populate_trades_full()
        self._populate_website_strategies_tab()

    # ── Signals tab ─────────────────────────────────────────────────────────
    @work(thread=True, group="website-strategies-picks", exclusive=True)
    def _load_website_signals(self, force: bool = False) -> None:
        strategies = self._website_strategies
        if strategies is None:
            return
        if not strategies.loaded:
            strategies.refresh()
        if strategies.assigned:
            strategies.picks(strategies.assigned, force=force)
        self.call_from_thread(self._render_website_signals)

    def _render_website_signals(self) -> None:
        strategies = self._website_strategies
        if strategies is None:
            return
        assigned = strategies.assigned_strategy()
        cols = [("  #", "n"), ("SYMBOL", "sym"), ("SCORE", "score"), ("LTP", "ltp"), ("SECTOR", "sec"), ("WHY", "why")]
        rows: list[list[Text]] = []
        if assigned is None:
            title = "SIGNALS — no strategy assigned (0: Strategies → ASSIGN)"
            rows.append([Text("—", style=DIM), Text("Assign one of your nepsequant.com strategies", style=LABEL)] + [Text("")] * 4)
        else:
            picks = strategies.cached_picks(assigned["id"])
            name = str(assigned.get("name") or assigned["id"])
            if picks is None:
                title = f"SIGNALS — {name} · loading…"
            elif picks.warming and not picks.rows:
                title = f"SIGNALS — {name} · preparing today's picks…"
            else:
                when = picks.as_of or "?"
                err = f"  |  {picks.error[:60]}" if picks.error else ""
                title = f"SIGNALS — {name} · {len(picks.rows)} picks · {when}{err}"
            for r in (picks.rows if picks else []):
                score = r.get("score")
                ltp = r.get("ltp")
                rows.append([
                    Text(str(r.get("rank") or ""), style=DIM),
                    Text(str(r.get("symbol") or ""), style=f"bold {WHITE}"),
                    Text("—" if score is None else f"{float(score):.3f}", style=GAIN_HI),
                    Text("—" if not ltp else f"{float(ltp):,.1f}", style=WHITE),
                    Text(str(r.get("sector") or ""), style=CYAN),
                    Text(str(r.get("reason") or "")[:60], style=LABEL),
                ])
            if picks and not picks.rows and not picks.warming:
                rows.append([Text("—", style=DIM), Text(picks.note or "No stocks pass this strategy today", style=LABEL)] + [Text("")] * 4)
        picks_now = strategies.cached_picks((assigned or {}).get("id"))
        self._signals_last_strategy_name = str((assigned or {}).get("name") or "")
        self._signals_last_count = len(picks_now.rows) if picks_now else 0
        self._signals_last_loaded_at = datetime.now().strftime("%H:%M:%S")
        try:
            self._set_signals_table(cols, rows)
            self.query_one("#signals-table-title", Static).update(Text(title))
        except Exception:
            pass
