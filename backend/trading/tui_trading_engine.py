"""TUI Auto-Trading Engine — runs inside the Textual dashboard.

Pipeline: generate signals → agent filter → execute buys → monitor exits.
Uses separate state files (tui_paper_*) so it doesn't interfere with live_trader.py.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from configs.long_term import LONG_TERM_CONFIG
from backend.quant_pro.paths import (
    ensure_dir,
    get_project_root,
    get_trading_runtime_dir,
    migrate_legacy_path,
)
from backend.trading.live_trader import (
    Position,
    TradeRecord,
    append_nav_log,
    append_trade_log,
    check_exits,
    fetch_prices_for_symbols,
    generate_signals,
    is_market_open,
    load_portfolio,
    now_nst,
    save_portfolio,
)
from backend.quant_pro.config import (
    HARD_STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRAILING_STOP_PCT,
)
from backend.quant_pro.database import get_db_path
from backend.quant_pro.nepse_calendar import is_trading_day
from backend.backtesting.simple_backtest import get_symbol_sector, load_all_prices
from validation.transaction_costs import TransactionCostModel as NepseFees

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# File paths (separate from live_trader's paper_* files)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = get_project_root(__file__)
TRADING_RUNTIME_DIR = ensure_dir(get_trading_runtime_dir(__file__))
PORTFOLIO_FILE = migrate_legacy_path(TRADING_RUNTIME_DIR / "tui_paper_portfolio.csv", [PROJECT_ROOT / "tui_paper_portfolio.csv"])
TRADE_LOG_FILE = migrate_legacy_path(TRADING_RUNTIME_DIR / "tui_paper_trade_log.csv", [PROJECT_ROOT / "tui_paper_trade_log.csv"])
NAV_LOG_FILE = migrate_legacy_path(TRADING_RUNTIME_DIR / "tui_paper_nav_log.csv", [PROJECT_ROOT / "tui_paper_nav_log.csv"])
STATE_FILE = migrate_legacy_path(TRADING_RUNTIME_DIR / "tui_paper_state.json", [PROJECT_ROOT / "tui_paper_state.json"])
LIVE_PORTFOLIO_FILE = migrate_legacy_path(TRADING_RUNTIME_DIR / "paper_portfolio.csv", [PROJECT_ROOT / "paper_portfolio.csv"])
LIVE_TRADE_LOG_FILE = migrate_legacy_path(TRADING_RUNTIME_DIR / "paper_trade_log.csv", [PROJECT_ROOT / "paper_trade_log.csv"])
LIVE_NAV_LOG_FILE = migrate_legacy_path(TRADING_RUNTIME_DIR / "paper_nav_log.csv", [PROJECT_ROOT / "paper_nav_log.csv"])
LIVE_STATE_FILE = migrate_legacy_path(TRADING_RUNTIME_DIR / "paper_state.json", [PROJECT_ROOT / "paper_state.json"])


class TUITradingEngine:
    """Self-contained paper trading engine for the TUI dashboard.

    Lifecycle:
        engine = TUITradingEngine(...)
        engine.run_loop()   # blocking — call from @work(thread=True)
        engine.stop()       # from another thread to shut down
    """

    def __init__(
        self,
        capital: float = LONG_TERM_CONFIG["initial_capital"],
        signal_types: Optional[List[str]] = None,
        max_positions: int = LONG_TERM_CONFIG["max_positions"],
        holding_days: int = LONG_TERM_CONFIG["holding_days"],
        sector_limit: float = LONG_TERM_CONFIG["sector_limit"],
        refresh_secs: int = 300,
        on_status: Optional[Callable[[str], None]] = None,
        on_activity: Optional[Callable[[str], None]] = None,
        on_portfolio_changed: Optional[Callable[[], None]] = None,
        on_agent_updated: Optional[Callable[[], None]] = None,
    ):
        self.capital = capital
        self.signal_types = signal_types or list(LONG_TERM_CONFIG["signal_types"])
        self.max_positions = max_positions
        self.holding_days = holding_days
        self.sector_limit = sector_limit
        self.refresh_secs = refresh_secs

        # Callbacks (called from engine thread — caller wraps with call_from_thread)
        self._on_status = on_status or (lambda msg: None)
        self._on_activity = on_activity or (lambda msg: None)
        self._on_portfolio_changed = on_portfolio_changed or (lambda: None)
        self._on_agent_updated = on_agent_updated or (lambda: None)

        # State
        self.positions: Dict[str, Position] = {}
        self.cash: float = capital
        self.phase: str = "idle"
        self.regime: str = "unknown"
        self.signals_today: List[dict] = []
        self.signals_generated_today: bool = False
        self._last_signal_date: str = ""
        self._last_refresh: float = 0.0
        self._eod_logged_date: str = ""
        self._stop_flag: bool = False
        self._prices_df: Optional[pd.DataFrame] = None
        self._consecutive_losses: int = 0
        self._daily_start_nav: float = capital
        self._halted: bool = False
        self._halt_reason: str = ""

        # Load persisted state
        self._load_state()

    # ─── Status helpers ─────────────────────────────────────────────────────

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        n = len(self.positions)
        nav = self._calc_nav()
        self._on_status(
            f"PAPER AUTO  |  {phase.upper().replace('_', ' ')}  |  "
            f"{n}/{self.max_positions} pos  |  NAV: {nav:,.0f}"
        )

    def _log(self, msg: str) -> None:
        nst = now_nst()
        ts = nst.strftime("%H:%M:%S")
        self._on_activity(f"{ts}  {msg}")
        logger.info("[TUI-engine] %s", msg)

    # ─── NAV ────────────────────────────────────────────────────────────────

    def _calc_nav(self) -> float:
        pos_value = sum(
            pos.shares * (pos.last_ltp or pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash + pos_value

    def _calc_positions_value(self) -> float:
        return sum(
            pos.shares * (pos.last_ltp or pos.entry_price)
            for pos in self.positions.values()
        )

    # ─── Persistence ────────────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Resume from saved state, or bootstrap from existing paper portfolio."""
        # Check if TUI state files exist
        if not PORTFOLIO_FILE.exists() and LIVE_PORTFOLIO_FILE.exists():
            # First run: copy existing paper portfolio files over
            for src, dst in [
                (LIVE_PORTFOLIO_FILE, PORTFOLIO_FILE),
                (LIVE_TRADE_LOG_FILE, TRADE_LOG_FILE),
                (LIVE_NAV_LOG_FILE, NAV_LOG_FILE),
            ]:
                if Path(src).exists():
                    shutil.copy2(src, dst)

            # Copy state JSON with adapted cash
            if LIVE_STATE_FILE.exists():
                try:
                    s = json.loads(LIVE_STATE_FILE.read_text())
                    self.cash = s.get("cash", self.capital)
                    STATE_FILE.write_text(json.dumps({
                        "cash": self.cash,
                        "last_signal_date": "",
                        "consecutive_losses": 0,
                        "daily_start_nav": self.cash,
                        "eod_logged_date": "",
                        "timestamp": time.time(),
                    }, indent=2))
                except Exception:
                    pass

        # Positions
        self.positions = load_portfolio(str(PORTFOLIO_FILE))

        # State JSON
        p = STATE_FILE
        if p.exists():
            try:
                s = json.loads(p.read_text())
                self.cash = s.get("cash", self.capital)
                self._last_signal_date = s.get("last_signal_date", "")
                self._consecutive_losses = s.get("consecutive_losses", 0)
                self._daily_start_nav = s.get("daily_start_nav", self.capital)
                self._eod_logged_date = s.get("eod_logged_date", "")
            except Exception:
                pass
        elif not self.positions:
            # Fresh start
            self.cash = self.capital

    def _persist_state(self) -> None:
        """Save portfolio + state to disk."""
        save_portfolio(self.positions, str(PORTFOLIO_FILE))
        state = {
            "cash": round(self.cash, 2),
            "last_signal_date": self._last_signal_date,
            "consecutive_losses": self._consecutive_losses,
            "daily_start_nav": round(self._daily_start_nav, 2),
            "eod_logged_date": self._eod_logged_date,
            "timestamp": time.time(),
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))

    # ─── Main loop ──────────────────────────────────────────────────────────

    def run_loop(self) -> None:
        """Main trading loop — blocks until stop() is called."""
        self._log("Engine started")
        self._set_phase("initializing")

        # Load price data once
        try:
            conn = sqlite3.connect(str(get_db_path()))
            self._prices_df = load_all_prices(conn)
            conn.close()
            self._log(f"Price data loaded: {len(self._prices_df)} rows")
        except Exception as e:
            self._log(f"ERROR loading prices: {e}")
            self._set_phase("error")
            return

        while not self._stop_flag:
            try:
                self._tick()
            except Exception as e:
                self._log(f"Tick error: {e}")
                logger.exception("TUI engine tick error")

            # Sleep in small increments so stop() is responsive
            for _ in range(10):
                if self._stop_flag:
                    break
                time.sleep(1)

        self._persist_state()
        self._log("Engine stopped")
        self._set_phase("stopped")

    def _tick(self) -> None:
        """One iteration of the trading loop."""
        nst = now_nst()
        today_str = nst.strftime("%Y-%m-%d")

        # Reset for new day
        if self._last_signal_date and self._last_signal_date != today_str:
            self.signals_generated_today = False
            self._daily_start_nav = self._calc_nav()

        # Check if halted
        if self._halted:
            self._set_phase(f"halted: {self._halt_reason}")
            return

        # Market closed
        if not is_market_open():
            # EOD logging (once per day, after market close)
            if (nst.hour >= 15 and is_trading_day(nst.date())
                    and self._eod_logged_date != today_str
                    and self.signals_generated_today):
                self._log_eod(today_str)
            self._set_phase("market_closed")
            return

        # ── Market is open ──────────────────────────────────────────────

        # Step 1: Generate signals (once per day)
        if not self.signals_generated_today:
            self._set_phase("generating_signals")
            self._generate_signals()

            # Step 2: Agent filter
            self._set_phase("agent_reviewing")
            approved = self._run_agent_filter(self.signals_today)

            # Step 3: Execute buys
            if approved:
                self._set_phase("executing_buys")
                self._execute_buys(approved)

            self.signals_generated_today = True
            self._last_signal_date = today_str
            self._persist_state()
            self._on_portfolio_changed()

        # Step 4: Periodic price refresh + exit checks
        if time.monotonic() - self._last_refresh > self.refresh_secs:
            self._set_phase("monitoring")
            self._refresh_and_check_exits()
            self._check_kill_switch()
            self._last_refresh = time.monotonic()

        self._set_phase("monitoring")

    def stop(self) -> None:
        """Signal the loop to stop (thread-safe)."""
        self._stop_flag = True

    # ─── Signal generation ──────────────────────────────────────────────────

    def _generate_signals(self) -> None:
        self._log(f"Generating signals... types={self.signal_types}")

        try:
            signals, regime = generate_signals(
                self._prices_df,
                self.signal_types,
                use_regime_filter=True,
            )
            self.regime = regime
            self.signals_today = signals
            self._log(f"{len(signals)} raw signals, regime={regime}")
        except Exception as e:
            self._log(f"Signal generation failed: {e}")
            self.signals_today = []
            self.regime = "unknown"

    # ─── Agent filter ───────────────────────────────────────────────────────

    def _run_agent_filter(self, signals: List[dict]) -> List[dict]:
        """Send signals to Claude Sonnet agent, return only APPROVED ones."""
        if not signals:
            self._log("No signals to review")
            return []

        self._log(f"Sending {len(signals)} signals to agent for review...")

        # Call agent analysis (this triggers a Sonnet subprocess call)
        try:
            from backend.agents.agent_analyst import analyze, check_trade_approval
            analyze(force=True)
            self._on_agent_updated()
        except Exception as e:
            self._log(f"Agent analysis failed: {e} — allowing all signals")
            return signals

        # Filter signals through agent verdicts
        approved = []
        rejected = []
        held = []

        try:
            from backend.agents.agent_analyst import check_trade_approval
            for sig in signals:
                sym = sig["symbol"]
                ok, reason = check_trade_approval(sym, "BUY")
                if ok:
                    sig["agent_reason"] = reason
                    approved.append(sig)
                    if "APPROVED" in reason.upper():
                        held.append(None)  # just for counting
                else:
                    rejected.append(sig)
        except Exception as e:
            self._log(f"Agent filter error: {e} — allowing all")
            return signals

        n_approve = len(approved)
        n_reject = len(rejected)
        n_hold = len([s for s in approved if "HOLD" in s.get("agent_reason", "").upper()])

        self._log(
            f"Agent: {n_approve} APPROVE, {n_reject} REJECT, {n_hold} HOLD"
        )
        return approved

    # ─── Buy execution ──────────────────────────────────────────────────────

    def _execute_buys(self, signals: List[dict]) -> None:
        """Execute paper buys for approved signals."""
        available_slots = self.max_positions - len(self.positions)
        if available_slots <= 0:
            self._log("No position slots available")
            return

        # Fetch LTPs for all candidate symbols
        candidate_syms = [s["symbol"] for s in signals if s["symbol"] not in self.positions]
        if not candidate_syms:
            return

        ltps = fetch_prices_for_symbols(candidate_syms)
        today_str = now_nst().strftime("%Y-%m-%d")
        per_position = self.capital / self.max_positions
        bought = 0

        for sig in signals:
            if bought >= available_slots:
                break

            sym = sig["symbol"]
            if sym in self.positions:
                continue

            ltp = ltps.get(sym)
            if not ltp or ltp <= 0:
                self._log(f"SKIP {sym}: no LTP available")
                continue

            # Sector concentration check
            sector = get_symbol_sector(sym) or "Other"
            nav = self._calc_nav()
            sector_value = sum(
                pos.shares * (pos.last_ltp or pos.entry_price)
                for pos in self.positions.values()
                if (get_symbol_sector(pos.symbol) or "Other") == sector
            )
            if nav > 0 and (sector_value / nav) > self.sector_limit:
                self._log(f"SKIP {sym}: sector {sector} over {self.sector_limit:.0%} limit")
                continue

            # Position sizing
            budget = min(per_position, self.cash * 0.95)
            if budget < ltp * 10:  # minimum 10 shares
                self._log(f"SKIP {sym}: insufficient cash ({self.cash:.0f})")
                continue

            shares = int(budget / ltp)
            if shares <= 0:
                continue

            # Calculate fees
            fees = NepseFees.total_fees(shares, ltp)
            total_cost = shares * ltp + fees

            if total_cost > self.cash:
                shares = int((self.cash - fees) / ltp)
                if shares <= 0:
                    continue
                fees = NepseFees.total_fees(shares, ltp)
                total_cost = shares * ltp + fees

            # Execute buy
            self.cash -= total_cost
            self.positions[sym] = Position(
                symbol=sym,
                shares=shares,
                entry_price=ltp,
                entry_date=today_str,
                buy_fees=fees,
                signal_type=sig.get("signal_type", "unknown"),
                high_watermark=ltp,
                last_ltp=ltp,
            )

            # Log trade
            append_trade_log(
                TradeRecord(
                    date=today_str,
                    action="BUY",
                    symbol=sym,
                    shares=shares,
                    price=ltp,
                    fees=fees,
                    reason=sig.get("signal_type", ""),
                ),
                TRADE_LOG_FILE,
            )

            agent_note = sig.get("agent_reason", "")[:40]
            self._log(
                f"BUY {sym} {shares} @ {ltp:.1f} [{sig.get('signal_type', '')}] "
                f"✓ {agent_note}"
            )
            bought += 1

        if bought > 0:
            self._persist_state()
            self._on_portfolio_changed()
            self._log(f"Executed {bought} buys, cash remaining: {self.cash:,.0f}")

    # ─── Price refresh + exit checks ────────────────────────────────────────

    def _refresh_and_check_exits(self) -> None:
        """Refresh LTPs for held positions and check exit conditions."""
        if not self.positions:
            return

        symbols = list(self.positions.keys())
        ltps = fetch_prices_for_symbols(symbols)

        # Update positions with new prices
        for sym, ltp in ltps.items():
            if ltp and sym in self.positions:
                pos = self.positions[sym]
                pos.last_ltp = ltp
                if ltp > pos.high_watermark:
                    pos.high_watermark = ltp

        # Check exits
        exits = check_exits(self.positions, self.holding_days)
        if not exits:
            self._persist_state()
            return

        today_str = now_nst().strftime("%Y-%m-%d")
        for sym, reason in exits:
            pos = self.positions.get(sym)
            if not pos:
                continue

            sell_price = pos.last_ltp or pos.entry_price
            sell_fees = NepseFees.total_fees(pos.shares, sell_price)
            gross_proceeds = pos.shares * sell_price - sell_fees
            pnl = gross_proceeds - pos.cost_basis
            pnl_pct = pnl / pos.cost_basis * 100 if pos.cost_basis > 0 else 0

            self.cash += gross_proceeds

            # Log trade
            append_trade_log(
                TradeRecord(
                    date=today_str,
                    action="SELL",
                    symbol=sym,
                    shares=pos.shares,
                    price=sell_price,
                    fees=sell_fees,
                    reason=reason,
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 2),
                ),
                TRADE_LOG_FILE,
            )

            # Track consecutive losses
            if pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

            pnl_str = f"{pnl:+,.0f} ({pnl_pct:+.1f}%)"
            self._log(f"SELL {sym} {pos.shares} @ {sell_price:.1f} [{reason}] P&L: {pnl_str}")
            del self.positions[sym]

        self._persist_state()
        self._on_portfolio_changed()

    # ─── Kill switch ────────────────────────────────────────────────────────

    def _check_kill_switch(self) -> None:
        """Halt trading if risk limits breached."""
        nav = self._calc_nav()

        # Daily loss check (3%)
        if self._daily_start_nav > 0:
            daily_loss = (nav - self._daily_start_nav) / self._daily_start_nav
            if daily_loss < -0.03:
                self._halt("daily loss > 3%")
                return

        # Drawdown from capital (15%)
        dd = (nav - self.capital) / self.capital
        if dd < -0.15:
            self._halt("drawdown > 15%")
            return

        # Consecutive losses (5)
        if self._consecutive_losses >= 5:
            self._halt("5 consecutive losses")
            return

    def _halt(self, reason: str) -> None:
        self._halted = True
        self._halt_reason = reason
        self._log(f"KILL SWITCH: {reason} — trading halted")
        self._set_phase(f"HALTED: {reason}")
        self._persist_state()

    def resume(self) -> None:
        """Resume trading after halt (manual action)."""
        self._halted = False
        self._halt_reason = ""
        self._consecutive_losses = 0
        self._log("Trading resumed")

    # ─── EOD ────────────────────────────────────────────────────────────────

    def _log_eod(self, today_str: str) -> None:
        """Log end-of-day NAV snapshot."""
        nav = self._calc_nav()
        pos_value = self._calc_positions_value()
        append_nav_log(today_str, self.cash, pos_value, nav, len(self.positions), NAV_LOG_FILE)
        self._eod_logged_date = today_str
        self._persist_state()

        day_ret = (nav - self._daily_start_nav) / self._daily_start_nav * 100 if self._daily_start_nav > 0 else 0
        total_ret = (nav - self.capital) / self.capital * 100
        self._log(
            f"EOD: NAV {nav:,.0f}  Day {day_ret:+.2f}%  Total {total_ret:+.2f}%  "
            f"{len(self.positions)} positions"
        )

    # ─── Stats for dashboard display ────────────────────────────────────────

    def get_portfolio_stats(self) -> dict:
        """Return portfolio stats dict compatible with dashboard's _populate_portfolio_tab."""
        positions = []
        total_cost = total_value = 0.0
        sector_exposure = {}

        for pos in self.positions.values():
            cur = pos.last_ltp or pos.entry_price
            cost = pos.cost_basis
            val = pos.shares * cur
            pnl = val - cost
            ret = pnl / cost * 100 if cost > 0 else 0
            total_cost += cost
            total_value += val

            sector = get_symbol_sector(pos.symbol) or "Other"
            sector_exposure[sector] = sector_exposure.get(sector, 0) + val

            from backend.trading.live_trader import count_trading_days_since
            days = count_trading_days_since(pos.entry_date)

            positions.append({
                "sym": pos.symbol, "qty": pos.shares,
                "entry": pos.entry_price, "cur": cur,
                "cost": cost, "val": val, "pnl": pnl, "ret": ret,
                "prev_close": cur, "day_pnl": 0.0, "day_ret": 0.0,
                "signal": pos.signal_type, "date": pos.entry_date,
                "days": days, "sector": sector,
            })

        nav = self.cash + total_value
        day_pnl = nav - self._daily_start_nav
        day_ret = (day_pnl / self._daily_start_nav * 100) if self._daily_start_nav > 0 else 0.0
        total_return = (nav - self.capital) / self.capital * 100

        positions.sort(key=lambda x: x["val"], reverse=True)
        top3_conc = sum(p["val"] for p in positions[:3]) / total_value * 100 if total_value > 0 else 0
        winners = [p for p in positions if p["pnl"] > 0]
        losers = [p for p in positions if p["pnl"] < 0]

        # Load trade log for stats
        tl = pd.DataFrame()
        if Path(TRADE_LOG_FILE).exists():
            try:
                tl = pd.read_csv(TRADE_LOG_FILE)
            except Exception:
                pass

        # Load NAV log
        nl = pd.DataFrame()
        if Path(NAV_LOG_FILE).exists():
            try:
                nl = pd.read_csv(NAV_LOG_FILE)
            except Exception:
                pass

        # Max drawdown from NAV log
        max_dd = 0.0
        dd_date = ""
        peak_nav = self.capital
        if not nl.empty and "NAV" in nl.columns:
            for _, row in nl.iterrows():
                n = float(row["NAV"])
                if n > peak_nav:
                    peak_nav = n
                dd = (n - peak_nav) / peak_nav * 100
                if dd < max_dd:
                    max_dd = dd
                    dd_date = str(row.get("Date", ""))

        # Holding age buckets
        age_0_5 = sum(1 for p in positions if p["days"] <= 5)
        age_6_15 = sum(1 for p in positions if 6 <= p["days"] <= 15)
        age_16 = sum(1 for p in positions if p["days"] > 15)

        return {
            "positions": positions,
            "total_cost": total_cost, "total_value": total_value,
            "cash": self.cash, "nav": nav, "total_return": total_return,
            "day_pnl": day_pnl, "day_ret": day_ret,
            "realized": 0, "unrealized": total_value - total_cost,
            "max_dd": max_dd, "dd_date": dd_date, "peak_nav": peak_nav,
            "nepse_ret": 0, "alpha": total_return,
            "n_positions": len(positions),
            "sector_exposure": sector_exposure,
            "top3_conc": top3_conc,
            "winners": winners, "losers": losers,
            "age_0_5": age_0_5, "age_6_15": age_6_15, "age_16": age_16,
            "trade_log": tl, "nav_log": nl,
        }

    def get_trade_log(self) -> pd.DataFrame:
        """Return the trade log DataFrame."""
        if Path(TRADE_LOG_FILE).exists():
            try:
                return pd.read_csv(TRADE_LOG_FILE)
            except Exception:
                pass
        return pd.DataFrame()
