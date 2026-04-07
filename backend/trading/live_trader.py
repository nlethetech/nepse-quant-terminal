#!/usr/bin/env python3
"""
NEPSE Live Paper Trading Bot — Rich TUI + Interactive Telegram Bot

Monitors live prices during NEPSE regular hours (11 AM – 3 PM NPT, Mon–Fri),
generates signals at market open, manages paper positions with risk exits,
and displays everything in a Rich terminal dashboard.

The interactive Telegram bot provides two-way control: view portfolio,
signals, status, and execute manual buy/sell orders with inline buttons.

Usage:
    python live_trader.py                          # Today's session (TUI + Telegram)
    python live_trader.py --continuous              # Run indefinitely
    python live_trader.py --headless                # No TUI (Telegram-only, for VPS)
    python live_trader.py --capital 1000000         # Set capital
    python live_trader.py --signals volume,quality,low_vol
    python live_trader.py --refresh-secs 300        # Price refresh interval
    python live_trader.py --no-telegram             # TUI only, no Telegram
    python live_trader.py --dry-run                 # No portfolio writes
    python live_trader.py --paper-portfolio paper_portfolio.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from backend.quant_pro.config import (
    DEFAULT_CAPITAL,
    HARD_STOP_LOSS_PCT,
    MAX_POSITIONS,
    RECOMMENDED_HOLDING_DAYS,
    TAKE_PROFIT_PCT,
    TRAILING_STOP_PCT,
)
from backend.quant_pro.database import (
    get_db_path,
    load_benchmark_history,
    load_latest_market_quotes,
    save_benchmark_history,
)
from backend.quant_pro.data_io import SECTOR_INDEX_SYMBOLS
from backend.quant_pro.event_layer import (
    load_event_adjustment_context,
    refresh_daily_event_layer,
)
from backend.quant_pro.message_formatters import (
    format_live_order_summary_html,
    format_trade_activity_line,
)
from backend.quant_pro.nepse_calendar import is_trading_day, is_today_trading_day
from backend.quant_pro.paths import (
    ensure_dir,
    get_project_root,
    get_runtime_dir,
    get_trading_runtime_dir,
    migrate_legacy_path,
)
from backend.quant_pro.signal_ranking import rank_signal_candidates
from backend.quant_pro.control_plane.command_service import build_live_trader_control_plane
from backend.quant_pro.vendor_api import (
    fetch_latest_ltp,
    fetch_latest_ltps,
    fetch_ohlcv_chunk,
    get_latest_ltps_context,
)
from backend.quant_pro.tms_audit import (
    load_execution_intent,
    load_latest_live_orders,
    load_latest_live_positions,
    load_latest_tms_snapshot,
    mark_intent_notified,
    save_tms_snapshot,
    update_execution_intent,
)
from backend.quant_pro.tms_executor import (
    LocalTMSExecutionService,
    TMSBrowserExecutor,
)
from backend.quant_pro.tms_models import (
    ExecutionAction,
    ExecutionIntent,
    ExecutionResult,
    ExecutionSource,
    ExecutionStatus,
    FillState,
    utc_now_iso,
)
from backend.quant_pro.tms_session import load_tms_settings
from backend.quant_pro.tms_session import validate_live_setup

# Signal generators from simple_backtest
from backend.backtesting.simple_backtest import (
    compute_market_regime,
    generate_low_volatility_signals_at_date,
    generate_momentum_signals_at_date,
    generate_quality_signals_at_date,
    generate_volume_breakout_signals_at_date,
    generate_52wk_high_signals_at_date,
    apply_amihud_tilt,
    get_symbol_sector,
    load_all_prices,
)
from backend.quant_pro.disposition import generate_cgo_signals_at_date
from backend.quant_pro.lead_lag import generate_sector_leadlag_signals_at_date
from backend.quant_pro.macro_signals import get_nrb_policy_regime
from validation.transaction_costs import TransactionCostModel as NepseFees
from validation.kill_switch import KillSwitch

PROJECT_ROOT = get_project_root(__file__)
RUNTIME_DIR = ensure_dir(get_runtime_dir(__file__))
TRADING_RUNTIME_DIR = ensure_dir(get_trading_runtime_dir(__file__))
LOGS_DIR = ensure_dir(RUNTIME_DIR / "logs")
LIVE_TRADER_LOG_PATH = migrate_legacy_path(
    LOGS_DIR / "live_trader.log",
    [PROJECT_ROOT / "live_trader.log", PROJECT_ROOT / "backend.trading.live_trader.log"],
)
DEFAULT_PAPER_PORTFOLIO = migrate_legacy_path(
    TRADING_RUNTIME_DIR / "paper_portfolio.csv",
    [PROJECT_ROOT / "paper_portfolio.csv"],
)
DEFAULT_NAV_LOG = migrate_legacy_path(
    TRADING_RUNTIME_DIR / "paper_nav_log.csv",
    [PROJECT_ROOT / "paper_nav_log.csv"],
)
DEFAULT_STATE_FILE = migrate_legacy_path(
    TRADING_RUNTIME_DIR / "paper_state.json",
    [PROJECT_ROOT / "paper_state.json"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.FileHandler(LIVE_TRADER_LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Nepal is UTC+5:45
NST_OFFSET = timedelta(hours=5, minutes=45)

MARKET_OPEN_HOUR = 11   # 11:00 AM NST
MARKET_CLOSE_HOUR = 15  # 3:00 PM NST


# ─────────────────────────────────────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────────────────────────────────────

def now_nst() -> datetime:
    """Current time in Nepal Standard Time (naive datetime)."""
    from datetime import timezone
    utc_now = datetime.now(timezone.utc)
    return (utc_now + NST_OFFSET).replace(tzinfo=None)


def is_market_open() -> bool:
    """Check if NEPSE market is currently open."""
    nst = now_nst()
    if not is_trading_day(nst.date()):
        return False
    return MARKET_OPEN_HOUR <= nst.hour < MARKET_CLOSE_HOUR


def time_until_open() -> Optional[timedelta]:
    """Time until next market open (None if market is open)."""
    nst = now_nst()
    today = nst.date()
    if is_trading_day(today) and nst.hour < MARKET_OPEN_HOUR:
        open_time = datetime(today.year, today.month, today.day, MARKET_OPEN_HOUR, 0)
        return open_time - nst
    return None


def time_until_close() -> Optional[timedelta]:
    """Time until market close (None if market is closed)."""
    nst = now_nst()
    today = nst.date()
    if is_market_open():
        close_time = datetime(today.year, today.month, today.day, MARKET_CLOSE_HOUR, 0)
        return close_time - nst
    return None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _price_deviation_pct(trader: Any, default: float = 2.5) -> float:
    settings = getattr(trader, "live_settings", None)
    raw = getattr(settings, "max_price_deviation_pct", default)
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return float(default)


def estimate_execution_price(
    ltp: float,
    *,
    is_buy: bool,
    max_price_deviation_pct: float = 2.5,
) -> float:
    """Estimate a realistic executable price from the observed LTP."""
    if ltp <= 0:
        return 0.0
    tick = 0.1
    band = min(0.003, max(0.0, float(max_price_deviation_pct)) / 100.0)
    if is_buy:
        raw = ltp * (1.0 + band)
    else:
        raw = max(tick, ltp * (1.0 - band))
    return round(round(raw / tick) * tick, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio persistence (CSV compatible with paper_trade_tracker.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    shares: int
    entry_price: float
    entry_date: str       # YYYY-MM-DD
    buy_fees: float
    signal_type: str
    high_watermark: float  # Peak LTP since entry
    last_ltp: Optional[float] = None
    last_ltp_source: Optional[str] = None
    last_ltp_time_utc: Optional[str] = None
    entry_trading_days: int = 0  # Trading days held

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price + self.buy_fees

    @property
    def market_value(self) -> float:
        price = self.last_ltp if self.last_ltp else self.entry_price
        return self.shares * price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def pnl_pct(self) -> float:
        basis = self.cost_basis
        return (self.unrealized_pnl / basis) if basis > 0 else 0.0


@dataclass
class TradeRecord:
    date: str
    action: str           # BUY / SELL
    symbol: str
    shares: int
    price: float
    fees: float
    reason: str
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class TradeLot:
    symbol: str
    strategy: str
    sector: str
    shares: int
    entry_date: date
    entry_price: float
    buy_fees: float
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None
    sell_fees: float = 0.0
    mark_price: Optional[float] = None
    mark_source: Optional[str] = None
    mark_time_utc: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def cost_basis(self) -> float:
        return float(self.shares) * float(self.entry_price) + float(self.buy_fees)

    @property
    def end_price(self) -> Optional[float]:
        if self.is_open:
            return self.mark_price
        return self.exit_price

    @property
    def terminal_value(self) -> float:
        if self.is_open:
            price = self.mark_price or self.entry_price
            return float(self.shares) * float(price)
        price = self.exit_price or self.entry_price
        return float(self.shares) * float(price) - float(self.sell_fees)

    @property
    def total_fees(self) -> float:
        return float(self.buy_fees) + float(self.sell_fees)

    @property
    def pnl(self) -> float:
        return self.terminal_value - self.cost_basis

    @property
    def return_pct(self) -> float:
        basis = self.cost_basis
        return ((self.terminal_value / basis) - 1.0) * 100.0 if basis > 0 else 0.0


PORTFOLIO_COLS = [
    "Entry_Date", "Symbol", "Quantity", "Buy_Price",
    "Buy_Amount", "Buy_Fees", "Total_Cost_Basis",
    "Signal_Type", "High_Watermark", "Last_LTP", "Last_LTP_Source",
    "Last_LTP_Time_UTC",
]

TRADE_LOG_COLS = [
    "Date", "Action", "Symbol", "Shares", "Price",
    "Fees", "Reason", "PnL", "PnL_Pct",
]

NAV_LOG_COLS = ["Date", "Cash", "Positions_Value", "NAV", "Num_Positions"]

_BENCHMARK_CACHE: Dict[Tuple[str, str, str], Tuple[float, Dict[str, Any]]] = {}
_BENCHMARK_CACHE_TTL_SECS = 900
_BENCHMARK_NO_DATA_CACHE: Dict[Tuple[str, str, str], float] = {}
_BENCHMARK_NO_DATA_TTL_SECS = 1800
_SPARKLINE_BARS = "▁▂▃▄▅▆▇█"


def _ensure_parent_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    ensure_dir(p.parent)
    return p


def _derive_companion_paths(portfolio_path: str | Path) -> tuple[Path, Path, Path, Path]:
    portfolio = Path(portfolio_path).expanduser()
    ensure_dir(portfolio.parent)
    name = portfolio.name
    trade_name = name.replace("portfolio", "trade_log") if "portfolio" in name else f"{portfolio.stem}_trade_log{portfolio.suffix}"
    nav_name = name.replace("portfolio", "nav_log") if "portfolio" in name else f"{portfolio.stem}_nav_log{portfolio.suffix}"
    if name.endswith("portfolio.csv"):
        state_name = name.replace("portfolio.csv", "state.json")
    elif "portfolio" in name:
        state_name = name.replace("portfolio", "state")
        if "." not in state_name:
            state_name = f"{state_name}.json"
    else:
        state_name = f"{portfolio.stem}_state.json"
    return portfolio, portfolio.with_name(trade_name), portfolio.with_name(nav_name), portfolio.with_name(state_name)


def _signal_marker_path(marker_dir: str | Path, when: Optional[datetime] = None) -> Path:
    stamp = (when or now_nst()).strftime("%Y%m%d")
    return Path(marker_dir) / f".signals_{stamp}"


def load_portfolio(path: str) -> Dict[str, Position]:
    """Load open positions from CSV."""
    positions: Dict[str, Position] = {}
    p = Path(path).expanduser()
    if not p.exists():
        return positions

    df = pd.read_csv(p)
    for _, row in df.iterrows():
        sym = str(row["Symbol"])
        last_ltp = None
        raw_last_ltp = row.get("Last_LTP")
        try:
            if pd.notna(raw_last_ltp):
                parsed_last_ltp = float(raw_last_ltp)
                if parsed_last_ltp > 0:
                    last_ltp = parsed_last_ltp
        except (TypeError, ValueError):
            last_ltp = None
        last_ltp_source = row.get("Last_LTP_Source")
        if pd.isna(last_ltp_source):
            last_ltp_source = None
        last_ltp_time_utc = row.get("Last_LTP_Time_UTC")
        if pd.isna(last_ltp_time_utc):
            last_ltp_time_utc = None
        positions[sym] = Position(
            symbol=sym,
            shares=int(row["Quantity"]),
            entry_price=float(row["Buy_Price"]),
            entry_date=str(row["Entry_Date"]),
            buy_fees=float(row.get("Buy_Fees", 0)),
            signal_type=str(row.get("Signal_Type", "")),
            high_watermark=float(row.get("High_Watermark", row["Buy_Price"])),
            last_ltp=last_ltp,
            last_ltp_source=str(last_ltp_source) if last_ltp_source else None,
            last_ltp_time_utc=str(last_ltp_time_utc) if last_ltp_time_utc else None,
        )
    return positions


def save_portfolio(positions: Dict[str, Position], path: str) -> None:
    """Write open positions to CSV."""
    target = _ensure_parent_path(path)
    rows = []
    for pos in positions.values():
        rows.append({
            "Entry_Date": pos.entry_date,
            "Symbol": pos.symbol,
            "Quantity": pos.shares,
            "Buy_Price": pos.entry_price,
            "Buy_Amount": pos.shares * pos.entry_price,
            "Buy_Fees": pos.buy_fees,
            "Total_Cost_Basis": pos.cost_basis,
            "Signal_Type": pos.signal_type,
            "High_Watermark": pos.high_watermark,
            "Last_LTP": pos.last_ltp,
            "Last_LTP_Source": pos.last_ltp_source,
            "Last_LTP_Time_UTC": pos.last_ltp_time_utc,
        })
    df = pd.DataFrame(rows, columns=PORTFOLIO_COLS)
    df.to_csv(target, index=False)


def append_trade_log(record: TradeRecord, path: str) -> None:
    """Append a trade to the trade log CSV."""
    target = _ensure_parent_path(path)
    write_header = not target.exists()
    with open(target, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_LOG_COLS)
        if write_header:
            w.writeheader()
        w.writerow({
            "Date": record.date,
            "Action": record.action,
            "Symbol": record.symbol,
            "Shares": record.shares,
            "Price": record.price,
            "Fees": record.fees,
            "Reason": record.reason,
            "PnL": record.pnl,
            "PnL_Pct": record.pnl_pct,
        })


def append_nav_log(date_str: str, cash: float, positions_value: float,
                   nav: float, num_positions: int, path: str) -> None:
    """Append daily NAV snapshot."""
    target = _ensure_parent_path(path)
    write_header = not target.exists()
    with open(target, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NAV_LOG_COLS)
        if write_header:
            w.writeheader()
        w.writerow({
            "Date": date_str,
            "Cash": round(cash, 2),
            "Positions_Value": round(positions_value, 2),
            "NAV": round(nav, 2),
            "Num_Positions": num_positions,
        })


def load_runtime_state(path: str) -> Dict[str, Any]:
    """Load persisted runtime state (cash, refresh metadata) from JSON."""
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def save_runtime_state(path: str, state: Dict[str, Any]) -> None:
    """Persist runtime state atomically."""
    p = _ensure_parent_path(path)
    payload = json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(p)


def resolve_daily_start_nav(nav_log_path: str, *, fallback_nav: float) -> float:
    """Use the latest prior NAV row as today's baseline when available."""
    try:
        nav_df = load_nav_log_df(nav_log_path)
    except Exception:
        nav_df = pd.DataFrame(columns=NAV_LOG_COLS)
    if nav_df.empty:
        return float(fallback_nav)
    today = now_nst().date()
    prior_rows = nav_df[nav_df["Date"].dt.date < today]
    if prior_rows.empty:
        return float(fallback_nav)
    try:
        return float(prior_rows.iloc[-1]["NAV"])
    except Exception:
        return float(fallback_nav)


def load_trade_log(path: str, limit: int = 10) -> List[TradeRecord]:
    """Load last N trades from the trade log."""
    p = Path(path)
    if not p.exists():
        return []
    df = pd.read_csv(p)
    records = []
    for _, row in df.tail(limit).iterrows():
        records.append(TradeRecord(
            date=str(row.get("Date", "")),
            action=str(row.get("Action", "")),
            symbol=str(row.get("Symbol", "")),
            shares=int(row.get("Shares", 0)),
            price=float(row.get("Price", 0)),
            fees=float(row.get("Fees", 0)),
            reason=str(row.get("Reason", "")),
            pnl=float(row.get("PnL", 0)),
            pnl_pct=float(row.get("PnL_Pct", 0)),
        ))
    return records


def load_trade_log_df(path: str) -> pd.DataFrame:
    """Load the full trade log as a normalized DataFrame."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=TRADE_LOG_COLS)
    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=TRADE_LOG_COLS)
    df["Action"] = df["Action"].astype(str).str.upper()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ("Shares", "Price", "Fees", "PnL", "PnL_Pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def calculate_cash_from_trade_log(capital: float, trade_log_path: str) -> Optional[float]:
    """Rebuild cash from historical trade flows so realized P&L survives restarts."""
    df = load_trade_log_df(trade_log_path)
    if df.empty:
        return None

    cash = float(capital)
    for _, row in df.iterrows():
        gross = float(row["Shares"]) * float(row["Price"])
        fees = float(row["Fees"])
        action = str(row["Action"]).upper()
        if action == "BUY":
            cash -= gross + fees
        elif action == "SELL":
            cash += gross - fees
    return round(cash, 2)


def compute_deployed_performance(
    capital: float,
    positions: Dict[str, Position],
    trade_log_path: str,
) -> Optional[Dict[str, Any]]:
    """Compute cash-neutral strategy performance from realized + unrealized P&L."""
    trades = load_trade_log_df(trade_log_path)
    if trades.empty:
        return None

    buy_trades = trades[trades["Action"] == "BUY"].copy()
    if buy_trades.empty:
        return None

    sell_trades = trades[trades["Action"] == "SELL"].copy()
    open_cost_basis = sum(pos.cost_basis for pos in positions.values())
    open_market_value = sum(pos.market_value for pos in positions.values())
    open_unrealized_pnl = open_market_value - open_cost_basis
    realized_pnl = float(sell_trades["PnL"].sum()) if not sell_trades.empty else 0.0
    total_pnl = realized_pnl + open_unrealized_pnl

    gross_buys_with_fees = float((buy_trades["Shares"] * buy_trades["Price"] + buy_trades["Fees"]).sum())
    if gross_buys_with_fees <= 0:
        return None

    start_date = pd.Timestamp(buy_trades["Date"].min()).date()
    start_day_buys = buy_trades[buy_trades["Date"].dt.date == start_date]
    initial_deployed = float((start_day_buys["Shares"] * start_day_buys["Price"] + start_day_buys["Fees"]).sum())
    if initial_deployed <= 0:
        initial_deployed = gross_buys_with_fees

    return {
        "start_date": start_date,
        "open_cost_basis": open_cost_basis,
        "open_market_value": open_market_value,
        "open_unrealized_pnl": open_unrealized_pnl,
        "open_return_pct": ((open_market_value / open_cost_basis) - 1.0) * 100.0 if open_cost_basis > 0 else 0.0,
        "realized_pnl": realized_pnl,
        "realized_return_pct": (realized_pnl / initial_deployed) * 100.0,
        "total_pnl": total_pnl,
        "gross_buys_with_fees": gross_buys_with_fees,
        "deployed_return_pct": (total_pnl / initial_deployed) * 100.0,
        "initial_deployed": initial_deployed,
        "capital": capital,
    }


def compute_benchmark_return(
    benchmark: str,
    start_date: date,
    end_date: Optional[date] = None,
) -> Optional[Dict[str, Any]]:
    """Compute a benchmark return from locally cached daily history."""
    end_date = end_date or now_nst().date()
    benchmark = str(benchmark).strip().upper()
    if not benchmark:
        return None

    cache_key = (benchmark, start_date.isoformat(), end_date.isoformat())
    cached = _BENCHMARK_CACHE.get(cache_key)
    now_ts = time.monotonic()
    if cached and (now_ts - cached[0]) < _BENCHMARK_CACHE_TTL_SECS:
        return cached[1]

    try:
        df = ensure_benchmark_history(benchmark, start_date, end_date)
        if df is None or df.empty:
            return None
        df = df.sort_values("Date").copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        base_candidates = df[df["Date"] <= start_date]
        latest_candidates = df[df["Date"] <= end_date]
        if base_candidates.empty or latest_candidates.empty:
            return None
        base_row = base_candidates.iloc[-1]
        latest_row = latest_candidates.iloc[-1]
        result = {
            "benchmark": benchmark,
            "base_date": base_row["Date"],
            "base_close": float(base_row["Close"]),
            "latest_date": latest_row["Date"],
            "latest_close": float(latest_row["Close"]),
            "return_pct": ((float(latest_row["Close"]) / float(base_row["Close"])) - 1.0) * 100.0,
        }
        _BENCHMARK_CACHE[cache_key] = (now_ts, result)
        return result
    except Exception as e:
        logger.warning("Failed to compute benchmark return for %s: %s", benchmark, e)
        return None


def ensure_benchmark_history(
    benchmark: str,
    start_date: date,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Load benchmark history from SQLite, fetching and persisting gaps if needed."""
    end_date = end_date or now_nst().date()
    benchmark = str(benchmark).strip().upper()
    if not benchmark:
        return pd.DataFrame()
    local = load_benchmark_history(benchmark, start_date=start_date, end_date=end_date)
    local_dates = set(pd.to_datetime(local["Date"]).dt.date) if not local.empty else set()

    if not local.empty and start_date in local_dates and end_date in local_dates:
        return local

    miss_key = (benchmark, start_date.isoformat(), end_date.isoformat())
    miss_ts = _BENCHMARK_NO_DATA_CACHE.get(miss_key)
    now_ts = time.monotonic()
    if miss_ts is not None and (now_ts - miss_ts) < _BENCHMARK_NO_DATA_TTL_SECS:
        return local

    try:
        start_fetch = datetime.combine(start_date - timedelta(days=7), datetime.min.time())
        end_fetch = datetime.combine(end_date, datetime.min.time())
        fetched = fetch_ohlcv_chunk(benchmark, int(start_fetch.timestamp()), int(end_fetch.timestamp()))
        if fetched is not None and not fetched.empty:
            _BENCHMARK_NO_DATA_CACHE.pop(miss_key, None)
            save_benchmark_history(
                benchmark,
                fetched.to_dict("records"),
                source="merolagani_chart",
            )
            local = load_benchmark_history(benchmark, start_date=start_date, end_date=end_date)
        else:
            _BENCHMARK_NO_DATA_CACHE[miss_key] = now_ts
    except Exception as e:
        logger.warning("Failed to refresh benchmark history for %s: %s", benchmark, e)
    return local


def compute_nepse_benchmark_return(start_date: date, end_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
    """Fetch NEPSE index history and compute the return over matching dates."""
    return compute_benchmark_return("NEPSE", start_date, end_date=end_date)


def compute_portfolio_vs_nepse(
    capital: float,
    positions: Dict[str, Position],
    trade_log_path: str,
    cash: Optional[float] = None,
    nav_log_path: str = str(DEFAULT_NAV_LOG),
) -> Optional[Dict[str, Any]]:
    """Compare deployed-capital strategy return against NEPSE on matching dates."""
    inferred_cash = float(cash) if cash is not None else calculate_cash_from_trade_log(capital, trade_log_path)
    if inferred_cash is None:
        inferred_cash = max(0.0, capital - sum(pos.cost_basis for pos in positions.values()))
    intelligence = compute_portfolio_intelligence(
        capital,
        inferred_cash,
        positions,
        trade_log_path,
        nav_log_path,
    )
    if intelligence is not None:
        performance = intelligence["performance"]
        benchmark = intelligence.get("global_benchmark")
        alpha_pct = None
        if benchmark is not None and not intelligence["data_quality"].get("freeze_alpha"):
            alpha_pct = performance["deployed_return_pct"] - benchmark["return_pct"]
        return {
            "performance": performance,
            "benchmark": benchmark,
            "alpha_pct": alpha_pct,
            "data_quality": intelligence["data_quality"],
            "attribution_stack": intelligence["attribution_stack"],
            "positions": intelligence["positions"],
        }

    performance = compute_deployed_performance(capital, positions, trade_log_path)
    if performance is None:
        return None

    benchmark = compute_nepse_benchmark_return(performance["start_date"])
    if benchmark is None:
        return {
            "performance": performance,
            "benchmark": None,
            "alpha_pct": None,
        }

    return {
        "performance": performance,
        "benchmark": benchmark,
        "alpha_pct": performance["deployed_return_pct"] - benchmark["return_pct"],
    }


def compute_position_vs_nepse(pos: Position) -> Optional[Dict[str, Any]]:
    """Compare a single holding's price appreciation against NEPSE since entry."""
    if pos.last_ltp is None or pos.entry_price <= 0:
        return None
    try:
        entry_date = datetime.strptime(pos.entry_date, "%Y-%m-%d").date()
    except ValueError:
        return None
    benchmark = compute_nepse_benchmark_return(entry_date)
    stock_return_pct = ((pos.last_ltp / pos.entry_price) - 1.0) * 100.0
    result = {
        "stock_return_pct": stock_return_pct,
        "benchmark": benchmark,
        "alpha_pct": None,
    }
    if benchmark is not None:
        result["alpha_pct"] = stock_return_pct - benchmark["return_pct"]
    return result


def compute_position_benchmark_summary(
    pos: Position,
    *,
    initial_deployed: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Compare one open holding against both sector and NEPSE benchmarks."""
    if pos.last_ltp is None or pos.entry_price <= 0:
        return None
    try:
        entry_date = datetime.strptime(pos.entry_date, "%Y-%m-%d").date()
    except ValueError:
        return None
    sector = get_symbol_sector(pos.symbol) or "Unknown"
    sector_benchmark_symbol = SECTOR_INDEX_SYMBOLS.get(sector)
    sector_benchmark = compute_benchmark_return(sector_benchmark_symbol, entry_date) if sector_benchmark_symbol else None
    nepse_benchmark = compute_nepse_benchmark_return(entry_date)
    stock_return_pct = pos.pnl_pct * 100.0
    active_vs_nepse = stock_return_pct - nepse_benchmark["return_pct"] if nepse_benchmark else None
    active_vs_sector = stock_return_pct - sector_benchmark["return_pct"] if sector_benchmark else None
    contribution_vs_nepse = None
    if active_vs_nepse is not None and initial_deployed and initial_deployed > 0:
        contribution_vs_nepse = (pos.cost_basis / initial_deployed) * active_vs_nepse
    return {
        "sector": sector,
        "stock_return_pct": stock_return_pct,
        "nepse_benchmark": nepse_benchmark,
        "sector_benchmark": sector_benchmark,
        "active_vs_nepse_pct": active_vs_nepse,
        "active_vs_sector_pct": active_vs_sector,
        "contribution_vs_nepse_pts": contribution_vs_nepse,
    }


def compute_sector_attribution(positions: Dict[str, Position]) -> List[Dict[str, Any]]:
    """Aggregate open-book P&L and returns by sector."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for symbol, pos in positions.items():
        sector = get_symbol_sector(symbol) or "Unknown"
        bucket = buckets.setdefault(
            sector,
            {
                "sector": sector,
                "cost_basis": 0.0,
                "market_value": 0.0,
                "pnl": 0.0,
                "entry_dates": [],
                "symbols": [],
            },
        )
        bucket["cost_basis"] += pos.cost_basis
        bucket["market_value"] += pos.market_value
        bucket["pnl"] += pos.unrealized_pnl
        if pos.entry_date:
            bucket["entry_dates"].append(pos.entry_date)
        bucket["symbols"].append(symbol)

    results = list(buckets.values())
    for item in results:
        cost_basis = item["cost_basis"]
        item["return_pct"] = ((item["market_value"] / cost_basis) - 1.0) * 100.0 if cost_basis > 0 else 0.0
        item["symbols"] = sorted(item["symbols"])
        benchmark_symbol = SECTOR_INDEX_SYMBOLS.get(item["sector"])
        item["benchmark"] = None
        item["alpha_pct"] = None
        if benchmark_symbol and item["entry_dates"]:
            try:
                start_date = min(
                    datetime.strptime(entry_date, "%Y-%m-%d").date()
                    for entry_date in item["entry_dates"]
                )
            except ValueError:
                start_date = None
            if start_date is not None:
                benchmark = compute_benchmark_return(benchmark_symbol, start_date)
                if benchmark:
                    item["benchmark"] = benchmark
                    item["alpha_pct"] = item["return_pct"] - benchmark["return_pct"]
        item.pop("entry_dates", None)
    results.sort(key=lambda item: item["pnl"], reverse=True)
    return results


def compute_strategy_attribution(
    positions: Dict[str, Position],
    trade_log_path: str,
) -> List[Dict[str, Any]]:
    """Aggregate realized and unrealized P&L by strategy tag."""
    trades = load_trade_log_df(trade_log_path)
    buckets: Dict[str, Dict[str, Any]] = {}
    open_strategy_by_symbol: Dict[str, str] = {}

    def _bucket(tag: str) -> Dict[str, Any]:
        clean = str(tag or "unknown").strip() or "unknown"
        return buckets.setdefault(
            clean,
            {
                "strategy": clean,
                "buy_cost": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "open_cost_basis": 0.0,
                "symbols": set(),
            },
        )

    if not trades.empty:
        for _, row in trades.iterrows():
            action = str(row.get("Action", "")).upper()
            symbol = str(row.get("Symbol", "")).strip().upper()
            if not symbol:
                continue
            if action == "BUY":
                tag = str(row.get("Reason", "")).strip() or "unknown"
                bucket = _bucket(tag)
                bucket["buy_cost"] += float(row.get("Shares", 0.0)) * float(row.get("Price", 0.0)) + float(row.get("Fees", 0.0))
                bucket["symbols"].add(symbol)
                open_strategy_by_symbol[symbol] = tag
            elif action == "SELL":
                tag = open_strategy_by_symbol.pop(symbol, "unknown")
                bucket = _bucket(tag)
                bucket["realized_pnl"] += float(row.get("PnL", 0.0))
                bucket["symbols"].add(symbol)

    for symbol, pos in positions.items():
        tag = str(pos.signal_type or open_strategy_by_symbol.get(symbol) or "unknown").strip() or "unknown"
        bucket = _bucket(tag)
        bucket["unrealized_pnl"] += pos.unrealized_pnl
        bucket["open_cost_basis"] += pos.cost_basis
        bucket["symbols"].add(symbol)

    results = list(buckets.values())
    for item in results:
        item["total_pnl"] = item["realized_pnl"] + item["unrealized_pnl"]
        deployed = item["buy_cost"] or item["open_cost_basis"]
        item["return_pct"] = (item["total_pnl"] / deployed) * 100.0 if deployed > 0 else 0.0
        item["symbols"] = sorted(item["symbols"])
    results.sort(key=lambda item: item["total_pnl"], reverse=True)
    return results


def _parse_utc_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def previous_trading_day(reference_day: date) -> date:
    """Return the most recent NEPSE trading day before `reference_day`."""
    current = reference_day - timedelta(days=1)
    while not is_trading_day(current):
        current -= timedelta(days=1)
    return current


def _load_latest_quote_details(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    clean_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if not clean_symbols:
        return {}
    try:
        return load_latest_market_quotes(clean_symbols, max_age_seconds=7 * 24 * 60 * 60)
    except Exception as exc:
        logger.warning("Failed to load latest quote details: %s", exc)
        return {}


def reconstruct_trade_lots(
    positions: Dict[str, Position],
    trade_log_path: str,
) -> List[TradeLot]:
    """Rebuild realized and open lots from the trade log plus current marks."""
    trades = load_trade_log_df(trade_log_path)
    quote_details = _load_latest_quote_details(list(positions.keys()))
    open_queues: Dict[str, List[Dict[str, Any]]] = {}
    closed_lots: List[TradeLot] = []

    if not trades.empty:
        for _, row in trades.iterrows():
            symbol = str(row.get("Symbol", "")).strip().upper()
            if not symbol:
                continue
            action = str(row.get("Action", "")).upper()
            shares = int(float(row.get("Shares", 0.0) or 0.0))
            if shares <= 0:
                continue
            trade_date = pd.Timestamp(row["Date"]).date()
            price = float(row.get("Price", 0.0) or 0.0)
            fees = float(row.get("Fees", 0.0) or 0.0)
            if action == "BUY":
                open_queues.setdefault(symbol, []).append(
                    {
                        "symbol": symbol,
                        "strategy": str(row.get("Reason", "") or "unknown").strip() or "unknown",
                        "sector": get_symbol_sector(symbol) or "Unknown",
                        "entry_date": trade_date,
                        "entry_price": price,
                        "remaining_shares": shares,
                        "remaining_buy_fees": fees,
                    }
                )
                continue

            if action != "SELL":
                continue

            sell_remaining = shares
            sell_fees_remaining = fees
            queue = open_queues.get(symbol, [])
            while sell_remaining > 0 and queue:
                open_lot = queue[0]
                available = int(open_lot["remaining_shares"])
                matched = min(available, sell_remaining)
                if matched <= 0:
                    queue.pop(0)
                    continue
                buy_fee_alloc = float(open_lot["remaining_buy_fees"]) * (matched / available) if available > 0 else 0.0
                sell_fee_alloc = sell_fees_remaining * (matched / sell_remaining) if sell_remaining > 0 else 0.0
                closed_lots.append(
                    TradeLot(
                        symbol=symbol,
                        strategy=str(open_lot["strategy"]),
                        sector=str(open_lot["sector"]),
                        shares=matched,
                        entry_date=open_lot["entry_date"],
                        entry_price=float(open_lot["entry_price"]),
                        buy_fees=buy_fee_alloc,
                        exit_date=trade_date,
                        exit_price=price,
                        sell_fees=sell_fee_alloc,
                    )
                )
                open_lot["remaining_shares"] = available - matched
                open_lot["remaining_buy_fees"] = max(0.0, float(open_lot["remaining_buy_fees"]) - buy_fee_alloc)
                sell_remaining -= matched
                sell_fees_remaining = max(0.0, sell_fees_remaining - sell_fee_alloc)
                if int(open_lot["remaining_shares"]) <= 0:
                    queue.pop(0)

    open_lots: List[TradeLot] = []
    for symbol, position in positions.items():
        queue = open_queues.get(symbol, [])
        quote = quote_details.get(symbol, {})
        if queue:
            for open_lot in queue:
                if int(open_lot["remaining_shares"]) <= 0:
                    continue
                open_lots.append(
                    TradeLot(
                        symbol=symbol,
                        strategy=str(open_lot["strategy"]),
                        sector=str(open_lot["sector"]),
                        shares=int(open_lot["remaining_shares"]),
                        entry_date=open_lot["entry_date"],
                        entry_price=float(open_lot["entry_price"]),
                        buy_fees=float(open_lot["remaining_buy_fees"]),
                        mark_price=position.last_ltp,
                        mark_source=position.last_ltp_source,
                        mark_time_utc=position.last_ltp_time_utc or quote.get("fetched_at_utc"),
                    )
                )
        else:
            open_lots.append(
                TradeLot(
                    symbol=symbol,
                    strategy=str(position.signal_type or "unknown"),
                    sector=get_symbol_sector(symbol) or "Unknown",
                    shares=int(position.shares),
                    entry_date=datetime.strptime(position.entry_date, "%Y-%m-%d").date(),
                    entry_price=float(position.entry_price),
                    buy_fees=float(position.buy_fees),
                    mark_price=position.last_ltp,
                    mark_source=position.last_ltp_source,
                    mark_time_utc=position.last_ltp_time_utc or quote.get("fetched_at_utc"),
                )
            )

    all_lots = closed_lots + open_lots
    all_lots.sort(key=lambda lot: (lot.entry_date, lot.symbol, 0 if not lot.is_open else 1))
    return all_lots


def _determine_common_benchmark_date(
    benchmark_histories: Dict[str, pd.DataFrame],
) -> Tuple[Optional[date], List[str]]:
    issues: List[str] = []
    latest_dates: Dict[str, date] = {}
    for benchmark, df in benchmark_histories.items():
        if df is None or df.empty:
            issues.append(f"missing {benchmark} history")
            continue
        latest_dates[benchmark] = pd.Timestamp(df["Date"].iloc[-1]).date()
    if not latest_dates:
        return None, issues or ["no benchmark history"]
    unique_dates = sorted(set(latest_dates.values()))
    if len(unique_dates) > 1:
        issues.append(
            "benchmark date mismatch: "
            + ", ".join(f"{bench}={latest_dates[bench]}" for bench in sorted(latest_dates))
        )
    return min(unique_dates), issues


def _aligned_lot_end_price(
    lot: TradeLot,
    *,
    benchmark_date: Optional[date],
    quote_details: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[float], Optional[str], Optional[str], bool]:
    """Return a benchmark-aligned mark price, source, time, and whether alignment used a fallback."""
    if not lot.is_open:
        if lot.exit_price is None:
            return None, None, None, False
        return lot.exit_price, "realized_exit", None, False

    if lot.mark_price is None:
        return None, None, None, False

    quote = quote_details.get(lot.symbol, {})
    mark_time = _parse_utc_datetime(lot.mark_time_utc)
    mark_date = mark_time.date() if mark_time else (benchmark_date or now_nst().date())
    if benchmark_date is None or mark_date <= benchmark_date:
        return lot.mark_price, lot.mark_source, lot.mark_time_utc, False

    previous_close = quote.get("previous_close")
    try:
        previous_close = float(previous_close) if previous_close is not None else None
    except (TypeError, ValueError):
        previous_close = None
    if previous_close and previous_close > 0:
        fetched_at = quote.get("fetched_at_utc") or lot.mark_time_utc
        return previous_close, "previous_close_alignment", fetched_at, True
    return None, None, None, False


def _compute_alpha_data_quality(
    positions: Dict[str, Position],
    *,
    benchmark_histories: Dict[str, pd.DataFrame],
    common_benchmark_date: Optional[date],
) -> Dict[str, Any]:
    from datetime import timezone

    now_dt = datetime.now(timezone.utc)
    stale_threshold_secs = max(1800, 3 * 60 * 5) if is_market_open() else 36 * 3600
    stale_symbols: List[str] = []
    missing_time_symbols: List[str] = []
    fallback_symbols: List[str] = []
    cache_symbols: List[str] = []
    primary_symbols: List[str] = []
    source_counts: Dict[str, int] = {}
    mark_dates: set[date] = set()

    for symbol, pos in positions.items():
        source = str(pos.last_ltp_source or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        if source == "nepalstock":
            primary_symbols.append(symbol)
        elif source == "merolagani":
            fallback_symbols.append(symbol)
        elif source == "sqlite_cache":
            cache_symbols.append(symbol)

        parsed_time = _parse_utc_datetime(pos.last_ltp_time_utc)
        if parsed_time is None:
            missing_time_symbols.append(symbol)
            continue
        mark_dates.add(parsed_time.date())
        age_secs = max(0.0, (now_dt - parsed_time.astimezone(timezone.utc)).total_seconds())
        if age_secs > stale_threshold_secs:
            stale_symbols.append(symbol)

    benchmark_issues: List[str] = []
    latest_benchmark_dates: Dict[str, str] = {}
    for benchmark, df in benchmark_histories.items():
        if df is None or df.empty:
            benchmark_issues.append(f"missing {benchmark}")
            continue
        latest_date = pd.Timestamp(df["Date"].iloc[-1]).date()
        latest_benchmark_dates[benchmark] = latest_date.isoformat()
        if common_benchmark_date is not None and latest_date != common_benchmark_date:
            benchmark_issues.append(f"{benchmark}={latest_date}")

    confidence_score = 100
    confidence_score -= 18 * len(stale_symbols)
    confidence_score -= 10 * len(missing_time_symbols)
    confidence_score -= 8 * len(fallback_symbols)
    confidence_score -= 4 * len(cache_symbols)
    confidence_score -= 20 if benchmark_issues else 0
    confidence_score -= 8 if len(mark_dates) > 1 else 0
    confidence_score = max(0, min(100, confidence_score))

    alerts: List[str] = []
    if stale_symbols:
        alerts.append(f"stale marks: {', '.join(sorted(stale_symbols))}")
    if missing_time_symbols:
        alerts.append(f"missing mark timestamps: {', '.join(sorted(missing_time_symbols))}")
    if fallback_symbols and primary_symbols:
        alerts.append(
            f"mixed mark sources: fallback {', '.join(sorted(fallback_symbols))} while others are primary"
        )
    elif fallback_symbols:
        alerts.append(f"fallback marks in use: {', '.join(sorted(fallback_symbols))}")
    if benchmark_issues:
        alerts.append("benchmark mismatch: " + ", ".join(sorted(benchmark_issues)))

    return {
        "confidence_score": confidence_score,
        "stale_symbols": sorted(stale_symbols),
        "missing_time_symbols": sorted(missing_time_symbols),
        "fallback_symbols": sorted(fallback_symbols),
        "cache_symbols": sorted(cache_symbols),
        "primary_symbols": sorted(primary_symbols),
        "source_counts": source_counts,
        "latest_benchmark_dates": latest_benchmark_dates,
        "benchmark_issues": benchmark_issues,
        "alerts": alerts,
        "freeze_alpha": bool(benchmark_issues),
        "common_benchmark_date": common_benchmark_date.isoformat() if common_benchmark_date else None,
    }


def compute_portfolio_intelligence(
    capital: float,
    cash: float,
    positions: Dict[str, Position],
    trade_log_path: str,
    nav_log_path: str,
) -> Optional[Dict[str, Any]]:
    """Build a benchmark-aligned analytics snapshot for alpha and risk views."""
    performance = compute_deployed_performance(capital, positions, trade_log_path)
    if performance is None:
        return None

    lots = reconstruct_trade_lots(positions, trade_log_path)
    if not lots:
        return None

    strategy_start = performance["start_date"]
    today = now_nst().date()
    required_benchmarks = {"NEPSE"}
    for lot in lots:
        sector_benchmark = SECTOR_INDEX_SYMBOLS.get(lot.sector)
        if sector_benchmark:
            required_benchmarks.add(str(sector_benchmark).strip().upper())

    benchmark_histories: Dict[str, pd.DataFrame] = {}
    for benchmark in sorted(required_benchmarks):
        benchmark_histories[benchmark] = ensure_benchmark_history(benchmark, strategy_start, today)

    common_benchmark_date, benchmark_date_issues = _determine_common_benchmark_date(benchmark_histories)
    quote_details = _load_latest_quote_details(sorted(positions.keys()))

    global_benchmark = compute_benchmark_return("NEPSE", strategy_start, end_date=common_benchmark_date or today)
    if global_benchmark is None:
        return None

    data_quality = _compute_alpha_data_quality(
        positions,
        benchmark_histories=benchmark_histories,
        common_benchmark_date=common_benchmark_date,
    )
    if benchmark_date_issues:
        for issue in benchmark_date_issues:
            if issue not in data_quality["alerts"]:
                data_quality["alerts"].append(issue)
        data_quality["freeze_alpha"] = True

    initial_buy_fees = 0.0
    turnover_fees = 0.0
    turnover_notional = 0.0
    initial_buy_notional = 0.0
    trades = load_trade_log_df(trade_log_path)
    if not trades.empty:
        for _, row in trades.iterrows():
            trade_date = pd.Timestamp(row["Date"]).date()
            gross = float(row["Shares"]) * float(row["Price"])
            fees = float(row["Fees"])
            if trade_date == strategy_start and str(row["Action"]).upper() == "BUY":
                initial_buy_fees += fees
                initial_buy_notional += gross
            else:
                turnover_fees += fees
                turnover_notional += gross
    alpha_initial_deployed = initial_buy_notional if initial_buy_notional > 0 else max(
        performance["initial_deployed"] - initial_buy_fees,
        0.0,
    )

    lots_rows: List[Dict[str, Any]] = []
    realized_alpha_pnl = 0.0
    unrealized_alpha_pnl = 0.0
    stock_selection_alpha_pnl = 0.0
    sector_allocation_alpha_pnl = 0.0
    timing_alpha_pnl = 0.0
    gross_realized_strategy_pnl = 0.0
    gross_unrealized_strategy_pnl = 0.0

    for lot in lots:
        end_date = lot.exit_date or (common_benchmark_date or today)
        if end_date < lot.entry_date:
            end_date = lot.entry_date
        aligned_price, aligned_source, aligned_time, used_alignment_fallback = _aligned_lot_end_price(
            lot,
            benchmark_date=common_benchmark_date,
            quote_details=quote_details,
        )
        if aligned_price is None:
            data_quality["freeze_alpha"] = True
            data_quality["alerts"].append(f"unaligned mark for {lot.symbol}")
            continue

        sector_benchmark_symbol = SECTOR_INDEX_SYMBOLS.get(lot.sector)
        sector_benchmark = compute_benchmark_return(
            sector_benchmark_symbol,
            lot.entry_date,
            end_date=end_date,
        ) if sector_benchmark_symbol else None
        nepse_benchmark = compute_benchmark_return("NEPSE", lot.entry_date, end_date=end_date)
        strategy_window_nepse = compute_benchmark_return("NEPSE", strategy_start, end_date=end_date)
        if nepse_benchmark is None:
            data_quality["freeze_alpha"] = True
            data_quality["alerts"].append(f"missing NEPSE benchmark for {lot.symbol}")
            continue

        entry_notional = float(lot.shares) * float(lot.entry_price)
        aligned_terminal_value = float(lot.shares) * float(aligned_price)
        aligned_return_pct = ((aligned_terminal_value / entry_notional) - 1.0) * 100.0 if entry_notional > 0 else 0.0
        sector_return_pct = sector_benchmark["return_pct"] if sector_benchmark else 0.0
        nepse_return_pct = nepse_benchmark["return_pct"]
        strategy_window_nepse_pct = strategy_window_nepse["return_pct"] if strategy_window_nepse else nepse_return_pct
        active_vs_sector_pct = aligned_return_pct - sector_return_pct if sector_benchmark else None
        active_vs_nepse_pct = aligned_return_pct - nepse_return_pct
        active_vs_sector_pnl = entry_notional * (active_vs_sector_pct / 100.0) if active_vs_sector_pct is not None else None
        active_vs_nepse_pnl = entry_notional * (active_vs_nepse_pct / 100.0)
        sector_allocation_pnl = entry_notional * ((sector_return_pct - nepse_return_pct) / 100.0) if sector_benchmark else 0.0
        timing_pnl = entry_notional * ((nepse_return_pct - strategy_window_nepse_pct) / 100.0)
        contribution_vs_nepse_pts = (entry_notional / alpha_initial_deployed) * active_vs_nepse_pct if alpha_initial_deployed > 0 else 0.0
        contribution_vs_sector_pts = (
            (entry_notional / alpha_initial_deployed) * active_vs_sector_pct
            if alpha_initial_deployed > 0 and active_vs_sector_pct is not None
            else None
        )
        gross_lot_pnl = aligned_terminal_value - entry_notional

        row = {
            "symbol": lot.symbol,
            "strategy": lot.strategy,
            "sector": lot.sector,
            "shares": lot.shares,
            "entry_date": lot.entry_date.isoformat(),
            "end_date": end_date.isoformat(),
            "is_open": lot.is_open,
            "cost_basis": lot.cost_basis,
            "entry_notional": entry_notional,
            "aligned_price": aligned_price,
            "aligned_source": aligned_source,
            "aligned_time_utc": aligned_time,
            "used_alignment_fallback": used_alignment_fallback,
            "aligned_return_pct": aligned_return_pct,
            "sector_return_pct": sector_return_pct if sector_benchmark else None,
            "nepse_return_pct": nepse_return_pct,
            "active_vs_sector_pct": active_vs_sector_pct,
            "active_vs_nepse_pct": active_vs_nepse_pct,
            "active_vs_sector_pnl": active_vs_sector_pnl,
            "active_vs_nepse_pnl": active_vs_nepse_pnl,
            "contribution_vs_nepse_pts": contribution_vs_nepse_pts,
            "contribution_vs_sector_pts": contribution_vs_sector_pts,
            "realized_pnl": lot.pnl if not lot.is_open else 0.0,
            "unrealized_pnl": lot.pnl if lot.is_open else 0.0,
            "mark_source": lot.mark_source,
            "mark_time_utc": lot.mark_time_utc,
        }
        lots_rows.append(row)

        realized_alpha_pnl += 0.0 if lot.is_open else active_vs_nepse_pnl
        unrealized_alpha_pnl += active_vs_nepse_pnl if lot.is_open else 0.0
        stock_selection_alpha_pnl += active_vs_sector_pnl or 0.0
        sector_allocation_alpha_pnl += sector_allocation_pnl
        timing_alpha_pnl += timing_pnl
        gross_realized_strategy_pnl += 0.0 if lot.is_open else gross_lot_pnl
        gross_unrealized_strategy_pnl += gross_lot_pnl if lot.is_open else 0.0

    if not lots_rows:
        return None

    nav_df = load_nav_log_df(nav_log_path)
    nav_since_start = nav_df[nav_df["Date"].dt.date >= strategy_start].copy() if not nav_df.empty else pd.DataFrame()
    current_row = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp(now_nst().date()),
                "Cash": cash,
                "Positions_Value": sum(pos.market_value for pos in positions.values()),
                "NAV": cash + sum(pos.market_value for pos in positions.values()),
                "Num_Positions": len(positions),
            }
        ]
    )
    if nav_since_start.empty:
        nav_since_start = current_row
    else:
        nav_since_start = pd.concat([nav_since_start, current_row], ignore_index=True)
        nav_since_start = nav_since_start.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    average_cash = float(nav_since_start["Cash"].mean()) if "Cash" in nav_since_start else float(cash)
    cash_drag_pnl = -average_cash * (global_benchmark["return_pct"] / 100.0)
    gross_total_strategy_pnl = gross_realized_strategy_pnl + gross_unrealized_strategy_pnl
    gross_strategy_return_pct = (gross_total_strategy_pnl / alpha_initial_deployed) * 100.0 if alpha_initial_deployed > 0 else 0.0
    gross_alpha_return_pct = gross_strategy_return_pct - global_benchmark["return_pct"]
    gross_alpha_pnl = (
        stock_selection_alpha_pnl
        + sector_allocation_alpha_pnl
        + timing_alpha_pnl
        + cash_drag_pnl
    )
    alpha_cost_drag_pnl = -(turnover_fees + initial_buy_fees)
    net_alpha_pnl = gross_alpha_pnl + alpha_cost_drag_pnl

    positions_rows = [row for row in lots_rows if row["is_open"]]
    positions_rows.sort(key=lambda row: row["contribution_vs_nepse_pts"], reverse=True)

    attribution_stack = {
        "stock_selection_alpha_pnl": stock_selection_alpha_pnl,
        "sector_allocation_alpha_pnl": sector_allocation_alpha_pnl,
        "timing_alpha_pnl": timing_alpha_pnl,
        "cash_drag_pnl": cash_drag_pnl,
        "turnover_drag_pnl": -turnover_fees,
        "fee_drag_pnl": -initial_buy_fees,
        "realized_alpha_pnl": realized_alpha_pnl,
        "unrealized_alpha_pnl": unrealized_alpha_pnl,
        "gross_realized_strategy_pnl": gross_realized_strategy_pnl,
        "gross_unrealized_strategy_pnl": gross_unrealized_strategy_pnl,
        "gross_total_strategy_pnl": gross_total_strategy_pnl,
        "gross_strategy_return_pct": gross_strategy_return_pct,
        "gross_alpha_return_pct": gross_alpha_return_pct,
        "gross_alpha_pnl": gross_alpha_pnl,
        "alpha_cost_drag_pnl": alpha_cost_drag_pnl,
        "net_alpha_pnl": net_alpha_pnl,
        "turnover_ratio_pct": (turnover_notional / alpha_initial_deployed) * 100.0 if alpha_initial_deployed > 0 else 0.0,
        "alpha_initial_deployed": alpha_initial_deployed,
        "total_fees_paid": initial_buy_fees + turnover_fees,
    }

    return {
        "performance": performance,
        "global_benchmark": global_benchmark,
        "data_quality": data_quality,
        "positions": positions_rows,
        "lots": lots_rows,
        "attribution_stack": attribution_stack,
        "benchmark_histories": benchmark_histories,
        "common_benchmark_date": common_benchmark_date,
    }


def load_nav_log_df(path: str) -> pd.DataFrame:
    """Load NAV history as a normalized DataFrame."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=NAV_LOG_COLS)
    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=NAV_LOG_COLS)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ("Cash", "Positions_Value", "NAV"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return df


def _build_sparkline(values: List[float]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return _SPARKLINE_BARS[-1]
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin <= 1e-12:
        return _SPARKLINE_BARS[0] * len(values)
    chars = []
    for value in values:
        scaled = (value - vmin) / (vmax - vmin)
        idx = min(len(_SPARKLINE_BARS) - 1, max(0, int(round(scaled * (len(_SPARKLINE_BARS) - 1)))))
        chars.append(_SPARKLINE_BARS[idx])
    return "".join(chars)


def _build_direction_strip(values: List[float]) -> str:
    """Render day-over-day direction using colored square emoji."""
    if not values:
        return ""
    if len(values) == 1:
        return "⬜"
    parts = []
    previous = values[0]
    for value in values[1:]:
        delta = value - previous
        if delta > 1e-9:
            parts.append("🟩")
        elif delta < -1e-9:
            parts.append("🟥")
        else:
            parts.append("⬜")
        previous = value
    return "".join(parts)


def compute_deployed_nav_chart_data(
    capital: float,
    positions: Dict[str, Position],
    trade_log_path: str,
    nav_log_path: str,
) -> Optional[Dict[str, Any]]:
    """Build a normalized deployed-sleeve-vs-NEPSE chart data structure."""
    perf = compute_deployed_performance(capital, positions, trade_log_path)
    if perf is None:
        return None

    nav_df = load_nav_log_df(nav_log_path)
    if nav_df.empty:
        return None

    baseline_cash = capital - perf["initial_deployed"]
    nav_df = nav_df.copy()
    nav_df["Date"] = pd.to_datetime(nav_df["Date"]).dt.date
    nav_df["Deployed_NAV"] = nav_df["Positions_Value"] + (nav_df["Cash"] - baseline_cash).clip(lower=0.0)

    today = now_nst().date()
    current_deployed_nav = perf["initial_deployed"] + perf["total_pnl"]
    current_row = pd.DataFrame(
        [
            {
                "Date": today,
                "Deployed_NAV": current_deployed_nav,
            }
        ]
    )
    nav_df = pd.concat([nav_df[["Date", "Deployed_NAV"]], current_row], ignore_index=True)
    nav_df = nav_df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    nav_df = nav_df[nav_df["Date"] >= perf["start_date"]]
    if nav_df.empty:
        return None

    benchmark_df = ensure_benchmark_history("NEPSE", perf["start_date"], today)
    if benchmark_df.empty:
        return None
    benchmark_df = benchmark_df.copy()
    benchmark_df["Date"] = pd.to_datetime(benchmark_df["Date"]).dt.date
    benchmark_df = benchmark_df.sort_values("Date")

    merged_rows = []
    for _, row in nav_df.iterrows():
        bench_subset = benchmark_df[benchmark_df["Date"] <= row["Date"]]
        if bench_subset.empty:
            continue
        bench_row = bench_subset.iloc[-1]
        merged_rows.append(
            {
                "date": row["Date"],
                "deployed_nav": float(row["Deployed_NAV"]),
                "benchmark_close": float(bench_row["Close"]),
            }
        )
    if not merged_rows:
        return None

    sleeve_base = merged_rows[0]["deployed_nav"]
    bench_base = merged_rows[0]["benchmark_close"]
    for item in merged_rows:
        item["sleeve_index"] = (item["deployed_nav"] / sleeve_base) * 100.0 if sleeve_base > 0 else 100.0
        item["benchmark_index"] = (item["benchmark_close"] / bench_base) * 100.0 if bench_base > 0 else 100.0

    sleeve_values = [item["sleeve_index"] for item in merged_rows]
    benchmark_values = [item["benchmark_index"] for item in merged_rows]
    return {
        "dates": [item["date"] for item in merged_rows],
        "sleeve_index": sleeve_values,
        "benchmark_index": benchmark_values,
        "sleeve_sparkline": _build_sparkline(sleeve_values),
        "benchmark_sparkline": _build_sparkline(benchmark_values),
        "sleeve_direction_strip": _build_direction_strip(sleeve_values),
        "benchmark_direction_strip": _build_direction_strip(benchmark_values),
        "latest_sleeve": sleeve_values[-1],
        "latest_benchmark": benchmark_values[-1],
    }


def compute_risk_snapshot(
    capital: float,
    cash: float,
    positions: Dict[str, Position],
    nav_log_path: str,
) -> Dict[str, Any]:
    """Summarize current risk: concentration, exposures, and drawdowns."""
    nav = float(cash) + sum(pos.market_value for pos in positions.values())
    nav_denom = nav if nav > 0 else (float(capital) if capital > 0 else 1.0)

    position_rows = []
    sector_buckets: Dict[str, Dict[str, Any]] = {}
    signal_buckets: Dict[str, Dict[str, Any]] = {}
    holding_buckets: Dict[str, Dict[str, Any]] = {}
    winner_exposure_value = 0.0
    laggard_exposure_value = 0.0
    for symbol, pos in positions.items():
        market_value = pos.market_value
        weight_pct = (market_value / nav_denom) * 100.0
        holding_days = count_trading_days_since(pos.entry_date)
        position_rows.append(
            {
                "symbol": symbol,
                "market_value": market_value,
                "weight_pct": weight_pct,
                "pnl": pos.unrealized_pnl,
                "return_pct": pos.pnl_pct * 100.0,
                "signal_type": pos.signal_type,
                "holding_days": holding_days,
            }
        )
        sector = get_symbol_sector(symbol) or "Unknown"
        bucket = sector_buckets.setdefault(
            sector,
            {"sector": sector, "market_value": 0.0, "pnl": 0.0},
        )
        bucket["market_value"] += market_value
        bucket["pnl"] += pos.unrealized_pnl
        signal_key = str(pos.signal_type or "unknown")
        signal_bucket = signal_buckets.setdefault(
            signal_key,
            {"signal_type": signal_key, "market_value": 0.0, "count": 0},
        )
        signal_bucket["market_value"] += market_value
        signal_bucket["count"] += 1

        if holding_days <= 5:
            holding_key = "0-5d"
        elif holding_days <= 15:
            holding_key = "6-15d"
        else:
            holding_key = "16d+"
        holding_bucket = holding_buckets.setdefault(
            holding_key,
            {"holding_bucket": holding_key, "market_value": 0.0, "count": 0},
        )
        holding_bucket["market_value"] += market_value
        holding_bucket["count"] += 1

        if pos.unrealized_pnl >= 0:
            winner_exposure_value += market_value
        else:
            laggard_exposure_value += market_value

    position_rows.sort(key=lambda item: item["weight_pct"], reverse=True)
    sector_rows = list(sector_buckets.values())
    for item in sector_rows:
        item["weight_pct"] = (item["market_value"] / nav_denom) * 100.0
    sector_rows.sort(key=lambda item: item["weight_pct"], reverse=True)
    signal_rows = list(signal_buckets.values())
    for item in signal_rows:
        item["weight_pct"] = (item["market_value"] / nav_denom) * 100.0
    signal_rows.sort(key=lambda item: item["weight_pct"], reverse=True)
    holding_rows = list(holding_buckets.values())
    for item in holding_rows:
        item["weight_pct"] = (item["market_value"] / nav_denom) * 100.0
    holding_rows.sort(key=lambda item: item["weight_pct"], reverse=True)

    nav_df = load_nav_log_df(nav_log_path)
    peak_nav = nav
    peak_date = now_nst().date()
    rolling_drawdown_5d = 0.0
    rolling_drawdown_20d = 0.0
    if not nav_df.empty:
        peak_idx = nav_df["NAV"].idxmax()
        historical_peak_nav = float(nav_df.loc[peak_idx, "NAV"])
        historical_peak_date = pd.Timestamp(nav_df.loc[peak_idx, "Date"]).date()
        if historical_peak_nav > peak_nav:
            peak_nav = historical_peak_nav
            peak_date = historical_peak_date
        nav_values = list(nav_df["NAV"].astype(float))
        nav_values.append(nav)
        if nav_values:
            window_5 = nav_values[-5:]
            window_20 = nav_values[-20:]
            peak_5 = max(window_5)
            peak_20 = max(window_20)
            rolling_drawdown_5d = ((nav / peak_5) - 1.0) * 100.0 if peak_5 > 0 else 0.0
            rolling_drawdown_20d = ((nav / peak_20) - 1.0) * 100.0 if peak_20 > 0 else 0.0
    drawdown_pct = ((nav / peak_nav) - 1.0) * 100.0 if peak_nav > 0 else 0.0

    winners = sorted(position_rows, key=lambda item: item["pnl"], reverse=True)
    losers = sorted(position_rows, key=lambda item: item["pnl"])
    alerts: List[str] = []
    if position_rows and position_rows[0]["weight_pct"] > 18.0:
        alerts.append(f"single-name concentration high: {position_rows[0]['symbol']} {position_rows[0]['weight_pct']:.1f}%")
    if sector_rows and sector_rows[0]["weight_pct"] > 35.0:
        alerts.append(f"sector concentration high: {sector_rows[0]['sector']} {sector_rows[0]['weight_pct']:.1f}%")
    if rolling_drawdown_20d < -8.0:
        alerts.append(f"20-session drawdown elevated: {rolling_drawdown_20d:.1f}%")
    return {
        "nav": nav,
        "cash": cash,
        "cash_weight_pct": (float(cash) / nav_denom) * 100.0,
        "positions": position_rows,
        "sectors": sector_rows,
        "signals": signal_rows,
        "holding_buckets": holding_rows,
        "top_positions": position_rows[:3],
        "top_sectors": sector_rows[:3],
        "winners": winners[:3],
        "losers": losers[:3],
        "peak_nav": peak_nav,
        "peak_date": peak_date,
        "drawdown_pct": drawdown_pct,
        "rolling_drawdown_5d": rolling_drawdown_5d,
        "rolling_drawdown_20d": rolling_drawdown_20d,
        "winner_exposure_pct": (winner_exposure_value / nav_denom) * 100.0,
        "laggard_exposure_pct": (laggard_exposure_value / nav_denom) * 100.0,
        "alerts": alerts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trading day counter (reused from paper_trade_tracker logic)
# ─────────────────────────────────────────────────────────────────────────────

def count_trading_days_since(entry_date_str: str) -> int:
    """Count NEPSE trading days from entry to today (exclusive of entry day)."""
    entry = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
    today = now_nst().date()
    count = 0
    current = entry + timedelta(days=1)
    while current <= today:
        if is_trading_day(current):
            count += 1
        current += timedelta(days=1)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Signal generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(
    prices_df: pd.DataFrame,
    signal_types: List[str],
    use_regime_filter: bool = True,
) -> Tuple[List[dict], str]:
    """Generate buy signals at current date using historical price data.

    Includes: volume breakout, quality, low_vol, mean_reversion (core) +
    CGO disposition, lead-lag sector spillover, 52-week high proximity,
    Amihud illiquidity tilt, NRB macro overlay (extended signals).

    Returns (signals_list, regime_str).
    """
    today = pd.Timestamp(now_nst().date())

    regime = compute_market_regime(prices_df, today)
    if use_regime_filter and regime == "bear":
        logger.info("Bear regime detected — skipping signal generation")
        return [], regime

    # Regime confidence multiplier (matches generate_daily_signals.py logic)
    regime_mult = {"bull": 1.15, "neutral": 0.90}.get(regime, 1.0)

    # NRB monetary policy rate overlay
    nrb_mult = 1.0
    nrb_sector_adj: dict = {}
    try:
        nrb = get_nrb_policy_regime(db_path=str(get_db_path()))
        nrb_mult = nrb["multiplier"]
        nrb_sector_adj = nrb.get("sector_adjustments", {})
        if nrb["cycle"] != "no_data":
            logger.info(
                "NRB policy: %s (rate=%.1f%%, %+.0fbps, mult=%.2f)",
                nrb["cycle"].upper(), nrb["latest_rate_pct"] or 0,
                nrb["rate_change_bps"], nrb_mult,
            )
    except Exception as e:
        logger.warning("NRB policy regime failed (skipping): %s", e)

    all_signals = []

    # Core signals
    if "momentum" in signal_types:
        try:
            all_signals.extend(generate_momentum_signals_at_date(prices_df, today))
        except Exception as e:
            logger.warning("Momentum signals failed: %s", e)
    if "volume" in signal_types:
        try:
            all_signals.extend(generate_volume_breakout_signals_at_date(prices_df, today))
        except Exception as e:
            logger.warning("Volume signals failed: %s", e)
    if "low_vol" in signal_types:
        try:
            all_signals.extend(generate_low_volatility_signals_at_date(prices_df, today))
        except Exception as e:
            logger.warning("Low-vol signals failed: %s", e)
    if "quality" in signal_types:
        try:
            all_signals.extend(generate_quality_signals_at_date(prices_df, today))
        except Exception as e:
            logger.warning("Quality signals failed: %s", e)

    # Extended signals — all fail-open
    try:
        cgo = generate_cgo_signals_at_date(prices_df, today)
        logger.info("CGO signals: %d", len(cgo))
        all_signals.extend(cgo)
    except Exception as e:
        logger.warning("CGO signals failed (skipping): %s", e)

    try:
        ll = generate_sector_leadlag_signals_at_date(prices_df, today)
        logger.info("Lead-lag signals: %d", len(ll))
        all_signals.extend(ll)
    except Exception as e:
        logger.warning("Lead-lag signals failed (skipping): %s", e)

    try:
        wk52 = generate_52wk_high_signals_at_date(prices_df, today)
        logger.info("52Wk-high signals: %d", len(wk52))
        all_signals.extend(wk52)
    except Exception as e:
        logger.warning("52Wk-high signals failed (skipping): %s", e)

    # Amihud illiquidity tilt (overlay — modifies strength in-place)
    try:
        all_signals = apply_amihud_tilt(all_signals, prices_df, today)
    except Exception as e:
        logger.warning("Amihud tilt failed (skipping): %s", e)

    # Apply regime + NRB confidence multipliers
    combined_mult = regime_mult * nrb_mult
    for sig in all_signals:
        sig.confidence = min(sig.confidence * combined_mult, 0.95)
        sector = get_symbol_sector(sig.symbol)
        sector_mult = nrb_sector_adj.get(sector, 1.0)
        if abs(sector_mult - 1.0) > 1e-6:
            sig.confidence = min(sig.confidence * sector_mult, 0.95)

    # Rank by strength * confidence
    all_signals.sort(key=lambda s: s.strength * s.confidence, reverse=True)

    results = []
    for sig in all_signals:
        results.append({
            "symbol": sig.symbol,
            "signal_type": sig.signal_type.value,
            "strength": sig.strength,
            "confidence": sig.confidence,
            "reasoning": sig.reasoning,
            "score": sig.strength * sig.confidence,
        })

    return results, regime


# ─────────────────────────────────────────────────────────────────────────────
# Price fetching (batch)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices_for_symbols(symbols: List[str]) -> Dict[str, Optional[float]]:
    """Fetch LTPs in one batch from the shared snapshot provider."""
    try:
        prices = fetch_latest_ltps(symbols)
    except Exception as e:
        logger.warning("Batch LTP fetch failed: %s", e)
        prices = {str(sym).strip().upper(): None for sym in symbols}

    for sym, ltp in prices.items():
        if ltp is not None:
            logger.debug("Fetched LTP %s: NPR %.1f", sym, ltp)
        else:
            logger.warning("LTP unavailable for %s", sym)
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# Risk monitoring
# ─────────────────────────────────────────────────────────────────────────────

def check_exits(
    positions: Dict[str, Position],
    holding_days: int,
) -> List[Tuple[str, str]]:
    """Check all positions for risk-based exits.

    Returns list of (symbol, reason) tuples for positions that should be sold.
    """
    exits = []
    for sym, pos in positions.items():
        if pos.last_ltp is None:
            continue

        price = pos.last_ltp
        reason = None

        # Hard stop loss
        if price < pos.entry_price * (1 - HARD_STOP_LOSS_PCT):
            reason = "stop_loss"

        # Trailing stop (only if we've moved above entry)
        if reason is None and pos.high_watermark > pos.entry_price:
            trailing_stop_price = pos.high_watermark * (1 - TRAILING_STOP_PCT)
            if price < trailing_stop_price:
                reason = "trailing_stop"

        # Take profit
        if reason is None and price >= pos.entry_price * (1 + TAKE_PROFIT_PCT):
            reason = "take_profit"

        # Holding period expiry
        if reason is None:
            days_held = count_trading_days_since(pos.entry_date)
            pos.entry_trading_days = days_held
            if days_held >= holding_days:
                reason = "holding_period"

        if reason:
            exits.append((sym, reason))

    return exits


# ─────────────────────────────────────────────────────────────────────────────
# Execution engine
# ─────────────────────────────────────────────────────────────────────────────

class LiveTrader:
    """Core trading engine that ties everything together."""

    def __init__(self, args: argparse.Namespace):
        self.capital = args.capital
        self.signal_types = args.signals.split(",")
        self.refresh_secs = args.refresh_secs
        self.no_telegram = args.no_telegram
        self.dry_run = args.dry_run
        self.continuous = args.continuous
        self.headless = args.headless
        self.max_positions = MAX_POSITIONS
        self.holding_days = RECOMMENDED_HOLDING_DAYS
        self.execution_mode = str(getattr(args, "mode", "paper") or "paper").strip().lower()
        self.live_settings = load_tms_settings()
        self.live_settings.mode = self.execution_mode
        self.live_settings.enabled = self.execution_mode in {"live", "dual", "shadow_live"} or self.live_settings.enabled
        self.live_execution_enabled = bool(self.live_settings.enabled and self.execution_mode in {"live", "dual"})
        self.dual_execution_mode = self.execution_mode == "dual"
        self.shadow_live_mode = self.execution_mode == "shadow_live"
        self.live_execution_service: Optional[LocalTMSExecutionService] = None
        self.tms_monitor_enabled = _env_flag("NEPSE_TMS_MONITOR_ENABLED", True)
        self.tms_monitor_interval_secs = max(60, _env_int("NEPSE_TMS_MONITOR_INTERVAL_SECS", 300))
        self.tms_monitor_market_hours_only = _env_flag("NEPSE_TMS_MONITOR_MARKET_HOURS_ONLY", True)
        self.tms_browser: Optional[TMSBrowserExecutor] = None
        self.tms_monitor_bundle: Dict[str, Dict[str, Any]] = {}
        self.tms_last_sync_utc: Optional[str] = None
        self.tms_last_sync_nst: Optional[datetime] = None
        self.tms_last_error: str = ""
        self._tms_monitor_lock = threading.RLock()
        self._tms_last_poll_monotonic = 0.0
        self._control_plane_service = None

        live_setup_errors = validate_live_setup(self.live_settings, interactive=not bool(self.headless))
        if live_setup_errors and self.execution_mode in {"live", "dual", "shadow_live"}:
            raise RuntimeError("; ".join(live_setup_errors))

        # File paths
        portfolio_path, trade_log_path, nav_log_path, state_path = _derive_companion_paths(args.paper_portfolio)
        self.portfolio_file = str(portfolio_path)
        self.trade_log_file = str(trade_log_path)
        self.nav_log_file = str(nav_log_path)
        self.state_file = str(state_path)
        self.signal_marker_dir = ensure_dir(portfolio_path.parent / "signal_markers")

        # Thread safety
        self._state_lock = threading.RLock()

        # State
        self.positions: Dict[str, Position] = load_portfolio(self.portfolio_file)
        self.runtime_state = load_runtime_state(self.state_file)
        self._hydrate_position_ltps()
        self.cash = self._calculate_initial_cash()
        self.signals_today: List[dict] = []
        self.activity_log: List[str] = []
        self.regime = "unknown"
        self.prices_df: Optional[pd.DataFrame] = None
        self.signals_generated_today = False
        # Check for today's signal marker file (survives restarts)
        _today_marker = _signal_marker_path(self.signal_marker_dir)
        migrate_legacy_path(_today_marker, [PROJECT_ROOT / _today_marker.name])
        if _today_marker.exists():
            self.signals_generated_today = True
            logger.info("Restored signals_generated_today=True from marker file")
        self.daily_start_nav: Optional[float] = None
        self.last_refresh: Optional[datetime] = None
        self.last_price_source_label: str = "startup"
        self.last_price_source_detail: str = ""
        self.last_price_snapshot_time_utc: Optional[str] = None
        self.live_halt_level: str = "none"
        self.live_freeze_reason: str = ""
        self.num_signals_today = 0
        self._restore_runtime_state()
        self._load_cached_tms_monitor_bundle()

        # Kill switch — auto-halt on excessive loss
        self.kill_switch = KillSwitch(
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.15,
            max_consecutive_losses=5,
            stale_data_minutes=30,
        )
        self.peak_nav: float = self.capital
        self.consecutive_losses: int = 0

        # Telegram
        if not self.no_telegram:
            from backend.quant_pro.telegram_alerts import send_alert  # noqa: F401

        if self.tms_monitor_enabled:
            try:
                self.tms_browser = TMSBrowserExecutor(self.live_settings)
            except Exception as e:
                self.tms_last_error = str(e)
                logger.warning("TMS monitor unavailable: %s", e)

        self._persist_runtime_state()
        self._log_activity("System started")

    def _restore_runtime_state(self) -> None:
        """Hydrate runtime state persisted across restarts."""
        state = self.runtime_state or {}
        saved_cash = state.get("cash")
        if isinstance(saved_cash, (int, float)):
            self.cash = float(saved_cash)
        saved_refresh = state.get("last_refresh_nst")
        if saved_refresh:
            try:
                self.last_refresh = datetime.fromisoformat(str(saved_refresh))
            except ValueError:
                self.last_refresh = None
        saved_label = state.get("last_price_source_label")
        if saved_label:
            self.last_price_source_label = str(saved_label)
        saved_detail = state.get("last_price_source_detail")
        if saved_detail:
            self.last_price_source_detail = str(saved_detail)
        saved_snapshot_time = state.get("last_price_snapshot_time_utc")
        if saved_snapshot_time:
            self.last_price_snapshot_time_utc = str(saved_snapshot_time)
        saved_daily_start_nav = state.get("daily_start_nav")
        if isinstance(saved_daily_start_nav, (int, float)):
            self.daily_start_nav = float(saved_daily_start_nav)
        self.live_halt_level = str(state.get("live_halt_level") or "none")
        self.live_freeze_reason = str(state.get("live_freeze_reason") or "")

    def _persist_runtime_state(self) -> None:
        """Persist cash and quote-source metadata for restart-safe startup."""
        cash_value = getattr(self, "cash", None)
        if cash_value is None or (
            float(cash_value) == 0.0 and getattr(self, "trade_log_file", None)
        ):
            rebuilt_cash = calculate_cash_from_trade_log(
                float(getattr(self, "capital", 0.0)),
                str(getattr(self, "trade_log_file", "")),
            )
            if rebuilt_cash is not None:
                cash_value = rebuilt_cash
        if cash_value is None:
            cash_value = 0.0
        save_runtime_state(
            self.state_file,
            {
                "cash": round(float(cash_value), 2),
                "last_refresh_nst": getattr(self, "last_refresh", None).isoformat() if getattr(self, "last_refresh", None) else None,
                "last_price_source_label": getattr(self, "last_price_source_label", "startup"),
                "last_price_source_detail": getattr(self, "last_price_source_detail", ""),
                "last_price_snapshot_time_utc": getattr(self, "last_price_snapshot_time_utc", None),
                "daily_start_nav": round(float(getattr(self, "daily_start_nav", 0.0) or 0.0), 2) if getattr(self, "daily_start_nav", None) is not None else None,
                "live_halt_level": getattr(self, "live_halt_level", "none"),
                "live_freeze_reason": getattr(self, "live_freeze_reason", ""),
                "updated_at_nst": now_nst().isoformat(),
            },
        )

    def _load_cached_tms_monitor_bundle(self) -> None:
        snapshot_types = (
            "tms_health",
            "tms_account",
            "tms_funds",
            "tms_holdings",
            "tms_orders_daily",
            "tms_orders_historic",
            "tms_trades_daily",
            "tms_trades_historic",
        )
        bundle: Dict[str, Dict[str, Any]] = {}
        for snapshot_type in snapshot_types:
            try:
                payload = load_latest_tms_snapshot(snapshot_type)
            except Exception as e:
                logger.warning("Failed to load cached %s snapshot: %s", snapshot_type, e)
                payload = None
            if payload:
                bundle[snapshot_type] = dict(payload)
        self.tms_monitor_bundle = bundle
        health = bundle.get("tms_health") or {}
        last_sync_utc = health.get("snapshot_time_utc") or health.get("snapshot_recorded_at")
        if last_sync_utc:
            self.tms_last_sync_utc = str(last_sync_utc)
            try:
                self.tms_last_sync_nst = datetime.fromisoformat(str(last_sync_utc).replace("Z", "+00:00")).replace(tzinfo=None) + NST_OFFSET
            except Exception:
                self.tms_last_sync_nst = None
        if health.get("detail") and not bool(health.get("ready", True)):
            self.tms_last_error = str(health.get("detail"))

    def _capture_price_refresh_context(self) -> None:
        """Record the last batch price-source metadata for display/debugging."""
        context = get_latest_ltps_context()
        if not context:
            return
        endpoint = context.get("snapshot_endpoint") or context.get("snapshot_source") or "unknown"
        self.last_price_source_label = str(endpoint)
        primary = int(context.get("primary_count") or 0)
        cache = int(context.get("db_cache_count") or 0)
        fallback = int(context.get("fallback_count") or 0)
        missing = len(context.get("missing_symbols") or [])
        self.last_price_source_detail = (
            f"primary={primary} cache={cache} fallback={fallback} missing={missing}"
        )
        snapshot_time = context.get("snapshot_fetched_at_utc")
        if snapshot_time:
            self.last_price_snapshot_time_utc = str(snapshot_time)

    def _apply_position_source_map(self, symbols: Optional[List[str]] = None) -> None:
        """Push the latest batch source map into position state for per-symbol display."""
        context = get_latest_ltps_context()
        source_map = context.get("source_map") if isinstance(context, dict) else None
        timestamp_map = context.get("timestamp_map") if isinstance(context, dict) else None
        if not isinstance(source_map, dict):
            return
        target_symbols = symbols or list(source_map.keys())
        for symbol in target_symbols:
            source = source_map.get(symbol)
            if source and symbol in self.positions:
                self.positions[symbol].last_ltp_source = str(source)
            if isinstance(timestamp_map, dict):
                timestamp = timestamp_map.get(symbol)
                if timestamp and symbol in self.positions:
                    self.positions[symbol].last_ltp_time_utc = str(timestamp)

    def _hydrate_position_ltps(self) -> None:
        """Restore cached LTPs on startup so restarts do not reset NAV display."""
        if not self.positions:
            return

        symbols = list(self.positions.keys())
        restored = 0

        try:
            cached_quotes = load_latest_market_quotes(symbols, max_age_seconds=7 * 24 * 60 * 60)
        except Exception as e:
            logger.warning("Failed to load cached quote snapshot: %s", e)
            cached_quotes = {}

        missing = []
        for symbol in symbols:
            pos = self.positions[symbol]
            quote = cached_quotes.get(symbol)
            price = float(quote["last_traded_price"]) if quote and quote.get("last_traded_price") else None
            if price and price > 0:
                pos.last_ltp = price
                pos.last_ltp_source = str(quote.get("source") or "sqlite_cache")
                fetched_at = quote.get("fetched_at_utc")
                if fetched_at:
                    pos.last_ltp_time_utc = str(fetched_at)
                if price > pos.high_watermark:
                    pos.high_watermark = price
                restored += 1
            else:
                missing.append(symbol)

        if missing:
            try:
                live_prices = fetch_prices_for_symbols(missing)
                self._capture_price_refresh_context()
            except Exception as e:
                logger.warning("Failed to hydrate live prices on startup: %s", e)
                live_prices = {}
            self._apply_position_source_map(missing)
            for symbol in missing:
                price = live_prices.get(symbol)
                if price and price > 0:
                    pos = self.positions[symbol]
                    pos.last_ltp = price
                    if not pos.last_ltp_time_utc:
                        pos.last_ltp_time_utc = self.last_price_snapshot_time_utc
                    if price > pos.high_watermark:
                        pos.high_watermark = price
                    restored += 1

        if restored:
            try:
                save_portfolio(self.positions, self.portfolio_file)
            except Exception as e:
                logger.warning("Failed to persist hydrated LTPs: %s", e)
            logger.info("Restored LTPs for %d/%d positions on startup", restored, len(symbols))
        self._persist_runtime_state()

    def _calculate_initial_cash(self) -> float:
        """Restore cash from actual trade history, falling back to open cost basis."""
        saved_cash = self.runtime_state.get("cash") if isinstance(self.runtime_state, dict) else None
        if isinstance(saved_cash, (int, float)):
            return float(saved_cash)
        rebuilt_cash = calculate_cash_from_trade_log(self.capital, self.trade_log_file)
        if rebuilt_cash is not None:
            return rebuilt_cash
        total_invested = sum(pos.cost_basis for pos in self.positions.values())
        return max(0, self.capital - total_invested)

    def _log_activity(self, msg: str) -> None:
        nst = now_nst()
        ts = nst.strftime("%H:%M")
        entry = f"{ts}  {msg}"
        self.activity_log.append(entry)
        # Keep last 20 entries
        if len(self.activity_log) > 20:
            self.activity_log = self.activity_log[-20:]
        logger.info(msg)

    def _send_telegram(self, func_name: str, **kwargs) -> None:
        """Call a telegram_alerts function if Telegram is enabled."""
        if self.no_telegram:
            return
        try:
            import backend.quant_pro.telegram_alerts as tg
            func = getattr(tg, func_name)
            func(**kwargs)
        except Exception as e:
            logger.warning("Telegram alert failed: %s", e)

    def _execution_snapshot(self) -> Dict[str, Any]:
        return {
            "capital": self.capital,
            "cash": self.cash,
            "positions": dict(self.positions),
            "max_positions": self.max_positions,
        }

    def get_control_plane(self):
        return build_live_trader_control_plane(self)

    def _start_live_execution_service(self) -> None:
        if not self.live_execution_enabled:
            return
        if self.live_execution_service is not None:
            return
        self.live_execution_service = LocalTMSExecutionService(
            self.live_settings,
            snapshot_provider=self._execution_snapshot,
            result_callback=self._handle_live_execution_result,
        )
        if self.live_halt_level != "none":
            self.live_execution_service.set_halt(self.live_halt_level, reason=self.live_freeze_reason)
        self.live_execution_service.start()
        self._log_activity(f"Live execution service ready ({self.execution_mode})")

    def _best_effort_limit_price(self, symbol: str, side: ExecutionAction, ltp: float) -> float:
        return estimate_execution_price(
            ltp,
            is_buy=side == ExecutionAction.BUY,
            max_price_deviation_pct=_price_deviation_pct(self),
        )

    def live_session_status(self, *, force: bool = False) -> Dict[str, Any]:
        if self.live_execution_service is None:
            return {"enabled": False, "detail": "Live execution disabled"}
        status = self.live_execution_service.session_status(force=force)
        return {
            "enabled": True,
            "ready": status.ready,
            "login_required": status.login_required,
            "url": status.current_url,
            "detail": status.detail,
            "screenshot_path": status.screenshot_path,
            "halt_level": self.live_halt_level,
            "freeze_reason": self.live_freeze_reason,
        }

    def live_mode_summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.live_execution_enabled,
            "mode": self.execution_mode,
            "strategy_automation": bool(self.live_settings.strategy_automation_enabled),
            "auto_exits": bool(self.live_settings.auto_exits_enabled),
            "owner_confirm_required": bool(self.live_settings.owner_confirm_required),
            "halt_level": self.live_halt_level,
            "freeze_reason": self.live_freeze_reason,
        }

    def _tms_snapshot_name_map(self) -> Dict[str, str]:
        return {
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

    def _cache_tms_bundle(self, bundle: Dict[str, Dict[str, Any]]) -> None:
        mapped = self._tms_snapshot_name_map()
        for key, snapshot_name in mapped.items():
            payload = bundle.get(key)
            if payload:
                self.tms_monitor_bundle[snapshot_name] = dict(payload)
                save_tms_snapshot(snapshot_name, dict(payload), status="ok")
        health = bundle.get("health") or {}
        self.tms_last_sync_utc = str(health.get("snapshot_time_utc") or utc_now_iso())
        self.tms_last_sync_nst = now_nst()
        self.tms_last_error = ""
        self._tms_last_poll_monotonic = time.monotonic()

    def _normalize_tms_trade_signature(self, row: Dict[str, Any]) -> str:
        fields = [
            str(row.get("order_no") or row.get("order_number") or row.get("trade_no") or row.get("contract_no") or "").strip(),
            str(row.get("symbol") or row.get("scrip") or row.get("scrip_name") or "").strip().upper(),
            str(row.get("qty") or row.get("quantity") or row.get("trade_qty") or "").strip(),
            str(row.get("price") or row.get("rate") or row.get("trade_price") or "").strip(),
            str(row.get("status") or row.get("type") or row.get("buy_sell") or "").strip().lower(),
            str(row.get("trade_date") or row.get("date") or row.get("business_date") or "").strip(),
        ]
        return "|".join(fields)

    def _normalize_tms_transaction_signature(self, row: Dict[str, Any]) -> str:
        fields = [
            str(row.get("date") or row.get("transaction_date") or row.get("created_at") or "").strip(),
            str(row.get("particular") or row.get("description") or row.get("remarks") or "").strip().lower(),
            str(row.get("amount") or row.get("credit") or row.get("debit") or "").strip(),
            str(row.get("voucher_no") or row.get("reference_no") or "").strip(),
        ]
        return "|".join(fields)

    def _emit_tms_monitor_alerts(self, previous: Dict[str, Dict[str, Any]], current: Dict[str, Dict[str, Any]]) -> None:
        prev_health = previous.get("tms_health") or {}
        curr_health = current.get("health") or {}
        if prev_health.get("ready") and not curr_health.get("ready"):
            self._send_telegram(
                "send_alert",
                message=(
                    "<b>TMS SESSION EXPIRED</b>\n"
                    f"<code>  Detail: {curr_health.get('detail') or 'Session no longer valid'}</code>"
                ),
            )

        prev_funds = previous.get("tms_funds") or {}
        curr_funds = current.get("funds") or {}
        prev_available = float(prev_funds.get("collateral_available") or 0.0)
        curr_available = float(curr_funds.get("collateral_available") or 0.0)
        if prev_funds and curr_funds and curr_available < (prev_available - 0.009):
            delta = curr_available - prev_available
            self._send_telegram(
                "send_alert",
                message=(
                    "<b>TMS COLLATERAL DROP</b>\n"
                    "<code>"
                    f"  Prev  : NPR {prev_available:,.2f}\n"
                    f"  Now   : NPR {curr_available:,.2f}\n"
                    f"  Delta : NPR {delta:+,.2f}"
                    "</code>"
                ),
            )

        prev_tx = {
            self._normalize_tms_transaction_signature(row)
            for row in (prev_funds.get("recent_transactions") or [])
            if isinstance(row, dict)
        }
        curr_tx_rows = [row for row in (curr_funds.get("recent_transactions") or []) if isinstance(row, dict)]
        new_tx = [row for row in curr_tx_rows if self._normalize_tms_transaction_signature(row) not in prev_tx]
        if prev_funds and new_tx:
            head = new_tx[0]
            title = str(head.get("particular") or head.get("description") or "Collateral activity").strip()
            amount = str(head.get("amount") or head.get("credit") or head.get("debit") or "-").strip()
            when = str(head.get("date") or head.get("transaction_date") or head.get("created_at") or "").strip()
            self._send_telegram(
                "send_alert",
                message=(
                    "<b>TMS REFUND/LOAD CHANGE</b>\n"
                    "<code>"
                    f"  Event  : {title[:48]}\n"
                    f"  Amount : {amount}\n"
                    f"  Time   : {when[:32]}"
                    "</code>"
                ),
            )

        prev_trade_signatures = {
            self._normalize_tms_trade_signature(row)
            for row in (previous.get("tms_trades_daily") or {}).get("records", [])
            if isinstance(row, dict)
        }
        curr_trade_rows = [row for row in (current.get("trades_daily") or {}).get("records", []) if isinstance(row, dict)]
        new_trade_rows = [
            row for row in curr_trade_rows
            if self._normalize_tms_trade_signature(row) not in prev_trade_signatures
        ]
        if previous.get("tms_trades_daily") and new_trade_rows:
            head = new_trade_rows[0]
            symbol = str(head.get("symbol") or head.get("scrip") or head.get("scrip_name") or "-").strip().upper()
            qty = str(head.get("qty") or head.get("quantity") or head.get("trade_qty") or "-").strip()
            price = str(head.get("price") or head.get("rate") or head.get("trade_price") or "-").strip()
            side = str(head.get("buy_sell") or head.get("type") or head.get("status") or "trade").strip().upper()
            msg = (
                "<b>NEW TMS TRADE</b>\n"
                "<code>"
                f"  {side:<6} {symbol}\n"
                f"  Qty   : {qty}\n"
                f"  Price : {price}"
                "</code>"
            )
            self._send_telegram("send_alert", message=msg)
            self._send_telegram("send_public_alert", message=msg)

    def refresh_tms_monitor(self, *, force: bool = False, emit_alerts: bool = False) -> Dict[str, Dict[str, Any]]:
        if not self.tms_monitor_enabled or self.tms_browser is None:
            return {}
        with self._tms_monitor_lock:
            if not force and self.tms_monitor_bundle and (time.monotonic() - self._tms_last_poll_monotonic) < self.tms_monitor_interval_secs:
                return dict(self.tms_monitor_bundle)
            previous = dict(self.tms_monitor_bundle)
            bundle = self.tms_browser.fetch_monitor_bundle()
            if not bundle:
                return previous
            self._cache_tms_bundle(bundle)
            if emit_alerts:
                try:
                    self._emit_tms_monitor_alerts(previous, bundle)
                except Exception as e:
                    logger.warning("Failed to emit TMS monitor alerts: %s", e)
            return dict(self.tms_monitor_bundle)

    def get_tms_snapshot(self, kind: str, *, force: bool = False, max_age_secs: int = 180) -> Optional[Dict[str, Any]]:
        mapping = self._tms_snapshot_name_map()
        snapshot_name = mapping.get(str(kind))
        if snapshot_name is None:
            return None
        if force:
            self.refresh_tms_monitor(force=True, emit_alerts=False)
        snapshot = self.tms_monitor_bundle.get(snapshot_name)
        if snapshot:
            return dict(snapshot)
        return load_latest_tms_snapshot(snapshot_name)

    def get_tms_monitor_bundle(self, *, force: bool = False, max_age_secs: int = 180) -> Dict[str, Dict[str, Any]]:
        if force:
            self.refresh_tms_monitor(force=True, emit_alerts=False)
        return dict(self.tms_monitor_bundle)

    def get_tms_health_summary(self, *, force: bool = False) -> Dict[str, Any]:
        health = self.get_tms_snapshot("health", force=force, max_age_secs=60) or {}
        return {
            "enabled": bool(self.tms_monitor_enabled and self.tms_browser is not None),
            "last_sync_utc": self.tms_last_sync_utc or health.get("snapshot_time_utc") or health.get("snapshot_recorded_at"),
            "last_sync_nst": self.tms_last_sync_nst.isoformat() if self.tms_last_sync_nst else None,
            "error": self.tms_last_error or health.get("detail") or "",
            **health,
        }

    def kill_live(self, level: str = "all", reason: str = "manual owner halt") -> None:
        self.live_halt_level = level
        self.live_freeze_reason = reason
        if self.live_execution_service is not None:
            self.live_execution_service.set_halt(level, reason=reason)
        self._persist_runtime_state()
        self._log_activity(f"Live execution halted ({level})")

    def resume_live(self) -> None:
        self.live_halt_level = "none"
        self.live_freeze_reason = ""
        if self.live_execution_service is not None:
            self.live_execution_service.resume()
        self._persist_runtime_state()
        self._log_activity("Live execution resumed")

    def _format_live_receipt_html(self, intent: ExecutionIntent, result: Optional[ExecutionResult] = None) -> str:
        return format_live_order_summary_html(intent, result=result)

    def create_live_owner_buy_intent(self, symbol: str, shares: int, limit_price: float) -> Tuple[bool, str, Optional[ExecutionIntent]]:
        result = self.get_control_plane().create_live_intent(
            action=str(ExecutionAction.BUY),
            symbol=str(symbol).upper(),
            quantity=int(shares),
            limit_price=float(limit_price),
            mode=self.execution_mode,
            source=str(ExecutionSource.OWNER_MANUAL),
            reason="owner_manual_buy",
            metadata={"interactive": True},
            operator_surface="owner_manual",
        )
        intent = load_execution_intent(result.intent_id) if result.intent_id else None
        return result.ok, result.message, intent

    def create_live_owner_sell_intent(self, symbol: str, shares: int, limit_price: float) -> Tuple[bool, str, Optional[ExecutionIntent]]:
        result = self.get_control_plane().create_live_intent(
            action=str(ExecutionAction.SELL),
            symbol=str(symbol).upper(),
            quantity=int(shares),
            limit_price=float(limit_price),
            mode=self.execution_mode,
            source=str(ExecutionSource.OWNER_MANUAL),
            reason="owner_manual_sell",
            metadata={"interactive": True},
            operator_surface="owner_manual",
        )
        intent = load_execution_intent(result.intent_id) if result.intent_id else None
        return result.ok, result.message, intent

    def create_live_owner_cancel_intent(self, order_ref: str) -> Tuple[bool, str, Optional[ExecutionIntent]]:
        result = self.get_control_plane().create_live_intent(
            action=str(ExecutionAction.CANCEL),
            symbol="ORDER",
            quantity=0,
            limit_price=None,
            target_order_ref=str(order_ref),
            mode=self.execution_mode,
            source=str(ExecutionSource.OWNER_MANUAL),
            reason="owner_manual_cancel",
            metadata={"interactive": True},
            operator_surface="owner_manual",
        )
        intent = load_execution_intent(result.intent_id) if result.intent_id else None
        return result.ok, result.message, intent

    def create_live_owner_modify_intent(self, order_ref: str, limit_price: float, quantity: Optional[int] = None) -> Tuple[bool, str, Optional[ExecutionIntent]]:
        result = self.get_control_plane().create_live_intent(
            action=str(ExecutionAction.MODIFY),
            symbol="ORDER",
            quantity=int(quantity or 0),
            limit_price=float(limit_price),
            target_order_ref=str(order_ref),
            mode=self.execution_mode,
            source=str(ExecutionSource.OWNER_MANUAL),
            reason="owner_manual_modify",
            metadata={"interactive": True},
            operator_surface="owner_manual",
        )
        intent = load_execution_intent(result.intent_id) if result.intent_id else None
        return result.ok, result.message, intent

    def confirm_live_intent(self, intent_id: str, *, timeout: float = 90.0) -> Tuple[Optional[ExecutionIntent], Optional[ExecutionResult]]:
        if self.execution_mode == "shadow_live":
            result = self.get_control_plane().confirm_live_intent(intent_id, mode="shadow_live")
            return load_execution_intent(intent_id), None if not result.ok else None
        if self.live_execution_service is None:
            return None, None
        result = self.get_control_plane().confirm_live_intent(intent_id, mode=self.execution_mode)
        payload = dict(result.payload or {})
        live_result = payload.get("result")
        return load_execution_intent(intent_id), ExecutionResult(**live_result) if isinstance(live_result, dict) else None

    def list_live_orders(self, limit: int = 10) -> List[Dict[str, Any]]:
        return load_latest_live_orders(limit=limit)

    def list_live_positions(self, limit: int = 20) -> List[Dict[str, Any]]:
        return load_latest_live_positions(limit=limit)

    def reconcile_live_orders(self) -> Dict[str, Any]:
        if self.live_execution_service is None:
            return {"enabled": False, "detail": "Live execution disabled"}
        result = self.get_control_plane().reconcile_live_state()
        return dict(result.payload or {})

    def _apply_paper_buy_fill(self, symbol: str, shares: int, ltp: float, *, signal_type: str) -> Tuple[bool, str]:
        if symbol in self.positions:
            return False, f"Already holding {symbol}."
        buy_fees = NepseFees.total_fees(shares, ltp)
        total_cost = shares * ltp + buy_fees
        if total_cost > self.cash:
            return False, "Insufficient cash for mirrored paper fill."
        self.cash -= total_cost
        today_str = now_nst().strftime("%Y-%m-%d")
        self.positions[symbol] = Position(
            symbol=symbol,
            shares=shares,
            entry_price=ltp,
            entry_date=today_str,
            buy_fees=buy_fees,
            signal_type=signal_type,
            high_watermark=ltp,
            last_ltp=ltp,
            last_ltp_source="tms_live_fill",
            last_ltp_time_utc=self.last_price_snapshot_time_utc or utc_now_iso(),
        )
        append_trade_log(
            TradeRecord(
                date=today_str,
                action="BUY",
                symbol=symbol,
                shares=shares,
                price=ltp,
                fees=buy_fees,
                reason=signal_type,
            ),
            self.trade_log_file,
        )
        save_portfolio(self.positions, self.portfolio_file)
        self._persist_runtime_state()
        return True, "paper mirrored"

    def _apply_paper_sell_fill(self, symbol: str, price: float, *, reason: str) -> Tuple[bool, str]:
        if symbol not in self.positions:
            return False, f"No paper position in {symbol}."
        pos = self.positions[symbol]
        sell_fees = NepseFees.total_fees(pos.shares, price, is_sell=True)
        proceeds = pos.shares * price - sell_fees
        gross_pnl = (price - pos.entry_price) * pos.shares
        net_pnl = gross_pnl - pos.buy_fees - sell_fees
        pnl_pct = net_pnl / pos.cost_basis if pos.cost_basis > 0 else 0.0
        self.cash += proceeds
        today_str = now_nst().strftime("%Y-%m-%d")
        append_trade_log(
            TradeRecord(
                date=today_str,
                action="SELL",
                symbol=symbol,
                shares=pos.shares,
                price=price,
                fees=sell_fees,
                reason=reason,
                pnl=round(net_pnl, 2),
                pnl_pct=round(pnl_pct, 4),
            ),
            self.trade_log_file,
        )
        del self.positions[symbol]
        save_portfolio(self.positions, self.portfolio_file)
        self._persist_runtime_state()
        return True, "paper mirrored"

    def _handle_live_execution_result(self, intent: ExecutionIntent, result: ExecutionResult, origin: str) -> None:
        if result.status == ExecutionStatus.FROZEN:
            self.live_halt_level = "all"
            self.live_freeze_reason = result.status_text or "Uncertain submission state"
            self._persist_runtime_state()

        if origin == "reconcile" and not intent.paper_mirrored:
            if result.fill_state == FillState.FILLED and intent.action == ExecutionAction.BUY:
                ok, _ = self._apply_paper_buy_fill(
                    intent.symbol,
                    int(result.observed_qty or intent.quantity),
                    float(result.observed_price or intent.limit_price or 0.0),
                    signal_type=intent.strategy_tag or "live_buy",
                )
                if ok:
                    update_execution_intent(intent.intent_id, paper_mirrored=True)
            elif result.fill_state == FillState.FILLED and intent.action == ExecutionAction.SELL:
                ok, _ = self._apply_paper_sell_fill(
                    intent.symbol,
                    float(result.observed_price or intent.limit_price or 0.0),
                    reason=intent.reason or "live_sell",
                )
                if ok:
                    update_execution_intent(intent.intent_id, paper_mirrored=True)

        interactive = bool((intent.metadata or {}).get("interactive"))
        receipt_html = self._format_live_receipt_html(intent, result=result)
        if origin != "submit" or not interactive:
            self._send_telegram("send_alert", message=receipt_html)
            if result.screenshot_path:
                self._send_telegram("send_alert_photo", caption=receipt_html, photo_path=result.screenshot_path)

        if result.fill_state == FillState.FILLED and not intent.viewer_notified:
            public_msg = self._format_live_receipt_html(intent, result=result)
            self._send_telegram("send_public_alert", message=public_msg)
            mark_intent_notified(intent.intent_id, viewer=True)

    def load_price_data(self) -> None:
        """Load historical prices from the database for signal generation."""
        self._log_activity("Loading price data from DB...")
        conn = sqlite3.connect(str(get_db_path()))
        self.prices_df = load_all_prices(conn)
        conn.close()
        n_symbols = self.prices_df["symbol"].nunique()
        self._log_activity(f"Loaded prices for {n_symbols} symbols")

    def _sector_exposure_snapshot(self) -> Dict[str, float]:
        with self._state_lock:
            nav = self.calculate_nav()
            if nav <= 0:
                return {}
            exposure: Dict[str, float] = {}
            for symbol, position in self.positions.items():
                sector = str(get_symbol_sector(symbol) or "").strip().upper()
                if not sector:
                    continue
                exposure[sector] = exposure.get(sector, 0.0) + float(position.market_value) / float(nav)
            return exposure

    def _rank_entry_signals(self, signals: List[dict], *, as_of_date: Optional[date] = None) -> List[dict]:
        with self._state_lock:
            held_symbols = set(self.positions.keys())
        ranked = rank_signal_candidates(
            signals,
            held_symbols=held_symbols,
            sector_exposure=self._sector_exposure_snapshot(),
            sector_lookup=get_symbol_sector,
            event_context=load_event_adjustment_context(as_of_date or now_nst().date()),
        )
        return ranked

    def run_premarket_health_check(self) -> None:
        """Run at ~10:55 AM NST to verify price API works before market opens."""
        self._log_activity("Pre-market health check starting...")

        # Test price fetcher with 3 liquid stocks
        test_symbols = ["NABIL", "ADBL", "NICA"]
        ltp_map = fetch_prices_for_symbols(test_symbols)
        results = [(sym, ltp_map.get(sym)) for sym in test_symbols]

        ok = sum(1 for _, ltp in results if ltp is not None)
        total = len(results)

        # Check existing positions
        with self._state_lock:
            n_pos = len(self.positions)
            cash = self.cash

        event_result = refresh_daily_event_layer()
        self._log_activity(f"Event layer: {event_result.status} ({event_result.detail})")

        prices_str = "\n".join(
            f"  {sym}: NPR {ltp:,.1f}" if ltp else f"  {sym}: FAILED"
            for sym, ltp in results
        )

        if ok == total:
            status = "ALL SYSTEMS GO"
        elif ok > 0:
            status = "PARTIAL — some fetches failed"
        else:
            status = "CRITICAL — all fetches failed!"

        msg = (
            f"<b>PRE-MARKET HEALTH CHECK</b>\n\n"
            f"<code>"
            f"  Status    : {status}\n"
            f"  API Test  : {ok}/{total} OK\n"
            f"{prices_str}\n"
            f"  Positions : {n_pos}\n"
            f"  Cash      : NPR {cash:,.0f}\n"
            f"  Events    : {event_result.status}\n"
            f"  Signals   : {','.join(self.signal_types)}"
            f"</code>\n\n"
        )
        if ok < total:
            msg += "Primary intraday data feed has issues. Check Nepalstock access / fallback path."

        self._send_telegram("send_alert", message=msg)
        self._log_activity(f"Health check: {status} ({ok}/{total} API OK)")

    def generate_signals_at_open(self) -> None:
        """Generate signals once at market open."""
        if self.signals_generated_today:
            return
        if self.kill_switch.is_triggered:
            self._log_activity("Kill switch active — skipping signal generation")
            return
        if self.prices_df is None:
            self.load_price_data()

        # Notify: market opening
        self._send_telegram(
            "send_alert",
            message="<b>MARKET OPEN</b>\nGenerating signals...",
        )

        self._log_activity("Generating signals...")
        signals, regime = generate_signals(
            self.prices_df, self.signal_types, use_regime_filter=True
        )
        ranked_signals = self._rank_entry_signals(signals, as_of_date=now_nst().date())

        with self._state_lock:
            self.regime = regime
            self.signals_today = ranked_signals
            self.num_signals_today = len(ranked_signals)
            self.signals_generated_today = True
            # Persist marker to disk so restarts don't re-generate signals
            _signal_marker_path(self.signal_marker_dir).touch()

        self._log_activity(
            f"Regime: {regime} | {len(ranked_signals)} ranked signals generated"
        )

        # Send detailed signal summary before executing
        slots = self.max_positions - len(self.positions)
        top_signals = ranked_signals[:slots] if slots > 0 else []
        sig_lines = "\n".join(
            f"  {s['symbol']:<8} score={s['score']:.2f}  {s['signal_type']}"
            for s in ranked_signals[:8]
        )
        self._send_telegram(
            "send_alert",
            message=(
                f"<b>SIGNALS GENERATED</b>\n\n"
                f"<code>"
                f"  Regime  : {regime.upper()}\n"
                f"  Total   : {len(ranked_signals)} signals\n"
                f"  Slots   : {slots} available\n"
                f"  Will buy: {len(top_signals)} stocks\n\n"
                f"{sig_lines}"
                f"</code>"
            ),
        )

        # Execute top signals as buys
        self._execute_buy_signals(signals)

    def _execute_buy_signals(self, signals: List[dict]) -> None:
        """Execute top-ranked signals as paper or live buys."""
        with self._state_lock:
            slots = self.max_positions - len(self.positions)
            if slots <= 0:
                self._log_activity("No position slots available")
                return
        candidates = self._rank_entry_signals(signals, as_of_date=now_nst().date())

        if not candidates:
            return

        ltp_map = fetch_prices_for_symbols([sig["symbol"] for sig in candidates])
        ltp_context = get_latest_ltps_context()
        ltp_source_map = ltp_context.get("source_map") if isinstance(ltp_context, dict) else {}
        live_specs: List[Dict[str, Any]] = []

        with self._state_lock:
            slots = self.max_positions - len(self.positions)
            executed = 0

            for sig in candidates:
                if executed >= slots:
                    break
                sym = sig["symbol"]
                if sym in self.positions:
                    continue

                ltp = ltp_map.get(sym)
                if ltp is None or ltp <= 0:
                    self._log_activity(f"Skip {sym}: no LTP available")
                    continue

                per_position = self.capital / self.max_positions
                available = min(per_position, self.cash * 0.95)
                if available < 10000:
                    continue

                fill_price = estimate_execution_price(
                    float(ltp),
                    is_buy=True,
                    max_price_deviation_pct=_price_deviation_pct(self),
                )
                if fill_price <= 0:
                    self._log_activity(f"Skip {sym}: invalid estimated fill price")
                    continue

                shares = int(available / fill_price)
                if shares < 10:
                    continue

                buy_fees = NepseFees.total_fees(shares, fill_price)
                total_cost = shares * fill_price + buy_fees
                if total_cost > self.cash:
                    shares = int(self.cash / fill_price)
                    if shares < 10:
                        continue
                    buy_fees = NepseFees.total_fees(shares, fill_price)
                    total_cost = shares * fill_price + buy_fees
                    while shares >= 10 and total_cost > self.cash:
                        shares -= 1
                        buy_fees = NepseFees.total_fees(shares, fill_price)
                        total_cost = shares * fill_price + buy_fees
                    if shares < 10:
                        continue

                sym_sector = get_symbol_sector(sym)
                if sym_sector:
                    nav = self.calculate_nav()
                    sector_value = sum(
                        p.market_value for s, p in self.positions.items()
                        if get_symbol_sector(s) == sym_sector
                    )
                    if nav > 0 and (sector_value + shares * ltp) / nav > 0.35:
                        self._log_activity(f"Skip {sym}: sector limit reached")
                        continue

                if self.live_execution_enabled:
                    live_specs.append(
                        {
                            "symbol": sym,
                            "shares": shares,
                            "ltp": float(ltp),
                            "signal_type": str(sig["signal_type"]),
                            "score": float(sig["score"]),
                        }
                    )
                    executed += 1
                    continue

                if self.dry_run:
                    self._log_activity(
                        f"[DRY] BUY {sym} {shares}@{fill_price:,.1f} "
                        f"(LTP {ltp:,.1f}) [{sig['signal_type']}]"
                    )
                    continue

                self.cash -= total_cost
                today_str = now_nst().strftime("%Y-%m-%d")
                self.positions[sym] = Position(
                    symbol=sym,
                    shares=shares,
                    entry_price=fill_price,
                    entry_date=today_str,
                    buy_fees=buy_fees,
                    signal_type=sig["signal_type"],
                    high_watermark=ltp,
                    last_ltp=ltp,
                    last_ltp_source=str(ltp_source_map.get(sym) or "unknown"),
                    last_ltp_time_utc=str((ltp_context.get("timestamp_map") or {}).get(sym) or self.last_price_snapshot_time_utc or ""),
                )
                record = TradeRecord(
                    date=today_str, action="BUY", symbol=sym,
                    shares=shares, price=fill_price, fees=buy_fees,
                    reason=sig["signal_type"],
                )
                try:
                    append_trade_log(record, self.trade_log_file)
                    save_portfolio(self.positions, self.portfolio_file)
                    self._persist_runtime_state()
                except Exception as e:
                    logger.error(f"Failed to persist BUY {sym}: {e}")

                self._log_activity(
                    f"BUY {sym} {shares} shares @ {fill_price:,.1f} "
                    f"(LTP {ltp:,.1f}) [{sig['signal_type']}]"
                )
                self._send_telegram(
                    "send_buy_signal",
                    symbol=sym, shares=shares, price=fill_price,
                    signal_type=sig["signal_type"], strength=sig["score"],
                )
                executed += 1

        if not self.live_execution_enabled:
            return
        if not live_specs:
            return
        if not self.live_settings.strategy_automation_enabled:
            for spec in live_specs:
                self._log_activity(
                    f"Live mode active; skipped strategy entry {spec['symbol']} because strategy automation is disabled"
                )
            return
        if self.live_execution_service is None:
            self._log_activity("Live execution service unavailable; strategy entries skipped")
            return

        for spec in live_specs:
            limit_price = self._best_effort_limit_price(spec["symbol"], ExecutionAction.BUY, spec["ltp"])
            if self.dry_run:
                self._log_activity(
                    f"[DRY] LIVE BUY {spec['symbol']} {spec['shares']} @ {limit_price:,.1f} [{spec['signal_type']}]"
                )
                continue
            intent = ExecutionIntent(
                action=ExecutionAction.BUY,
                symbol=spec["symbol"],
                quantity=int(spec["shares"]),
                limit_price=float(limit_price),
                source=ExecutionSource.STRATEGY_ENTRY,
                requires_confirmation=False,
                status=ExecutionStatus.QUEUED,
                strategy_tag=spec["signal_type"],
                reason=f"strategy_entry:{spec['signal_type']}",
                metadata={
                    "interactive": False,
                    "signal_type": spec["signal_type"],
                    "signal_score": spec["score"],
                    "ltp_anchor": spec["ltp"],
                },
            )
            ok, detail, _, _ = self.live_execution_service.submit_intent(intent, wait=False)
            if not ok:
                self._log_activity(f"Skip live BUY {spec['symbol']}: {detail}")
                continue
            self._log_activity(
                f"LIVE BUY queued {spec['symbol']} {spec['shares']} @ {limit_price:,.1f} [{spec['signal_type']}]"
            )

    def refresh_prices(self) -> None:
        """Fetch latest prices for all held positions.

        Caller must hold _state_lock OR call from the main loop (single writer).
        """
        if not self.positions:
            self.last_refresh = now_nst()
            return

        symbols = list(self.positions.keys())
        self._log_activity(f"Refreshing prices for {len(symbols)} positions...")

        prices = fetch_prices_for_symbols(symbols)
        self._capture_price_refresh_context()
        self._apply_position_source_map(symbols)

        for sym, ltp in prices.items():
            if ltp is not None and sym in self.positions:
                pos = self.positions[sym]
                pos.last_ltp = ltp
                if ltp > pos.high_watermark:
                    pos.high_watermark = ltp

        self.last_refresh = now_nst()
        try:
            save_portfolio(self.positions, self.portfolio_file)
        except Exception as e:
            logger.warning("Failed to persist refreshed portfolio marks: %s", e)
        self._persist_runtime_state()
        self._log_activity("Prices updated")

    def check_and_execute_exits(self) -> None:
        """Check risk exits and execute sells."""
        live_exits: List[Dict[str, Any]] = []
        with self._state_lock:
            exits = check_exits(self.positions, self.holding_days)

            for sym, reason in exits:
                pos = self.positions[sym]
                ltp = pos.last_ltp
                if ltp is None:
                    continue
                price = estimate_execution_price(
                    float(ltp),
                    is_buy=False,
                    max_price_deviation_pct=_price_deviation_pct(self),
                )
                if price <= 0:
                    continue

                if self.live_execution_enabled:
                    live_exits.append(
                        {
                            "symbol": sym,
                            "reason": str(reason),
                            "shares": int(pos.shares),
                            "price": float(price),
                        }
                    )
                    continue

                sell_fees = NepseFees.total_fees(pos.shares, price, is_sell=True)
                proceeds = pos.shares * price - sell_fees
                gross_pnl = (price - pos.entry_price) * pos.shares
                net_pnl = gross_pnl - pos.buy_fees - sell_fees
                pnl_pct = net_pnl / pos.cost_basis if pos.cost_basis > 0 else 0.0

                # Track consecutive losses for kill switch
                if net_pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

                if self.dry_run:
                    self._log_activity(
                        f"[DRY] SELL {sym} {pos.shares}@{price:,.1f} "
                        f"(LTP {ltp:,.1f}) reason={reason} pnl={net_pnl:+,.0f}"
                    )
                    continue

                self.cash += proceeds

                today_str = now_nst().strftime("%Y-%m-%d")
                record = TradeRecord(
                    date=today_str, action="SELL", symbol=sym,
                    shares=pos.shares, price=price, fees=sell_fees,
                    reason=reason, pnl=round(net_pnl, 2),
                    pnl_pct=round(pnl_pct, 4),
                )
                del self.positions[sym]
                # Persist both files together
                try:
                    append_trade_log(record, self.trade_log_file)
                    save_portfolio(self.positions, self.portfolio_file)
                    self._persist_runtime_state()
                except Exception as e:
                    logger.error(f"Failed to persist SELL {sym}: {e}")

                self._log_activity(
                    f"SELL {sym} {pos.shares} shares @ {price:,.1f} "
                    f"(LTP {ltp:,.1f}) "
                    f"[{reason}] P&L: {net_pnl:+,.0f} ({pnl_pct:+.1%})"
                )
                self._send_telegram(
                    "send_sell_signal",
                    symbol=sym, shares=pos.shares, price=price,
                    reason=reason, pnl=net_pnl, pnl_pct=pnl_pct,
                )

        if not self.live_execution_enabled:
            return
        if not live_exits:
            return
        if not self.live_settings.auto_exits_enabled:
            for spec in live_exits:
                self._log_activity(
                    f"Live mode active; exit for {spec['symbol']} blocked because auto exits are disabled"
                )
            return
        if self.live_execution_service is None:
            self._log_activity("Live execution service unavailable; exits skipped")
            return

        for spec in live_exits:
            limit_price = self._best_effort_limit_price(spec["symbol"], ExecutionAction.SELL, spec["price"])
            if self.dry_run:
                self._log_activity(
                    f"[DRY] LIVE SELL {spec['symbol']} {spec['shares']} @ {limit_price:,.1f} [{spec['reason']}]"
                )
                continue
            intent = ExecutionIntent(
                action=ExecutionAction.SELL,
                symbol=spec["symbol"],
                quantity=int(spec["shares"]),
                limit_price=float(limit_price),
                source=ExecutionSource.RISK_EXIT,
                requires_confirmation=False,
                status=ExecutionStatus.QUEUED,
                strategy_tag="risk_exit",
                reason=spec["reason"],
                metadata={"interactive": False, "ltp_anchor": spec["price"]},
            )
            ok, detail, _, _ = self.live_execution_service.submit_intent(intent, wait=False)
            if not ok:
                self._log_activity(f"Skip live SELL {spec['symbol']}: {detail}")
                continue
            self._log_activity(
                f"LIVE SELL queued {spec['symbol']} {spec['shares']} @ {limit_price:,.1f} [{spec['reason']}]"
            )

    def calculate_nav(self) -> float:
        """Current net asset value."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    def send_daily_summary(self) -> None:
        """Send end-of-day summary via Telegram and log NAV."""
        nav = self.calculate_nav()
        positions_value = sum(pos.market_value for pos in self.positions.values())
        day_pnl = (nav - self.daily_start_nav) if self.daily_start_nav else 0
        portfolio_day_return = None
        if self.daily_start_nav and self.daily_start_nav > 0:
            portfolio_day_return = day_pnl / self.daily_start_nav
        comparison = compute_portfolio_vs_nepse(
            self.capital,
            self.positions,
            self.trade_log_file,
            cash=self.cash,
            nav_log_path=self.nav_log_file,
        )
        today_date = now_nst().date()
        benchmark_day_return = None
        deployed_since_start_return = None
        try:
            day_benchmark = compute_nepse_benchmark_return(previous_trading_day(today_date), today_date)
            if day_benchmark and day_benchmark.get("return_pct") is not None:
                benchmark_day_return = float(day_benchmark["return_pct"]) / 100.0
        except Exception as e:
            logger.warning("Failed to compute daily NEPSE change for summary: %s", e)
        benchmark_return = None
        alpha_points = None
        realized_pnl = None
        if comparison:
            performance = comparison.get("performance") or {}
            if performance.get("deployed_return_pct") is not None:
                deployed_since_start_return = float(performance["deployed_return_pct"]) / 100.0
            if performance.get("realized_pnl") is not None:
                realized_pnl = float(performance["realized_pnl"])
            benchmark = comparison.get("benchmark")
            if benchmark and benchmark.get("return_pct") is not None:
                benchmark_return = float(benchmark["return_pct"]) / 100.0
            if comparison.get("alpha_pct") is not None:
                alpha_points = float(comparison["alpha_pct"])

        today_str = now_nst().strftime("%Y-%m-%d")
        try:
            today_date = datetime.strptime(today_str, "%Y-%m-%d").date()
            ensure_benchmark_history("NEPSE", today_date, today_date)
        except Exception as e:
            logger.warning("Failed to persist daily NEPSE benchmark snapshot: %s", e)
        for sector in {get_symbol_sector(symbol) for symbol in self.positions}:
            benchmark = SECTOR_INDEX_SYMBOLS.get(sector or "")
            if not benchmark:
                continue
            try:
                ensure_benchmark_history(benchmark, today_date, today_date)
            except Exception as e:
                logger.warning("Failed to persist daily sector benchmark snapshot for %s: %s", benchmark, e)

        if not self.dry_run:
            append_nav_log(
                today_str, self.cash, positions_value,
                nav, len(self.positions), self.nav_log_file,
            )
            self._persist_runtime_state()

        self._log_activity(
            f"EOD — NAV: NPR {nav:,.0f} | Day P&L: {day_pnl:+,.0f}"
        )
        self._send_telegram(
            "send_daily_summary",
            portfolio_value=nav, day_pnl=day_pnl,
            open_positions=len(self.positions),
            signals_generated=self.num_signals_today,
            capital=self.capital,
            since_start_return=deployed_since_start_return,
            portfolio_day_return=portfolio_day_return,
            benchmark_day_return=benchmark_day_return,
            benchmark_return=benchmark_return,
            alpha_points=alpha_points,
            realized_pnl=realized_pnl,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Manual trading (called by Telegram bot under _state_lock)
    # ─────────────────────────────────────────────────────────────────────

    def execute_manual_buy(self, symbol: str, shares: int, ltp: float) -> Tuple[bool, str]:
        """Execute a manual buy order.

        Must be called while holding _state_lock.
        Returns (success, receipt_message_html).
        """
        if symbol in self.positions:
            return False, f"Already holding {symbol}."

        if len(self.positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached."

        fill_price = estimate_execution_price(
            float(ltp),
            is_buy=True,
            max_price_deviation_pct=_price_deviation_pct(self),
        )
        if fill_price <= 0:
            return False, f"Invalid execution price for {symbol}."

        buy_fees = NepseFees.total_fees(shares, fill_price)
        total_cost = shares * fill_price + buy_fees

        if total_cost > self.cash:
            return False, f"Insufficient cash. Need {total_cost:,.0f}, have {self.cash:,.0f}."

        if self.dry_run:
            return False, "[DRY RUN] Buy not executed."

        self.cash -= total_cost
        today_str = now_nst().strftime("%Y-%m-%d")
        ltp_context = get_latest_ltps_context()
        ltp_source_map = ltp_context.get("source_map") if isinstance(ltp_context, dict) else {}

        self.positions[symbol] = Position(
            symbol=symbol,
            shares=shares,
            entry_price=fill_price,
            entry_date=today_str,
            buy_fees=buy_fees,
            signal_type="manual",
            high_watermark=ltp,
            last_ltp=ltp,
            last_ltp_source=str(ltp_source_map.get(symbol) or "manual"),
            last_ltp_time_utc=str((ltp_context.get("timestamp_map") or {}).get(symbol) or self.last_price_snapshot_time_utc or ""),
        )

        record = TradeRecord(
            date=today_str, action="BUY", symbol=symbol,
            shares=shares, price=fill_price, fees=buy_fees,
            reason="manual",
        )
        append_trade_log(record, self.trade_log_file)
        save_portfolio(self.positions, self.portfolio_file)
        self._persist_runtime_state()

        self._log_activity(
            f"MANUAL BUY {symbol} {shares} shares @ {fill_price:,.1f} (LTP {ltp:,.1f})"
        )

        msg = (
            f"<b>BUY EXECUTED</b>\n\n"
            f"<code>"
            f"  Symbol : {symbol}\n"
            f"  Shares : {shares}\n"
            f"  Price  : NPR {fill_price:,.1f}\n"
            f"  Fees   : NPR {buy_fees:,.0f}\n"
            f"  Total  : NPR {total_cost:,.0f}\n"
            f"  Cash   : NPR {self.cash:,.0f}"
            f"</code>"
        )
        return True, msg

    def execute_manual_sell(self, symbol: str) -> Tuple[bool, str]:
        """Execute a manual sell order.

        Must be called while holding _state_lock.
        Returns (success, receipt_message_html).
        """
        if symbol not in self.positions:
            return False, f"No position in {symbol}."

        pos = self.positions[symbol]
        ltp = pos.last_ltp
        if ltp is None:
            # Try to fetch
            ltp = fetch_latest_ltp(symbol)
            if ltp is None:
                return False, f"Could not fetch LTP for {symbol}."
        price = estimate_execution_price(
            float(ltp),
            is_buy=False,
            max_price_deviation_pct=_price_deviation_pct(self),
        )
        if price <= 0:
            return False, f"Invalid execution price for {symbol}."

        if self.dry_run:
            return False, "[DRY RUN] Sell not executed."

        sell_fees = NepseFees.total_fees(pos.shares, price, is_sell=True)
        proceeds = pos.shares * price - sell_fees
        gross_pnl = (price - pos.entry_price) * pos.shares
        net_pnl = gross_pnl - pos.buy_fees - sell_fees
        pnl_pct = net_pnl / pos.cost_basis if pos.cost_basis > 0 else 0.0
        days = count_trading_days_since(pos.entry_date)

        self.cash += proceeds

        today_str = now_nst().strftime("%Y-%m-%d")
        record = TradeRecord(
            date=today_str, action="SELL", symbol=symbol,
            shares=pos.shares, price=price, fees=sell_fees,
            reason="manual", pnl=round(net_pnl, 2),
            pnl_pct=round(pnl_pct, 4),
        )
        append_trade_log(record, self.trade_log_file)
        del self.positions[symbol]
        save_portfolio(self.positions, self.portfolio_file)
        self._persist_runtime_state()

        self._log_activity(
            f"MANUAL SELL {symbol} {pos.shares} shares @ {price:,.1f} "
            f"(LTP {ltp:,.1f}) "
            f"P&L: {net_pnl:+,.0f} ({pnl_pct:+.1%})"
        )

        emoji = "✅" if net_pnl >= 0 else "❌"
        msg = (
            f"<b>SELL EXECUTED</b>\n\n"
            f"<code>"
            f"  Symbol   : {symbol}\n"
            f"  Shares   : {pos.shares}\n"
            f"  Price    : NPR {price:,.1f}\n"
            f"  Proceeds : NPR {proceeds:,.0f}\n"
            f"  P&L      : {emoji} NPR {net_pnl:+,.0f} ({pnl_pct:+.1%})\n"
            f"  Held     : {days} trading days\n"
            f"  Cash     : NPR {self.cash:,.0f}"
            f"</code>"
        )
        return True, msg

    # ─────────────────────────────────────────────────────────────────────
    # Rich TUI
    # ─────────────────────────────────────────────────────────────────────

    def build_header(self) -> Panel:
        """Build the header panel."""
        nst = now_nst()

        if is_market_open():
            market_status = "[bold green]OPEN[/]"
        elif is_trading_day(nst.date()) and nst.hour < MARKET_OPEN_HOUR:
            market_status = "[bold yellow]PRE-MARKET[/]"
        elif is_trading_day(nst.date()) and nst.hour >= MARKET_CLOSE_HOUR:
            market_status = "[bold red]CLOSED[/]"
        else:
            market_status = "[dim]HOLIDAY/WEEKEND[/]"

        regime_color = {
            "bull": "green", "neutral": "yellow", "bear": "red", "unknown": "dim",
        }.get(self.regime, "dim")

        # Next refresh countdown
        if self.last_refresh and is_market_open():
            elapsed = (nst - self.last_refresh).total_seconds()
            remaining = max(0, self.refresh_secs - elapsed)
            mins, secs = divmod(int(remaining), 60)
            refresh_str = f"{mins}m {secs:02d}s"
        else:
            refresh_str = "--"

        day_name = nst.strftime("%a")
        date_str = nst.strftime("%Y-%m-%d %H:%M")

        text = Text()
        text.append("NEPSE LIVE PAPER TRADER", style="bold white")
        text.append(f"          {day_name} {date_str} NST\n", style="dim")
        text.append(f"Market: {market_status}", style="white")
        text.append(f"  |  Regime: [{regime_color}]{self.regime.upper()}[/]")
        text.append(f"  |  Next refresh: {refresh_str}", style="dim")

        if self.dry_run:
            text.append("  |  [bold yellow]DRY RUN[/]")

        return Panel(text, style="bold blue")

    def build_portfolio_table(self) -> Panel:
        """Build the portfolio positions table."""
        table = Table(
            show_header=True, header_style="bold cyan",
            expand=True, box=None, padding=(0, 1),
        )
        table.add_column("Symbol", style="bold white", width=8)
        table.add_column("Shares", justify="right", width=7)
        table.add_column("Entry", justify="right", width=9)
        table.add_column("LTP", justify="right", width=9)
        table.add_column("P&L", justify="right", width=10)
        table.add_column("P&L%", justify="right", width=8)
        table.add_column("Days", justify="right", width=5)
        table.add_column("Signal", width=12)

        for sym in sorted(self.positions.keys()):
            pos = self.positions[sym]
            days = count_trading_days_since(pos.entry_date)
            ltp = pos.last_ltp if pos.last_ltp else pos.entry_price
            pnl = pos.unrealized_pnl
            pnl_pct = pos.pnl_pct

            pnl_style = "green" if pnl >= 0 else "red"
            pnl_str = f"[{pnl_style}]{pnl:+,.0f}[/]"
            pnl_pct_str = f"[{pnl_style}]{pnl_pct:+.1%}[/]"

            table.add_row(
                sym, str(pos.shares),
                f"{pos.entry_price:,.0f}", f"{ltp:,.0f}",
                pnl_str, pnl_pct_str, str(days),
                pos.signal_type,
            )

        nav = self.calculate_nav()
        positions_value = sum(pos.market_value for pos in self.positions.values())
        total_return = (nav / self.capital - 1) if self.capital > 0 else 0
        ret_style = "green" if total_return >= 0 else "red"

        footer = Text()
        footer.append(f"\nCash: NPR {self.cash:,.0f}", style="white")
        footer.append(f"  |  NAV: NPR {nav:,.0f}", style="bold white")
        footer.append(f"  |  ", style="dim")
        footer.append(f"{total_return:+.1%}", style=ret_style)

        content = Text()
        content.append_text(Text.from_ansi(str(table)))

        title = f"PORTFOLIO  (NPR {self.capital:,.0f} capital, {len(self.positions)} positions)"
        return Panel(
            table, title=title, subtitle=footer,
            style="white", border_style="blue",
        )

    def build_activity_panel(self) -> Panel:
        """Build the today's activity panel."""
        text = Text()
        for entry in self.activity_log[-8:]:
            text.append(entry + "\n", style="white")
        if not self.activity_log:
            text.append("No activity yet...", style="dim")
        return Panel(text, title="TODAY'S ACTIVITY", border_style="blue")

    def build_signals_panel(self) -> Panel:
        """Build the signals panel."""
        text = Text()
        if not self.signals_today:
            text.append("No signals generated yet...", style="dim")
        else:
            for sig in self.signals_today[:8]:
                score = sig["score"]
                color = "green" if score > 0.5 else ("yellow" if score > 0.3 else "dim")
                text.append(f"{sig['symbol']:<8}", style="bold white")
                text.append(f"score={score:.2f}  ", style=color)
                reasoning = sig["reasoning"]
                if len(reasoning) > 50:
                    reasoning = reasoning[:47] + "..."
                text.append(f"{reasoning}\n", style="dim")
        return Panel(text, title=f"SIGNALS ({len(self.signals_today)} generated)", border_style="blue")

    def build_trades_panel(self) -> Panel:
        """Build recent trades panel."""
        trades = load_trade_log(self.trade_log_file, limit=8)
        text = Text()
        if not trades:
            text.append("No trades yet...", style="dim")
        else:
            for t in reversed(trades):
                line = format_trade_activity_line(
                    date=t.date,
                    action=t.action,
                    symbol=t.symbol,
                    shares=t.shares,
                    price=t.price,
                    pnl=t.pnl if t.action == "SELL" else None,
                )
                text.append(line + "\n", style="white")

        return Panel(text, title="RECENT TRADES", border_style="blue")

    def build_layout(self) -> Layout:
        """Compose the full Rich layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="main"),
            Layout(name="bottom", size=12),
        )
        layout["main"].split_column(
            Layout(name="portfolio", ratio=3),
            Layout(name="activity", ratio=2),
        )
        layout["bottom"].split_row(
            Layout(name="signals"),
            Layout(name="trades"),
        )

        layout["header"].update(self.build_header())
        layout["portfolio"].update(self.build_portfolio_table())
        layout["activity"].update(self.build_activity_panel())
        layout["signals"].update(self.build_signals_panel())
        layout["trades"].update(self.build_trades_panel())

        return layout

    # ─────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main entry point — starts bot thread, then TUI or headless loop."""
        # Load price data at startup
        try:
            self.load_price_data()
        except Exception as e:
            logger.error("Failed to load price data: %s", e)
            self._send_telegram("send_error", error_message=f"Failed to load prices: {e}")
            return

        current_nav = self.calculate_nav()
        if not self.daily_start_nav or self.daily_start_nav <= 0:
            self.daily_start_nav = resolve_daily_start_nav(self.nav_log_file, fallback_nav=current_nav)
        self._persist_runtime_state()
        self._start_live_execution_service()

        # Start Telegram bot thread (if enabled)
        if not self.no_telegram:
            owner_token = os.environ.get("NEPSE_TELEGRAM_BOT_TOKEN")
            owner_chat_id = os.environ.get("NEPSE_TELEGRAM_CHAT_ID")
            viewer_token = os.environ.get("NEPSE_VIEWER_TELEGRAM_BOT_TOKEN")
            viewer_chat_ids = os.environ.get("NEPSE_VIEWER_TELEGRAM_ALLOWED_CHAT_IDS")
            viewer_startup_chat_id = os.environ.get("NEPSE_VIEWER_TELEGRAM_POST_CHAT_ID")
            if owner_token and owner_chat_id:
                from backend.quant_pro.telegram_bot import start_bot_threads

                start_bot_threads(
                    self,
                    owner_token=owner_token,
                    owner_chat_id=owner_chat_id,
                    viewer_token=viewer_token,
                    viewer_chat_ids=viewer_chat_ids,
                    viewer_startup_chat_id=viewer_startup_chat_id,
                )

        if self.headless:
            self._run_headless()
        else:
            self._run_tui()

    def _run_tui(self) -> None:
        """Run with Rich Live TUI display."""
        console = Console()
        with Live(self.build_layout(), console=console, refresh_per_second=1,
                   screen=True) as live:
            retries = 0
            max_retries = 3
            while retries < max_retries:
                try:
                    self._run_loop(live)
                    break  # Clean exit
                except KeyboardInterrupt:
                    self._log_activity("Shutting down (Ctrl+C)")
                    break
                except Exception as e:
                    retries += 1
                    logger.exception(f"Loop error ({retries}/{max_retries}) — recovering in 60s")
                    self._send_telegram("send_error", error_message=f"Loop error ({retries}/{max_retries}): {e}")
                    if retries >= max_retries:
                        logger.critical("Max retries exceeded — shutting down")
                        break
                    time.sleep(60)

    def _run_headless(self) -> None:
        """Run without Rich TUI (Telegram-only, for VPS/server)."""
        logger.info("Running in headless mode (no TUI). Telegram bot active.")
        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                self._run_loop(live=None)
                break  # Clean exit
            except KeyboardInterrupt:
                self._log_activity("Shutting down (Ctrl+C)")
                break
            except Exception as e:
                retries += 1
                logger.exception(f"Loop error ({retries}/{max_retries}) — recovering in 60s")
                self._send_telegram("send_error", error_message=f"Loop error ({retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    logger.critical("Max retries exceeded — shutting down")
                    break
                time.sleep(60)

    def _run_loop(self, live: Optional[Live] = None) -> None:
        """Inner event loop. When live is None, runs headless (no TUI updates)."""
        last_refresh_ts = 0.0
        daily_summary_sent = False
        health_check_done = False
        last_tms_poll_ts = 0.0

        while True:
            nst = now_nst()
            now_ts = time.monotonic()
            should_poll_tms = self.tms_monitor_enabled and self.tms_browser is not None
            if should_poll_tms and self.tms_monitor_market_hours_only:
                should_poll_tms = is_market_open()

            if should_poll_tms and (now_ts - last_tms_poll_ts) >= self.tms_monitor_interval_secs:
                try:
                    self.refresh_tms_monitor(force=True, emit_alerts=True)
                except Exception as e:
                    logger.warning("TMS monitor refresh failed: %s", e)
                    self.tms_last_error = str(e)
                last_tms_poll_ts = now_ts

            # ── Pre-market health check at 10:55 AM ──
            if (is_trading_day(nst.date())
                    and nst.hour == 10 and nst.minute >= 55
                    and not health_check_done
                    and not self.signals_generated_today):
                self.run_premarket_health_check()
                health_check_done = True

            # ── Market open: generate signals ──
            if is_market_open() and not self.signals_generated_today:
                self.generate_signals_at_open()

            # ── During market hours: periodic price refresh + risk check ──
            if is_market_open() and (now_ts - last_refresh_ts) >= self.refresh_secs:
                with self._state_lock:
                    self.refresh_prices()
                self.check_and_execute_exits()
                if self.live_execution_enabled and self.live_execution_service is not None:
                    try:
                        summary = self.reconcile_live_orders()
                        if summary.get("matched_intents"):
                            self._log_activity(
                                f"Live reconcile: {summary.get('matched_intents')} intents, {summary.get('orders')} orders"
                            )
                    except Exception as e:
                        logger.warning("Live reconciliation failed: %s", e)
                last_refresh_ts = now_ts

                # ── Kill switch check ──
                if not self.kill_switch.is_triggered:
                    current_nav = self.calculate_nav()
                    self.peak_nav = max(self.peak_nav, current_nav)
                    daily_pnl = current_nav - (self.daily_start_nav or current_nav)
                    should_halt, reason = self.kill_switch.check(
                        current_nav=current_nav,
                        peak_nav=self.peak_nav,
                        daily_pnl=daily_pnl,
                        daily_start_nav=self.daily_start_nav or current_nav,
                        consecutive_losses=self.consecutive_losses,
                        last_data_time=self.last_refresh,
                    )
                    if should_halt:
                        self._log_activity(f"KILL SWITCH: {reason}")
                        self._send_telegram(
                            "send_alert",
                            message=f"\U0001f6a8 <b>KILL SWITCH TRIGGERED</b>\n{reason}\nAll trading halted.",
                        )

            # ── Market just closed: send daily summary ──
            if (is_trading_day(nst.date())
                    and nst.hour >= MARKET_CLOSE_HOUR
                    and not daily_summary_sent
                    and self.signals_generated_today):
                self.send_daily_summary()
                daily_summary_sent = True

            # ── New day reset ──
            if nst.hour < MARKET_OPEN_HOUR and self.signals_generated_today:
                with self._state_lock:
                    self.signals_generated_today = False
                    daily_summary_sent = False
                    health_check_done = False
                    self.signals_today = []
                    self.activity_log = []
                    self.daily_start_nav = self.calculate_nav()
                    self.num_signals_today = 0
                    self.prices_df = None  # Force fresh load for next session
                    self.consecutive_losses = 0
                    # Clean up old signal marker files
                    for old_marker in self.signal_marker_dir.glob(".signals_*"):
                        old_marker.unlink(missing_ok=True)
                    for old_marker in PROJECT_ROOT.glob(".signals_*"):
                        old_marker.unlink(missing_ok=True)
                self._log_activity("New day — waiting for market open")

            # ── Exit if single-session mode and market is done ──
            if (not self.continuous
                    and is_trading_day(nst.date())
                    and nst.hour >= MARKET_CLOSE_HOUR
                    and daily_summary_sent):
                self._log_activity("Session complete. Exiting.")
                if live:
                    live.update(self.build_layout())
                time.sleep(3)
                return

            # ── Non-trading day in single-session mode ──
            if not self.continuous and not is_trading_day(nst.date()):
                self._log_activity("Not a trading day. Exiting.")
                if live:
                    live.update(self.build_layout())
                time.sleep(3)
                return

            # Update display (skip in headless mode)
            if live:
                live.update(self.build_layout())
            time.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NEPSE Live Paper Trading Bot with Rich TUI + Telegram Alerts"
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run indefinitely (auto-skip weekends/holidays)",
    )
    parser.add_argument(
        "--capital", type=float, default=DEFAULT_CAPITAL,
        help=f"Starting capital in NPR (default: {DEFAULT_CAPITAL:,.0f})",
    )
    parser.add_argument(
        "--signals", type=str, default="volume,quality,low_vol",
        help="Comma-separated signal types (default: volume,quality,low_vol)",
    )
    parser.add_argument(
        "--refresh-secs", type=int, default=300,
        help="Price refresh interval in seconds (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--no-telegram", action="store_true",
        help="Disable Telegram alerts (TUI only)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without Rich TUI (Telegram-only, for VPS/server)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Display only, no portfolio writes",
    )
    parser.add_argument(
        "--paper-portfolio", type=str, default=str(DEFAULT_PAPER_PORTFOLIO),
        help="Path to portfolio CSV file",
    )
    parser.add_argument(
        "--mode", type=str, default="paper", choices=("paper", "live", "dual"),
        help="Execution mode: paper, live, or dual (default: paper)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Default to continuous+headless when no TUI is explicitly requested
    # so the Telegram bot stays alive on weekends/holidays.
    if not hasattr(args, "continuous") or not args.continuous:
        if args.headless:
            args.continuous = True
    trader = LiveTrader(args)
    trader.run()
    # If the trading loop exits (non-trading day) but the Telegram bot thread
    # is still running, keep the main thread alive so the daemon doesn't die.
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
