#!/usr/bin/env python3
"""
Portfolio-Based Walk-Forward Backtest for NEPSE Signal Validation

Tracks actual capital allocation, position sizing, NEPSE tiered fees,
daily portfolio NAV, and computes institutional-grade risk metrics.

No lookahead bias - uses only past data for each decision.
Entry at T+1 open after signal on T. Stop checks use open price.

Holding period is counted in actual NEPSE trading days (Sun-Thu, excluding
holidays), not calendar days. 40 trading days ≈ 8 NEPSE weeks ≈ 56 calendar days.

Usage:
    python simple_backtest.py --start 2023-01-01 --end 2025-12-31 --holding-days 40
    python simple_backtest.py --sweep
"""

import argparse
import logging
import multiprocessing
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
import numpy as np

from backend.quant_pro.alpha_practical import (
    AlphaSignal,
    MomentumScanner,
    LiquidityScanner,
    SignalType,
)
from backend.quant_pro.sectors import SECTOR_GROUPS
from backend.quant_pro.database import get_db_path
from backend.quant_pro.event_layer import load_event_adjustment_context
from backend.quant_pro.signal_ranking import rank_signal_candidates

# Agent 3 signal imports (Citadel upgrade)
from backend.quant_pro.disposition import generate_cgo_signals_at_date
# Signal plugin: lead_lag (not included in public release)
from backend.quant_pro.pairs_trading import generate_pairs_signals_at_date
# Signal plugin: informed_trading (not included in public release)

# Sprint 4-5 signal imports (Alt data + Earnings + Sentiment)
from backend.quant_pro.satellite_data import generate_hydro_rainfall_signals_at_date
from backend.quant_pro.macro_signals import generate_remittance_signals_at_date
# Signal plugin: earnings_drift (not included in public release)
from backend.quant_pro.nepali_sentiment import generate_sentiment_signals_at_date
from backend.quant_pro.quarterly_fundamental import QuarterlyFundamentalModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# NEPSE FEE MODEL — unified source in validation.transaction_costs
# =============================================================================

from validation.transaction_costs import TransactionCostModel as NepseFees  # noqa: E402

# NEPSE trades Sun-Thu (~240 trading days/year vs NYSE's 252)
NEPSE_TRADING_DAYS = 240


# =============================================================================
# NEPSE CIRCUIT BREAKER
# =============================================================================

CIRCUIT_BREAKER_PCT = 0.10  # ±10% daily price limit


def apply_circuit_breaker(price: float, prev_close: float) -> float:
    """Cap price at NEPSE ±10% daily limit relative to previous close."""
    if prev_close <= 0:
        return price
    max_price = prev_close * (1 + CIRCUIT_BREAKER_PCT)
    min_price = prev_close * (1 - CIRCUIT_BREAKER_PCT)
    return max(min_price, min(max_price, price))


# =============================================================================
# TRADE AND PORTFOLIO RESULT DATACLASSES
# =============================================================================

@dataclass
class Trade:
    """A single trade with actual position sizing and fees."""
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int = 0
    position_value: float = 0.0   # entry_price * shares
    buy_fees: float = 0.0
    sell_fees: float = 0.0
    signal_date: Optional[datetime] = None
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    signal_type: str = ""
    direction: int = 1  # 1 = long
    exit_reason: str = ""
    max_price: float = 0.0  # For trailing stop tracking
    entry_trading_idx: int = 0  # Index in trading_dates array at entry
    target_exit_date: Optional[datetime] = None  # Event-driven exit date (e.g. bookclose T-1)
    max_holding_days: int = 40  # Regime-adaptive hold cap (set at entry time)

    @property
    def gross_pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) * self.shares * self.direction

    @property
    def net_pnl(self) -> Optional[float]:
        gross = self.gross_pnl
        if gross is None:
            return None
        return gross - self.buy_fees - self.sell_fees

    @property
    def net_return(self) -> Optional[float]:
        if self.position_value == 0:
            return None
        pnl = self.net_pnl
        if pnl is None:
            return None
        return pnl / self.position_value

    @property
    def gross_return(self) -> Optional[float]:
        if self.exit_price is None or self.entry_price == 0:
            return None
        return (self.exit_price / self.entry_price - 1) * self.direction

    @property
    def holding_days(self) -> Optional[int]:
        if self.exit_date is None:
            return None
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    """Results from a portfolio-based backtest with institutional-grade metrics."""
    trades: List[Trade]
    start_date: datetime
    end_date: datetime
    holding_period: int
    initial_capital: float
    daily_nav: List[Tuple[Any, float]]  # [(date, nav_value), ...]

    @property
    def completed_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.net_return is not None]

    @property
    def total_trades(self) -> int:
        return len(self.completed_trades)

    @property
    def winning_trades(self) -> int:
        return len([t for t in self.completed_trades if t.net_pnl is not None and t.net_pnl > 0])

    @property
    def losing_trades(self) -> int:
        return len([t for t in self.completed_trades if t.net_pnl is not None and t.net_pnl <= 0])

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_win(self) -> float:
        wins = [t.net_return for t in self.completed_trades if t.net_pnl is not None and t.net_pnl > 0]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.net_return for t in self.completed_trades if t.net_pnl is not None and t.net_pnl <= 0]
        return float(np.mean(losses)) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_wins = sum(t.net_pnl for t in self.completed_trades if t.net_pnl is not None and t.net_pnl > 0)
        gross_losses = abs(sum(t.net_pnl for t in self.completed_trades if t.net_pnl is not None and t.net_pnl <= 0))
        if gross_losses == 0:
            return float('inf') if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    @property
    def daily_returns(self) -> np.ndarray:
        navs = np.array([nav for _, nav in self.daily_nav])
        if len(navs) < 2:
            return np.array([])
        return np.diff(navs) / navs[:-1]

    @property
    def total_return(self) -> float:
        if len(self.daily_nav) < 2:
            return 0.0
        return self.daily_nav[-1][1] / self.daily_nav[0][1] - 1

    @property
    def annualized_return(self) -> float:
        if len(self.daily_nav) < 2:
            return 0.0
        total_days = (self.daily_nav[-1][0] - self.daily_nav[0][0]).days
        if total_days == 0:
            return 0.0
        total_ret = self.total_return
        if total_ret <= -1:
            return -1.0
        return (1 + total_ret) ** (365 / total_days) - 1

    @property
    def volatility(self) -> float:
        rets = self.daily_returns
        if len(rets) < 30:
            return 0.0
        return float(np.std(rets, ddof=1) * np.sqrt(NEPSE_TRADING_DAYS))

    @property
    def sharpe_ratio(self) -> float:
        rets = self.daily_returns
        if len(rets) < 30:
            return 0.0
        mean_ret = np.mean(rets)
        std_ret = np.std(rets, ddof=1)
        if std_ret == 0:
            return 0.0
        return float(mean_ret / std_ret * np.sqrt(NEPSE_TRADING_DAYS))

    @property
    def sortino_ratio(self) -> float:
        rets = self.daily_returns
        if len(rets) < 30:
            return 0.0
        downside = rets[rets < 0]
        if len(downside) == 0:
            return float('inf') if np.mean(rets) > 0 else 0.0
        downside_std = np.std(downside, ddof=1)
        if downside_std == 0:
            return 0.0
        return float(np.mean(rets) / downside_std * np.sqrt(NEPSE_TRADING_DAYS))

    @property
    def max_drawdown(self) -> float:
        navs = np.array([nav for _, nav in self.daily_nav])
        if len(navs) < 2:
            return 0.0
        running_max = np.maximum.accumulate(navs)
        drawdowns = (navs - running_max) / running_max
        return float(np.min(drawdowns))

    @property
    def max_drawdown_duration(self) -> int:
        """Duration of longest drawdown in trading days."""
        navs = np.array([nav for _, nav in self.daily_nav])
        if len(navs) < 2:
            return 0
        running_max = np.maximum.accumulate(navs)
        in_drawdown = navs < running_max
        max_dur = 0
        current_dur = 0
        for dd in in_drawdown:
            if dd:
                current_dur += 1
                max_dur = max(max_dur, current_dur)
            else:
                current_dur = 0
        return max_dur

    @property
    def calmar_ratio(self) -> float:
        dd = abs(self.max_drawdown)
        if dd == 0:
            return 0.0
        ann_ret = self.annualized_return
        return ann_ret / dd

    @property
    def max_consecutive_losses(self) -> int:
        trades = sorted(self.completed_trades, key=lambda t: t.entry_date)
        if not trades:
            return 0
        max_streak = 0
        current_streak = 0
        for t in trades:
            if t.net_pnl is not None and t.net_pnl <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    @property
    def total_pnl(self) -> float:
        return sum(t.net_pnl for t in self.completed_trades if t.net_pnl is not None)

    @property
    def total_fees_paid(self) -> float:
        return sum(t.buy_fees + t.sell_fees for t in self.completed_trades)

    @property
    def avg_holding_days(self) -> float:
        days = [t.holding_days for t in self.completed_trades if t.holding_days is not None]
        return float(np.mean(days)) if days else 0.0

    def by_signal_type(self) -> Dict[str, Dict[str, Any]]:
        """Break down performance by signal type."""
        by_type: Dict[str, List[Trade]] = {}
        for t in self.completed_trades:
            if t.signal_type not in by_type:
                by_type[t.signal_type] = []
            by_type[t.signal_type].append(t)

        results = {}
        for sig_type, sig_trades in by_type.items():
            returns = [t.net_return for t in sig_trades if t.net_return is not None]
            pnls = [t.net_pnl for t in sig_trades if t.net_pnl is not None]
            if not returns:
                continue
            results[sig_type] = {
                "count": len(returns),
                "win_rate": len([r for r in returns if r > 0]) / len(returns),
                "avg_return": float(np.mean(returns)),
                "total_pnl": sum(pnls),
                "std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0,
            }
        return results

    def by_exit_reason(self) -> Dict[str, int]:
        """Count trades by exit reason."""
        reasons: Dict[str, int] = {}
        for t in self.completed_trades:
            reason = t.exit_reason or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
        return reasons

    def monthly_returns(self) -> pd.Series:
        """Compute monthly return series from daily NAV."""
        if len(self.daily_nav) < 2:
            return pd.Series(dtype=float)
        nav_df = pd.DataFrame(self.daily_nav, columns=["date", "nav"])
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        nav_df = nav_df.set_index("date")
        monthly = nav_df["nav"].resample("ME").last()
        return monthly.pct_change(fill_method=None).dropna()

    def summary(self) -> str:
        final_nav = self.daily_nav[-1][1] if self.daily_nav else self.initial_capital
        lines = [
            "=" * 70,
            f"BACKTEST RESULTS: {self.start_date.date()} to {self.end_date.date()}",
            f"Capital: NPR {self.initial_capital:,.0f}  ->  Final NAV: NPR {final_nav:,.0f}",
            "=" * 70,
            "",
            "PORTFOLIO METRICS (from daily NAV):",
            f"  Total Return:        {self.total_return:>10.2%}",
            f"  Annualized Return:   {self.annualized_return:>10.2%}",
            f"  Volatility (ann.):   {self.volatility:>10.2%}",
            f"  Sharpe Ratio:        {self.sharpe_ratio:>10.2f}",
            f"  Sortino Ratio:       {self.sortino_ratio:>10.2f}",
            f"  Max Drawdown:        {self.max_drawdown:>10.2%}",
            f"  Max DD Duration:     {self.max_drawdown_duration:>10d} days",
            f"  Calmar Ratio:        {self.calmar_ratio:>10.2f}",
            "",
            "TRADE STATISTICS:",
            f"  Total Trades:        {self.total_trades:>10d}",
            f"  Win Rate:            {self.win_rate:>10.1%}",
            f"  Avg Win:             {self.avg_win:>10.2%}",
            f"  Avg Loss:            {self.avg_loss:>10.2%}",
            f"  Profit Factor:       {self.profit_factor:>10.2f}",
            f"  Max Consec. Losses:  {self.max_consecutive_losses:>10d}",
            f"  Avg Holding Days:    {self.avg_holding_days:>10.1f}",
            f"  Total P&L:           NPR {self.total_pnl:>12,.0f}",
            f"  Total Fees Paid:     NPR {self.total_fees_paid:>12,.0f}",
        ]

        sig_breakdown = self.by_signal_type()
        if sig_breakdown:
            lines.append("")
            lines.append("BY SIGNAL TYPE:")
            for sig_type, stats in sig_breakdown.items():
                lines.append(
                    f"  {sig_type}: {stats['count']} trades, "
                    f"{stats['win_rate']:.1%} win, "
                    f"{stats['avg_return']:.2%} avg ret, "
                    f"NPR {stats['total_pnl']:,.0f} P&L"
                )

        exit_reasons = self.by_exit_reason()
        if exit_reasons:
            lines.append("")
            lines.append("EXIT REASONS:")
            for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
                pct = count / self.total_trades if self.total_trades > 0 else 0
                lines.append(f"  {reason}: {count} ({pct:.0%})")

        # Monthly returns summary
        monthly = self.monthly_returns()
        if len(monthly) > 0:
            positive_months = (monthly > 0).sum()
            lines.append("")
            lines.append(f"MONTHLY: {positive_months}/{len(monthly)} positive "
                         f"({positive_months/len(monthly):.0%}), "
                         f"best {monthly.max():.1%}, worst {monthly.min():.1%}")

        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# DATA LOADING AND PRICE HELPERS
# =============================================================================

def load_all_prices(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all price data from database."""
    query = """
        SELECT symbol, date, open, high, low, close, volume
        FROM stock_prices
        ORDER BY symbol, date
    """
    df = pd.read_sql_query(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def precompute_prev_close(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Add prev_close column for circuit breaker calculations."""
    prices_df = prices_df.sort_values(["symbol", "date"])
    prices_df["prev_close"] = prices_df.groupby("symbol")["close"].shift(1)
    return prices_df


def build_price_lookup(prices_df: pd.DataFrame, column: str) -> Dict[str, Dict[Any, float]]:
    """Build fast symbol -> {date -> price} lookup for daily computations."""
    lookup: Dict[str, Dict[Any, float]] = defaultdict(dict)
    for symbol, group in prices_df.groupby("symbol"):
        dates = group["date"].values
        prices = group[column].values
        sym_dict = {}
        for d, price in zip(dates, prices):
            sym_dict[pd.Timestamp(d)] = float(price)
        lookup[symbol] = sym_dict
    return lookup


def build_close_lookup(prices_df: pd.DataFrame) -> Dict[str, Dict[Any, float]]:
    """Build fast symbol -> {date -> close} lookup for daily NAV computation."""
    return build_price_lookup(prices_df, "close")


def build_vol_lookup(prices_df: pd.DataFrame, window: int = 20) -> Dict[str, Dict[Any, float]]:
    """Precompute trailing 20-day realized daily-return volatility per symbol/date."""
    close_pivot = prices_df.pivot(index="date", columns="symbol", values="close").sort_index()
    rolling_vol = close_pivot.pct_change(fill_method=None).rolling(window=window, min_periods=10).std()
    result: Dict[str, Dict[Any, float]] = {}
    for symbol in rolling_vol.columns:
        sym_series = rolling_vol[symbol].dropna()
        result[str(symbol)] = {pd.Timestamp(d): float(v) for d, v in sym_series.items()}
    return result


def build_ma_lookup(prices_df: pd.DataFrame, window: int = 20) -> Dict[str, Dict[Any, float]]:
    """Precompute trailing N-day simple moving average of close per symbol/date."""
    close_pivot = prices_df.pivot(index="date", columns="symbol", values="close").sort_index()
    ma = close_pivot.rolling(window=window, min_periods=10).mean()
    result: Dict[str, Dict[Any, float]] = {}
    for symbol in ma.columns:
        sym_series = ma[symbol].dropna()
        result[str(symbol)] = {pd.Timestamp(d): float(v) for d, v in sym_series.items()}
    return result


def build_symbol_price_cache(
    prices_df: pd.DataFrame,
) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    """Build per-symbol sorted frames for faster historical lookups."""
    cache: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {}
    for symbol, group in prices_df.groupby("symbol", sort=False):
        ordered = group.sort_values("date").reset_index(drop=True)
        cache[symbol] = (ordered, ordered["date"].to_numpy(dtype="datetime64[ns]"))
    return cache


def get_symbol_history(
    symbol_cache: Optional[Dict[str, Tuple[pd.DataFrame, np.ndarray]]],
    symbol: str,
    target_date: datetime,
    *,
    tail: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Return symbol history on or before target_date, optionally truncated to tail rows."""
    if symbol_cache is None:
        return None
    cached = symbol_cache.get(symbol)
    if cached is None:
        return None
    frame, dates = cached
    end_idx = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(target_date)), side="right"))
    if end_idx <= 0:
        return None
    start_idx = max(0, end_idx - tail) if tail is not None else 0
    return frame.iloc[start_idx:end_idx]


def get_price_at_date(prices_df: pd.DataFrame, symbol: str, target_date: datetime,
                      look_forward_days: int = 5, use_open: bool = False) -> Optional[float]:
    """Get price at or after target date (for execution)."""
    result = get_price_with_fill_date(
        prices_df=prices_df,
        symbol=symbol,
        target_date=target_date,
        look_forward_days=look_forward_days,
        use_open=use_open,
    )
    if result is None:
        return None
    _, price = result
    return price


def get_price_with_fill_date(
    prices_df: pd.DataFrame,
    symbol: str,
    target_date: datetime,
    look_forward_days: int = 5,
    use_open: bool = False,
) -> Optional[Tuple[pd.Timestamp, float]]:
    """Return (fill_date, price) at or after target_date within look_forward_days."""
    sym_prices = prices_df[prices_df["symbol"] == symbol].sort_values("date")
    if sym_prices.empty:
        return None

    target = pd.Timestamp(target_date)
    future = sym_prices[sym_prices["date"] >= target]
    if future.empty:
        return None

    first_available = future.iloc[0]
    if (first_available["date"] - target).days > look_forward_days:
        return None

    price_col = "open" if use_open else "close"
    return pd.Timestamp(first_available["date"]), float(first_available[price_col])


def get_price_on_or_before_date(
    prices_df: pd.DataFrame,
    symbol: str,
    target_date: datetime,
    use_open: bool = False,
) -> Optional[Tuple[pd.Timestamp, float]]:
    """Return (date, price) using the latest available row on or before target_date."""
    sym_prices = prices_df[prices_df["symbol"] == symbol].sort_values("date")
    if sym_prices.empty:
        return None

    target = pd.Timestamp(target_date)
    past = sym_prices[sym_prices["date"] <= target]
    if past.empty:
        return None

    row = past.iloc[-1]
    price_col = "open" if use_open else "close"
    return pd.Timestamp(row["date"]), float(row[price_col])


def get_prev_close(prices_df: pd.DataFrame, symbol: str, target_date: datetime) -> Optional[float]:
    """Get previous trading day's close for circuit breaker calculation."""
    sym_prices = prices_df[
        (prices_df["symbol"] == symbol) &
        (prices_df["date"] < target_date)
    ].sort_values("date")
    if sym_prices.empty:
        return None
    return float(sym_prices.iloc[-1]["close"])


# =============================================================================
# REGIME FILTER
# =============================================================================

def compute_market_regime(
    prices_df: pd.DataFrame,
    date: datetime,
    lookback: int = 60,
    bear_threshold: float = -0.05,
    market_return_cache: Optional[Dict[pd.Timestamp, float]] = None,
) -> str:
    """
    Compute market regime from broad market returns.
    Uses median stock return over lookback period as a regime indicator.

    Parameters
    ----------
    bear_threshold : float
        Median return threshold below which market is classified as bear.
        Default -0.05 (5% decline). Use -0.03 for tighter detection.

    Returns: 'bull', 'neutral', or 'bear'
    """
    cache_key = pd.Timestamp(date)
    if market_return_cache is not None and cache_key in market_return_cache:
        market_return = float(market_return_cache[cache_key])
    else:
        market_data = prices_df[prices_df["date"] <= date]
        recent_dates = sorted(market_data["date"].unique())
        if len(recent_dates) < lookback:
            return "neutral"

        recent_dates = recent_dates[-lookback:]
        first_date = recent_dates[0]
        last_date = recent_dates[-1]

        first_prices = market_data[market_data["date"] == first_date].set_index("symbol")["close"]
        last_prices = market_data[market_data["date"] == last_date].set_index("symbol")["close"]

        common = first_prices.index.intersection(last_prices.index)
        if len(common) < 20:
            return "neutral"

        returns = (last_prices[common] / first_prices[common] - 1)
        market_return = float(returns.median())

    if market_return < bear_threshold:
        return "bear"
    elif market_return < 0.02:
        return "neutral"
    else:
        return "bull"


def get_symbol_sector(symbol: str) -> Optional[str]:
    """Look up sector for a symbol using SECTOR_GROUPS."""
    for sector, symbols in SECTOR_GROUPS.items():
        if symbol in symbols:
            return sector
    return None


def build_market_return_cache(
    prices_df: pd.DataFrame,
    lookback: int = 60,
) -> Dict[pd.Timestamp, float]:
    """Precompute median market return over the rolling lookback window."""
    close_pivot = prices_df.pivot(index="date", columns="symbol", values="close").sort_index()
    shifted = close_pivot.shift(lookback - 1)
    returns = close_pivot / shifted - 1.0
    medians = returns.median(axis=1, skipna=True)
    counts = returns.notna().sum(axis=1)
    cache: Dict[pd.Timestamp, float] = {}
    for date, value in medians.items():
        if counts.loc[date] >= 20 and pd.notna(value):
            cache[pd.Timestamp(date)] = float(value)
    return cache


# =============================================================================
# SIGNAL GENERATORS (unchanged logic, fixed labels)
# =============================================================================

def generate_momentum_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    lookback_short: int = 20,
    lookback_long: int = 50,
    min_volume: float = 100000,
    symbol_cache: Optional[Dict[str, Tuple[pd.DataFrame, np.ndarray]]] = None,
) -> List[AlphaSignal]:
    """Generate momentum signals using only data available up to date."""
    signals = []
    symbols = list(symbol_cache.keys()) if symbol_cache is not None else prices_df["symbol"].unique()

    for symbol in symbols:
        sym_df = get_symbol_history(symbol_cache, symbol, date, tail=lookback_long + 10)
        if sym_df is None:
            sym_df = prices_df[
                (prices_df["symbol"] == symbol) &
                (prices_df["date"] <= date)
            ].sort_values("date")

        if len(sym_df) < lookback_long + 10:
            continue

        recent = sym_df.tail(lookback_long + 10)
        close = recent["close"]
        volume = recent["volume"]

        avg_volume = volume.iloc[-20:].mean()
        if avg_volume < min_volume:
            continue

        sma_short = close.rolling(lookback_short).mean()
        sma_long = close.rolling(lookback_long).mean()
        vol_avg = volume.rolling(lookback_short).mean()

        current_price = close.iloc[-1]
        current_sma_short = sma_short.iloc[-1]
        current_sma_long = sma_long.iloc[-1]
        current_volume = volume.iloc[-1]
        current_vol_avg = vol_avg.iloc[-1]

        if pd.isna(current_sma_short) or pd.isna(current_sma_long):
            continue

        if not (current_price > current_sma_short > current_sma_long):
            continue

        strength = 0.0
        reasons = []

        trend_strength = (current_price / current_sma_long - 1)
        strength += min(trend_strength * 2, 0.4)
        reasons.append(f"Uptrend: Price > SMA20 > SMA50")

        if current_volume > current_vol_avg * 1.5:
            strength += 0.2
            reasons.append(f"Volume {current_volume/current_vol_avg:.1f}x avg")

        roc_20 = (current_price / close.iloc[-lookback_short] - 1)
        if roc_20 > 0.05:
            strength += min(roc_20, 0.3)
            reasons.append(f"ROC(20) = {roc_20:.1%}")

        if strength > 0.2:
            signals.append(AlphaSignal(
                symbol=symbol,
                signal_type=SignalType.MOMENTUM,
                direction=1,
                strength=min(strength, 1.0),
                confidence=0.5,
                reasoning="; ".join(reasons),
            ))

    return signals


def generate_volume_breakout_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    spike_threshold: float = 3.0,
    min_turnover: float = 1000000,
    symbol_cache: Optional[Dict[str, Tuple[pd.DataFrame, np.ndarray]]] = None,
) -> List[AlphaSignal]:
    """Generate volume breakout signals using only data available up to date."""
    signals = []
    symbols = list(symbol_cache.keys()) if symbol_cache is not None else prices_df["symbol"].unique()

    for symbol in symbols:
        sym_df = get_symbol_history(symbol_cache, symbol, date, tail=65)
        if sym_df is None:
            sym_df = prices_df[
                (prices_df["symbol"] == symbol) &
                (prices_df["date"] <= date)
            ].sort_values("date")

        if len(sym_df) < 60:
            continue

        recent = sym_df.tail(65)
        close = recent["close"]
        volume = recent["volume"]

        current_turnover = close.iloc[-1] * volume.iloc[-1]
        if current_turnover < min_turnover:
            continue

        vol_60d_avg = volume.iloc[-60:].mean()
        vol_5d_avg = volume.iloc[-5:].mean()

        if vol_60d_avg == 0:
            continue

        if vol_5d_avg < vol_60d_avg * spike_threshold:
            continue

        spike_ratio = vol_5d_avg / vol_60d_avg
        strength = min((spike_ratio - 1) * 0.15, 0.5)

        price_change_5d = close.iloc[-1] / close.iloc[-5] - 1 if close.iloc[-5] > 0 else 0
        if price_change_5d < 0:
            strength *= 0.5

        if strength > 0.15:
            signals.append(AlphaSignal(
                symbol=symbol,
                signal_type=SignalType.LIQUIDITY,
                direction=1,
                strength=min(strength, 1.0),
                confidence=0.45,
                reasoning=f"Volume spike {spike_ratio:.1f}x 60-day avg, price change {price_change_5d:.1%}",
            ))

    return signals


def generate_corporate_action_signals_at_date(
    prices_df: pd.DataFrame,
    corp_actions_df: pd.DataFrame,
    date: datetime,
    days_before_event: int = 21,
    min_volume: float = 100000,
    short_term_mode: bool = False,
    trading_dates_list: Optional[List] = None,
) -> List[AlphaSignal]:
    """Generate signals for upcoming corporate actions (dividends/bonuses).

    When short_term_mode=True (event-driven portfolio):
    - Entry window: T-7 to T-5 (5-8 calendar days before bookclose)
    - Filter: cash_dividend_pct >= 5 OR bonus_share_pct >= 10
    - Sets target_exit_date to bookclose T-1 (1 trading day before)
    - Higher confidence (0.70) and yield-weighted strength
    """
    signals = []

    if corp_actions_df is None or corp_actions_df.empty:
        return signals

    current_date = pd.Timestamp(date)

    if short_term_mode:
        # Short-term: look 5-12 calendar days ahead (T-10 to T-5 window)
        # Wider than original T-7 to capture more events across weekends/holidays
        window_min = current_date + timedelta(days=5)
        window_max = current_date + timedelta(days=12)
        upcoming = corp_actions_df[
            (corp_actions_df["bookclose_date"] >= window_min) &
            (corp_actions_df["bookclose_date"] <= window_max)
        ]
    else:
        cutoff_date = current_date + timedelta(days=days_before_event)
        upcoming = corp_actions_df[
            (corp_actions_df["bookclose_date"] > current_date) &
            (corp_actions_df["bookclose_date"] <= cutoff_date)
        ]

    for _, action in upcoming.iterrows():
        symbol = action["symbol"]

        sym_prices = prices_df[
            (prices_df["symbol"] == symbol) &
            (prices_df["date"] <= date)
        ]

        if len(sym_prices) < 20:
            continue

        recent = sym_prices.tail(20)
        avg_volume = recent["volume"].mean()

        if avg_volume < min_volume:
            continue

        cash_div = action.get("cash_dividend_pct") or 0
        bonus = action.get("bonus_share_pct") or 0

        if short_term_mode:
            # Filter: only high-yield events
            if cash_div < 5 and bonus < 10:
                continue

            # Strength weighted by yield
            strength = min(cash_div / 15 + bonus / 20, 0.8)
            days_to_event = (action["bookclose_date"] - current_date).days

            reasons = []
            if cash_div > 0:
                reasons.append(f"Cash Div: {cash_div}%")
            if bonus > 0:
                reasons.append(f"Bonus: {bonus}%")

            # Compute target exit date: 1 trading day before bookclose
            bookclose = action["bookclose_date"]
            target_exit = None
            if trading_dates_list is not None:
                # Find the trading day just before bookclose
                bc_ts = pd.Timestamp(bookclose)
                prior_dates = [d for d in trading_dates_list if pd.Timestamp(d) < bc_ts]
                if prior_dates:
                    target_exit = pd.Timestamp(prior_dates[-1])
            else:
                # Fallback: 1 calendar day before
                target_exit = bookclose - timedelta(days=1)

            signals.append(AlphaSignal(
                symbol=symbol,
                signal_type=SignalType.CORPORATE_ACTION,
                direction=1,
                strength=strength,
                confidence=0.70,
                reasoning=f"[ST] Bookclose in {days_to_event}d: {', '.join(reasons)}",
                target_exit_date=target_exit,
            ))
        else:
            # Original long-term logic
            strength = 0.0
            reasons = []

            if cash_div > 0:
                strength += min(cash_div / 20, 0.3)
                reasons.append(f"Cash Div: {cash_div}%")

            if bonus > 0:
                strength += min(bonus / 30, 0.4)
                reasons.append(f"Bonus: {bonus}%")

            days_to_event = (action["bookclose_date"] - current_date).days
            urgency_bonus = max(0, (days_before_event - days_to_event) / days_before_event * 0.2)
            strength += urgency_bonus

            if strength > 0.15:
                signals.append(AlphaSignal(
                    symbol=symbol,
                    signal_type=SignalType.CORPORATE_ACTION,
                    direction=1,
                    strength=min(strength, 0.8),
                    confidence=0.6,
                    reasoning=f"Bookclose in {days_to_event}d: {', '.join(reasons)}",
                ))

    return signals


def generate_settlement_pressure_signals_at_date(
    prices_df: pd.DataFrame,
    corp_actions_df: pd.DataFrame,
    date: datetime,
    trading_dates_list: Optional[List] = None,
    min_volume: float = 100000,
    min_drop_pct: float = 0.03,
) -> List[AlphaSignal]:
    """Generate signals from T+2 settlement selling pressure after bookclose.

    After bookclose, shareholders who bought just for the dividend are forced
    to sell (or face holding an ex-dividend stock). T+2 settlement creates a
    predictable 2-4 trading day window of forced selling pressure. This is a
    mean-reversion play: buy the dip from forced selling.

    Logic:
    - Look for bookclose dates that occurred 2-4 trading days ago
    - If stock dropped >3% from bookclose to now → settlement selling pressure
    - Filter: div >= 5% or bonus >= 10% (same high-yield filter)
    - Direction: +1 (buy the dip)
    """
    signals = []

    if corp_actions_df is None or corp_actions_df.empty:
        return signals

    current_date = pd.Timestamp(date)

    # Look for bookclose dates 2-8 calendar days ago (covers 2-4 trading days)
    lookback_start = current_date - timedelta(days=8)
    lookback_end = current_date - timedelta(days=2)

    recent_bookcloses = corp_actions_df[
        (corp_actions_df["bookclose_date"] >= lookback_start) &
        (corp_actions_df["bookclose_date"] <= lookback_end)
    ]

    for _, action in recent_bookcloses.iterrows():
        symbol = action["symbol"]
        cash_div = action.get("cash_dividend_pct") or 0
        bonus = action.get("bonus_share_pct") or 0

        # Only high-yield events
        if cash_div < 5 and bonus < 10:
            continue

        sym_prices = prices_df[
            (prices_df["symbol"] == symbol) &
            (prices_df["date"] <= date)
        ].sort_values("date")

        if len(sym_prices) < 20:
            continue

        # Volume check
        recent = sym_prices.tail(20)
        avg_volume = recent["volume"].mean()
        if avg_volume < min_volume:
            continue

        # Get price at bookclose date and current price
        bookclose_date = action["bookclose_date"]
        bc_prices = sym_prices[sym_prices["date"] <= bookclose_date]
        if bc_prices.empty:
            continue

        bookclose_price = float(bc_prices.iloc[-1]["close"])
        current_price = float(sym_prices.iloc[-1]["close"])

        if bookclose_price <= 0:
            continue

        # Calculate drop from bookclose
        drop_pct = (bookclose_price - current_price) / bookclose_price

        if drop_pct < min_drop_pct:
            continue

        # Strength scales with drop size (max 0.6)
        strength = min(drop_pct / 0.08, 0.6)

        reasons = []
        if cash_div > 0:
            reasons.append(f"Div:{cash_div}%")
        if bonus > 0:
            reasons.append(f"Bonus:{bonus}%")

        signals.append(AlphaSignal(
            symbol=symbol,
            signal_type=SignalType.SETTLEMENT_PRESSURE,
            direction=1,
            strength=strength,
            confidence=0.50,
            reasoning=f"Post-bookclose drop {drop_pct:.1%} ({', '.join(reasons)})",
        ))

    return signals


def generate_mean_reversion_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    rsi_period: int = 14,
    oversold_threshold: float = 30,
    min_volume: float = 100000,
    symbol_cache: Optional[Dict[str, Tuple[pd.DataFrame, np.ndarray]]] = None,
) -> List[AlphaSignal]:
    """Generate mean reversion signals (RSI oversold bounce)."""
    signals = []
    symbols = list(symbol_cache.keys()) if symbol_cache is not None else prices_df["symbol"].unique()

    for symbol in symbols:
        sym_df = get_symbol_history(symbol_cache, symbol, date, tail=rsi_period + 20)
        if sym_df is None:
            sym_df = prices_df[
                (prices_df["symbol"] == symbol) &
                (prices_df["date"] <= date)
            ].sort_values("date")

        if len(sym_df) < rsi_period + 20:
            continue

        recent = sym_df.tail(rsi_period + 20)
        close = recent["close"]
        volume = recent["volume"]

        if volume.iloc[-20:].mean() < min_volume:
            continue

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()

        if loss.iloc[-1] == 0:
            continue

        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))

        if pd.isna(rsi):
            continue

        if rsi < oversold_threshold:
            if close.iloc[-1] > close.iloc[-2]:
                strength = (oversold_threshold - rsi) / oversold_threshold * 0.5
                # Volume confirmation: if current volume > 1.5x 20-day avg, add bonus
                avg_vol_20 = volume.iloc[-20:].mean()
                if avg_vol_20 > 0 and volume.iloc[-1] > 1.5 * avg_vol_20:
                    strength += 0.1
                signals.append(AlphaSignal(
                    symbol=symbol,
                    signal_type=SignalType.MEAN_REVERSION,
                    direction=1,
                    strength=min(strength, 0.6),
                    confidence=0.45,
                    reasoning=f"RSI oversold at {rsi:.1f}, price recovering"
                             f"{', high volume' if avg_vol_20 > 0 and volume.iloc[-1] > 1.5 * avg_vol_20 else ''}",
                ))

    return signals


def generate_low_volatility_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    lookback: int = 60,
    min_volume: float = 100000,
    volatility_percentile: float = 30,
    symbol_cache: Optional[Dict[str, Tuple[pd.DataFrame, np.ndarray]]] = None,
) -> List[AlphaSignal]:
    """Generate low volatility factor signals."""
    signals = []
    symbols = list(symbol_cache.keys()) if symbol_cache is not None else prices_df["symbol"].unique()

    volatilities = {}
    for symbol in symbols:
        sym_df = get_symbol_history(symbol_cache, symbol, date, tail=lookback)
        if sym_df is None:
            sym_df = prices_df[
                (prices_df["symbol"] == symbol) &
                (prices_df["date"] <= date)
            ].sort_values("date")

        if len(sym_df) < lookback:
            continue

        recent = sym_df.tail(lookback)
        close = recent["close"]
        volume = recent["volume"]

        if volume.iloc[-20:].mean() < min_volume:
            continue

        returns = close.pct_change().dropna()
        if len(returns) < 20:
            continue
        vol = returns.std() * np.sqrt(NEPSE_TRADING_DAYS)
        volatilities[symbol] = vol

    if not volatilities:
        return signals

    vol_values = list(volatilities.values())
    threshold = np.percentile(vol_values, volatility_percentile)

    for symbol, vol in volatilities.items():
        if vol > threshold:
            continue

        sym_df = get_symbol_history(symbol_cache, symbol, date, tail=lookback)
        if sym_df is None:
            sym_df = prices_df[
                (prices_df["symbol"] == symbol) &
                (prices_df["date"] <= date)
            ].sort_values("date").tail(lookback)

        close = sym_df["close"]

        sma_20 = close.rolling(20).mean().iloc[-1]
        current_price = close.iloc[-1]

        if current_price < sma_20:
            continue

        vol_rank = 1 - (vol / threshold)
        strength = 0.3 + vol_rank * 0.3

        signals.append(AlphaSignal(
            symbol=symbol,
            signal_type=SignalType.FUNDAMENTAL,
            direction=1,
            strength=min(strength, 0.6),
            confidence=0.5,
            reasoning=f"Low volatility ({vol:.1%} annual), price above SMA20",
        ))

    return signals


# =============================================================================
# LIQUIDITY PRE-FILTER (Phase 2A)
# =============================================================================

def compute_liquid_universe(
    prices_df: pd.DataFrame,
    current_date: datetime,
    percentile_cutoff: int = 30,
    lookback: int = 20,
) -> List[str]:
    """
    Compute the liquid universe by removing bottom N% by 20-day average turnover.

    Returns list of symbols that pass the liquidity filter.
    This should be applied BEFORE signal computation to reduce noise
    from illiquid names in the ranking.
    """
    # Exclude non-tradable symbols (market index, sector indices)
    all_symbols = prices_df["symbol"].unique()
    symbols = [s for s in all_symbols if s != "NEPSE" and not str(s).startswith("SECTOR::")]
    turnover_map = {}

    for symbol in symbols:
        sym_df = prices_df[
            (prices_df["symbol"] == symbol) &
            (prices_df["date"] <= current_date)
        ].sort_values("date").tail(lookback)

        if len(sym_df) < lookback // 2:
            continue

        avg_turnover = (sym_df["close"] * sym_df["volume"]).mean()
        if avg_turnover > 0:
            turnover_map[symbol] = avg_turnover

    if not turnover_map:
        return list(symbols)

    threshold = np.percentile(list(turnover_map.values()), percentile_cutoff)
    return [s for s, t in turnover_map.items() if t >= threshold]


# =============================================================================
# CROSS-SECTIONAL MOMENTUM (Phase 2B)
# =============================================================================

def generate_xsec_momentum_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    min_volume: float = 100000,
    top_pct: float = 0.10,
    liquid_symbols: Optional[List[str]] = None,
) -> List[AlphaSignal]:
    """
    Cross-sectional momentum: rank stocks by 6-month return minus last month
    (skip recent month to avoid short-term reversal).

    Unlike absolute momentum (price > SMA), this ranks stocks AGAINST each other.
    In bull markets, random picks also go up (beta), but cross-sectional ranking
    selects relative WINNERS — the key to separating alpha from beta.
    """
    signals = []
    symbols = liquid_symbols if liquid_symbols else prices_df["symbol"].unique()

    scores = {}
    for symbol in symbols:
        sym_df = prices_df[
            (prices_df["symbol"] == symbol) &
            (prices_df["date"] <= date)
        ].sort_values("date")

        if len(sym_df) < 130:  # Need ~6 months of data
            continue

        recent = sym_df.tail(130)
        close = recent["close"].values
        volume = recent["volume"].values

        # Volume filter
        if np.mean(volume[-20:]) < min_volume:
            continue

        # 6-month return
        if close[-126] <= 0:
            continue
        ret_6m = close[-1] / close[-126] - 1

        # 1-month return (to skip — reversal avoidance)
        if close[-21] <= 0:
            continue
        ret_1m = close[-1] / close[-21] - 1

        # Cross-sectional momentum = 6m return minus last month
        xsec_score = ret_6m - ret_1m
        scores[symbol] = xsec_score

    if len(scores) < 10:
        return signals

    # Rank cross-sectionally
    sorted_syms = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
    n = len(sorted_syms)
    cutoff = max(1, int(n * top_pct))
    top_syms = set(sorted_syms[:cutoff])

    for symbol in top_syms:
        rank_in_top = sorted_syms.index(symbol)
        strength = (1.0 - rank_in_top / cutoff) * 0.5  # Scale 0.0-0.5, no floor

        signals.append(AlphaSignal(
            symbol=symbol,
            signal_type=SignalType.XSEC_MOMENTUM,
            direction=1,
            strength=min(strength, 0.5),
            confidence=0.45,
            reasoning=f"XSec momentum rank {rank_in_top+1}/{n}, "
                      f"score={scores[symbol]:.2%}",
        ))

    return signals


# =============================================================================
# ACCUMULATION/DISTRIBUTION (Phase 2C)
# =============================================================================

def generate_accumulation_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    min_volume: float = 100000,
    obv_roc_threshold: float = 0.25,
    cmf_threshold: float = 0.10,
    liquid_symbols: Optional[List[str]] = None,
) -> List[AlphaSignal]:
    """
    Detect sustained accumulation using OBV rate-of-change and Chaikin Money Flow.

    In NEPSE's retail market, volume leads price. Current volume_breakout only
    detects spikes. OBV/CMF detect sustained buying pressure (accumulation).

    OBV rising + CMF positive = smart money accumulating.
    """
    signals = []
    symbols = liquid_symbols if liquid_symbols else prices_df["symbol"].unique()

    for symbol in symbols:
        sym_df = prices_df[
            (prices_df["symbol"] == symbol) &
            (prices_df["date"] <= date)
        ].sort_values("date")

        if len(sym_df) < 30:
            continue

        recent = sym_df.tail(30)
        close = recent["close"].values
        high = recent["high"].values
        low = recent["low"].values
        volume = recent["volume"].values

        # Volume filter
        if np.mean(volume[-20:]) < min_volume:
            continue

        # On Balance Volume (OBV)
        close_change = np.diff(close)
        obv_changes = np.where(
            close_change > 0, volume[1:],
            np.where(close_change < 0, -volume[1:], 0)
        )
        obv = np.cumsum(obv_changes)

        if len(obv) < 21:
            continue

        # OBV rate-of-change (20-day)
        obv_old = obv[-21]
        obv_new = obv[-1]
        if abs(obv_old) < 1:
            continue
        obv_roc = (obv_new - obv_old) / abs(obv_old)

        # Chaikin Money Flow (21-day)
        hl_range = high[-21:] - low[-21:]
        hl_range = np.where(hl_range == 0, 1e-10, hl_range)
        mf_mult = ((close[-21:] - low[-21:]) - (high[-21:] - close[-21:])) / hl_range
        mf_volume = mf_mult * volume[-21:]
        total_volume = np.sum(volume[-21:])
        if total_volume == 0:
            continue
        cmf_21 = np.sum(mf_volume) / total_volume

        # Price confirmation: skip accumulation in downtrends
        sma_20 = np.mean(close[-20:])
        if close[-1] < sma_20:
            continue

        # Signal: OBV rising + CMF positive + price above SMA20 = accumulation
        if obv_roc > obv_roc_threshold and cmf_21 > cmf_threshold:
            strength = min(obv_roc * 0.3 + cmf_21 * 0.5, 0.5)
            signals.append(AlphaSignal(
                symbol=symbol,
                signal_type=SignalType.ACCUMULATION,
                direction=1,
                strength=max(strength, 0.15),
                confidence=0.45,
                reasoning=f"Accumulation: OBV ROC={obv_roc:.1%}, CMF={cmf_21:.3f}",
            ))

    return signals


# =============================================================================
# CIRCUIT-BREAKER REVERSAL FILTER (Phase 2D)
# =============================================================================

def is_circuit_breaker_hit(
    prices_df: pd.DataFrame,
    symbol: str,
    date: datetime,
    lookback_days: int = 3,
    threshold: float = 0.095,
) -> bool:
    """
    Check if a stock hit or nearly hit the +10% circuit breaker in the last
    N trading days. Stocks hitting CB from retail herding tend to reverse
    within 5-10 days — filter them out of buy signals.
    """
    sym_df = prices_df[
        (prices_df["symbol"] == symbol) &
        (prices_df["date"] <= date)
    ].sort_values("date").tail(lookback_days + 1)

    if len(sym_df) < 2:
        return False

    close = sym_df["close"].values
    daily_returns = np.diff(close) / close[:-1]

    return bool(np.any(daily_returns > threshold))


# =============================================================================
# 52-WEEK HIGH PROXIMITY (George & Hwang 2004) — Citadel Sprint 1
# =============================================================================

def generate_52wk_high_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    proximity_threshold: float = 0.90,  # within 10% of 52W high
    min_volume: float = 100000,
    liquid_symbols: Optional[List[str]] = None,
) -> List[AlphaSignal]:
    """
    52-Week High Proximity signal (George & Hwang 2004).

    Stocks near their 252-day high exhibit anchoring bias: retail investors
    anchor on the round-number high and are reluctant to push through it.
    When the stock finally breaks through, strong continuation follows.

    In NEPSE's retail-dominated market, this anchoring effect is amplified —
    ShareSansar/MeroLagani prominently display 52-week highs and retail
    traders fixate on these levels.

    NO lookahead bias: uses only data up to and including `date`.
    """
    signals = []
    symbols = liquid_symbols if liquid_symbols else prices_df["symbol"].unique()

    for symbol in symbols:
        # Skip index entries
        if symbol == "NEPSE" or str(symbol).startswith("SECTOR::"):
            continue

        sym_df = prices_df[
            (prices_df["symbol"] == symbol) &
            (prices_df["date"] <= date)
        ].sort_values("date")

        # Need at least 252 trading days of history
        if len(sym_df) < 252:
            continue

        recent = sym_df.tail(252)
        high = recent["high"].values
        close = recent["close"].values
        volume = recent["volume"].values

        # Volume filter: 20-day average volume
        if np.mean(volume[-20:]) < min_volume:
            continue

        # 252-day rolling high
        high_252 = np.max(high)
        current_close = close[-1]

        if high_252 <= 0 or current_close <= 0:
            continue

        # Proximity to 52-week high
        proximity = current_close / high_252

        # Filter: must be within threshold of 52-week high
        if proximity < proximity_threshold:
            continue

        # Signal strength = proximity (closer to 1.0 = stronger)
        strength = float(proximity)

        # Confidence scales with proximity: 0.6 + 0.3 * proximity
        confidence = 0.6 + 0.3 * float(proximity)

        signals.append(AlphaSignal(
            symbol=symbol,
            signal_type=SignalType.ANCHORING_52WK,
            direction=1,
            strength=min(strength, 1.0),
            confidence=min(confidence, 1.0),
            reasoning=f"52W high proximity: {proximity:.1%} "
                      f"(close={current_close:.0f}, high={high_252:.0f})",
        ))

    return signals


# =============================================================================
# VALUE BOUNCE — 52-week low proximity + price recovery (contrarian, anti-momentum)
# =============================================================================

def generate_value_bounce_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    low_proximity_threshold: float = 1.35,   # within 35% above 52W low
    recovery_days: int = 10,                  # must be +ve over last N days
    min_volume: float = 40000,                # lower than other signals — captures small hydro/value
    liquid_symbols: Optional[List[str]] = None,
) -> List[AlphaSignal]:
    """
    Value Bounce / Deep Reversal signal.

    Stocks near their 52-week LOW that have begun recovering (price positive
    over last recovery_days) are targeted for mean reversion at the multi-month
    timeframe. This is a contrarian, anti-momentum signal designed to catch
    sector rotations and beaten-down value stocks returning to fair value.

    In NEPSE's retail-dominated market, cyclical sectors (hydro, manufacturing)
    often overshoot to the downside in bear regimes then snap back sharply.

    Filter logic:
    1. Stock is within low_proximity_threshold×52w_low (e.g. price < 1.35×52w_low)
    2. Price has been rising for last recovery_days (trend reversal confirmation)
    3. Volume is adequate (liquid enough to trade)

    This is the OPPOSITE of 52wk_high — catches cheap stocks bouncing, not
    expensive stocks breaking out.

    NO lookahead bias: uses only data up to and including `date`.
    """
    signals: List[AlphaSignal] = []
    symbols = liquid_symbols if liquid_symbols else prices_df["symbol"].unique()

    for symbol in symbols:
        if symbol == "NEPSE" or str(symbol).startswith("SECTOR::"):
            continue

        sym_df = prices_df[
            (prices_df["symbol"] == symbol) &
            (prices_df["date"] <= date)
        ].sort_values("date")

        if len(sym_df) < 252:
            continue

        recent = sym_df.tail(252)
        close_vals = recent["close"].values
        volume_vals = recent["volume"].values

        if np.mean(volume_vals[-20:]) < min_volume:
            continue

        current_close = close_vals[-1]
        low_252 = np.min(close_vals)

        if low_252 <= 0 or current_close <= 0:
            continue

        # How far above the 52-week low are we? (1.0 = at the low, 2.0 = 100% above)
        dist_from_low = current_close / low_252

        # Only signal if we're still close to the 52-week low
        if dist_from_low > low_proximity_threshold:
            continue

        # Recovery confirmation: price must be higher than N days ago
        if len(close_vals) < recovery_days + 1:
            continue

        price_n_days_ago = close_vals[-(recovery_days + 1)]
        if price_n_days_ago <= 0:
            continue

        recovery_pct = (current_close / price_n_days_ago) - 1.0
        if recovery_pct <= 0:
            continue  # still falling — wait for reversal

        # Signal strength: stronger when closer to the 52w low + recovering faster
        # dist_from_low = 1.0 → strongest; dist_from_low = 1.35 → weakest
        proximity_score = (low_proximity_threshold - dist_from_low) / (low_proximity_threshold - 1.0)
        proximity_score = max(0.0, min(1.0, proximity_score))

        recovery_score = min(recovery_pct / 0.05, 1.0)  # 5% recovery = max score

        strength = 0.3 + 0.5 * proximity_score + 0.2 * recovery_score
        strength = min(strength, 0.85)

        # Confidence: 0.40 — deliberately conservative (value bounce is anti-momentum,
        # higher confidence in neutral/bear where value rotation is more predictable)
        confidence = 0.40

        signals.append(AlphaSignal(
            symbol=symbol,
            signal_type=SignalType.VALUE_BOUNCE,
            direction=1,
            strength=min(strength, 1.0),
            confidence=confidence,
            reasoning=(
                f"Near 52W low: {dist_from_low:.2f}x ({current_close:.0f}/{low_252:.0f}), "
                f"recovery +{recovery_pct:.1%} over {recovery_days}d"
            ),
        ))

    return signals


# =============================================================================
# RESIDUAL MOMENTUM / IMOM (Blitz, Huij & Martens 2011) — Citadel Sprint 1
# =============================================================================

def generate_residual_momentum_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    formation_period: int = 252,
    skip_period: int = 21,  # skip last month
    min_volume: float = 100000,
    liquid_symbols: Optional[List[str]] = None,
) -> List[AlphaSignal]:
    """
    Residual Momentum / IMOM (Blitz, Huij & Martens 2011).

    Standard momentum captures both market beta and stock-specific alpha.
    In NEPSE's highly correlated market (all stocks rise/fall together),
    raw momentum is mostly beta. Residual momentum strips out market beta
    via OLS regression and ranks on the residual (stock-specific) return.

    This is a strict upgrade over generate_xsec_momentum_signals_at_date():
    - Same concept (cross-sectional ranking) but on beta-adjusted returns
    - Skip period avoids short-term reversal (same as xsec_momentum)

    NO lookahead bias: uses only data up to and including `date`.
    """
    signals = []
    required_days = formation_period + skip_period

    # Step 1: Get NEPSE index returns (the market factor)
    nepse_df = prices_df[
        (prices_df["symbol"] == "NEPSE") &
        (prices_df["date"] <= date)
    ].sort_values("date")

    if len(nepse_df) < required_days + 1:
        logger.debug(f"IMOM: Insufficient NEPSE index data ({len(nepse_df)} rows)")
        return signals

    nepse_close = nepse_df["close"].values
    nepse_returns = np.diff(nepse_close) / nepse_close[:-1]
    nepse_dates = nepse_df["date"].values[1:]  # dates aligned with returns

    # Build a date->return lookup for NEPSE
    nepse_ret_map = {}
    for d, r in zip(nepse_dates, nepse_returns):
        nepse_ret_map[pd.Timestamp(d)] = r

    # Step 2: For each stock, regress on NEPSE and compute residual momentum
    symbols = liquid_symbols if liquid_symbols else prices_df["symbol"].unique()
    residual_scores = {}

    for symbol in symbols:
        # Skip index entries
        if symbol == "NEPSE" or str(symbol).startswith("SECTOR::"):
            continue

        sym_df = prices_df[
            (prices_df["symbol"] == symbol) &
            (prices_df["date"] <= date)
        ].sort_values("date")

        if len(sym_df) < required_days + 1:
            continue

        recent = sym_df.tail(required_days + 1)
        close = recent["close"].values
        volume = recent["volume"].values
        dates = recent["date"].values

        # Volume filter: 20-day average
        if np.mean(volume[-20:]) < min_volume:
            continue

        # Compute daily returns for the stock
        stock_returns = np.diff(close) / close[:-1]
        stock_dates = dates[1:]

        # Align stock returns with NEPSE returns over the formation window
        # Window: from (skip_period) days ago back to (formation_period + skip_period) days ago
        # i.e., we skip the most recent skip_period days
        if len(stock_returns) < required_days:
            continue

        # Formation window indices (skip the last skip_period returns)
        formation_stock_rets = stock_returns[-(required_days):-skip_period]
        formation_dates = stock_dates[-(required_days):-skip_period]

        # Get aligned NEPSE returns
        formation_nepse_rets = []
        valid_stock_rets = []
        for j, d in enumerate(formation_dates):
            ts = pd.Timestamp(d)
            if ts in nepse_ret_map:
                formation_nepse_rets.append(nepse_ret_map[ts])
                valid_stock_rets.append(formation_stock_rets[j])

        if len(valid_stock_rets) < formation_period * 0.5:
            # Need at least 50% overlap
            continue

        X_market = np.array(formation_nepse_rets)
        Y_stock = np.array(valid_stock_rets)

        # OLS regression: R_stock = alpha + beta * R_NEPSE + epsilon
        # Using numpy.linalg.lstsq
        A = np.column_stack([np.ones(len(X_market)), X_market])
        try:
            result, _, _, _ = np.linalg.lstsq(A, Y_stock, rcond=None)
        except np.linalg.LinAlgError:
            continue

        alpha_hat = result[0]
        beta_hat = result[1]

        # Compute residuals
        predicted = alpha_hat + beta_hat * X_market
        residuals = Y_stock - predicted

        # Cumulative residual over formation window = residual momentum score
        cum_residual = float(np.sum(residuals))
        residual_scores[symbol] = cum_residual

    if len(residual_scores) < 10:
        return signals

    # Step 3: Rank cross-sectionally, take top 20%
    sorted_syms = sorted(residual_scores.keys(),
                         key=lambda s: residual_scores[s], reverse=True)
    n = len(sorted_syms)
    cutoff = max(1, int(n * 0.20))
    top_syms = sorted_syms[:cutoff]

    for rank, symbol in enumerate(top_syms):
        # Normalized rank score: top rank gets 1.0, bottom of top-20% gets ~0.0
        strength = (1.0 - rank / cutoff) * 0.6  # Scale 0.0-0.6

        signals.append(AlphaSignal(
            symbol=symbol,
            signal_type=SignalType.RESIDUAL_MOMENTUM,
            direction=1,
            strength=min(max(strength, 0.1), 0.6),
            confidence=0.50,
            reasoning=f"IMOM rank {rank+1}/{n}, "
                      f"cum_residual={residual_scores[symbol]:.3f}",
        ))

    return signals


# =============================================================================
# AMIHUD ILLIQUIDITY TILT (Amihud 2002) — Citadel Sprint 1
# =============================================================================

def apply_amihud_tilt(
    signals: List[AlphaSignal],
    prices_df: pd.DataFrame,
    date: datetime,
    lookback: int = 60,
    tilt_boost: float = 0.10,  # 10% score boost for optimal illiquidity
) -> List[AlphaSignal]:
    """
    Amihud Illiquidity Factor Tilt (Amihud 2002).

    This is NOT a standalone signal — it is a scoring overlay applied
    post-aggregation. Stocks in the moderately illiquid range (quintile 3-4)
    earn a liquidity premium. The most liquid (quintile 1) are efficiently
    priced. The most illiquid (quintile 5) have execution risk.

    Modifies signal strengths in-place and returns the modified list.

    NO lookahead bias: uses only data up to and including `date`.
    """
    if not signals:
        return signals

    # Step 1: Compute Amihud ILLIQ for each signaled symbol
    symbol_illiq = {}

    for sig in signals:
        if sig.symbol in symbol_illiq:
            continue  # Already computed for this symbol

        sym_df = prices_df[
            (prices_df["symbol"] == sig.symbol) &
            (prices_df["date"] <= date)
        ].sort_values("date").tail(lookback)

        if len(sym_df) < lookback // 2:
            continue

        close = sym_df["close"].values
        volume = sym_df["volume"].values

        # Compute daily returns
        if len(close) < 2:
            continue
        daily_returns = np.diff(close) / close[:-1]
        close_for_turnover = close[1:]
        volume_for_turnover = volume[1:]

        # Turnover in NPR = close * volume (volume is in shares)
        turnover = close_for_turnover * volume_for_turnover

        # Avoid division by zero
        valid = turnover > 0
        if np.sum(valid) < 10:
            continue

        # ILLIQ = mean( |daily_return| / turnover )
        illiq_values = np.abs(daily_returns[valid]) / turnover[valid]
        illiq = float(np.mean(illiq_values))

        symbol_illiq[sig.symbol] = illiq

    if len(symbol_illiq) < 5:
        return signals  # Not enough data to rank into quintiles

    # Step 2: Rank by ILLIQ and assign quintiles
    illiq_values = np.array(list(symbol_illiq.values()))
    illiq_symbols = list(symbol_illiq.keys())

    # Quintile boundaries (5 equal groups)
    quintile_bounds = np.percentile(illiq_values, [20, 40, 60, 80])

    def get_quintile(illiq_val):
        """Return quintile 1-5 (1=most liquid, 5=most illiquid)."""
        if illiq_val <= quintile_bounds[0]:
            return 1
        elif illiq_val <= quintile_bounds[1]:
            return 2
        elif illiq_val <= quintile_bounds[2]:
            return 3
        elif illiq_val <= quintile_bounds[3]:
            return 4
        else:
            return 5

    symbol_quintile = {sym: get_quintile(symbol_illiq[sym])
                       for sym in illiq_symbols}

    # Step 3: Apply tilt — quintile 3-4 get strength boost
    for sig in signals:
        q = symbol_quintile.get(sig.symbol)
        if q is not None and q in (3, 4):
            sig.strength *= (1 + tilt_boost)
            sig.strength = min(sig.strength, 1.0)  # Cap at 1.0

    return signals


def generate_quality_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    lookback: int = 60,
    min_volume: float = 100000,
    symbol_cache: Optional[Dict[str, Tuple[pd.DataFrame, np.ndarray]]] = None,
) -> List[AlphaSignal]:
    """Generate quality/stability signals."""
    signals = []
    symbols = list(symbol_cache.keys()) if symbol_cache is not None else prices_df["symbol"].unique()

    for symbol in symbols:
        sym_df = get_symbol_history(symbol_cache, symbol, date, tail=lookback)
        if sym_df is None:
            sym_df = prices_df[
                (prices_df["symbol"] == symbol) &
                (prices_df["date"] <= date)
            ].sort_values("date")

        if len(sym_df) < lookback:
            continue

        recent = sym_df.tail(lookback)
        close = recent["close"]
        volume = recent["volume"]

        avg_volume = volume.iloc[-20:].mean()
        if avg_volume < min_volume:
            continue

        returns = close.pct_change().dropna()
        if len(returns) < 20:
            continue

        vol = returns.std() * np.sqrt(NEPSE_TRADING_DAYS)
        if vol > 0.6:
            continue

        total_return = close.iloc[-1] / close.iloc[0] - 1
        if total_return < 0:
            continue

        positive_days = (returns > 0).mean()
        if positive_days < 0.48:
            continue

        vol_cv = volume.std() / volume.mean() if volume.mean() > 0 else float('inf')
        if vol_cv > 2.0:
            continue

        quality_score = 0.0
        reasons = []

        vol_score = max(0, (0.5 - vol) / 0.5) * 0.3
        quality_score += vol_score
        reasons.append(f"Vol: {vol:.0%}")

        ret_score = min(total_return * 2, 0.3)
        quality_score += ret_score
        reasons.append(f"Ret: {total_return:.1%}")

        cons_score = (positive_days - 0.48) * 0.6
        quality_score += cons_score
        reasons.append(f"Win%: {positive_days:.0%}")

        if quality_score > 0.25:
            signals.append(AlphaSignal(
                symbol=symbol,
                signal_type=SignalType.FUNDAMENTAL,
                direction=1,
                strength=min(quality_score, 0.7),
                confidence=0.55,
                reasoning=f"Quality: {', '.join(reasons)}",
            ))

    return signals


def generate_quarterly_fundamental_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    model: QuarterlyFundamentalModel,
    *,
    min_volume: float = 100000,
    liquid_symbols: Optional[List[str]] = None,
) -> List[AlphaSignal]:
    """Generate point-in-time quarterly fundamental signals."""
    symbols = liquid_symbols if liquid_symbols else list(prices_df["symbol"].unique())
    current_prices: Dict[str, float] = {}
    eligible_symbols: List[str] = []

    for symbol in symbols:
        sym_df = prices_df[
            (prices_df["symbol"] == symbol) &
            (prices_df["date"] <= date)
        ].sort_values("date")
        if len(sym_df) < 20:
            continue

        recent = sym_df.tail(20)
        avg_volume = recent["volume"].mean()
        if avg_volume < min_volume:
            continue

        current_price = float(recent["close"].iloc[-1])
        if current_price <= 0:
            continue

        eligible_symbols.append(symbol)
        current_prices[symbol] = current_price

    if not eligible_symbols:
        return []

    return model.generate_signals(
        date,
        current_prices,
        sector_lookup=get_symbol_sector,
        symbols=eligible_symbols,
    )


def load_corporate_actions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load corporate actions from database."""
    query = """
        SELECT symbol, bookclose_date, cash_dividend_pct, bonus_share_pct,
               right_share_ratio, agenda
        FROM corporate_actions
        WHERE bookclose_date IS NOT NULL
        ORDER BY bookclose_date
    """
    try:
        df = pd.read_sql_query(query, conn)
        df["bookclose_date"] = pd.to_datetime(df["bookclose_date"])
        return df
    except Exception as e:
        logger.warning(f"Could not load corporate actions: {e}")
        return pd.DataFrame()


# =============================================================================
# PORTFOLIO-BASED BACKTEST ENGINE
# =============================================================================

def run_backtest(
    start_date: str,
    end_date: str,
    holding_days: int = 40,
    max_positions: int = 5,
    signal_types: List[str] = None,
    initial_capital: float = 1_000_000,
    rebalance_frequency: int = 5,
    use_trailing_stop: bool = True,
    trailing_stop_pct: float = 0.10,
    stop_loss_pct: float = 0.08,
    use_regime_filter: bool = True,
    sector_limit: float = 0.35,  # Max 35% in one sector
    regime_max_positions: Optional[Dict[str, int]] = None,
    bear_threshold: float = -0.05,
    profit_target_pct: Optional[float] = None,
    event_exit_mode: bool = False,
    regime_adaptive_hold: bool = False,
    regime_hold_days: Optional[Dict[str, int]] = None,
    regime_sector_limits: Optional[Dict[str, float]] = None,
    use_broker_exit: bool = False,
) -> BacktestResult:
    """
    Run walk-forward portfolio backtest with realistic capital tracking.

    Key properties:
    - Tracks actual capital (cash + position MTM)
    - Uses NEPSE tiered fee structure
    - Entries at T+1 open, stop checks at open price
    - Circuit breaker relative to previous close
    - Optional regime filter and sector concentration limits
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if signal_types is None:
        signal_types = ["volume", "quality"]

    logger.info("Loading price data...")
    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_all_prices(conn)

    corp_actions_df = None
    quarterly_fundamental_model = None
    if "corp_action" in signal_types or "settlement_pressure" in signal_types:
        logger.info("Loading corporate actions...")
        corp_actions_df = load_corporate_actions(conn)
        logger.info(f"Loaded {len(corp_actions_df)} corporate actions")

    if "quarterly_fundamental" in signal_types:
        logger.info("Loading point-in-time quarterly fundamentals...")
        quarterly_fundamental_model = QuarterlyFundamentalModel.from_connection(conn)

    broker_summary_df = None  # informed_trading signal not included in public release

    broker_v2_df = None
    if "smart_money" in signal_types or "smart_money_pure" in signal_types:
        logger.info("Loading broker_signals_v2 (smart money / circular detection)...")
        try:
            broker_v2_df = pd.read_sql(
                "SELECT symbol, as_of_date, hhi_buy, hhi_sell, circular_score, "
                "top_pair_pct, smart_money_score, pump_score "
                "FROM broker_signals_v2 ORDER BY symbol, as_of_date",
                conn,
                parse_dates=["as_of_date"],
            )
            logger.info(f"Loaded {len(broker_v2_df)} broker_v2 rows "
                        f"for {broker_v2_df['symbol'].nunique()} symbols")
        except Exception as e:
            logger.warning(f"Could not load broker_signals_v2: {e}")
            broker_v2_df = None

    # ── Broker pump-exit lookup ──────────────────────────────────────────────
    # Fast (symbol, date_str) → (circular_score, pump_score) dict for O(1) exit checks.
    broker_exit_lookup: Dict[tuple, tuple] = {}
    if use_broker_exit:
        try:
            _bex_conn = sqlite3.connect(str(DB_PATH))
            _bex_df = pd.read_sql(
                "SELECT symbol, as_of_date, circular_score, pump_score "
                "FROM broker_signals_v2",
                _bex_conn,
            )
            _bex_conn.close()
            for _, row in _bex_df.iterrows():
                broker_exit_lookup[(row["symbol"], str(row["as_of_date"])[:10])] = (
                    float(row["circular_score"] or 0),
                    float(row["pump_score"] or 0),
                )
            logger.info(f"Broker exit lookup: {len(broker_exit_lookup):,} entries")
        except Exception as e:
            logger.warning(f"Could not load broker exit data: {e}")

    conn.close()

    logger.info(f"Loaded {len(prices_df)} price records for {prices_df['symbol'].nunique()} symbols")
    symbol_cache = build_symbol_price_cache(prices_df)

    # Generate trading dates
    all_dates = sorted(prices_df["date"].unique())
    trading_dates = [d for d in all_dates if start <= d <= end]

    if not trading_dates:
        logger.error("No trading dates in range")
        return BacktestResult([], start, end, holding_days, initial_capital, [])

    logger.info(f"Testing on {len(trading_dates)} trading days")

    # Build close price lookup for fast NAV computation
    close_lookup = build_close_lookup(prices_df)
    open_lookup = build_price_lookup(prices_df, "open")
    prev_close_lookup = build_price_lookup(precompute_prev_close(prices_df), "prev_close")
    market_return_cache = build_market_return_cache(prices_df)

    # Portfolio state
    cash = initial_capital
    current_positions: Dict[str, Trade] = {}
    completed_trades: List[Trade] = []
    pending_entries: List[dict] = []  # Signals waiting for next-day execution
    daily_nav: List[Tuple[Any, float]] = []
    last_signal_date = None
    current_regime = "neutral"  # Tracks last-known regime for STEP 1 use

    for i, current_date in enumerate(trading_dates):
        current_date = pd.Timestamp(current_date)

        # ----- STEP 1: Execute pending entries at today's open -----
        for entry_info in pending_entries:
            symbol = entry_info["symbol"]
            if symbol in current_positions:
                continue
            if len(current_positions) >= max_positions:
                break

            # Get today's open price for this symbol
            open_price = open_lookup.get(symbol, {}).get(current_date)
            if open_price is None or not (open_price > 0):  # catches None, NaN, 0, negative
                continue

            # Apply circuit breaker to entry price
            prev_close = prev_close_lookup.get(symbol, {}).get(current_date)
            if prev_close is not None:
                open_price = apply_circuit_breaker(open_price, prev_close)

            # Position sizing: equal-weight
            per_position = initial_capital / max_positions
            # Don't exceed available cash
            available = min(per_position, cash * 0.95)  # Keep 5% cash buffer
            if available < 10000:  # Min NPR 10K position
                continue

            if not (open_price > 0):  # re-check after circuit breaker (may produce NaN)
                continue
            shares = int(available / open_price)
            if shares < 10:
                continue

            # Calculate buy fees
            buy_fees = NepseFees.total_fees(shares, open_price)
            total_cost = shares * open_price + buy_fees

            if total_cost > cash:
                shares = int((cash - buy_fees) / open_price)
                if shares < 10:
                    continue
                buy_fees = NepseFees.total_fees(shares, open_price)
                total_cost = shares * open_price + buy_fees

            # Sector concentration check (regime-dependent if regime_sector_limits provided)
            _eff_sector_limit = (
                regime_sector_limits.get(current_regime, sector_limit)
                if regime_sector_limits else sector_limit
            )
            if _eff_sector_limit < 1.0:
                symbol_sector = get_symbol_sector(symbol)
                if symbol_sector:
                    sector_value = sum(
                        t.shares * (close_lookup.get(t.symbol, {}).get(current_date, t.entry_price))
                        for t in current_positions.values()
                        if get_symbol_sector(t.symbol) == symbol_sector
                    )
                    nav_estimate = cash + sum(
                        t.shares * close_lookup.get(t.symbol, {}).get(current_date, t.entry_price)
                        for t in current_positions.values()
                    )
                    if nav_estimate > 0 and (sector_value + shares * open_price) / nav_estimate > _eff_sector_limit:
                        continue

            cash -= total_cost

            # Regime-adaptive hold: use shorter hold in neutral/bear if enabled
            if regime_adaptive_hold:
                _default_hold = {"bull": holding_days, "neutral": max(10, holding_days // 2), "bear": max(5, holding_days // 4)}
                _rhold = regime_hold_days if regime_hold_days else _default_hold
                trade_max_hold = _rhold.get(regime, holding_days)
            else:
                trade_max_hold = holding_days

            trade = Trade(
                symbol=symbol,
                signal_date=entry_info.get("signal_date"),
                entry_date=current_date,
                entry_price=open_price,
                shares=shares,
                position_value=shares * open_price,
                buy_fees=buy_fees,
                signal_type=entry_info.get("signal_type", ""),
                direction=1,
                max_price=open_price,
                entry_trading_idx=i,
                target_exit_date=entry_info.get("target_exit_date"),
                max_holding_days=trade_max_hold,
            )
            current_positions[symbol] = trade

        pending_entries = []

        # ----- STEP 2: Check exits -----
        to_close = []
        for symbol, trade in current_positions.items():
            # Use open price for intraday risk checks
            current_price = open_lookup.get(symbol, {}).get(current_date)
            if current_price is None:
                continue

            # Apply circuit breaker relative to previous close
            prev_close = prev_close_lookup.get(symbol, {}).get(current_date)
            if prev_close is not None:
                current_price = apply_circuit_breaker(current_price, prev_close)

            # Update max price for trailing stop
            if current_price > trade.max_price:
                trade.max_price = current_price

            trading_days_held = i - trade.entry_trading_idx
            exit_reason = None

            # Check hard stop loss
            if current_price < trade.entry_price * (1 - stop_loss_pct):
                exit_reason = "stop_loss"

            # Check trailing stop
            if exit_reason is None and use_trailing_stop and trade.max_price > trade.entry_price:
                trailing_stop_price = trade.max_price * (1 - trailing_stop_pct)
                if current_price < trailing_stop_price:
                    exit_reason = "trailing_stop"

            # Broker pump/circular exit: if the stock is showing wash-trade or
            # pump-and-dump patterns, exit before the distribution dump hits.
            # Only fires after holding >= 5 trading days (avoid day-1 noise).
            if exit_reason is None and broker_exit_lookup and trading_days_held >= 5:
                date_str = current_date.strftime("%Y-%m-%d")
                bsig = broker_exit_lookup.get((symbol, date_str))
                if bsig is not None:
                    circ, pump = bsig
                    if circ > 0.35 or pump > 0.10:
                        exit_reason = "broker_pump_exit"

            # Check profit target
            if exit_reason is None and profit_target_pct is not None:
                if current_price >= trade.entry_price * (1 + profit_target_pct):
                    exit_reason = "profit_target"

            # Event-driven exit: if trade has a target_exit_date, exit on that date
            if exit_reason is None and event_exit_mode and trade.target_exit_date is not None:
                if current_date >= trade.target_exit_date:
                    exit_reason = "event_exit"

            # Check holding period (in trading days) — use close price for planned exit
            # This fires independently of trailing stop state
            if exit_reason is None and trading_days_held >= holding_days:
                exit_reason = "holding_period"
                close_price = get_price_at_date(prices_df, symbol, current_date,
                                                look_forward_days=0, use_open=False)
                if close_price is not None:
                    if prev_close is not None:
                        close_price = apply_circuit_breaker(close_price, prev_close)
                    current_price = close_price

            if exit_reason:
                # Calculate sell fees
                sell_fees = NepseFees.total_fees(trade.shares, current_price, is_sell=True)
                proceeds = trade.shares * current_price - sell_fees
                cash += proceeds

                trade.exit_date = current_date
                trade.exit_price = current_price
                trade.sell_fees = sell_fees
                trade.exit_reason = exit_reason
                completed_trades.append(trade)
                to_close.append(symbol)

        for symbol in to_close:
            del current_positions[symbol]

        # ----- STEP 3: Record end-of-day NAV -----
        nav = cash
        for sym, trade in current_positions.items():
            # Use close price for MTM; fall back to entry price
            close_price = close_lookup.get(sym, {}).get(current_date)
            if close_price is None:
                close_price = trade.entry_price
            nav += trade.shares * close_price
        daily_nav.append((current_date, nav))

        # ----- STEP 4: Generate signals -----
        if last_signal_date is None or (current_date - last_signal_date).days >= rebalance_frequency:
            # Skip if near end (won't have exit data)
            remaining_trading_days = len(trading_dates) - i - 1
            if remaining_trading_days < holding_days:
                continue

            # Regime filter — compute regime for both filtering and weighting
            regime = (
                compute_market_regime(
                    prices_df,
                    current_date,
                    bear_threshold=bear_threshold,
                    market_return_cache=market_return_cache,
                )
                if use_regime_filter else "neutral"
            )
            current_regime = regime  # Persist for STEP 1 sector limit use
            if use_regime_filter and regime == "bear" and regime_max_positions is None:
                # Legacy behavior: skip all entries in bear markets
                last_signal_date = current_date
                continue

            # Graduated regime position limits (when regime_max_positions is set)
            regime_pos_limit = max_positions
            if regime_max_positions is not None and regime in regime_max_positions:
                regime_pos_limit = regime_max_positions[regime]

            # Liquidity pre-filter: remove bottom 30% by turnover
            # Only computed when new signals need it (avoids 2s overhead per signal day)
            liquid_symbols = None
            if any(st in signal_types for st in [
                "xsec_momentum", "accumulation",
                "52wk_high", "residual_momentum",
                "disposition", "pairs_trade",
                "satellite_hydro", "macro_remittance",
                "nlp_sentiment", "quarterly_fundamental",
                "value_bounce", "smart_money", "smart_money_pure",
            ]):
                liquid_symbols = compute_liquid_universe(prices_df, current_date)

            # Generate signals
            all_signals: List[AlphaSignal] = []

            if "momentum" in signal_types:
                all_signals.extend(generate_momentum_signals_at_date(prices_df, current_date, symbol_cache=symbol_cache))

            if "volume" in signal_types:
                all_signals.extend(generate_volume_breakout_signals_at_date(prices_df, current_date, symbol_cache=symbol_cache))

            if "mean_reversion" in signal_types:
                all_signals.extend(generate_mean_reversion_signals_at_date(prices_df, current_date, symbol_cache=symbol_cache))

            if "corp_action" in signal_types and corp_actions_df is not None:
                all_signals.extend(generate_corporate_action_signals_at_date(
                    prices_df, corp_actions_df, current_date,
                    short_term_mode=event_exit_mode,
                    trading_dates_list=trading_dates,
                ))

            if "settlement_pressure" in signal_types and corp_actions_df is not None:
                all_signals.extend(generate_settlement_pressure_signals_at_date(
                    prices_df, corp_actions_df, current_date,
                    trading_dates_list=trading_dates,
                ))

            if "low_vol" in signal_types:
                all_signals.extend(generate_low_volatility_signals_at_date(prices_df, current_date, symbol_cache=symbol_cache))

            if "quality" in signal_types:
                all_signals.extend(generate_quality_signals_at_date(prices_df, current_date, symbol_cache=symbol_cache))

            if "quarterly_fundamental" in signal_types and quarterly_fundamental_model is not None:
                all_signals.extend(generate_quarterly_fundamental_signals_at_date(
                    prices_df, current_date, quarterly_fundamental_model, liquid_symbols=liquid_symbols
                ))

            if "xsec_momentum" in signal_types:
                all_signals.extend(generate_xsec_momentum_signals_at_date(
                    prices_df, current_date, liquid_symbols=liquid_symbols
                ))

            if "accumulation" in signal_types:
                all_signals.extend(generate_accumulation_signals_at_date(
                    prices_df, current_date, liquid_symbols=liquid_symbols
                ))

            if "52wk_high" in signal_types:
                all_signals.extend(generate_52wk_high_signals_at_date(
                    prices_df, current_date, liquid_symbols=liquid_symbols
                ))

            if "value_bounce" in signal_types:
                all_signals.extend(generate_value_bounce_signals_at_date(
                    prices_df, current_date, liquid_symbols=liquid_symbols
                ))

            if "residual_momentum" in signal_types:
                all_signals.extend(generate_residual_momentum_signals_at_date(
                    prices_df, current_date, liquid_symbols=liquid_symbols
                ))

            if "disposition" in signal_types:
                all_signals.extend(generate_cgo_signals_at_date(
                    prices_df, current_date, liquid_symbols=liquid_symbols
                ))

            # lead_lag signal: not included in public release

            if "pairs_trade" in signal_types:
                all_signals.extend(generate_pairs_signals_at_date(
                    prices_df, current_date, liquid_symbols=liquid_symbols
                ))

            # Sprint 4-5: Alternative data, earnings, sentiment signals
            if "satellite_hydro" in signal_types:
                all_signals.extend(generate_hydro_rainfall_signals_at_date(
                    prices_df, current_date, liquid_symbols=liquid_symbols
                ))

            if "macro_remittance" in signal_types:
                all_signals.extend(generate_remittance_signals_at_date(
                    prices_df, current_date, liquid_symbols=liquid_symbols
                ))

            # earnings_drift signal: not included in public release

            if "nlp_sentiment" in signal_types:
                all_signals.extend(generate_sentiment_signals_at_date(
                    prices_df, current_date, liquid_symbols=liquid_symbols
                ))

            # informed_trading / broker_flow signal: not included in public release

            # Signal plugin: smart_money (not included in public release)

            # Amihud illiquidity tilt: boost moderately illiquid stocks
            if "amihud_tilt" in signal_types and all_signals:
                all_signals = apply_amihud_tilt(all_signals, prices_df, current_date)

            # Regime-adaptive signal weighting
            # Always applied when use_regime_filter=True (not gated by signal types)
            _REGIME_WEIGHTS = {
                # SignalType: {regime: multiplier}
                SignalType.LIQUIDITY:          {"bull": 1.0, "neutral": 1.0, "bear": 0.5},
                SignalType.FUNDAMENTAL:        {"bull": 0.9, "neutral": 1.2, "bear": 1.0},
                SignalType.QUARTERLY_FUNDAMENTAL: {"bull": 0.9, "neutral": 1.2, "bear": 1.0},
                SignalType.XSEC_MOMENTUM:      {"bull": 1.1, "neutral": 0.8, "bear": 0.3},
                SignalType.MEAN_REVERSION:     {"bull": 0.5, "neutral": 1.0, "bear": 1.5},
                SignalType.MOMENTUM:           {"bull": 1.1, "neutral": 0.7, "bear": 0.3},
                SignalType.ACCUMULATION:       {"bull": 1.0, "neutral": 1.0, "bear": 0.5},
                SignalType.ANCHORING_52WK:     {"bull": 1.2, "neutral": 1.0, "bear": 0.5},
                SignalType.RESIDUAL_MOMENTUM:  {"bull": 1.0, "neutral": 1.1, "bear": 0.7},
                SignalType.DISPOSITION:         {"bull": 1.0, "neutral": 1.1, "bear": 0.6},
                SignalType.LEAD_LAG:           {"bull": 1.1, "neutral": 0.9, "bear": 0.4},
                SignalType.PAIRS_TRADE:        {"bull": 0.8, "neutral": 1.2, "bear": 1.0},
                SignalType.INFORMED_TRADING:   {"bull": 1.0, "neutral": 1.0, "bear": 0.8},
                # Sprint 4-5: Alt data, earnings, sentiment
                SignalType.SATELLITE_HYDRO:    {"bull": 1.0, "neutral": 1.0, "bear": 0.6},
                SignalType.MACRO_REMITTANCE:   {"bull": 1.2, "neutral": 1.0, "bear": 0.7},
                SignalType.EARNINGS_DRIFT:     {"bull": 1.0, "neutral": 1.1, "bear": 0.9},
                SignalType.NLP_SENTIMENT:      {"bull": 1.1, "neutral": 1.0, "bear": 0.5},
                SignalType.SETTLEMENT_PRESSURE: {"bull": 0.8, "neutral": 1.0, "bear": 1.3},
                # Tier 6: Contrarian / Deep Value
                # bull=0.5: value bounce historically underperforms in bull (banks/ins at lows lose more)
                # neutral=1.2: boost in neutral where sector rotation is common
                # bear=1.8: dominant signal in bear for contrarian plays
                SignalType.VALUE_BOUNCE:       {"bull": 0.5, "neutral": 1.2, "bear": 1.8},
            }
            if use_regime_filter and regime != "neutral":
                for sig in all_signals:
                    wt = _REGIME_WEIGHTS.get(sig.signal_type, {}).get(regime, 1.0)
                    sig.strength *= wt
            elif use_regime_filter and regime == "neutral":
                # In neutral regime: only apply weights for explicitly contrarian signals
                # (xsec_momentum/momentum neutral weights hurt performance — that invariant holds)
                # But value_bounce and mean_reversion benefit from their neutral boost.
                _NEUTRAL_CONTRARIAN_SIGNALS = {
                    SignalType.VALUE_BOUNCE,
                    SignalType.MEAN_REVERSION,
                    SignalType.SETTLEMENT_PRESSURE,
                }
                for sig in all_signals:
                    if sig.signal_type in _NEUTRAL_CONTRARIAN_SIGNALS:
                        wt = _REGIME_WEIGHTS.get(sig.signal_type, {}).get("neutral", 1.0)
                        sig.strength *= wt

            # Circuit-breaker reversal filter: skip stocks that hit +10% in last 3 days
            # CB filter fires on +9.5% moves — extremely rare for quality/low_vol stocks
            all_signals = [
                sig for sig in all_signals
                if not is_circuit_breaker_hit(prices_df, sig.symbol, current_date)
            ]

            # Queue top signals for next-day execution using the shared ranker.
            effective_max = regime_pos_limit if regime_max_positions is not None else max_positions
            slots_available = effective_max - len(current_positions)
            queued = 0

            nav_estimate = cash + sum(
                trade.shares * close_lookup.get(trade.symbol, {}).get(current_date, trade.entry_price)
                for trade in current_positions.values()
            )
            sector_exposure: Dict[str, float] = {}
            if nav_estimate > 0:
                for trade in current_positions.values():
                    sector = str(get_symbol_sector(trade.symbol) or "").strip().upper()
                    if not sector:
                        continue
                    mark = close_lookup.get(trade.symbol, {}).get(current_date, trade.entry_price)
                    sector_exposure[sector] = sector_exposure.get(sector, 0.0) + (trade.shares * mark) / nav_estimate

            ranked_signals = rank_signal_candidates(
                [
                    {
                        "symbol": signal.symbol,
                        "signal_type": signal.signal_type.value,
                        "strength": signal.strength,
                        "confidence": signal.confidence,
                        "reasoning": signal.reasoning,
                        "target_exit_date": getattr(signal, "target_exit_date", None),
                    }
                    for signal in all_signals
                ],
                held_symbols=current_positions.keys(),
                sector_exposure=sector_exposure,
                sector_lookup=get_symbol_sector,
                event_context=load_event_adjustment_context(current_date),
            )

            for signal in ranked_signals:
                if queued >= slots_available:
                    break
                if signal["symbol"] in current_positions:
                    continue
                if any(p["symbol"] == signal["symbol"] for p in pending_entries):
                    continue

                pending_entries.append({
                    "symbol": signal["symbol"],
                    "signal_type": signal["signal_type"],
                    "signal_date": current_date,
                    "strength": signal["strength"],
                    "confidence": signal["confidence"],
                    "target_exit_date": signal.get("target_exit_date"),
                    "rank_score": signal["score"],
                    "raw_score": signal["raw_score"],
                    "event_adjustment": signal["event_adjustment"],
                })
                queued += 1

            last_signal_date = current_date

        # Progress logging
        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(trading_dates)} days, "
                        f"{len(completed_trades)} trades, "
                        f"NAV: NPR {nav:,.0f}")

    # Close remaining positions at last available price
    for symbol, trade in current_positions.items():
        exit_fill = get_price_on_or_before_date(prices_df, symbol, end)
        if exit_fill:
            exit_date, exit_price = exit_fill
            sell_fees = NepseFees.total_fees(trade.shares, exit_price, is_sell=True)
            trade.exit_date = exit_date
            trade.exit_price = exit_price
            trade.sell_fees = sell_fees
            trade.exit_reason = "end_of_backtest"
            cash += trade.shares * exit_price - sell_fees
            completed_trades.append(trade)

    return BacktestResult(
        trades=completed_trades,
        start_date=start,
        end_date=end,
        holding_period=holding_days,
        initial_capital=initial_capital,
        daily_nav=daily_nav,
    )


# =============================================================================
# PARAMETER SWEEP
# =============================================================================

def _sweep_worker(args: Tuple) -> dict:
    """Top-level worker for multiprocessing — runs a single sweep config."""
    start_date, end_date, holding, signals, use_regime = args
    period_name = f"{start_date[:4]}-{end_date[2:4]}"
    try:
        result = run_backtest(
            start_date=start_date,
            end_date=end_date,
            holding_days=holding,
            signal_types=signals,
            max_positions=5,
            use_trailing_stop=True,
            use_regime_filter=use_regime,
        )
        return {
            "period": period_name,
            "holding_days": holding,
            "signals": "+".join(signals),
            "regime_filter": use_regime,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "total_return": result.total_return,
            "ann_return": result.annualized_return,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "max_dd": result.max_drawdown,
            "calmar": result.calmar_ratio,
            "profit_factor": result.profit_factor if result.profit_factor != float('inf') else 99.9,
        }
    except Exception as e:
        logger.error(f"Sweep worker failed: period={period_name} hold={holding} "
                     f"signals={'+'.join(signals)} regime={use_regime}: {e}")
        return None


def parameter_sweep(test_periods: List[Tuple[str, str]] = None, n_workers: int = None) -> pd.DataFrame:
    """Run parameter sweep across strategies, holding periods, and regimes.

    Uses multiprocessing to parallelise across all available CPU cores.
    """
    if test_periods is None:
        test_periods = [
            ("2020-01-01", "2021-12-31"),  # Bull market
            ("2022-01-01", "2023-12-31"),  # Bear/recovery
            ("2024-01-01", "2025-12-31"),  # Recent
        ]

    holding_periods = [20, 40, 60]  # Trading days (≈ 4, 8, 12 NEPSE weeks)
    signal_combos = [
        ["volume"],
        ["quality"],
        ["low_vol"],
        ["mean_reversion"],
        ["xsec_momentum"],
        ["accumulation"],
        ["volume", "quality"],
        ["quality", "low_vol"],
        ["volume", "quality", "low_vol"],
        ["volume", "quality", "low_vol", "xsec_momentum"],
        ["volume", "quality", "low_vol", "xsec_momentum", "accumulation"],
    ]

    # Build all (198) configs upfront
    configs = [
        (start_date, end_date, holding, signals, use_regime)
        for start_date, end_date in test_periods
        for holding in holding_periods
        for signals in signal_combos
        for use_regime in [True, False]
    ]

    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), len(configs))

    logger.info(f"Running {len(configs)} sweep configs across {n_workers} parallel workers...")

    with multiprocessing.Pool(processes=n_workers) as pool:
        raw = pool.map(_sweep_worker, configs)

    results = [r for r in raw if r is not None]
    df = pd.DataFrame(results)

    if len(df) > 0:
        # Average metrics across periods for each config
        group_cols = ["signals", "holding_days", "regime_filter"]
        avg = df.groupby(group_cols).agg({
            "sharpe": "mean",
            "sortino": "mean",
            "max_dd": "min",  # Worst drawdown
            "total_return": "mean",
            "win_rate": "mean",
        }).reset_index()
        avg.columns = [*group_cols, "avg_sharpe", "avg_sortino", "worst_dd",
                        "avg_return", "avg_win_rate"]
        df = df.merge(avg, on=group_cols)
        df = df.sort_values(["avg_sharpe", "period"], ascending=[False, True])

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NEPSE Portfolio Backtest")
    parser.add_argument("--start", type=str, default="2023-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--holding-days", type=int, default=40,
                        help="Holding period in trading days (40 ≈ 8 NEPSE weeks)")
    parser.add_argument("--max-positions", type=int, default=5,
                        help="Max concurrent positions")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="Initial capital in NPR")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep")
    parser.add_argument("--signals", type=str, default="volume,quality",
                        help="Signal types (comma-separated: volume,quality,low_vol,mean_reversion,"
                             "momentum,corp_action,xsec_momentum,accumulation,"
                             "52wk_high,residual_momentum,amihud_tilt,satellite_hydro,"
                             "quarterly_fundamental,nlp_sentiment,disposition,pairs_trade)")
    parser.add_argument("--no-regime-filter", action="store_true",
                        help="Disable market regime filter")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for sweep (default: all CPUs)")

    args = parser.parse_args()

    if not args.sweep:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
        if start >= end:
            parser.error("--start must be before --end")
        if not (1 <= args.holding_days <= 252):
            parser.error("--holding-days must be between 1 and 252")
        if not (1 <= args.max_positions <= 50):
            parser.error("--max-positions must be between 1 and 50")
        if args.capital <= 0:
            parser.error("--capital must be positive")

    if args.sweep:
        logger.info("Running parameter sweep...")
        results = parameter_sweep(n_workers=args.workers)
        print("\nPARAMETER SWEEP RESULTS:")
        print(results.to_string(index=False))
        results.to_csv("backtest_sweep_results.csv", index=False)
        print("\nSaved to backtest_sweep_results.csv")
    else:
        signal_types = args.signals.split(",")
        result = run_backtest(
            start_date=args.start,
            end_date=args.end,
            holding_days=args.holding_days,
            max_positions=args.max_positions,
            signal_types=signal_types,
            initial_capital=args.capital,
            use_regime_filter=not args.no_regime_filter,
        )
        print(result.summary())

        # Save trades
        if result.trades:
            trades_df = pd.DataFrame([{
                "symbol": t.symbol,
                "signal_date": t.signal_date,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "shares": t.shares,
                "position_value": t.position_value,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "signal_type": t.signal_type,
                "exit_reason": t.exit_reason,
                "gross_return": t.gross_return,
                "net_return": t.net_return,
                "net_pnl": t.net_pnl,
                "buy_fees": t.buy_fees,
                "sell_fees": t.sell_fees,
                "holding_days": t.holding_days,
            } for t in result.trades if t.net_return is not None])
            trades_df.to_csv("backtest_trades.csv", index=False)
            print(f"\nSaved {len(trades_df)} trades to backtest_trades.csv")

        # Save daily NAV
        if result.daily_nav:
            nav_df = pd.DataFrame(result.daily_nav, columns=["date", "nav"])
            nav_df.to_csv("backtest_nav.csv", index=False)
            print(f"Saved {len(nav_df)} daily NAV records to backtest_nav.csv")


if __name__ == "__main__":
    main()
