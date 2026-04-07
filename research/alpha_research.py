#!/usr/bin/env python3
"""
NEPSE Alpha Research Framework
Tournament-Grade Signal Discovery System

Goal: Find robust, leak-free alpha that works out-of-sample
Method: Test multiple factors, combine best performers, validate rigorously

Author: Alpha Research System
"""

import logging
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

from backend.quant_pro.database import get_db_path

# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all price data and pivot to wide format."""
    conn = sqlite3.connect(str(get_db_path()))

    query = """
        SELECT symbol, date, open, high, low, close, volume
        FROM stock_prices
        WHERE volume > 0
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn)
    df["date"] = pd.to_datetime(df["date"])

    # Load corporate actions
    try:
        corp_query = """
            SELECT symbol, bookclose_date, cash_dividend_pct, bonus_share_pct
            FROM corporate_actions
            WHERE bookclose_date IS NOT NULL
        """
        corp_df = pd.read_sql_query(corp_query, conn)
        corp_df["bookclose_date"] = pd.to_datetime(corp_df["bookclose_date"])
    except (sqlite3.OperationalError, pd.errors.DatabaseError) as e:
        logger.warning("Could not load corporate actions: %s", e)
        corp_df = pd.DataFrame()

    conn.close()

    # Pivot to wide format
    prices = df.pivot(index="date", columns="symbol", values="close").sort_index()
    volumes = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    opens = df.pivot(index="date", columns="symbol", values="open").sort_index()
    highs = df.pivot(index="date", columns="symbol", values="high").sort_index()
    lows = df.pivot(index="date", columns="symbol", values="low").sort_index()

    return prices, volumes, corp_df, opens, highs, lows


# =============================================================================
# ALPHA FACTORS - Each returns a DataFrame of signals (positive = long)
# =============================================================================

def factor_volume_breakout(prices: pd.DataFrame, volumes: pd.DataFrame,
                           lookback: int = 60, spike_mult: float = 2.5) -> pd.DataFrame:
    """
    Volume Breakout: High volume precedes price moves.
    Signal when 5-day avg volume > lookback avg * spike_mult
    """
    vol_avg = volumes.rolling(lookback).mean()
    vol_5d = volumes.rolling(5).mean()

    signal = (vol_5d / vol_avg) - 1  # Excess volume ratio
    signal = signal.clip(lower=0)  # Only positive signals

    # Confirm with price momentum (volume + price going same direction)
    price_mom = prices.pct_change(5)
    signal = signal * (price_mom > 0).astype(float)  # Zero out if price falling

    return signal


def factor_price_momentum(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Price Momentum: Past winners continue winning.
    Classic momentum factor.
    """
    returns = prices.pct_change(lookback)

    # Normalize to cross-sectional z-score
    mean = returns.mean(axis=1)
    std = returns.std(axis=1)
    z_score = returns.sub(mean, axis=0).div(std, axis=0)

    return z_score.clip(lower=0)  # Long only


def factor_mean_reversion_rsi(prices: pd.DataFrame, period: int = 14,
                               oversold: float = 30) -> pd.DataFrame:
    """
    Mean Reversion: Buy oversold stocks showing recovery.
    RSI < oversold AND price starting to recover.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Signal when oversold AND recovering (today > yesterday)
    recovering = prices > prices.shift(1)
    signal = ((rsi < oversold) & recovering).astype(float)

    # Strength based on how oversold
    strength = (oversold - rsi) / oversold
    signal = signal * strength.clip(lower=0)

    return signal


def factor_dividend_anticipation(prices: pd.DataFrame, corp_df: pd.DataFrame,
                                  days_before: int = 21) -> pd.DataFrame:
    """
    Dividend Anticipation: Buy before book closure date.
    NEPSE-specific: Stocks rally before dividend/bonus announcements.
    """
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    if corp_df.empty:
        return signal

    for _, row in corp_df.iterrows():
        symbol = row["symbol"]
        if symbol not in prices.columns:
            continue

        bookclose = row["bookclose_date"]
        cash_div = row.get("cash_dividend_pct") or 0
        bonus = row.get("bonus_share_pct") or 0

        # Signal strength based on dividend size
        strength = min((cash_div + bonus * 1.5) / 30, 1.0)  # Cap at 1.0

        if strength < 0.1:
            continue

        # Create signal window: days_before to 3 days before bookclose
        start_date = bookclose - timedelta(days=days_before)
        end_date = bookclose - timedelta(days=3)

        mask = (signal.index >= start_date) & (signal.index <= end_date)
        signal.loc[mask, symbol] = strength

    return signal


def factor_sector_momentum_lag(prices: pd.DataFrame,
                                sector_map: Dict[str, str]) -> pd.DataFrame:
    """
    Sector Momentum Lag: Laggards in rising sectors catch up.
    When sector is rising, buy stocks that haven't moved yet.
    """
    # Calculate sector returns
    sector_returns = {}
    for symbol in prices.columns:
        sector = sector_map.get(symbol, "Others")
        if sector not in sector_returns:
            sector_returns[sector] = []
        sector_returns[sector].append(symbol)

    # Calculate sector average 10-day return
    sector_avg_ret = pd.DataFrame(index=prices.index)
    for sector, symbols in sector_returns.items():
        valid_symbols = [s for s in symbols if s in prices.columns]
        if valid_symbols:
            sector_avg_ret[sector] = prices[valid_symbols].pct_change(10).mean(axis=1)

    # For each stock, signal = sector_return - stock_return (laggard premium)
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    stock_returns = prices.pct_change(10)

    for symbol in prices.columns:
        sector = sector_map.get(symbol, "Others")
        if sector in sector_avg_ret.columns:
            # Signal when sector is up but stock is lagging
            sector_up = sector_avg_ret[sector] > 0.02  # Sector up >2%
            stock_lag = stock_returns[symbol] < sector_avg_ret[sector] - 0.01

            lag_amount = sector_avg_ret[sector] - stock_returns[symbol]
            signal[symbol] = (sector_up & stock_lag).astype(float) * lag_amount.clip(lower=0)

    return signal


def factor_volatility_breakout(prices: pd.DataFrame, highs: pd.DataFrame,
                                lows: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Volatility Breakout: Buy on breakout from consolidation.
    When price breaks above recent high after low volatility period.
    """
    # Calculate ATR (Average True Range)
    tr1 = highs - lows
    tr2 = abs(highs - prices.shift(1))
    tr3 = abs(lows - prices.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, level=0)

    # Handle the case where tr might be a Series instead of DataFrame
    if isinstance(tr, pd.Series):
        tr = pd.DataFrame(tr)

    atr = tr.rolling(lookback).mean()

    # Volatility contraction: current ATR < 70% of 60-day ATR
    atr_60 = tr.rolling(60).mean()
    vol_contraction = atr < (atr_60 * 0.7)

    # Breakout: price > 20-day high
    high_20 = prices.rolling(lookback).max()
    breakout = prices > high_20.shift(1)

    # Signal: breakout after volatility contraction
    signal = (vol_contraction & breakout).astype(float)

    return signal


def factor_relative_strength(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Relative Strength: Buy stocks outperforming the market.
    Cross-sectional momentum - buy top performers.
    """
    returns = prices.pct_change(lookback)

    # Rank stocks each day (higher = stronger)
    ranks = returns.rank(axis=1, pct=True)

    # Signal for top 30% performers
    signal = (ranks > 0.7).astype(float) * (ranks - 0.7) / 0.3

    return signal


def factor_liquidity_improvement(prices: pd.DataFrame, volumes: pd.DataFrame,
                                  lookback: int = 60) -> pd.DataFrame:
    """
    Liquidity Improvement: Buy when trading activity increases.
    Institutional accumulation shows up as gradual volume increase.
    """
    # 20-day avg volume vs 60-day avg volume
    vol_20 = volumes.rolling(20).mean()
    vol_60 = volumes.rolling(lookback).mean()

    # Improvement ratio
    improvement = vol_20 / vol_60 - 1

    # Signal when volume improving AND price stable/rising
    price_stable = prices.pct_change(20) > -0.05
    signal = improvement.clip(lower=0) * price_stable.astype(float)

    return signal


def factor_earnings_quality(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Quality Proxy: Low volatility + steady gains + consistent volume.
    Quality stocks tend to outperform in uncertain markets.
    """
    # Calculate metrics over 60 days
    returns = prices.pct_change()

    # 1. Low volatility
    vol = returns.rolling(60).std() * np.sqrt(252)
    vol_score = 1 - vol.rank(axis=1, pct=True)  # Lower vol = higher score

    # 2. Positive returns
    ret_60 = prices.pct_change(60)
    ret_score = (ret_60 > 0).astype(float) * ret_60.clip(upper=0.3)

    # 3. Consistency (% positive days)
    pos_days = (returns > 0).rolling(60).mean()
    cons_score = (pos_days > 0.5).astype(float) * (pos_days - 0.5)

    # Combined quality score
    signal = (vol_score * 0.4 + ret_score * 0.4 + cons_score * 0.2).clip(lower=0)

    return signal


# =============================================================================
# BACKTEST ENGINE (Leak-free)
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: str
    end_date: str
    holding_days: int = 21
    max_positions: int = 5
    position_size: float = 0.15  # 15% per position
    transaction_cost: float = 0.012  # 1.2% round trip
    min_signal_strength: float = 0.1


def backtest_factor(signal: pd.DataFrame, prices: pd.DataFrame, opens: pd.DataFrame,
                    config: BacktestConfig) -> Dict:
    """
    Backtest a factor signal with proper leak-free methodology.

    Rules:
    - Signal generated at close of day T
    - Entry at open of day T+1
    - Exit at close of day T+holding_days
    """
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)

    # Filter to date range
    mask = (signal.index >= start) & (signal.index <= end)
    signal = signal.loc[mask]
    prices = prices.loc[mask]
    opens = opens.loc[mask]

    trades = []
    positions = {}  # symbol -> (entry_date, entry_price)

    trading_days = signal.index.tolist()

    for i, date in enumerate(trading_days[:-config.holding_days-1]):
        # Check for exits
        to_exit = []
        for symbol, (entry_date, entry_price) in positions.items():
            days_held = (date - entry_date).days
            if days_held >= config.holding_days:
                # Exit at today's close
                exit_price = prices.loc[date, symbol] if symbol in prices.columns else None
                if exit_price and not pd.isna(exit_price):
                    gross_ret = (exit_price / entry_price - 1)
                    net_ret = gross_ret - config.transaction_cost
                    trades.append({
                        "symbol": symbol,
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return": gross_ret,
                        "net_return": net_ret,
                    })
                to_exit.append(symbol)

        for symbol in to_exit:
            del positions[symbol]

        # Check for entries (if slots available)
        slots_available = config.max_positions - len(positions)
        if slots_available <= 0:
            continue

        # Get today's signals
        day_signals = signal.loc[date].dropna()
        day_signals = day_signals[day_signals > config.min_signal_strength]
        day_signals = day_signals.sort_values(ascending=False)

        # Entry at NEXT day's open
        next_day_idx = i + 1
        if next_day_idx >= len(trading_days):
            continue
        next_day = trading_days[next_day_idx]

        entries = 0
        for symbol in day_signals.index:
            if entries >= slots_available:
                break
            if symbol in positions:
                continue
            if symbol not in opens.columns:
                continue

            entry_price = opens.loc[next_day, symbol]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            positions[symbol] = (next_day, entry_price)
            entries += 1

    # Close remaining positions at end
    if positions:
        last_date = trading_days[-1]
        for symbol, (entry_date, entry_price) in positions.items():
            if symbol in prices.columns:
                exit_price = prices.loc[last_date, symbol]
                if not pd.isna(exit_price):
                    gross_ret = (exit_price / entry_price - 1)
                    net_ret = gross_ret - config.transaction_cost
                    trades.append({
                        "symbol": symbol,
                        "entry_date": entry_date,
                        "exit_date": last_date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return": gross_ret,
                        "net_return": net_ret,
                    })

    # Calculate metrics
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "total_return": 0,
            "sharpe": 0,
            "max_drawdown": 0,
        }

    returns = [t["net_return"] for t in trades]
    wins = [r for r in returns if r > 0]

    sharpe = 0
    if len(returns) > 1 and np.std(returns) > 0:
        trades_per_year = 252 / config.holding_days
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(trades_per_year)

    # Max drawdown
    cumulative = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0

    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0,
        "avg_return": np.mean(returns),
        "total_return": np.sum(returns),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": trades,
    }


# =============================================================================
# FACTOR SCREENING
# =============================================================================

def screen_all_factors(prices: pd.DataFrame, volumes: pd.DataFrame,
                       corp_df: pd.DataFrame, opens: pd.DataFrame,
                       highs: pd.DataFrame, lows: pd.DataFrame,
                       sector_map: Dict[str, str]) -> pd.DataFrame:
    """Screen all factors on in-sample and out-of-sample periods."""

    # Define test periods
    periods = {
        "IS_2022": ("2022-01-01", "2022-12-31"),
        "IS_2023": ("2023-01-01", "2023-12-31"),
        "OOS_2024": ("2024-01-01", "2024-12-31"),
        "OOS_2025": ("2025-01-01", "2025-12-31"),
    }

    # Define factors to test
    factors = {
        "volume_breakout": lambda: factor_volume_breakout(prices, volumes),
        "price_momentum_20": lambda: factor_price_momentum(prices, 20),
        "price_momentum_60": lambda: factor_price_momentum(prices, 60),
        "mean_reversion_rsi": lambda: factor_mean_reversion_rsi(prices),
        "dividend_anticipation": lambda: factor_dividend_anticipation(prices, corp_df),
        "sector_lag": lambda: factor_sector_momentum_lag(prices, sector_map),
        "volatility_breakout": lambda: factor_volatility_breakout(prices, highs, lows),
        "relative_strength": lambda: factor_relative_strength(prices),
        "liquidity_improvement": lambda: factor_liquidity_improvement(prices, volumes),
        "quality": lambda: factor_earnings_quality(prices, volumes),
    }

    results = []

    for factor_name, factor_fn in factors.items():
        print(f"Testing {factor_name}...")
        try:
            signal = factor_fn()
        except Exception as e:
            print(f"  Error: {e}")
            continue

        for period_name, (start, end) in periods.items():
            config = BacktestConfig(start_date=start, end_date=end)

            try:
                result = backtest_factor(signal, prices, opens, config)
                results.append({
                    "factor": factor_name,
                    "period": period_name,
                    "trades": result["total_trades"],
                    "win_rate": result["win_rate"],
                    "avg_return": result["avg_return"],
                    "sharpe": result["sharpe"],
                    "max_dd": result["max_drawdown"],
                })
            except Exception as e:
                print(f"  {period_name} error: {e}")

    return pd.DataFrame(results)


def find_best_factors(results_df: pd.DataFrame, min_trades: int = 20) -> List[str]:
    """Find factors that work both in-sample and out-of-sample."""

    # Filter to factors with enough trades
    results_df = results_df[results_df["trades"] >= min_trades]

    # Calculate average OOS Sharpe for each factor
    oos_results = results_df[results_df["period"].str.startswith("OOS")]
    oos_sharpe = oos_results.groupby("factor")["sharpe"].mean()

    # Calculate IS Sharpe
    is_results = results_df[results_df["period"].str.startswith("IS")]
    is_sharpe = is_results.groupby("factor")["sharpe"].mean()

    # Find factors with positive OOS Sharpe and not too much degradation
    good_factors = []
    for factor in oos_sharpe.index:
        if factor not in is_sharpe.index:
            continue
        oos = oos_sharpe[factor]
        is_ = is_sharpe[factor]

        # Criteria: OOS Sharpe > 0.3 AND degradation < 50%
        if oos > 0.3 and (is_ <= 0 or (is_ - oos) / abs(is_) < 0.5):
            good_factors.append((factor, oos, is_))

    good_factors.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in good_factors]


# =============================================================================
# COMBINED SIGNAL
# =============================================================================

def create_combined_signal(prices: pd.DataFrame, volumes: pd.DataFrame,
                           corp_df: pd.DataFrame, opens: pd.DataFrame,
                           highs: pd.DataFrame, lows: pd.DataFrame,
                           sector_map: Dict[str, str],
                           factor_weights: Dict[str, float]) -> pd.DataFrame:
    """Create a combined signal from multiple factors."""

    factor_fns = {
        "volume_breakout": lambda: factor_volume_breakout(prices, volumes),
        "price_momentum_20": lambda: factor_price_momentum(prices, 20),
        "price_momentum_60": lambda: factor_price_momentum(prices, 60),
        "mean_reversion_rsi": lambda: factor_mean_reversion_rsi(prices),
        "dividend_anticipation": lambda: factor_dividend_anticipation(prices, corp_df),
        "sector_lag": lambda: factor_sector_momentum_lag(prices, sector_map),
        "volatility_breakout": lambda: factor_volatility_breakout(prices, highs, lows),
        "relative_strength": lambda: factor_relative_strength(prices),
        "liquidity_improvement": lambda: factor_liquidity_improvement(prices, volumes),
        "quality": lambda: factor_earnings_quality(prices, volumes),
    }

    combined = None
    total_weight = 0

    for factor_name, weight in factor_weights.items():
        if factor_name not in factor_fns:
            continue

        try:
            signal = factor_fns[factor_name]()
            # Normalize signal to 0-1 range
            signal = signal.clip(lower=0)
            signal_max = signal.max().max()
            if signal_max > 0:
                signal = signal / signal_max

            if combined is None:
                combined = signal * weight
            else:
                combined = combined.add(signal * weight, fill_value=0)
            total_weight += weight
        except Exception as e:
            print(f"Warning: {factor_name} failed: {e}")

    if combined is not None and total_weight > 0:
        combined = combined / total_weight

    return combined


# =============================================================================
# MAIN RESEARCH FLOW
# =============================================================================

def main():
    print("="*70)
    print("NEPSE ALPHA RESEARCH - FACTOR SCREENING")
    print("="*70)
    print()

    # Load data
    print("Loading data...")
    prices, volumes, corp_df, opens, highs, lows = load_all_data()
    print(f"Loaded {len(prices.columns)} symbols, {len(prices)} days")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")
    print()

    # Load sector mapping
    from backend.quant_pro.sectors import SECTOR_GROUPS
    sector_map = {}
    for sector, symbols in SECTOR_GROUPS.items():
        for symbol in symbols:
            sector_map[symbol] = sector

    # Screen all factors
    print("Screening factors across time periods...")
    print("-"*70)
    results = screen_all_factors(prices, volumes, corp_df, opens, highs, lows, sector_map)

    # Display results
    print()
    print("="*70)
    print("FACTOR PERFORMANCE SUMMARY")
    print("="*70)

    # Pivot for easy viewing
    pivot = results.pivot_table(
        index="factor",
        columns="period",
        values=["sharpe", "avg_return", "trades"],
        aggfunc="first"
    )

    print("\nSHARPE RATIOS BY PERIOD:")
    print(pivot["sharpe"].round(2).to_string())

    print("\nAVERAGE RETURNS BY PERIOD:")
    print((pivot["avg_return"] * 100).round(2).astype(str) + "%")

    # Find best factors
    print()
    print("="*70)
    print("BEST FACTORS (OOS Sharpe > 0.3, low degradation)")
    print("="*70)

    best = find_best_factors(results)
    if best:
        for i, f in enumerate(best, 1):
            oos_data = results[(results["factor"] == f) & (results["period"].str.startswith("OOS"))]
            avg_oos_sharpe = oos_data["sharpe"].mean()
            avg_oos_ret = oos_data["avg_return"].mean()
            print(f"{i}. {f}: OOS Sharpe={avg_oos_sharpe:.2f}, OOS Return={avg_oos_ret:.2%}")
    else:
        print("No factors passed the robustness criteria!")
        print("Trying with relaxed criteria...")

        # Relax criteria
        oos_results = results[results["period"].str.startswith("OOS")]
        oos_avg = oos_results.groupby("factor").agg({
            "sharpe": "mean",
            "avg_return": "mean",
            "trades": "sum"
        }).sort_values("sharpe", ascending=False)

        print("\nAll factors by OOS Sharpe:")
        print(oos_avg.round(3).to_string())

        # Take top 3 by OOS performance
        best = oos_avg.head(3).index.tolist()

    # Test combined signal
    if best:
        print()
        print("="*70)
        print("TESTING COMBINED SIGNAL")
        print("="*70)

        # Equal weight the best factors
        weights = {f: 1.0 for f in best[:3]}
        print(f"Combining: {list(weights.keys())}")

        combined = create_combined_signal(
            prices, volumes, corp_df, opens, highs, lows, sector_map, weights
        )

        # Test on each period
        for period_name, (start, end) in [
            ("IS_2023", ("2023-01-01", "2023-12-31")),
            ("OOS_2024", ("2024-01-01", "2024-12-31")),
            ("OOS_2025", ("2025-01-01", "2025-12-31")),
        ]:
            config = BacktestConfig(start_date=start, end_date=end)
            result = backtest_factor(combined, prices, opens, config)
            print(f"\n{period_name}:")
            print(f"  Trades: {result['total_trades']}")
            print(f"  Win Rate: {result['win_rate']:.1%}")
            print(f"  Avg Return: {result['avg_return']:.2%}")
            print(f"  Sharpe: {result['sharpe']:.2f}")
            print(f"  Max DD: {result['max_drawdown']:.2%}")

    # Save results
    results.to_csv("factor_screening_results.csv", index=False)
    print()
    print("Results saved to factor_screening_results.csv")

    return results, best


if __name__ == "__main__":
    results, best_factors = main()
