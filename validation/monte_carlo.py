"""
Monte Carlo trade resampling and bootstrap confidence intervals.

Implements:
- Trade-level resampling (shuffle trade sequence with replacement)
- Block bootstrap for daily returns (preserves autocorrelation)
- Terminal wealth distribution + probability of ruin
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

NEPSE_TRADING_DAYS = 240


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo trade resampling."""
    n_simulations: int
    n_trades: int
    initial_capital: float
    terminal_wealth_pcts: dict         # {5, 25, 50, 75, 95} percentiles
    max_dd_pcts: dict                  # {50, 75, 90, 95} percentiles
    sharpe_ci: Tuple[float, float]     # 95% CI for annualised Sharpe
    prob_ruin: float                   # P(terminal wealth < 50% of start)
    prob_loss: float                   # P(terminal wealth < start)
    all_terminal_wealth: np.ndarray = field(repr=False)
    all_max_dd: np.ndarray = field(repr=False)
    all_sharpe: np.ndarray = field(repr=False)


@dataclass
class BootstrapResult:
    """Results from block bootstrap on daily returns."""
    sharpe_ci: Tuple[float, float]
    cagr_ci: Tuple[float, float]
    sharpe_mean: float
    cagr_mean: float
    n_bootstrap: int
    block_size: int


def _equity_curve_from_returns(
    trade_returns: np.ndarray, initial_capital: float
) -> np.ndarray:
    """Build equity curve from per-trade returns."""
    equity = [initial_capital]
    for ret in trade_returns:
        equity.append(equity[-1] * (1 + ret))
    return np.array(equity)


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown from an equity curve."""
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    return float(np.min(dd))


def _sharpe_from_returns(returns: np.ndarray) -> float:
    """Annualised Sharpe from an array of per-trade returns."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(NEPSE_TRADING_DAYS))


def monte_carlo_trade_resample(
    trade_net_returns: List[float],
    initial_capital: float = 1_000_000,
    n_simulations: int = 10_000,
    rng_seed: Optional[int] = 42,
) -> MonteCarloResult:
    """
    Monte Carlo trade resampling with replacement.

    Shuffles the sequence of realised per-trade net returns to build
    alternative equity paths, giving distributions of terminal wealth,
    max drawdown, and Sharpe ratio.

    Parameters
    ----------
    trade_net_returns : List of net return per completed trade
    initial_capital : Starting capital (NPR)
    n_simulations : Number of Monte Carlo paths
    rng_seed : Random seed for reproducibility (None for random)

    Returns
    -------
    MonteCarloResult with percentile distributions
    """
    rng = np.random.default_rng(rng_seed)
    returns = np.array(trade_net_returns, dtype=np.float64)
    n_trades = len(returns)

    if n_trades == 0:
        empty = np.zeros(1)
        return MonteCarloResult(
            n_simulations=0,
            n_trades=0,
            initial_capital=initial_capital,
            terminal_wealth_pcts={5: initial_capital, 25: initial_capital,
                                   50: initial_capital, 75: initial_capital,
                                   95: initial_capital},
            max_dd_pcts={50: 0.0, 75: 0.0, 90: 0.0, 95: 0.0},
            sharpe_ci=(0.0, 0.0),
            prob_ruin=0.0,
            prob_loss=0.0,
            all_terminal_wealth=empty,
            all_max_dd=empty,
            all_sharpe=empty,
        )

    terminal_wealths = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)
    sharpes = np.empty(n_simulations)

    for i in range(n_simulations):
        sampled = rng.choice(returns, size=n_trades, replace=True)
        equity = _equity_curve_from_returns(sampled, initial_capital)
        terminal_wealths[i] = equity[-1]
        max_drawdowns[i] = _max_drawdown(equity)
        sharpes[i] = _sharpe_from_returns(sampled)

    tw_pcts = {
        p: float(np.percentile(terminal_wealths, p))
        for p in [5, 25, 50, 75, 95]
    }
    dd_pcts = {
        p: float(np.percentile(max_drawdowns, p))
        for p in [50, 75, 90, 95]
    }
    sharpe_ci = (
        float(np.percentile(sharpes, 2.5)),
        float(np.percentile(sharpes, 97.5)),
    )
    prob_ruin = float(np.mean(terminal_wealths < initial_capital * 0.50))
    prob_loss = float(np.mean(terminal_wealths < initial_capital))

    return MonteCarloResult(
        n_simulations=n_simulations,
        n_trades=n_trades,
        initial_capital=initial_capital,
        terminal_wealth_pcts=tw_pcts,
        max_dd_pcts=dd_pcts,
        sharpe_ci=sharpe_ci,
        prob_ruin=prob_ruin,
        prob_loss=prob_loss,
        all_terminal_wealth=terminal_wealths,
        all_max_dd=max_drawdowns,
        all_sharpe=sharpes,
    )


def block_bootstrap_ci(
    daily_returns: np.ndarray,
    n_bootstrap: int = 10_000,
    block_size: int = 21,
    confidence: float = 0.95,
    rng_seed: Optional[int] = 42,
) -> BootstrapResult:
    """
    Block bootstrap for Sharpe and CAGR confidence intervals.

    Uses non-overlapping blocks to preserve autocorrelation structure
    in daily returns.

    Parameters
    ----------
    daily_returns : Array of daily strategy returns
    n_bootstrap : Number of bootstrap replications
    block_size : Block size in trading days (21 ≈ 1 NEPSE month)
    confidence : Confidence level (default 0.95 → 95% CI)
    rng_seed : Random seed for reproducibility

    Returns
    -------
    BootstrapResult with confidence intervals
    """
    rng = np.random.default_rng(rng_seed)
    n = len(daily_returns)

    if n < block_size:
        return BootstrapResult(
            sharpe_ci=(0.0, 0.0),
            cagr_ci=(0.0, 0.0),
            sharpe_mean=0.0,
            cagr_mean=0.0,
            n_bootstrap=0,
            block_size=block_size,
        )

    # Number of blocks needed to cover original series length
    n_blocks_needed = math.ceil(n / block_size)
    # Valid starting indices for blocks
    max_start = n - block_size
    if max_start < 0:
        max_start = 0

    sharpes = np.empty(n_bootstrap)
    cagrs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # Sample random block starting indices
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        # Concatenate blocks and trim to original length
        bootstrapped = np.concatenate(
            [daily_returns[s : s + block_size] for s in starts]
        )[:n]

        # Sharpe
        std = np.std(bootstrapped, ddof=1)
        sharpes[i] = (
            np.mean(bootstrapped) / std * math.sqrt(NEPSE_TRADING_DAYS)
            if std > 0 else 0.0
        )

        # CAGR
        cumulative = np.prod(1 + bootstrapped)
        years = n / NEPSE_TRADING_DAYS
        cagrs[i] = cumulative ** (1 / years) - 1 if years > 0 else 0.0

    alpha = (1 - confidence) / 2
    lo = alpha * 100
    hi = (1 - alpha) * 100

    return BootstrapResult(
        sharpe_ci=(float(np.percentile(sharpes, lo)), float(np.percentile(sharpes, hi))),
        cagr_ci=(float(np.percentile(cagrs, lo)), float(np.percentile(cagrs, hi))),
        sharpe_mean=float(np.mean(sharpes)),
        cagr_mean=float(np.mean(cagrs)),
        n_bootstrap=n_bootstrap,
        block_size=block_size,
    )
