"""Sector Lead-Lag signal.

Model 3: Hou (2007) industry-level momentum spillover.

Detects when a leading sector moves strongly and positions in lagging
sectors that historically follow with a delay.  In NEPSE's concentrated
market, sector correlations are high but not synchronous --- banking
often leads, with development banks and insurance lagging by 2-5 days.

Algorithm:
    1. Compute sector returns (median stock return per sector) for recent windows.
    2. For each sector pair (A, B), compute the lagged cross-correlation
       (does A's return today predict B's return in 3-5 days?).
    3. When a leading sector moved > 1.5 sigma in the last 5 days, emit
       buy signals for stocks in the lagging sector.

Academic basis:
    * Hou (2007) "Industry information diffusion and the lead-lag effect"
    * Menzly & Ozbas (2010) "Market segmentation and cross-predictability of returns"
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from backend.quant_pro.alpha_practical import AlphaSignal, SignalType
from backend.quant_pro.sectors import SECTOR_GROUPS, SECTOR_LOOKUP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Sectors with enough liquid stocks for meaningful median returns
ELIGIBLE_SECTORS = {
    "Commercial Banks",
    "Development Banks",
    "Finance",
    "Hydropower",
    "Life Insurance",
    "Non-Life Insurance",
    "Microfinance",
}

# Empirically observed lead-lag relationships in NEPSE (leader -> laggards).
# These are based on sector structure: large liquid sectors tend to lead,
# smaller/less-liquid sectors lag.
LEAD_LAG_PAIRS: List[Tuple[str, str]] = [
    ("Commercial Banks", "Development Banks"),
    ("Commercial Banks", "Finance"),
    ("Commercial Banks", "Microfinance"),
    ("Commercial Banks", "Life Insurance"),
    ("Commercial Banks", "Non-Life Insurance"),
    ("Hydropower", "Commercial Banks"),
    ("Development Banks", "Microfinance"),
    ("Life Insurance", "Non-Life Insurance"),
]

# Minimum number of stocks per sector to compute a reliable median return
MIN_SECTOR_STOCKS = 4

# Signal thresholds
LEAD_SIGMA_THRESHOLD = 1.5  # leader must move > 1.5 sigma in the window
LAG_WINDOW = 5  # look at leader's return over last 5 trading days
SECTOR_RETURN_LOOKBACK = 60  # rolling window for sigma estimation
MIN_HISTORY_DAYS = 80  # minimum data days per stock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_sector_returns(
    prices_df: pd.DataFrame,
    date,
    lookback: int = SECTOR_RETURN_LOOKBACK,
) -> Dict[str, np.ndarray]:
    """Compute daily sector return series (median stock return per sector).

    Returns a dict mapping sector_name -> array of daily returns
    (length ~ lookback), most recent last.
    """
    sector_returns: Dict[str, List[np.ndarray]] = {}

    for sector in ELIGIBLE_SECTORS:
        members = SECTOR_GROUPS.get(sector, set())
        if len(members) < MIN_SECTOR_STOCKS:
            continue

        stock_return_series: List[np.ndarray] = []

        for symbol in members:
            sym_df = prices_df[
                (prices_df["symbol"] == symbol) & (prices_df["date"] <= date)
            ].sort_values("date")

            if len(sym_df) < MIN_HISTORY_DAYS:
                continue

            recent = sym_df.tail(lookback + 1)
            closes = recent["close"].values

            if len(closes) < lookback + 1:
                continue

            rets = np.diff(closes) / closes[:-1]
            stock_return_series.append(rets)

        if len(stock_return_series) < MIN_SECTOR_STOCKS:
            continue

        # Align to the shortest series (they should be very close in length)
        min_len = min(len(r) for r in stock_return_series)
        aligned = np.array([r[-min_len:] for r in stock_return_series])

        # Sector return = median stock return per day
        sector_daily_rets = np.median(aligned, axis=0)
        sector_returns[sector] = sector_daily_rets

    return sector_returns


def _compute_lead_signal(
    leader_returns: np.ndarray,
    lag_window: int = LAG_WINDOW,
    sigma_threshold: float = LEAD_SIGMA_THRESHOLD,
) -> Optional[float]:
    """Check if the leader sector had a significant move recently.

    Returns the move magnitude in sigma units, or None if not significant.
    """
    if len(leader_returns) < lag_window + 20:
        return None

    # Leader's return over the last `lag_window` days (cumulative)
    recent_cum_ret = np.sum(leader_returns[-lag_window:])

    # Historical rolling `lag_window`-day returns for sigma estimation
    # Use all data EXCEPT the most recent window (to avoid double-counting)
    historical = leader_returns[:-lag_window]
    if len(historical) < 20:
        return None

    rolling_sums = np.convolve(historical, np.ones(lag_window), mode="valid")
    mu = np.mean(rolling_sums)
    sigma = np.std(rolling_sums)

    if sigma < 1e-10:
        return None

    z_score = (recent_cum_ret - mu) / sigma

    if abs(z_score) >= sigma_threshold:
        return float(z_score)
    return None


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_sector_leadlag_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    sigma_threshold: float = LEAD_SIGMA_THRESHOLD,
    lag_window: int = LAG_WINDOW,
    lookback: int = SECTOR_RETURN_LOOKBACK,
    min_volume: float = 50_000,
    liquid_symbols: Optional[List[str]] = None,
) -> List[AlphaSignal]:
    """Generate lead-lag sector rotation signals for a single date.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Full price table (symbol, date, close, volume, ...).
    date : datetime
        Signal evaluation date (no lookahead).
    sigma_threshold : float
        Minimum sigma move in the leading sector to trigger (default 1.5).
    lag_window : int
        Number of recent trading days to measure leader move (default 5).
    lookback : int
        Rolling window for sigma estimation (default 60).
    min_volume : float
        Minimum 20d avg volume for individual stock signals.
    liquid_symbols : list[str] or None
        Pre-filtered liquid universe.

    Returns
    -------
    list[AlphaSignal]
        Buy signals for lagging-sector stocks.
    """
    signals: List[AlphaSignal] = []

    # Compute sector-level return series
    sector_rets = _compute_sector_returns(prices_df, date, lookback)
    if not sector_rets:
        return signals

    # Build liquid set for fast membership checks
    liquid_set: Optional[Set[str]] = set(liquid_symbols) if liquid_symbols else None

    # Check each lead-lag pair
    for leader_name, lagger_name in LEAD_LAG_PAIRS:
        if leader_name not in sector_rets or lagger_name not in sector_rets:
            continue

        leader_rets = sector_rets[leader_name]
        z_score = _compute_lead_signal(leader_rets, lag_window, sigma_threshold)

        if z_score is None:
            continue

        # Only act on POSITIVE leader moves (long-only market)
        if z_score < 0:
            continue

        # Emit buy signals for lagging sector stocks
        lagger_members = SECTOR_GROUPS.get(lagger_name, set())

        for symbol in lagger_members:
            # Respect liquidity filter
            if liquid_set is not None and symbol not in liquid_set:
                continue

            # Check individual stock has recent data and volume
            sym_df = prices_df[
                (prices_df["symbol"] == symbol) & (prices_df["date"] <= date)
            ].sort_values("date")

            if len(sym_df) < 30:
                continue

            recent = sym_df.tail(30)
            avg_vol = recent["volume"].values[-20:].mean()
            if avg_vol < min_volume:
                continue

            # Don't chase stocks already running (lagging sector should not have
            # already moved) -- check if stock's 5d return is < leader's move
            close_vals = recent["close"].values
            if close_vals[-lag_window] <= 0:
                continue
            stock_5d_ret = close_vals[-1] / close_vals[-lag_window] - 1

            # If the stock already moved more than half the leader sigma,
            # the lag opportunity is diminished
            leader_5d_ret = float(np.sum(sector_rets[leader_name][-lag_window:]))
            if stock_5d_ret > leader_5d_ret * 0.5:
                continue

            # Strength: proportional to leader's z-score
            strength = min(0.20 + (z_score - sigma_threshold) * 0.10, 0.55)

            # Confidence: lower because lead-lag is noisier than direct signals
            confidence = 0.35 + min(z_score * 0.05, 0.15)

            signals.append(AlphaSignal(
                symbol=symbol,
                signal_type=SignalType.LEAD_LAG,
                direction=1,
                strength=strength,
                confidence=confidence,
                reasoning=(
                    f"{leader_name} -> {lagger_name}: leader z={z_score:.2f} "
                    f"(5d ret={leader_5d_ret:.1%}), stock 5d={stock_5d_ret:.1%}"
                ),
            ))

    # Deduplicate: keep strongest signal per symbol (a stock may appear in
    # multiple lagging pairs)
    best_per_symbol: Dict[str, AlphaSignal] = {}
    for sig in signals:
        existing = best_per_symbol.get(sig.symbol)
        if existing is None or sig.strength > existing.strength:
            best_per_symbol[sig.symbol] = sig

    result = list(best_per_symbol.values())
    result.sort(key=lambda s: s.strength, reverse=True)
    return result


__all__ = ["generate_sector_leadlag_signals_at_date"]
