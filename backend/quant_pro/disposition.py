"""Disposition Effect / Capital Gains Overhang signal.

Model 1: CGO (Grinblatt & Han 2005) -- pure OHLCV variant.

Capital Gains Overhang measures the distance between the current price
and a volume-weighted reference price.  When CGO is high and positive,
existing holders are sitting on paper gains.  The disposition effect
predicts they are reluctant to sell (the "winners hold too long" bias),
which creates a *ceiling* of potential supply.  However, when this ceiling
is breached (volume breakout), the supply vacuum triggers strong upward
continuation -- the core alpha.

CGO = (Price - ReferencePrice) / Price
ReferencePrice = 260-day VWAP (volume-weighted average price).

Signal logic:
    HIGH CGO (>0.15) + Volume breakout (>1.5x avg) = breakout through
    disposition ceiling.  Buy signal.

Academic basis:
    * Grinblatt & Han (2005) "Prospect theory, mental accounting, and momentum"
    * Frazzini (2006) "The disposition effect and underreaction to news"

Note: the earnings-surprise variant (Frazzini 2006) is deferred until
quarterly EPS data is available.
"""

import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from backend.quant_pro.alpha_practical import AlphaSignal, SignalType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CGO_LOOKBACK = 260  # ~1 year of NEPSE trading days
CGO_THRESHOLD = 0.15  # minimum CGO to consider signal
VOLUME_SPIKE_MULTIPLIER = 1.5  # volume must exceed this * 20d avg
MIN_VOLUME = 50_000  # minimum 20d avg volume (shares)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_cgo(
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = CGO_LOOKBACK,
) -> Optional[float]:
    """Compute Capital Gains Overhang for the most recent observation.

    Parameters
    ----------
    close : array-like
        Closing prices, most recent last.  Length must be >= lookback.
    volume : array-like
        Volumes (shares), aligned with close.

    Returns
    -------
    float or None
        CGO value.  Positive means price is above VWAP reference.
        Returns None if computation is invalid.
    """
    if len(close) < lookback or len(volume) < lookback:
        return None

    window_close = close[-lookback:]
    window_volume = volume[-lookback:]

    total_volume = np.sum(window_volume)
    if total_volume <= 0:
        return None

    # Volume-weighted average price over lookback window
    vwap = np.sum(window_close * window_volume) / total_volume

    current_price = close[-1]
    if current_price <= 0:
        return None

    cgo = (current_price - vwap) / current_price
    return float(cgo)


# ---------------------------------------------------------------------------
# Signal generator (follows simple_backtest.py pattern exactly)
# ---------------------------------------------------------------------------

def generate_cgo_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    cgo_threshold: float = CGO_THRESHOLD,
    volume_spike: float = VOLUME_SPIKE_MULTIPLIER,
    min_volume: float = MIN_VOLUME,
    cgo_lookback: int = CGO_LOOKBACK,
    liquid_symbols: Optional[List[str]] = None,
) -> List[AlphaSignal]:
    """Generate Capital Gains Overhang signals for a single date.

    Follows the exact pattern used by ``generate_xsec_momentum_signals_at_date``
    in simple_backtest.py: accepts ``prices_df`` + ``date``, filters by
    ``date <= signal_date``, and returns ``List[AlphaSignal]``.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Full price table with columns: symbol, date, open, high, low, close, volume.
    date : datetime
        Signal evaluation date.  Only data on or before this date is used
        (strict no-lookahead).
    cgo_threshold : float
        Minimum CGO to fire a signal (default 0.15).
    volume_spike : float
        Today's volume must exceed this multiple of 20d average (default 1.5).
    min_volume : float
        Minimum 20d average volume in shares (default 50 000).
    cgo_lookback : int
        Trailing window for VWAP reference price (default 260).
    liquid_symbols : list[str] or None
        Pre-filtered liquid universe.  If None, scan all symbols.

    Returns
    -------
    list[AlphaSignal]
        Disposition-effect buy signals, sorted by strength descending.
    """
    signals: List[AlphaSignal] = []
    symbols = liquid_symbols if liquid_symbols else prices_df["symbol"].unique()

    for symbol in symbols:
        sym_df = prices_df[
            (prices_df["symbol"] == symbol) & (prices_df["date"] <= date)
        ].sort_values("date")

        # Need enough history for 260-day VWAP + some buffer
        if len(sym_df) < cgo_lookback + 5:
            continue

        recent = sym_df.tail(cgo_lookback + 5)
        close = recent["close"].values
        volume = recent["volume"].values

        # Volume filter: 20d average volume
        avg_vol_20d = np.mean(volume[-20:])
        if avg_vol_20d < min_volume:
            continue

        # Compute CGO
        cgo = _compute_cgo(close, volume, lookback=cgo_lookback)
        if cgo is None:
            continue

        # Only interested in stocks with high positive CGO (holders sitting on gains)
        if cgo < cgo_threshold:
            continue

        # Volume breakout condition: today's volume > multiplier * 20d average
        current_volume = volume[-1]
        if current_volume <= avg_vol_20d * volume_spike:
            continue

        # Price confirmation: close should be positive today
        if close[-1] <= 0:
            continue

        # Strength: higher CGO + bigger volume spike = stronger signal
        cgo_strength = min((cgo - cgo_threshold) / 0.30, 1.0)  # 0 at threshold, 1 at 0.45
        vol_ratio = current_volume / avg_vol_20d
        vol_strength = min((vol_ratio - volume_spike) / 3.0, 0.5)  # scale volume boost

        strength = 0.25 + cgo_strength * 0.35 + vol_strength * 0.15
        strength = min(strength, 0.75)

        # Confidence: higher CGO = more confident in disposition effect
        confidence = 0.40 + min(cgo * 0.3, 0.25)

        signals.append(AlphaSignal(
            symbol=symbol,
            signal_type=SignalType.DISPOSITION,
            direction=1,
            strength=strength,
            confidence=confidence,
            reasoning=(
                f"CGO={cgo:.1%} (VWAP ref {cgo_lookback}d), "
                f"vol spike {vol_ratio:.1f}x avg"
            ),
        ))

    # Sort by strength descending for consistent ranking
    signals.sort(key=lambda s: s.strength, reverse=True)
    return signals


__all__ = ["generate_cgo_signals_at_date"]
