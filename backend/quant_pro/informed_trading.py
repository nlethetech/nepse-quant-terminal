"""Informed Trading Detection from NEPSE floorsheet data.

Model 5: Broker concentration analysis (Phase A).

NEPSE is unique globally in publishing individual trade buyer/seller
broker IDs.  This allows detection of *informed accumulation* patterns
where a single broker (or small cluster) is responsible for a
disproportionate share of buying volume --- a signature of institutional
or insider positioning.

Signals:
    1. Buyer HHI (Herfindahl-Hirschman Index) > threshold: concentrated buying.
    2. Volume Z-score > threshold: unusually high volume.
    3. Price confirmation: close should be positive or neutral (not a dump day).

Combining all three yields the informed accumulation signal.

Data sources:
    * broker_flow_daily: pre-aggregated broker-level buy/sell volumes.
    * broker_flow_summary: per-symbol daily aggregates with HHI already computed.
    * stock_prices: OHLCV for volume Z-scores and price confirmation.

Limitation: only 61 symbols with ~13 months of floorsheet data as of
2026-02.  Treat as paper-trade-only until 24+ months of data available.

Academic basis:
    * Easley, Lopez de Prado & O'Hara (2012) "Flow toxicity and liquidity"
    * Chordia & Subrahmanyam (2004) "Order imbalance and stock returns"
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backend.quant_pro.alpha_practical import AlphaSignal, SignalType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_HHI_THRESHOLD = 0.15  # single broker > 15% of buy volume
DEFAULT_VOLUME_Z_THRESHOLD = 2.0  # volume Z-score
VOLUME_Z_LOOKBACK = 20  # rolling window for Z-score
MIN_DAILY_TRADES = 10  # minimum trades on the day to compute HHI
MIN_HISTORY_DAYS = 20  # minimum days of price data for Z-score


# ---------------------------------------------------------------------------
# HHI computation from broker flow data
# ---------------------------------------------------------------------------

def _compute_buyer_hhi(
    broker_flow_df: pd.DataFrame,
    symbol: str,
    date,
) -> Optional[float]:
    """Compute buyer-side HHI for a symbol on a given date.

    HHI = sum(s_i^2) where s_i = broker_i's buy_qty / total_buy_qty.
    HHI ranges from 1/N (perfectly dispersed) to 1.0 (single buyer).

    Parameters
    ----------
    broker_flow_df : pd.DataFrame
        Broker flow daily data with columns: symbol, date, broker,
        buy_qty (or buy_amount), sell_qty, ...
    symbol : str
        Stock symbol.
    date : datetime-like
        Target date.

    Returns
    -------
    float or None
        Buyer HHI on [0, 1], or None if insufficient data.
    """
    # Filter to this symbol and date
    day_data = broker_flow_df[
        (broker_flow_df["symbol"] == symbol)
        & (broker_flow_df["date"] == date)
    ]

    if day_data.empty:
        return None

    # Determine buy quantity column name
    buy_col = None
    for candidate in ["buy_qty", "buy_quantity", "buy_amount"]:
        if candidate in day_data.columns:
            buy_col = candidate
            break

    if buy_col is None:
        logger.debug("No buy quantity column found in broker_flow_df")
        return None

    # Filter to brokers with positive buys
    buys = day_data[day_data[buy_col] > 0].copy()
    if len(buys) < 1:
        return None

    total_buy = buys[buy_col].sum()
    if total_buy <= 0:
        return None

    # Market shares
    shares = buys[buy_col].values / total_buy

    # HHI
    hhi = float(np.sum(shares ** 2))
    return hhi


def _compute_buyer_hhi_from_summary(
    broker_summary_df: pd.DataFrame,
    symbol: str,
    date,
) -> Optional[float]:
    """Extract pre-computed HHI from the broker_flow_summary table.

    This is faster than computing from raw broker flows when the summary
    table already has an HHI column.
    """
    row = broker_summary_df[
        (broker_summary_df["symbol"] == symbol)
        & (broker_summary_df["date"] == date)
    ]

    if row.empty:
        return None

    # Check for HHI column
    for col in ["hhi", "buyer_hhi", "hhi_buy"]:
        if col in row.columns:
            val = row[col].iloc[0]
            if pd.notna(val) and val > 0:
                return float(val)

    return None


# ---------------------------------------------------------------------------
# Volume Z-score from price data
# ---------------------------------------------------------------------------

def _compute_volume_zscore(
    prices_df: pd.DataFrame,
    symbol: str,
    date,
    lookback: int = VOLUME_Z_LOOKBACK,
) -> Optional[float]:
    """Compute volume Z-score: (today's volume - mean) / std over lookback.

    Returns None if insufficient history.
    """
    sym_df = prices_df[
        (prices_df["symbol"] == symbol) & (prices_df["date"] <= date)
    ].sort_values("date")

    if len(sym_df) < lookback + 1:
        return None

    recent = sym_df.tail(lookback + 1)
    volumes = recent["volume"].values

    # Today's volume
    vol_today = volumes[-1]

    # Historical (excluding today)
    hist = volumes[:-1]
    mu = np.mean(hist)
    sigma = np.std(hist)

    if sigma < 1e-6:
        return None

    z = (vol_today - mu) / sigma
    return float(z)


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_informed_flow_signals_at_date(
    prices_df: pd.DataFrame,
    broker_flow_df: pd.DataFrame,
    date: datetime,
    hhi_threshold: float = DEFAULT_HHI_THRESHOLD,
    volume_z_threshold: float = DEFAULT_VOLUME_Z_THRESHOLD,
    min_daily_trades: int = MIN_DAILY_TRADES,
    liquid_symbols: Optional[List[str]] = None,
    broker_summary_df: Optional[pd.DataFrame] = None,
) -> List[AlphaSignal]:
    """Detect informed buying from broker concentration patterns.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Full price table (symbol, date, close, volume, ...).
    broker_flow_df : pd.DataFrame
        Broker-level daily buy/sell data.  Columns must include:
        symbol, date, broker (or broker_id), buy_qty (or buy_amount).
    date : datetime
        Signal evaluation date (no lookahead on prices; broker data is
        same-day because floorsheet is published EOD).
    hhi_threshold : float
        Minimum buyer HHI to consider concentrated (default 0.15).
    volume_z_threshold : float
        Minimum volume Z-score (default 2.0).
    min_daily_trades : int
        Minimum number of unique broker entries to compute HHI (default 10).
    liquid_symbols : list[str] or None
        Pre-filtered liquid universe.
    broker_summary_df : pd.DataFrame or None
        Optional pre-aggregated summary with HHI column.  If provided,
        uses this instead of computing HHI from broker_flow_df.

    Returns
    -------
    list[AlphaSignal]
        Informed trading buy signals.
    """
    signals: List[AlphaSignal] = []

    # Determine symbol universe for this date
    if liquid_symbols:
        symbols = liquid_symbols
    else:
        # Use symbols present in broker_flow_df on this date
        day_brokers = broker_flow_df[broker_flow_df["date"] == date]
        if day_brokers.empty:
            return signals
        symbols = day_brokers["symbol"].unique().tolist()

    for symbol in symbols:
        try:
            # Step 1: Compute buyer HHI
            if broker_summary_df is not None:
                hhi = _compute_buyer_hhi_from_summary(
                    broker_summary_df, symbol, date
                )
            else:
                hhi = _compute_buyer_hhi(broker_flow_df, symbol, date)

            if hhi is None or hhi < hhi_threshold:
                continue

            # Step 2: Check minimum broker participation
            # (HHI on 2 trades is meaningless)
            day_data = broker_flow_df[
                (broker_flow_df["symbol"] == symbol)
                & (broker_flow_df["date"] == date)
            ]
            if len(day_data) < min_daily_trades:
                continue

            # Step 3: Volume Z-score
            vol_z = _compute_volume_zscore(
                prices_df, symbol, date, VOLUME_Z_LOOKBACK
            )
            if vol_z is None or vol_z < volume_z_threshold:
                continue

            # Step 4: Price confirmation -- close should not be sharply negative
            sym_df = prices_df[
                (prices_df["symbol"] == symbol) & (prices_df["date"] <= date)
            ].sort_values("date")

            if len(sym_df) < 2:
                continue

            close_today = sym_df["close"].iloc[-1]
            close_prev = sym_df["close"].iloc[-2]

            if close_prev <= 0 or close_today <= 0:
                continue

            daily_ret = (close_today / close_prev) - 1

            # Don't signal on days with big drops (>-3%) -- that's likely
            # informed SELLING, not buying
            if daily_ret < -0.03:
                continue

            # Step 5: Compute signal strength
            # Higher HHI + higher volume Z = stronger informed buying
            hhi_strength = min((hhi - hhi_threshold) / 0.30, 0.5)
            vol_z_strength = min((vol_z - volume_z_threshold) / 3.0, 0.3)

            strength = 0.25 + hhi_strength * 0.40 + vol_z_strength * 0.20
            strength = min(strength, 0.70)

            # Confidence: this is the highest-conviction NEPSE-specific signal
            # but limited by data coverage
            confidence = 0.45 + min(hhi * 0.20, 0.15)

            signals.append(AlphaSignal(
                symbol=symbol,
                signal_type=SignalType.INFORMED_TRADING,
                direction=1,
                strength=strength,
                confidence=confidence,
                reasoning=(
                    f"Informed flow: buyer HHI={hhi:.3f}, "
                    f"vol z={vol_z:.1f}, day ret={daily_ret:.1%}"
                ),
            ))

        except Exception as e:
            logger.debug("Error processing %s for informed trading: %s", symbol, e)
            continue

    signals.sort(key=lambda s: s.strength, reverse=True)
    return signals


__all__ = ["generate_informed_flow_signals_at_date"]
