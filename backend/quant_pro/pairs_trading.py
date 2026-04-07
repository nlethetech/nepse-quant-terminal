"""Pairs Trading using Ornstein-Uhlenbeck process.

Model 13: Gatev et al. (2006), refined with OU half-life estimation.

Identifies cointegrated pairs and trades mean reversion of spreads.
NEPSE constraint: long-only, so we BUY the cheap leg only (the stock
that is below its fair value relative to the partner).

Algorithm:
    1. Maintain a list of known cointegrated pairs (pre-tested via
       Engle-Granger in the agent 3 analysis).
    2. For each active pair, compute the OLS hedge ratio and spread.
    3. Compute Z-score of current spread vs rolling mean/std.
    4. Entry: |Z| > entry_z => buy cheap leg.
    5. Exit: |Z| < exit_z (mean reversion).
    6. Stop: |Z| > stop_z (spread blowup).
    7. OU half-life filter: skip pairs with half-life > 60 days.

Academic basis:
    * Gatev, Goetzmann & Rouwenhorst (2006) "Pairs trading: Performance
      of a relative-value arbitrage rule"
    * Krauss (2017) "Statistical Arbitrage Pairs Trading Strategies"
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.quant_pro.alpha_practical import AlphaSignal, SignalType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known cointegrated pairs (from Agent 3 empirical testing)
# ---------------------------------------------------------------------------

KNOWN_PAIRS: List[Tuple[str, str]] = [
    # Banking (p < 0.05, Engle-Granger)
    ("KBL", "NBL"),
    ("PCBL", "GBIME"),
    ("PCBL", "ADBL"),
    ("NABIL", "NBL"),
    ("GBIME", "ADBL"),
    ("SCB", "ADBL"),
    ("SCB", "SBI"),
    ("SCB", "NBL"),
    ("GBIME", "KBL"),
    ("KBL", "MBL"),
    # Hydropower (p < 0.05)
    ("AHPC", "UPCL"),
    ("AHPC", "API"),
    ("AHPC", "CHL"),
    ("API", "CHL"),
    ("NHPC", "AKPL"),
    ("BARUN", "SHPC"),
    ("CHL", "UPCL"),
    # Insurance (p < 0.05)
    ("LICN", "UAIL"),
    ("ALICL", "NIL"),
    ("ALICL", "LICN"),
    ("HLI", "SICL"),
    ("LICN", "NIL"),
    ("ALICL", "UAIL"),
    # Microfinance (p < 0.05)
    ("GBLBS", "LLBS"),
    ("GBLBS", "VLBS"),
    ("LLBS", "VLBS"),
    ("DDBL", "SWBBL"),
    ("CBBL", "SKBBL"),
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ENTRY_Z = 2.0
DEFAULT_EXIT_Z = 0.5
DEFAULT_STOP_Z = 3.5
DEFAULT_LOOKBACK = 252
MAX_OU_HALFLIFE = 60  # skip pairs with HL > 60 trading days
MIN_HISTORY = 200  # minimum common observations


# ---------------------------------------------------------------------------
# OU process estimation
# ---------------------------------------------------------------------------

def _estimate_ou_halflife(spread: np.ndarray) -> Optional[float]:
    """Estimate Ornstein-Uhlenbeck half-life from AR(1) regression on spread.

    OU process: dS = theta * (mu - S) dt + sigma dW
    Discretized: delta_S = phi * S_{t-1} + c + eps
    where phi = exp(-theta * dt) - 1
    Half-life = -ln(2) / ln(1 + phi) = -ln(2) / ln(1 - theta_dt)

    Returns None if estimation fails or half-life is negative/infinite.
    """
    if len(spread) < 30:
        return None

    delta_s = np.diff(spread)
    s_lag = spread[:-1]

    # OLS: delta_s = phi * s_lag + c + eps
    n = len(delta_s)
    X = np.column_stack([s_lag, np.ones(n)])

    try:
        # Use least-squares (numerically stable)
        result = np.linalg.lstsq(X, delta_s, rcond=None)
        phi = result[0][0]

        if phi >= 0:
            # Not mean-reverting
            return None

        # Half-life = -ln(2) / ln(1 + phi)
        inner = 1 + phi
        if inner <= 0 or inner >= 1:
            return None

        halflife = -np.log(2) / np.log(inner)

        if halflife <= 0 or np.isinf(halflife) or np.isnan(halflife):
            return None

        return float(halflife)

    except (np.linalg.LinAlgError, ValueError):
        return None


def _compute_hedge_ratio(prices_a: np.ndarray, prices_b: np.ndarray) -> float:
    """OLS hedge ratio: prices_a = alpha + beta * prices_b + eps.

    Returns beta (hedge ratio).
    """
    n = len(prices_a)
    X = np.column_stack([prices_b, np.ones(n)])
    try:
        result = np.linalg.lstsq(X, prices_a, rcond=None)
        beta = result[0][0]
        return float(beta)
    except (np.linalg.LinAlgError, ValueError):
        return 1.0  # fallback: 1:1 ratio


# ---------------------------------------------------------------------------
# Pairs Trader
# ---------------------------------------------------------------------------

class PairsTrader:
    """Pairs trading signal generator using OU mean-reversion.

    Parameters
    ----------
    entry_z : float
        Z-score threshold for entry (default 2.0).
    exit_z : float
        Z-score threshold for exit / mean reversion (default 0.5).
    stop_z : float
        Z-score threshold for stop-loss (default 3.5).
    lookback : int
        Rolling window for spread statistics (default 252).
    max_halflife : float
        Maximum OU half-life to consider a pair tradeable (default 60).
    """

    def __init__(
        self,
        entry_z: float = DEFAULT_ENTRY_Z,
        exit_z: float = DEFAULT_EXIT_Z,
        stop_z: float = DEFAULT_STOP_Z,
        lookback: int = DEFAULT_LOOKBACK,
        max_halflife: float = MAX_OU_HALFLIFE,
    ):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.lookback = lookback
        self.max_halflife = max_halflife

    def compute_spread(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, float, Optional[float]]:
        """Compute spread, z-score, hedge ratio, and OU half-life.

        Parameters
        ----------
        prices_a, prices_b : np.ndarray
            Aligned price series for leg A and leg B.

        Returns
        -------
        tuple of (spread, z_score, hedge_ratio, spread_mean, halflife)
        """
        beta = _compute_hedge_ratio(prices_a, prices_b)
        spread = prices_a - beta * prices_b

        spread_mean = np.mean(spread)
        spread_std = np.std(spread)

        if spread_std < 1e-10:
            z_score = 0.0
        else:
            z_score = (spread[-1] - spread_mean) / spread_std

        halflife = _estimate_ou_halflife(spread)

        return spread, float(z_score), float(beta), float(spread_mean), halflife

    def _get_pair_prices(
        self,
        prices_df: pd.DataFrame,
        sym_a: str,
        sym_b: str,
        date,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract aligned price arrays for a pair up to date.

        Returns None if insufficient history.
        """
        df_a = prices_df[
            (prices_df["symbol"] == sym_a) & (prices_df["date"] <= date)
        ].sort_values("date").tail(self.lookback + 10)

        df_b = prices_df[
            (prices_df["symbol"] == sym_b) & (prices_df["date"] <= date)
        ].sort_values("date").tail(self.lookback + 10)

        if len(df_a) < MIN_HISTORY or len(df_b) < MIN_HISTORY:
            return None

        # Align by date (inner join)
        merged = pd.merge(
            df_a[["date", "close"]].rename(columns={"close": "close_a"}),
            df_b[["date", "close"]].rename(columns={"close": "close_b"}),
            on="date",
            how="inner",
        ).sort_values("date")

        if len(merged) < MIN_HISTORY:
            return None

        # Use the most recent `lookback` observations
        merged = merged.tail(self.lookback)

        prices_a_arr = merged["close_a"].values.astype(float)
        prices_b_arr = merged["close_b"].values.astype(float)

        # Validate: no zeros or negatives
        if np.any(prices_a_arr <= 0) or np.any(prices_b_arr <= 0):
            return None

        return prices_a_arr, prices_b_arr

    def generate_signals(
        self,
        prices_df: pd.DataFrame,
        date: datetime,
        pairs: Optional[List[Tuple[str, str]]] = None,
        liquid_symbols: Optional[List[str]] = None,
    ) -> List[AlphaSignal]:
        """Generate pairs trading signals for all known/given pairs.

        In NEPSE's long-only constraint, we can only BUY the cheap leg.
        When the spread is significantly positive (A overvalued relative
        to B), buy B.  When significantly negative, buy A.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Full price table.
        date : datetime
            Signal date (no lookahead).
        pairs : list of (str, str) or None
            Pairs to evaluate.  Defaults to KNOWN_PAIRS.
        liquid_symbols : list[str] or None
            If provided, only emit signals for symbols in this set.

        Returns
        -------
        list[AlphaSignal]
            Buy signals for cheap legs of active pairs.
        """
        signals: List[AlphaSignal] = []
        pairs_to_check = pairs if pairs is not None else KNOWN_PAIRS
        liquid_set = set(liquid_symbols) if liquid_symbols else None

        for sym_a, sym_b in pairs_to_check:
            try:
                result = self._get_pair_prices(prices_df, sym_a, sym_b, date)
                if result is None:
                    continue

                prices_a_arr, prices_b_arr = result
                spread, z_score, beta, spread_mean, halflife = self.compute_spread(
                    prices_a_arr, prices_b_arr
                )

                # Filter: skip if half-life is too long or not mean-reverting
                if halflife is not None and halflife > self.max_halflife:
                    continue

                # Skip if not at extreme
                if abs(z_score) < self.entry_z:
                    continue

                # Skip if past stop level (divergence is too extreme)
                if abs(z_score) > self.stop_z:
                    logger.debug(
                        "Pair %s-%s at stop level z=%.2f, skipping",
                        sym_a, sym_b, z_score,
                    )
                    continue

                # Determine cheap leg
                if z_score > self.entry_z:
                    # Spread is high: A is expensive relative to B => buy B
                    cheap_leg = sym_b
                    expensive_leg = sym_a
                elif z_score < -self.entry_z:
                    # Spread is low: A is cheap relative to B => buy A
                    cheap_leg = sym_a
                    expensive_leg = sym_b
                else:
                    continue

                # Respect liquidity filter
                if liquid_set is not None and cheap_leg not in liquid_set:
                    continue

                # Strength: proportional to z-score (higher = more mispriced)
                abs_z = abs(z_score)
                strength = min(
                    0.25 + (abs_z - self.entry_z) * 0.15,
                    0.65,
                )

                # Confidence: higher with shorter half-life (faster reversion)
                base_conf = 0.40
                if halflife is not None:
                    hl_bonus = max(0, (self.max_halflife - halflife) / self.max_halflife) * 0.15
                    base_conf += hl_bonus

                hl_str = f"{halflife:.0f}d" if halflife else "N/A"

                signals.append(AlphaSignal(
                    symbol=cheap_leg,
                    signal_type=SignalType.PAIRS_TRADE,
                    direction=1,
                    strength=strength,
                    confidence=min(base_conf, 0.60),
                    reasoning=(
                        f"Pair {sym_a}-{sym_b}: z={z_score:.2f}, "
                        f"beta={beta:.3f}, HL={hl_str}, "
                        f"buy {cheap_leg} (cheap leg)"
                    ),
                ))

            except Exception as e:
                logger.debug("Error processing pair %s-%s: %s", sym_a, sym_b, e)
                continue

        # Deduplicate: if a symbol appears as cheap leg in multiple pairs,
        # keep the strongest signal
        best_per_symbol: Dict[str, AlphaSignal] = {}
        for sig in signals:
            existing = best_per_symbol.get(sig.symbol)
            if existing is None or sig.strength > existing.strength:
                best_per_symbol[sig.symbol] = sig

        result = list(best_per_symbol.values())
        result.sort(key=lambda s: s.strength, reverse=True)
        return result


# ---------------------------------------------------------------------------
# Convenience function (matches pattern of other signal generators)
# ---------------------------------------------------------------------------

def generate_pairs_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    entry_z: float = DEFAULT_ENTRY_Z,
    exit_z: float = DEFAULT_EXIT_Z,
    stop_z: float = DEFAULT_STOP_Z,
    lookback: int = DEFAULT_LOOKBACK,
    liquid_symbols: Optional[List[str]] = None,
) -> List[AlphaSignal]:
    """Functional interface for pairs trading signals.

    Wraps PairsTrader.generate_signals() to match the standard pattern
    used by simple_backtest.py signal generators.
    """
    trader = PairsTrader(
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        lookback=lookback,
    )
    return trader.generate_signals(
        prices_df, date, liquid_symbols=liquid_symbols
    )


__all__ = [
    "PairsTrader",
    "KNOWN_PAIRS",
    "generate_pairs_signals_at_date",
]
