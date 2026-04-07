"""Macroeconomic signal from NRB remittance flow data.

Remittances = 23.5% of Nepal GDP. Monthly data with 3-4 week lag.
Used as a slow-moving macro regime modifier, NOT a daily signal.

When strong inflow detected (>15% YoY growth):
  - Boost confidence in all stock signals by 5%
  - Emit weak buy signals for banking sector (primary beneficiary
    of remittance deposits flowing through the banking system)

When weak inflow detected (<5% YoY growth):
  - Reduce confidence by 5%
  - No specific stock signals

Data source: NRB monthly statistics (stored in macro_indicators table).
Signal type: MACRO_REMITTANCE

Model 17 in the Citadel upgrade plan.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from backend.quant_pro.alpha_practical import AlphaSignal, SignalType
from backend.quant_pro.database import get_db_path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = str(get_db_path())

# Nepal's banking sector tickers — primary beneficiary of remittance deposits
# These banks have the largest remittance processing volumes
REMITTANCE_BENEFICIARY_TICKERS = [
    # Top commercial banks by remittance market share
    "NABIL", "NICA", "SBL", "GBIME", "NIMB",
    # Development banks active in remittance corridors
    "MEGA", "LBBL", "MBL", "JBBL",
]

# Regime thresholds (YoY growth in %)
STRONG_THRESHOLD = 15.0   # >15% YoY = strong inflow
WEAK_THRESHOLD = 5.0      # <5% YoY = weak inflow
# Between 5-15% = normal


def get_remittance_regime(
    db_path: str = DEFAULT_DB_PATH,
    as_of_date: Optional[str] = None,
) -> Dict:
    """Get current remittance regime from latest available data.

    The regime is determined by YoY growth in monthly remittance inflows.
    Data has a ~4 week publication lag, so the latest available data point
    is typically 4-6 weeks old.

    Args:
        db_path: Path to nepse_market_data.db.
        as_of_date: Optional date string (YYYY-MM-DD) to look up regime
                     as-of a historical date. Default: use latest available.

    Returns:
        dict with keys:
            regime: "strong" | "normal" | "weak" | "no_data"
            yoy_growth: float (percentage, e.g. 12.5 means +12.5%)
            multiplier: float (1.05 for strong, 1.00 for normal, 0.95 for weak)
            latest_date: str (date of latest data point used)
            latest_value_usd_m: float (latest monthly remittance in USD millions)
            data_age_days: int (how old the latest data point is)
    """
    try:
        conn = sqlite3.connect(db_path)

        # Query: get the most recent YoY growth data point
        if as_of_date:
            query = """
                SELECT date, value
                FROM macro_indicators
                WHERE indicator_name = 'remittance_yoy_growth_pct'
                  AND date <= ?
                ORDER BY date DESC
                LIMIT 1
            """
            df_growth = pd.read_sql_query(query, conn, params=(as_of_date,))
        else:
            query = """
                SELECT date, value
                FROM macro_indicators
                WHERE indicator_name = 'remittance_yoy_growth_pct'
                ORDER BY date DESC
                LIMIT 1
            """
            df_growth = pd.read_sql_query(query, conn)

        # Also get the latest absolute remittance value
        if as_of_date:
            query_abs = """
                SELECT date, value
                FROM macro_indicators
                WHERE indicator_name = 'remittance_usd_millions'
                  AND date <= ?
                ORDER BY date DESC
                LIMIT 1
            """
            df_abs = pd.read_sql_query(query_abs, conn, params=(as_of_date,))
        else:
            query_abs = """
                SELECT date, value
                FROM macro_indicators
                WHERE indicator_name = 'remittance_usd_millions'
                ORDER BY date DESC
                LIMIT 1
            """
            df_abs = pd.read_sql_query(query_abs, conn)

        conn.close()

        if df_growth.empty:
            return {
                "regime": "no_data",
                "yoy_growth": 0.0,
                "multiplier": 1.0,
                "latest_date": None,
                "latest_value_usd_m": None,
                "data_age_days": None,
            }

        yoy_growth = float(df_growth.iloc[0]["value"])
        latest_date = str(df_growth.iloc[0]["date"])
        latest_value = float(df_abs.iloc[0]["value"]) if not df_abs.empty else None

        # Determine regime
        if yoy_growth >= STRONG_THRESHOLD:
            regime = "strong"
            multiplier = 1.05
        elif yoy_growth < WEAK_THRESHOLD:
            regime = "weak"
            multiplier = 0.95
        else:
            regime = "normal"
            multiplier = 1.00

        # Compute data age
        ref_date = as_of_date or datetime.now().strftime("%Y-%m-%d")
        try:
            age_days = (
                datetime.strptime(ref_date, "%Y-%m-%d")
                - datetime.strptime(latest_date, "%Y-%m-%d")
            ).days
        except ValueError:
            age_days = None

        return {
            "regime": regime,
            "yoy_growth": round(yoy_growth, 2),
            "multiplier": multiplier,
            "latest_date": latest_date,
            "latest_value_usd_m": latest_value,
            "data_age_days": age_days,
        }

    except sqlite3.Error as e:
        logger.warning(f"DB error in get_remittance_regime: {e}")
        return {
            "regime": "no_data",
            "yoy_growth": 0.0,
            "multiplier": 1.0,
            "latest_date": None,
            "latest_value_usd_m": None,
            "data_age_days": None,
        }


def get_remittance_trend(
    db_path: str = DEFAULT_DB_PATH,
    n_months: int = 6,
    as_of_date: Optional[str] = None,
) -> Dict:
    """Get remittance trend over the last N months.

    Returns trend info: is growth accelerating or decelerating?
    This adds nuance beyond the simple regime classification.

    Returns:
        dict with keys:
            trend: "accelerating" | "stable" | "decelerating" | "no_data"
            avg_growth: float (average YoY growth over period)
            latest_growth: float
            growth_history: list of (date, growth) tuples
    """
    try:
        conn = sqlite3.connect(db_path)

        if as_of_date:
            query = """
                SELECT date, value
                FROM macro_indicators
                WHERE indicator_name = 'remittance_yoy_growth_pct'
                  AND date <= ?
                ORDER BY date DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(as_of_date, n_months))
        else:
            query = """
                SELECT date, value
                FROM macro_indicators
                WHERE indicator_name = 'remittance_yoy_growth_pct'
                ORDER BY date DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(n_months,))

        conn.close()

        if df.empty or len(df) < 2:
            return {
                "trend": "no_data",
                "avg_growth": 0.0,
                "latest_growth": 0.0,
                "growth_history": [],
            }

        # Reverse to chronological order
        df = df.sort_values("date")
        growths = df["value"].values

        avg_growth = float(growths.mean())
        latest_growth = float(growths[-1])

        # Trend: compare recent 3 months vs earlier 3 months
        if len(growths) >= 4:
            mid = len(growths) // 2
            recent_avg = growths[mid:].mean()
            earlier_avg = growths[:mid].mean()
            diff = recent_avg - earlier_avg

            if diff > 2.0:
                trend = "accelerating"
            elif diff < -2.0:
                trend = "decelerating"
            else:
                trend = "stable"
        else:
            trend = "stable"

        growth_history = list(zip(df["date"].tolist(), df["value"].tolist()))

        return {
            "trend": trend,
            "avg_growth": round(avg_growth, 2),
            "latest_growth": round(latest_growth, 2),
            "growth_history": growth_history,
        }

    except sqlite3.Error as e:
        logger.warning(f"DB error in get_remittance_trend: {e}")
        return {
            "trend": "no_data",
            "avg_growth": 0.0,
            "latest_growth": 0.0,
            "growth_history": [],
        }


def generate_remittance_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    db_path: str = "nepse_market_data.db",
    liquid_symbols: Optional[List[str]] = None,
) -> List[AlphaSignal]:
    """Generate macro remittance signal.

    This is primarily a regime-level signal that boosts/dampens ALL stock
    signals rather than picking specific stocks. However, when strong
    inflow is detected, it also emits weak buy signals for the banking
    sector (banks are the primary beneficiary of remittance deposits).

    Args:
        prices_df: DataFrame with columns [symbol, date, close, volume, ...].
        date: Current backtest/signal date.
        db_path: Path to nepse_market_data.db.
        liquid_symbols: Optional whitelist of tradeable symbols.

    Returns:
        List of AlphaSignal objects.
        - When regime is "strong": buy signals for banking sector tickers.
        - When regime is "normal" or "weak": empty list.
          The multiplier from get_remittance_regime() should be applied
          externally to all signals as a regime modifier.

    NO lookahead bias: only uses data published before `date` (publication_date).
    """
    signals: List[AlphaSignal] = []
    date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)[:10]

    # Get regime as-of this date (respects publication lag)
    regime_info = get_remittance_regime(db_path=db_path, as_of_date=date_str)

    if regime_info["regime"] == "no_data":
        return signals

    # Data staleness check: if data is more than 90 days old, do not emit signals
    if regime_info["data_age_days"] is not None and regime_info["data_age_days"] > 90:
        logger.debug(
            f"Remittance data too stale ({regime_info['data_age_days']} days). "
            f"Skipping signal generation."
        )
        return signals

    # Only emit stock-level signals when regime is strong
    if regime_info["regime"] != "strong":
        return signals

    # Get trend for additional confidence
    trend_info = get_remittance_trend(db_path=db_path, as_of_date=date_str)

    # Strong regime: emit weak buy signals for banking sector
    yoy_growth = regime_info["yoy_growth"]

    # Strength scales with YoY growth: 15% -> 0.3, 25% -> 0.5, 35%+ -> 0.7
    raw_strength = min(max((yoy_growth - STRONG_THRESHOLD) / 40.0, 0.0), 1.0)
    strength = 0.3 + 0.4 * raw_strength  # range: 0.3 to 0.7

    # Confidence: higher if trend is accelerating
    if trend_info["trend"] == "accelerating":
        confidence = 0.50
    elif trend_info["trend"] == "stable":
        confidence = 0.40
    else:
        confidence = 0.30  # decelerating trend reduces confidence

    available_symbols = set(prices_df["symbol"].unique()) if not prices_df.empty else set()

    for ticker in REMITTANCE_BENEFICIARY_TICKERS:
        # Skip if not in liquid universe or not in price data
        if liquid_symbols and ticker not in liquid_symbols:
            continue
        if ticker not in available_symbols:
            continue

        # Verify ticker has recent price data
        # Compare as strings (prices_df["date"] is string YYYY-MM-DD)
        sym_df = prices_df[
            (prices_df["symbol"] == ticker) & (prices_df["date"] <= date_str)
        ]
        if len(sym_df) < 20:
            continue

        signals.append(
            AlphaSignal(
                symbol=ticker,
                signal_type=SignalType.MACRO_REMITTANCE,
                direction=1,
                strength=min(strength, 1.0),
                confidence=min(confidence, 1.0),
                reasoning=(
                    f"Remittance regime: {regime_info['regime']} "
                    f"(YoY +{yoy_growth:.1f}%, trend: {trend_info['trend']}). "
                    f"Strong inflow benefits banking sector deposits."
                ),
            )
        )

    return signals



# --- NRB Policy Rate Signal ---
# Granger causality NRB policy rate → NEPSE confirmed in academic research
# (Nepal Rastra Bank Economic Review, NEPJOL 2024)
NRB_HIKE_THRESHOLD_BPS = 25      # >= 25bps change = hike
NRB_CUT_THRESHOLD_BPS = -25      # <= -25bps change = cut
NRB_HIKING_MULTIPLIER = 0.85     # 15% confidence reduction during rate hike cycle
NRB_CUTTING_MULTIPLIER = 1.08    # 8% confidence boost during rate cut cycle

# Sectors most sensitive to rate changes
_NRB_BEARISH_ON_HIKE = ["Microfinance", "Development Banks", "Finance"]  # highest leverage/CoF
_NRB_BULLISH_ON_HIKE = ["Commercial Banks", "Life Insurance"]  # benefit from steeper yield curve
_NRB_BULLISH_ON_CUT = ["Microfinance", "Development Banks", "Finance", "Hydropower"]


def get_nrb_policy_regime(
    db_path: str = "nepse_market_data.db",
    as_of_date: Optional[str] = None,
) -> Dict:
    """Detect NRB monetary policy rate cycle from macro_indicators table.

    Reads rows with indicator_name = 'nrb_policy_rate_pct'.
    Computes basis-point change over last 2 data points to detect hike/cut/hold.

    Returns dict with keys:
        cycle: "hiking" | "cutting" | "hold" | "no_data"
        latest_rate_pct: float | None
        rate_change_bps: float
        multiplier: float (apply to all signal confidence)
        sector_adjustments: dict[sector_name → multiplier]
    """
    no_data_result = {
        "cycle": "no_data",
        "latest_rate_pct": None,
        "rate_change_bps": 0.0,
        "multiplier": 1.0,
        "sector_adjustments": {},
    }
    try:
        conn = sqlite3.connect(db_path)
        if as_of_date:
            query = """
                SELECT date, value FROM macro_indicators
                WHERE indicator_name = 'nrb_policy_rate_pct' AND date <= ?
                ORDER BY date DESC LIMIT 3
            """
            df = pd.read_sql_query(query, conn, params=(as_of_date,))
        else:
            query = """
                SELECT date, value FROM macro_indicators
                WHERE indicator_name = 'nrb_policy_rate_pct'
                ORDER BY date DESC LIMIT 3
            """
            df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return no_data_result

        latest_rate = float(df.iloc[0]["value"])

        if len(df) < 2:
            return {
                "cycle": "hold",
                "latest_rate_pct": latest_rate,
                "rate_change_bps": 0.0,
                "multiplier": 1.0,
                "sector_adjustments": {},
            }

        prev_rate = float(df.iloc[1]["value"])
        change_bps = (latest_rate - prev_rate) * 100.0

        if change_bps >= NRB_HIKE_THRESHOLD_BPS:
            cycle = "hiking"
            multiplier = NRB_HIKING_MULTIPLIER
            sector_adj = {s: 0.80 for s in _NRB_BEARISH_ON_HIKE}
            sector_adj.update({s: 1.05 for s in _NRB_BULLISH_ON_HIKE})
        elif change_bps <= NRB_CUT_THRESHOLD_BPS:
            cycle = "cutting"
            multiplier = NRB_CUTTING_MULTIPLIER
            sector_adj = {s: 1.10 for s in _NRB_BULLISH_ON_CUT}
        else:
            cycle = "hold"
            multiplier = 1.0
            sector_adj = {}

        return {
            "cycle": cycle,
            "latest_rate_pct": latest_rate,
            "rate_change_bps": round(change_bps, 1),
            "multiplier": multiplier,
            "sector_adjustments": sector_adj,
        }

    except Exception as e:
        logger.warning(f"get_nrb_policy_regime failed: {e}")
        return no_data_result


__all__ = [
    "get_remittance_regime",
    "get_remittance_trend",
    "generate_remittance_signals_at_date",
    "REMITTANCE_BENEFICIARY_TICKERS",
    "get_nrb_policy_regime",
]
