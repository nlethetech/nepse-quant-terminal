"""Satellite rainfall -> hydropower stock signal.

Rain in Nepal directly determines run-of-river hydro output.
Above-normal rainfall = bullish for hydro stocks.
Below-normal rainfall = bearish for hydro stocks.

Data source: Open-Meteo (stored in weather_data table).
Signal type: SATELLITE_HYDRO

Nepal monsoon calendar:
  - Pre-monsoon (Mar-May): increasing rainfall
  - Monsoon (Jun-Sep): 80% of annual precipitation
  - Post-monsoon (Oct-Nov): tapering
  - Winter (Dec-Feb): minimal rainfall (<15mm/month)

Model 18 in the Citadel upgrade plan.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from backend.quant_pro.alpha_practical import AlphaSignal, SignalType
from backend.quant_pro.database import get_db_path

logger = logging.getLogger(__name__)

# Hydropower basin -> ticker mapping (confirmed from NEPSE company prospectuses)
HYDRO_BASINS = {
    "koshi": {
        "tickers": ["UPPER", "NHPC", "AKPL", "AHPC"],
        "desc": "Koshi basin (eastern Nepal)",
    },
    "gandaki": {
        "tickers": ["KGAL", "BARUN", "CHL", "API"],
        "desc": "Gandaki basin (central Nepal)",
    },
    "karnali": {
        "tickers": ["SHPC", "UPCL"],
        "desc": "Karnali basin (western Nepal)",
    },
}

# All hydro tickers across all basins
ALL_HYDRO_TICKERS = []
for _basin_info in HYDRO_BASINS.values():
    ALL_HYDRO_TICKERS.extend(_basin_info["tickers"])

# Ticker -> basin reverse lookup
TICKER_TO_BASIN = {}
for _basin_name, _basin_info in HYDRO_BASINS.items():
    for _ticker in _basin_info["tickers"]:
        TICKER_TO_BASIN[_ticker] = _basin_name

# Monthly rainfall climatology (mm) for Nepal — long-run averages
# Source: DHM Nepal / CHIRPS 1981-2020 averages
MONTHLY_BASELINE_MM = {
    1: 15.0,    # January - dry winter
    2: 25.0,    # February
    3: 35.0,    # March - pre-monsoon starts
    4: 60.0,    # April
    5: 120.0,   # May
    6: 250.0,   # June - monsoon onset
    7: 450.0,   # July - peak monsoon
    8: 380.0,   # August
    9: 250.0,   # September - monsoon retreat
    10: 60.0,   # October
    11: 10.0,   # November
    12: 10.0,   # December - dry winter
}

DEFAULT_DB_PATH = str(get_db_path())


def _get_basin_rainfall(
    db_path: str,
    basin: str,
    end_date: str,
    lookback_days: int = 30,
) -> Optional[pd.DataFrame]:
    """Query weather_data for a basin's recent rainfall.

    Returns DataFrame with columns [date, rainfall_mm, temperature_c]
    or None if insufficient data.
    """
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            """SELECT date, rainfall_mm, temperature_c
               FROM weather_data
               WHERE basin = ? AND date >= ? AND date <= ?
               ORDER BY date""",
            conn,
            params=(basin, start_date, end_date),
        )
        conn.close()

        if df.empty or len(df) < lookback_days * 0.5:
            logger.debug(
                f"Insufficient weather data for {basin}: "
                f"{len(df)} rows (need >= {lookback_days * 0.5:.0f})"
            )
            return None

        return df
    except sqlite3.Error as e:
        logger.warning(f"DB error fetching weather for {basin}: {e}")
        return None


def _compute_rainfall_anomaly(
    rainfall_df: pd.DataFrame,
    lookback_days: int = 30,
) -> float:
    """Compute rainfall anomaly vs monthly climatology.

    Returns fractional anomaly: (actual - baseline) / baseline
    E.g., +0.5 means 50% above normal, -0.3 means 30% below normal.
    """
    if rainfall_df is None or rainfall_df.empty:
        return 0.0

    # Sum actual rainfall over lookback period
    actual_mm = rainfall_df["rainfall_mm"].dropna().sum()

    # Compute expected baseline from monthly climatology
    # Weight each day by its month's baseline
    baseline_mm = 0.0
    for _, row in rainfall_df.iterrows():
        try:
            month = int(row["date"].split("-")[1]) if isinstance(row["date"], str) else row["date"].month
            # Daily baseline = monthly baseline / days in month (~30)
            daily_baseline = MONTHLY_BASELINE_MM.get(month, 30.0) / 30.0
            baseline_mm += daily_baseline
        except (ValueError, IndexError, AttributeError):
            baseline_mm += 1.0  # fallback ~1mm/day

    if baseline_mm <= 0:
        return 0.0

    anomaly = (actual_mm - baseline_mm) / baseline_mm
    return anomaly


def generate_hydro_rainfall_signals_at_date(
    prices_df: pd.DataFrame,
    date: datetime,
    db_path: str = DEFAULT_DB_PATH,
    lookback_days: int = 30,
    anomaly_threshold: float = 0.3,
    liquid_symbols: Optional[List[str]] = None,
) -> List[AlphaSignal]:
    """Generate signals for hydro stocks based on rainfall anomaly.

    Logic:
    1. Query weather_data table for last lookback_days of rainfall per basin.
    2. Compute rolling total rainfall for each basin.
    3. Compare to monthly baseline (climatology).
    4. anomaly = (actual - baseline) / baseline
    5. If anomaly > +30%: bullish for hydro stocks in that basin.
    6. If anomaly < -30%: skip (we are long-only, no short signal).

    Args:
        prices_df: DataFrame with columns [symbol, date, close, volume, ...].
        date: Current backtest/signal date.
        db_path: Path to nepse_market_data.db.
        lookback_days: Number of days for rainfall lookback window.
        anomaly_threshold: Minimum anomaly fraction to emit signal (default 0.3 = 30%).
        liquid_symbols: Optional whitelist of tradeable symbols.

    Returns:
        List of AlphaSignal objects for hydro tickers with above-normal rainfall.

    NO lookahead bias: queries weather data up to and including `date` only.
    """
    signals: List[AlphaSignal] = []
    date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)[:10]

    # Determine which symbols are available in the price data
    available_symbols = set(prices_df["symbol"].unique()) if not prices_df.empty else set()

    for basin_name, basin_info in HYDRO_BASINS.items():
        # Fetch rainfall data for this basin
        rain_df = _get_basin_rainfall(
            db_path=db_path,
            basin=basin_name,
            end_date=date_str,
            lookback_days=lookback_days,
        )

        if rain_df is None:
            continue

        # Compute rainfall anomaly
        anomaly = _compute_rainfall_anomaly(rain_df, lookback_days)

        # Only emit bullish signals (long-only system)
        if anomaly < anomaly_threshold:
            continue

        # Signal strength: scales linearly with anomaly above threshold
        # Cap at 1.0. Anomaly of +100% (double normal) = strength 0.85
        raw_strength = min(anomaly, 2.0) / 2.0  # normalize to 0-1
        strength = 0.3 + 0.7 * raw_strength  # range: 0.3 to 1.0

        # Confidence is higher during monsoon (Jun-Sep) when rainfall
        # is more predictable and more impactful for hydro generation
        month = date.month if isinstance(date, datetime) else int(date_str.split("-")[1])
        if month in (6, 7, 8, 9):
            confidence = 0.55  # monsoon season: stronger signal
        elif month in (5, 10):
            confidence = 0.45  # shoulder months
        else:
            confidence = 0.35  # dry season: rainfall anomaly less meaningful

        # Total rainfall for reasoning string
        total_rain = rain_df["rainfall_mm"].dropna().sum()

        # Emit signal for each hydro ticker in this basin
        for ticker in basin_info["tickers"]:
            # Skip if not in liquid universe or not in price data
            if liquid_symbols and ticker not in liquid_symbols:
                continue
            if ticker not in available_symbols:
                continue

            # Verify ticker has recent price data (not stale)
            # Compare as strings (prices_df["date"] is string YYYY-MM-DD)
            sym_df = prices_df[
                (prices_df["symbol"] == ticker) & (prices_df["date"] <= date_str)
            ]
            if len(sym_df) < 20:
                continue

            signals.append(
                AlphaSignal(
                    symbol=ticker,
                    signal_type=SignalType.SATELLITE_HYDRO,
                    direction=1,
                    strength=min(strength, 1.0),
                    confidence=min(confidence, 1.0),
                    reasoning=(
                        f"Rainfall anomaly +{anomaly:.0%} in {basin_name} basin "
                        f"({total_rain:.0f}mm in {lookback_days}d). "
                        f"Above-normal rain = higher hydro generation."
                    ),
                )
            )

    return signals


def get_basin_rainfall_summary(
    db_path: str = "nepse_market_data.db",
    date_str: Optional[str] = None,
    lookback_days: int = 30,
) -> dict:
    """Get rainfall summary for all basins (useful for monitoring/dashboard).

    Returns dict with per-basin anomaly, total rainfall, and signal direction.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    summary = {}
    for basin_name in list(HYDRO_BASINS.keys()) + ["national_avg"]:
        rain_df = _get_basin_rainfall(db_path, basin_name, date_str, lookback_days)
        if rain_df is None:
            summary[basin_name] = {"status": "no_data"}
            continue

        anomaly = _compute_rainfall_anomaly(rain_df, lookback_days)
        total_mm = rain_df["rainfall_mm"].dropna().sum()
        avg_temp = rain_df["temperature_c"].dropna().mean() if "temperature_c" in rain_df.columns else None

        summary[basin_name] = {
            "total_rainfall_mm": round(total_mm, 1),
            "anomaly_pct": round(anomaly * 100, 1),
            "avg_temperature_c": round(avg_temp, 1) if avg_temp is not None else None,
            "days_with_data": len(rain_df),
            "signal": "bullish" if anomaly > 0.3 else ("neutral" if anomaly > -0.3 else "bearish"),
        }

    return summary


__all__ = [
    "generate_hydro_rainfall_signals_at_date",
    "get_basin_rainfall_summary",
    "HYDRO_BASINS",
    "ALL_HYDRO_TICKERS",
    "TICKER_TO_BASIN",
    "MONTHLY_BASELINE_MM",
]
