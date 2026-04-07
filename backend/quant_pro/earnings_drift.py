"""
Post-Earnings Announcement Drift (PEAD) signal generator.

Paper: Bernard & Thomas (1989)
Key finding: Stocks with positive earnings surprises drift upward for 60+ trading
days after announcement. The effect is strongest in small/mid-cap stocks with
low analyst coverage -- exactly the NEPSE profile.

SUE = (EPS_q - EPS_q-4) / |EPS_q-4|  (Standardized Unexpected Earnings)

NEPSE-specific considerations:
- Quarterly results published 30-45 days after quarter end (slow disclosure)
- Retail market = slow information incorporation = stronger drift
- Use announcement_date to prevent lookahead bias
- When announcement_date is NULL, use conservative estimate: report_date + 30 days
- Signal decays linearly over holding period (strongest right after announcement)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.quant_pro.alpha_practical import AlphaSignal, SignalType
from backend.quant_pro.database import get_db_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default parameters
DEFAULT_SUE_THRESHOLD = 0.10     # 10% earnings surprise minimum to trigger signal
DEFAULT_HOLDING_PERIOD = 60      # Trading days to hold after announcement
DEFAULT_LOOKBACK_DAYS = 90       # Calendar days to look back for recent announcements
DEFAULT_ANNOUNCEMENT_BUFFER = 30 # Calendar days to add when announcement_date is NULL
DEFAULT_DECAY_HALF_LIFE = 30     # Trading days for signal strength half-life


# ---------------------------------------------------------------------------
# Core SUE computation
# ---------------------------------------------------------------------------

def compute_sue(eps_current: float, eps_year_ago: float) -> float:
    """
    Compute Standardized Unexpected Earnings (SUE).

    SUE = (EPS_q - EPS_{q-4}) / |EPS_{q-4}|

    Uses seasonal random walk model (compare to same quarter last year)
    to avoid seasonal earnings patterns.

    Parameters
    ----------
    eps_current : float
        Current quarter EPS (annualized or raw).
    eps_year_ago : float
        Same quarter from previous fiscal year.

    Returns
    -------
    float
        SUE value. Positive = positive surprise, negative = negative surprise.
        Returns 0.0 if prior EPS is zero or None (undefined surprise).
    """
    if eps_year_ago is None or eps_year_ago == 0.0:
        return 0.0
    if eps_current is None:
        return 0.0
    return (eps_current - eps_year_ago) / abs(eps_year_ago)


def compute_eps_growth_qoq(eps_current: float, eps_prev_quarter: float) -> float:
    """
    Compute quarter-over-quarter EPS growth as a simpler alternative to SUE.

    Used when we don't have 4 quarters of history for seasonal comparison.
    """
    if eps_prev_quarter is None or eps_prev_quarter == 0.0:
        return 0.0
    if eps_current is None:
        return 0.0
    return (eps_current - eps_prev_quarter) / abs(eps_prev_quarter)


# ---------------------------------------------------------------------------
# Database queries
# ---------------------------------------------------------------------------

def _get_db_path() -> str:
    """Resolve the database file path."""
    return str(get_db_path())


def _load_recent_earnings(
    db_path: str,
    as_of_date: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    announcement_buffer_days: int = DEFAULT_ANNOUNCEMENT_BUFFER,
) -> pd.DataFrame:
    """
    Load quarterly earnings with announcement dates on or before `as_of_date`.

    Critical: This function enforces NO LOOKAHEAD by filtering on announcement_date.
    - If announcement_date is set: use it directly
    - If announcement_date is NULL: use report_date + buffer (conservative estimate)
    - If both are NULL: use scraped_at_utc as last resort

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    as_of_date : str
        Date string (YYYY-MM-DD) representing the current backtesting date.
    lookback_days : int
        How many calendar days back to look for recent announcements.
    announcement_buffer_days : int
        Days to add to report_date when announcement_date is NULL.

    Returns
    -------
    pd.DataFrame with columns:
        symbol, fiscal_year, quarter, eps, net_profit, revenue, book_value,
        effective_announcement_date
    """
    conn = sqlite3.connect(db_path, timeout=30)

    # Calculate the earliest date to consider
    as_of_dt = pd.Timestamp(as_of_date)
    earliest_date = (as_of_dt - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    query = """
        SELECT
            symbol,
            fiscal_year,
            quarter,
            eps,
            net_profit,
            revenue,
            book_value,
            announcement_date,
            report_date,
            scraped_at_utc
        FROM quarterly_earnings
        ORDER BY symbol, fiscal_year, quarter
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return df

    # Compute effective announcement date (no lookahead)
    def _effective_date(row):
        """Determine when this earnings data became public."""
        if pd.notna(row["announcement_date"]) and row["announcement_date"]:
            return row["announcement_date"]
        if pd.notna(row["report_date"]) and row["report_date"]:
            try:
                rd = pd.Timestamp(row["report_date"])
                return (rd + pd.Timedelta(days=announcement_buffer_days)).strftime("%Y-%m-%d")
            except Exception:
                pass
        # Last resort: scraped_at_utc (when we first saw it)
        if pd.notna(row["scraped_at_utc"]) and row["scraped_at_utc"]:
            try:
                return pd.Timestamp(row["scraped_at_utc"]).strftime("%Y-%m-%d")
            except Exception:
                pass
        return None

    df["effective_announcement_date"] = df.apply(_effective_date, axis=1)

    # Filter: only include earnings that were announced ON or BEFORE as_of_date
    # Convert dates to strings for comparison (effective_announcement_date is str)
    as_of_str = as_of_date.strftime("%Y-%m-%d") if hasattr(as_of_date, 'strftime') else str(as_of_date)[:10]
    earliest_str = earliest_date.strftime("%Y-%m-%d") if hasattr(earliest_date, 'strftime') else str(earliest_date)[:10]
    df = df[
        df["effective_announcement_date"].notna()
        & (df["effective_announcement_date"] <= as_of_str)
        & (df["effective_announcement_date"] >= earliest_str)
    ].copy()

    return df


def _load_all_earnings(db_path: str) -> pd.DataFrame:
    """Load ALL quarterly earnings (for SUE computation, no date filter)."""
    conn = sqlite3.connect(db_path, timeout=30)
    query = """
        SELECT symbol, fiscal_year, quarter, eps, net_profit, revenue, book_value,
               announcement_date, report_date
        FROM quarterly_earnings
        ORDER BY symbol, fiscal_year, quarter
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# SUE computation across all symbols
# ---------------------------------------------------------------------------

def _compute_sue_for_all(
    all_earnings: pd.DataFrame,
) -> Dict[str, Dict[Tuple[str, int], float]]:
    """
    Compute SUE for all symbol/quarter combinations.

    For each (symbol, fiscal_year, quarter), compute:
        SUE = (EPS_q - EPS_{q-4}) / |EPS_{q-4}|

    where EPS_{q-4} is the same quarter from the previous fiscal year.

    Returns
    -------
    dict: {symbol: {(fiscal_year, quarter): sue_value}}
    """
    sue_map: Dict[str, Dict[Tuple[str, int], float]] = {}

    for symbol, group in all_earnings.groupby("symbol"):
        group = group.sort_values(["fiscal_year", "quarter"]).reset_index(drop=True)
        sue_map[symbol] = {}

        for idx, row in group.iterrows():
            fy = row["fiscal_year"]
            q = row["quarter"]
            eps = row["eps"]

            if eps is None or pd.isna(eps):
                continue

            # Find same quarter from previous fiscal year
            # Fiscal years in NEPSE format: "082-083", "2082/2083", etc.
            # We match by quarter number and look for the most recent previous entry
            prior_entries = group[
                (group["quarter"] == q)
                & (group["fiscal_year"] < fy)
                & (group["eps"].notna())
            ]

            if not prior_entries.empty:
                prior_eps = prior_entries.iloc[-1]["eps"]
                sue = compute_sue(eps, prior_eps)
                sue_map[symbol][(fy, q)] = sue
            else:
                # Fallback: QoQ growth if no year-ago comparison available
                prev_q_entries = group[
                    (group.index < idx)
                    & (group["eps"].notna())
                ]
                if not prev_q_entries.empty:
                    prev_eps = prev_q_entries.iloc[-1]["eps"]
                    qoq = compute_eps_growth_qoq(eps, prev_eps)
                    # Mark as QoQ with lower confidence (applied later)
                    sue_map[symbol][(fy, q)] = qoq

    return sue_map


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_pead_signals_at_date(
    prices_df: pd.DataFrame,
    date: str,
    db_path: str = "",
    sue_threshold: float = DEFAULT_SUE_THRESHOLD,
    holding_period: int = DEFAULT_HOLDING_PERIOD,
    liquid_symbols: Optional[List[str]] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> List[AlphaSignal]:
    """
    Generate PEAD signals based on recent earnings surprises.

    Logic:
    1. Query quarterly_earnings for announcements in last `lookback_days` calendar days
    2. For each announcement, compute SUE (vs same quarter last year)
    3. Filter: only use announcements where announcement_date <= date (no lookahead!)
    4. If |SUE| > threshold AND within holding window: emit signal
       - SUE > threshold: BUY signal (positive surprise -> upward drift)
       - SUE < -threshold: AVOID signal (negative surprise -> downward drift)
    5. Signal strength proportional to SUE magnitude
    6. Confidence decays with days since announcement

    Parameters
    ----------
    prices_df : pd.DataFrame
        Price data with DatetimeIndex and columns like 'Close' or per-symbol.
        Used to compute trading days elapsed since announcement.
    date : str
        Current date (YYYY-MM-DD format) for signal generation.
    db_path : str
        Path to SQLite database containing quarterly_earnings table.
    sue_threshold : float
        Minimum |SUE| to trigger a signal. Default 0.10 (10%).
    holding_period : int
        Number of trading days to hold the PEAD signal.
    liquid_symbols : list of str, optional
        If provided, only generate signals for these symbols.
    lookback_days : int
        Calendar days to look back for recent announcements.

    Returns
    -------
    list of AlphaSignal
    """
    signals: List[AlphaSignal] = []

    # Resolve DB path
    if not db_path or db_path == "nepse_market_data.db":
        db_path = _get_db_path()

    # Check if table exists
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='quarterly_earnings'"
        )
        if not cursor.fetchone():
            conn.close()
            logger.debug("quarterly_earnings table does not exist yet")
            return signals
        conn.close()
    except sqlite3.Error:
        return signals

    # Load recent earnings (already filtered for no-lookahead)
    recent = _load_recent_earnings(db_path, date, lookback_days)

    if recent.empty:
        logger.debug("No recent earnings announcements found for date %s", date)
        return signals

    # Load ALL earnings for SUE computation
    all_earnings = _load_all_earnings(db_path)
    sue_map = _compute_sue_for_all(all_earnings)

    # Build trading day index from prices_df
    if hasattr(prices_df, "index") and isinstance(prices_df.index, pd.DatetimeIndex):
        trading_dates = prices_df.index
    else:
        # Try to parse dates from the dataframe
        try:
            trading_dates = pd.DatetimeIndex(prices_df.index)
        except Exception:
            trading_dates = None

    date_ts = pd.Timestamp(date)

    for _, row in recent.iterrows():
        symbol = row["symbol"]

        # Filter by liquid symbols if provided
        if liquid_symbols is not None and symbol not in liquid_symbols:
            continue

        fy = row["fiscal_year"]
        q = row["quarter"]
        eps = row["eps"]

        if eps is None or pd.isna(eps):
            continue

        # Get SUE for this quarter
        sue = sue_map.get(symbol, {}).get((fy, q))

        if sue is None:
            # If we only have one quarter, use EPS level as a rough signal
            # Positive EPS = mild positive signal for NEPSE where many stocks have negative EPS
            if eps > 0:
                sue = 0.05  # Mild positive
            else:
                sue = -0.05  # Mild negative

        # Check threshold
        if abs(sue) < sue_threshold:
            continue

        # Compute days since announcement
        ann_date_str = row["effective_announcement_date"]
        try:
            ann_date = pd.Timestamp(ann_date_str)
        except Exception:
            continue

        # Count trading days since announcement
        if trading_dates is not None and len(trading_dates) > 0:
            # Count trading days between announcement and current date
            mask = (trading_dates >= ann_date) & (trading_dates <= date_ts)
            trading_days_elapsed = int(mask.sum())
        else:
            # Approximate: calendar days * 5/7
            calendar_days = (date_ts - ann_date).days
            trading_days_elapsed = int(calendar_days * 5 / 7)

        # Skip if outside holding period
        if trading_days_elapsed > holding_period:
            continue

        # Skip if announcement is in the future (shouldn't happen due to earlier filter, but safety)
        if trading_days_elapsed < 0:
            continue

        # Compute signal strength and confidence
        # Strength: proportional to |SUE|, capped at 1.0
        raw_strength = min(abs(sue), 2.0) / 2.0  # Normalize: SUE of 2.0 -> strength 1.0

        # Confidence: decays linearly with time since announcement
        # Full confidence on day 0, zero confidence at holding_period
        if holding_period > 0:
            time_decay = max(0.0, 1.0 - (trading_days_elapsed / holding_period))
        else:
            time_decay = 1.0

        # Base confidence: 0.5 (earnings signals are moderately reliable)
        confidence = 0.5 * time_decay

        # Boost confidence if we have full financial data (from ShareSansar)
        if pd.notna(row.get("net_profit")) and pd.notna(row.get("revenue")):
            confidence = min(confidence + 0.1, 0.8)

        # Direction: positive SUE -> buy, negative SUE -> avoid (can't short in NEPSE)
        direction = 1 if sue > 0 else -1

        # For NEPSE (long-only), only emit buy signals for positive surprises
        # Negative surprises are still useful as "avoid" signals
        if direction == 1:
            reasoning = (
                f"PEAD: Positive earnings surprise "
                f"(SUE={sue:.1%}, EPS={eps:.2f}, FY{fy} Q{q}). "
                f"{trading_days_elapsed}d since announcement. "
                f"Drift window: {holding_period - trading_days_elapsed}d remaining."
            )
        else:
            reasoning = (
                f"PEAD: Negative earnings surprise "
                f"(SUE={sue:.1%}, EPS={eps:.2f}, FY{fy} Q{q}). "
                f"{trading_days_elapsed}d since announcement. "
                f"AVOID for {holding_period - trading_days_elapsed}d."
            )

        # Compute expiry date
        try:
            expiry = ann_date + timedelta(days=int(holding_period * 7 / 5))  # Approx trading days to calendar
            expiry_dt = expiry.to_pydatetime() if hasattr(expiry, "to_pydatetime") else expiry
        except Exception:
            expiry_dt = None

        signals.append(AlphaSignal(
            symbol=symbol,
            signal_type=SignalType.EARNINGS_DRIFT,
            direction=direction,
            strength=raw_strength,
            confidence=confidence,
            reasoning=reasoning,
            expires=expiry_dt,
        ))

    # Sort by score (strength * confidence * direction) descending
    signals.sort(key=lambda s: s.score, reverse=True)

    logger.info(
        "PEAD signals at %s: %d total (%d buy, %d avoid)",
        date,
        len(signals),
        sum(1 for s in signals if s.direction > 0),
        sum(1 for s in signals if s.direction < 0),
    )

    return signals


# ---------------------------------------------------------------------------
# Convenience: SUE summary for all stocks
# ---------------------------------------------------------------------------

def get_sue_summary(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Compute SUE for all stocks with sufficient quarterly earnings data.

    Returns DataFrame with: symbol, fiscal_year, quarter, eps, sue, eps_year_ago
    """
    db_path = db_path or _get_db_path()
    all_earnings = _load_all_earnings(db_path)

    if all_earnings.empty:
        return pd.DataFrame()

    sue_map = _compute_sue_for_all(all_earnings)

    rows = []
    for symbol, quarters in sue_map.items():
        for (fy, q), sue in quarters.items():
            eps_row = all_earnings[
                (all_earnings["symbol"] == symbol)
                & (all_earnings["fiscal_year"] == fy)
                & (all_earnings["quarter"] == q)
            ]
            eps_val = eps_row["eps"].iloc[0] if not eps_row.empty else None
            rows.append({
                "symbol": symbol,
                "fiscal_year": fy,
                "quarter": q,
                "eps": eps_val,
                "sue": sue,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Module-level test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick test
    db_path = _get_db_path()
    print(f"DB path: {db_path}")

    # Check if we have earnings data
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COUNT(DISTINCT symbol) FROM quarterly_earnings")
        total, symbols = cursor.fetchone()
        print(f"quarterly_earnings: {total} rows, {symbols} symbols")

        if total > 0:
            # Generate signals for today
            today = datetime.now().strftime("%Y-%m-%d")
            prices_df = pd.DataFrame(index=pd.date_range("2020-01-01", today, freq="B"))
            signals = generate_pead_signals_at_date(prices_df, today, db_path)
            print(f"\nPEAD signals: {len(signals)}")
            for s in signals[:5]:
                print(f"  {s.symbol}: dir={s.direction}, str={s.strength:.3f}, "
                      f"conf={s.confidence:.3f}, score={s.score:.4f}")
                print(f"    {s.reasoning}")

        conn.close()
    except sqlite3.OperationalError as e:
        print(f"Table not found: {e}")
        print("Run the earnings scraper first: python -m backend.quant_pro.earnings_scraper")
