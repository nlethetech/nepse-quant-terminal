#!/usr/bin/env python3
"""
NEPSE Daily Signal Generator

Generates buy_orders.csv for paper_trade_tracker.py using proven alpha signals:
1. Corporate Actions (dividend/bonus pre-event plays)
2. Momentum (SMA crossovers with volume confirmation)
3. Volume Breakouts (liquidity spikes)
4. Fundamentals (P/E discounts)

Usage:
    python -m scripts.signals.generate_daily_signals --capital 1000000 --max-positions 7

Output:
    buy_orders.csv with columns: Symbol, Shares, Weight, Signal_Type, Confidence, Sector
"""

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Local imports
from backend.quant_pro.alpha_practical import (
    AlphaAggregator,
    AlphaSignal,
    CorporateAction,
    CorporateActionScanner,
    FundamentalData,
    FundamentalScanner,
    MomentumScanner,
    LiquidityScanner,
    SignalType,
)
from backend.quant_pro.sectors import SECTOR_GROUPS
from backend.quant_pro.database import get_db_connection, get_db_path
from backend.quant_pro.exceptions import DataStalenessError
from backend.quant_pro.paths import ensure_dir, get_data_dir, get_trading_runtime_dir
from backend.quant_pro.config import (
    TRAILING_STOP_PCT,
    HARD_STOP_LOSS_PCT,
    DEFAULT_CAPITAL,
    MAX_POSITIONS,
    RECOMMENDED_HOLDING_DAYS,
    PORTFOLIO_METHOD,
    CALENDAR_WEDNESDAY_BOOST,
    CALENDAR_THURSDAY_BOOST,
    CALENDAR_SUNDAY_PENALTY,
    CALENDAR_DASHAIN_RALLY_BOOST,
    CALENDAR_PRE_HOLIDAY_PENALTY,
    CALENDAR_DASHAIN_PRE_DAYS,
    CALENDAR_DASHAIN_SELLOFF_DAYS,
    CALENDAR_POST_DASHAIN_CORRECTION_DAYS,
    CALENDAR_POST_DASHAIN_PENALTY,
)
from backend.quant_pro.disposition import generate_cgo_signals_at_date
from backend.quant_pro.earnings_drift import generate_pead_signals_at_date
from backend.quant_pro.lead_lag import generate_sector_leadlag_signals_at_date
from backend.quant_pro.macro_signals import get_nrb_policy_regime
from backend.quant_pro.nepse_calendar import (
    is_today_trading_day,
    next_trading_day,
    is_dashain_period,
    days_until_dashain,
)

# Deferred imports: only loaded if data is available (fail-open pattern)
# from backend.backtesting.simple_backtest import generate_52wk_high_signals_at_date, apply_amihud_tilt

# Configuration
OUTPUT_FILE = ensure_dir(get_trading_runtime_dir(__file__)) / "buy_orders.csv"
QUARTERLY_REPORTS_DIR = get_data_dir(__file__) / "quarterly_reports"

# Risk limits (DEFAULT_CAPITAL, MAX_POSITIONS imported from config)
MAX_SINGLE_POSITION_PCT = 0.15  # 15% max per stock
MAX_SECTOR_PCT = 0.35  # 35% max per sector
MIN_TURNOVER_NPR = 500_000  # Minimum daily turnover for liquidity
MIN_CONFIDENCE = 0.45  # Raised back to 0.45 for better signal quality

# Strategy Configuration (based on cross-regime backtest 2022-2025)
# Volume breakouts = best average Sharpe (0.05) across all regimes
# Quality signals = best in neutral/sideways markets (Sharpe 1.75)
# No strategy works perfectly in bear markets - regime filter is critical
# RECOMMENDED_HOLDING_DAYS imported from config
ENABLE_MOMENTUM_SIGNALS = False  # Disabled - negative alpha
ENABLE_QUALITY_SIGNALS = True  # Added for better neutral market performance
VOLUME_SPIKE_MULTIPLIER = 2.5

# Risk Management (TRAILING_STOP_PCT, HARD_STOP_LOSS_PCT imported from config)
USE_TRAILING_STOP = True

# Market Regime Filter
# CRITICAL: All strategies lose money in bear markets
# Regime filter reduces exposure during downturns
ENABLE_REGIME_FILTER = True
REGIME_LOOKBACK_DAYS = 20
ENABLE_BOCPD_REGIME = True  # BOCPD changepoint detector as downgrade-only overlay

# Data freshness guardrails
DB_STALENESS_WARN_DAYS = 5
DB_STALENESS_HARD_FAIL_DAYS = 14

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _safe_float(value) -> Optional[float]:
    """Best-effort numeric coercion for mixed DB / OCR cache payloads."""
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _annualize_eps(eps: Optional[float], quarter: Optional[int]) -> Optional[float]:
    if eps is None:
        return None
    if not quarter or quarter <= 0:
        return eps
    return eps * (4.0 / quarter)


def _growth_ratio(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous in (None, 0):
        return None
    return (current - previous) / abs(previous)


def _fiscal_digits(raw: str) -> Tuple[int, int]:
    digits = [int(chunk) for chunk in "".join(ch if ch.isdigit() else " " for ch in str(raw)).split()]
    if not digits:
        return (0, 0)
    if len(digits) == 1:
        return (digits[0], 0)
    return (digits[0], digits[1])


def _quarter_sort_key(fiscal_year: str, quarter: Optional[int], fallback_date: str = "") -> Tuple[int, int, str]:
    left, right = _fiscal_digits(fiscal_year)
    return (left, right + int(quarter or 0), str(fallback_date or ""))


def _load_latest_quarterly_report_metrics(symbol: str) -> Dict[str, Optional[float]]:
    """Load the newest cached quarterly-report metrics for a symbol, if available."""
    cache_path = QUARTERLY_REPORTS_DIR / f"{symbol}.json"
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text())
    except Exception:
        return {}

    reports = []
    for report in payload.get("reports", []):
        financials = report.get("financials") or {}
        if not financials or "error" in financials:
            continue
        quarter_raw = str(financials.get("quarter") or "")
        quarter_digits = "".join(ch for ch in quarter_raw if ch.isdigit())
        quarter = int(quarter_digits) if quarter_digits else None
        reports.append(
            {
                "fiscal_year": str(financials.get("fiscal_year") or ""),
                "quarter": quarter,
                "report_date": str(report.get("date") or report.get("announcement_date") or ""),
                "eps": _safe_float((financials.get("per_share") or {}).get("eps")),
                "book_value": _safe_float((financials.get("per_share") or {}).get("book_value")),
                "revenue": _safe_float((financials.get("income_statement") or {}).get("total_revenue")),
                "net_profit": _safe_float((financials.get("income_statement") or {}).get("net_profit")),
                "equity": _safe_float((financials.get("balance_sheet") or {}).get("shareholders_equity")),
                "liabilities": _safe_float((financials.get("balance_sheet") or {}).get("total_liabilities")),
                "npl_pct": _safe_float((financials.get("ratios") or {}).get("npl_pct")),
                "capital_adequacy_pct": _safe_float((financials.get("ratios") or {}).get("capital_adequacy_pct")),
                "cost_income_ratio": _safe_float((financials.get("ratios") or {}).get("cost_income_ratio")),
                "sector": str(financials.get("sector") or "").strip(),
            }
        )

    if not reports:
        return {}

    reports.sort(
        key=lambda row: _quarter_sort_key(row["fiscal_year"], row["quarter"], row["report_date"]),
        reverse=True,
    )
    return reports[0]


def get_latest_market_date(conn: sqlite3.Connection) -> Optional[datetime]:
    """Return latest date available in stock_prices, or None."""
    query = "SELECT MAX(date) AS last_date FROM stock_prices"
    row = pd.read_sql_query(query, conn)
    if row.empty or pd.isna(row.loc[0, "last_date"]):
        return None
    return pd.to_datetime(row.loc[0, "last_date"])


def validate_data_freshness(conn: sqlite3.Connection) -> bool:
    """
    Validate DB freshness against wall clock.

    Returns:
        True if data can be used.

    Raises:
        DataStalenessError: if no data or data is too stale.
    """
    latest_date = get_latest_market_date(conn)
    if latest_date is None:
        raise DataStalenessError("No market data found in stock_prices")

    staleness_days = (datetime.now().date() - latest_date.date()).days
    if staleness_days > DB_STALENESS_HARD_FAIL_DAYS:
        raise DataStalenessError(
            f"Market data is {staleness_days} days stale (max {DB_STALENESS_HARD_FAIL_DAYS}). Last data: {latest_date.date()}"
        )

    if staleness_days > DB_STALENESS_WARN_DAYS:
        logger.warning(
            "Market data is stale (last=%s, staleness=%sd). Signals may be outdated.",
            latest_date.date(),
            staleness_days,
        )
    else:
        logger.info("Market data freshness OK (last=%s)", latest_date.date())

    return True


def get_symbol_sector(symbol: str) -> str:
    """Look up sector for a symbol."""
    for sector, symbols in SECTOR_GROUPS.items():
        if symbol in symbols:
            return sector
    return "Others"


def get_calendar_multiplier(dt) -> float:
    """
    Return signal confidence multiplier based on NEPSE calendar effects.

    Academic evidence (NRB Economic Review, NepJOL):
    - Wednesday and Thursday show statistically significant positive returns
    - Sunday (first day of NEPSE week) shows slight negative bias
    - Pre-Dashain rally period (2-3 weeks before) shows elevated returns
    - Last 3 days before Dashain show pre-holiday selloff

    Args:
        dt: date or datetime object

    Returns:
        Multiplier in approximately [0.90, 1.12] to scale signal confidence.
    """
    if hasattr(dt, 'date'):
        dt = dt.date()

    multiplier = 1.0
    dow = dt.weekday()  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun

    # NEPSE trading days: Sun(6), Mon(0), Tue(1), Wed(2), Thu(3)
    if dow == 2:  # Wednesday
        multiplier *= CALENDAR_WEDNESDAY_BOOST
    elif dow == 3:  # Thursday
        multiplier *= CALENDAR_THURSDAY_BOOST
    elif dow == 6:  # Sunday
        multiplier *= CALENDAR_SUNDAY_PENALTY

    # Dashain rally period and post-Dashain correction
    d = days_until_dashain(dt)
    if d is not None and 0 < d <= CALENDAR_DASHAIN_PRE_DAYS:
        if d > CALENDAR_DASHAIN_SELLOFF_DAYS:
            multiplier *= CALENDAR_DASHAIN_RALLY_BOOST
        else:
            multiplier *= CALENDAR_PRE_HOLIDAY_PENALTY
    elif d is not None and d <= 0:
        # Dashain has passed — apply correction window penalty
        # Research: NEPSE declined in the first week after Dashain 6 consecutive years
        days_after = abs(d)
        if days_after <= CALENDAR_POST_DASHAIN_CORRECTION_DAYS:
            multiplier *= CALENDAR_POST_DASHAIN_PENALTY

    return multiplier


def calculate_market_regime(prices: pd.DataFrame, lookback: int = REGIME_LOOKBACK_DAYS) -> dict:
    """
    Calculate market regime based on breadth and momentum.

    Returns dict with:
        - regime: 'bull', 'bear', or 'neutral'
        - confidence_multiplier: 0.5 to 1.2 (scale signals based on regime)
        - breadth: % of stocks above their 20-day SMA
        - momentum: average 20-day return across stocks
    """
    if prices.empty or len(prices) < lookback:
        return {"regime": "neutral", "confidence_multiplier": 1.0, "breadth": 0.5, "momentum": 0.0}

    # Calculate breadth: % of stocks above their 20-day SMA
    sma = prices.rolling(lookback).mean()
    above_sma = (prices.iloc[-1] > sma.iloc[-1]).mean()

    # Calculate average momentum
    if len(prices) >= lookback:
        returns = (prices.iloc[-1] / prices.iloc[-lookback] - 1)
        avg_momentum = returns.mean()
    else:
        avg_momentum = 0.0

    # Determine regime
    if above_sma > 0.6 and avg_momentum > 0.02:
        regime = "bull"
        multiplier = 1.15  # Boost confidence in bull market
    elif above_sma < 0.4 or avg_momentum < -0.05:
        regime = "bear"
        multiplier = 0.6  # Reduce confidence in bear market
    else:
        regime = "neutral"
        multiplier = 0.9  # Slightly cautious in neutral

    return {
        "regime": regime,
        "confidence_multiplier": multiplier,
        "breadth": above_sma,
        "momentum": avg_momentum,
    }


def calculate_market_regime_with_bocpd(
    prices: pd.DataFrame,
    lookback: int = REGIME_LOOKBACK_DAYS,
    bocpd_threshold: float = 0.5,
    hazard_lambda: int = 60,
) -> dict:
    """Enhanced regime detection combining breadth/momentum with BOCPD changepoint detection.

    BOCPD (Bayesian Online Changepoint Detection, Adams & MacKay 2007) acts as a
    downgrade-only overlay on the base regime — it can move bull→neutral or
    neutral→bear when it detects a structural break, but NEVER upgrades the regime.
    This is conservative by design: we'd rather miss a bull entry than ride a bear.

    hazard_lambda=60: expects regime changes every ~60 trading days (~3 months).
    bocpd_threshold=0.5: changepoint must be more likely than not before overriding.

    Falls back to base regime on any exception.
    """
    base = calculate_market_regime(prices, lookback)

    try:
        from backend.quant_pro.regime_detection import BOCPDDetector

        if prices.empty or len(prices) < 40:
            return base

        # Use median return across all stocks as broad market proxy
        market_returns = prices.pct_change().dropna().median(axis=1).values

        detector = BOCPDDetector(hazard_lambda=hazard_lambda)
        cp_prob = 0.0
        for obs in market_returns:
            cp_prob = detector.update(float(obs))

        if cp_prob >= bocpd_threshold:
            current = base["regime"]
            if current == "bull":
                base["regime"] = "neutral"
                base["confidence_multiplier"] = min(base["confidence_multiplier"], 0.90)
                logger.info(
                    f"BOCPD changepoint p={cp_prob:.2f}: regime downgraded bull→neutral"
                )
            elif current == "neutral":
                base["regime"] = "bear"
                base["confidence_multiplier"] = min(base["confidence_multiplier"], 0.60)
                logger.info(
                    f"BOCPD changepoint p={cp_prob:.2f}: regime downgraded neutral→bear"
                )

    except Exception as e:
        logger.warning(f"BOCPD regime detection failed, using base regime: {e}")

    return base


def get_liquid_universe(conn: sqlite3.Connection, min_days: int = 60, recent_days: int = 14) -> List[str]:
    """Get symbols with sufficient trading history and liquidity."""
    query = """
        SELECT symbol,
               COUNT(*) as days,
               AVG(close * volume) as avg_turnover,
               MAX(date) as last_date
        FROM stock_prices
        WHERE volume > 0
        GROUP BY symbol
        HAVING days >= ? AND avg_turnover >= ?
        ORDER BY avg_turnover DESC
    """
    df = pd.read_sql_query(query, conn, params=(min_days, MIN_TURNOVER_NPR))

    # Filter to symbols that have traded recently relative to the latest
    # known session in the DB (not relative to wall-clock time).
    latest_market_date = get_latest_market_date(conn)
    if latest_market_date is not None:
        recent_cutoff = (latest_market_date - timedelta(days=recent_days)).strftime("%Y-%m-%d")
        df = df[df["last_date"] >= recent_cutoff]

    logger.info(f"Found {len(df)} liquid symbols")
    return df["symbol"].tolist()


def load_price_data(conn: sqlite3.Connection, symbols: List[str], days: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load price and volume data for symbols."""
    if not symbols:
        return pd.DataFrame(), pd.DataFrame()

    placeholders = ",".join(["?"] * len(symbols))
    query = f"""
        WITH ranked AS (
            SELECT
                symbol,
                date,
                close,
                volume,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
            FROM stock_prices
            WHERE symbol IN ({placeholders})
        )
        SELECT symbol, date, close, volume
        FROM ranked
        WHERE rn <= ?
        ORDER BY date ASC
    """
    params = [*symbols, days]
    df = pd.read_sql_query(query, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])

    # Pivot to get symbols as columns
    prices = df.pivot(index="date", columns="symbol", values="close").sort_index()
    volumes = df.pivot(index="date", columns="symbol", values="volume").sort_index()

    return prices, volumes


def load_price_data_long(
    conn: sqlite3.Connection,
    symbols: List[str],
    days: int = 300,
) -> pd.DataFrame:
    """Load OHLCV data in long format (symbol, date, open, high, low, close, volume).

    Required by CGO (260-day VWAP lookback), lead-lag (sector correlation),
    52-week high (252-day high), and Amihud tilt signals.
    Returns empty DataFrame on failure — all callers handle this gracefully.
    """
    if not symbols:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(symbols))
    query = f"""
        WITH ranked AS (
            SELECT symbol, date, open, high, low, close, volume,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
            FROM stock_prices
            WHERE symbol IN ({placeholders})
        )
        SELECT symbol, date, open, high, low, close, volume
        FROM ranked
        WHERE rn <= ?
        ORDER BY symbol, date ASC
    """
    try:
        df = pd.read_sql_query(query, conn, params=[*symbols, days])
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        logger.warning(f"load_price_data_long failed: {e}")
        return pd.DataFrame()


def load_fundamentals(
    conn: sqlite3.Connection,
    symbols: List[str],
    latest_prices: Optional[Dict[str, float]] = None,
) -> Dict[str, FundamentalData]:
    """Load fundamental data for symbols (if available)."""
    fundamentals = {}
    latest_prices = latest_prices or {}

    # Try to get fundamental data from the database
    try:
        query = """
            SELECT symbol, pe_ratio, pb_ratio, dividend_yield, eps, book_value_per_share, roe
            FROM fundamentals
            WHERE symbol IN ({})
        """.format(",".join(["?"] * len(symbols)))
        df = pd.read_sql_query(query, conn, params=symbols)

        for _, row in df.iterrows():
            fundamentals[row["symbol"]] = FundamentalData(
                symbol=row["symbol"],
                sector=get_symbol_sector(row["symbol"]),
                pe_ratio=row.get("pe_ratio"),
                pb_ratio=row.get("pb_ratio"),
                dividend_yield=row.get("dividend_yield"),
                eps=row.get("eps"),
                book_value=row.get("book_value_per_share"),
                roe=row.get("roe"),
            )
    except Exception as e:
        logger.warning(f"Could not load fundamentals: {e}")

    # Enrich with the most recent two quarters from quarterly_earnings.
    try:
        query = """
            WITH ranked AS (
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
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol
                        ORDER BY
                            COALESCE(announcement_date, report_date, '') DESC,
                            fiscal_year DESC,
                            quarter DESC
                    ) AS rn
                FROM quarterly_earnings
                WHERE symbol IN ({})
            )
            SELECT *
            FROM ranked
            WHERE rn <= 2
            ORDER BY symbol, rn ASC
        """.format(",".join(["?"] * len(symbols)))
        q_df = pd.read_sql_query(query, conn, params=symbols)
        for symbol, group in q_df.groupby("symbol"):
            latest = group.iloc[0]
            previous = group.iloc[1] if len(group) > 1 else None
            fd = fundamentals.get(symbol) or FundamentalData(
                symbol=symbol,
                sector=get_symbol_sector(symbol),
            )

            q_num = int(latest["quarter"]) if pd.notna(latest["quarter"]) else None
            annualized_eps = _annualize_eps(_safe_float(latest.get("eps")), q_num)
            book_value = _safe_float(latest.get("book_value"))
            last_price = _safe_float(latest_prices.get(symbol))
            if annualized_eps and annualized_eps > 0 and last_price and last_price > 0:
                fd.pe_ratio = last_price / annualized_eps
            if book_value and book_value > 0 and last_price and last_price > 0:
                fd.pb_ratio = last_price / book_value

            fd.eps = annualized_eps or fd.eps
            fd.book_value = book_value or fd.book_value
            fd.latest_net_profit = _safe_float(latest.get("net_profit"))
            fd.latest_revenue = _safe_float(latest.get("revenue"))
            fd.revenue_growth_qoq = _growth_ratio(
                fd.latest_revenue,
                _safe_float(previous.get("revenue")) if previous is not None else None,
            )
            fd.profit_growth_qoq = _growth_ratio(
                fd.latest_net_profit,
                _safe_float(previous.get("net_profit")) if previous is not None else None,
            )
            fd.eps_growth_qoq = _growth_ratio(
                _safe_float(latest.get("eps")),
                _safe_float(previous.get("eps")) if previous is not None else None,
            )
            fd.book_value_growth_qoq = _growth_ratio(
                book_value,
                _safe_float(previous.get("book_value")) if previous is not None else None,
            )
            if fd.data_source:
                fd.data_source = f"{fd.data_source}+quarterly_earnings"
            else:
                fd.data_source = "quarterly_earnings"
            fundamentals[symbol] = fd
    except Exception as e:
        logger.warning(f"Could not load quarterly earnings enrichment: {e}")

    # Overlay richer balance-sheet / asset-quality metrics from the quarterly-report cache.
    for symbol in symbols:
        latest_report = _load_latest_quarterly_report_metrics(symbol)
        if not latest_report:
            continue
        fd = fundamentals.get(symbol) or FundamentalData(
            symbol=symbol,
            sector=get_symbol_sector(symbol),
        )

        q_num = latest_report.get("quarter")
        annualized_eps = _annualize_eps(latest_report.get("eps"), q_num if isinstance(q_num, int) else None)
        book_value = latest_report.get("book_value")
        last_price = _safe_float(latest_prices.get(symbol))
        if annualized_eps and annualized_eps > 0 and last_price and last_price > 0:
            fd.pe_ratio = last_price / annualized_eps
        if book_value and book_value > 0 and last_price and last_price > 0:
            fd.pb_ratio = last_price / book_value
        if annualized_eps is not None:
            fd.eps = annualized_eps
        if book_value is not None:
            fd.book_value = book_value
        if latest_report.get("revenue") is not None:
            fd.latest_revenue = latest_report.get("revenue")
        if latest_report.get("net_profit") is not None:
            fd.latest_net_profit = latest_report.get("net_profit")
        equity = latest_report.get("equity")
        liabilities = latest_report.get("liabilities")
        if equity not in (None, 0) and liabilities is not None:
            fd.debt_to_equity = liabilities / equity
        fd.npl_pct = latest_report.get("npl_pct")
        fd.capital_adequacy_pct = latest_report.get("capital_adequacy_pct")
        fd.cost_income_ratio = latest_report.get("cost_income_ratio")
        if latest_report.get("sector"):
            fd.sector = latest_report["sector"]
        if fd.data_source:
            fd.data_source = f"{fd.data_source}+quarterly_cache"
        else:
            fd.data_source = "quarterly_cache"
        fundamentals[symbol] = fd

    return fundamentals


def load_corporate_actions(conn: sqlite3.Connection) -> List[CorporateAction]:
    """Load upcoming corporate actions."""
    actions = []
    today = datetime.now().date()
    future_cutoff = today + timedelta(days=45)

    try:
        query = """
            SELECT symbol, fiscal_year, bookclose_date,
                   cash_dividend_pct, bonus_share_pct, right_share_ratio, agenda
            FROM corporate_actions
            WHERE bookclose_date >= ? AND bookclose_date <= ?
            ORDER BY bookclose_date
        """
        df = pd.read_sql_query(query, conn, params=(str(today), str(future_cutoff)))

        for _, row in df.iterrows():
            # Determine action type
            if row.get("right_share_ratio"):
                action_type = "rights"
            elif row.get("bonus_share_pct") and row["bonus_share_pct"] > 0:
                action_type = "bonus"
            elif row.get("cash_dividend_pct") and row["cash_dividend_pct"] > 0:
                action_type = "dividend"
            else:
                action_type = "agm"

            try:
                record_date = datetime.strptime(str(row["bookclose_date"]), "%Y-%m-%d")
            except (ValueError, TypeError) as e:
                logger.warning("Skipping malformed corporate action: %s", e)
                continue

            actions.append(CorporateAction(
                symbol=row["symbol"],
                action_type=action_type,
                announce_date=datetime.now(),
                record_date=record_date,
                details=row.get("agenda") or f"{action_type} - FY {row.get('fiscal_year', 'N/A')}",
            ))

        logger.info(f"Found {len(actions)} upcoming corporate actions")
    except Exception as e:
        logger.warning(f"Could not load corporate actions: {e}")

    return actions


def calculate_position_sizes(
    signals: List[AlphaSignal],
    capital: float,
    prices: pd.DataFrame,
    max_positions: int = MAX_POSITIONS,
    prices_long: Optional[pd.DataFrame] = None,
    portfolio_method: str = "equal_weight",
    signal_date=None,
) -> pd.DataFrame:
    """Convert signals to position sizes with risk limits.

    If portfolio_method != "equal_weight" and prices_long is provided, uses
    HRP/CVaR to compute correlation-aware weights. Falls back to equal-weight
    on any failure so buy_orders.csv is always produced.
    """
    if not signals:
        return pd.DataFrame()

    # Get latest prices
    latest_prices = prices.iloc[-1].to_dict()

    # Calculate raw allocations
    allocations = []
    for signal in signals:
        if signal.symbol not in latest_prices:
            continue

        price = latest_prices[signal.symbol]
        if pd.isna(price) or price <= 0:
            continue

        # Position size based on signal strength and confidence
        raw_weight = signal.strength * signal.confidence
        capped_weight = min(raw_weight, MAX_SINGLE_POSITION_PCT)

        allocations.append({
            "Symbol": signal.symbol,
            "Signal_Type": signal.signal_type.value,
            "Strength": signal.strength,
            "Confidence": signal.confidence,
            "Raw_Weight": raw_weight,
            "Weight": capped_weight,
            "Price": price,
            "Sector": get_symbol_sector(signal.symbol),
            "Reasoning": signal.reasoning,
        })

    if not allocations:
        return pd.DataFrame()

    df = pd.DataFrame(allocations)

    # Sort by score (strength * confidence)
    df["Score"] = df["Strength"] * df["Confidence"]
    df = df.sort_values("Score", ascending=False)

    # Apply sector limits
    sector_totals: Dict[str, float] = {}
    keep_rows = []

    for idx, row in df.iterrows():
        sector = row["Sector"]
        current_sector = sector_totals.get(sector, 0.0)

        # Check if adding this position exceeds sector limit
        if current_sector + row["Weight"] > MAX_SECTOR_PCT:
            # Reduce weight to fit
            available = MAX_SECTOR_PCT - current_sector
            if available > 0.05:  # At least 5% allocation
                row["Weight"] = available
            else:
                continue

        sector_totals[sector] = current_sector + row["Weight"]
        keep_rows.append(row)

        if len(keep_rows) >= max_positions:
            break

    if not keep_rows:
        return pd.DataFrame()

    df = pd.DataFrame(keep_rows)

    # HRP/CVaR portfolio construction — replace equal-weight with correlation-aware sizing
    if portfolio_method != "equal_weight" and prices_long is not None and not prices_long.empty and len(df) > 1:
        try:
            from backend.quant_pro.portfolio_construction import allocate_portfolio
            syms = df["Symbol"].tolist()
            sd = signal_date if signal_date is not None else pd.Timestamp.now()
            hrp_alloc = allocate_portfolio(
                method=portfolio_method,
                prices_df=prices_long,
                symbols=syms,
                date=sd,
                capital=capital * 0.80,  # allocate 80% to match cash reserve below
            )
            if hrp_alloc:
                for i, row in df.iterrows():
                    sym = row["Symbol"]
                    if sym in hrp_alloc and hrp_alloc[sym] > 0:
                        df.at[i, "Weight"] = min(hrp_alloc[sym] / capital, MAX_SINGLE_POSITION_PCT)
                logger.info(f"HRP allocation applied via method='{portfolio_method}'")
        except Exception as e:
            logger.warning(f"HRP allocation failed, falling back to equal-weight: {e}")

    # Normalize weights to sum to ~80% (keep 20% cash)
    total_weight = df["Weight"].sum()
    if total_weight > 0.80:
        df["Weight"] = df["Weight"] * 0.80 / total_weight

    # Calculate shares
    df["Value"] = capital * df["Weight"]
    df["Shares"] = (df["Value"] / df["Price"]).astype(int)

    # Recalculate actual values
    df["Value"] = df["Shares"] * df["Price"]
    df["Weight"] = df["Value"] / capital

    # Format output
    df["Hold_Days"] = RECOMMENDED_HOLDING_DAYS  # Add holding period recommendation
    output_cols = ["Symbol", "Shares", "Weight", "Signal_Type", "Confidence", "Sector", "Price", "Value", "Hold_Days", "Reasoning"]
    df = df[output_cols].copy()
    df["Weight"] = df["Weight"].apply(lambda x: f"{x:.1%}")
    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.1%}")
    df["Value"] = df["Value"].apply(lambda x: f"{x:,.0f}")

    return df


def generate_signals(
    capital: float = DEFAULT_CAPITAL,
    max_positions: int = MAX_POSITIONS,
    output_file: str = OUTPUT_FILE,
    dry_run: bool = False,
    allow_stale: bool = False,
) -> pd.DataFrame:
    """Generate daily signals and output buy_orders.csv."""
    logger.info(f"Generating signals with capital={capital:,.0f}, max_positions={max_positions}")

    # Connect to database
    if not get_db_path().exists():
        logger.error(f"Database not found: {get_db_path()}")
        return pd.DataFrame()

    conn = sqlite3.connect(str(get_db_path()))

    try:
        try:
            if not validate_data_freshness(conn):
                return pd.DataFrame()
        except DataStalenessError as e:
            if allow_stale:
                logger.warning("Stale data allowed via --allow-stale: %s", e)
            else:
                logger.error(str(e))
                return pd.DataFrame()

        # 1. Get liquid universe
        symbols = get_liquid_universe(conn)
        if not symbols:
            logger.error("No liquid symbols found")
            return pd.DataFrame()

        # 2. Load price/volume data
        prices, volumes = load_price_data(conn, symbols)
        logger.info(f"Loaded data for {len(prices.columns)} symbols, {len(prices)} days")

        # 2b. Load long-format OHLCV for advanced signals (CGO, lead-lag, 52wk, Amihud)
        prices_long = load_price_data_long(conn, symbols, days=300)
        logger.info(f"Loaded long-format data: {len(prices_long)} rows")

        # Signal date = last available trading date in DB
        signal_date = prices.index[-1] if not prices.empty else pd.Timestamp.now()

        # 2a. Calculate market regime (with BOCPD changepoint overlay if enabled)
        regime_multiplier = 1.0
        if ENABLE_REGIME_FILTER:
            if ENABLE_BOCPD_REGIME:
                regime = calculate_market_regime_with_bocpd(prices)
            else:
                regime = calculate_market_regime(prices)
            regime_multiplier = regime["confidence_multiplier"]
            logger.info(
                f"Market regime: {regime['regime'].upper()} "
                f"(breadth={regime['breadth']:.1%}, momentum={regime['momentum']:.1%}, "
                f"confidence_mult={regime_multiplier:.2f})"
            )
            if regime["regime"] == "bear":
                logger.warning("BEAR MARKET DETECTED - reducing position sizes")

        # 2c. NRB monetary policy rate regime (macro overlay)
        nrb_mult = 1.0
        nrb_sector_adj: Dict[str, float] = {}
        try:
            nrb = get_nrb_policy_regime(db_path=str(get_db_path()))
            nrb_mult = nrb["multiplier"]
            nrb_sector_adj = nrb.get("sector_adjustments", {})
            if nrb["cycle"] != "no_data":
                logger.info(
                    f"NRB policy cycle: {nrb['cycle'].upper()} "
                    f"(rate={nrb['latest_rate_pct']}%, "
                    f"change={nrb['rate_change_bps']}bps, mult={nrb_mult:.2f})"
                )
        except Exception as e:
            logger.warning(f"NRB policy regime failed (skipping): {e}")

        # 3. Initialize scanners
        all_signals: List[AlphaSignal] = []

        # 3a. Momentum scanner - DISABLED based on backtest results
        # Backtest showed momentum signals have NEGATIVE alpha (-0.92% avg return)
        # Adding momentum to volume signals HURTS performance (Sharpe drops from 1.17 to 0.38)
        if ENABLE_MOMENTUM_SIGNALS:
            momentum_scanner = MomentumScanner(lookback_short=20, lookback_long=50)
            try:
                momentum_signals = momentum_scanner.calculate_signals(prices, volumes)
                logger.info(f"Momentum scanner found {len(momentum_signals)} signals")
                all_signals.extend(momentum_signals)
            except Exception as e:
                logger.warning(f"Momentum scanner failed: {e}")
        else:
            logger.info("Momentum scanner DISABLED (negative alpha in backtest)")

        # 3b. Liquidity/Volume scanner - PRIMARY SIGNAL SOURCE
        # Backtest: Volume breakouts with 21-day hold = Sharpe 1.17, 4.91% avg return
        liquidity_scanner = LiquidityScanner(min_volume_spike=VOLUME_SPIKE_MULTIPLIER)
        try:
            # Estimate market caps
            market_caps = {}
            for symbol in prices.columns:
                if symbol in volumes.columns:
                    avg_vol = volumes[symbol].mean()
                    last_price = prices[symbol].iloc[-1]
                    market_caps[symbol] = last_price * avg_vol * 1000  # Rough estimate

            liquidity_signals = liquidity_scanner.calculate_signals(prices, volumes, market_caps)
            # Boost confidence for volume signals (our best alpha source)
            # Apply regime multiplier to scale down in bear markets
            for sig in liquidity_signals:
                base_boost = 1.15
                sig.confidence = min(sig.confidence * base_boost * regime_multiplier, 0.85)
            logger.info(f"Liquidity scanner found {len(liquidity_signals)} signals (PRIMARY)")
            all_signals.extend(liquidity_signals)
        except Exception as e:
            logger.warning(f"Liquidity scanner failed: {e}")

        # 3c. Corporate action scanner
        corp_scanner = CorporateActionScanner()
        try:
            corp_actions = load_corporate_actions(conn)
            for action in corp_actions:
                corp_scanner.add_action(action)
            corp_signals = corp_scanner.scan(datetime.now())
            logger.info(f"Corporate action scanner found {len(corp_signals)} signals")
            all_signals.extend(corp_signals)
        except Exception as e:
            logger.warning(f"Corporate action scanner failed: {e}")

        # 3d. Fundamental scanner (optional)
        try:
            fund_scanner = FundamentalScanner()
            fundamentals = load_fundamentals(conn, symbols, latest_prices=prices.iloc[-1].to_dict())
            for fd in fundamentals.values():
                fund_scanner.update_fundamentals(fd)
            fund_signals = fund_scanner.scan()
            logger.info(f"Fundamental scanner found {len(fund_signals)} signals")
            all_signals.extend(fund_signals)
        except Exception as e:
            logger.warning(f"Fundamental scanner failed: {e}")

        # 3e. PEAD / earnings-drift overlay from quarterly_earnings.
        try:
            pead_signals = generate_pead_signals_at_date(
                prices,
                signal_date.strftime("%Y-%m-%d"),
                db_path=str(get_db_path()),
                liquid_symbols=symbols,
            )
            positive_pead = []
            for sig in pead_signals:
                if sig.direction <= 0:
                    continue
                sig.confidence = min(sig.confidence * regime_multiplier, 0.85)
                positive_pead.append(sig)
            logger.info(f"PEAD scanner found {len(positive_pead)} buy signals")
            all_signals.extend(positive_pead)
        except Exception as e:
            logger.warning(f"PEAD scanner failed: {e}")

        # 3f. CGO Disposition signal (Grinblatt & Han 2005)
        # Stocks breaking through their 260-day VWAP cost basis attract new buyers
        if not prices_long.empty:
            try:
                cgo_signals = generate_cgo_signals_at_date(
                    prices_long, signal_date, liquid_symbols=symbols
                )
                for sig in cgo_signals:
                    sig.confidence = min(sig.confidence * regime_multiplier, 0.85)
                logger.info(f"CGO scanner found {len(cgo_signals)} signals")
                all_signals.extend(cgo_signals)
            except Exception as e:
                logger.warning(f"CGO scanner failed (skipping): {e}")

        # 3g. Lead-lag sector spillover (Hou 2007)
        # When leading sectors break out, lagging correlated sectors follow with a delay
        if not prices_long.empty:
            try:
                ll_signals = generate_sector_leadlag_signals_at_date(
                    prices_long, signal_date, liquid_symbols=symbols
                )
                for sig in ll_signals:
                    sig.confidence = min(sig.confidence * regime_multiplier, 0.85)
                logger.info(f"Lead-lag scanner found {len(ll_signals)} signals")
                all_signals.extend(ll_signals)
            except Exception as e:
                logger.warning(f"Lead-lag scanner failed (skipping): {e}")

        # 3h. 52-week high proximity (George & Hwang 2004)
        # Retail anchoring at round-number highs — breakout above creates strong continuation
        if not prices_long.empty:
            try:
                from backend.backtesting.simple_backtest import generate_52wk_high_signals_at_date
                wk52_signals = generate_52wk_high_signals_at_date(
                    prices_long, signal_date, liquid_symbols=symbols
                )
                for sig in wk52_signals:
                    sig.confidence = min(sig.confidence * regime_multiplier, 0.85)
                logger.info(f"52Wk-high scanner found {len(wk52_signals)} signals")
                all_signals.extend(wk52_signals)
            except Exception as e:
                logger.warning(f"52Wk-high scanner failed (skipping): {e}")

        # 3i. Amihud illiquidity tilt (post-aggregation overlay)
        # Boosts strength for stocks in the illiquidity premium zone (quintiles 3-4)
        if not prices_long.empty and all_signals:
            try:
                from backend.backtesting.simple_backtest import apply_amihud_tilt
                all_signals = apply_amihud_tilt(all_signals, prices_long, signal_date)
                logger.info("Amihud illiquidity tilt applied")
            except Exception as e:
                logger.warning(f"Amihud tilt failed (skipping): {e}")

        logger.info(f"Total signals before aggregation: {len(all_signals)}")

        # 4. Aggregate signals
        aggregator = AlphaAggregator(max_position_size=MAX_SINGLE_POSITION_PCT)
        composite = aggregator.aggregate(all_signals)
        top_picks = aggregator.get_top_picks(composite, n=max_positions * 2, min_confidence=MIN_CONFIDENCE)

        logger.info(f"Top picks after aggregation: {len(top_picks)}")

        # Convert composite signals back to AlphaSignal for position sizing
        final_signals = []
        for pick in top_picks:
            # Use the strongest signal from the composite
            if pick.signals:
                best = max(pick.signals, key=lambda s: s.strength * s.confidence)
                final_signals.append(best)

        # 4b. Apply calendar effect multiplier (Model 16, includes post-Dashain correction)
        cal_mult = get_calendar_multiplier(datetime.now())
        if abs(cal_mult - 1.0) > 1e-6:
            logger.info(f"Calendar multiplier: {cal_mult:.3f}")
            for sig in final_signals:
                sig.confidence = min(sig.confidence * cal_mult, 0.95)

        # 4c. Apply NRB monetary policy rate multiplier
        if abs(nrb_mult - 1.0) > 1e-6 or nrb_sector_adj:
            for sig in final_signals:
                # Base macro multiplier (hiking=0.85, cutting=1.08, hold=1.0)
                sig.confidence = min(sig.confidence * nrb_mult, 0.95)
                # Sector-specific adjustment (e.g. penalize microfinance during hikes)
                sig_sector = get_symbol_sector(sig.symbol)
                sector_mult = nrb_sector_adj.get(sig_sector, 1.0)
                if abs(sector_mult - 1.0) > 1e-6:
                    sig.confidence = min(sig.confidence * sector_mult, 0.95)

        # 5. Calculate position sizes (with HRP if configured)
        result = calculate_position_sizes(
            final_signals, capital, prices, max_positions,
            prices_long=prices_long if not prices_long.empty else None,
            portfolio_method=PORTFOLIO_METHOD,
            signal_date=signal_date,
        )

        if result.empty:
            logger.warning("No positions to take")
            return result

        logger.info(f"Generated {len(result)} positions")

        # 6. Output
        if not dry_run:
            result.to_csv(output_file, index=False)
            logger.info(f"Saved to {output_file}")

        # Print summary
        print("\n" + "=" * 60)
        print(f"DAILY SIGNALS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 60)
        print(result.to_string(index=False))
        print("=" * 60)
        print(f"Total positions: {len(result)}")
        print(f"Capital deployed: {result['Value'].apply(lambda x: float(x.replace(',', ''))).sum():,.0f} NPR")
        print(f"Recommended holding: {RECOMMENDED_HOLDING_DAYS} days (based on backtest)")
        print("=" * 60)
        print("\nSTRATEGY NOTES:")
        print("- Volume breakout signals are PRIMARY (Sharpe 1.17 in backtest)")
        print("- Momentum signals DISABLED (negative alpha)")
        print(f"- Target exit: ~{RECOMMENDED_HOLDING_DAYS} trading days")
        print("=" * 60 + "\n")

        return result

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="NEPSE Daily Signal Generator")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL,
                        help=f"Trading capital in NPR (default: {DEFAULT_CAPITAL:,})")
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS,
                        help=f"Maximum number of positions (default: {MAX_POSITIONS})")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help=f"Output file (default: {OUTPUT_FILE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write output file, just print signals")
    parser.add_argument("--allow-stale", action="store_true",
                        help="Allow stale data (for research/backtest use)")

    args = parser.parse_args()

    if args.capital <= 0:
        parser.error("--capital must be positive")
    if not (1 <= args.max_positions <= 50):
        parser.error("--max-positions must be between 1 and 50")

    # Trading day check (skip on weekends/holidays unless forced)
    if not is_today_trading_day() and not args.allow_stale:
        from datetime import date
        today = date.today()
        nxt = next_trading_day(today)
        logger.warning(
            "Today (%s, %s) is not a NEPSE trading day. "
            "Next trading day: %s (%s). Use --allow-stale to override.",
            today, today.strftime("%A"), nxt, nxt.strftime("%A"),
        )
        return

    generate_signals(
        capital=args.capital,
        max_positions=args.max_positions,
        output_file=args.output,
        dry_run=args.dry_run,
        allow_stale=args.allow_stale,
    )


if __name__ == "__main__":
    main()
