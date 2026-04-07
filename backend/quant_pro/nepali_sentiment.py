"""
Nepali Sentiment Analysis signal for NEPSE stocks.

Uses NLP models (XLM-RoBERTa or keyword fallback) to score sentiment
from ShareSansar/MeroLagani news and comments. Aggregates recent
sentiment scores from the sentiment_scores DB table and generates
AlphaSignal objects for the backtest/live trading pipeline.

Integration:
    - Data source: sentiment_scores table (populated by sentiment_ingestion.py)
    - Output: List[AlphaSignal] with type=NLP_SENTIMENT
    - Called from: simple_backtest.py run_backtest() signal dispatch
    - Frequency: Daily (sentiment scored once per day after market close)

Signal logic:
    1. Query sentiment_scores for last `lookback_days` (default 3)
    2. Aggregate per symbol: mean score, total documents
    3. If mean_score > threshold AND n_docs >= min_documents: BUY signal
    4. If mean_score < -threshold: avoid (direction=0, no short in NEPSE)
    5. Strength proportional to sentiment score magnitude
    6. Confidence proportional to number of documents
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.quant_pro.alpha_practical import SignalType, AlphaSignal
from backend.quant_pro.database import get_db_path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = str(get_db_path())


def _query_sentiment_scores(
    db_path: str,
    end_date: str,
    lookback_days: int = 3,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query sentiment scores from the database.

    Args:
        db_path: Path to SQLite database
        end_date: End date (inclusive) in YYYY-MM-DD format
        lookback_days: Number of days to look back
        symbol: Optional symbol filter (None = all symbols)

    Returns:
        DataFrame with columns: date, symbol, source, model, score, confidence, n_documents
    """
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    conn = sqlite3.connect(db_path)

    query = """
        SELECT date, symbol, source, model, score, confidence, n_documents
        FROM sentiment_scores
        WHERE date >= ? AND date <= ?
    """
    params = [start_date, end_date]

    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    else:
        # Get both per-symbol and market-wide scores
        pass

    query += " ORDER BY date DESC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def _aggregate_sentiment(
    scores_df: pd.DataFrame,
    min_documents: int = 2,
) -> Dict[str, Dict]:
    """
    Aggregate sentiment scores per symbol.

    Args:
        scores_df: DataFrame from _query_sentiment_scores
        min_documents: Minimum total documents required

    Returns:
        Dict mapping symbol -> {mean_score, total_docs, mean_confidence, n_sources}
    """
    if scores_df.empty:
        return {}

    results = {}

    # Separate market-wide (symbol is NULL/None) from per-symbol
    per_symbol = scores_df[scores_df["symbol"].notna()]
    market_wide = scores_df[scores_df["symbol"].isna()]

    # Per-symbol aggregation
    if not per_symbol.empty:
        grouped = per_symbol.groupby("symbol").agg(
            mean_score=("score", "mean"),
            total_docs=("n_documents", "sum"),
            mean_confidence=("confidence", "mean"),
            n_sources=("source", "nunique"),
        ).reset_index()

        for _, row in grouped.iterrows():
            if row["total_docs"] >= min_documents:
                results[row["symbol"]] = {
                    "mean_score": float(row["mean_score"]),
                    "total_docs": int(row["total_docs"]),
                    "mean_confidence": float(row["mean_confidence"]),
                    "n_sources": int(row["n_sources"]),
                }

    # Market-wide sentiment (use as baseline for stocks without specific sentiment)
    market_sentiment = None
    if not market_wide.empty:
        market_sentiment = {
            "mean_score": float(market_wide["score"].mean()),
            "total_docs": int(market_wide["n_documents"].sum()),
            "mean_confidence": float(market_wide["confidence"].mean()),
        }

    # Store market-wide under special key
    if market_sentiment and market_sentiment["total_docs"] >= min_documents:
        results["__MARKET__"] = market_sentiment

    return results


def generate_sentiment_signals_at_date(
    prices_df: pd.DataFrame,
    date,
    db_path: str = DEFAULT_DB_PATH,
    lookback_days: int = 3,
    min_documents: int = 2,
    sentiment_threshold: float = 0.3,
    liquid_symbols: Optional[set] = None,
    market_sentiment_weight: float = 0.3,
) -> List[AlphaSignal]:
    """
    Generate signals from Nepali text sentiment.

    Logic:
    1. Query sentiment_scores for last lookback_days
    2. Aggregate per symbol: mean score, total documents
    3. If mean_score > threshold AND n_docs >= min_documents: buy signal
    4. Strength proportional to sentiment score magnitude
    5. Confidence proportional to number of documents and sources
    6. Market-wide sentiment provides a baseline tilt

    Args:
        prices_df: DataFrame with OHLCV data (used for symbol universe)
        date: Signal generation date
        db_path: Path to SQLite database
        lookback_days: Days of sentiment to aggregate (default 3)
        min_documents: Minimum documents needed to generate signal (default 2)
        sentiment_threshold: Minimum |score| to generate signal (default 0.3)
        liquid_symbols: Optional set of liquid symbols to filter to
        market_sentiment_weight: Weight of market-wide sentiment in scoring

    Returns:
        List of AlphaSignal objects with type=NLP_SENTIMENT
    """
    signals = []

    # Convert date to string
    if isinstance(date, (datetime, pd.Timestamp)):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = str(date)

    # Query sentiment scores
    try:
        scores_df = _query_sentiment_scores(
            db_path=db_path,
            end_date=date_str,
            lookback_days=lookback_days,
        )
    except Exception as e:
        logger.debug(f"Sentiment query failed for {date_str}: {e}")
        return signals

    if scores_df.empty:
        return signals

    # Aggregate per symbol
    aggregated = _aggregate_sentiment(scores_df, min_documents=min_documents)

    if not aggregated:
        return signals

    # Get market-wide sentiment
    market_sent = aggregated.pop("__MARKET__", None)
    market_score = market_sent["mean_score"] if market_sent else 0.0

    # Get valid symbols from prices_df
    if liquid_symbols:
        valid_symbols = liquid_symbols
    else:
        # Get symbols that have price data near this date
        valid_symbols = set(prices_df["symbol"].unique()) if "symbol" in prices_df.columns else set()

    for symbol, data in aggregated.items():
        if valid_symbols and symbol not in valid_symbols:
            continue

        mean_score = data["mean_score"]
        total_docs = data["total_docs"]
        mean_conf = data["mean_confidence"]
        n_sources = data["n_sources"]

        # Blend with market sentiment
        blended_score = (
            mean_score * (1 - market_sentiment_weight)
            + market_score * market_sentiment_weight
        )

        # Check threshold
        if abs(blended_score) < sentiment_threshold:
            continue

        # Direction: NEPSE is long-only, so only generate buy signals
        if blended_score <= 0:
            continue
        direction = 1

        # Strength: map sentiment score (0.3-1.0) to signal strength (0.3-1.0)
        strength = min(abs(blended_score), 1.0)

        # Confidence: based on document count and source diversity
        doc_confidence = min(total_docs / 10.0, 1.0)  # 10+ docs = max
        source_confidence = min(n_sources / 2.0, 1.0)  # 2+ sources = max
        model_confidence = mean_conf

        confidence = (doc_confidence * 0.3 + source_confidence * 0.3 + model_confidence * 0.4)
        confidence = max(0.1, min(confidence, 1.0))

        reasoning = (
            f"Sentiment score {blended_score:.3f} ({total_docs} docs, "
            f"{n_sources} sources, {lookback_days}d lookback)"
        )

        signals.append(AlphaSignal(
            symbol=symbol,
            signal_type=SignalType.NLP_SENTIMENT,
            direction=direction,
            strength=round(strength, 4),
            confidence=round(confidence, 4),
            reasoning=reasoning,
        ))

    # If market sentiment is strongly positive but no per-symbol signals,
    # we could optionally generate signals for top liquid stocks.
    # For now, we only signal when per-symbol sentiment is available.

    if signals:
        logger.info(
            f"Sentiment signals for {date_str}: {len(signals)} signals "
            f"(market sentiment: {market_score:.3f})"
        )

    return signals


def get_sentiment_summary(
    db_path: str = DEFAULT_DB_PATH,
    days: int = 7,
) -> Dict:
    """
    Get a summary of recent sentiment data for diagnostics.

    Returns dict with counts, date range, and distribution.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(*) FROM sentiment_scores")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(date), MAX(date) FROM sentiment_scores")
        min_date, max_date = cursor.fetchone()

        cursor.execute("""
            SELECT source, model, COUNT(*), AVG(score), AVG(confidence)
            FROM sentiment_scores
            GROUP BY source, model
        """)
        breakdown = cursor.fetchall()

        conn.close()

        return {
            "total_scores": total,
            "date_range": (min_date, max_date),
            "breakdown": [
                {
                    "source": row[0],
                    "model": row[1],
                    "count": row[2],
                    "avg_score": round(row[3], 4) if row[3] else None,
                    "avg_confidence": round(row[4], 4) if row[4] else None,
                }
                for row in breakdown
            ],
        }
    except Exception as e:
        conn.close()
        return {"error": str(e), "total_scores": 0}
