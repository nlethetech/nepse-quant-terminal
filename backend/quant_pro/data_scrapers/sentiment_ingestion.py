"""
Sentiment data ingestion from MeroLagani and ShareSansar news.

Scrapes NEPSE-related news headlines, scores them using NLP models
(XLM-RoBERTa multilingual or keyword-based fallback), and stores
aggregated daily sentiment scores in the sentiment_scores table.

Usage:
    python3 -m backend.quant_pro.data_scrapers.sentiment_ingestion
    python3 -m backend.quant_pro.data_scrapers.sentiment_ingestion --pages 5 --model keyword
    python3 -m backend.quant_pro.data_scrapers.sentiment_ingestion --symbol NABIL --days 7

Sources:
    - MeroLagani news: https://merolagani.com/NewsList.aspx
    - ShareSansar news: https://www.sharesansar.com/category/latest

Models (in priority order):
    1. cardiffnlp/twitter-xlm-roberta-base-sentiment (multilingual, Nepali-capable)
    2. Keyword-based fallback (Nepali + English financial keywords)
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from backend.quant_pro.paths import get_data_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = get_data_dir(__file__) / "nepse_market_data.db"

# Rate limit: 1 request per second
REQUEST_DELAY = 1.0

# --- Keyword-based sentiment scorer (fallback) ---

POSITIVE_KEYWORDS_NP = [
    "बढ्यो", "बढेको", "वृद्धि", "लाभांश", "बोनस", "मुनाफा", "सकारात्मक",
    "राम्रो", "उच्च", "बढ्दो", "शेयर बढ्यो", "नाफा", "सुधार", "उत्साहजनक",
    "आईपीओ", "लगानी", "प्रगति", "विकास", "सफल", "रेकर्ड", "उत्कृष्ट",
]
POSITIVE_KEYWORDS_EN = [
    "profit", "increase", "bullish", "growth", "dividend", "bonus",
    "ipo", "record", "high", "surge", "rally", "gain", "rise",
    "strong", "positive", "improve", "uptrend", "breakout",
]

NEGATIVE_KEYWORDS_NP = [
    "घट्यो", "घटेको", "गिरावट", "नोक्सान", "ऋणात्मक", "खराब", "न्यून",
    "घट्दो", "शेयर घट्यो", "क्षति", "समस्या", "चिन्ता", "जोखिम",
    "मन्दी", "दुर्घटना", "बन्द", "स्थगित", "कारबाही", "जरिवाना",
]
NEGATIVE_KEYWORDS_EN = [
    "loss", "decrease", "bearish", "decline", "crash", "fall",
    "drop", "negative", "risk", "concern", "weak", "downtrend",
    "circuit breaker", "suspend", "penalty", "fine", "fraud",
]

ALL_POSITIVE = set(POSITIVE_KEYWORDS_NP + POSITIVE_KEYWORDS_EN)
ALL_NEGATIVE = set(NEGATIVE_KEYWORDS_NP + NEGATIVE_KEYWORDS_EN)

# Common NEPSE stock symbols for entity extraction
# (loaded from DB at runtime if available)
_CACHED_SYMBOLS: Optional[set] = None


def _load_symbols_from_db(db_path: str) -> set:
    """Load NEPSE symbols from the database."""
    global _CACHED_SYMBOLS
    if _CACHED_SYMBOLS is not None:
        return _CACHED_SYMBOLS
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM stock_prices WHERE symbol NOT LIKE 'SECTOR::%' AND symbol != 'NEPSE'")
        symbols = {row[0] for row in cursor.fetchall()}
        conn.close()
        _CACHED_SYMBOLS = symbols
        return symbols
    except Exception:
        _CACHED_SYMBOLS = set()
        return set()


def _extract_symbols_from_text(text: str, known_symbols: set) -> List[str]:
    """Extract stock symbols mentioned in text."""
    found = []
    text_upper = text.upper()
    for sym in known_symbols:
        # Match as a word boundary (not substring of another word)
        if re.search(rf'\b{re.escape(sym)}\b', text_upper):
            found.append(sym)
    return found


# --- Scrapers ---

def _fetch_url(url: str, timeout: int = 15) -> str:
    """Fetch a URL with user-agent header and timeout."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5,ne;q=0.3",
    }
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (URLError, HTTPError) as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return ""


def scrape_merolagani_news(pages: int = 3) -> List[Dict]:
    """
    Scrape news headlines from MeroLagani.

    URL pattern: https://merolagani.com/NewsList.aspx?page=N
    Each headline has: title, date, URL with newsID

    Returns list of dicts with keys: title, url, date_str, source
    """
    articles = []
    base_url = "https://merolagani.com"

    for page in range(1, pages + 1):
        url = f"{base_url}/NewsList.aspx?page={page}"
        logger.info(f"Fetching MeroLagani page {page}/{pages}: {url}")
        html = _fetch_url(url)
        if not html:
            continue

        # Extract news articles using regex on the HTML
        # Pattern: <a href="/NewsDetail.aspx?newsID=XXXXX">HEADLINE</a>
        # Date pattern: "Feb 13, 2026 05:55 PM" or similar
        news_pattern = re.compile(
            r'<a[^>]*href="(/NewsDetail\.aspx\?newsID=\d+)"[^>]*>(.*?)</a>',
            re.DOTALL
        )
        date_pattern = re.compile(
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s+(?:AM|PM))'
        )

        matches = news_pattern.findall(html)
        dates = date_pattern.findall(html)

        for i, (href, title_raw) in enumerate(matches):
            title = unescape(re.sub(r'<[^>]+>', '', title_raw)).strip()
            if not title or len(title) < 10:
                continue

            # Try to associate a date
            date_str = ""
            if i < len(dates):
                date_str = dates[i]

            articles.append({
                "title": title,
                "url": f"{base_url}{href}",
                "date_str": date_str,
                "source": "merolagani",
            })

        time.sleep(REQUEST_DELAY)

    logger.info(f"Scraped {len(articles)} articles from MeroLagani")
    return articles


def scrape_sharesansar_news(pages: int = 3) -> List[Dict]:
    """
    Scrape news headlines from ShareSansar.

    URL pattern: https://www.sharesansar.com/category/latest?page=N
    Uses similar HTML parsing approach.

    Returns list of dicts with keys: title, url, date_str, source
    """
    articles = []
    base_url = "https://www.sharesansar.com"

    for page in range(1, pages + 1):
        url = f"{base_url}/category/latest?page={page}"
        logger.info(f"Fetching ShareSansar page {page}/{pages}: {url}")
        html = _fetch_url(url)
        if not html:
            # Try alternative URL pattern
            url = f"{base_url}/category/latest/{page}"
            html = _fetch_url(url)
            if not html:
                continue

        # ShareSansar article links typically follow pattern:
        # <a href="https://www.sharesansar.com/newsdetail/SLUG-NNNN">TITLE</a>
        news_pattern = re.compile(
            r'<a[^>]*href="((?:https?://www\.sharesansar\.com)?/newsdetail/[^"]+)"[^>]*>(.*?)</a>',
            re.DOTALL
        )
        date_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'
        )

        matches = news_pattern.findall(html)
        dates = date_pattern.findall(html)

        for i, (href, title_raw) in enumerate(matches):
            title = unescape(re.sub(r'<[^>]+>', '', title_raw)).strip()
            if not title or len(title) < 10:
                continue

            full_url = href if href.startswith("http") else f"{base_url}{href}"
            date_str = dates[i] if i < len(dates) else ""

            articles.append({
                "title": title,
                "url": full_url,
                "date_str": date_str,
                "source": "sharesansar",
            })

        time.sleep(REQUEST_DELAY)

    logger.info(f"Scraped {len(articles)} articles from ShareSansar")
    return articles


# --- Sentiment Scoring ---

def _load_xlm_roberta():
    """
    Load XLM-RoBERTa sentiment model.
    Returns (pipeline, model_name) or (None, None) if loading fails.
    """
    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading cardiffnlp/twitter-xlm-roberta-base-sentiment...")
        classifier = hf_pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            top_k=None,  # Return all labels with scores
            truncation=True,
            max_length=512,
        )
        logger.info("XLM-RoBERTa model loaded successfully")
        return classifier, "xlm-roberta-base-sentiment"
    except Exception as e:
        logger.warning(f"Failed to load XLM-RoBERTa: {e}")
        return None, None


def score_with_xlm_roberta(texts: List[str], classifier) -> List[Dict]:
    """
    Score texts using XLM-RoBERTa sentiment model.
    Returns list of dicts with keys: score (-1 to +1), confidence (0-1), label
    """
    results = []
    # Process in batches to avoid memory issues
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            outputs = classifier(batch)
            for output in outputs:
                # output is a list of dicts: [{"label": "positive", "score": 0.8}, ...]
                if isinstance(output, list):
                    label_scores = {item["label"].lower(): item["score"] for item in output}
                else:
                    label_scores = {output["label"].lower(): output["score"]}

                # Map to -1 to +1 scale
                pos = label_scores.get("positive", 0.0)
                neg = label_scores.get("negative", 0.0)
                neu = label_scores.get("neutral", 0.0)

                # Weighted score: positive contributes +1, negative -1, neutral 0
                score = pos - neg
                confidence = max(pos, neg, neu)

                results.append({
                    "score": round(score, 4),
                    "confidence": round(confidence, 4),
                    "label": max(label_scores, key=label_scores.get),
                })
        except Exception as e:
            logger.warning(f"XLM-RoBERTa batch scoring failed: {e}")
            # Fall back to keyword for this batch
            for text in batch:
                results.append(score_with_keywords(text))

    return results


def score_with_keywords(text: str) -> Dict:
    """
    Keyword-based sentiment scoring fallback.
    Returns dict with keys: score (-1 to +1), confidence (0-1), label
    """
    text_lower = text.lower()
    pos_count = sum(1 for kw in ALL_POSITIVE if kw.lower() in text_lower)
    neg_count = sum(1 for kw in ALL_NEGATIVE if kw.lower() in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return {"score": 0.0, "confidence": 0.1, "label": "neutral"}

    score = (pos_count - neg_count) / total
    confidence = min(total / 5.0, 1.0)  # Scale confidence: 5+ keywords = 1.0

    if score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return {"score": round(score, 4), "confidence": round(confidence, 4), "label": label}


def score_articles(
    articles: List[Dict],
    model: str = "auto",
) -> List[Dict]:
    """
    Score articles using the specified or best available model.

    Args:
        articles: List of article dicts with 'title' key
        model: "xlm-roberta", "keyword", or "auto" (try xlm first)

    Returns: articles with added keys: score, confidence, label, model_name
    """
    if not articles:
        return []

    classifier = None
    model_name = "keyword"

    if model in ("auto", "xlm-roberta"):
        classifier, loaded_name = _load_xlm_roberta()
        if classifier is not None:
            model_name = loaded_name

    texts = [a["title"] for a in articles]

    if classifier is not None:
        logger.info(f"Scoring {len(texts)} articles with {model_name}")
        scores = score_with_xlm_roberta(texts, classifier)
    else:
        logger.info(f"Scoring {len(texts)} articles with keyword-based scorer")
        model_name = "keyword"
        scores = [score_with_keywords(t) for t in texts]

    # Merge scores into articles
    for article, score_dict in zip(articles, scores):
        article["sentiment_score"] = score_dict["score"]
        article["sentiment_confidence"] = score_dict["confidence"]
        article["sentiment_label"] = score_dict["label"]
        article["model_name"] = model_name

    return articles


# --- Aggregation and Storage ---

def _parse_date(date_str: str) -> Optional[str]:
    """Parse various date formats to YYYY-MM-DD."""
    if not date_str:
        return None

    formats = [
        "%b %d, %Y %I:%M %p",      # "Feb 13, 2026 05:55 PM"
        "%Y-%m-%d %H:%M:%S",        # "2026-02-13 17:55:00"
        "%Y-%m-%d",                  # "2026-02-13"
        "%d %b %Y",                  # "13 Feb 2026"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def aggregate_and_store(
    articles: List[Dict],
    db_path: str,
    known_symbols: Optional[set] = None,
):
    """
    Aggregate scored articles by (date, symbol, source) and store in DB.

    For each article:
    1. Parse date
    2. Extract mentioned symbols (if any)
    3. Store per-symbol scores AND market-wide score (symbol=NULL)
    """
    if not articles:
        logger.warning("No articles to store")
        return 0

    if known_symbols is None:
        known_symbols = _load_symbols_from_db(db_path)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today = datetime.now().strftime("%Y-%m-%d")

    # Group articles by (date, source, model)
    # For each group, compute: mean score, mean confidence, count
    from collections import defaultdict

    # key = (date, symbol_or_none, source, model)
    groups = defaultdict(lambda: {"scores": [], "confidences": []})

    for article in articles:
        date = _parse_date(article.get("date_str", "")) or today
        source = article.get("source", "unknown")
        model = article.get("model_name", "keyword")
        score = article.get("sentiment_score", 0.0)
        conf = article.get("sentiment_confidence", 0.0)

        # Market-wide entry (symbol = empty string for SQLite NULL handling)
        key = (date, "", source, model)
        groups[key]["scores"].append(score)
        groups[key]["confidences"].append(conf)

        # Per-symbol entries
        symbols = _extract_symbols_from_text(article.get("title", ""), known_symbols)
        for sym in symbols:
            key = (date, sym, source, model)
            groups[key]["scores"].append(score)
            groups[key]["confidences"].append(conf)

    # Store aggregated scores
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    stored = 0

    for (date, symbol, source, model), data in groups.items():
        scores = data["scores"]
        confidences = data["confidences"]
        mean_score = sum(scores) / len(scores)
        mean_conf = sum(confidences) / len(confidences)
        n_docs = len(scores)

        # Use None for market-wide (no symbol)
        sym_value = symbol if symbol else None

        try:
            cursor.execute(
                """INSERT OR REPLACE INTO sentiment_scores
                   (date, symbol, source, model, score, confidence, n_documents, scraped_at_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (date, sym_value, source, model, round(mean_score, 4),
                 round(mean_conf, 4), n_docs, now_utc),
            )
            stored += 1
        except sqlite3.Error as e:
            logger.warning(f"Failed to store ({date}, {symbol}, {source}): {e}")

    conn.commit()
    conn.close()
    logger.info(f"Stored {stored} aggregated sentiment scores")
    return stored


# --- Main pipeline ---

def run_sentiment_pipeline(
    pages: int = 3,
    model: str = "auto",
    db_path: str = str(DEFAULT_DB_PATH),
    sources: Optional[List[str]] = None,
) -> int:
    """
    Full sentiment ingestion pipeline:
    1. Scrape news from MeroLagani and/or ShareSansar
    2. Score with NLP model (XLM-RoBERTa or keyword fallback)
    3. Aggregate and store in sentiment_scores table

    Args:
        pages: Number of pages to scrape per source
        model: "auto", "xlm-roberta", or "keyword"
        db_path: Path to SQLite database
        sources: List of sources to scrape ("merolagani", "sharesansar")

    Returns: Number of aggregated scores stored
    """
    if sources is None:
        sources = ["merolagani", "sharesansar"]

    all_articles = []

    if "merolagani" in sources:
        ml_articles = scrape_merolagani_news(pages=pages)
        all_articles.extend(ml_articles)

    if "sharesansar" in sources:
        ss_articles = scrape_sharesansar_news(pages=pages)
        all_articles.extend(ss_articles)

    if not all_articles:
        logger.warning("No articles scraped from any source")
        return 0

    logger.info(f"Total articles scraped: {len(all_articles)}")

    # Score all articles
    scored_articles = score_articles(all_articles, model=model)

    # Log sentiment distribution
    scores = [a["sentiment_score"] for a in scored_articles]
    if scores:
        pos = sum(1 for s in scores if s > 0.1)
        neg = sum(1 for s in scores if s < -0.1)
        neu = len(scores) - pos - neg
        logger.info(f"Sentiment distribution: {pos} positive, {neg} negative, {neu} neutral")
        logger.info(f"Mean sentiment: {sum(scores)/len(scores):.4f}")

    # Aggregate and store
    stored = aggregate_and_store(scored_articles, db_path)
    return stored


def main():
    parser = argparse.ArgumentParser(
        description="Scrape NEPSE news and compute sentiment scores"
    )
    parser.add_argument(
        "--pages", type=int, default=3,
        help="Number of pages to scrape per source (default: 3)"
    )
    parser.add_argument(
        "--model", choices=["auto", "xlm-roberta", "keyword"], default="auto",
        help="Sentiment model to use (default: auto = try xlm-roberta first)"
    )
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB_PATH),
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--sources", nargs="+", default=["merolagani", "sharesansar"],
        choices=["merolagani", "sharesansar"],
        help="News sources to scrape"
    )
    parser.add_argument(
        "--keyword-only", action="store_true",
        help="Force keyword-based scoring (skip model download)"
    )

    args = parser.parse_args()

    model = "keyword" if args.keyword_only else args.model

    stored = run_sentiment_pipeline(
        pages=args.pages,
        model=model,
        db_path=args.db,
        sources=args.sources,
    )
    print(f"\nDone. Stored {stored} aggregated sentiment scores.")
    return 0 if stored > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
