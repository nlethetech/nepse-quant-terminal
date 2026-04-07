"""NepalOSINT-driven event layer for signal ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from .database import get_db_connection, init_db
from .data_scrapers.sentiment_ingestion import (
    scrape_merolagani_news,
    scrape_sharesansar_news,
)

logger = logging.getLogger(__name__)

NST_OFFSET = timedelta(hours=5, minutes=45)
DEFAULT_PROMPT_VERSION = "nepse_event_layer_v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_nst(now_utc: Optional[datetime] = None) -> datetime:
    base = now_utc or _now_utc()
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return (base + NST_OFFSET).replace(tzinfo=None)


def _normalize_headline(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9]+", " ", (text or "").strip().lower()))
    return cleaned.strip()


def _headline_hash(text: str) -> str:
    return hashlib.sha1(_normalize_headline(text).encode("utf-8")).hexdigest()


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raw = str(value).strip()
    if not raw:
        return None
    candidates = [
        raw,
        raw.replace("Z", "+00:00"),
    ]
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%b %d, %Y %I:%M %p"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _normalize_entity_key(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().upper())


def _find_first(mapping: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


@dataclass
class EventLayerConfig:
    enabled: bool = False
    api_base_url: str = ""
    api_key: str = ""
    api_email: str = ""
    api_password: str = ""
    use_guest_login: bool = True
    lookback_hours: int = 12
    model: str = "gpt-4.1-mini"
    min_confidence: float = 0.45
    openai_api_key: str = ""
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    prompt_version: str = DEFAULT_PROMPT_VERSION
    request_timeout_secs: int = 20
    min_nepalosint_items: int = 5


@dataclass
class EventLayerResult:
    status: str
    detail: str
    run_date: str
    news_saved: int = 0
    scored_rows: int = 0
    nepalosint_items: int = 0
    fallback_items: int = 0


@dataclass
class EventAdjustmentContext:
    run_date: str = ""
    market_adjustment: float = 0.0
    sector_adjustments: Dict[str, float] = field(default_factory=dict)
    symbol_adjustments: Dict[str, float] = field(default_factory=dict)
    market_rationale: str = ""
    sector_rationales: Dict[str, str] = field(default_factory=dict)
    symbol_rationales: Dict[str, str] = field(default_factory=dict)

    def details_for(self, symbol: str, sector: Optional[str]) -> Dict[str, Any]:
        symbol_key = _normalize_entity_key(symbol)
        sector_key = _normalize_entity_key(sector)
        symbol_adj = float(self.symbol_adjustments.get(symbol_key, 0.0))
        sector_adj = float(self.sector_adjustments.get(sector_key, 0.0)) if sector_key else 0.0
        effective_sector = 0.0 if abs(symbol_adj) > 1e-12 else sector_adj
        total = _clamp(self.market_adjustment + effective_sector + symbol_adj, -0.20, 0.20)

        rationale_parts: List[str] = []
        if self.market_rationale and abs(self.market_adjustment) > 1e-12:
            rationale_parts.append(f"market {self.market_adjustment:+.2f}: {self.market_rationale}")
        if abs(symbol_adj) > 1e-12:
            rationale = self.symbol_rationales.get(symbol_key, "")
            rationale_parts.append(f"symbol {symbol_adj:+.2f}: {rationale}".strip())
        elif abs(sector_adj) > 1e-12:
            rationale = self.sector_rationales.get(sector_key, "")
            rationale_parts.append(f"sector {sector_adj:+.2f}: {rationale}".strip())

        return {
            "event_adjustment": total,
            "market_adjustment": self.market_adjustment,
            "sector_adjustment": effective_sector,
            "symbol_adjustment": symbol_adj,
            "event_rationale": " | ".join(part for part in rationale_parts if part).strip(),
        }


def load_event_layer_config() -> EventLayerConfig:
    return EventLayerConfig(
        enabled=_env_flag("NEPSE_EVENT_LAYER_ENABLED", False),
        api_base_url=str(os.environ.get("NEPALOSINT_API_BASE_URL", "")).strip(),
        api_key=str(os.environ.get("NEPALOSINT_API_KEY", "")).strip(),
        api_email=str(os.environ.get("NEPALOSINT_API_EMAIL", os.environ.get("OSINT_EMAIL", ""))).strip(),
        api_password=str(os.environ.get("NEPALOSINT_API_PASSWORD", os.environ.get("OSINT_PASSWORD", ""))).strip(),
        use_guest_login=_env_flag("NEPALOSINT_API_USE_GUEST_LOGIN", True),
        lookback_hours=max(1, _env_int("NEPSE_EVENT_LOOKBACK_HOURS", 12)),
        model=str(os.environ.get("NEPSE_EVENT_MODEL", "gpt-4.1-mini")).strip() or "gpt-4.1-mini",
        min_confidence=_clamp(_env_float("NEPSE_EVENT_MIN_CONFIDENCE", 0.45), 0.0, 1.0),
        openai_api_key=str(os.environ.get("OPENAI_API_KEY", "")).strip(),
        openai_base_url=str(os.environ.get("OPENAI_API_BASE_URL", DEFAULT_OPENAI_BASE_URL)).strip() or DEFAULT_OPENAI_BASE_URL,
    )


def _fetch_json(
    url: str,
    *,
    api_key: str = "",
    bearer_token: str = "",
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
) -> Dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": "NEPSE-Quant-EventLayer/1.0"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    if api_key:
        headers["X-API-Key"] = api_key
        if not bearer_token:
            headers["Authorization"] = f"Bearer {api_key}"
    response = requests.get(url, headers=headers, params=params or {}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"items": payload}


def _post_json(
    url: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    api_key: str = "",
    bearer_token: str = "",
    timeout: int = 20,
) -> Dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "NEPSE-Quant-EventLayer/1.0",
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    if api_key:
        headers["X-API-Key"] = api_key
        if not bearer_token:
            headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(url, headers=headers, json=payload or {}, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


def _normalize_nepalosint_api_root(raw_url: str) -> Optional[str]:
    cleaned = str(raw_url or "").strip().rstrip("/")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered.endswith("/api/v1"):
        return cleaned
    if "/api/v1/" not in lowered and re.match(r"^https?://[^/]+$", cleaned):
        return f"{cleaned}/api/v1"
    return None


def _build_national_assessment_text(
    executive_summary: Dict[str, Any],
    analytics_summary: Dict[str, Any],
    announcements_payload: Dict[str, Any],
) -> str:
    parts: List[str] = []

    situation = str(executive_summary.get("situation_overview") or "").strip()
    key_judgment = str(executive_summary.get("key_judgment") or "").strip()
    watch_items = executive_summary.get("watch_items") or []
    if situation:
        parts.append(situation)
    if key_judgment:
        parts.append(key_judgment)
    if isinstance(watch_items, list):
        compact_watch = [str(item).strip() for item in watch_items[:3] if str(item).strip()]
        if compact_watch:
            parts.append("Watch items: " + "; ".join(compact_watch))

    if not parts and analytics_summary:
        stories = int(analytics_summary.get("stories") or 0)
        events = int(analytics_summary.get("events") or 0)
        active_alerts = int(analytics_summary.get("active_alerts") or 0)
        parts.append(
            f"Last-window NepalOSINT summary: {stories} stories, {events} events, {active_alerts} active alerts."
        )

    latest = announcements_payload.get("latest") or []
    if isinstance(latest, list) and latest:
        titles = [str(item.get("title") or "").strip() for item in latest[:3] if isinstance(item, dict)]
        titles = [title for title in titles if title]
        if titles:
            parts.append("Latest government actions: " + "; ".join(titles))

    return " ".join(part for part in parts if part).strip()


def _login_nepalosint_api(api_root: str, config: EventLayerConfig) -> str:
    if config.api_key:
        return config.api_key
    if config.api_email and config.api_password:
        payload = _post_json(
            f"{api_root}/auth/login",
            payload={"email": config.api_email, "password": config.api_password},
            timeout=config.request_timeout_secs,
        )
        token = str(payload.get("access_token") or "").strip()
        if token:
            return token
    if config.use_guest_login:
        payload = _post_json(
            f"{api_root}/auth/guest",
            payload={},
            timeout=config.request_timeout_secs,
        )
        token = str(payload.get("access_token") or "").strip()
        if token:
            return token
    return ""


def _fetch_nepalosint_root_payload(
    api_root: str,
    *,
    config: EventLayerConfig,
) -> Dict[str, Any]:
    token = _login_nepalosint_api(api_root, config)
    fetch_kwargs = {
        "api_key": config.api_key,
        "bearer_token": token,
        "timeout": config.request_timeout_secs,
    }

    stories_payload = _fetch_json(
        f"{api_root}/stories/export",
        params={"hours": config.lookback_hours, "limit": 200},
        **fetch_kwargs,
    )
    announcements_payload = _fetch_json(
        f"{api_root}/announcements/summary",
        params={"hours": config.lookback_hours, "limit": 30},
        **fetch_kwargs,
    )

    executive_summary: Dict[str, Any] = {}
    analytics_summary: Dict[str, Any] = {}
    try:
        executive_summary = _fetch_json(
            f"{api_root}/analytics/executive-summary",
            params={"hours": config.lookback_hours},
            **fetch_kwargs,
        )
    except Exception:
        try:
            analytics_summary = _fetch_json(
                f"{api_root}/analytics/summary",
                params={"hours": config.lookback_hours},
                **fetch_kwargs,
            )
        except Exception:
            analytics_summary = {}

    items: List[Dict[str, Any]] = []
    for story in stories_payload.get("stories") or []:
        if not isinstance(story, dict):
            continue
        items.append(story)
    for announcement in announcements_payload.get("latest") or []:
        if not isinstance(announcement, dict):
            continue
        items.append(
            {
                "id": announcement.get("id") or announcement.get("external_id"),
                "headline": announcement.get("title"),
                "title": announcement.get("title"),
                "url": announcement.get("url"),
                "published_at": announcement.get("published_at") or announcement.get("date_ad") or announcement.get("created_at"),
                "source": announcement.get("source_name") or announcement.get("source") or "government",
                "source_id": announcement.get("source"),
                "category": announcement.get("category") or "government_announcement",
                "classification": announcement.get("category") or "government_announcement",
                "summary": announcement.get("content"),
                "content": announcement.get("content"),
            }
        )

    return {
        "national_assessment": _build_national_assessment_text(
            executive_summary,
            analytics_summary,
            announcements_payload,
        ),
        "items": items,
        "stories_export": stories_payload,
        "announcements_summary": announcements_payload,
        "executive_summary": executive_summary,
        "analytics_summary": analytics_summary,
    }


def _coerce_symbols(item: Dict[str, Any]) -> List[str]:
    raw = _find_first(item, "symbols", "tickers", "stocks", "affected_symbols")
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        values = re.split(r"[,;/|]\s*|\s+", raw.strip())
    else:
        values = []
    result = []
    for value in values:
        key = _normalize_entity_key(value)
        if key and key not in result:
            result.append(key)
    return result


def _coerce_sectors(item: Dict[str, Any]) -> List[str]:
    raw = _find_first(item, "sectors", "affected_sectors", "sector")
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        values = re.split(r"[,;/|]", raw)
    else:
        values = []
    result = []
    for value in values:
        key = _normalize_entity_key(value)
        if key and key not in result:
            result.append(key)
    return result


def _extract_payload_items(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    if not isinstance(payload, dict):
        return [], ""

    summary = ""
    for key in ("national_assessment", "nationalAssessment", "summary", "assessment", "executive_summary"):
        value = payload.get(key)
        if isinstance(value, dict):
            summary = str(_find_first(value, "summary", "text", "content", "headline", "title") or "").strip()
        elif isinstance(value, str):
            summary = value.strip()
        if summary:
            break

    candidates: Any = payload.get("items")
    if not isinstance(candidates, list):
        for key in ("events", "articles", "news", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates = value
                break
            if isinstance(value, dict):
                nested = value.get("items") or value.get("events") or value.get("articles")
                if isinstance(nested, list):
                    candidates = nested
                    break
    return list(candidates or []), summary


def _normalize_nepalosint_items(
    payload: Dict[str, Any],
    *,
    now_utc: datetime,
    lookback_hours: int,
) -> Tuple[List[Dict[str, Any]], str]:
    raw_items, summary = _extract_payload_items(payload)
    window_start = now_utc - timedelta(hours=lookback_hours)
    normalized: List[Dict[str, Any]] = []

    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        headline = str(_find_first(raw, "headline", "title", "summary", "text", "content") or "").strip()
        if len(headline) < 8:
            continue
        published_at = _parse_timestamp(_find_first(raw, "published_at", "publishedAt", "timestamp", "created_at", "date"))
        if published_at and published_at < window_start:
            continue
        source = str(_find_first(raw, "source", "publisher", "feed") or "nepalosint").strip() or "nepalosint"
        source_id = str(_find_first(raw, "source_id", "sourceId", "id", "event_id") or "").strip()
        url = str(_find_first(raw, "url", "link", "permalink") or "").strip()
        symbols = _coerce_symbols(raw)
        sectors = _coerce_sectors(raw)
        content_json = json.dumps(
            {
                "source_id": source_id,
                "symbols": symbols,
                "sectors": sectors,
                "classification": _find_first(raw, "classification", "classification_label", "event_type", "category"),
                "raw": raw,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        synthetic_id = source_id or _headline_hash(headline)
        normalized.append(
            {
                "source_id": source_id,
                "headline": headline,
                "url": url or f"nepalosint://{source}/{synthetic_id}",
                "date": (published_at or now_utc).date().isoformat(),
                "published_at_utc": (published_at or now_utc).replace(microsecond=0).isoformat(),
                "source": source,
                "category": str(_find_first(raw, "category", "event_type", "classification") or "nepalosint").strip() or "nepalosint",
                "sentiment_score": _find_first(raw, "sentiment_score", "sentiment", "score"),
                "sentiment_label": str(_find_first(raw, "sentiment_label", "sentiment", "polarity") or "").strip() or None,
                "symbol": symbols[0] if len(symbols) == 1 else None,
                "symbols": symbols,
                "sectors": sectors,
                "content": content_json,
            }
        )

    return normalized, summary


def _normalize_fallback_items(now_utc: datetime) -> List[Dict[str, Any]]:
    fallback: List[Dict[str, Any]] = []
    for article in scrape_merolagani_news(pages=1) + scrape_sharesansar_news(pages=1):
        headline = str(article.get("title") or "").strip()
        if len(headline) < 8:
            continue
        fallback.append(
            {
                "source_id": "",
                "headline": headline,
                "url": str(article.get("url") or f"fallback://{_headline_hash(headline)}"),
                "date": _parse_timestamp(article.get("date_str")) and _parse_timestamp(article.get("date_str")).date().isoformat() or now_utc.date().isoformat(),
                "published_at_utc": (_parse_timestamp(article.get("date_str")) or now_utc).replace(microsecond=0).isoformat(),
                "source": str(article.get("source") or "fallback_news"),
                "category": "fallback_news",
                "sentiment_score": None,
                "sentiment_label": None,
                "symbol": None,
                "symbols": [],
                "sectors": [],
                "content": json.dumps({"raw": article}, ensure_ascii=False, sort_keys=True),
            }
        )
    return fallback


def _dedupe_items(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_source_ids = set()
    seen_urls = set()
    seen_hashes = set()
    deduped: List[Dict[str, Any]] = []

    for item in items:
        source_id = str(item.get("source_id") or "").strip()
        url = str(item.get("url") or "").strip()
        headline_hash = _headline_hash(str(item.get("headline") or ""))
        if source_id and source_id in seen_source_ids:
            continue
        if url and url in seen_urls:
            continue
        if headline_hash and headline_hash in seen_hashes:
            continue

        if source_id:
            seen_source_ids.add(source_id)
        if url:
            seen_urls.add(url)
        if headline_hash:
            seen_hashes.add(headline_hash)
        deduped.append(item)

    return deduped


def _store_news_items(items: List[Dict[str, Any]]) -> int:
    if not items:
        return 0

    conn = get_db_connection()
    cur = conn.cursor()
    stored = 0
    for item in items:
        cur.execute(
            """
            INSERT OR REPLACE INTO news
            (symbol, date, headline, url, source, sentiment_score, sentiment_label, category, content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.get("symbol"),
                item.get("date"),
                item.get("headline"),
                item.get("url"),
                item.get("source"),
                item.get("sentiment_score"),
                item.get("sentiment_label"),
                item.get("category"),
                item.get("content"),
            ),
        )
        stored += 1
    conn.commit()
    conn.close()
    return stored


def _build_assessment_payload(
    items: List[Dict[str, Any]],
    summary_text: str,
    *,
    window_start_utc: str,
    window_end_utc: str,
) -> Dict[str, Any]:
    compact_items = []
    for item in items[:60]:
        compact_items.append(
            {
                "ref": item.get("url"),
                "headline": item.get("headline"),
                "source": item.get("source"),
                "published_at_utc": item.get("published_at_utc"),
                "category": item.get("category"),
                "sentiment_label": item.get("sentiment_label"),
                "sentiment_score": item.get("sentiment_score"),
                "symbols": item.get("symbols") or [],
                "sectors": item.get("sectors") or [],
            }
        )
    return {
        "window_start_utc": window_start_utc,
        "window_end_utc": window_end_utc,
        "national_assessment": summary_text,
        "items": compact_items,
    }


def _call_openai_structured_scores(
    assessment_payload: Dict[str, Any],
    config: EventLayerConfig,
) -> List[Dict[str, Any]]:
    schema = {
        "name": "news_event_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_type": {"type": "string", "enum": ["market", "sector", "symbol"]},
                            "entity_key": {"type": "string"},
                            "impact_direction": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                            "impact_score": {"type": "number"},
                            "confidence": {"type": "number"},
                            "event_type": {"type": "string"},
                            "source_refs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "rationale_short": {"type": "string"},
                        },
                        "required": [
                            "entity_type",
                            "entity_key",
                            "impact_direction",
                            "impact_score",
                            "confidence",
                            "event_type",
                            "source_refs",
                            "rationale_short",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["rows"],
            "additionalProperties": False,
        },
    }

    system_prompt = (
        "You are scoring NEPSE market events for a trading ranker. "
        "Use only the provided NepalOSINT/news assessment. "
        "Return rows only for material 1-5 trading day impacts. "
        "entity_key must be 'NEPSE' for market rows, an uppercase sector name for sector rows, "
        "or an uppercase stock symbol for symbol rows. "
        "impact_score must be signed and between -0.20 and 0.20. "
        "If a row is neutral, impact_score must be 0."
    )

    response = requests.post(
        f"{config.openai_base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.openai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(assessment_payload, ensure_ascii=False)},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_schema", "json_schema": schema},
        },
        timeout=config.request_timeout_secs,
    )
    response.raise_for_status()
    payload = response.json()
    content = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    data = json.loads(content or "{}")
    rows = data.get("rows", [])
    return rows if isinstance(rows, list) else []


def _sanitize_scored_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        entity_type = str(row.get("entity_type") or "").strip().lower()
        if entity_type not in {"market", "sector", "symbol"}:
            continue
        entity_key = _normalize_entity_key(row.get("entity_key"))
        if entity_type == "market":
            entity_key = "NEPSE"
        if not entity_key:
            continue
        direction = str(row.get("impact_direction") or "neutral").strip().lower()
        if direction not in {"positive", "negative", "neutral"}:
            direction = "neutral"
        raw_score = _clamp(float(row.get("impact_score") or 0.0), -0.20, 0.20)
        if direction == "positive" and raw_score < 0:
            raw_score = abs(raw_score)
        elif direction == "negative" and raw_score > 0:
            raw_score = -abs(raw_score)
        elif direction == "neutral":
            raw_score = 0.0
        confidence = _clamp(float(row.get("confidence") or 0.0), 0.0, 1.0)
        source_refs = [str(item).strip() for item in (row.get("source_refs") or []) if str(item).strip()]
        cleaned.append(
            {
                "entity_type": entity_type,
                "entity_key": entity_key,
                "impact_direction": direction,
                "impact_score": raw_score,
                "confidence": confidence,
                "event_type": str(row.get("event_type") or "news_event").strip() or "news_event",
                "source_refs_json": json.dumps(source_refs, ensure_ascii=False),
                "source_count": len(source_refs),
                "rationale_short": str(row.get("rationale_short") or "").strip()[:300],
            }
        )
    return cleaned


def _replace_scored_rows(
    *,
    run_date: str,
    window_start_utc: str,
    window_end_utc: str,
    rows: List[Dict[str, Any]],
    model_name: str,
    prompt_version: str,
) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM news_event_scores WHERE run_date = ?", (run_date,))
    created_at = _now_utc().replace(microsecond=0).isoformat()
    for row in rows:
        cur.execute(
            """
            INSERT INTO news_event_scores
            (run_date, window_start_utc, window_end_utc, entity_type, entity_key,
             impact_direction, impact_score, confidence, event_type, source_count,
             source_refs_json, rationale_short, model_name, prompt_version, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_date,
                window_start_utc,
                window_end_utc,
                row["entity_type"],
                row["entity_key"],
                row["impact_direction"],
                row["impact_score"],
                row["confidence"],
                row["event_type"],
                row["source_count"],
                row["source_refs_json"],
                row["rationale_short"],
                model_name,
                prompt_version,
                created_at,
            ),
        )
    conn.commit()
    conn.close()
    return len(rows)


def refresh_daily_event_layer(
    *,
    config: Optional[EventLayerConfig] = None,
    now_utc: Optional[datetime] = None,
    force: bool = False,
) -> EventLayerResult:
    init_db()
    cfg = config or load_event_layer_config()
    current_utc = now_utc or _now_utc()
    run_date = _now_nst(current_utc).date().isoformat()

    if not cfg.enabled:
        return EventLayerResult(status="disabled", detail="event layer disabled", run_date=run_date)
    if not cfg.api_base_url:
        return EventLayerResult(status="unavailable", detail="NepalOSINT API base URL not configured", run_date=run_date)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM news_event_scores WHERE run_date = ?", (run_date,))
    existing_rows = int(cur.fetchone()[0] or 0)
    conn.close()
    if existing_rows and not force:
        return EventLayerResult(
            status="cached",
            detail="using existing event-layer scores",
            run_date=run_date,
            scored_rows=existing_rows,
        )

    window_end = current_utc.replace(microsecond=0)
    window_start = window_end - timedelta(hours=cfg.lookback_hours)
    try:
        api_root = _normalize_nepalosint_api_root(cfg.api_base_url)
        if api_root:
            payload = _fetch_nepalosint_root_payload(api_root, config=cfg)
        else:
            payload = _fetch_json(
                cfg.api_base_url,
                api_key=cfg.api_key,
                params={"lookback_hours": cfg.lookback_hours},
                timeout=cfg.request_timeout_secs,
            )
    except Exception as exc:
        return EventLayerResult(status="unavailable", detail=f"NepalOSINT fetch failed: {exc}", run_date=run_date)

    normalized_items, summary_text = _normalize_nepalosint_items(
        payload,
        now_utc=current_utc,
        lookback_hours=cfg.lookback_hours,
    )
    valid_primary = bool(normalized_items or summary_text)
    if not valid_primary:
        return EventLayerResult(status="unavailable", detail="NepalOSINT payload empty or invalid", run_date=run_date)

    fallback_items: List[Dict[str, Any]] = []
    if len(normalized_items) < cfg.min_nepalosint_items:
        try:
            fallback_items = _normalize_fallback_items(current_utc)
        except Exception as exc:
            logger.warning("Fallback news scrape failed: %s", exc)
            fallback_items = []

    merged_items = _dedupe_items(normalized_items + fallback_items)
    news_saved = _store_news_items(merged_items)

    if not cfg.openai_api_key:
        return EventLayerResult(
            status="unavailable",
            detail="OPENAI_API_KEY not configured; raw news stored only",
            run_date=run_date,
            news_saved=news_saved,
            nepalosint_items=len(normalized_items),
            fallback_items=len(fallback_items),
        )

    assessment_payload = _build_assessment_payload(
        merged_items,
        summary_text,
        window_start_utc=window_start.isoformat(),
        window_end_utc=window_end.isoformat(),
    )
    try:
        rows = _call_openai_structured_scores(assessment_payload, cfg)
    except Exception as exc:
        return EventLayerResult(
            status="unavailable",
            detail=f"OpenAI event scoring failed: {exc}",
            run_date=run_date,
            news_saved=news_saved,
            nepalosint_items=len(normalized_items),
            fallback_items=len(fallback_items),
        )

    sanitized_rows = [
        row for row in _sanitize_scored_rows(rows)
        if float(row["confidence"]) >= cfg.min_confidence
    ]
    scored_rows = _replace_scored_rows(
        run_date=run_date,
        window_start_utc=window_start.isoformat(),
        window_end_utc=window_end.isoformat(),
        rows=sanitized_rows,
        model_name=cfg.model,
        prompt_version=cfg.prompt_version,
    )
    detail = "event layer refreshed" if scored_rows else "no material event rows returned"
    return EventLayerResult(
        status="ok",
        detail=detail,
        run_date=run_date,
        news_saved=news_saved,
        scored_rows=scored_rows,
        nepalosint_items=len(normalized_items),
        fallback_items=len(fallback_items),
    )


def load_event_adjustment_context(
    as_of_date: date | datetime | str,
    *,
    min_confidence: Optional[float] = None,
) -> EventAdjustmentContext:
    init_db()
    if isinstance(as_of_date, datetime):
        target_date = as_of_date.date().isoformat()
    elif isinstance(as_of_date, date):
        target_date = as_of_date.isoformat()
    else:
        target_date = str(as_of_date)

    cfg = load_event_layer_config()
    confidence_floor = cfg.min_confidence if min_confidence is None else float(min_confidence)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT run_date
        FROM news_event_scores
        WHERE run_date <= ?
        ORDER BY run_date DESC
        LIMIT 1
        """,
        (target_date,),
    )
    row = cur.fetchone()
    if row is None:
        conn.close()
        return EventAdjustmentContext()
    run_date = str(row[0])
    cur.execute(
        """
        SELECT entity_type, entity_key, impact_score, confidence, rationale_short
        FROM news_event_scores
        WHERE run_date = ? AND confidence >= ?
        """,
        (run_date, confidence_floor),
    )
    rows = cur.fetchall()
    conn.close()

    context = EventAdjustmentContext(run_date=run_date)
    for entity_type, entity_key, impact_score, confidence, rationale_short in rows:
        score = float(impact_score or 0.0)
        key = _normalize_entity_key(entity_key)
        rationale = str(rationale_short or "").strip()
        if entity_type == "market":
            context.market_adjustment = _clamp(context.market_adjustment + score, -0.20, 0.20)
            if rationale:
                context.market_rationale = rationale
        elif entity_type == "sector":
            context.sector_adjustments[key] = _clamp(context.sector_adjustments.get(key, 0.0) + score, -0.20, 0.20)
            if rationale and key:
                context.sector_rationales[key] = rationale
        elif entity_type == "symbol":
            context.symbol_adjustments[key] = _clamp(context.symbol_adjustments.get(key, 0.0) + score, -0.20, 0.20)
            if rationale and key:
                context.symbol_rationales[key] = rationale
    return context


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh NepalOSINT event-layer scores")
    parser.add_argument("--force", action="store_true", help="Recompute scores even if today's rows already exist")
    args = parser.parse_args(argv)
    result = refresh_daily_event_layer(force=args.force)
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
