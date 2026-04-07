"""Unit tests for NepalOSINT event-layer ingestion and loading."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone

from backend.quant_pro.database import get_db_connection, init_db
from backend.quant_pro.event_layer import (
    EventLayerConfig,
    load_event_adjustment_context,
    refresh_daily_event_layer,
)


def test_refresh_daily_event_layer_normalizes_dedupes_and_stores(monkeypatch, tmp_path):
    db_file = tmp_path / "event_layer.db"
    monkeypatch.setenv("NEPSE_DB_FILE", str(db_file))

    import backend.quant_pro.database as db_mod
    db_mod._wal_initialized = False
    init_db()

    monkeypatch.setattr(
        "backend.quant_pro.event_layer._fetch_json",
        lambda *args, **kwargs: {
            "national_assessment": "Government policy is supportive for banks.",
            "items": [
                {
                    "id": "evt-1",
                    "headline": "NRB eases lending rules for banks",
                    "url": "https://example.com/a",
                    "published_at": "2026-03-30T02:00:00Z",
                    "source": "nepalosint",
                    "symbols": ["NABIL"],
                    "sectors": ["Commercial Banks"],
                    "category": "policy",
                },
                {
                    "id": "evt-1",
                    "headline": "NRB eases lending rules for banks",
                    "url": "https://example.com/a",
                    "published_at": "2026-03-30T02:05:00Z",
                    "source": "nepalosint",
                    "symbols": ["NABIL"],
                    "sectors": ["Commercial Banks"],
                    "category": "policy",
                },
                {
                    "id": "evt-2",
                    "headline": "Cabinet approves tourism promotion package",
                    "published_at": "2026-03-30T03:00:00Z",
                    "source": "nepalosint",
                    "sectors": ["Hotels & Tourism"],
                    "category": "government",
                },
            ],
        },
    )
    monkeypatch.setattr("backend.quant_pro.event_layer._normalize_fallback_items", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "backend.quant_pro.event_layer._call_openai_structured_scores",
        lambda *_args, **_kwargs: [
            {
                "entity_type": "symbol",
                "entity_key": "NABIL",
                "impact_direction": "positive",
                "impact_score": 0.12,
                "confidence": 0.88,
                "event_type": "policy",
                "source_refs": ["https://example.com/a"],
                "rationale_short": "Bank regulation became more supportive.",
            }
        ],
    )

    result = refresh_daily_event_layer(
        config=EventLayerConfig(
            enabled=True,
            api_base_url="https://api.nepalosint.test/events",
            lookback_hours=12,
            model="test-model",
            min_confidence=0.4,
            openai_api_key="test-key",
            min_nepalosint_items=1,
        ),
        now_utc=datetime(2026, 3, 30, 4, 0, tzinfo=timezone.utc),
        force=True,
    )

    assert result.status == "ok"
    assert result.news_saved == 2
    assert result.scored_rows == 1

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM news")
    assert cur.fetchone()[0] == 2
    cur.execute("SELECT headline, url FROM news ORDER BY headline")
    rows = cur.fetchall()
    assert rows[0][0] == "Cabinet approves tourism promotion package"
    assert rows[1][1] == "https://example.com/a"
    cur.execute("SELECT entity_type, entity_key, impact_score FROM news_event_scores")
    assert cur.fetchone() == ("symbol", "NABIL", 0.12)
    conn.close()


def test_load_event_adjustment_context_uses_latest_run_and_symbol_override(monkeypatch, tmp_path):
    db_file = tmp_path / "event_context.db"
    monkeypatch.setenv("NEPSE_DB_FILE", str(db_file))

    import backend.quant_pro.database as db_mod
    db_mod._wal_initialized = False
    init_db()

    conn = get_db_connection()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO news_event_scores
        (run_date, window_start_utc, window_end_utc, entity_type, entity_key,
         impact_direction, impact_score, confidence, event_type, source_count,
         source_refs_json, rationale_short, model_name, prompt_version, created_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "2026-03-29",
                "2026-03-28T12:00:00+00:00",
                "2026-03-29T00:00:00+00:00",
                "sector",
                "HYDROPOWER",
                "negative",
                -0.15,
                0.80,
                "government",
                1,
                '["ref-1"]',
                "Hydropower faced a policy setback.",
                "test-model",
                "v1",
                "2026-03-29T00:00:00+00:00",
            ),
            (
                "2026-03-29",
                "2026-03-28T12:00:00+00:00",
                "2026-03-29T00:00:00+00:00",
                "symbol",
                "API",
                "positive",
                0.10,
                0.90,
                "company",
                1,
                '["ref-2"]',
                "API received a direct positive catalyst.",
                "test-model",
                "v1",
                "2026-03-29T00:00:00+00:00",
            ),
        ],
    )
    conn.commit()
    conn.close()

    context = load_event_adjustment_context(date(2026, 3, 30), min_confidence=0.5)
    details = context.details_for("API", "Hydropower")

    assert context.run_date == "2026-03-29"
    assert details["symbol_adjustment"] == 0.10
    assert details["sector_adjustment"] == 0.0
    assert round(details["event_adjustment"], 4) == 0.10


def test_refresh_daily_event_layer_supports_api_root_mode(monkeypatch, tmp_path):
    db_file = tmp_path / "event_layer_root_mode.db"
    monkeypatch.setenv("NEPSE_DB_FILE", str(db_file))

    import backend.quant_pro.database as db_mod
    db_mod._wal_initialized = False
    init_db()

    auth_calls = []
    fetch_calls = []

    def fake_post_json(url, **kwargs):
        auth_calls.append((url, kwargs.get("payload")))
        assert url == "https://nepalosint.com/api/v1/auth/guest"
        return {"access_token": "guest-token"}

    def fake_fetch_json(url, **kwargs):
        fetch_calls.append((url, kwargs.get("params"), kwargs.get("bearer_token")))
        assert kwargs.get("bearer_token") == "guest-token"
        if url.endswith("/stories/export"):
            return {
                "stories": [
                    {
                        "id": "story-1",
                        "title": "Nepal Rastra Bank signals easier liquidity conditions",
                        "source_name": "NepalOSINT",
                        "published_at": "2026-03-30T02:00:00Z",
                        "category": "economic",
                        "summary": "Banking conditions improved for lenders.",
                    }
                ]
            }
        if url.endswith("/announcements/summary"):
            return {
                "latest": [
                    {
                        "id": "ann-1",
                        "title": "Cabinet clears tourism promotion package",
                        "url": "https://example.com/ann",
                        "source": "opmcm.gov.np",
                        "source_name": "OPMCM",
                        "category": "cabinet",
                        "content": "Government approved tourism-sector support measures.",
                        "published_at": "2026-03-30T01:00:00Z",
                    }
                ]
            }
        if url.endswith("/analytics/executive-summary"):
            return {
                "situation_overview": "Economic activity is stable with selective policy support.",
                "key_judgment": "Financial and tourism-sensitive names may benefit from the current policy tone.",
                "watch_items": ["Monitor banking liquidity", "Watch cabinet policy execution"],
            }
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr("backend.quant_pro.event_layer._post_json", fake_post_json)
    monkeypatch.setattr("backend.quant_pro.event_layer._fetch_json", fake_fetch_json)
    monkeypatch.setattr("backend.quant_pro.event_layer._normalize_fallback_items", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("backend.quant_pro.event_layer._call_openai_structured_scores", lambda *_args, **_kwargs: [])

    result = refresh_daily_event_layer(
        config=EventLayerConfig(
            enabled=True,
            api_base_url="https://nepalosint.com/api/v1",
            openai_api_key="test-key",
            use_guest_login=True,
            min_nepalosint_items=1,
        ),
        now_utc=datetime(2026, 3, 30, 4, 0, tzinfo=timezone.utc),
        force=True,
    )

    assert result.status == "ok"
    assert result.news_saved == 2
    assert auth_calls == [("https://nepalosint.com/api/v1/auth/guest", {})]
    assert [call[0] for call in fetch_calls] == [
        "https://nepalosint.com/api/v1/stories/export",
        "https://nepalosint.com/api/v1/announcements/summary",
        "https://nepalosint.com/api/v1/analytics/executive-summary",
    ]

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT headline FROM news ORDER BY headline")
    headlines = [row[0] for row in cur.fetchall()]
    assert headlines == [
        "Cabinet clears tourism promotion package",
        "Nepal Rastra Bank signals easier liquidity conditions",
    ]
    conn.close()
