from __future__ import annotations

from backend.agents.agent_analyst import (
    _analysis_cache_is_fresh,
    _build_directional_market_answer,
    _merge_agent_output_with_shortlist,
    _question_is_directional_market_call,
    _response_is_hedged_market_call,
    append_external_agent_chat_message,
    load_agent_analysis,
    load_agent_archive_history,
    load_agent_history,
    publish_external_agent_analysis,
)
from scripts.agents.run_codex_agent import extract_json_object


def test_publish_external_agent_analysis_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.agents.agent_analyst.ANALYSIS_FILE",
        tmp_path / "agent_analysis.json",
        raising=False,
    )

    payload = publish_external_agent_analysis(
        {
            "market_view": "Test view",
            "trade_today": True,
            "stocks": [{"symbol": "NABIL", "verdict": "APPROVE", "conviction": 0.8}],
        },
        source="mcp_external",
        provider="ollama",
    )

    restored = load_agent_analysis()

    assert restored["market_view"] == "Test view"
    assert restored["agent_runtime_meta"]["provider"] == "ollama"
    assert payload["stocks"][0]["symbol"] == "NABIL"


def test_append_external_agent_chat_message_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.agents.agent_analyst.AGENT_HISTORY_FILE",
        tmp_path / "agent_chat_history.json",
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst.AGENT_ARCHIVE_FILE",
        tmp_path / "agent_chat_archive.json",
        raising=False,
    )

    append_external_agent_chat_message("AGENT", "hello from mcp", source="mcp_external", provider="claude")
    append_external_agent_chat_message("YOU", "follow up", source="mcp_external", provider="claude")
    history = load_agent_history()

    assert len(history) == 2
    assert history[0]["message"] == "hello from mcp"
    assert history[1]["role"] == "YOU"


def test_agent_chat_rolls_older_messages_into_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.agents.agent_analyst.AGENT_HISTORY_FILE",
        tmp_path / "agent_chat_history.json",
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst.AGENT_ARCHIVE_FILE",
        tmp_path / "agent_chat_archive.json",
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst.MAX_AGENT_HISTORY_ITEMS",
        3,
        raising=False,
    )

    for idx in range(5):
        append_external_agent_chat_message("AGENT", f"message {idx}", source="mcp_external", provider="codex")

    active = load_agent_history()
    archived = load_agent_archive_history()
    combined = load_agent_history(include_archive=True)

    assert [item["message"] for item in active] == ["message 2", "message 3", "message 4"]
    assert [item["message"] for item in archived] == ["message 0", "message 1"]
    assert [item["message"] for item in combined] == [
        "message 0",
        "message 1",
        "message 2",
        "message 3",
        "message 4",
    ]


def test_merge_agent_output_with_shortlist_keeps_ranked_algo_context():
    merged = _merge_agent_output_with_shortlist(
        {
            "trade_today": True,
            "stocks": [
                {
                    "symbol": "NABIL",
                    "verdict": "APPROVE",
                    "conviction": 0.93,
                    "reasoning": "Best bank setup.",
                    "what_matters": "Breakout with earnings support.",
                }
            ],
        },
        {
            "signals": [
                {
                    "symbol": "NABIL",
                    "type": "volume",
                    "direction": "BUY",
                    "strength": 1.2,
                    "confidence": 0.82,
                    "score": 0.98,
                    "reasoning": "Volume breakout",
                    "rank": 1,
                },
                {
                    "symbol": "SBL",
                    "type": "quality",
                    "direction": "BUY",
                    "strength": 0.7,
                    "confidence": 0.6,
                    "score": 0.42,
                    "reasoning": "Quality composite",
                    "rank": 2,
                },
            ],
            "portfolio": [{"symbol": "SBL"}],
            "prices": {"NABIL": 550.0, "SBL": 640.0},
            "signal_metrics": {
                "NABIL": {"profit_margin_pct": 18.0, "pe_ratio": 9.2, "revenue_growth_qoq_pct": 12.0},
                "SBL": {"profit_margin_pct": 11.0, "pbv_ratio": 1.4},
            },
            "symbol_intelligence": {
                "NABIL": {
                    "story_count": 1,
                    "social_count": 1,
                    "related_count": 1,
                    "story_items": [{"title": "NABIL profit jumps on stronger spread income", "source_name": "My Republica"}],
                    "social_items": [{"text": "NABIL looks strong into earnings", "author_username": "nepsealpha"}],
                    "related_items": [],
                },
                "SBL": {"story_count": 0, "social_count": 0, "related_count": 0},
            },
        },
    )

    assert [row["symbol"] for row in merged["stocks"]] == ["NABIL", "SBL"]
    assert merged["stocks"][0]["action_label"] == "BUY"
    assert merged["stocks"][0]["auto_entry_candidate"] is True
    assert merged["stocks"][1]["verdict"] in {"HOLD", "REJECT"}
    assert merged["stocks"][1]["action_label"] in {"HOLD", "SELL"}
    assert merged["stocks"][1]["is_held"] is True


def test_merge_agent_output_with_shortlist_synthesizes_actionable_verdicts():
    merged = _merge_agent_output_with_shortlist(
        {
            "trade_today": True,
            "stocks": [],
        },
        {
            "signals": [
                {
                    "symbol": "HRL",
                    "type": "anchoring",
                    "direction": "BUY",
                    "strength": 1.1,
                    "confidence": 0.84,
                    "score": 0.91,
                    "reasoning": "52w proximity with expanding volume",
                    "rank": 1,
                },
            ],
            "portfolio": [],
            "prices": {"HRL": 612.0},
            "signal_metrics": {
                "HRL": {
                    "sector": "insurance",
                    "profit_margin_pct": 21.0,
                    "revenue_growth_qoq_pct": 14.0,
                    "profit_growth_qoq_pct": 16.0,
                    "pe_ratio": 10.5,
                    "pbv_ratio": 1.8,
                    "roe_pct": 13.5,
                }
            },
            "symbol_intelligence": {
                "HRL": {
                    "story_count": 2,
                    "social_count": 1,
                    "related_count": 2,
                    "story_items": [
                        {
                            "title": "Himalayan Reinsurance posts profit growth and expands treaty book",
                            "source_name": "My Republica",
                        }
                    ],
                    "social_items": [
                        {
                            "text": "HRL earnings momentum looks strong ahead of results",
                            "author_username": "NepseStock",
                        }
                    ],
                    "related_items": [],
                }
            },
        },
    )

    stock = merged["stocks"][0]
    assert stock["verdict"] in {"APPROVE", "REJECT"}
    assert stock["action_label"] in {"BUY", "PASS"}
    assert stock["conviction"] > 0.35
    assert stock["what_matters"]


def test_merge_agent_output_maps_unheld_hold_to_pass():
    merged = _merge_agent_output_with_shortlist(
        {
            "trade_today": True,
            "stocks": [
                {
                    "symbol": "NABIL",
                    "verdict": "HOLD",
                    "conviction": 0.62,
                    "reasoning": "Interesting setup but not enough.",
                    "what_matters": "Needs confirmation.",
                }
            ],
        },
        {
            "signals": [
                {
                    "symbol": "NABIL",
                    "type": "anchoring",
                    "direction": "BUY",
                    "strength": 0.9,
                    "confidence": 0.7,
                    "score": 0.71,
                    "reasoning": "Anchoring setup",
                    "rank": 1,
                },
            ],
            "portfolio": [],
            "prices": {"NABIL": 550.0},
            "signal_metrics": {"NABIL": {"profit_margin_pct": 15.0}},
            "symbol_intelligence": {"NABIL": {"story_count": 0, "social_count": 0, "related_count": 0}},
        },
    )

    stock = merged["stocks"][0]
    assert stock["is_held"] is False
    assert stock["verdict"] == "REJECT"
    assert stock["action_label"] == "PASS"


def test_analysis_cache_requires_current_session_and_recent_timestamp(monkeypatch):
    monkeypatch.setattr(
        "backend.agents.agent_analyst._current_nst_session_date",
        lambda: "2026-04-07",
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst.time.time",
        lambda: 1_000.0,
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst._active_account_context",
        lambda: {"id": "account_1", "name": "Account 1"},
        raising=False,
    )

    assert _analysis_cache_is_fresh(
        {"stocks": [{"symbol": "NABIL"}], "timestamp": 700.0, "context_date": "2026-04-07", "account_id": "account_1"}
    ) is True
    assert _analysis_cache_is_fresh(
        {"stocks": [{"symbol": "NABIL"}], "timestamp": 700.0, "context_date": "2026-04-06", "account_id": "account_1"}
    ) is False
    assert _analysis_cache_is_fresh(
        {"stocks": [{"symbol": "NABIL"}], "timestamp": 1.0, "context_date": "2026-04-07", "account_id": "account_1"}
    ) is False
    assert _analysis_cache_is_fresh(
        {"stocks": [{"symbol": "NABIL"}], "timestamp": 700.0, "context_date": "2026-04-07", "account_id": "account_2"}
    ) is False


def test_extract_json_object_from_codex_output():
    payload = extract_json_object('preface {"regime":"unknown","signal_count":0} tail')

    assert payload == {"regime": "unknown", "signal_count": 0}


def test_directional_market_question_detection_and_fallback():
    assert _question_is_directional_market_call(
        "How would NEPSE react after the news of KP Oli's release?"
    ) is True
    assert _response_is_hedged_market_call("It depends entirely on the content of the news.") is True

    answer = _build_directional_market_answer(
        "How would NEPSE react after the news of KP Oli's release?",
        {"regime": "bull", "fresh_market": {"advancers": 264, "decliners": 3}},
        {"market_phase": "PREOPEN"},
        {"bias": 0.08, "stories": [{"title": "KP Oli release eases political uncertainty"}], "social": []},
    )

    assert answer.startswith("Base case:")
    assert "pressure" in answer
