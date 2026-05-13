"""Unit tests for shared signal ranking."""

from __future__ import annotations

from backend.quant_pro.event_layer import EventAdjustmentContext
from backend.quant_pro.signal_ranking import (
    blocked_signal_symbol_reason,
    is_tradeable_signal_symbol,
    rank_signal_candidates,
)


def test_tradeable_policy_blocks_indexes_and_suspended_symbols():
    assert not is_tradeable_signal_symbol("NEPSE")
    assert not is_tradeable_signal_symbol("SECTOR::BANKING")
    assert not is_tradeable_signal_symbol("KRBL")
    assert not is_tradeable_signal_symbol("UIC")
    assert blocked_signal_symbol_reason("KRBL") == "suspended_or_non_tradeable_symbol"
    assert is_tradeable_signal_symbol("NABIL")


def test_tradeable_policy_honors_env_blocklist(monkeypatch):
    monkeypatch.setenv("NEPSE_BLOCKED_SYMBOLS", "AAA, BBB")

    assert not is_tradeable_signal_symbol("AAA")
    assert not is_tradeable_signal_symbol("BBB")
    assert blocked_signal_symbol_reason("AAA") == "suspended_or_non_tradeable_symbol"


def test_rank_signal_candidates_merges_duplicate_symbol_support():
    ranked = rank_signal_candidates(
        [
            {
                "symbol": "AAA",
                "signal_type": "fundamental",
                "strength": 0.40,
                "confidence": 0.50,
                "reasoning": "Strong quality metrics",
            },
            {
                "symbol": "AAA",
                "signal_type": "liquidity",
                "strength": 0.30,
                "confidence": 0.50,
                "reasoning": "Volume accelerated",
            },
        ],
        sector_lookup=lambda _symbol: None,
    )

    assert len(ranked) == 1
    assert ranked[0]["symbol"] == "AAA"
    assert ranked[0]["support_count"] == 2
    assert ranked[0]["signal_types"] == ["fundamental", "liquidity"]
    assert round(ranked[0]["raw_score"], 4) == 0.23


def test_positive_symbol_event_boosts_candidate_over_peer():
    ranked = rank_signal_candidates(
        [
            {"symbol": "AAA", "signal_type": "fundamental", "strength": 0.40, "confidence": 0.50, "reasoning": "A"},
            {"symbol": "BBB", "signal_type": "fundamental", "strength": 0.39, "confidence": 0.50, "reasoning": "B"},
        ],
        sector_lookup=lambda _symbol: None,
        event_context=EventAdjustmentContext(symbol_adjustments={"BBB": 0.20}),
    )

    assert ranked[0]["symbol"] == "BBB"
    assert ranked[0]["event_adjustment"] == 0.20


def test_negative_sector_event_penalizes_sector():
    ranked = rank_signal_candidates(
        [
            {"symbol": "AAA", "signal_type": "fundamental", "strength": 0.40, "confidence": 0.50, "reasoning": "A"},
            {"symbol": "BBB", "signal_type": "fundamental", "strength": 0.40, "confidence": 0.50, "reasoning": "B"},
        ],
        sector_lookup=lambda symbol: "Hydropower" if symbol == "AAA" else "Commercial Banks",
        event_context=EventAdjustmentContext(sector_adjustments={"HYDROPOWER": -0.15}),
    )

    assert ranked[0]["symbol"] == "BBB"
    assert ranked[1]["symbol"] == "AAA"


def test_symbol_override_beats_negative_sector_tone():
    ranked = rank_signal_candidates(
        [
            {"symbol": "AAA", "signal_type": "fundamental", "strength": 0.40, "confidence": 0.50, "reasoning": "A"},
            {"symbol": "BBB", "signal_type": "fundamental", "strength": 0.40, "confidence": 0.50, "reasoning": "B"},
        ],
        sector_lookup=lambda _symbol: "Hydropower",
        event_context=EventAdjustmentContext(
            sector_adjustments={"HYDROPOWER": -0.15},
            symbol_adjustments={"AAA": 0.10},
        ),
    )

    assert ranked[0]["symbol"] == "AAA"
    assert ranked[0]["sector_adjustment"] == 0.0
    assert ranked[1]["sector_adjustment"] == -0.15


def test_missing_event_data_keeps_base_order():
    ranked = rank_signal_candidates(
        [
            {"symbol": "AAA", "signal_type": "fundamental", "strength": 0.45, "confidence": 0.50, "reasoning": "A"},
            {"symbol": "BBB", "signal_type": "fundamental", "strength": 0.35, "confidence": 0.50, "reasoning": "B"},
        ],
        sector_lookup=lambda _symbol: None,
        event_context=EventAdjustmentContext(),
    )

    assert [item["symbol"] for item in ranked] == ["AAA", "BBB"]
