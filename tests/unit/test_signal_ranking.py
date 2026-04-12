"""Unit tests for shared signal ranking."""

from __future__ import annotations

from backend.quant_pro.event_layer import EventAdjustmentContext
from backend.quant_pro.signal_ranking import is_tradeable_signal_symbol, rank_signal_candidates


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


def test_tradeable_signal_symbol_rejects_index_and_sector_proxies():
    assert is_tradeable_signal_symbol("NABIL")
    assert not is_tradeable_signal_symbol("NEPSE")
    assert not is_tradeable_signal_symbol("SECTOR::BANKING")


def test_rank_signal_candidates_skips_non_tradeable_symbols():
    ranked = rank_signal_candidates(
        [
            {"symbol": "NEPSE", "signal_type": "fundamental", "strength": 0.50, "confidence": 0.50},
            {"symbol": "SECTOR::BANKING", "signal_type": "fundamental", "strength": 0.60, "confidence": 0.50},
            {"symbol": "AAA", "signal_type": "fundamental", "strength": 0.40, "confidence": 0.50},
        ],
        sector_lookup=lambda _symbol: None,
    )

    assert [item["symbol"] for item in ranked] == ["AAA"]
