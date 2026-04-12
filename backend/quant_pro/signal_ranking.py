"""Shared signal ranking helpers for paper trading and backtests."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional

from .event_layer import EventAdjustmentContext


def is_tradeable_signal_symbol(symbol: object) -> bool:
    clean = str(symbol or "").strip().upper()
    if not clean:
        return False
    if clean == "NEPSE":
        return False
    if clean.startswith("SECTOR::"):
        return False
    return True


def _base_signal_score(signal: Dict[str, Any]) -> float:
    return float(signal.get("strength", 0.0) or 0.0) * float(signal.get("confidence", 0.0) or 0.0)


def _coerce_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    strength = float(signal.get("strength", 0.0) or 0.0)
    confidence = float(signal.get("confidence", 0.0) or 0.0)
    return {
        "symbol": str(signal.get("symbol") or "").strip().upper(),
        "signal_type": str(signal.get("signal_type") or "unknown").strip() or "unknown",
        "strength": strength,
        "confidence": confidence,
        "reasoning": str(signal.get("reasoning") or "").strip(),
        "score": float(signal.get("score", strength * confidence) or 0.0),
        "target_exit_date": signal.get("target_exit_date"),
    }


def _sector_weight_penalty(weight: float) -> float:
    if weight >= 0.30:
        return 0.75
    if weight >= 0.20:
        return 0.90
    return 1.0


def _signal_type_penalty(count: int) -> float:
    if count <= 0:
        return 1.0
    if count == 1:
        return 0.95
    return 0.85


def merge_signal_candidates(signals: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for raw in signals:
        signal = _coerce_signal(raw)
        if not is_tradeable_signal_symbol(signal["symbol"]):
            continue
        grouped[signal["symbol"]].append(signal)

    merged: List[Dict[str, Any]] = []
    for symbol, entries in grouped.items():
        ranked_entries = sorted(
            entries,
            key=lambda item: (
                -_base_signal_score(item),
                -float(item["confidence"]),
                str(item["signal_type"]),
            ),
        )
        primary = ranked_entries[0]
        support_count = len(ranked_entries)
        raw_score = _base_signal_score(primary) + min(0.08, 0.03 * max(0, support_count - 1))
        reasons: List[str] = []
        for item in ranked_entries:
            reason = str(item.get("reasoning") or "").strip()
            if reason and reason not in reasons:
                reasons.append(reason)
            if len(reasons) >= 2:
                break
        merged.append(
            {
                "symbol": symbol,
                "signal_type": primary["signal_type"],
                "primary_signal_type": primary["signal_type"],
                "signal_types": sorted({str(item["signal_type"]) for item in ranked_entries}),
                "strength": float(primary["strength"]),
                "confidence": float(primary["confidence"]),
                "support_count": support_count,
                "raw_score": float(raw_score),
                "reasoning": " | ".join(reasons) if reasons else str(primary.get("reasoning") or ""),
                "target_exit_date": primary.get("target_exit_date"),
            }
        )
    return merged


def rank_signal_candidates(
    signals: Iterable[Dict[str, Any]],
    *,
    held_symbols: Optional[Iterable[str]] = None,
    sector_exposure: Optional[Dict[str, float]] = None,
    sector_lookup: Optional[Callable[[str], Optional[str]]] = None,
    event_context: Optional[EventAdjustmentContext] = None,
) -> List[Dict[str, Any]]:
    held = {str(symbol).strip().upper() for symbol in (held_symbols or []) if str(symbol).strip()}
    exposure = {str(k).strip().upper(): float(v) for k, v in (sector_exposure or {}).items()}
    lookup = sector_lookup or (lambda _symbol: None)
    context = event_context or EventAdjustmentContext()

    remaining = [item for item in merge_signal_candidates(signals) if item["symbol"] not in held]
    ordered: List[Dict[str, Any]] = []
    selected_sectors: Dict[str, int] = defaultdict(int)
    selected_signal_types: Dict[str, int] = defaultdict(int)

    while remaining:
        scored: List[Dict[str, Any]] = []
        for candidate in remaining:
            sector = str(lookup(candidate["symbol"]) or "").strip()
            sector_key = sector.upper()
            base_penalty = _sector_weight_penalty(exposure.get(sector_key, 0.0))
            cycle_sector_penalty = 0.90 if sector_key and selected_sectors.get(sector_key, 0) > 0 else 1.0
            signal_type_penalty = _signal_type_penalty(selected_signal_types.get(candidate["signal_type"], 0))
            base_rank_score = float(candidate["raw_score"]) * base_penalty * cycle_sector_penalty * signal_type_penalty
            event_details = context.details_for(candidate["symbol"], sector)
            final_rank_score = base_rank_score * (1.0 + float(event_details["event_adjustment"]))
            reasoning = str(candidate.get("reasoning") or "").strip()
            if event_details["event_rationale"]:
                reasoning = f"{reasoning} | {event_details['event_rationale']}".strip(" |")
            scored.append(
                {
                    **candidate,
                    "sector": sector or None,
                    "base_rank_score": float(base_rank_score),
                    "score": float(final_rank_score),
                    "event_adjustment": float(event_details["event_adjustment"]),
                    "market_adjustment": float(event_details["market_adjustment"]),
                    "sector_adjustment": float(event_details["sector_adjustment"]),
                    "symbol_adjustment": float(event_details["symbol_adjustment"]),
                    "reasoning": reasoning,
                }
            )

        scored.sort(
            key=lambda item: (
                -float(item["score"]),
                -float(item["raw_score"]),
                -int(item["support_count"]),
                -float(item["confidence"]),
                str(item["symbol"]),
            )
        )
        chosen = scored[0]
        chosen["rank_position"] = len(ordered) + 1
        ordered.append(chosen)
        remaining = [item for item in remaining if item["symbol"] != chosen["symbol"]]
        sector_key = str(chosen.get("sector") or "").upper()
        if sector_key:
            selected_sectors[sector_key] += 1
        selected_signal_types[str(chosen["signal_type"])] += 1

    return ordered
