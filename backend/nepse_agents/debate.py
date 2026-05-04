"""Deterministic debate passes for the public paper-only agent workflow."""

from __future__ import annotations

from .schemas import DebateTurn, EvidencePacket, SignalEnvelope


def run_research_debate(candidate: SignalEnvelope, evidence: list[EvidencePacket]) -> list[DebateTurn]:
    bullish = [item.packet_id for item in evidence if item.stance == "bullish"]
    bearish = [item.packet_id for item in evidence if item.stance == "bearish"]
    return [
        DebateTurn(
            side="bull",
            thesis=(
                f"Bull case for {candidate.symbol}: score {candidate.score:.2f}, public catalysts "
                f"{', '.join(candidate.catalysts[:3]) or 'not supplied'}, and no private data required."
            ),
            evidence_cited=bullish or [item.packet_id for item in evidence[:2]],
            concession="Upside case is weaker when public catalysts are sparse.",
        ),
        DebateTurn(
            side="bear",
            thesis=(
                f"Bear case for {candidate.symbol}: confidence {candidate.confidence:.2f}, explicit risks "
                f"{', '.join(candidate.risks[:3]) or 'not supplied'}, and public build must fail closed."
            ),
            evidence_cited=bearish or [item.packet_id for item in evidence[-2:]],
            concession="Bear case concedes a high-confidence, well-sourced quant setup can proceed in paper mode.",
        ),
    ]


def run_risk_debate(candidate: SignalEnvelope) -> list[DebateTurn]:
    qty = int(candidate.suggested_quantity or 0)
    return [
        DebateTurn("aggressive", f"Aggressive risk view accepts paper quantity up to {qty} if evidence quorum passes."),
        DebateTurn("neutral", f"Neutral risk view sizes {candidate.symbol} only within supplied lot and confidence constraints."),
        DebateTurn("conservative", f"Conservative risk view blocks approval if risks are unresolved or price is missing."),
    ]
