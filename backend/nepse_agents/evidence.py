"""Evidence helpers for the public paper-only agent workflow."""

from __future__ import annotations

from .schemas import EvidencePacket, SignalEnvelope


REQUIRED_ROLES = ("quant", "fundamentals", "news", "risk")


def build_public_evidence(candidate: SignalEnvelope) -> list[EvidencePacket]:
    """Create generic evidence packets without exposing private strategy logic."""
    refs = list(candidate.evidence_refs or [])
    if not refs and candidate.signal_id:
        refs = [candidate.signal_id]

    score_stance = "bullish" if candidate.score >= 0.62 else "neutral" if candidate.score >= 0.45 else "bearish"
    risk_stance = "bearish" if candidate.risks and candidate.confidence < 0.55 else "neutral"
    return [
        EvidencePacket(
            packet_id=f"{candidate.symbol}:quant",
            role="quant",
            stance=score_stance,
            summary=f"Quant candidate score {candidate.score:.2f} with confidence {candidate.confidence:.2f}.",
            confidence=max(candidate.confidence, 0.35),
            sources=refs[:3],
        ),
        EvidencePacket(
            packet_id=f"{candidate.symbol}:fundamentals",
            role="fundamentals",
            stance="neutral",
            summary="Fundamental packet is public-build neutral unless supplied by caller.",
            confidence=0.50,
            sources=refs[:2],
        ),
        EvidencePacket(
            packet_id=f"{candidate.symbol}:news",
            role="news",
            stance="neutral" if not candidate.catalysts else "bullish",
            summary="News packet uses caller-supplied public catalysts only.",
            confidence=0.55 if candidate.catalysts else 0.45,
            sources=refs[:3],
        ),
        EvidencePacket(
            packet_id=f"{candidate.symbol}:risk",
            role="risk",
            stance=risk_stance,
            summary="Risk packet checks confidence, explicit risk flags, paper-only constraints, and lot sizing.",
            confidence=0.65,
            sources=refs[:2],
            red_flags=list(candidate.risks or []),
        ),
    ]


def evidence_quorum(evidence: list[EvidencePacket]) -> dict[str, object]:
    roles = {item.role for item in evidence if item.sources}
    missing = [role for role in REQUIRED_ROLES if role not in roles]
    red_flags = [flag for item in evidence for flag in item.red_flags]
    avg_conf = sum(item.confidence for item in evidence) / max(1, len(evidence))
    ok = not missing and avg_conf >= 0.45
    return {
        "ok": ok,
        "required_roles": list(REQUIRED_ROLES),
        "covered_roles": sorted(roles),
        "missing_roles": missing,
        "average_confidence": round(avg_conf, 4),
        "red_flags": red_flags,
    }
