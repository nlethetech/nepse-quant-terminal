"""Dataclass contracts for the public paper-only agent workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from uuid import uuid4


def _bounded(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value or 0.0)))


@dataclass
class SignalEnvelope:
    """Generic quant candidate passed into the agent workflow."""

    symbol: str
    as_of_date: str = ""
    sector: str = ""
    signal_id: str = ""
    strategy_id: str = ""
    score: float = 0.0
    strength: float = 0.0
    confidence: float = 0.0
    rank: int = 0
    regime: str = "neutral"
    suggested_quantity: int = 0
    suggested_limit_price: float = 0.0
    suggested_horizon_days: int = 20
    thesis: str = ""
    catalysts: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol or "").upper().strip()
        self.sector = str(self.sector or "Unknown").strip()
        self.score = _bounded(self.score)
        self.strength = _bounded(self.strength)
        self.confidence = _bounded(self.confidence)
        self.suggested_quantity = max(0, int(self.suggested_quantity or 0))
        self.suggested_limit_price = max(0.0, float(self.suggested_limit_price or 0.0))
        self.suggested_horizon_days = max(1, int(self.suggested_horizon_days or 20))

    @classmethod
    def from_record(cls, payload: dict[str, Any]) -> "SignalEnvelope":
        raw = dict(payload or {})
        return cls(
            symbol=raw.get("symbol", ""),
            as_of_date=raw.get("as_of_date") or raw.get("date") or "",
            sector=raw.get("sector", ""),
            signal_id=raw.get("signal_id", ""),
            strategy_id=raw.get("strategy_id", ""),
            score=raw.get("score") or raw.get("composite_score") or 0.0,
            strength=raw.get("strength") or raw.get("score") or 0.0,
            confidence=raw.get("confidence") or 0.0,
            rank=raw.get("rank") or raw.get("rank_among_universe") or 0,
            regime=raw.get("regime", "neutral"),
            suggested_quantity=raw.get("suggested_quantity") or raw.get("quantity") or 0,
            suggested_limit_price=raw.get("suggested_limit_price") or raw.get("limit_price") or raw.get("price") or 0.0,
            suggested_horizon_days=raw.get("suggested_horizon_days") or raw.get("horizon_days") or 20,
            thesis=raw.get("thesis") or raw.get("reasoning") or "",
            catalysts=list(raw.get("catalysts") or []),
            risks=list(raw.get("risks") or raw.get("risk") or []),
            evidence_refs=list(raw.get("evidence_refs") or raw.get("source_signals") or []),
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidencePacket:
    packet_id: str
    role: str
    stance: Literal["bullish", "bearish", "neutral"]
    summary: str
    confidence: float
    sources: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = _bounded(self.confidence)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DebateTurn:
    side: Literal["bull", "bear", "aggressive", "neutral", "conservative"]
    thesis: str
    evidence_cited: list[str] = field(default_factory=list)
    concession: str = ""

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchPlan:
    rating: Literal["Buy", "Overweight", "Hold", "Underweight", "Skip"]
    synthesis: str
    confidence: float
    horizon_days: int
    expected_move_pct: float = 0.0

    def __post_init__(self) -> None:
        self.confidence = _bounded(self.confidence)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraderProposal:
    action: Literal["BUY", "HOLD"]
    symbol: str
    quantity: int
    limit_price: float
    horizon_days: int
    rationale: str
    stop_loss_pct: float = 0.08
    profit_target_pct: float = 0.16

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskVerdict:
    final_quantity: int
    final_stop_pct: float
    final_horizon_days: int
    blocked: bool = False
    block_reason: str = ""
    committee_notes: dict[str, str] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioDecision:
    symbol: str
    action: Literal["APPROVE", "HOLD", "REJECT"]
    final_quantity: int
    final_limit_price: float
    horizon_days: int
    conviction: float
    thesis: str
    evidence_packet_ids: list[str]
    risks: list[str] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)
    rating: str = "Hold"
    execution_mode: Literal["paper"] = "paper"
    evidence_gate_ok: bool = False
    decision_id: str = field(default_factory=lambda: f"agent_{uuid4().hex[:18]}")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.conviction = _bounded(self.conviction)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
