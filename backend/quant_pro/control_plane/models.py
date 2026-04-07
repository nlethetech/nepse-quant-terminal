"""Typed models for the NEPSE trading control plane."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class TradingMode(StrEnum):
    PAPER = "paper"
    LIVE = "live"
    SHADOW_LIVE = "shadow_live"


class PolicyDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class AgentDecision:
    action: str
    symbol: str
    quantity: int
    limit_price: Optional[float]
    thesis: str
    catalysts: List[str] = field(default_factory=list)
    risk: List[str] = field(default_factory=list)
    confidence: float = 0.0
    horizon: str = ""
    source_signals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    decision_id: str = field(default_factory=lambda: f"agent_{uuid4().hex[:18]}")

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyVerdict:
    decision: PolicyDecision
    reasons: List[str] = field(default_factory=list)
    machine_reasons: List[Dict[str, Any]] = field(default_factory=list)
    requires_approval: bool = False
    approved_mode: Optional[TradingMode] = None

    @property
    def allowed(self) -> bool:
        return self.decision != PolicyDecision.DENY

    def to_record(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = str(self.decision)
        payload["approved_mode"] = str(self.approved_mode) if self.approved_mode else None
        return payload


@dataclass
class ApprovalRequest:
    intent_id: str
    summary: str
    operator_surface: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    expires_at: Optional[str] = None
    requested_at: Optional[str] = None
    decision_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = str(self.status)
        return payload


@dataclass
class CommandResult:
    ok: bool
    status: str
    message: str
    mode: TradingMode
    payload: Dict[str, Any] = field(default_factory=dict)
    intent_id: Optional[str] = None
    approval_request: Optional[ApprovalRequest] = None

    def to_record(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = str(self.mode)
        if self.approval_request is not None:
            payload["approval_request"] = self.approval_request.to_record()
        return payload


@dataclass
class SignalCandidate:
    symbol: str
    signal_type: str
    score: float
    confidence: float
    reasoning: str
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PositionSnapshotItem:
    symbol: str
    quantity: int
    entry_price: float
    last_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    pnl_pct: float
    signal_type: str = ""
    sector: str = ""

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioSnapshot:
    cash: float
    nav: float
    capital: float
    open_positions: int
    positions: List[PositionSnapshotItem] = field(default_factory=list)
    runtime_state: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return {
            "cash": self.cash,
            "nav": self.nav,
            "capital": self.capital,
            "open_positions": self.open_positions,
            "positions": [item.to_record() for item in self.positions],
            "runtime_state": dict(self.runtime_state),
        }


@dataclass
class MarketSnapshot:
    as_of: str
    regime: str
    market_open: bool
    signal_count: int
    last_refresh_nst: Optional[str] = None
    price_source: str = ""
    signals: List[SignalCandidate] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return {
            "as_of": self.as_of,
            "regime": self.regime,
            "market_open": self.market_open,
            "signal_count": self.signal_count,
            "last_refresh_nst": self.last_refresh_nst,
            "price_source": self.price_source,
            "signals": [item.to_record() for item in self.signals],
            "metadata": dict(self.metadata),
        }


@dataclass
class RiskSnapshot:
    halt_level: str
    freeze_reason: str
    max_positions: int
    open_positions: int
    cash: float
    daily_order_cap: Optional[int] = None
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    risk_flags: List[str] = field(default_factory=list)

    def to_record(self) -> Dict[str, Any]:
        return {
            "halt_level": self.halt_level,
            "freeze_reason": self.freeze_reason,
            "max_positions": self.max_positions,
            "open_positions": self.open_positions,
            "cash": self.cash,
            "daily_order_cap": self.daily_order_cap,
            "sector_exposure": dict(self.sector_exposure),
            "risk_flags": list(self.risk_flags),
        }
