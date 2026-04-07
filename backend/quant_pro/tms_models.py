"""Models for local TMS browser automation and execution routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, Optional
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ExecutionSource(StrEnum):
    OWNER_MANUAL = "owner_manual"
    STRATEGY_ENTRY = "strategy_entry"
    STRATEGY_EXIT = "strategy_exit"
    RISK_EXIT = "risk_exit"


class ExecutionAction(StrEnum):
    BUY = "buy"
    SELL = "sell"
    CANCEL = "cancel"
    MODIFY = "modify"


class PriceType(StrEnum):
    LIMIT = "limit"


class ExecutionStatus(StrEnum):
    PENDING_CONFIRMATION = "pending_confirmation"
    QUEUED = "queued"
    SUBMITTING = "submitting"
    ACCEPTED = "accepted"
    REJECTED_PRETRADE = "rejected_pretrade"
    SUBMIT_FAILED = "submit_failed"
    SUBMITTED_PENDING = "submitted_pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    MODIFY_FAILED = "modify_failed"
    FROZEN = "frozen"


class FillState(StrEnum):
    UNKNOWN = "unknown"
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


def new_intent_id(prefix: str = "tms") -> str:
    return f"{prefix}_{uuid4().hex[:18]}"


@dataclass
class ExecutionIntent:
    action: ExecutionAction
    symbol: str
    quantity: int = 0
    limit_price: Optional[float] = None
    source: ExecutionSource = ExecutionSource.OWNER_MANUAL
    price_type: PriceType = PriceType.LIMIT
    time_in_force: str = "DAY"
    strategy_tag: str = ""
    reason: str = ""
    requires_confirmation: bool = False
    status: ExecutionStatus = ExecutionStatus.QUEUED
    intent_id: str = field(default_factory=new_intent_id)
    created_at: str = field(default_factory=utc_now_iso)
    target_order_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    broker_order_ref: Optional[str] = None
    last_error: Optional[str] = None
    confirmed_at: Optional[str] = None
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    paper_mirrored: bool = False
    owner_notified: bool = False
    viewer_notified: bool = False

    def to_record(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["action"] = str(self.action)
        payload["source"] = str(self.source)
        payload["price_type"] = str(self.price_type)
        payload["status"] = str(self.status)
        return payload

    @classmethod
    def from_record(cls, row: Dict[str, Any]) -> "ExecutionIntent":
        data = dict(row)
        data["action"] = ExecutionAction(str(data["action"]))
        data["source"] = ExecutionSource(str(data["source"]))
        data["price_type"] = PriceType(str(data.get("price_type") or PriceType.LIMIT))
        data["status"] = ExecutionStatus(str(data.get("status") or ExecutionStatus.QUEUED))
        data["metadata"] = dict(data.get("metadata") or {})
        return cls(**data)


@dataclass
class ExecutionResult:
    intent_id: str
    status: ExecutionStatus
    submitted: bool
    fill_state: FillState = FillState.UNKNOWN
    status_text: str = ""
    broker_order_ref: Optional[str] = None
    observed_price: Optional[float] = None
    observed_qty: Optional[int] = None
    screenshot_path: Optional[str] = None
    dom_error_text: Optional[str] = None
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    uncertain_submission: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = str(self.status)
        payload["fill_state"] = str(self.fill_state)
        return payload


@dataclass
class SessionStatus:
    ready: bool
    login_required: bool
    current_url: str = ""
    dashboard_text: str = ""
    screenshot_path: Optional[str] = None
    detail: str = ""


@dataclass
class OrderSnapshot:
    broker_order_ref: str
    symbol: str
    action: str
    quantity: int
    price: float
    status_text: str
    fill_state: FillState
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionSnapshot:
    symbol: str
    quantity: int
    average_price: float
    market_value: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)
