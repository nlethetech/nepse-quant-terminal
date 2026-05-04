"""
TMS execution model stubs.

Live brokerage execution (TMS19) is not included in this public release.
These stubs allow the codebase to import without errors.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class ExecutionAction(_ValueEnum):
    BUY = "BUY"
    SELL = "SELL"
    CANCEL = "CANCEL"
    MODIFY = "MODIFY"


class ExecutionSource(_ValueEnum):
    SIGNAL = "signal"
    MANUAL = "manual"
    AGENT = "agent"
    OWNER_MANUAL = "owner_manual"
    STRATEGY_ENTRY = "strategy_entry"
    RISK_EXIT = "risk_exit"


class ExecutionStatus(_ValueEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FILLED = "filled"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PENDING_CONFIRMATION = "pending_confirmation"
    QUEUED = "queued"
    SUBMITTING = "submitting"
    ACCEPTED = "accepted"
    SUBMITTED_PENDING = "submitted_pending"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED_PRETRADE = "rejected_pretrade"
    SUBMIT_FAILED = "submit_failed"
    MODIFY_FAILED = "modify_failed"
    FROZEN = "frozen"


class FillState(_ValueEnum):
    PENDING = "pending"
    PARTIAL = "partial"
    FULL = "full"
    FILLED = "filled"


@dataclass
class ExecutionIntent:
    intent_id: str = ""
    symbol: str = ""
    action: ExecutionAction = ExecutionAction.BUY
    source: ExecutionSource = ExecutionSource.SIGNAL
    status: ExecutionStatus = ExecutionStatus.PENDING
    quantity: int = 0
    shares: int = 0
    limit_price: Optional[float] = None
    created_at: str = ""
    completed_at: str = ""
    reason: str = ""
    notes: str = ""
    requires_confirmation: bool = False
    target_order_ref: str = ""
    broker_order_ref: str = ""
    last_error: str = ""
    viewer_notified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0 and self.shares > 0:
            self.quantity = int(self.shares)
        elif self.shares <= 0 and self.quantity > 0:
            self.shares = int(self.quantity)


@dataclass
class ExecutionResult:
    intent_id: str = ""
    success: bool = False
    status: ExecutionStatus = ExecutionStatus.PENDING
    submitted: bool = False
    filled_shares: int = 0
    fill_price: Optional[float] = None
    error: str = ""
    fill_state: FillState = FillState.FULL
    observed_price: Optional[float] = None
    observed_qty: Optional[int] = None
    broker_order_ref: str = ""
    status_text: str = ""
    dom_error_text: str = ""
    screenshot_path: str = ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
