"""Decision journal persisted in the local control-plane audit database."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Optional

from backend.quant_pro.tms_audit import get_live_audit_db_path, init_live_audit_db
from backend.quant_pro.tms_models import utc_now_iso

from .models import AgentDecision, ApprovalRequest, ApprovalStatus, PolicyVerdict


def _connect() -> sqlite3.Connection:
    init_live_audit_db()
    db_path = get_live_audit_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    _ensure_tables(conn)
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_decisions (
            decision_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            limit_price REAL,
            confidence REAL NOT NULL,
            horizon TEXT,
            thesis TEXT,
            catalysts_json TEXT,
            risk_json TEXT,
            source_signals_json TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            mode TEXT NOT NULL,
            policy_decision TEXT NOT NULL,
            requires_approval INTEGER NOT NULL,
            reasons_json TEXT,
            machine_reasons_json TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS approval_requests (
            intent_id TEXT PRIMARY KEY,
            decision_id TEXT,
            status TEXT NOT NULL,
            operator_surface TEXT,
            summary TEXT,
            expires_at TEXT,
            requested_at TEXT,
            metadata_json TEXT
        )
        """
    )
    conn.commit()


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)


def _load(raw: Optional[str]) -> Any:
    if not raw:
        return None
    return json.loads(raw)


def record_agent_decision(decision: AgentDecision) -> None:
    conn = _connect()
    conn.execute(
        """
        INSERT OR REPLACE INTO agent_decisions (
            decision_id, symbol, action, quantity, limit_price, confidence, horizon,
            thesis, catalysts_json, risk_json, source_signals_json, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision.decision_id,
            decision.symbol,
            decision.action,
            int(decision.quantity),
            float(decision.limit_price) if decision.limit_price is not None else None,
            float(decision.confidence),
            decision.horizon,
            decision.thesis,
            _dump(decision.catalysts),
            _dump(decision.risk),
            _dump(decision.source_signals),
            _dump(decision.metadata),
            utc_now_iso(),
        ),
    )
    conn.commit()
    conn.close()


def record_policy_event(
    *,
    decision_id: Optional[str],
    symbol: str,
    action: str,
    mode: str,
    verdict: PolicyVerdict,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    conn = _connect()
    conn.execute(
        """
        INSERT INTO policy_events (
            decision_id, symbol, action, mode, policy_decision,
            requires_approval, reasons_json, machine_reasons_json, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision_id,
            symbol,
            action,
            mode,
            str(verdict.decision),
            1 if verdict.requires_approval else 0,
            _dump(verdict.reasons),
            _dump(verdict.machine_reasons),
            _dump(metadata or {}),
            utc_now_iso(),
        ),
    )
    conn.commit()
    conn.close()


def create_approval_request(request: ApprovalRequest) -> None:
    conn = _connect()
    conn.execute(
        """
        INSERT OR REPLACE INTO approval_requests (
            intent_id, decision_id, status, operator_surface, summary,
            expires_at, requested_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request.intent_id,
            request.decision_id,
            str(request.status),
            request.operator_surface,
            request.summary,
            request.expires_at,
            request.requested_at or utc_now_iso(),
            _dump(request.metadata),
        ),
    )
    conn.commit()
    conn.close()


def update_approval_request(intent_id: str, *, status: ApprovalStatus, metadata: Optional[Dict[str, Any]] = None) -> None:
    conn = _connect()
    conn.execute(
        """
        UPDATE approval_requests
        SET status = ?, metadata_json = COALESCE(?, metadata_json)
        WHERE intent_id = ?
        """,
        (
            str(status),
            _dump(metadata) if metadata is not None else None,
            intent_id,
        ),
    )
    conn.commit()
    conn.close()


def load_approval_request(intent_id: str) -> Optional[ApprovalRequest]:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM approval_requests WHERE intent_id = ?",
        (intent_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    payload = dict(row)
    return ApprovalRequest(
        intent_id=str(payload["intent_id"]),
        decision_id=payload.get("decision_id"),
        summary=str(payload.get("summary") or ""),
        operator_surface=str(payload.get("operator_surface") or ""),
        status=ApprovalStatus(str(payload.get("status") or ApprovalStatus.PENDING)),
        expires_at=payload.get("expires_at"),
        requested_at=payload.get("requested_at"),
        metadata=_load(payload.get("metadata_json")) or {},
    )
