"""SQLite audit journal for local TMS execution and read-only monitoring."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .tms_models import ExecutionIntent, ExecutionResult, FillState, utc_now_iso
from .paths import ensure_dir, get_runtime_dir


def get_live_audit_db_path() -> Path:
    raw = os.environ.get("NEPSE_LIVE_AUDIT_DB_FILE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return ensure_dir(get_runtime_dir(__file__)) / "nepse_live_audit.db"


def _connect() -> sqlite3.Connection:
    path = get_live_audit_db_path()
    conn = sqlite3.connect(str(path), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)


def _json_loads(raw: Optional[str]) -> Any:
    if not raw:
        return None
    return json.loads(raw)


def init_live_audit_db() -> None:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_intents (
            intent_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            action TEXT NOT NULL,
            symbol TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 0,
            price_type TEXT NOT NULL,
            limit_price REAL,
            time_in_force TEXT NOT NULL,
            strategy_tag TEXT,
            reason TEXT,
            target_order_ref TEXT,
            requires_confirmation INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            confirmed_at TEXT,
            submitted_at TEXT,
            completed_at TEXT,
            broker_order_ref TEXT,
            last_error TEXT,
            metadata_json TEXT,
            paper_mirrored INTEGER NOT NULL DEFAULT 0,
            owner_notified INTEGER NOT NULL DEFAULT 0,
            viewer_notified INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_execution_intents_status ON execution_intents (status, created_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_execution_intents_symbol ON execution_intents (symbol, created_at DESC)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_attempts (
            attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            detail TEXT,
            broker_order_ref TEXT,
            screenshot_path TEXT,
            dom_error_text TEXT,
            payload_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_execution_attempts_intent ON execution_attempts (intent_id, created_at DESC)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS live_order_states (
            state_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent_id TEXT,
            broker_order_ref TEXT,
            symbol TEXT,
            action TEXT,
            quantity INTEGER,
            price REAL,
            status_text TEXT,
            fill_state TEXT,
            source TEXT NOT NULL,
            payload_json TEXT,
            recorded_at TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_live_order_states_ref ON live_order_states (broker_order_ref, recorded_at DESC)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS live_position_states (
            position_state_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            average_price REAL,
            market_value REAL,
            source TEXT NOT NULL,
            payload_json TEXT,
            recorded_at TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_live_position_states_symbol ON live_position_states (symbol, recorded_at DESC)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reconciliation_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL,
            summary_json TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tms_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_type TEXT NOT NULL,
            status TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tms_snapshots_type_time ON tms_snapshots (snapshot_type, recorded_at DESC, snapshot_id DESC)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_decisions (
            decision_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 0,
            limit_price REAL,
            confidence REAL NOT NULL DEFAULT 0,
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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_decisions_symbol_time ON agent_decisions (symbol, created_at DESC)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS approval_requests (
            intent_id TEXT PRIMARY KEY,
            decision_id TEXT,
            status TEXT NOT NULL,
            operator_surface TEXT NOT NULL,
            summary TEXT NOT NULL,
            expires_at TEXT,
            requested_at TEXT NOT NULL,
            metadata_json TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_approval_requests_status_time ON approval_requests (status, requested_at DESC)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            mode TEXT NOT NULL,
            policy_decision TEXT NOT NULL,
            requires_approval INTEGER NOT NULL DEFAULT 0,
            reasons_json TEXT,
            machine_reasons_json TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_policy_events_symbol_time ON policy_events (symbol, created_at DESC)")
    conn.commit()
    conn.close()


def save_execution_intent(intent: ExecutionIntent) -> None:
    init_live_audit_db()
    conn = _connect()
    conn.execute(
        """
        INSERT OR REPLACE INTO execution_intents (
            intent_id, source, action, symbol, quantity, price_type, limit_price,
            time_in_force, strategy_tag, reason, target_order_ref,
            requires_confirmation, status, created_at, confirmed_at, submitted_at,
            completed_at, broker_order_ref, last_error, metadata_json,
            paper_mirrored, owner_notified, viewer_notified
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            intent.intent_id,
            str(intent.source),
            str(intent.action),
            intent.symbol,
            int(intent.quantity),
            str(intent.price_type),
            float(intent.limit_price) if intent.limit_price is not None else None,
            intent.time_in_force,
            intent.strategy_tag,
            intent.reason,
            intent.target_order_ref,
            1 if intent.requires_confirmation else 0,
            str(intent.status),
            intent.created_at,
            intent.confirmed_at,
            intent.submitted_at,
            intent.completed_at,
            intent.broker_order_ref,
            intent.last_error,
            _json_dumps(intent.metadata),
            1 if intent.paper_mirrored else 0,
            1 if intent.owner_notified else 0,
            1 if intent.viewer_notified else 0,
        ),
    )
    conn.commit()
    conn.close()


def update_execution_intent(intent_id: str, **fields: Any) -> None:
    if not fields:
        return
    init_live_audit_db()
    conn = _connect()
    cur = conn.cursor()
    assignments = []
    values: List[Any] = []
    for key, value in fields.items():
        column = {
            "metadata": "metadata_json",
        }.get(key, key)
        assignments.append(f"{column} = ?")
        if key == "metadata":
            values.append(_json_dumps(value))
        elif isinstance(value, bool):
            values.append(1 if value else 0)
        else:
            values.append(str(value) if key in {"status", "fill_state"} and value is not None else value)
    values.append(intent_id)
    cur.execute(
        f"UPDATE execution_intents SET {', '.join(assignments)} WHERE intent_id = ?",
        values,
    )
    conn.commit()
    conn.close()


def load_execution_intent(intent_id: str) -> Optional[ExecutionIntent]:
    init_live_audit_db()
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM execution_intents WHERE intent_id = ?",
        (intent_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    payload = dict(row)
    payload["metadata"] = _json_loads(payload.pop("metadata_json")) or {}
    payload["requires_confirmation"] = bool(payload["requires_confirmation"])
    payload["paper_mirrored"] = bool(payload["paper_mirrored"])
    payload["owner_notified"] = bool(payload["owner_notified"])
    payload["viewer_notified"] = bool(payload["viewer_notified"])
    return ExecutionIntent.from_record(payload)


def list_execution_intents(
    *,
    statuses: Optional[Iterable[str]] = None,
    limit: int = 25,
) -> List[ExecutionIntent]:
    init_live_audit_db()
    conn = _connect()
    query = "SELECT * FROM execution_intents"
    params: List[Any] = []
    if statuses:
        clean = [str(status) for status in statuses]
        query += f" WHERE status IN ({','.join('?' for _ in clean)})"
        params.extend(clean)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(int(limit))
    rows = conn.execute(query, params).fetchall()
    conn.close()
    items = []
    for row in rows:
        payload = dict(row)
        payload["metadata"] = _json_loads(payload.pop("metadata_json")) or {}
        payload["requires_confirmation"] = bool(payload["requires_confirmation"])
        payload["paper_mirrored"] = bool(payload["paper_mirrored"])
        payload["owner_notified"] = bool(payload["owner_notified"])
        payload["viewer_notified"] = bool(payload["viewer_notified"])
        items.append(ExecutionIntent.from_record(payload))
    return items


def count_intents_for_day(day_str: str) -> int:
    init_live_audit_db()
    conn = _connect()
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM execution_intents WHERE substr(created_at, 1, 10) = ?",
        (str(day_str),),
    ).fetchone()
    conn.close()
    return int(row["c"]) if row else 0


def find_recent_open_intent(symbol: str, *, within_seconds: Optional[int] = None) -> Optional[ExecutionIntent]:
    init_live_audit_db()
    conn = _connect()
    query = """
        SELECT * FROM execution_intents
        WHERE symbol = ?
          AND status IN ('pending_confirmation', 'queued', 'submitting', 'accepted', 'submitted_pending', 'partially_filled')
        ORDER BY created_at DESC
        LIMIT 1
    """
    row = conn.execute(query, (str(symbol).upper(),)).fetchone()
    conn.close()
    if row is None:
        return None
    intent = load_execution_intent(str(row["intent_id"]))
    if intent is None:
        return None
    if within_seconds is not None:
        try:
            recent_cutoff = datetime.fromisoformat(utc_now_iso())  # type: ignore[name-defined]
        except Exception:
            recent_cutoff = None
        try:
            created = datetime.fromisoformat(intent.created_at)  # type: ignore[name-defined]
        except Exception:
            created = None
        if recent_cutoff and created and (recent_cutoff - created).total_seconds() > float(within_seconds):
            return None
    return intent


def log_execution_attempt(
    intent_id: str,
    *,
    stage: str,
    status: str,
    detail: str = "",
    broker_order_ref: Optional[str] = None,
    screenshot_path: Optional[str] = None,
    dom_error_text: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    init_live_audit_db()
    conn = _connect()
    conn.execute(
        """
        INSERT INTO execution_attempts (
            intent_id, stage, status, detail, broker_order_ref, screenshot_path,
            dom_error_text, payload_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            intent_id,
            stage,
            status,
            detail,
            broker_order_ref,
            screenshot_path,
            dom_error_text,
            _json_dumps(payload or {}),
            utc_now_iso(),
        ),
    )
    conn.commit()
    conn.close()


def save_execution_result(intent: ExecutionIntent, result: ExecutionResult, *, source: str) -> None:
    init_live_audit_db()
    conn = _connect()
    conn.execute(
        """
        INSERT INTO live_order_states (
            intent_id, broker_order_ref, symbol, action, quantity, price,
            status_text, fill_state, source, payload_json, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            intent.intent_id,
            result.broker_order_ref,
            intent.symbol,
            str(intent.action),
            int(result.observed_qty if result.observed_qty is not None else intent.quantity),
            float(result.observed_price if result.observed_price is not None else (intent.limit_price or 0.0)),
            result.status_text,
            str(result.fill_state),
            source,
            _json_dumps(result.payload),
            utc_now_iso(),
        ),
    )
    conn.commit()
    conn.close()
    update_execution_intent(
        intent.intent_id,
        status=str(result.status),
        broker_order_ref=result.broker_order_ref or intent.broker_order_ref,
        submitted_at=result.submitted_at or intent.submitted_at,
        completed_at=result.completed_at,
        last_error=result.dom_error_text,
    )
    log_execution_attempt(
        intent.intent_id,
        stage="result",
        status=str(result.status),
        detail=result.status_text,
        broker_order_ref=result.broker_order_ref,
        screenshot_path=result.screenshot_path,
        dom_error_text=result.dom_error_text,
        payload=result.payload,
    )


def save_live_orders(orders: Iterable[Dict[str, Any]], *, source: str) -> int:
    init_live_audit_db()
    conn = _connect()
    count = 0
    for row in orders:
        conn.execute(
            """
            INSERT INTO live_order_states (
                intent_id, broker_order_ref, symbol, action, quantity, price,
                status_text, fill_state, source, payload_json, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("intent_id"),
                row.get("broker_order_ref"),
                row.get("symbol"),
                row.get("action"),
                int(row.get("quantity") or 0),
                float(row.get("price") or 0.0),
                row.get("status_text") or "",
                str(row.get("fill_state") or FillState.UNKNOWN),
                source,
                _json_dumps(row),
                utc_now_iso(),
            ),
        )
        count += 1
    conn.commit()
    conn.close()
    return count


def save_live_positions(positions: Iterable[Dict[str, Any]], *, source: str) -> int:
    init_live_audit_db()
    conn = _connect()
    count = 0
    for row in positions:
        conn.execute(
            """
            INSERT INTO live_position_states (
                symbol, quantity, average_price, market_value, source, payload_json, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("symbol"),
                int(row.get("quantity") or 0),
                float(row.get("average_price") or 0.0),
                float(row.get("market_value") or 0.0) if row.get("market_value") is not None else None,
                source,
                _json_dumps(row),
                utc_now_iso(),
            ),
        )
        count += 1
    conn.commit()
    conn.close()
    return count


def load_latest_live_orders(limit: int = 25) -> List[Dict[str, Any]]:
    init_live_audit_db()
    conn = _connect()
    rows = conn.execute(
        """
        SELECT broker_order_ref, symbol, action, quantity, price, status_text, fill_state, source, recorded_at
        FROM live_order_states
        ORDER BY recorded_at DESC, state_id DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def load_latest_live_positions(limit: int = 50) -> List[Dict[str, Any]]:
    init_live_audit_db()
    conn = _connect()
    rows = conn.execute(
        """
        SELECT symbol, quantity, average_price, market_value, source, recorded_at
        FROM live_position_states
        ORDER BY recorded_at DESC, position_state_id DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def list_executed_trade_events(limit: int = 20) -> List[Dict[str, Any]]:
    init_live_audit_db()
    conn = _connect()
    rows = conn.execute(
        """
        SELECT i.intent_id, i.action, i.symbol, i.quantity, i.limit_price, i.strategy_tag,
               i.completed_at, i.broker_order_ref, s.status_text, s.fill_state
        FROM execution_intents i
        LEFT JOIN live_order_states s ON s.intent_id = i.intent_id
        WHERE i.status IN ('filled', 'partially_filled')
        ORDER BY COALESCE(i.completed_at, i.created_at) DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def mark_intent_notified(intent_id: str, *, owner: Optional[bool] = None, viewer: Optional[bool] = None) -> None:
    updates: Dict[str, Any] = {}
    if owner is not None:
        updates["owner_notified"] = owner
    if viewer is not None:
        updates["viewer_notified"] = viewer
    update_execution_intent(intent_id, **updates)


def record_reconciliation_run(status: str, summary: Dict[str, Any]) -> None:
    init_live_audit_db()
    conn = _connect()
    conn.execute(
        """
        INSERT INTO reconciliation_runs (status, summary_json, started_at, completed_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            status,
            _json_dumps(summary),
            utc_now_iso(),
            utc_now_iso(),
        ),
    )
    conn.commit()
    conn.close()


def save_tms_snapshot(snapshot_type: str, payload: Dict[str, Any], *, status: str = "ok") -> int:
    init_live_audit_db()
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tms_snapshots (snapshot_type, status, summary_json, recorded_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            str(snapshot_type),
            str(status),
            _json_dumps(payload or {}),
            str((payload or {}).get("snapshot_time_utc") or utc_now_iso()),
        ),
    )
    snapshot_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return snapshot_id


def load_latest_tms_snapshot(snapshot_type: str) -> Optional[Dict[str, Any]]:
    init_live_audit_db()
    conn = _connect()
    row = conn.execute(
        """
        SELECT snapshot_id, snapshot_type, status, summary_json, recorded_at
        FROM tms_snapshots
        WHERE snapshot_type = ?
        ORDER BY recorded_at DESC, snapshot_id DESC
        LIMIT 1
        """,
        (str(snapshot_type),),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    payload = _json_loads(row["summary_json"]) or {}
    if isinstance(payload, dict):
        payload.setdefault("snapshot_type", str(row["snapshot_type"]))
        payload.setdefault("snapshot_status", str(row["status"]))
        payload.setdefault("snapshot_recorded_at", str(row["recorded_at"]))
        payload.setdefault("snapshot_id", int(row["snapshot_id"]))
        return payload
    return {
        "snapshot_type": str(row["snapshot_type"]),
        "snapshot_status": str(row["status"]),
        "snapshot_recorded_at": str(row["recorded_at"]),
        "snapshot_id": int(row["snapshot_id"]),
        "payload": payload,
    }


def list_tms_snapshots(snapshot_type: str, *, limit: int = 10) -> List[Dict[str, Any]]:
    init_live_audit_db()
    conn = _connect()
    rows = conn.execute(
        """
        SELECT snapshot_id, snapshot_type, status, summary_json, recorded_at
        FROM tms_snapshots
        WHERE snapshot_type = ?
        ORDER BY recorded_at DESC, snapshot_id DESC
        LIMIT ?
        """,
        (str(snapshot_type), int(limit)),
    ).fetchall()
    conn.close()
    items: List[Dict[str, Any]] = []
    for row in rows:
        payload = _json_loads(row["summary_json"]) or {}
        if isinstance(payload, dict):
            payload.setdefault("snapshot_type", str(row["snapshot_type"]))
            payload.setdefault("snapshot_status", str(row["status"]))
            payload.setdefault("snapshot_recorded_at", str(row["recorded_at"]))
            payload.setdefault("snapshot_id", int(row["snapshot_id"]))
            items.append(payload)
        else:
            items.append(
                {
                    "snapshot_type": str(row["snapshot_type"]),
                    "snapshot_status": str(row["status"]),
                    "snapshot_recorded_at": str(row["recorded_at"]),
                    "snapshot_id": int(row["snapshot_id"]),
                    "payload": payload,
                }
            )
    return items
