"""
Institutional reliability components for NEPSE quant workflows.

This module provides:
1) Deterministic ingestion run metadata schema/helpers.
2) Immutable trade ledger + portfolio state-machine engine.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


INGESTION_RUN_STATUS = {"RUNNING", "SUCCESS", "PARTIAL", "FAILED"}
INGESTION_SYMBOL_STATUS = {"SUCCESS", "NO_DATA", "FAILED"}
POSITION_STATUS = {"OPEN", "CLOSED"}
LEDGER_SIDE = {"BUY", "SELL", "NONE"}


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def init_institutional_tables(conn: sqlite3.Connection) -> None:
    """Create ingestion metadata and institutional portfolio tables."""
    cur = conn.cursor()

    # Ingestion run metadata
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at_utc TEXT NOT NULL,
            ended_at_utc TEXT,
            status TEXT NOT NULL,
            source TEXT NOT NULL,
            universe_size INTEGER NOT NULL DEFAULT 0,
            symbols_succeeded INTEGER NOT NULL DEFAULT 0,
            symbols_failed INTEGER NOT NULL DEFAULT 0,
            rows_fetched INTEGER NOT NULL DEFAULT 0,
            latest_market_date_before TEXT,
            latest_market_date_after TEXT,
            freshness_days_after INTEGER,
            sla_max_staleness_days INTEGER,
            notes TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_run_symbols (
            run_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            started_at_utc TEXT NOT NULL,
            ended_at_utc TEXT,
            status TEXT NOT NULL,
            rows_fetched INTEGER NOT NULL DEFAULT 0,
            rows_added INTEGER NOT NULL DEFAULT 0,
            latest_date_before TEXT,
            latest_date_after TEXT,
            error TEXT,
            PRIMARY KEY (run_id, symbol),
            FOREIGN KEY (run_id) REFERENCES ingestion_runs(run_id)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ingestion_runs_started ON ingestion_runs(started_at_utc)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ingestion_run_symbols_status ON ingestion_run_symbols(run_id, status)"
    )

    # Portfolio position registry (mutable state)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_positions (
            position_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            status TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            avg_entry_price REAL NOT NULL,
            opened_at_utc TEXT NOT NULL,
            closed_at_utc TEXT,
            high_watermark REAL NOT NULL,
            hard_stop_pct REAL NOT NULL DEFAULT 0.08,
            trailing_stop_pct REAL NOT NULL DEFAULT 0.10,
            take_profit_pct REAL NOT NULL DEFAULT 0.20,
            strategy_tag TEXT,
            metadata_json TEXT
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_portfolio_positions_status ON portfolio_positions(status)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_portfolio_positions_symbol ON portfolio_positions(symbol)"
    )

    # Immutable ledger: append-only event store
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_ledger (
            ledger_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_ts_utc TEXT NOT NULL,
            position_id INTEGER,
            symbol TEXT NOT NULL,
            event_type TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            fees REAL NOT NULL DEFAULT 0.0,
            cash_impact REAL NOT NULL,
            realized_pnl REAL,
            reason TEXT,
            metadata_json TEXT,
            FOREIGN KEY (position_id) REFERENCES portfolio_positions(position_id)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_trade_ledger_position ON trade_ledger(position_id, event_ts_utc)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_trade_ledger_symbol ON trade_ledger(symbol, event_ts_utc)"
    )

    # Enforce append-only semantics for ledger events.
    cur.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_trade_ledger_no_update
        BEFORE UPDATE ON trade_ledger
        BEGIN
            SELECT RAISE(ABORT, 'trade_ledger is immutable');
        END
        """
    )
    cur.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_trade_ledger_no_delete
        BEFORE DELETE ON trade_ledger
        BEGIN
            SELECT RAISE(ABORT, 'trade_ledger is immutable');
        END
        """
    )

    conn.commit()


def get_db_latest_market_date(conn: sqlite3.Connection) -> Optional[str]:
    """Return latest date from stock_prices as YYYY-MM-DD."""
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM stock_prices")
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return str(row[0])


@dataclass
class PositionSnapshot:
    position_id: int
    symbol: str
    quantity: int
    avg_entry_price: float
    high_watermark: float
    hard_stop_pct: float
    trailing_stop_pct: float
    take_profit_pct: float
    status: str
    opened_at_utc: str
    closed_at_utc: Optional[str]


@dataclass
class RiskSignal:
    position_id: int
    symbol: str
    ltp: float
    action: str
    reason: str
    hard_stop_price: float
    trailing_stop_price: float
    take_profit_price: float


class PortfolioStateMachine:
    """
    Institutional portfolio engine with immutable trade ledger.

    State model:
    - OPEN: active position.
    - CLOSED: terminal state (no re-open).
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        init_institutional_tables(self.conn)

    def _insert_ledger_event(
        self,
        *,
        position_id: Optional[int],
        symbol: str,
        event_type: str,
        side: str,
        quantity: int,
        price: float,
        fees: float,
        cash_impact: float,
        realized_pnl: Optional[float],
        reason: str,
        metadata: Optional[Dict] = None,
    ) -> int:
        if side not in LEDGER_SIDE:
            raise ValueError(f"Invalid side: {side}")
        if quantity < 0:
            raise ValueError("Quantity must be >= 0")
        if price < 0:
            raise ValueError("Price must be >= 0")

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO trade_ledger (
                event_ts_utc, position_id, symbol, event_type, side, quantity,
                price, fees, cash_impact, realized_pnl, reason, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                position_id,
                symbol,
                event_type,
                side,
                quantity,
                float(price),
                float(fees),
                float(cash_impact),
                realized_pnl,
                reason,
                json.dumps(metadata or {}, separators=(",", ":")),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def open_position(
        self,
        *,
        symbol: str,
        quantity: int,
        entry_price: float,
        fees: float = 0.0,
        strategy_tag: str = "default",
        hard_stop_pct: float = 0.08,
        trailing_stop_pct: float = 0.10,
        take_profit_pct: float = 0.20,
        metadata: Optional[Dict] = None,
    ) -> int:
        if quantity <= 0:
            raise ValueError("Quantity must be > 0")
        if entry_price <= 0:
            raise ValueError("entry_price must be > 0")

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO portfolio_positions (
                symbol, status, quantity, avg_entry_price, opened_at_utc,
                high_watermark, hard_stop_pct, trailing_stop_pct, take_profit_pct,
                strategy_tag, metadata_json
            )
            VALUES (?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                int(quantity),
                float(entry_price),
                utc_now_iso(),
                float(entry_price),
                float(hard_stop_pct),
                float(trailing_stop_pct),
                float(take_profit_pct),
                strategy_tag,
                json.dumps(metadata or {}, separators=(",", ":")),
            ),
        )
        position_id = int(cur.lastrowid)
        # No intermediate commit — let _insert_ledger_event() commit both
        # the position INSERT and ledger INSERT in one transaction.

        gross = quantity * entry_price
        self._insert_ledger_event(
            position_id=position_id,
            symbol=symbol,
            event_type="OPEN_POSITION",
            side="BUY",
            quantity=quantity,
            price=entry_price,
            fees=fees,
            cash_impact=-(gross + fees),
            realized_pnl=None,
            reason="ENTRY",
            metadata=metadata,
        )
        return position_id

    def get_position(self, position_id: int) -> Optional[PositionSnapshot]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT position_id, symbol, quantity, avg_entry_price, high_watermark,
                   hard_stop_pct, trailing_stop_pct, take_profit_pct, status,
                   opened_at_utc, closed_at_utc
            FROM portfolio_positions
            WHERE position_id = ?
            """,
            (position_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return PositionSnapshot(*row)

    def list_open_positions(self) -> List[PositionSnapshot]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT position_id, symbol, quantity, avg_entry_price, high_watermark,
                   hard_stop_pct, trailing_stop_pct, take_profit_pct, status,
                   opened_at_utc, closed_at_utc
            FROM portfolio_positions
            WHERE status = 'OPEN'
            ORDER BY position_id ASC
            """
        )
        return [PositionSnapshot(*row) for row in cur.fetchall()]

    def close_position(
        self,
        *,
        position_id: int,
        exit_price: float,
        fees: float = 0.0,
        reason: str = "MANUAL_EXIT",
        metadata: Optional[Dict] = None,
    ) -> None:
        if exit_price <= 0:
            raise ValueError("exit_price must be > 0")

        pos = self.get_position(position_id)
        if pos is None:
            raise ValueError(f"position_id={position_id} does not exist")
        if pos.status != "OPEN":
            raise ValueError(f"position_id={position_id} is not OPEN")

        gross_entry = pos.quantity * pos.avg_entry_price
        gross_exit = pos.quantity * exit_price
        realized_pnl = gross_exit - gross_entry - fees

        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE portfolio_positions
            SET status = 'CLOSED',
                closed_at_utc = ?
            WHERE position_id = ? AND status = 'OPEN'
            """,
            (utc_now_iso(), position_id),
        )
        if cur.rowcount != 1:
            raise RuntimeError(f"Failed to close position_id={position_id}")
        # No intermediate commit — let _insert_ledger_event() commit both
        # the position UPDATE and ledger INSERT in one transaction.

        self._insert_ledger_event(
            position_id=position_id,
            symbol=pos.symbol,
            event_type="CLOSE_POSITION",
            side="SELL",
            quantity=pos.quantity,
            price=exit_price,
            fees=fees,
            cash_impact=(gross_exit - fees),
            realized_pnl=realized_pnl,
            reason=reason,
            metadata=metadata,
        )

    def _update_high_watermark(self, position_id: int, ltp: float) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE portfolio_positions
            SET high_watermark = CASE WHEN high_watermark < ? THEN ? ELSE high_watermark END
            WHERE position_id = ? AND status = 'OPEN'
            """,
            (ltp, ltp, position_id),
        )
        self.conn.commit()

    def evaluate_risk_signals(
        self,
        ltp_by_symbol: Dict[str, float],
    ) -> List[RiskSignal]:
        """
        Evaluate all open positions against hard stop, trailing stop, and take profit.
        """
        signals: List[RiskSignal] = []
        positions = self.list_open_positions()
        for pos in positions:
            ltp = ltp_by_symbol.get(pos.symbol)
            if ltp is None or ltp <= 0:
                continue

            self._update_high_watermark(pos.position_id, ltp)
            pos = self.get_position(pos.position_id) or pos

            hard_stop_price = pos.avg_entry_price * (1.0 - pos.hard_stop_pct)
            trailing_stop_price = pos.high_watermark * (1.0 - pos.trailing_stop_pct)
            take_profit_price = pos.avg_entry_price * (1.0 + pos.take_profit_pct)

            action = "HOLD"
            reason = "NO_TRIGGER"
            if ltp <= hard_stop_price:
                action = "EXIT"
                reason = "HARD_STOP"
            elif pos.high_watermark > pos.avg_entry_price and ltp <= trailing_stop_price:
                action = "EXIT"
                reason = "TRAILING_STOP"
            elif ltp >= take_profit_price:
                action = "EXIT"
                reason = "TAKE_PROFIT"

            if action == "EXIT":
                signals.append(
                    RiskSignal(
                        position_id=pos.position_id,
                        symbol=pos.symbol,
                        ltp=float(ltp),
                        action=action,
                        reason=reason,
                        hard_stop_price=float(hard_stop_price),
                        trailing_stop_price=float(trailing_stop_price),
                        take_profit_price=float(take_profit_price),
                    )
                )
        return signals

    def apply_risk_actions(
        self,
        signals: List[RiskSignal],
        *,
        fees_bps: float = 0.0,
    ) -> None:
        """
        Apply risk actions by closing positions and recording immutable ledger events.
        """
        for sig in signals:
            pos = self.get_position(sig.position_id)
            if pos is None or pos.status != "OPEN":
                continue
            fees = (fees_bps / 10_000.0) * (sig.ltp * pos.quantity)
            self.close_position(
                position_id=sig.position_id,
                exit_price=sig.ltp,
                fees=fees,
                reason=sig.reason,
                metadata={
                    "hard_stop_price": sig.hard_stop_price,
                    "trailing_stop_price": sig.trailing_stop_price,
                    "take_profit_price": sig.take_profit_price,
                },
            )

    def ledger_summary(self) -> Dict[str, float]:
        """Return simple ledger-level statistics."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM trade_ledger")
        events = int(cur.fetchone()[0] or 0)
        cur.execute(
            """
            SELECT COALESCE(SUM(realized_pnl), 0.0)
            FROM trade_ledger
            WHERE event_type = 'CLOSE_POSITION'
            """
        )
        realized = float(cur.fetchone()[0] or 0.0)
        cur.execute(
            """
            SELECT COUNT(*)
            FROM portfolio_positions
            WHERE status = 'OPEN'
            """
        )
        open_positions = int(cur.fetchone()[0] or 0)
        return {
            "ledger_events": events,
            "realized_pnl": realized,
            "open_positions": open_positions,
        }

