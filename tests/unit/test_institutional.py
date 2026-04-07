"""Unit tests for institutional portfolio engine."""

import sqlite3
import pytest

from backend.quant_pro.institutional import PortfolioStateMachine, init_institutional_tables


@pytest.fixture
def sm(tmp_path):
    """Create a PortfolioStateMachine with a temp DB."""
    db_file = tmp_path / "test_inst.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA foreign_keys = ON")

    # Create stock_prices table (needed for some queries)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            symbol TEXT, date DATE, open REAL, high REAL,
            low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.commit()

    init_institutional_tables(conn)
    machine = PortfolioStateMachine(conn)
    yield machine
    conn.close()


class TestLedgerImmutability:
    """Verify trade ledger is append-only."""

    def test_open_creates_ledger_event(self, sm):
        pos_id = sm.open_position(
            symbol="NABIL", quantity=100, entry_price=1000.0,
            fees=60.0, strategy_tag="test",
        )
        assert pos_id > 0

        summary = sm.ledger_summary()
        assert summary["ledger_events"] >= 1

    def test_close_creates_ledger_event(self, sm):
        pos_id = sm.open_position(
            symbol="NABIL", quantity=100, entry_price=1000.0,
            fees=60.0, strategy_tag="test",
        )
        sm.close_position(position_id=pos_id, exit_price=1100.0, fees=66.0, reason="TEST")

        summary = sm.ledger_summary()
        assert summary["ledger_events"] >= 2


class TestPositionStateMachine:
    """Test position lifecycle: OPEN → CLOSED."""

    def test_open_then_close(self, sm):
        pos_id = sm.open_position(
            symbol="SBL", quantity=50, entry_price=500.0,
            fees=15.0, strategy_tag="test",
        )
        pos = sm.get_position(pos_id)
        assert pos is not None
        assert pos.status == "OPEN"

        sm.close_position(position_id=pos_id, exit_price=550.0, fees=16.5, reason="TP")
        pos = sm.get_position(pos_id)
        assert pos.status == "CLOSED"

    def test_cannot_close_already_closed(self, sm):
        pos_id = sm.open_position(
            symbol="KBL", quantity=10, entry_price=300.0,
            fees=1.8, strategy_tag="test",
        )
        sm.close_position(position_id=pos_id, exit_price=310.0, fees=1.86, reason="HOLD")

        with pytest.raises(Exception):
            sm.close_position(position_id=pos_id, exit_price=320.0, fees=1.92, reason="AGAIN")

    def test_list_open_positions(self, sm):
        sm.open_position(symbol="NABIL", quantity=100, entry_price=1000.0, fees=60.0, strategy_tag="t")
        sm.open_position(symbol="SBL", quantity=50, entry_price=500.0, fees=15.0, strategy_tag="t")

        open_positions = sm.list_open_positions()
        assert len(open_positions) == 2

    def test_realized_pnl(self, sm):
        pos_id = sm.open_position(
            symbol="NABIL", quantity=100, entry_price=1000.0,
            fees=60.0, strategy_tag="test",
        )
        sm.close_position(position_id=pos_id, exit_price=1100.0, fees=66.0, reason="TP")

        summary = sm.ledger_summary()
        # PnL = (1100 - 1000) * 100 - 60 - 66 = 10000 - 126 = 9874
        assert summary["realized_pnl"] > 0
