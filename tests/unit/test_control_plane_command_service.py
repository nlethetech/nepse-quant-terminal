from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from backend.quant_pro.control_plane.command_service import ControlPlaneCommandService
from backend.quant_pro.control_plane.decision_journal import load_approval_request
from backend.quant_pro.control_plane.models import TradingMode
from backend.quant_pro.tms_audit import load_execution_intent
from backend.quant_pro.tms_models import ExecutionResult, ExecutionStatus, FillState


class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@dataclass
class DummyPosition:
    shares: int
    entry_price: float = 500.0
    buy_fees: float = 100.0
    last_ltp: float = 520.0
    signal_type: str = "manual"

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price + self.buy_fees

    @property
    def market_value(self) -> float:
        return self.shares * self.last_ltp

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def pnl_pct(self) -> float:
        return self.unrealized_pnl / self.cost_basis


class DummyLiveService:
    def __init__(self):
        self.confirmed = []
        self.reconciled = False

    def submit_intent(self, intent, wait=False, timeout=90.0):
        return True, "pending_confirmation" if intent.requires_confirmation else "queued", intent, None

    def confirm_intent(self, intent_id, wait=True):
        self.confirmed.append(intent_id)
        return ExecutionResult(
            intent_id=intent_id,
            status=ExecutionStatus.SUBMITTED_PENDING,
            submitted=True,
            fill_state=FillState.PENDING,
            status_text="Pending",
        )

    def reconcile(self):
        self.reconciled = True
        return {"orders_saved": 1, "positions_saved": 1, "matched_intents": 1}


class DummyTrader:
    def __init__(self):
        self._state_lock = DummyLock()
        self.cash = 1_000_000.0
        self.capital = 1_000_000.0
        self.max_positions = 10
        self.positions = {}
        self.runtime_state = {}
        self.live_execution_enabled = True
        self.execution_mode = "live"
        self.live_execution_service = DummyLiveService()
        self.live_settings = SimpleNamespace(
            max_order_notional=500_000.0,
            max_daily_orders=20,
            symbol_cooldown_secs=120,
            max_price_deviation_pct=20.0,
            owner_confirm_required=True,
        )

    def execute_manual_buy(self, symbol, shares, ltp):
        self.positions[symbol] = DummyPosition(shares=shares, entry_price=ltp, last_ltp=ltp)
        self.cash -= shares * ltp
        return True, f"BUY {symbol}"

    def execute_manual_sell(self, symbol):
        self.positions.pop(symbol, None)
        return True, f"SELL {symbol}"

    def calculate_nav(self):
        return self.cash + sum(pos.market_value for pos in self.positions.values())

    def _sector_exposure_snapshot(self):
        return {}

    def kill_live(self, level="all", reason="manual"):
        self.live_halt_level = level
        self.live_freeze_reason = reason

    def resume_live(self):
        self.live_halt_level = "none"
        self.live_freeze_reason = ""

    def reconcile_live_orders(self):
        return self.live_execution_service.reconcile()


def test_submit_paper_order_uses_trader_execution(monkeypatch):
    monkeypatch.setattr("backend.quant_pro.control_plane.command_service.fetch_latest_ltp", lambda symbol: 500.0)
    monkeypatch.setattr("backend.quant_pro.control_plane.command_service.count_intents_for_day", lambda day: 0)
    monkeypatch.setattr("backend.quant_pro.control_plane.command_service.find_recent_open_intent", lambda symbol, within_seconds=None: None)

    trader = DummyTrader()
    trader.execution_mode = "paper"
    trader.live_execution_enabled = False
    service = ControlPlaneCommandService(trader=trader, mode=TradingMode.PAPER)

    result = service.submit_paper_order(action="buy", symbol="NABIL", quantity=10, limit_price=500.0)

    assert result.ok is True
    assert "NABIL" in trader.positions


def test_shadow_live_records_approval_request(monkeypatch, tmp_path):
    monkeypatch.setenv("NEPSE_LIVE_AUDIT_DB_FILE", str(tmp_path / "audit.db"))
    monkeypatch.setattr("backend.quant_pro.control_plane.command_service.fetch_latest_ltp", lambda symbol: 500.0)
    monkeypatch.setattr("backend.quant_pro.control_plane.command_service.count_intents_for_day", lambda day: 0)
    monkeypatch.setattr("backend.quant_pro.control_plane.command_service.find_recent_open_intent", lambda symbol, within_seconds=None: None)

    trader = DummyTrader()
    trader.execution_mode = "shadow_live"
    trader.live_execution_enabled = False
    service = ControlPlaneCommandService(trader=trader, mode=TradingMode.SHADOW_LIVE)
    monkeypatch.setattr(service, "_is_market_open", lambda: True)

    result = service.create_live_intent(
        action="buy",
        symbol="NABIL",
        quantity=10,
        limit_price=500.0,
        mode="shadow_live",
        source="strategy_entry",
        reason="shadow_test",
    )

    assert result.ok is True
    assert result.status == "pending_confirmation"
    assert result.intent_id is not None
    assert load_execution_intent(result.intent_id) is not None
    approval = load_approval_request(result.intent_id)
    assert approval is not None
    assert approval.status.value == "pending"


def test_live_confirm_routes_through_execution_service(monkeypatch, tmp_path):
    monkeypatch.setenv("NEPSE_LIVE_AUDIT_DB_FILE", str(tmp_path / "audit.db"))
    monkeypatch.setattr("backend.quant_pro.control_plane.command_service.fetch_latest_ltp", lambda symbol: 500.0)
    monkeypatch.setattr("backend.quant_pro.control_plane.command_service.count_intents_for_day", lambda day: 0)
    monkeypatch.setattr("backend.quant_pro.control_plane.command_service.find_recent_open_intent", lambda symbol, within_seconds=None: None)

    trader = DummyTrader()
    service = ControlPlaneCommandService(
        trader=trader,
        live_service=trader.live_execution_service,
        mode=TradingMode.LIVE,
        reconcile_callback=trader.reconcile_live_orders,
    )
    monkeypatch.setattr(service, "_is_market_open", lambda: True)
    created = service.create_live_intent(
        action="buy",
        symbol="NABIL",
        quantity=10,
        limit_price=500.0,
        mode="live",
        source="owner_manual",
        reason="live_test",
    )

    confirmed = service.confirm_live_intent(created.intent_id or "", mode="live")

    assert confirmed.ok is True
    assert trader.live_execution_service.confirmed == [created.intent_id]
    assert confirmed.payload["result"]["status"] == ExecutionStatus.SUBMITTED_PENDING
