"""Read-model builders for market, portfolio, and risk snapshots."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from backend.quant_pro.tms_audit import load_latest_live_orders, load_latest_live_positions

from .models import MarketSnapshot, PortfolioSnapshot, PositionSnapshotItem, RiskSnapshot, SignalCandidate


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def build_signal_candidates(signals: Iterable[dict] | None) -> List[SignalCandidate]:
    items: List[SignalCandidate] = []
    for sig in list(signals or []):
        items.append(
            SignalCandidate(
                symbol=str(sig.get("symbol") or "").upper(),
                signal_type=str(sig.get("signal_type") or sig.get("type") or ""),
                score=_safe_float(sig.get("score")),
                confidence=_safe_float(sig.get("confidence")),
                reasoning=str(sig.get("reasoning") or ""),
                raw=dict(sig),
            )
        )
    return items


def build_market_snapshot_from_trader(trader: Any, *, market_open: bool) -> MarketSnapshot:
    last_refresh = getattr(trader, "last_refresh", None)
    signals = build_signal_candidates(getattr(trader, "signals_today", []))
    return MarketSnapshot(
        as_of=str(getattr(trader, "last_price_snapshot_time_utc", "") or ""),
        regime=str(getattr(trader, "regime", "unknown") or "unknown"),
        market_open=bool(market_open),
        signal_count=len(signals),
        last_refresh_nst=last_refresh.isoformat() if last_refresh is not None else None,
        price_source=str(getattr(trader, "last_price_source_label", "") or ""),
        signals=signals,
        metadata={
            "price_source_detail": str(getattr(trader, "last_price_source_detail", "") or ""),
            "num_signals_today": int(getattr(trader, "num_signals_today", len(signals)) or len(signals)),
        },
    )


def build_portfolio_snapshot_from_trader(trader: Any) -> PortfolioSnapshot:
    positions = []
    raw_positions = dict(getattr(trader, "positions", {}) or {})
    for symbol, pos in raw_positions.items():
        positions.append(
            PositionSnapshotItem(
                symbol=str(symbol).upper(),
                quantity=_safe_int(getattr(pos, "shares", 0)),
                entry_price=_safe_float(getattr(pos, "entry_price", 0.0)),
                last_price=_safe_float(getattr(pos, "last_ltp", None), _safe_float(getattr(pos, "entry_price", 0.0))),
                market_value=_safe_float(getattr(pos, "market_value", 0.0)),
                cost_basis=_safe_float(getattr(pos, "cost_basis", 0.0)),
                unrealized_pnl=_safe_float(getattr(pos, "unrealized_pnl", 0.0)),
                pnl_pct=_safe_float(getattr(pos, "pnl_pct", 0.0)),
                signal_type=str(getattr(pos, "signal_type", "") or ""),
                sector="",
            )
        )
    nav = float(getattr(trader, "calculate_nav", lambda: getattr(trader, "cash", 0.0))())
    return PortfolioSnapshot(
        cash=_safe_float(getattr(trader, "cash", 0.0)),
        nav=nav,
        capital=_safe_float(getattr(trader, "capital", nav)),
        open_positions=len(positions),
        positions=positions,
        runtime_state=dict(getattr(trader, "runtime_state", {}) or {}),
    )


def build_risk_snapshot_from_trader(trader: Any) -> RiskSnapshot:
    exposure = {}
    if hasattr(trader, "_sector_exposure_snapshot"):
        try:
            exposure = dict(trader._sector_exposure_snapshot())
        except Exception:
            exposure = {}
    flags: List[str] = []
    halt_level = str(getattr(trader, "live_halt_level", "none") or "none")
    if halt_level != "none":
        flags.append(f"halted:{halt_level}")
    freeze_reason = str(getattr(trader, "live_freeze_reason", "") or "")
    if freeze_reason:
        flags.append(freeze_reason)
    return RiskSnapshot(
        halt_level=halt_level,
        freeze_reason=freeze_reason,
        max_positions=_safe_int(getattr(trader, "max_positions", 0)),
        open_positions=len(dict(getattr(trader, "positions", {}) or {})),
        cash=_safe_float(getattr(trader, "cash", 0.0)),
        daily_order_cap=_safe_int(getattr(getattr(trader, "live_settings", None), "max_daily_orders", 0)) or None,
        sector_exposure=exposure,
        risk_flags=flags,
    )


def build_live_state_snapshot(limit_orders: int = 20, limit_positions: int = 20) -> Dict[str, Any]:
    return {
        "orders": load_latest_live_orders(limit=limit_orders),
        "positions": load_latest_live_positions(limit=limit_positions),
    }
