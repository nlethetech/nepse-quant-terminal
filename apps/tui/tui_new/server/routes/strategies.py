"""Strategy configuration and backtesting endpoints."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
async def get_strategies(request: Request):
    try:
        from backend.trading import strategy_registry

        return strategy_registry.list_strategies()
    except Exception as exc:
        return {"error": str(exc), "strategies": []}


@router.post("/backtest")
async def run_backtest(request: Request):
    try:
        from backend.trading import strategy_registry

        body = await request.json()
        strategy_id = str(body.get("strategy_id") or body.get("id") or "").strip()
        strategy = strategy_registry.load_strategy(strategy_id) if strategy_id else None
        if strategy is None and isinstance(body.get("strategy"), dict):
            strategy = body["strategy"]
        if strategy is None:
            return {"error": "strategy not found"}

        start_date = str(body.get("start_date") or body.get("start") or "").strip()
        end_date = str(body.get("end_date") or body.get("end") or "").strip()
        if not start_date or not end_date:
            return {"error": "start_date and end_date are required"}

        capital = float(body.get("capital") or body.get("initial_capital") or 1_000_000)
        return await asyncio.to_thread(
            strategy_registry.run_strategy_backtest,
            strategy,
            start_date=start_date,
            end_date=end_date,
            capital=capital,
        )
    except Exception as exc:
        return {"error": str(exc)}
