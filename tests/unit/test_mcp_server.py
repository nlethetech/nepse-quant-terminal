from __future__ import annotations

from backend.quant_pro.control_plane.models import CommandResult, TradingMode
from apps.mcp.server import build_tool_adapters


class DummyService:
    def get_market_snapshot(self):
        return {"market": "ok"}

    def get_portfolio_snapshot(self):
        return {"portfolio": "ok"}

    def get_signal_candidates(self):
        return {"signals": []}

    def get_risk_status(self):
        return {"risk": "ok"}

    def review_trade_candidate(self, **kwargs):
        return {"reviewed": kwargs["symbol"]}

    def submit_paper_order(self, **kwargs):
        return CommandResult(True, "submitted", "paper_ok", TradingMode.PAPER, payload=kwargs)

    def create_live_intent(self, **kwargs):
        return CommandResult(True, "pending_confirmation", "live_ok", TradingMode.LIVE, payload=kwargs, intent_id="intent_1")

    def confirm_live_intent(self, intent_id, mode="live"):
        return CommandResult(True, "confirmed", "confirm_ok", TradingMode.LIVE, intent_id=intent_id)

    def cancel_live_intent(self, order_ref, operator_surface="mcp"):
        return CommandResult(True, "ok", "cancel_ok", TradingMode.LIVE, payload={"order_ref": order_ref})

    def modify_live_intent(self, order_ref, limit_price, quantity=None, operator_surface="mcp"):
        return CommandResult(True, "ok", "modify_ok", TradingMode.LIVE, payload={"order_ref": order_ref, "limit_price": limit_price, "quantity": quantity})

    def reconcile_live_state(self):
        return CommandResult(True, "ok", "reconcile_ok", TradingMode.LIVE, payload={"matched_intents": 1})

    def halt_trading(self, level="all", reason="mcp halt"):
        return CommandResult(True, "halted", "halt_ok", TradingMode.LIVE, payload={"level": level})

    def resume_trading(self):
        return CommandResult(True, "resumed", "resume_ok", TradingMode.LIVE)

    def sync_watchlist(self, action="fetch", symbol=None):
        return CommandResult(True, "ok", "watch_ok", TradingMode.LIVE, payload={"action": action, "symbol": symbol})


def test_mcp_tool_adapters_route_to_service():
    tools = build_tool_adapters(DummyService())

    assert tools["get_market_snapshot"]() == {"market": "ok"}
    assert tools["review_trade_candidate"](mode="paper", action="buy", symbol="NABIL", quantity=10, limit_price=500.0) == {"reviewed": "NABIL"}

    live = tools["create_live_intent"](action="buy", symbol="NABIL", quantity=10, limit_price=500.0)
    assert live["intent_id"] == "intent_1"

    paper = tools["submit_paper_order"](action="buy", symbol="NABIL", quantity=10, limit_price=500.0)
    assert paper["payload"]["symbol"] == "NABIL"


def test_mcp_tool_adapters_expose_nepalosint_tools(monkeypatch):
    monkeypatch.setattr(
        "apps.mcp.server.nepalosint_semantic_story_search",
        lambda query, hours=720, top_k=10, min_similarity=0.45: {"query": query, "results": [{"story_id": "s1"}]},
        raising=False,
    )
    monkeypatch.setattr(
        "apps.mcp.server.nepalosint_unified_search",
        lambda query, limit=10, election_year=None: {"query": query, "categories": {"stories": {"items": [{"id": "s1"}], "total": 1}}},
        raising=False,
    )
    monkeypatch.setattr(
        "apps.mcp.server.nepalosint_related_stories",
        lambda story_id, top_k=8, min_similarity=0.55, hours=8760: {"source_story_id": story_id, "similar_stories": [{"story_id": "s2"}]},
        raising=False,
    )

    tools = build_tool_adapters(DummyService())

    assert tools["semantic_story_search"](query="HRL")["results"][0]["story_id"] == "s1"
    assert tools["unified_osint_search"](query="HRL")["categories"]["stories"]["total"] == 1
    assert tools["related_story_search"](story_id="s1")["similar_stories"][0]["story_id"] == "s2"
