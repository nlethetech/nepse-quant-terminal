"""Optional MCP server exposing the NEPSE trading control plane."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

from backend.agents.agent_analyst import (
    append_external_agent_chat_message,
    load_agent_analysis,
    load_agent_archive_history,
    load_agent_history,
    publish_external_agent_analysis,
)
from backend.quant_pro.nepalosint_client import (
    related_stories as nepalosint_related_stories,
    semantic_story_search as nepalosint_semantic_story_search,
    unified_search as nepalosint_unified_search,
)
from backend.quant_pro.control_plane.command_service import (
    ControlPlaneCommandService,
    build_env_live_trader_control_plane,
)


def build_tool_adapters(service: ControlPlaneCommandService) -> Dict[str, Callable[..., Dict[str, Any]]]:
    return {
        "get_market_snapshot": lambda: service.get_market_snapshot(),
        "get_portfolio_snapshot": lambda: service.get_portfolio_snapshot(),
        "get_signal_candidates": lambda: service.get_signal_candidates(),
        "get_risk_status": lambda: service.get_risk_status(),
        "semantic_story_search": lambda query, hours=720, top_k=10, min_similarity=0.45: nepalosint_semantic_story_search(
            query,
            hours=hours,
            top_k=top_k,
            min_similarity=min_similarity,
        ),
        "unified_osint_search": lambda query, limit=10, election_year=None: nepalosint_unified_search(
            query,
            limit=limit,
            election_year=election_year,
        ),
        "related_story_search": lambda story_id, top_k=8, min_similarity=0.55, hours=8760: nepalosint_related_stories(
            story_id,
            top_k=top_k,
            min_similarity=min_similarity,
            hours=hours,
        ),
        "review_trade_candidate": lambda **kwargs: service.review_trade_candidate(**kwargs),
        "submit_paper_order": lambda **kwargs: service.submit_paper_order(**kwargs).to_record(),
        "create_live_intent": lambda **kwargs: service.create_live_intent(**kwargs).to_record(),
        "confirm_live_intent": lambda intent_id, mode="live": service.confirm_live_intent(intent_id, mode=mode).to_record(),
        "cancel_live_intent": lambda order_ref, operator_surface="mcp": service.cancel_live_intent(order_ref, operator_surface=operator_surface).to_record(),
        "modify_live_intent": lambda order_ref, limit_price, quantity=None, operator_surface="mcp": service.modify_live_intent(
            order_ref,
            limit_price=limit_price,
            quantity=quantity,
            operator_surface=operator_surface,
        ).to_record(),
        "reconcile_live_state": lambda: service.reconcile_live_state().to_record(),
        "halt_trading": lambda level="all", reason="mcp halt": service.halt_trading(level=level, reason=reason).to_record(),
        "resume_trading": lambda: service.resume_trading().to_record(),
        "sync_watchlist": lambda action="fetch", symbol=None: service.sync_watchlist(action=action, symbol=symbol).to_record(),
        "get_agent_tab_state": lambda: {
            "analysis": load_agent_analysis(),
            "history": load_agent_history(limit=20),
            "archive_count": len(load_agent_archive_history()),
        },
        "publish_agent_analysis": lambda analysis, source="mcp_external", provider="external": publish_external_agent_analysis(
            analysis,
            source=source,
            provider=provider,
        ),
        "append_agent_chat_message": lambda role, message, source="mcp_external", provider="external": {
            "history": append_external_agent_chat_message(
                role,
                message,
                source=source,
                provider=provider,
            )
        },
    }


def build_server(
    service: ControlPlaneCommandService | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
):
    service = service or build_env_live_trader_control_plane()
    adapters = build_tool_adapters(service)
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "The optional `mcp` package is required to run apps.mcp.server. "
            "Install it in the project environment, then rerun the server."
        ) from exc

    server = FastMCP(
        "nepse-control-plane",
        host=host,
        port=int(port),
        streamable_http_path=streamable_http_path,
    )

    @server.tool()
    def get_market_snapshot():
        return adapters["get_market_snapshot"]()

    @server.tool()
    def get_portfolio_snapshot():
        return adapters["get_portfolio_snapshot"]()

    @server.tool()
    def get_signal_candidates():
        return adapters["get_signal_candidates"]()

    @server.tool()
    def get_risk_status():
        return adapters["get_risk_status"]()

    @server.tool()
    def semantic_story_search(
        query: str,
        hours: int = 720,
        top_k: int = 10,
        min_similarity: float = 0.45,
    ):
        return adapters["semantic_story_search"](
            query=query,
            hours=hours,
            top_k=top_k,
            min_similarity=min_similarity,
        )

    @server.tool()
    def unified_osint_search(
        query: str,
        limit: int = 10,
        election_year: int | None = None,
    ):
        return adapters["unified_osint_search"](
            query=query,
            limit=limit,
            election_year=election_year,
        )

    @server.tool()
    def related_story_search(
        story_id: str,
        top_k: int = 8,
        min_similarity: float = 0.55,
        hours: int = 8760,
    ):
        return adapters["related_story_search"](
            story_id=story_id,
            top_k=top_k,
            min_similarity=min_similarity,
            hours=hours,
        )

    @server.tool()
    def review_trade_candidate(
        mode: str,
        action: str,
        symbol: str,
        quantity: int,
        limit_price: float | None = None,
        thesis: str = "",
        catalysts: list[str] | None = None,
        risk: list[str] | None = None,
        confidence: float = 0.0,
        horizon: str = "",
        source_signals: list[str] | None = None,
        target_order_ref: str | None = None,
        operator_surface: str = "mcp",
    ):
        return adapters["review_trade_candidate"](
            mode=mode,
            action=action,
            symbol=symbol,
            quantity=quantity,
            limit_price=limit_price,
            thesis=thesis,
            catalysts=catalysts,
            risk=risk,
            confidence=confidence,
            horizon=horizon,
            source_signals=source_signals,
            target_order_ref=target_order_ref,
            operator_surface=operator_surface,
        )

    @server.tool()
    def submit_paper_order(action: str, symbol: str, quantity: int, limit_price: float):
        return adapters["submit_paper_order"](
            action=action,
            symbol=symbol,
            quantity=quantity,
            limit_price=limit_price,
        )

    @server.tool()
    def create_live_intent(
        action: str,
        symbol: str,
        quantity: int = 0,
        limit_price: float | None = None,
        target_order_ref: str | None = None,
        mode: str = "live",
        source: str = "strategy_entry",
        reason: str = "",
        strategy_tag: str = "",
        operator_surface: str = "mcp",
    ):
        return adapters["create_live_intent"](
            action=action,
            symbol=symbol,
            quantity=quantity,
            limit_price=limit_price,
            target_order_ref=target_order_ref,
            mode=mode,
            source=source,
            reason=reason,
            strategy_tag=strategy_tag,
            operator_surface=operator_surface,
        )

    @server.tool()
    def confirm_live_intent(intent_id: str, mode: str = "live"):
        return adapters["confirm_live_intent"](intent_id=intent_id, mode=mode)

    @server.tool()
    def cancel_live_intent(order_ref: str, operator_surface: str = "mcp"):
        return adapters["cancel_live_intent"](order_ref=order_ref, operator_surface=operator_surface)

    @server.tool()
    def modify_live_intent(order_ref: str, limit_price: float, quantity: int | None = None, operator_surface: str = "mcp"):
        return adapters["modify_live_intent"](
            order_ref=order_ref,
            limit_price=limit_price,
            quantity=quantity,
            operator_surface=operator_surface,
        )

    @server.tool()
    def reconcile_live_state():
        return adapters["reconcile_live_state"]()

    @server.tool()
    def halt_trading(level: str = "all", reason: str = "mcp halt"):
        return adapters["halt_trading"](level=level, reason=reason)

    @server.tool()
    def resume_trading():
        return adapters["resume_trading"]()

    @server.tool()
    def sync_watchlist(action: str = "fetch", symbol: str | None = None):
        return adapters["sync_watchlist"](action=action, symbol=symbol)

    @server.tool()
    def get_agent_tab_state():
        return adapters["get_agent_tab_state"]()

    @server.tool()
    def publish_agent_analysis(
        analysis: dict,
        source: str = "mcp_external",
        provider: str = "external",
    ):
        return adapters["publish_agent_analysis"](analysis=analysis, source=source, provider=provider)

    @server.tool()
    def append_agent_chat_message(
        role: str,
        message: str,
        source: str = "mcp_external",
        provider: str = "external",
    ):
        return adapters["append_agent_chat_message"](
            role=role,
            message=message,
            source=source,
            provider=provider,
        )

    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NEPSE MCP control-plane server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport to expose.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP-based transports.")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP-based transports.")
    parser.add_argument(
        "--streamable-http-path",
        default="/mcp",
        help="Path to mount the streamable HTTP endpoint.",
    )
    parser.add_argument(
        "--mount-path",
        default="/",
        help="Mount path used by the SSE transport.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = build_server(
        host=args.host,
        port=args.port,
        streamable_http_path=args.streamable_http_path,
    )
    server.run(transport=args.transport, mount_path=args.mount_path)


if __name__ == "__main__":
    main()
