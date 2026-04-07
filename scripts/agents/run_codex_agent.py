#!/usr/bin/env python3
"""Run Codex against the local MCP control plane and publish into the TUI agent tab."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.agent_analyst import (
    append_external_agent_chat_message,
    publish_external_agent_analysis,
)
from backend.quant_pro.paths import get_project_root


PROMPT = """Use the MCP server named nepse_control_plane_http.
Call these tools before answering:
- get_market_snapshot
- get_portfolio_snapshot
- get_signal_candidates
- get_risk_status
- get_agent_tab_state

Return raw JSON only, no markdown, no prose, with this exact top-level shape:
{
  "market_view": "short market summary",
  "trade_today": true,
  "trade_today_reason": "why",
  "risks": ["risk 1", "risk 2"],
  "portfolio_note": "short portfolio note",
  "regime": "bull|neutral|bear|unknown",
  "stocks": [
    {
      "symbol": "NABIL",
      "algo_signal": "BUY",
      "sector": "Banking",
      "verdict": "APPROVE|REJECT|HOLD",
      "conviction": 0.0,
      "bull_case": "short bull case",
      "bear_case": "short bear case",
      "what_matters": "what matters now",
      "reasoning": "2-3 sentence reasoning"
    }
  ]
}

If there are no candidates, keep stocks as an empty list and still produce the full JSON object.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8765")))
    parser.add_argument("--path", default=os.environ.get("MCP_PATH", "/mcp"))
    parser.add_argument("--mode", default=os.environ.get("NEPSE_MCP_TRADING_MODE", "paper"))
    parser.add_argument("--dry-run", default=os.environ.get("NEPSE_MCP_DRY_RUN", "true"))
    parser.add_argument("--model", default=os.environ.get("CODEX_AGENT_MODEL", "gpt-5.4"))
    parser.add_argument("--provider-label", default="codex")
    return parser.parse_args()


def extract_json_object(text: str) -> dict:
    raw = str(text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in Codex output")
    return json.loads(raw[start:end])


def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.5):
            return True
    except OSError:
        return False


def wait_for_port(host: str, port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if is_port_open(host, port):
            return
        time.sleep(0.25)
    raise TimeoutError(f"MCP server did not become ready on {host}:{port}")


def start_http_server(root: Path, host: str, port: int, path: str, mode: str, dry_run: str) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["MCP_HOST"] = host
    env["MCP_PORT"] = str(port)
    env["MCP_PATH"] = path
    env["NEPSE_MCP_TRADING_MODE"] = mode
    env["NEPSE_MCP_DRY_RUN"] = dry_run
    env.setdefault("PYTHONWARNINGS", "ignore")
    return subprocess.Popen(
        [str(root / "scripts" / "mcp" / "run_http_server.sh")],
        cwd=str(root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def run_codex(root: Path, *, url: str, model: str) -> dict:
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("codex CLI not found in PATH")

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        cmd = [
            codex,
            "exec",
            "--skip-git-repo-check",
            "-C",
            str(root),
            "--color",
            "never",
            "-m",
            str(model),
            "-c",
            f'mcp_servers.nepse_control_plane_http.url="{url}"',
            "-o",
            str(output_path),
            PROMPT,
        ]
        result = subprocess.run(cmd, cwd=str(root), text=True, capture_output=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(
                "Codex exec failed:\n"
                f"stdout:\n{result.stdout[-4000:]}\n"
                f"stderr:\n{result.stderr[-4000:]}"
            )
        return extract_json_object(output_path.read_text(encoding="utf-8"))
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    root = get_project_root(__file__)
    url = f"http://{args.host}:{args.port}{args.path}"
    started_server: subprocess.Popen[str] | None = None

    try:
        if not is_port_open(args.host, args.port):
            started_server = start_http_server(
                root,
                host=args.host,
                port=args.port,
                path=args.path,
                mode=args.mode,
                dry_run=str(args.dry_run),
            )
            wait_for_port(args.host, args.port)

        append_external_agent_chat_message(
            "AGENT",
            f"Codex agent run started via MCP ({args.mode}).",
            source="codex_cli",
            provider=args.provider_label,
        )
        analysis = run_codex(root, url=url, model=args.model)
        published = publish_external_agent_analysis(
            analysis,
            source="codex_cli",
            provider=args.provider_label,
        )
        append_external_agent_chat_message(
            "AGENT",
            "Codex agent analysis refreshed via MCP.",
            source="codex_cli",
            provider=args.provider_label,
        )
        print(json.dumps({"ok": True, "provider": args.provider_label, "stocks": len(published.get("stocks", []))}, indent=2))
    finally:
        if started_server is not None:
            started_server.terminate()
            try:
                started_server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                started_server.kill()


if __name__ == "__main__":
    main()
