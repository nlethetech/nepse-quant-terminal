#!/usr/bin/env python3
"""Run the currently active built-in agent, or override it for one execution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.agent_analyst import analyze, ask
from backend.agents.runtime_config import load_active_agent_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--provider-label", default="")
    parser.add_argument("--source-label", default="")
    parser.add_argument("--fallback", default="")
    parser.add_argument("--question", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _apply_overrides(args: argparse.Namespace) -> None:
    if str(args.backend or "").strip():
        os.environ["NEPSE_AGENT_BACKEND"] = str(args.backend).strip()
    if str(args.model or "").strip():
        os.environ["NEPSE_AGENT_MODEL"] = str(args.model).strip()
    if str(args.provider_label or "").strip():
        os.environ["NEPSE_AGENT_PROVIDER_LABEL"] = str(args.provider_label).strip()
    if str(args.source_label or "").strip():
        os.environ["NEPSE_AGENT_SOURCE_LABEL"] = str(args.source_label).strip()
    if str(args.fallback or "").strip():
        os.environ["NEPSE_AGENT_FALLBACK_BACKEND"] = str(args.fallback).strip()


def main() -> None:
    args = parse_args()
    _apply_overrides(args)

    if str(args.question or "").strip():
        print(ask(str(args.question).strip()))
        return

    payload = analyze(force=bool(args.force))
    active = load_active_agent_config()
    print(
        json.dumps(
            {
                "ok": True,
                "backend": active.get("backend"),
                "provider": payload.get("agent_runtime_meta", {}).get("provider"),
                "model": active.get("model"),
                "stocks": len(payload.get("stocks", [])),
                "trade_today": payload.get("trade_today"),
                "regime": payload.get("regime"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
