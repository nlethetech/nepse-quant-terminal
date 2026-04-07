#!/usr/bin/env python3
"""Run the built-in NEPSE analyst with the local Gemma 4 MLX backend."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.agent_analyst import (
    DEFAULT_GEMMA4_EXPERIMENTAL_MODEL,
    DEFAULT_GEMMA4_MLX_MODEL,
    analyze,
    ask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("NEPSE_AGENT_MODEL", DEFAULT_GEMMA4_MLX_MODEL))
    parser.add_argument("--backend", default="gemma4_mlx")
    parser.add_argument("--fallback", default=os.environ.get("NEPSE_AGENT_FALLBACK_BACKEND", "claude"))
    parser.add_argument("--question", default="")
    parser.add_argument("--use-experimental", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["NEPSE_AGENT_BACKEND"] = str(args.backend)
    os.environ["NEPSE_AGENT_FALLBACK_BACKEND"] = str(args.fallback)
    os.environ["NEPSE_AGENT_PROVIDER_LABEL"] = "gemma4_mlx"
    os.environ["NEPSE_AGENT_SOURCE_LABEL"] = "local_gemma4_mlx"
    os.environ["NEPSE_AGENT_MODEL"] = (
        DEFAULT_GEMMA4_EXPERIMENTAL_MODEL if args.use_experimental else str(args.model)
    )

    if args.question.strip():
        print(ask(args.question.strip()))
        return

    payload = analyze(force=bool(args.force))
    print(json.dumps(
        {
            "ok": True,
            "provider": payload.get("agent_runtime_meta", {}).get("provider"),
            "model": os.environ["NEPSE_AGENT_MODEL"],
            "stocks": len(payload.get("stocks", [])),
            "trade_today": payload.get("trade_today"),
            "regime": payload.get("regime"),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
