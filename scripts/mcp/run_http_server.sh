#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HOST="${MCP_HOST:-127.0.0.1}"
PORT="${MCP_PORT:-8765}"
PATH_PREFIX="${MCP_PATH:-/mcp}"

export NEPSE_MCP_TRADING_MODE="paper"
export NEPSE_MCP_DRY_RUN="${NEPSE_MCP_DRY_RUN:-true}"

exec python3 -m apps.mcp.server \
  --transport streamable-http \
  --host "$HOST" \
  --port "$PORT" \
  --streamable-http-path "$PATH_PREFIX" \
  "$@"
