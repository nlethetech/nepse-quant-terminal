#!/bin/bash
#
# NEPSE Daily Trading Workflow (Production Hardened)
#
# Usage:
#   ./daily_run.sh                    # Default: NPR 1M capital
#   ./daily_run.sh 2000000            # Custom capital
#   ./daily_run.sh 1000000 --dry-run  # Dry run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

CAPITAL="${1:-1000000}"
EXTRA_ARGS="${@:2}"
LOG_DIR="data/runtime/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/daily_$(date +%Y%m%d).log"

# Tee all output to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

trap 'echo "[daily_run] FAILED at step — check $LOG_FILE"; exit 1' ERR

echo "========================================"
echo "NEPSE Daily Run — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Capital: NPR $(printf "%'d" "$CAPITAL")"
echo "Log: $LOG_FILE"
echo "========================================"

# Step 0: Check if today is a NEPSE trading day
TRADING_DAY=$(python3 -c "
from backend.quant_pro.nepse_calendar import is_today_trading_day
print('yes' if is_today_trading_day() else 'no')
")

if [ "$TRADING_DAY" = "no" ]; then
    echo ""
    python3 -c "
from backend.quant_pro.nepse_calendar import current_nepal_datetime, get_market_schedule, is_nepal_weekend, is_known_holiday, next_trading_day
today = current_nepal_datetime().date()
schedule = get_market_schedule()
reason = f'weekend ({schedule[\"weekend\"]})' if is_nepal_weekend(today) else 'public holiday' if is_known_holiday(today) else 'non-trading day'
nxt = next_trading_day(today)
print(f'Today ({today}, {today.strftime(\"%A\")}) is a {reason}.')
print(f'Next trading day: {nxt} ({nxt.strftime(\"%A\")})')
"
    echo ""
    echo "[daily_run] Skipping — not a NEPSE trading day."
    echo "========================================"
    exit 0
fi

# Step 1: Backup database
echo ""
echo "[1/4] Backing up database..."
if [ -x scripts/backup_db.sh ]; then
    bash scripts/backup_db.sh
else
    echo "  (skipped — scripts/backup_db.sh not found or not executable)"
fi

# Step 2: Data ingestion
echo ""
echo "[2/4] Running data ingestion..."
python3 -m scripts.ingestion.deterministic_daily_ingestion --source both --max-staleness-days 3 || {
    echo "  WARNING: Ingestion returned non-zero — checking data freshness..."
}

# Gate: verify data is fresh enough to trade on
FRESHNESS_OK=$(python3 -c "
import sqlite3, pandas as pd
from backend.quant_pro.database import get_db_path
from datetime import datetime
conn = sqlite3.connect(str(get_db_path()), timeout=10)
cur = conn.cursor()
cur.execute('SELECT MAX(date) FROM stock_prices')
row = cur.fetchone()
conn.close()
if row and row[0]:
    last = pd.Timestamp(row[0]).date()
    stale = (datetime.now().date() - last).days
    print('yes' if stale <= 3 else 'no')
else:
    print('no')
")

if [ "$FRESHNESS_OK" = "no" ]; then
    echo "  ABORT: Data is too stale (>3 days). Skipping signal generation."
    echo "  Fix: check ingestion source or manually update data."
    exit 1
fi

# Step 3: Check existing positions for exits
echo ""
echo "[3/5] Checking existing positions for risk exits..."
if [ -f "data/runtime/trading/paper_portfolio.csv" ]; then
    python3 -m backend.trading.paper_trade_tracker --action risk
    if [ -f "data/runtime/trading/sell_orders.csv" ]; then
        echo ""
        echo "========== SELL ORDERS =========="
        cat data/runtime/trading/sell_orders.csv
        echo "================================="
    fi
else
    echo "  (no existing portfolio — skipping risk check)"
fi

# Step 4: Generate new buy signals
echo ""
echo "[4/5] Generating trading signals..."
python3 -m scripts.signals.generate_daily_signals --capital "$CAPITAL" $EXTRA_ARGS

if [ -f "data/runtime/trading/buy_orders.csv" ]; then
    echo ""
    echo "========== BUY ORDERS =========="
    cat data/runtime/trading/buy_orders.csv
    echo "================================"
fi

# Step 5: Health check
echo ""
echo "[5/5] Running health check..."
python3 -c "
from backend.quant_pro.monitoring import run_health_check, print_health_report
report = run_health_check()
print_health_report(report)
if report['status'] == 'CRITICAL':
    raise SystemExit(1)
"

echo ""
echo "========================================"
echo "Done! $(date '+%H:%M:%S')"
echo "========================================"
