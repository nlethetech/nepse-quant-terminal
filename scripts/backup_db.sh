#!/bin/bash
# backup_db.sh - WAL-safe SQLite backup with 7-day retention
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DB_FILE="${NEPSE_DB_FILE:-${PROJECT_ROOT}/data/nepse_market_data.db}"
BACKUP_DIR="${NEPSE_BACKUP_DIR:-${PROJECT_ROOT}/data/backups}"
RETENTION_DAYS=7

mkdir -p "$BACKUP_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/nepse_market_data_${TIMESTAMP}.db"

echo "[backup] Starting WAL checkpoint..."
sqlite3 "$DB_FILE" "PRAGMA wal_checkpoint(TRUNCATE);"

echo "[backup] Copying database to ${BACKUP_FILE}..."
cp "$DB_FILE" "$BACKUP_FILE"

echo "[backup] Running integrity check..."
INTEGRITY=$(sqlite3 "$BACKUP_FILE" "PRAGMA integrity_check;" 2>&1)
if [ "$INTEGRITY" != "ok" ]; then
    echo "[backup] INTEGRITY CHECK FAILED: $INTEGRITY"
    rm -f "$BACKUP_FILE"
    exit 1
fi

echo "[backup] Pruning backups older than ${RETENTION_DAYS} days..."
find "$BACKUP_DIR" -name "nepse_market_data_*.db" -mtime +"$RETENTION_DAYS" -delete

BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "[backup] Success: ${BACKUP_FILE} (${BACKUP_SIZE})"
