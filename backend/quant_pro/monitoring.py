"""
Health check and monitoring utilities for NEPSE Quant Pro.
"""

import sqlite3
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .database import get_db_path

logger = logging.getLogger(__name__)


def run_health_check(db_path: Path | None = None) -> Dict[str, Any]:
    """
    Run system health checks and return a status dict.

    Checks:
    - Data freshness (days since last market data)
    - Database size
    - Open positions count (if institutional tables exist)
    - WAL file size
    """
    if db_path is None:
        db_path = get_db_path()

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "status": "OK",
        "checks": {},
    }

    # 1. Database existence and size
    if not db_path.exists():
        report["status"] = "CRITICAL"
        report["checks"]["db_exists"] = {"ok": False, "detail": f"DB not found: {db_path}"}
        return report

    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    report["checks"]["db_size_mb"] = {"ok": True, "value": round(db_size_mb, 1)}

    # WAL file
    wal_path = db_path.with_suffix(".db-wal")
    if wal_path.exists():
        wal_mb = wal_path.stat().st_size / (1024 * 1024)
        ok = wal_mb < 500  # WAL > 500 MB is concerning
        report["checks"]["wal_size_mb"] = {"ok": ok, "value": round(wal_mb, 1)}
        if not ok:
            report["status"] = "WARNING"

    # 2. Data freshness
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM stock_prices")
        row = cur.fetchone()
        if row and row[0]:
            from datetime import date as date_type
            import pandas as pd
            last_date = pd.Timestamp(row[0]).date()
            staleness = (datetime.now().date() - last_date).days
            ok = staleness <= 3
            report["checks"]["data_freshness"] = {
                "ok": ok,
                "last_date": str(last_date),
                "staleness_days": staleness,
            }
            if not ok:
                report["status"] = "WARNING" if staleness <= 30 else "CRITICAL"
        else:
            report["checks"]["data_freshness"] = {"ok": False, "detail": "No data"}
            report["status"] = "CRITICAL"

        # 3. Symbol count
        cur.execute("SELECT COUNT(DISTINCT symbol) FROM stock_prices WHERE symbol NOT LIKE 'SECTOR::%'")
        sym_count = cur.fetchone()[0] or 0
        report["checks"]["symbol_count"] = {"ok": sym_count > 0, "value": sym_count}

        # 4. Open positions (if table exists)
        try:
            cur.execute("SELECT COUNT(*) FROM portfolio_positions WHERE status = 'OPEN'")
            open_count = cur.fetchone()[0] or 0
            report["checks"]["open_positions"] = {"ok": True, "value": open_count}
        except sqlite3.OperationalError:
            pass  # Table doesn't exist yet

        conn.close()
    except sqlite3.Error as e:
        report["status"] = "CRITICAL"
        report["checks"]["db_connect"] = {"ok": False, "detail": str(e)}

    # Set overall status
    if any(not c.get("ok", True) for c in report["checks"].values()):
        if report["status"] == "OK":
            report["status"] = "WARNING"

    return report


def print_health_report(report: Dict[str, Any]) -> None:
    """Pretty-print a health check report."""
    status = report["status"]
    icon = {"OK": "OK", "WARNING": "WARN", "CRITICAL": "CRIT"}[status]
    print(f"[health] {icon} - {report['timestamp']}")
    for name, check in report["checks"].items():
        ok_str = "OK" if check.get("ok", True) else "FAIL"
        detail = check.get("value") or check.get("detail") or check.get("last_date", "")
        extra = ""
        if "staleness_days" in check:
            extra = f" ({check['staleness_days']}d stale)"
        print(f"  {ok_str:4s} {name}: {detail}{extra}")
