#!/usr/bin/env python3
"""
Institutional CI gates for NEPSE quant stack.

Gates:
1) DB freshness + integrity checks.
2) Ingestion run recency/SLA checks.
3) Leakage audit must pass.
4) Backtest regression thresholds.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import local backtest engine for metric gating.
from backend.backtesting.simple_backtest import run_backtest


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


def load_config(path: Path) -> Dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _to_date(s: Optional[str]) -> Optional[datetime.date]:
    if not s:
        return None
    return pd.Timestamp(s).date()


def data_freshness_and_integrity_gate(conn: sqlite3.Connection, cfg: Dict) -> GateResult:
    max_staleness = int(cfg["data_freshness"]["max_staleness_days"])
    min_symbols = int(cfg["data_freshness"]["min_symbol_count"])

    cur = conn.cursor()
    cur.execute("SELECT MAX(date), COUNT(DISTINCT symbol) FROM stock_prices")
    latest_date_raw, symbol_count = cur.fetchone()
    latest_date = _to_date(latest_date_raw)
    if latest_date is None:
        return GateResult("data_freshness", False, "stock_prices has no data")

    staleness = (datetime.now(timezone.utc).date() - latest_date).days
    if staleness > max_staleness:
        return GateResult(
            "data_freshness",
            False,
            f"staleness={staleness}d exceeds max={max_staleness}d (latest={latest_date})",
        )
    if int(symbol_count or 0) < min_symbols:
        return GateResult(
            "data_freshness",
            False,
            f"symbol_count={symbol_count} below minimum={min_symbols}",
        )

    # Data quality checks
    cur.execute(
        """
        SELECT COUNT(*)
        FROM stock_prices
        WHERE close <= 0 OR open <= 0 OR high <= 0 OR low <= 0 OR volume < 0
        """
    )
    bad_rows = int(cur.fetchone()[0] or 0)
    if bad_rows > 0:
        return GateResult("data_integrity", False, f"invalid ohlcv rows={bad_rows}")

    cur.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT symbol, date, COUNT(*) AS c
            FROM stock_prices
            GROUP BY symbol, date
            HAVING c > 1
        )
        """
    )
    dupes = int(cur.fetchone()[0] or 0)
    if dupes > 0:
        return GateResult("data_integrity", False, f"duplicate symbol/date keys={dupes}")

    return GateResult(
        "data_freshness",
        True,
        f"latest={latest_date} staleness={staleness}d symbols={symbol_count} integrity=ok",
    )


def ingestion_recency_gate(conn: sqlite3.Connection, cfg: Dict) -> GateResult:
    if not bool(cfg["ingestion"]["require_recent_run"]):
        return GateResult("ingestion_recency", True, "disabled by config")

    max_age_hours = int(cfg["ingestion"]["max_run_age_hours"])
    accepted_status = set(cfg["ingestion"]["accepted_status"])

    cur = conn.cursor()
    cur.execute(
        """
        SELECT run_id, ended_at_utc, status, latest_market_date_after, freshness_days_after
        FROM ingestion_runs
        WHERE ended_at_utc IS NOT NULL
        ORDER BY run_id DESC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if row is None:
        return GateResult("ingestion_recency", False, "no completed ingestion_runs found")

    run_id, ended_at_raw, status, latest_market_date, freshness_days = row
    if status not in accepted_status:
        return GateResult(
            "ingestion_recency",
            False,
            f"latest run_id={run_id} has status={status} not in {sorted(accepted_status)}",
        )

    try:
        ended_at = datetime.fromisoformat(str(ended_at_raw))
    except Exception:
        return GateResult("ingestion_recency", False, f"invalid ended_at_utc={ended_at_raw}")
    if ended_at.tzinfo is None:
        ended_at = ended_at.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - ended_at).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        return GateResult(
            "ingestion_recency",
            False,
            f"latest run age={age_hours:.1f}h exceeds max={max_age_hours}h",
        )

    return GateResult(
        "ingestion_recency",
        True,
        (
            f"run_id={run_id} status={status} age={age_hours:.1f}h "
            f"latest_market_date={latest_market_date} freshness_days={freshness_days}"
        ),
    )


def leakage_gate(cfg: Dict) -> GateResult:
    timeout_sec = int(cfg["timeouts"]["leakage_seconds"])
    cmd = [sys.executable, str(ROOT / "test_leakage.py")]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        snippet = "\n".join(proc.stdout.strip().splitlines()[-40:])
        return GateResult("leakage", False, f"test_leakage.py failed\n{snippet}")
    return GateResult("leakage", True, "test_leakage.py passed")


def backtest_regression_gate(cfg: Dict) -> GateResult:
    backtest_cfg = cfg["backtest"]
    result = run_backtest(
        start_date=backtest_cfg["start_date"],
        end_date=backtest_cfg["end_date"],
        holding_days=int(backtest_cfg["holding_days"]),
        max_positions=int(backtest_cfg["max_positions"]),
        signal_types=list(backtest_cfg["signals"]),
    )

    min_trades = int(backtest_cfg["min_trades"])
    min_sharpe = float(backtest_cfg["min_sharpe"])
    min_win_rate = float(backtest_cfg["min_win_rate"])
    max_drawdown = float(backtest_cfg["max_drawdown"])

    if result.total_trades < min_trades:
        return GateResult(
            "backtest_regression",
            False,
            f"total_trades={result.total_trades} < min_trades={min_trades}",
        )
    if result.sharpe_ratio < min_sharpe:
        return GateResult(
            "backtest_regression",
            False,
            f"sharpe={result.sharpe_ratio:.2f} < min_sharpe={min_sharpe:.2f}",
        )
    if result.win_rate < min_win_rate:
        return GateResult(
            "backtest_regression",
            False,
            f"win_rate={result.win_rate:.2%} < min_win_rate={min_win_rate:.2%}",
        )
    if result.max_drawdown < max_drawdown:
        return GateResult(
            "backtest_regression",
            False,
            f"max_drawdown={result.max_drawdown:.2%} < allowed_floor={max_drawdown:.2%}",
        )

    return GateResult(
        "backtest_regression",
        True,
        (
            f"trades={result.total_trades} sharpe={result.sharpe_ratio:.2f} "
            f"win_rate={result.win_rate:.2%} max_dd={result.max_drawdown:.2%}"
        ),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run institutional quality gates")
    p.add_argument("--db-file", default="nepse_market_data.db")
    p.add_argument("--config", default=str(ROOT / "ci" / "quality_gates.toml"))
    p.add_argument("--skip-leakage", action="store_true")
    p.add_argument("--skip-backtest", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    conn = sqlite3.connect(args.db_file)
    try:
        results = []
        results.append(data_freshness_and_integrity_gate(conn, cfg))
        results.append(ingestion_recency_gate(conn, cfg))
        if not args.skip_leakage:
            results.append(leakage_gate(cfg))
        if not args.skip_backtest:
            results.append(backtest_regression_gate(cfg))
    finally:
        conn.close()

    failed = [r for r in results if not r.passed]
    print("=" * 72)
    print("QUALITY GATES")
    print("=" * 72)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")
    print("-" * 72)
    print(f"Total={len(results)} Passed={len(results)-len(failed)} Failed={len(failed)}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
