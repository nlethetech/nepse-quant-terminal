#!/usr/bin/env python3
"""
Institutional portfolio/risk engine CLI.

Backed by:
- portfolio_positions state table
- immutable trade_ledger event store
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
from requests.exceptions import RequestException, Timeout

from backend.quant_pro.institutional import PortfolioStateMachine, init_institutional_tables
from backend.quant_pro.vendor_api import fetch_latest_ltp
from backend.quant_pro.database import get_db_path
from backend.quant_pro.paths import ensure_dir, get_trading_runtime_dir

logger = logging.getLogger(__name__)


DEFAULT_DB_FILE = str(get_db_path())
DEFAULT_ORDERS_FILE = ensure_dir(get_trading_runtime_dir(__file__)) / "buy_orders.csv"


def parse_ltp_overrides(raw: str) -> Dict[str, float]:
    """
    Parse price overrides in format: SYMBOL=123.4,SYM2=456.7
    """
    out: Dict[str, float] = {}
    raw = (raw or "").strip()
    if not raw:
        return out
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid ltp override token: {part}")
        symbol, value = part.split("=", 1)
        out[symbol.strip().upper()] = float(value.strip())
    return out


def fetch_ltp(conn: sqlite3.Connection, symbol: str) -> Optional[float]:
    # First try vendor live quote.
    try:
        ltp = fetch_latest_ltp(symbol)
        if ltp is not None and ltp > 0:
            return float(ltp)
    except (RequestException, Timeout, ValueError) as e:
        logger.warning("Live LTP fetch failed for %s: %s", symbol, e)

    # Fallback to latest DB close.
    cur = conn.cursor()
    cur.execute(
        """
        SELECT close, date
        FROM stock_prices
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT 1
        """,
        (symbol,),
    )
    row = cur.fetchone()
    if row and row[0] is not None and float(row[0]) > 0:
        logger.info("Using DB fallback price for %s (date=%s)", symbol, row[1])
        return float(row[0])
    logger.warning("No price available for %s from vendor or DB", symbol)
    return None


def command_open(
    sm: PortfolioStateMachine,
    symbol: str,
    quantity: int,
    price: float,
    fees_bps: float,
    strategy_tag: str,
) -> int:
    notional = quantity * price
    fees = (fees_bps / 10_000.0) * notional
    position_id = sm.open_position(
        symbol=symbol.upper(),
        quantity=quantity,
        entry_price=price,
        fees=fees,
        strategy_tag=strategy_tag,
    )
    print(
        f"[open] position_id={position_id} symbol={symbol.upper()} "
        f"qty={quantity} price={price:.2f} fees={fees:.2f}"
    )
    return 0


def validate_orders_csv(df: pd.DataFrame) -> list[str]:
    """Validate orders CSV schema. Returns list of error messages (empty = valid)."""
    errors = []
    required_cols = {"Symbol", "Shares"}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors

    if df["Symbol"].isna().any():
        errors.append(f"Found {df['Symbol'].isna().sum()} rows with missing Symbol")

    if df["Shares"].isna().any():
        errors.append(f"Found {df['Shares'].isna().sum()} rows with missing Shares")

    # Check shares are positive integers
    for idx, row in df.iterrows():
        try:
            shares = int(float(str(row["Shares"]).replace(",", "")))
            if shares <= 0:
                errors.append(f"Row {idx}: Shares must be positive, got {shares}")
        except (ValueError, TypeError):
            errors.append(f"Row {idx}: Invalid Shares value: {row['Shares']}")

    return errors


def command_open_from_orders(
    sm: PortfolioStateMachine,
    conn: sqlite3.Connection,
    orders_file: Path,
    fees_bps: float,
    strategy_tag: str,
    use_live_prices: bool,
) -> int:
    if not orders_file.exists():
        print(f"[open-from-orders] file not found: {orders_file}")
        return 1

    df = pd.read_csv(orders_file)

    # Validate CSV schema
    validation_errors = validate_orders_csv(df)
    if validation_errors:
        print("[open-from-orders] CSV validation failed:")
        for err in validation_errors:
            print(f"  - {err}")
        return 1

    opened = 0
    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).strip().upper()
        try:
            quantity = int(float(str(row["Shares"]).replace(",", "")))
        except Exception:
            print(f"[open-from-orders] skip {symbol}: invalid Shares={row.get('Shares')}")
            continue
        if quantity <= 0:
            continue

        price: Optional[float] = None
        if "Price" in df.columns and pd.notna(row.get("Price")):
            try:
                price = float(row["Price"])
            except Exception:
                price = None
        if (price is None or price <= 0) and use_live_prices:
            price = fetch_ltp(conn, symbol)
        if price is None or price <= 0:
            print(f"[open-from-orders] skip {symbol}: missing valid price")
            continue

        notional = quantity * price
        fees = (fees_bps / 10_000.0) * notional
        position_id = sm.open_position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            fees=fees,
            strategy_tag=strategy_tag,
            metadata={"source": str(orders_file)},
        )
        opened += 1
        print(
            f"[open-from-orders] opened position_id={position_id} symbol={symbol} "
            f"qty={quantity} price={price:.2f} fees={fees:.2f}"
        )

    print(f"[open-from-orders] opened={opened}")
    return 0


def command_close(
    sm: PortfolioStateMachine,
    conn: sqlite3.Connection,
    position_id: int,
    price: Optional[float],
    fees_bps: float,
    reason: str,
    use_live_price: bool,
) -> int:
    pos = sm.get_position(position_id)
    if pos is None:
        print(f"[close] position_id={position_id} not found")
        return 1
    if pos.status != "OPEN":
        print(f"[close] position_id={position_id} is not OPEN")
        return 1

    exit_price = price
    if (exit_price is None or exit_price <= 0) and use_live_price:
        exit_price = fetch_ltp(conn, pos.symbol)
    if exit_price is None or exit_price <= 0:
        print("[close] valid exit price required (or --use-live-price)")
        return 1

    fees = (fees_bps / 10_000.0) * (exit_price * pos.quantity)
    sm.close_position(
        position_id=position_id,
        exit_price=exit_price,
        fees=fees,
        reason=reason,
    )
    print(
        f"[close] position_id={position_id} symbol={pos.symbol} qty={pos.quantity} "
        f"price={exit_price:.2f} fees={fees:.2f} reason={reason}"
    )
    return 0


def command_risk_check(
    sm: PortfolioStateMachine,
    conn: sqlite3.Connection,
    ltp_overrides: Dict[str, float],
    use_live_prices: bool,
    apply_actions: bool,
    fees_bps: float,
) -> int:
    open_positions = sm.list_open_positions()
    if not open_positions:
        print("[risk-check] no open positions")
        return 0

    ltp_by_symbol: Dict[str, float] = dict(ltp_overrides)
    for pos in open_positions:
        if pos.symbol in ltp_by_symbol:
            continue
        if use_live_prices:
            ltp = fetch_ltp(conn, pos.symbol)
            if ltp is not None and ltp > 0:
                ltp_by_symbol[pos.symbol] = ltp

    signals = sm.evaluate_risk_signals(ltp_by_symbol)
    if not signals:
        print("[risk-check] no risk exits triggered")
        return 0

    print("[risk-check] triggers:")
    for sig in signals:
        print(
            f"  position_id={sig.position_id} symbol={sig.symbol} action={sig.action} "
            f"reason={sig.reason} ltp={sig.ltp:.2f} "
            f"hard={sig.hard_stop_price:.2f} trail={sig.trailing_stop_price:.2f} tp={sig.take_profit_price:.2f}"
        )

    if apply_actions:
        sm.apply_risk_actions(signals, fees_bps=fees_bps)
        print(f"[risk-check] applied_actions={len(signals)}")
    else:
        print("[risk-check] dry-run only (use --apply to execute exits)")
    return 0


def command_list_open(sm: PortfolioStateMachine) -> int:
    rows = sm.list_open_positions()
    if not rows:
        print("[list-open] no open positions")
        return 0
    print("[list-open] open positions:")
    for p in rows:
        print(
            f"  id={p.position_id} symbol={p.symbol} qty={p.quantity} entry={p.avg_entry_price:.2f} "
            f"high={p.high_watermark:.2f} hard_stop={p.hard_stop_pct:.2%} "
            f"trailing={p.trailing_stop_pct:.2%} tp={p.take_profit_pct:.2%}"
        )
    return 0


def command_summary(sm: PortfolioStateMachine) -> int:
    s = sm.ledger_summary()
    print(
        "[summary] "
        f"ledger_events={s['ledger_events']} open_positions={s['open_positions']} "
        f"realized_pnl={s['realized_pnl']:.2f}"
    )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Institutional portfolio state-machine engine")
    p.add_argument("--db-file", default=DEFAULT_DB_FILE)
    p.add_argument(
        "--action",
        required=True,
        choices=[
            "open",
            "open-from-orders",
            "close",
            "risk-check",
            "list-open",
            "summary",
        ],
    )

    # open
    p.add_argument("--symbol")
    p.add_argument("--qty", type=int, default=0)
    p.add_argument("--price", type=float, default=0.0)
    p.add_argument("--strategy-tag", default="default")

    # open-from-orders
    p.add_argument("--orders-file", type=Path, default=DEFAULT_ORDERS_FILE)

    # close
    p.add_argument("--position-id", type=int, default=0)
    p.add_argument("--reason", default="MANUAL_EXIT")

    # risk-check
    p.add_argument("--ltp", default="", help="Overrides: SYMBOL=123.4,SYM2=456.7")
    p.add_argument("--apply", action="store_true", help="Apply risk exits")

    # pricing behavior
    p.add_argument("--use-live-prices", action="store_true")
    p.add_argument("--use-live-price", action="store_true", help="Alias for close action")

    # transaction assumptions
    p.add_argument("--fees-bps", type=float, default=60.0, help="Fees in basis points per side")
    return p.parse_args(argv)


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    db_path = Path(args.db_file)
    if not db_path.exists():
        print(f"[engine] DB file not found: {db_path}")
        return 1

    if not (0 <= args.fees_bps <= 1000):
        print("[engine] --fees-bps must be between 0 and 1000")
        return 1

    conn = connect_db(db_path)
    try:
        init_institutional_tables(conn)
        sm = PortfolioStateMachine(conn)
        action = args.action

        if action == "open":
            if not args.symbol or args.qty <= 0 or args.price <= 0:
                print("[open] require --symbol, --qty > 0, --price > 0")
                return 1
            return command_open(
                sm=sm,
                symbol=args.symbol,
                quantity=args.qty,
                price=args.price,
                fees_bps=args.fees_bps,
                strategy_tag=args.strategy_tag,
            )

        if action == "open-from-orders":
            return command_open_from_orders(
                sm=sm,
                conn=conn,
                orders_file=args.orders_file,
                fees_bps=args.fees_bps,
                strategy_tag=args.strategy_tag,
                use_live_prices=args.use_live_prices,
            )

        if action == "close":
            if args.position_id <= 0:
                print("[close] require --position-id > 0")
                return 1
            return command_close(
                sm=sm,
                conn=conn,
                position_id=args.position_id,
                price=args.price if args.price > 0 else None,
                fees_bps=args.fees_bps,
                reason=args.reason,
                use_live_price=args.use_live_price or args.use_live_prices,
            )

        if action == "risk-check":
            overrides = parse_ltp_overrides(args.ltp)
            return command_risk_check(
                sm=sm,
                conn=conn,
                ltp_overrides=overrides,
                use_live_prices=args.use_live_prices,
                apply_actions=args.apply,
                fees_bps=args.fees_bps,
            )

        if action == "list-open":
            return command_list_open(sm)

        if action == "summary":
            return command_summary(sm)

        print(f"[engine] unsupported action: {action}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
