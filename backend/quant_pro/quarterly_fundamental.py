"""Point-in-time quarterly fundamental signals for backtesting."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, Optional

import pandas as pd

from backend.quant_pro.alpha_practical import (
    AlphaSignal,
    FundamentalData,
    FundamentalScanner,
    SignalType,
)


def _safe_float(value) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _annualize_eps(eps: Optional[float], quarter: Optional[int]) -> Optional[float]:
    if eps is None:
        return None
    if not quarter or quarter <= 0:
        return eps
    return eps * (4.0 / quarter)


def _growth_ratio(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous in (None, 0):
        return None
    return (current - previous) / abs(previous)


def _effective_announcement_date(row: pd.Series) -> Optional[pd.Timestamp]:
    raw = row.get("announcement_date")
    if pd.notna(raw) and raw:
        try:
            return pd.Timestamp(raw)
        except Exception:
            pass

    raw = row.get("report_date")
    if pd.notna(raw) and raw:
        try:
            return pd.Timestamp(raw) + pd.Timedelta(days=30)
        except Exception:
            pass

    raw = row.get("scraped_at_utc")
    if pd.notna(raw) and raw:
        try:
            return pd.Timestamp(raw)
        except Exception:
            pass
    return None


@dataclass
class QuarterlyFundamentalModel:
    """Historical quarterly-fundamental data prepared for point-in-time scans."""

    quarterly_earnings: pd.DataFrame
    fundamentals: pd.DataFrame

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> "QuarterlyFundamentalModel":
        try:
            quarterly = pd.read_sql_query("SELECT * FROM quarterly_earnings", conn)
        except Exception:
            quarterly = pd.DataFrame()
        try:
            fundamentals = pd.read_sql_query("SELECT * FROM fundamentals", conn)
        except Exception:
            fundamentals = pd.DataFrame()
        return cls.from_frames(quarterly, fundamentals)

    @classmethod
    def from_frames(
        cls,
        quarterly_earnings: pd.DataFrame,
        fundamentals: pd.DataFrame,
    ) -> "QuarterlyFundamentalModel":
        quarterly = quarterly_earnings.copy() if quarterly_earnings is not None else pd.DataFrame()
        if not quarterly.empty:
            quarterly["effective_announcement_date"] = quarterly.apply(_effective_announcement_date, axis=1)
            quarterly = quarterly[quarterly["effective_announcement_date"].notna()].copy()
            quarterly["quarter"] = pd.to_numeric(quarterly["quarter"], errors="coerce")
            quarterly = quarterly.sort_values(
                ["symbol", "effective_announcement_date", "fiscal_year", "quarter"]
            )

        fund = fundamentals.copy() if fundamentals is not None else pd.DataFrame()
        if not fund.empty and "date" in fund.columns:
            fund["date"] = pd.to_datetime(fund["date"], errors="coerce")
            fund = fund[fund["date"].notna()].copy()
            fund = fund.sort_values(["symbol", "date"])

        return cls(quarterly_earnings=quarterly, fundamentals=fund)

    def build_fundamentals(
        self,
        as_of_date: datetime | pd.Timestamp,
        current_prices: Dict[str, float],
        *,
        sector_lookup: Callable[[str], Optional[str]],
        symbols: Optional[Iterable[str]] = None,
    ) -> Dict[str, FundamentalData]:
        """Build a point-in-time fundamental snapshot from historical tables."""
        as_of_ts = pd.Timestamp(as_of_date)
        symbol_filter = {str(symbol).strip().upper() for symbol in (symbols or []) if str(symbol).strip()}
        out: Dict[str, FundamentalData] = {}

        fund_latest: Dict[str, pd.Series] = {}
        if not self.fundamentals.empty:
            fund_df = self.fundamentals[self.fundamentals["date"] <= as_of_ts]
            if symbol_filter:
                fund_df = fund_df[fund_df["symbol"].isin(symbol_filter)]
            if not fund_df.empty:
                fund_latest = {
                    symbol: grp.iloc[-1]
                    for symbol, grp in fund_df.groupby("symbol", sort=False)
                }

        qe_df = self.quarterly_earnings
        if qe_df.empty:
            return out

        qe_df = qe_df[qe_df["effective_announcement_date"] <= as_of_ts]
        if symbol_filter:
            qe_df = qe_df[qe_df["symbol"].isin(symbol_filter)]
        if qe_df.empty:
            return out

        for symbol, group in qe_df.groupby("symbol", sort=False):
            ordered = group.sort_values(
                ["effective_announcement_date", "fiscal_year", "quarter"],
                ascending=[False, False, False],
            )
            latest = ordered.iloc[0]
            previous = ordered.iloc[1] if len(ordered) > 1 else None
            snapshot = fund_latest.get(symbol)

            q_num = int(latest["quarter"]) if pd.notna(latest["quarter"]) else None
            latest_eps = _safe_float(latest.get("eps"))
            latest_book = _safe_float(latest.get("book_value"))
            annualized_eps = _annualize_eps(latest_eps, q_num)
            current_price = _safe_float(current_prices.get(symbol))

            snapshot_pe = _safe_float(snapshot.get("pe_ratio")) if snapshot is not None else None
            snapshot_pb = _safe_float(snapshot.get("pb_ratio")) if snapshot is not None else None
            snapshot_roe = _safe_float(snapshot.get("roe")) if snapshot is not None else None
            snapshot_dividend = _safe_float(snapshot.get("dividend_yield")) if snapshot is not None else None

            pe_ratio = snapshot_pe
            if current_price and annualized_eps and annualized_eps > 0:
                pe_ratio = current_price / annualized_eps

            pb_ratio = snapshot_pb
            if current_price and latest_book and latest_book > 0:
                pb_ratio = current_price / latest_book

            out[symbol] = FundamentalData(
                symbol=symbol,
                sector=str(snapshot.get("sector")).strip() if snapshot is not None and snapshot.get("sector") else (sector_lookup(symbol) or "Others"),
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                eps=annualized_eps,
                book_value=latest_book,
                dividend_yield=snapshot_dividend,
                roe=snapshot_roe,
                revenue_growth_qoq=_growth_ratio(
                    _safe_float(latest.get("revenue")),
                    _safe_float(previous.get("revenue")) if previous is not None else None,
                ),
                profit_growth_qoq=_growth_ratio(
                    _safe_float(latest.get("net_profit")),
                    _safe_float(previous.get("net_profit")) if previous is not None else None,
                ),
                eps_growth_qoq=_growth_ratio(
                    latest_eps,
                    _safe_float(previous.get("eps")) if previous is not None else None,
                ),
                book_value_growth_qoq=_growth_ratio(
                    latest_book,
                    _safe_float(previous.get("book_value")) if previous is not None else None,
                ),
                latest_net_profit=_safe_float(latest.get("net_profit")),
                latest_revenue=_safe_float(latest.get("revenue")),
                data_source="quarterly_earnings_pit" + ("+fundamentals_snapshot" if snapshot is not None else ""),
            )

        return out

    def generate_signals(
        self,
        as_of_date: datetime | pd.Timestamp,
        current_prices: Dict[str, float],
        *,
        sector_lookup: Callable[[str], Optional[str]],
        symbols: Optional[Iterable[str]] = None,
    ) -> list[AlphaSignal]:
        """Generate point-in-time quarterly fundamental signals."""
        scanner = FundamentalScanner()
        fundamentals = self.build_fundamentals(
            as_of_date,
            current_prices,
            sector_lookup=sector_lookup,
            symbols=symbols,
        )
        for item in fundamentals.values():
            scanner.update_fundamentals(item)

        signals = scanner.scan()
        for signal in signals:
            signal.signal_type = SignalType.QUARTERLY_FUNDAMENTAL
        return signals
