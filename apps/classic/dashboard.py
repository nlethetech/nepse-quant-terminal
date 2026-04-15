#!/usr/bin/env python3
"""
NEPSE Bloomberg-Style Terminal Dashboard

Keys:  1-5 = switch tab   B = Buy   S = Sell   L = Lookup   R = Refresh   Q = Quit
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sqlite3
import sys
import termios
import threading
import time
import tty
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from backend.quant_pro.database import get_db_path
from backend.quant_pro.paths import ensure_dir, get_trading_runtime_dir

console = Console(highlight=False)
PAPER_PORTFOLIO_FILE = ensure_dir(get_trading_runtime_dir(__file__)) / "paper_portfolio.csv"

# ── palette ───────────────────────────────────────────────────────────────────
AMBER   = "#ffaf00"
WHITE   = "#e8e8e8"
DIM     = "#555555"
LABEL   = "#888888"
BORDER  = "#333333"
GAIN_HI = "#00ff7f"
GAIN    = "#00cc60"
LOSS_HI = "#ff4545"
LOSS    = "#cc3333"
CYAN    = "#00cfff"
YELLOW  = "#ffd700"
PURPLE  = "#c87fff"
BLUE    = "#4d9fff"

# ── ANSI helpers ──────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

def _vlen(s: str) -> int:
    """Visible length of ANSI string."""
    return len(_ANSI_RE.sub('', s))

def _vpad(s: str, w: int) -> str:
    """Pad ANSI string to exact visible width w."""
    diff = w - _vlen(s)
    return s + ' ' * diff if diff > 0 else s

# ── Rich helpers ──────────────────────────────────────────────────────────────

def _pct(v: float, bold: bool = False) -> Text:
    s = "+" if v >= 0 else ""
    c = (GAIN_HI if v > 2 else GAIN) if v > 0 else (LOSS_HI if v < -2 else LOSS) if v < 0 else WHITE
    return Text(f"{s}{v:.2f}%", style=f"bold {c}" if bold else c)

def _vol(v: float) -> str:
    if v >= 1_000_000: return f"{v/1_000_000:.2f}M"
    if v >= 1_000:     return f"{v/1_000:.0f}K"
    return str(int(v))

def _npr(v: float) -> Text:
    s = "+" if v >= 0 else ""
    c = GAIN_HI if v > 0 else (LOSS_HI if v < 0 else WHITE)
    a = abs(v)
    t = f"{s}NPR {v/1_000_000:.2f}M" if a >= 1_000_000 else \
        f"{s}NPR {v/1_000:.1f}K"     if a >= 1_000 else f"{s}NPR {v:.0f}"
    return Text(t, style=f"bold {c}")

def _db() -> sqlite3.Connection:
    return sqlite3.connect(str(get_db_path()))

def _tbl(cols) -> Table:
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style=f"bold {AMBER}",
              border_style=BORDER, expand=True, padding=(0, 1))
    for h, j, w in cols:
        t.add_column(h, justify=j, min_width=w, no_wrap=True)
    return t

def _panel(content, title: str, colour: str = BORDER) -> Panel:
    return Panel(content, title=f"[bold {colour}]{title}[/bold {colour}]",
                 border_style=colour, padding=(0, 1))

def _rich_to_str(renderable, width: int) -> str:
    """Render a Rich object to an ANSI string at exact width."""
    buf = io.StringIO()
    c = Console(file=buf, width=width, force_terminal=True,
                color_system="truecolor", highlight=False)
    c.print(renderable, end="")
    return buf.getvalue()

def _hgrid(panels: list, total_width: int) -> str:
    """Render panels side-by-side: each panel rendered independently at exact width,
    then lines combined manually. No Rich grid layout involved."""
    n = len(panels)
    col_w = total_width // n

    all_lines: list[list[str]] = []
    for i, p in enumerate(panels):
        w = col_w + (total_width - col_w * n if i == n - 1 else 0)
        text = _rich_to_str(p, w)
        lines = text.split('\n')
        # Remove trailing empty line if present
        while lines and lines[-1].strip() == '':
            lines.pop()
        # Pad every line to exact visible width
        lines = [_vpad(line, w) for line in lines]
        all_lines.append((lines, w))

    max_h = max(len(lines) for lines, _ in all_lines)
    for lines, w in all_lines:
        while len(lines) < max_h:
            lines.append(' ' * w)

    combined = []
    for row in range(max_h):
        combined.append(''.join(lines[row] for lines, _ in all_lines))
    return '\n'.join(combined)

# ── data ──────────────────────────────────────────────────────────────────────

class MD:
    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.err: Optional[str] = None
        self.latest = self.prev_d = "—"
        self.df = self.gainers = self.losers = self.vol_top = pd.DataFrame()
        self.near_hi = self.near_lo = self.quotes = self.nepse = self.corp = pd.DataFrame()
        self.adv = self.dec = self.unch = 0
        self.ts = datetime.now()
        # Optional TMS live feed; when None, MD behaves exactly as before.
        self.tms_balance: Optional[Dict] = None
        self.tms_indices: Dict[str, object] = {}
        self.refresh()

    def refresh(self):
        try:
            c = _db()
            quotes = pd.read_sql_query(
                "SELECT symbol,last_traded_price ltp,previous_close prev_close,percentage_change pc,"
                "total_trade_quantity vol,fetched_at_utc ts FROM market_quotes "
                "ORDER BY fetched_at_utc DESC",
                c,
            ).drop_duplicates("symbol")
            for col in ("ltp", "prev_close", "pc", "vol"):
                quotes[col] = pd.to_numeric(quotes[col], errors="coerce").fillna(0.0)

            lat_d = pd.read_sql_query(
                "SELECT MAX(date) d FROM stock_prices WHERE symbol!='NEPSE'", c)["d"].iloc[0]
            prv_d = None
            if pd.notna(lat_d):
                prv_d = pd.read_sql_query(
                    "SELECT MAX(date) d FROM stock_prices WHERE symbol!='NEPSE' AND date<?",
                    c, params=(lat_d,))["d"].iloc[0]

            if pd.notna(lat_d):
                lat = pd.read_sql_query(
                    "SELECT symbol,open,high,low,close,volume FROM stock_prices WHERE date=?",
                    c, params=(lat_d,))
            else:
                lat = pd.DataFrame(columns=["symbol", "open", "high", "low", "close", "volume"])

            lat = lat[lat["symbol"] != "NEPSE"].copy()
            if not quotes.empty:
                quote_fallback = quotes.rename(
                    columns={"ltp": "quote_ltp", "prev_close": "quote_prev", "vol": "quote_vol"}
                )[["symbol", "quote_ltp", "quote_prev", "quote_vol"]]
            else:
                quote_fallback = pd.DataFrame(columns=["symbol", "quote_ltp", "quote_prev", "quote_vol"])

            if lat.empty and not quote_fallback.empty:
                df = pd.DataFrame({
                    "symbol": quote_fallback["symbol"],
                    "open": quote_fallback["quote_ltp"],
                    "high": quote_fallback["quote_ltp"],
                    "low": quote_fallback["quote_ltp"],
                    "close": quote_fallback["quote_ltp"],
                    "volume": quote_fallback["quote_vol"],
                })
            else:
                df = lat.copy()

            if not quote_fallback.empty:
                df = df.merge(quote_fallback, on="symbol", how="left")
            else:
                df["quote_ltp"] = pd.NA
                df["quote_prev"] = pd.NA
                df["quote_vol"] = pd.NA

            if pd.notna(prv_d):
                prv = pd.read_sql_query(
                    "SELECT symbol,close prev FROM stock_prices WHERE date=?",
                    c, params=(prv_d,))
                df = df.merge(prv, on="symbol", how="left")
            else:
                df["prev"] = pd.NA

            for col in ("open", "high", "low", "close", "volume", "quote_ltp", "quote_prev", "quote_vol", "prev"):
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["close"] = df["close"].fillna(df["quote_ltp"])
            df["open"] = df["open"].fillna(df["close"])
            df["high"] = df["high"].fillna(df["close"])
            df["low"] = df["low"].fillna(df["close"])
            df["volume"] = df["volume"].fillna(df["quote_vol"]).fillna(0.0)
            df["prev"] = df["prev"].fillna(df["quote_prev"])
            prev_mask = df["prev"].fillna(0) > 0
            df["chg"] = 0.0
            df.loc[prev_mask, "chg"] = (df.loc[prev_mask, "close"] - df.loc[prev_mask, "prev"]) / df.loc[prev_mask, "prev"] * 100
            df["turn"] = df["close"].fillna(0.0) * df["volume"].fillna(0.0)
            df = df.drop(columns=["quote_ltp", "quote_prev", "quote_vol"], errors="ignore")

            display_session = lat_d
            if pd.isna(display_session) and not quotes.empty and "ts" in quotes.columns:
                try:
                    display_session = pd.to_datetime(quotes["ts"].iloc[0]).strftime("%Y-%m-%d")
                except Exception:
                    display_session = "—"

            self.latest = display_session if pd.notna(display_session) else "—"
            self.prev_d = prv_d if pd.notna(prv_d) else "—"
            self.df = df
            filt = df[df["chg"].abs() <= 12]
            self.gainers = filt.nlargest(self.top_n, "chg")
            self.losers  = filt.nsmallest(self.top_n, "chg")
            self.vol_top = df.nlargest(self.top_n, "volume")
            self.adv  = int((df["chg"] > 0).sum())
            self.dec  = int((df["chg"] < 0).sum())
            self.unch = int((df["chg"] == 0).sum())

            yr = pd.read_sql_query(
                "SELECT symbol,MAX(high) h,MIN(low) l FROM stock_prices "
                "WHERE date>=date(?,'-365 days') AND symbol!='NEPSE' GROUP BY symbol",
                c, params=(lat_d,))
            d52 = df.merge(yr, on="symbol", how="left")
            self.near_hi = d52[d52["close"] >= d52["h"] * 0.97].nlargest(self.top_n, "chg")
            self.near_lo = d52[d52["close"] <= d52["l"] * 1.03].nsmallest(self.top_n, "chg")

            self.quotes = quotes

            self.nepse = pd.read_sql_query(
                "SELECT close FROM stock_prices WHERE symbol='NEPSE' "
                "ORDER BY date DESC LIMIT 2", c)

            try:
                corp = pd.read_sql_query(
                    "SELECT symbol,bookclose_date,cash_dividend_pct,bonus_share_pct "
                    "FROM corporate_actions WHERE bookclose_date>? AND bookclose_date<=? "
                    "ORDER BY bookclose_date", c,
                    params=(datetime.now().strftime("%Y-%m-%d"),
                            (datetime.now()+timedelta(days=30)).strftime("%Y-%m-%d")))
                corp["bookclose_date"] = pd.to_datetime(corp["bookclose_date"])
                self.corp = corp
            except Exception:
                self.corp = pd.DataFrame()

            c.close(); self.ts = datetime.now(); self.err = None
        except Exception as e:
            self.err = str(e)

        # Pull cached TMS snapshots if a source is attached — silent on failure.
        if self.tms is not None:
            try:
                self.tms_balance = self.tms.balance()
                self.tms_indices = self.tms.indices()
            except Exception:
                pass

    def ltps(self) -> Dict[str, float]:
        base: Dict[str, float] = {
            str(r.symbol): float(r.ltp)
            for r in self.quotes.itertuples() if float(r.ltp) > 0
        }
        # Live TMS ticker overlays DB quotes whenever the WS has fresh frames.
        if self.tms is not None:
            try:
                live = self.tms.ltps() if self.tms.is_live() else {}
            except Exception:
                live = {}
            if live:
                base.update(live)
        return base

    def detail(self, sym: str, limit: int = 60) -> Optional[Dict]:
        c = _db()
        h = pd.read_sql_query(
            "SELECT date,open,high,low,close,volume FROM stock_prices "
            "WHERE symbol=? ORDER BY date DESC LIMIT ?", c, params=(sym, limit))
        ca = pd.read_sql_query(
            "SELECT bookclose_date,cash_dividend_pct,bonus_share_pct "
            "FROM corporate_actions WHERE symbol=? ORDER BY bookclose_date DESC LIMIT 5",
            c, params=(sym,))
        c.close()
        if h.empty: return None
        lat = h.iloc[0]; prv = h.iloc[1] if len(h) > 1 else lat
        chg = (lat["close"]-prv["close"])/prv["close"]*100 if prv["close"] else 0
        return {"h": h, "lat": lat, "chg": chg, "ca": ca}

def load_port() -> pd.DataFrame:
    return pd.read_csv(PAPER_PORTFOLIO_FILE) if PAPER_PORTFOLIO_FILE.exists() else pd.DataFrame()

def save_port(df: pd.DataFrame):
    ensure_dir(PAPER_PORTFOLIO_FILE.parent)
    df.to_csv(PAPER_PORTFOLIO_FILE, index=False)

# ── panel builders ────────────────────────────────────────────────────────────

def p_gainers(d: MD) -> Panel:
    t = _tbl([("SYMBOL","left",6),("PRICE","right",6),("CHG%","right",7),("VOL","right",6)])
    for _, r in d.gainers.iterrows():
        t.add_row(Text(r["symbol"], style=f"bold {WHITE}"),
                  f"{r['close']:.1f}", _pct(r["chg"]), Text(_vol(r["volume"]), style=LABEL))
    return _panel(t, f"▲ GAINERS", GAIN)

def p_losers(d: MD) -> Panel:
    t = _tbl([("SYMBOL","left",6),("PRICE","right",6),("CHG%","right",7),("VOL","right",6)])
    for _, r in d.losers.iterrows():
        t.add_row(Text(r["symbol"], style=f"bold {WHITE}"),
                  f"{r['close']:.1f}", _pct(r["chg"]), Text(_vol(r["volume"]), style=LABEL))
    return _panel(t, f"▼ LOSERS", LOSS)

def p_volume(d: MD) -> Panel:
    t = _tbl([("SYMBOL","left",6),("PRICE","right",6),("CHG%","right",7),("VOL","right",7)])
    for _, r in d.vol_top.iterrows():
        t.add_row(Text(r["symbol"], style=f"bold {WHITE}"),
                  f"{r['close']:.1f}", _pct(r["chg"]),
                  Text(_vol(r["volume"]), style=CYAN))
    return _panel(t, "◉ VOLUME", CYAN)

def p_52wk(d: MD) -> Panel:
    t = _tbl([("SYMBOL","left",7),("PRICE","right",7),("REF","right",8),("SIGNAL","left",11)])
    for _, r in d.near_hi.iterrows():
        t.add_row(Text(r["symbol"], style=f"bold {WHITE}"), f"{r['close']:.1f}",
                  f"{r['h']:.1f}", Text("▲ NEAR HIGH", style=f"bold {GAIN_HI}"))
    for _, r in d.near_lo.iterrows():
        t.add_row(Text(r["symbol"], style=f"bold {WHITE}"), f"{r['close']:.1f}",
                  f"{r['l']:.1f}", Text("▼ NEAR LOW", style=f"bold {LOSS_HI}"))
    if d.near_hi.empty and d.near_lo.empty:
        t.add_row(Text("—", style=DIM), Text("None today", style=DIM), "", "")
    return _panel(t, "◆ 52-WEEK EXTREMES", YELLOW)

def p_quotes(d: MD) -> Panel:
    qq = d.quotes[d.quotes["vol"] > 0].nlargest(d.top_n * 3, "vol")
    t  = _tbl([("SYMBOL","left",7),("LTP","right",8),("CHG%","right",8),("VOL","right",9)])
    for _, r in qq.iterrows():
        t.add_row(Text(str(r["symbol"]), style=f"bold {WHITE}"),
                  f"{r['ltp']:.1f}", _pct(r["pc"]), Text(_vol(r["vol"]), style=LABEL))
    ts = d.quotes["ts"].iloc[0][:16] if not d.quotes.empty and "ts" in d.quotes.columns else "N/A"
    return _panel(t, f"◈ LIVE QUOTES  {ts}", PURPLE)

def p_portfolio(d: MD) -> Panel:
    ltps = d.ltps(); port = load_port()
    t = _tbl([("SYMBOL","left",7),("QTY","right",5),("ENTRY","right",7),
              ("LTP","right",7),("P&L","right",13),("RTN%","right",8),
              ("SIGNAL","left",12),("DATE","left",10)])
    tc = tv = 0.0
    if not port.empty:
        for _, r in port.iterrows():
            sym = str(r["Symbol"]); qty = int(r["Quantity"])
            entry = float(r["Buy_Price"])
            cost  = float(r.get("Total_Cost_Basis", entry*qty))
            cur   = ltps.get(sym) or float(r.get("Last_LTP") or entry)
            val   = cur*qty; pnl = val-cost; ret = pnl/cost*100 if cost else 0
            tc += cost; tv += val
            t.add_row(Text(sym, style=f"bold {WHITE}"), str(qty),
                      Text(f"{entry:.1f}", style=LABEL), f"{cur:.1f}",
                      _npr(pnl), _pct(ret),
                      Text(str(r.get("Signal_Type",""))[:12], style=DIM),
                      Text(str(r.get("Entry_Date",""))[:10], style=DIM))
    else:
        t.add_row(Text("No positions", style=DIM), *[""] * 7)
    title = f"◎ PORTFOLIO  —  {len(port)} pos"
    if tc > 0:
        tp = tv - tc; tr_ = tp/tc*100
        title += f"  │  {_npr(tp).markup}  {_pct(tr_, bold=True).markup}"
    return _panel(t, title, BLUE)

def p_signals(d: MD) -> Panel:
    t = _tbl([("#","right",3),("SYMBOL","left",8),("SCORE","right",7),
              ("TYPE","left",16),("STR","right",6),("CONF","right",6),("DIR","left",8)])
    try:
        from backend.backtesting.simple_backtest import load_all_prices, generate_signals_at_date
        conn = _db(); prices_df = load_all_prices(conn); conn.close()
        today = datetime.strptime(d.latest, "%Y-%m-%d")
        sigs = generate_signals_at_date(prices_df, today,
                   signal_types=["volume","quality","low_vol","mean_reversion",
                                 "xsec_momentum","accumulation"])
        for i, s in enumerate(sorted(sigs, key=lambda x: x.strength, reverse=True)[:40], 1):
            t.add_row(Text(str(i), style=DIM),
                      Text(s.symbol, style=f"bold {WHITE}"),
                      Text(f"{s.strength:.3f}", style=YELLOW),
                      Text(s.signal_type.value, style=CYAN),
                      f"{s.strength:.2f}", f"{s.confidence:.2f}",
                      Text("▲ LONG", style=f"bold {GAIN_HI}") if s.direction > 0
                      else Text("—", style=DIM))
        if not sigs:
            t.add_row("—", Text("No signals this session", style=DIM), *[""] * 5)
    except Exception as e:
        t.add_row("!", Text(str(e)[:80], style=LOSS), *[""] * 5)
    return _panel(t, f"◉ SIGNALS  —  {d.latest}", GAIN)

def p_calendar(d: MD) -> Panel:
    t = _tbl([("SYMBOL","left",8),("BOOK CLOSE","left",12),("DAYS","right",5),
              ("CASH%","right",7),("BONUS%","right",7),("BUY BY","left",12)])
    now = datetime.now()
    if d.corp.empty:
        t.add_row(Text("—", style=DIM), Text("No upcoming events", style=DIM), *[""] * 4)
    else:
        for _, r in d.corp.iterrows():
            bc = r["bookclose_date"]; days = (bc-now).days
            cash = float(r.get("cash_dividend_pct") or 0)
            bonus = float(r.get("bonus_share_pct") or 0)
            buy_by = (bc-timedelta(days=5)).strftime("%Y-%m-%d")
            uc = f"bold {YELLOW}" if days <= 7 else (YELLOW if days <= 14 else WHITE)
            t.add_row(Text(str(r["symbol"]), style=f"bold {WHITE}"),
                      bc.strftime("%Y-%m-%d"), Text(f"{days}d", style=uc),
                      Text(f"{cash:.1f}%", style=f"bold {GAIN_HI}") if cash >= 5 else Text("—", style=DIM),
                      Text(f"{bonus:.1f}%", style=f"bold {GAIN_HI}") if bonus >= 10 else Text("—", style=DIM),
                      Text(buy_by, style=CYAN))
    return _panel(t, "◆ CORPORATE ACTIONS  —  Next 30 Days", YELLOW)

def p_lookup(d: MD, sym: str) -> Panel:
    if not sym:
        return _panel(
            Text("\n  Press  L  to look up any stock.\n\n"
                 "  Shows 30 sessions OHLCV + corporate actions.\n", style=LABEL),
            "◎ STOCK LOOKUP", CYAN)
    det = d.detail(sym)
    if not det:
        return _panel(Text(f"\n  Symbol '{sym}' not found.\n", style=LOSS), "◎ LOOKUP", CYAN)
    lat = det["lat"]; chg = det["chg"]
    ltp = d.ltps().get(sym, float(lat["close"]))
    hdr_text = Text.assemble(
        (f"  {sym}  ", f"bold {CYAN}"),
        ("LTP ", LABEL), (f"{ltp:.1f}", f"bold {WHITE}"),
        ("  Close ", LABEL), (f"{lat['close']:.1f}", WHITE),
        ("  ", ""), _pct(chg, bold=True),
        (f"  O {lat['open']:.1f}  H {lat['high']:.1f}  L {lat['low']:.1f}  "
         f"Vol {_vol(lat['volume'])}", LABEL))
    t = _tbl([("DATE","left",11),("OPEN","right",7),("HIGH","right",7),
              ("LOW","right",7),("CLOSE","right",7),("VOL","right",9),("CHG%","right",8)])
    pc: Optional[float] = None
    for _, r in det["h"].iterrows():
        ct = _pct((r["close"]-pc)/pc*100) if pc else Text("—", style=DIM)
        t.add_row(Text(str(r["date"])[:10], style=LABEL),
                  f"{r['open']:.1f}", f"{r['high']:.1f}",
                  f"{r['low']:.1f}", f"{r['close']:.1f}",
                  Text(_vol(r["volume"]), style=LABEL), ct)
        pc = r["close"]
    ca = det["ca"]
    parts: list = [hdr_text, t]
    if not ca.empty:
        cat = _tbl([("BOOK CLOSE","left",12),("CASH DIV%","right",10),("BONUS%","right",8)])
        for _, r in ca.iterrows():
            cat.add_row(
                str(r["bookclose_date"])[:10],
                Text(f"{r['cash_dividend_pct']:.1f}%", style=GAIN_HI)
                    if pd.notna(r.get("cash_dividend_pct")) and r["cash_dividend_pct"] else Text("—", style=DIM),
                Text(f"{r['bonus_share_pct']:.1f}%", style=GAIN_HI)
                    if pd.notna(r.get("bonus_share_pct")) and r["bonus_share_pct"] else Text("—", style=DIM))
        parts.append(Text(f"\n  [{YELLOW}]{sym} — Corporate Actions[/{YELLOW}]"))
        parts.append(cat)
    return _panel(Group(*parts), f"◎ {sym}  —  Last 30 Sessions", CYAN)

# ── header bars ───────────────────────────────────────────────────────────────

TABS = {1: "MARKET", 2: "PORTFOLIO", 3: "SIGNALS", 4: "CALENDAR", 5: "LOOKUP", 6: "TMS"}

def _tab_bar(active: int) -> Text:
    t = Text(" ")
    for n, name in TABS.items():
        if n == active:
            t.append(f" [{n}]{name} ", style=f"bold reverse {AMBER}")
        else:
            t.append(f"  {n} {name}  ", style=LABEL)
    t.append("   │  B=Buy  S=Sell  L=Lookup  R=Refresh  Q=Quit", style=DIM)
    return t


def p_tms(d: MD) -> Panel:
    """TMS live panel — not available in public release."""
    return _panel(Text("\n  Live brokerage not available in this release.\n", style=DIM), "◎ TMS LIVE", CYAN)

def _idx_bar(d: MD) -> Text:
    if len(d.nepse) >= 2:
        ni = d.nepse.iloc[0]["close"]; np_ = d.nepse.iloc[1]["close"]
        chg = (ni - np_) / np_ * 100
        idx = Text.assemble(("NEPSE ", f"bold {LABEL}"),
                             (f"{ni:,.1f}", f"bold {WHITE}"), ("  ", ""), _pct(chg, True))
    else:
        idx = Text("NEPSE N/A", style=DIM)
    ts = d.quotes["ts"].iloc[0][:16] if not d.quotes.empty and "ts" in d.quotes.columns else ""
    return Text.assemble(
        ("  ", ""), idx, ("   │   ", DIM),
        (f"▲{d.adv}", f"bold {GAIN_HI}"), ("  ", ""),
        (f"▼{d.dec}", f"bold {LOSS_HI}"), (f"  ={d.unch}", DIM),
        ("   │   ", DIM), (f"session {d.latest}", LABEL),
        (f"   quotes {ts}" if ts else "", DIM),
        ("   │   ", DIM), (d.ts.strftime("%H:%M:%S"), LABEL))

def _status(msg: str) -> Text:
    return Text(f"  {msg}", style=LABEL)

# ── frame renderer ────────────────────────────────────────────────────────────

def _render_frame(tw: int, d: MD, tab: int, lk: str, msg: str) -> str:
    """Build full frame as string. Horizontal grids use manual line stitching."""
    parts: list[str] = []

    # Header (tab bar + index bar)
    parts.append(_rich_to_str(Group(_tab_bar(tab), _idx_bar(d)), tw))

    # Tab content
    if tab == 1:
        parts.append(_hgrid([p_gainers(d), p_losers(d), p_volume(d)], tw))
        parts.append(_hgrid([p_52wk(d), p_quotes(d)], tw))
    elif tab == 2:
        parts.append(_rich_to_str(p_portfolio(d), tw))
    elif tab == 3:
        parts.append(_rich_to_str(p_signals(d), tw))
    elif tab == 4:
        parts.append(_rich_to_str(p_calendar(d), tw))
    elif tab == 5:
        parts.append(_rich_to_str(p_lookup(d, lk), tw))
    elif tab == 6:
        parts.append(_rich_to_str(p_tms(d), tw))

    # Status bar
    parts.append(_rich_to_str(_status(msg), tw))

    return '\n'.join(parts)

# ── keyboard ──────────────────────────────────────────────────────────────────

def _read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
    except Exception:
        ch = ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

def _prompt_input(label: str) -> str:
    sys.stdout.write(f"  \033[38;5;214m{label}:\033[0m ")
    sys.stdout.flush()
    try:
        return input().strip()
    except (EOFError, KeyboardInterrupt):
        return ""

# ── buy / sell ────────────────────────────────────────────────────────────────

def _tms_dry_submit(side: str, sym: str, qty: int, price: float) -> str:
    """TMS order routing not available in public release."""
    return ""


def exec_buy(sym: str, sh: str, pr: str) -> str:
    try: shares = int(sh)
    except ValueError: return f"Invalid shares: {sh}"
    try:
        c = _db()
        r = pd.read_sql_query(
            "SELECT close FROM stock_prices WHERE symbol=? ORDER BY date DESC LIMIT 1",
            c, params=(sym,))
        c.close()
        if r.empty: return f"Symbol {sym} not found"
        price = float(pr) if pr else float(r.iloc[0]["close"])
    except Exception as e:
        return f"Price lookup: {e}"
    amt = price * shares; fees = round(amt * 0.004, 4); cost = round(amt + fees, 4)
    tms_tail = _tms_dry_submit("BUY", sym, shares, price)
    port = load_port()
    port = pd.concat([port, pd.DataFrame([{
        "Entry_Date": datetime.now().strftime("%Y-%m-%d"),
        "Symbol": sym, "Quantity": shares, "Buy_Price": price,
        "Buy_Amount": round(amt, 4), "Buy_Fees": fees, "Total_Cost_Basis": cost,
        "Signal_Type": "manual", "High_Watermark": price, "Last_LTP": price,
        "Last_LTP_Source": "manual",
        "Last_LTP_Time_UTC": datetime.now(timezone.utc).isoformat(),
    }])], ignore_index=True)
    save_port(port)
    return f"BUY  {shares}x{sym} @ {price:.1f}  cost {cost:,.0f} NPR  fees {fees:.0f}{tms_tail}"

def exec_sell(sym: str, sh: str, pr: str) -> str:
    port = load_port()
    if port.empty: return "Portfolio empty"
    mask = port["Symbol"] == sym
    if not mask.any(): return f"{sym} not in portfolio"
    total = int(port[mask]["Quantity"].sum())
    try: sell = total if sh.lower() == "all" else int(sh)
    except ValueError: return "Invalid qty"
    if sell <= 0 or sell > total: return f"Invalid qty — holding {total}"
    try:
        price = float(pr) if pr else None
        if not price:
            c = _db()
            r = pd.read_sql_query(
                "SELECT close FROM stock_prices WHERE symbol=? ORDER BY date DESC LIMIT 1",
                c, params=(sym,)); c.close()
            price = float(r.iloc[0]["close"]) if not r.empty else float(port[mask].iloc[0]["Buy_Price"])
    except Exception:
        price = float(port[mask].iloc[0]["Buy_Price"])
    if sell == total:
        port = port[~mask]
    else:
        port.at[port[mask].index[0], "Quantity"] = total - sell
    save_port(port)
    tms_tail = _tms_dry_submit("SELL", sym, sell, price)
    return f"SELL  {sell}x{sym} @ {price:.1f}  proceeds {price*sell:,.0f} NPR{tms_tail}"

# ── screen management ─────────────────────────────────────────────────────────

def _enter_alt():
    sys.stdout.write("\033[?1049h\033[?25l")
    sys.stdout.flush()

def _exit_alt():
    sys.stdout.write("\033[?25h\033[?1049l")
    sys.stdout.flush()

def _paint(frame: str):
    sys.stdout.write("\033[H\033[2J")
    sys.stdout.write(frame)
    sys.stdout.flush()

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top",     type=int, default=5)
    ap.add_argument("--refresh", type=int, default=30)
    ap.add_argument("--no-tms",  action="store_true",
                    help="Disable TMS live feed even if a session is available")
    args = ap.parse_args()

    # Opt-in TMS live feed: quietly skipped if no session or disabled.
    tms_src = None
    if not args.no_tms:
        try:
            if tms_src is not None:
                tms_src.start_background()
        except Exception:
            tms_src = None

    tab = 1; lk = ""
    tms_tag = "  │  TMS LIVE" if tms_src is not None else ""
    msg = f"Session: {d.latest}  │  ▲{d.adv} ▼{d.dec}  │  Auto-refresh {args.refresh}s{tms_tag}"

    stop = threading.Event()
    keys: List[str] = []; klock = threading.Lock()

    def _kt():
        while not stop.is_set():
            try:
                k = _read_key()
                if k:
                    with klock: keys.append(k)
            except Exception:
                break

    threading.Thread(target=_kt, daemon=True).start()
    last_ref = time.time()

    _enter_alt()
    try:
        while True:
            if time.time() - last_ref >= args.refresh:
                d.refresh(); last_ref = time.time()
                msg = f"Refreshed {d.ts.strftime('%H:%M:%S')}  │  ▲{d.adv} ▼{d.dec}"

            tw = os.get_terminal_size().columns
            frame = _render_frame(tw, d, tab, lk, msg)
            _paint(frame)

            with klock:
                ks = keys[:]; keys.clear()

            for k in ks:
                kl = k.lower()
                if kl in ("q", "\x03"):
                    stop.set(); return
                elif kl in "123456":
                    tab = int(kl); msg = f"Tab: {TABS[tab]}"
                elif kl == "r":
                    d.refresh(); last_ref = time.time()
                    msg = f"Refreshed {d.ts.strftime('%H:%M:%S')}  │  ▲{d.adv} ▼{d.dec}"
                elif kl == "b":
                    _exit_alt()
                    print(f"\n  \033[38;5;214m━━━  BUY ORDER  ━━━\033[0m")
                    sym = _prompt_input("Symbol").upper()
                    if sym:
                        sh = _prompt_input("Shares")
                        pr = _prompt_input("Price (blank=last close)")
                        res = exec_buy(sym, sh, pr)
                        print(f"  \033[1m{res}\033[0m"); d.refresh(); tab = 2
                    print(f"  \033[38;5;244mpress any key\033[0m")
                    _read_key()
                    _enter_alt()
                elif kl == "s":
                    _exit_alt()
                    print(f"\n  \033[38;5;214m━━━  SELL ORDER  ━━━\033[0m")
                    sym = _prompt_input("Symbol").upper()
                    if sym:
                        sh = _prompt_input("Shares (or 'all')")
                        pr = _prompt_input("Price (blank=last close)")
                        res = exec_sell(sym, sh, pr)
                        print(f"  \033[1m{res}\033[0m"); d.refresh(); tab = 2
                    print(f"  \033[38;5;244mpress any key\033[0m")
                    _read_key()
                    _enter_alt()
                elif kl == "l":
                    _exit_alt()
                    print(f"\n  \033[38;5;214m━━━  LOOKUP  ━━━\033[0m")
                    sym = _prompt_input("Symbol").upper()
                    if sym: lk = sym; tab = 5
                    print(f"  \033[38;5;244mpress any key\033[0m")
                    _read_key()
                    _enter_alt()

            time.sleep(0.15)
    finally:
        _exit_alt()
        if tms_src is not None:
            try:
                tms_src.stop()
            except Exception:
                pass

if __name__ == "__main__":
    main()
