"""The Lookup tab built from nepsequant.com's lookup page (``/api/cli/lookup``).

``build_payload()`` turns the website's reply into the same payload the tab
already renders (title, header, chart, OHLCV table, panels), plus the related
stocks table. Pure: no I/O, so it is easy to test.

Layout (linked to nepsequant.com):
  top     — the volume chart (no candlesticks)
  left    — daily / weekly / monthly OHLCV
  middle  — key facts, related stocks (Enter opens one), broker flow
  right   — quarterly results, dividends & bonus, news & filings
For the NEPSE index: index facts, the day's movers as "related", most active.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from rich.text import Text

from apps.tui.render.volume_chart import fmt_volume, render_volume_chart
from apps.tui.theme import AMBER, CYAN, GAIN, GAIN_HI, LABEL, LOSS, LOSS_HI, WHITE, YELLOW

TF_LABEL = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly", "I": "Daily"}
TF_COL = {"D": "DATE", "W": "WEEK", "M": "MONTH", "Y": "YEAR", "I": "DATE"}
TF_RULE = {"W": "W-FRI", "M": "ME", "Y": "YE"}


def _f(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _dim(s: str) -> Text:
    return Text(s, style=LABEL)


def _pct(v: Optional[float], *, plus: bool = True, nd: int = 1) -> Text:
    if v is None:
        return _dim("—")
    style = GAIN_HI if v > 2 else GAIN if v > 0 else LOSS_HI if v < -2 else LOSS if v < 0 else WHITE
    return Text(f"{v:+.{nd}f}%" if plus else f"{v:.{nd}f}%", style=style)


def _num(v: Optional[float], nd: int = 1, style: str = WHITE) -> Text:
    return _dim("—") if v is None else Text(f"{v:,.{nd}f}", style=style)


def rupees(v: Optional[float]) -> str:
    """Rs in arba (1e9) / crore (1e7) / lakh (1e5), the way NEPSE numbers are read."""
    if v is None:
        return "—"
    a = abs(v)
    sign = "-" if v < 0 else ""
    if a >= 1e9:
        return f"{sign}Rs {a / 1e9:,.2f} arba"
    if a >= 1e7:
        return f"{sign}Rs {a / 1e7:,.2f} cr"
    if a >= 1e5:
        return f"{sign}Rs {a / 1e5:,.2f} lakh"
    return f"{sign}Rs {a:,.0f}"


def history_frame(data: dict[str, Any]) -> pd.DataFrame:
    hist = data.get("history") or {}
    cols = hist.get("columns") or ["date", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(hist.get("rows") or [], columns=cols)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = TF_RULE.get(tf)
    if not rule or df.empty:
        return df
    g = df.set_index("date").resample(rule)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
    }).dropna(subset=["close"]).reset_index()
    return out


def _sparkline(values: list[float], width: int = 28) -> Text:
    blocks = "▁▂▃▄▅▆▇█"
    vals = [v for v in values if v is not None][-width:]
    if len(vals) < 2:
        return _dim("—")
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    style = GAIN if vals[-1] >= vals[0] else LOSS
    return Text("".join(blocks[int((v - lo) / span * 7)] for v in vals), style=style)


def _facts_panel(data: dict[str, Any], df: pd.DataFrame) -> Text:
    fa = data.get("facts") or {}
    out = Text()
    out.append("  KEY FACTS", style=f"bold {AMBER}")
    if fa.get("fundamentals_as_of"):
        out.append(f"  (results to {fa['fundamentals_as_of']})", style=LABEL)
    out.append("\n")

    def pair(label: str, value: Text, gap: str = "   ") -> None:
        out.append(f"  {label} ", style=LABEL)
        out.append_text(value)
        out.append(gap)

    pair("MKT CAP", Text(rupees(_f(fa.get("market_cap"))), style=WHITE))
    pair("P/E", _num(_f(fa.get("pe")), 1))
    pair("P/B", _num(_f(fa.get("pb")), 2), "\n")
    pair("ROE", _pct(_f(fa.get("roe")), plus=False))
    pair("EPS", _num(_f(fa.get("eps")), 2))
    pair("DIV YIELD", _pct(_f(fa.get("dividend_yield_pct")), plus=False, nd=2), "\n")
    hi, lo = _f(fa.get("high_52w")), _f(fa.get("low_52w"))
    out.append("  52W ", style=LABEL)
    out.append(f"{lo:,.1f} – {hi:,.1f}" if hi and lo else "—", style=WHITE)
    if _f(fa.get("pct_from_52w_high")) is not None:
        out.append("  from high ", style=LABEL)
        out.append_text(_pct(_f(fa.get("pct_from_52w_high"))))
    if _f(fa.get("rsi")) is not None:
        out.append("  RSI ", style=LABEL)
        out.append(f"{float(fa['rsi']):.0f}", style=YELLOW)
    out.append("\n  RETURNS ", style=LABEL)
    for label, key in (("1W", "ret_1w"), ("1M", "ret_1m"), ("3M", "ret_3m"), ("1Y", "ret_1y")):
        out.append(f"{label} ", style=LABEL)
        out.append_text(_pct(_f(fa.get(key))))
        out.append("  ")
    out.append("\n  PRICE  ", style=LABEL)
    out.append_text(_sparkline(df["close"].tolist() if not df.empty else [], width=40))
    if not fa:
        status = (data.get("parts") or {}).get("facts")
        out.append("\n  " + ("Key facts are being prepared on nepsequant.com; try again in a minute."
                             if status == "warming" else "No company facts on nepsequant.com for this symbol."),
                   style=LABEL)
    return out


def _index_panel(sym: str, df: pd.DataFrame, unit: str) -> Text:
    out = Text()
    out.append(f"  {sym} INDEX", style=f"bold {AMBER}")
    if df.empty:
        out.append("\n  No index history.", style=LABEL)
        return out
    first, last = float(df["close"].iloc[0]), float(df["close"].iloc[-1])
    hi, lo = float(df["high"].max()), float(df["low"].min())
    out.append(f"  ({len(df)} sessions, {df['date'].iloc[0]:%Y-%m-%d} → {df['date'].iloc[-1]:%Y-%m-%d})\n", style=LABEL)
    out.append("  CHANGE ", style=LABEL)
    out.append_text(_pct((last / first - 1) * 100 if first else None))
    out.append("   HIGH ", style=LABEL)
    out.append(f"{hi:,.1f}", style=GAIN_HI)
    out.append("   LOW ", style=LABEL)
    out.append(f"{lo:,.1f}\n", style=LOSS_HI)
    out.append("  OFF HIGH ", style=LABEL)
    out.append_text(_pct((last / hi - 1) * 100 if hi else None))
    out.append("   AVG TURNOVER ", style=LABEL)
    out.append(fmt_volume(float(df["volume"].mean()), unit) + "\n", style=CYAN)
    out.append("  PRICE  ", style=LABEL)
    out.append_text(_sparkline(df["close"].tolist(), width=40))
    return out


def _related_security(data: dict[str, Any]) -> tuple[str, list[tuple[str, str]], list[list[Text]], list[str]]:
    rel = data.get("related") or {}
    rows = rel.get("rows") or []
    sector = rel.get("sector") or "—"
    size = rel.get("sector_size") or len(rows)
    title = f"RELATED — {sector} · {size} in sector · Enter opens"
    cols = [("SYMBOL", "sym"), ("LTP", "ltp"), ("CHG%", "chg"), ("P/E", "pe"), ("P/B", "pb"),
            ("ROE", "roe"), ("MKT CAP", "mcap")]
    out, syms = [], []
    for r in rows:
        subject = bool(r.get("is_subject"))
        out.append([
            Text(("▸ " if subject else "  ") + str(r.get("symbol")), style=f"bold {AMBER}" if subject else CYAN),
            _num(_f(r.get("ltp")), 1),
            _pct(_f(r.get("change_pct")), nd=2),
            _num(_f(r.get("pe")), 1),
            _num(_f(r.get("pb")), 2),
            _pct(_f(r.get("roe")), plus=False),
            Text(rupees(_f(r.get("market_cap"))).replace("Rs ", ""), style=LABEL),
        ])
        syms.append(str(r.get("symbol")))
    if not rows:
        out.append([_dim("  —"), _dim("No related stocks found"), *[Text("")] * 5])
        syms.append("")
    return title, cols, out, syms


def _related_index(data: dict[str, Any]) -> tuple[str, list[tuple[str, str]], list[list[Text]], list[str]]:
    rel = data.get("related") or {}
    title = "WHAT MOVED THE MARKET — gainers ▲ losers ▼ · Enter opens"
    cols = [("SYMBOL", "sym"), ("LTP", "ltp"), ("CHG%", "chg"), ("VOLUME", "vol"), ("SECTOR", "sector")]
    out, syms = [], []
    for arrow, key, style in (("▲", "gainers", GAIN_HI), ("▼", "losers", LOSS_HI)):
        for r in (rel.get(key) or [])[:6]:
            out.append([
                Text(f"{arrow} {r.get('symbol')}", style=style),
                _num(_f(r.get("ltp")), 1),
                _pct(_f(r.get("change_pct")), nd=2),
                Text(fmt_volume(_f(r.get("volume")) or 0), style=LABEL),
                Text(str(r.get("sector") or "")[:22], style=LABEL),
            ])
            syms.append(str(r.get("symbol")))
    if not out:
        out.append([_dim("  —"), _dim("No movers yet"), *[Text("")] * 3])
        syms.append("")
    return title, cols, out, syms


def _brokers_panel(data: dict[str, Any]) -> Text:
    b = data.get("brokers")
    out = Text()
    if not b:
        out.append("  BROKERS", style=f"bold {AMBER}")
        out.append("\n  No broker flow on nepsequant.com for this stock yet.", style=LABEL)
        return out
    out.append(f"  BROKERS — last {b.get('sessions')} sessions", style=f"bold {AMBER}")
    if b.get("from") and b.get("to"):
        out.append(f"  ({b['from']} → {b['to']})", style=LABEL)
    if _f(b.get("top5_buy_share_pct")) is not None:
        out.append(f"\n  Top 5 buyers took {float(b['top5_buy_share_pct']):.0f}% of the buying"
                   f" across {b.get('brokers') or '?'} brokers", style=LABEL)
    for label, key, style in (("NET BUYERS", "buyers", GAIN_HI), ("NET SELLERS", "sellers", LOSS_HI)):
        out.append(f"\n  {label}\n", style=f"bold {style}")
        rows = b.get(key) or []
        if not rows:
            out.append("   —\n", style=LABEL)
        for r in rows:
            name = str(r.get("name") or f"#{r.get('broker')}")[:26]
            out.append(f"   {str(r.get('broker') or ''):>3} {name:<26} ", style=WHITE)
            qty = _f(r.get("net_qty")) or 0
            out.append(f"{qty:+,.0f} sh  ", style=style)
            out.append(rupees(_f(r.get("net_amount"))) + "\n", style=LABEL)
    return out


def _news_panel(data: dict[str, Any], width: int = 110) -> Text:
    out = Text()
    items = []
    for n in data.get("news") or []:
        items.append((str(n.get("published_at") or "")[:10], str(n.get("source") or "news"), str(n.get("title") or ""), CYAN))
    for f in data.get("filings") or []:
        items.append((str(f.get("published_at") or "")[:10], "FILING " + str(f.get("type") or "").upper(),
                      str(f.get("title") or ""), YELLOW))
    items.sort(key=lambda x: x[0], reverse=True)
    if not items:
        out.append("  No recent news or exchange filings on nepsequant.com.", style=LABEL)
        return out
    for day, src, title, style in items[:10]:
        out.append(f"  {day[5:]}  ", style=LABEL)
        out.append(f"{src[:24]}\n", style=style)
        text = " ".join(title.split())
        out.append("    " + (text if len(text) <= width else text[: width - 1] + "…") + "\n", style=WHITE)
    return out


def build_payload(sym: str, tf: str, data: dict[str, Any], *, width: int, height: int) -> dict[str, Any]:
    """The Lookup tab's payload from the website's lookup page."""
    tf = tf if tf in TF_LABEL else "D"
    index = data.get("kind") == "index"
    hist_unit = str((data.get("history") or {}).get("volume_unit") or ("rupees" if index else "shares"))
    daily = history_frame(data)
    df = resample(daily, tf)
    quote = data.get("quote") or {}
    name = quote.get("name") or ""
    sector = quote.get("sector") or ""
    view = f"{TF_LABEL[tf]} ({len(df)} {'sessions' if tf in ('D', 'I') else TF_LABEL[tf].lower().rstrip('ly') + 's'})"
    who = " · ".join(x for x in (name, sector) if x)
    title = (f"LOOKUP: {sym}" + (f" — {who}" if who else "") + f"  —  {view}  —  D/W/M/Y to switch  —  "
             "L to change  —  nepsequant.com")

    last = daily.iloc[-1] if not daily.empty else None
    ltp = _f(quote.get("ltp")) or (float(last["close"]) if last is not None else None)
    chg = _f(quote.get("change_pct"))
    if chg is None and len(daily) >= 2:
        prev = float(daily["close"].iloc[-2])
        chg = (float(daily["close"].iloc[-1]) / prev - 1) * 100 if prev else None
    header = Text()
    header.append(f"  {sym}  ", style=f"bold {CYAN}")
    header.append("LTP ", style=LABEL)
    header.append(f"{ltp:,.1f} " if ltp else "— ", style=f"bold {WHITE}")
    header.append_text(_pct(chg, nd=2))
    if last is not None:
        header.append(f"   O {last['open']:,.1f}  H {last['high']:,.1f}  L {last['low']:,.1f}  "
                      f"{'Turnover' if hist_unit == 'rupees' else 'Vol'} {fmt_volume(float(last['volume'] or 0), hist_unit)}"
                      f"  ({last['date']:%Y-%m-%d})", style=LABEL)

    if df.empty:
        chart = Text("  No price history on nepsequant.com for this symbol.", style=LABEL)
    else:
        chart = render_volume_chart([d.strftime("%Y-%m-%d") for d in df["date"]], df["close"].tolist(),
                                    df["volume"].tolist(), width=width, height=height, unit=hist_unit,
                                    label="sessions" if tf in ("D", "I") else TF_LABEL[tf].lower().replace("ly", "s"))

    ohlcv_cols = [(TF_COL[tf], "dt"), ("OPEN", "o"), ("HIGH", "h"), ("LOW", "l"), ("CLOSE", "c"),
                  ("TURNOVER" if hist_unit == "rupees" else "VOL", "vol"), ("CHG%", "chg")]
    ohlcv_rows: list[list[Text]] = []
    closes = df["close"].tolist()
    for i in range(len(df) - 1, -1, -1):  # newest first
        r = df.iloc[i]
        c = closes[i]
        prev = closes[i - 1] if i else None
        ohlcv_rows.append([
            _dim(f"{r['date']:%Y-%m-%d}"),
            _num(_f(r["open"])), _num(_f(r["high"])), _num(_f(r["low"])), _num(_f(c), style=f"bold {WHITE}"),
            Text(fmt_volume(float(r["volume"] or 0), hist_unit), style=LABEL),
            _pct((c / prev - 1) * 100 if prev else None, nd=2),
        ])

    if index:
        stats = _index_panel(sym, daily, hist_unit)
        rel_title, rel_cols, rel_rows, rel_syms = _related_index(data)
        report = _news_panel({"news": [n | {"source": f"{r.get('symbol')}"} for r in
                                       (data.get("related") or {}).get("gainers", []) + (data.get("related") or {}).get("losers", [])
                                       for n in (r.get("news") or [])]})
        report = Text("  NEWS BEHIND THE MOVES\n", style=f"bold {AMBER}") + report
        active = (data.get("related") or {}).get("most_active") or []
        fin_title = "MOST ACTIVE TODAY"
        fin_cols = [("SYMBOL", "sym"), ("LTP", "ltp"), ("CHG%", "chg"), ("VOLUME", "vol")]
        fin_rows = [[Text(str(r.get("symbol")), style=CYAN), _num(_f(r.get("ltp"))), _pct(_f(r.get("change_pct")), nd=2),
                     Text(fmt_volume(_f(r.get("volume")) or 0), style=LABEL)] for r in active[:8]] or \
                   [[_dim("—"), _dim("No data yet"), Text(""), Text("")]]
        ca_title, ca_cols, ca_rows = "", [], []
        intel_title, intel = "", Text("")
    else:
        stats = _facts_panel(data, daily)
        rel_title, rel_cols, rel_rows, rel_syms = _related_security(data)
        report = _brokers_panel(data)
        quarters = data.get("quarters") or []
        fin_title = f"QUARTERLY RESULTS — {sym}"
        fin_cols = [("PERIOD", "period"), ("EPS", "eps"), ("NET PROFIT", "np"), ("REPORTED", "rep")]
        fin_rows = [[_dim(f"{q.get('fiscal_year') or '?'} Q{q.get('quarter') or '?'}"),
                     _num(_f(q.get("eps")), 2, GAIN_HI if (_f(q.get("eps")) or 0) >= 0 else LOSS_HI),
                     Text(rupees(_f(q.get("net_profit"))), style=WHITE),
                     _dim(str(q.get("reported") or "—"))] for q in quarters] or \
                   [[_dim("—"), _dim("No quarterly results on nepsequant.com"), Text(""), Text("")]]
        actions = data.get("corporate_actions") or []
        ca_title = f"DIVIDENDS & BONUS — {sym}" + ("" if actions else " — none recorded")
        ca_cols = [("FY", "fy"), ("BOOK CLOSE", "bc"), ("CASH", "cash"), ("BONUS", "bonus"), ("RIGHTS", "rights")]
        ca_rows = [[_dim(str(a.get("fiscal_year") or "—")), Text(str(a.get("book_close") or "—"), style=WHITE),
                    Text(f"{a['cash_dividend_pct']:.2f}%", style=GAIN_HI) if a.get("cash_dividend_pct") else _dim("—"),
                    Text(f"{a['bonus_pct']:.2f}%", style=GAIN_HI) if a.get("bonus_pct") else _dim("—"),
                    Text(str(a.get("rights") or "—"), style=WHITE if a.get("rights") else LABEL)] for a in actions]
        intel_title = f"NEWS & FILINGS — {sym}"
        intel = _news_panel(data)

    return {
        "found": True,
        "source": "nepsequant.com",
        "title": title,
        "header": header,
        "chart": chart,
        "stats": stats,
        "report": report,
        "related_title": rel_title,
        "related_cols": rel_cols,
        "related_rows": rel_rows,
        "related_syms": rel_syms,
        "fin_title": fin_title,
        "fin_cols": fin_cols,
        "fin_rows": fin_rows,
        "intel_title": intel_title,
        "intel": intel,
        "ohlcv_title": f"OHLCV — {sym} ({TF_LABEL[tf]}, newest first)",
        "ohlcv_cols": ohlcv_cols,
        "ohlcv_rows": ohlcv_rows,
        "ca_title": ca_title,
        "ca_cols": ca_cols,
        "ca_rows": ca_rows,
    }
