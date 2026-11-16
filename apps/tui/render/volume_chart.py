"""A volume chart for the Lookup tab (in place of candlesticks).

One bar per session, green on a day the price closed up, red on a day it
closed down, grey when unchanged; a dotted line marks the average volume. A
volume axis on the left, the first / middle / last dates underneath. Pure text
(Rich), so it works in any terminal, including Apple Terminal's 256 colours.
"""

from __future__ import annotations

from typing import Sequence

from rich.text import Text

from apps.tui.theme import AMBER, GAIN, LABEL, LOSS, WHITE

BLOCKS = " ▁▂▃▄▅▆▇█"
FLAT = "#6f7b88"
AVG_LINE = "#5c5c5c"
AXIS_W = 14  # "Rs 41.6 arba ┤"


def fmt_volume(v: float, unit: str = "shares") -> str:
    """Shares as 7.84M / 812K; rupee turnover the NEPSE way (arba, crore, lakh)."""
    v = float(v or 0.0)
    steps = ((1e9, " arba"), (1e7, " cr"), (1e5, " lakh")) if unit == "rupees" else ((1e9, "B"), (1e6, "M"), (1e3, "K"))
    prefix = "Rs " if unit == "rupees" else ""
    for div, suffix in steps:
        if abs(v) >= div:
            return f"{prefix}{v / div:.2f}{suffix}" if abs(v) / div < 10 else f"{prefix}{v / div:.1f}{suffix}"
    return f"{prefix}{v:,.0f}"


def render_volume_chart(
    dates: Sequence[str],
    closes: Sequence[float],
    volumes: Sequence[float],
    *,
    width: int,
    height: int,
    unit: str = "shares",
    label: str = "sessions",
) -> Text:
    """The chart as Rich text, ``height`` lines tall and at most ``width`` wide."""
    n_all = min(len(dates), len(closes), len(volumes))
    out = Text()
    if n_all == 0:
        out.append("  No volume history yet.", style=LABEL)
        return out
    rows = max(3, int(height) - 2)  # one line for the header, one for the dates
    room = max(10, int(width) - AXIS_W - 2)
    # bars as wide as the room allows (1-3 columns), newest sessions kept
    bar_w = 3 if room >= n_all * 3 else 2 if room >= n_all * 2 else 1
    n = min(n_all, room // bar_w)
    dates, closes, volumes = list(dates)[-n:], list(closes)[-n:], [max(0.0, float(v or 0.0)) for v in list(volumes)[-n:]]
    prev_close = list(closes)[0]
    vmax = max(volumes) or 1.0
    avg = sum(volumes) / len(volumes)
    last = volumes[-1]

    out.append("  VOLUME ", style=f"bold {AMBER}")
    out.append(f"{n} {label} · ", style=LABEL)
    out.append("avg ", style=LABEL)
    out.append(fmt_volume(avg, unit), style=WHITE)
    out.append(" · last ", style=LABEL)
    out.append(fmt_volume(last, unit), style=f"bold {WHITE}")
    if avg > 0:
        ratio = last / avg
        out.append(f" ({ratio:.1f}× avg)", style=GAIN if ratio >= 1.5 else LABEL)
    out.append("   ▮", style=GAIN)
    out.append(" up day  ", style=LABEL)
    out.append("▮", style=LOSS)
    out.append(" down day  ", style=LABEL)
    out.append("┈", style=AVG_LINE)
    out.append(" average\n", style=LABEL)

    colors = []
    for i, c in enumerate(closes):
        before = closes[i - 1] if i else prev_close
        colors.append(FLAT if i == 0 or c == before else (GAIN if c > before else LOSS))
    eighths = [round(v / vmax * rows * 8) for v in volumes]
    avg_level = avg / vmax * rows  # in rows from the bottom
    avg_row = min(rows - 1, max(0, int(avg_level)))  # the row the dotted line sits in

    for r in range(rows - 1, -1, -1):  # top row first
        if r == rows - 1:
            axis = fmt_volume(vmax, unit)
        elif r == rows // 2:
            axis = fmt_volume(vmax * (r + 1) / rows, unit)
        else:
            axis = ""
        out.append(f"{axis:>{AXIS_W - 1}} ", style=LABEL)
        out.append("┤" if axis else "│", style=AVG_LINE)
        for i, e in enumerate(eighths):
            fill = e - r * 8
            if fill >= 8:
                ch = BLOCKS[8]
            elif fill > 0:
                ch = BLOCKS[fill]
            else:
                ch = ""
            style = f"bold {colors[i]}" if i == n - 1 else colors[i]
            if ch:
                out.append(ch * (bar_w if bar_w == 1 else bar_w - 1), style=style)
            else:
                out.append(("┈" if r == avg_row else " ") * (bar_w if bar_w == 1 else bar_w - 1),
                           style=AVG_LINE)
            if bar_w > 1:
                out.append("┈" if (r == avg_row and not ch) else " ", style=AVG_LINE)
        out.append("\n")

    # dates: first, middle, last under the bars
    span = n * bar_w
    line = [" "] * span
    marks = [(0, str(dates[0])[:10]), (span // 2 - 5, str(dates[n // 2])[:10]), (span - 10, str(dates[-1])[:10])]
    for pos, text in marks:
        pos = max(0, min(span - len(text), pos))
        if all(ch == " " for ch in line[max(0, pos - 1): pos + len(text) + 1]):
            line[pos: pos + len(text)] = list(text)
    out.append(" " * (AXIS_W + 1) + "".join(line).rstrip(), style=LABEL)
    return out
