"""Persistent chrome: header bar, index/NEPSE bar, status bar, badges.

All methods here are call_from_thread targets. _set_status is the most-targeted
method in the dashboard and stays reachable from every mixin via the MRO.
"""

from datetime import datetime

from rich.markup import escape as _escape_markup
from rich.text import Text
from textual.widgets import Static

from apps.tui.theme import GAIN_HI, LABEL, LOSS_HI, WHITE, YELLOW


class HeaderStatusMixin:
    def _display_live_badge(self) -> bool:
        return False

    def _display_mode_label(self) -> str:
        if self._website_book is not None:
            trader = self._autotrader
            return "PAPER · AUTO-TRADE" if trader is not None and trader.enabled else "PAPER · NEPSEQUANT.COM"
        return "PAPER AUTO"

    def _update_header(self):
        # Lazy import avoids a circular import at module load (dashboard_tui
        # imports this mixin before TAB_NAMES/TAB_LABELS are defined).
        from apps.tui.dashboard_tui import TAB_LABELS, TAB_NAMES
        parts = []
        # Mode indicator
        parts.append(f"[bold #111820 on #79ffb3] {self._display_mode_label()} [/]")
        for name, key in TAB_NAMES.items():
            label = TAB_LABELS.get(name, name.upper())
            if name == self.active_tab:
                parts.append(f"[bold #111820 on #f2b94b] {key}:{label} [/]")
            else:
                parts.append(f"[#7d8b99] {key}:{label} [/]")
        if self.active_tab == "lookup" and self.lookup_sym:
            tf = self.lookup_tf
            tf_parts = []
            for key, label in [("D", "Daily"), ("W", "Weekly"), ("M", "Monthly"), ("Y", "Yearly"), ("I", "Intraday")]:
                if key == tf:
                    tf_parts.append(f"[bold #f2b94b]{key}={label}[/]")
                else:
                    tf_parts.append(f"[#708091]{key}={label}[/]")
            parts.append("  [#3d4f5e]│[/]  " + "  [#3d4f5e]·[/]  ".join(tf_parts) + "  [#3d4f5e]│[/]  [#7a9ab0]l[/][#4a6070] lookup[/]  [#3d4f5e]·[/]  [#7a9ab0]q[/][#4a6070] quit[/]")
        else:
            parts.append("  [#3d4f5e]│[/]  [#7a9ab0]/[/][#4a6070] search[/]  [#3d4f5e]·[/]  [#7a9ab0]+[/][#4a6070] watch[/]  [#3d4f5e]·[/]  [#7a9ab0]r[/][#4a6070] refresh[/]  [#3d4f5e]·[/]  [#7a9ab0]q[/][#4a6070] quit[/]")
        self.query_one("#header-bar", Static).update(Text.from_markup(" ".join(parts)))

    def _update_index(self):
        parts = []
        # NEPSE index — prefer live TMS WS value when available.
        live_nepse = None
        tms_src = getattr(self.md, "tms", None)
        if tms_src is not None:
            try:
                if tms_src.is_live():
                    idx = (self.md.tms_indices or {}).get("NEPSE")
                    if idx is None:
                        idx = tms_src.indices().get("NEPSE")
                    if idx is not None:
                        live_nepse = idx
            except Exception:
                live_nepse = None

        if live_nepse is not None:
            ni = float(getattr(live_nepse, "value", 0.0))
            chg = float(getattr(live_nepse, "pct_change", 0.0))
            sign = "+" if chg >= 0 else ""
            cc = GAIN_HI if chg >= 0 else LOSS_HI
            parts.append(
                f"[bold #8da1b5]NEPSE[/] [bold {WHITE}]{ni:,.2f}[/] "
                f"[{cc}]{sign}{chg:.2f}%[/] [dim #6a8899]live[/]"
            )
        elif len(self.md.nepse) >= 2:
            ni = self.md.nepse.iloc[0]["close"]
            np_ = self.md.nepse.iloc[1]["close"]
            chg = (ni - np_) / np_ * 100
            sign = "+" if chg >= 0 else ""
            cc = GAIN_HI if chg >= 0 else LOSS_HI
            parts.append(f"[bold #8da1b5]NEPSE[/] [bold {WHITE}]{ni:,.1f}[/] [{cc}]{sign}{chg:.2f}%[/]")
        else:
            parts.append("[#8da1b5]NEPSE N/A[/]")
        # Breadth
        parts.append(f"[bold {GAIN_HI}]▲{self.md.adv}[/] [bold {LOSS_HI}]▼{self.md.dec}[/] [#888888]={self.md.unch}[/]")
        # Regime
        regime = getattr(self, '_regime', 'unknown')
        rc = {
            "bull": f"bold {GAIN_HI}", "neutral": f"bold {YELLOW}",
            "bear": f"bold {LOSS_HI}"
        }.get(regime, LABEL)
        parts.append(f"[{rc}]REGIME {regime.upper()}[/]")
        # Session
        parts.append(f"[#8da1b5]SESSION[/] [bold {WHITE}]{self.md.latest}[/]")
        # Quotes timestamp
        ts_q = ""
        if not self.md.quotes.empty and "ts" in self.md.quotes.columns:
            ts_q = self.md.quotes["ts"].iloc[0][:16]
        if ts_q:
            parts.append(f"[#708091]QUOTES {ts_q}[/]")
        parts.append(f"[#8da1b5]LOCAL[/] [bold {WHITE}]{self.md.ts.strftime('%H:%M:%S')}[/]")
        self.query_one("#index-bar", Static).update(
            Text.from_markup("   │   ".join(parts)))

    def _set_status(self, msg: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        compact = " ".join(str(msg).split())
        if len(compact) > 220:
            compact = compact[:217] + "..."
        lowered = compact.lower()
        body_style = "#95a4b5"
        if lowered.startswith("rejected:") or lowered.startswith("order cancelled:") or "same-day rule" in lowered:
            body_style = LOSS_HI
        self.query_one("#status-bar", Static).update(
            Text.from_markup(
                f"[bold #f2b94b]{stamp}[/] [#516273]::[/] [{body_style}]{_escape_markup(compact)}[/]"
            )
        )
