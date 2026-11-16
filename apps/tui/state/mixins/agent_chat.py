"""Agent analysis worker + chat worker + their UI readers.

The @work analysis/chat workers sit beside ALL their call_from_thread readers
(typing animation, chat append, status, picks/verdicts tab), runtime-state
load, env-context sync, and the auto-order super-signal path.

Lazy fallback imports only: any degrade-graceful optional import stays
in-function, never hoisted to this module top, to avoid a partial-init
circular-import trap.
"""

import os
from pathlib import Path
import shlex
import threading
import time

from rich.markup import escape
from rich.text import Text
from textual import work
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Static

from typing import Optional

from apps.tui.io.agent_io import _split_agent_messages_by_cutoff
from apps.tui.io.intraday import _format_nst_hm
from apps.tui.io.persistence import INITIAL_CAPITAL, _account_dir
from apps.tui.render.cells import _dim_text, _sym_text
from apps.tui.theme import (
    AMBER, BLUE, GAIN, GAIN_HI, LABEL, LOSS, LOSS_HI, PURPLE, WHITE, YELLOW,
)
from configs.long_term import LONG_TERM_CONFIG
from backend.trading import strategy_registry
from backend.agents.agent_analyst import (
    _analysis_cache_is_fresh,
    agent_display_label,
    analyze as agent_analyze,
    append_external_agent_chat_message,
    ask as agent_ask,
    build_algo_shortlist_snapshot,
    last_call_info,
    load_agent_analysis,
    load_agent_archive_history,
    load_agent_history,
)
from backend.agents.ai_settings import (
    CLOUD_PROVIDERS,
    ENV_KEYS,
    KEY_IN_CHAT_MESSAGE,
    get_api_key,
    looks_like_api_key,
    update_ai_settings,
)
from backend.agents.llm_anthropic import CHEAP_ANTHROPIC_MODEL
from backend.agents.runtime_config import (
    ACTIVE_AGENT_FILE,
    REMEMBERED_MODEL_PRESETS,
    list_agent_backends,
    load_active_agent_config,
    set_active_agent,
)

AGENT_ARCHIVE_RENDER_LIMIT = 60
# An answer may look things up first (a few tool rounds), so allow more than one model call.
AGENT_CHAT_TIMEOUT_SECS = 150
AGENT_BRIEF_RETRY_SECS = 600  # provider not set up: check again in 10 minutes
AGENT_SWITCH_USAGE = "Usage: /agent <ollama|openai|anthropic|claude|gemma4_mlx> [model]"


def _error_line(text: str) -> str:
    """Provider errors arrive as "ERROR: what happened. What to do." - show the sentence."""
    clean = " ".join(str(text or "").split())
    return clean[len("ERROR:"):].strip() if clean.startswith("ERROR:") else clean


class AgentChatMixin:
    def _load_agent_runtime_state(self) -> None:
        self._agent_analysis = load_agent_analysis() or {}
        current_account_id = str(getattr(self, "_current_account_id", "account_1") or "account_1")
        current_strategy_id = str((self._strategy_account_binding(current_account_id) or {}).get("strategy_id") or strategy_registry.default_strategy_for_account(current_account_id))
        if (
            str((self._agent_analysis or {}).get("account_id") or "") != current_account_id
            or str((self._agent_analysis or {}).get("strategy_id") or "") != current_strategy_id
        ):
            self._agent_analysis = {}
        all_recent = list(load_agent_history() or [])
        visible_cutoff = float(getattr(self, "_agent_visible_since", 0.0) or 0.0)
        self._agent_history, self._agent_hidden_recent_history = _split_agent_messages_by_cutoff(all_recent, visible_cutoff)
        archived_items = list(load_agent_archive_history() or [])
        self._agent_archive_count = len(archived_items) + len(self._agent_hidden_recent_history)
        if self._agent_show_archived:
            older = archived_items + list(self._agent_hidden_recent_history or [])
            self._agent_archived_history = older[-AGENT_ARCHIVE_RENDER_LIMIT:]
        else:
            self._agent_archived_history = []

    def _sync_agent_account_context_env(self) -> None:
        if getattr(self, "_website_strategies", None) is not None:
            self._sync_agent_context_from_website()
            return
        os.environ.pop("NEPSE_AGENT_SIGNALS_SOURCE", None)
        account_id = str(getattr(self, "_current_account_id", "account_1") or "account_1")
        account_dir = _account_dir(account_id)
        strategy_id = str((self._strategy_account_binding(account_id) or {}).get("strategy_id") or strategy_registry.default_strategy_for_account(account_id))
        os.environ["NEPSE_ACTIVE_ACCOUNT_ID"] = account_id
        os.environ["NEPSE_ACTIVE_ACCOUNT_NAME"] = self._active_account_name()
        os.environ["NEPSE_ACTIVE_ACCOUNT_DIR"] = str(account_dir)
        os.environ["NEPSE_ACTIVE_PORTFOLIO_FILE"] = str(account_dir / "paper_portfolio.csv")
        os.environ["NEPSE_ACTIVE_STRATEGY_ID"] = strategy_id
        os.environ["NEPSE_ACTIVE_STRATEGY_NAME"] = strategy_registry.strategy_name(strategy_id)

    def _sync_agent_context_from_website(self) -> None:
        """Linked to nepsequant.com: the agent reviews the shared paper book and
        the assigned website strategy's picks for today (never the local
        signal engine). Uses cached data only, so it is safe on the UI thread;
        the analysis worker fetches fresh picks before calling this."""
        from backend.agents.agent_analyst import _current_nst_session_date, publish_agent_signal_snapshot
        from backend.quant_pro.paths import ensure_dir, get_runtime_dir

        strategies, book = self._website_strategies, self._website_book
        folder = ensure_dir(Path(get_runtime_dir(__file__)) / "website_book")
        portfolio_file = folder / "agent_portfolio.csv"
        try:
            book.portfolio_frame().to_csv(portfolio_file, index=False)
        except Exception:
            pass
        assigned = strategies.assigned_strategy() if strategies is not None else None
        strategy_id = str((assigned or {}).get("id") or "nepsequant:none")
        strategy_name = str((assigned or {}).get("name") or "No strategy assigned")
        os.environ["NEPSE_ACTIVE_ACCOUNT_ID"] = "nepsequant_paper"
        os.environ["NEPSE_ACTIVE_ACCOUNT_NAME"] = "nepsequant.com paper book"
        os.environ["NEPSE_ACTIVE_ACCOUNT_DIR"] = str(folder)
        os.environ["NEPSE_ACTIVE_PORTFOLIO_FILE"] = str(portfolio_file)
        os.environ["NEPSE_ACTIVE_STRATEGY_ID"] = strategy_id
        os.environ["NEPSE_ACTIVE_STRATEGY_NAME"] = strategy_name
        os.environ["NEPSE_AGENT_SIGNALS_SOURCE"] = "website"
        picks = strategies.cached_picks(strategy_id) if assigned else None
        rows = list(picks.rows) if picks else []
        n = max(len(rows), 1)
        signals = [
            {
                "symbol": str(r.get("symbol") or "").upper(),
                "signal_type": f"nepsequant_{assigned.get('kind') or 'strategy'}",
                # rank-based 1.0 → 0: rule-set scores are sort values, not comparable across strategies
                "score": round(1.0 - (int(r.get("rank") or i + 1) - 1) / n, 3),
                "strength": round(1.0 - (int(r.get("rank") or i + 1) - 1) / n, 3),
                "confidence": 0.5,
                "reasoning": f"Rank {r.get('rank') or i + 1} in your nepsequant.com strategy \"{strategy_name}\""
                + (f": {r.get('reason')}" if r.get("reason") else ""),
            }
            for i, r in enumerate(rows)
            if r.get("symbol")
        ]
        try:
            publish_agent_signal_snapshot({
                "account_id": "nepsequant_paper",
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "context_date": _current_nst_session_date(),
                "regime": "unknown",
                "source": "nepsequant.com",
                "picks_as_of": picks.as_of if picks else None,
                "signals": signals,
            })
        except Exception:
            pass

    # ── what the analyst always knows: watchlist, holdings, trades, strategy ──
    def _install_agent_context_provider(self) -> None:
        from backend.agents import user_context

        user_context.set_provider(self._agent_user_context)

    def _agent_user_context(self) -> dict:
        """The user's own context from the terminal's in-memory state (no network).
        Runs in worker threads; reads cached copies only."""
        from backend.agents import user_context

        watch = [
            str(e.get("symbol") or e.get("label") or "").upper()
            for e in list(getattr(self, "_paper_watchlist", None) or [])
            if isinstance(e, dict) and str(e.get("kind") or "stock") == "stock"
        ]
        book = getattr(self, "_website_book", None)
        if book is not None:
            holdings = [
                {"symbol": p.get("symbol"), "qty": p.get("quantity"), "avg_cost": p.get("avg_cost"),
                 "last": p.get("last"), "return_pct": p.get("return_pct")}
                for p in (book.positions() if book.loaded else [])
            ]
            try:
                trades = user_context.trades_from_frame(book.trade_log_frame())
            except Exception:
                trades = []
            strategy = None
            strategies = getattr(self, "_website_strategies", None)
            assigned = strategies.assigned_strategy() if strategies is not None else None
            if assigned:
                picks = strategies.cached_picks(assigned.get("id"))
                strategy = {
                    "name": assigned.get("name") or assigned.get("id"),
                    "kind": assigned.get("kind"),
                    "as_of": picks.as_of if picks else None,
                    "picks": [dict(r) for r in (picks.rows if picks else [])][:8],
                }
            return {
                "linked": True,
                "account": "nepsequant.com paper book",
                "terminal_watchlist": watch,
                "holdings": holdings,
                "trades": trades,
                "strategy": strategy,
            }
        from apps.tui.io.persistence import _load_trade_log
        from backend.agents.agent_analyst import _load_active_portfolio

        rows, meta = _load_active_portfolio()
        try:
            trades = user_context.trades_from_frame(_load_trade_log())
        except Exception:
            trades = []
        return {
            "linked": False,
            "account": str(meta.get("name") or meta.get("id") or "local paper account"),
            "terminal_watchlist": watch,
            "holdings": [{"symbol": r["symbol"], "qty": r["qty"], "avg_cost": r.get("entry")} for r in rows],
            "trades": trades,
            "strategy": None,
        }

    # ── the automatic brief ──────────────────────────────────────────────────
    def _agent_proactive_tick(self) -> None:
        """Every minute: post a brief if one is due for this window (cheap check)."""
        from backend.agents import proactive
        from backend.quant_pro.nepse_calendar import current_nepal_datetime

        if getattr(self, "_agent_brief_running", False) or getattr(self, "_agent_chat_busy", False):
            return
        if time.time() < float(getattr(self, "_agent_brief_skip_until", 0.0) or 0.0):
            return
        try:
            window = proactive.due_window(current_nepal_datetime(), proactive.load_state())
        except Exception:
            return
        if window:
            self._run_agent_brief(window, manual=False)

    @work(thread=True, group="agent-brief")
    def _run_agent_brief(self, window: Optional[str] = None, manual: bool = False) -> None:
        from backend.agents import proactive
        from backend.quant_pro.nepse_calendar import current_nepal_datetime

        if getattr(self, "_agent_brief_running", False):
            if manual:
                self.call_from_thread(self._set_status, "A brief is already being prepared")
            return
        self._agent_brief_running = True
        try:
            ready, why = proactive.provider_ready()
            if not ready:
                if manual:
                    self.call_from_thread(self._append_chat_note, f"A brief needs an AI provider: {why}.")
                else:
                    self._agent_brief_skip_until = time.time() + AGENT_BRIEF_RETRY_SECS
                return
            now = current_nepal_datetime()
            window = window or proactive.current_window(now)
            self._sync_agent_account_context_env()
            label = proactive.WINDOW_LABEL.get(window, window)
            self.call_from_thread(self._set_status, f"⧖ Agent preparing the {label} brief... ({self._agent_header_label()})")
            result = proactive.run_brief(window, now=now)
            provider = self._current_agent_provider_label()
            text = str(result.get("text") or "")
            if result.get("ok"):
                append_external_agent_chat_message("AGENT", text, source="proactive_brief", provider=provider)
                proactive.mark_done(now.date(), window, text=text)
                self.call_from_thread(self._append_chat, "AGENT", text, PURPLE, keep_lines=True)
                self.call_from_thread(self._set_status, f"Brief posted ({label}) │ {self._agent_header_label()} │ /proactive off to stop automatic briefs")
                return
            # failed: say so once and do not retry this window automatically
            proactive.mark_done(now.date(), window)
            if manual:
                append_external_agent_chat_message("AGENT", text, source="proactive_brief", provider=provider)
                self.call_from_thread(self._append_chat, "AGENT", text, LOSS_HI)
            else:
                self.call_from_thread(self._append_chat_note, f"Automatic brief skipped: {_error_line(text)}")
            self.call_from_thread(self._set_status, _error_line(text))
        except Exception as exc:
            self.call_from_thread(self._set_status, f"Brief error: {exc}")
        finally:
            self._agent_brief_running = False

    def _current_agent_provider_label(self) -> str:
        try:
            cfg = load_active_agent_config()
        except Exception:
            cfg = {}
        configured = str(
            cfg.get("provider_label")
            or cfg.get("backend")
            or os.environ.get("NEPSE_AGENT_PROVIDER_LABEL")
            or "gemma4_mlx"
        ).strip()
        if configured:
            return configured
        meta = dict((getattr(self, "_agent_analysis", {}) or {}).get("agent_runtime_meta") or {})
        return str(meta.get("provider") or "gemma4_mlx")

    def _agent_header_label(self) -> str:
        """e.g. "CLAUDE · claude-opus-5-5", "OPENAI · gpt-5-mini", "OLLAMA · llama3.2"."""
        try:
            return agent_display_label()
        except Exception:
            return self._current_agent_provider_label().upper()

    def _agent_key_status(self, backend: str) -> str:
        if backend not in CLOUD_PROVIDERS:
            return ""
        _key, source = get_api_key(backend)
        if source == "env":
            return f"key: from {ENV_KEYS[backend]}"
        if source == "file":
            return "key: saved on this computer"
        return "key: none yet - run ./nqt-connect ai setup in a terminal"

    def _agent_runtime_summary(self) -> str:
        cfg = load_active_agent_config()
        backend = str(cfg.get("backend") or "ollama")
        parts = [self._agent_header_label()]
        key_status = self._agent_key_status(backend)
        if key_status:
            parts.append(key_status)
        if backend == "anthropic":
            parts.append(f"low-cost option: /model {CHEAP_ANTHROPIC_MODEL}")
        return " | ".join(parts)

    def _agent_backends_help(self) -> str:
        parts = [
            f"{row['id']} ({row.get('model') or 'no default model'})"
            for row in list_agent_backends()
            if row["id"] != "gemma4_experimental"
        ]
        return (
            "Agent backends: " + ", ".join(parts)
            + f". Switch: /agent <name> [model]. Cloud models (openai, anthropic) bill your own API key; "
            + f"low-cost Claude: /agent anthropic {CHEAP_ANTHROPIC_MODEL}. Keys: ./nqt-connect ai setup (never in chat)."
        )

    def _agent_ui(self, fn, *args) -> None:
        """Run ``fn`` on the UI thread whether we are on it already or in a worker."""
        if getattr(self, "_thread_id", None) == threading.get_ident():
            fn(*args)
        else:
            self.call_from_thread(fn, *args)

    def _refuse_secret_in_composer(self, text: str) -> bool:
        """Never send or store something that looks like an API key (chat history is on disk)."""
        if not looks_like_api_key(text):
            return False
        self._append_chat_note(KEY_IN_CHAT_MESSAGE)
        self._set_status("Not sent: that looked like an API key. Use ./nqt-connect ai setup")
        return True

    def _stop_active_agent_chat(self, *, announce: bool = True) -> bool:
        proc = getattr(self, "_agent_chat_process", None)
        if proc is None and not bool(getattr(self, "_agent_chat_busy", False)):
            return False
        # The answer still being worked on is dropped when it arrives.
        self._agent_chat_stop_requested = True
        self._agent_chat_busy = False
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            self._agent_chat_process = None
        self._agent_ui(self._set_typing, False)
        if announce:
            provider = self._current_agent_provider_label()
            append_external_agent_chat_message("AGENT", "Stopped by user.", source="tui_chat", provider=provider)
            self._agent_ui(self._append_chat, "AGENT", "Stopped by user.", PURPLE)
            self._agent_ui(self._set_status, "Agent response stopped")
        return True

    def _update_agent_chat_hint(self) -> None:
        hint = self.query_one("#agent-chat-hint", Static)
        if getattr(self, "_agent_show_archived", False):
            text = "Archive view loaded. Type /recent to collapse."
        else:
            text = ""
        hint.styles.height = 1 if text else 0
        hint.update(Text(text, style=LABEL))

    def _get_agent_typing_widget(self) -> Static:
        widget = getattr(self, "_agent_typing_widget", None)
        if widget is None:
            widget = Static("", classes="chat-msg chat-note", id="agent-typing-line")
            self._agent_typing_widget = widget
        return widget

    def _detach_agent_typing_widget(self) -> None:
        widget = getattr(self, "_agent_typing_widget", None)
        if widget is not None and widget.parent is not None:
            widget.remove()

    def _ensure_agent_typing_widget(self) -> Static:
        scroll = self.query_one("#agent-chat-scroll", VerticalScroll)
        widget = self._get_agent_typing_widget()
        if widget.parent is None:
            scroll.mount(widget)
        elif widget.parent is not scroll:
            widget.remove()
            scroll.mount(widget)
        return widget

    def _append_chat_note(self, message: str) -> None:
        scroll = self.query_one("#agent-chat-scroll", VerticalScroll)
        note_text = Text.assemble(
            ("[SYSTEM]: ", f"bold {LABEL}"),
            (str(message or "").strip(), LABEL),
        )
        widget = Static(note_text, classes="chat-msg chat-note")
        scroll.mount(widget)
        scroll.scroll_end(animate=False)

    def _render_agent_chat_history(self) -> None:
        scroll = self.query_one("#agent-chat-scroll", VerticalScroll)
        scroll.remove_children()
        archived = list(getattr(self, "_agent_archived_history", []) or [])
        recent = list(getattr(self, "_agent_history", []) or [])
        if getattr(self, "_agent_show_archived", False) and archived:
            self._append_chat_note(f"ARCHIVE · showing {len(archived)} older messages")
            for item in archived:
                role = str(item.get("role") or "AGENT").upper()
                color = BLUE if role == "YOU" else AMBER if role == "AGENT" else LABEL
                self._append_chat(
                    role,
                    str(item.get("message") or ""),
                    color,
                    ts=item.get("ts"),
                    provider=item.get("provider"),
                )
            self._append_chat_note("RECENT")
        for item in recent:
            role = str(item.get("role") or "AGENT").upper()
            color = BLUE if role == "YOU" else AMBER if role == "AGENT" else LABEL
            if role == "AGENT" and str(item.get("message") or "").startswith("ERROR:"):
                color = LOSS_HI
            self._append_chat(
                role,
                str(item.get("message") or ""),
                color,
                ts=item.get("ts"),
                provider=item.get("provider"),
                keep_lines=str(item.get("message") or "").startswith("BRIEF ·"),
            )
        if bool(getattr(self, "_agent_typing_visible", False)):
            self._animate_agent_typing()
        else:
            self._detach_agent_typing_widget()
        self._update_agent_chat_hint()

    @work(thread=True)
    def _run_agent_analysis(self, force: bool = True) -> None:
        if bool(getattr(self, "_agent_analysis_running", False)):
            # One analysis at a time: a second press would pay for the same answer twice.
            self.call_from_thread(self._set_status, "Agent analysis already running")
            return
        self._agent_analysis_running = True
        try:
            strategies = getattr(self, "_website_strategies", None)
            if strategies is not None:  # linked: fresh picks for the assigned strategy first
                if not strategies.loaded:
                    strategies.refresh()
                if strategies.assigned:
                    strategies.picks(strategies.assigned)
            self._sync_agent_account_context_env()
            self.call_from_thread(self._set_status, "⧖ Agent analyzing..." if force else "⧖ Loading agent...")
            self.call_from_thread(self._update_agent_status, "ANALYZING..." if force else "LOADING...", "running")
            cached_fresh = False
            if not force:
                try:
                    cached_fresh = _analysis_cache_is_fresh(load_agent_analysis())
                except Exception:
                    cached_fresh = False
            if not cached_fresh:
                preview = build_algo_shortlist_snapshot()
                self.call_from_thread(self._set_agent_shortlist_preview, preview)
            result = agent_analyze(force=force)
            self._agent_analysis = result
            self.call_from_thread(self._populate_agent_tab)
            self.call_from_thread(self._maybe_submit_agent_super_signal, result)
            age = time.time() - result.get("timestamp", 0)
            raw = str(result.get("raw_response") or "")
            if age < 5 and raw.startswith("ERROR:"):
                self.call_from_thread(
                    self._set_status, f"{_error_line(raw)} (showing rule-based picks meanwhile)"
                )
            elif age < 5:
                info = last_call_info()
                took = f" in {float(info.get('seconds') or 0):.0f}s" if info.get("seconds") else ""
                self.call_from_thread(self._set_status,
                    f"Agent analysis complete{took} │ {len(result.get('stocks', []))} stocks reviewed │ {self._agent_header_label()}")
            else:
                self.call_from_thread(self._set_status,
                    f"Agent loaded from cache ({int(age/60)}m ago) │ Press A+Enter to refresh")
        except Exception as e:
            self.call_from_thread(self._set_status, f"Agent error: {e}")
        finally:
            self._agent_analysis_running = False

    def _set_agent_shortlist_preview(self, preview: dict) -> None:
        current_rows = list((getattr(self, "_agent_analysis", {}) or {}).get("stocks") or [])
        if current_rows and not all(str(row.get("verdict") or "").upper() == "REVIEW" for row in current_rows):
            return
        self._agent_preview_override = dict(preview or {})
        self._populate_agent_tab()

    def _set_typing(self, visible: bool):
        self._agent_typing_visible = bool(visible)
        if not visible:
            self._agent_typing_frame = 0
            self._detach_agent_typing_widget()
        else:
            self._animate_agent_typing()

    def _animate_agent_typing(self) -> None:
        if not bool(getattr(self, "_agent_typing_visible", False)):
            self._detach_agent_typing_widget()
            return
        typing_w = self._ensure_agent_typing_widget()
        frames = ["[AGENT] : Typing.", "[AGENT] : Typing..", "[AGENT] : Typing..."]
        frame = frames[int(getattr(self, "_agent_typing_frame", 0)) % len(frames)]
        self._agent_typing_frame = int(getattr(self, "_agent_typing_frame", 0)) + 1
        typing_w.update(Text(frame, style=f"italic {LABEL}"))
        try:
            self.query_one("#agent-chat-scroll", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    @work(thread=True)
    def _agent_ask_async(self, question: str) -> None:
        if bool(getattr(self, "_agent_chat_busy", False)):
            self.call_from_thread(self._set_status, "Agent is already thinking | use /stop to cancel")
            return
        if looks_like_api_key(question):  # the composer refuses first; never send a key anyway
            self.call_from_thread(self._append_chat_note, KEY_IN_CHAT_MESSAGE)
            return

        self._sync_agent_account_context_env()
        self._agent_chat_request_id = int(getattr(self, "_agent_chat_request_id", 0)) + 1
        request_id = self._agent_chat_request_id
        self._agent_chat_stop_requested = False
        self._agent_chat_busy = True
        provider = self._current_agent_provider_label()
        append_external_agent_chat_message("YOU", question, source="tui_chat", provider=provider)
        self.call_from_thread(self._append_chat, "YOU", question, BLUE)
        self.call_from_thread(self._set_typing, True)
        self.call_from_thread(self._set_status, f"⧖ Agent thinking... ({self._agent_header_label()})")

        # Answer in this process (no new Python per question, and the briefing /
        # market caches stay warm); a helper thread keeps /stop and the timeout responsive.
        result: dict = {}

        def _answer() -> None:
            try:
                result["answer"] = agent_ask(question, persist_history=False)
                result["info"] = last_call_info()
            except Exception as exc:  # pragma: no cover - surfaced below
                result["error"] = exc

        helper = threading.Thread(target=_answer, name="agent-chat", daemon=True)
        helper.start()
        started = time.monotonic()

        def _superseded() -> bool:
            return request_id != int(getattr(self, "_agent_chat_request_id", 0)) or bool(
                getattr(self, "_agent_chat_stop_requested", False)
            )

        while helper.is_alive():
            if _superseded():
                return
            if (time.monotonic() - started) > AGENT_CHAT_TIMEOUT_SECS:
                self._agent_chat_busy = False
                self._agent_chat_request_id = request_id + 1  # drop the late answer
                note = f"Timed out after {AGENT_CHAT_TIMEOUT_SECS}s. Ask a tighter question or try again."
                append_external_agent_chat_message("AGENT", note, source="tui_chat", provider=provider)
                self.call_from_thread(self._set_typing, False)
                self.call_from_thread(self._append_chat, "AGENT", note, PURPLE)
                self.call_from_thread(self._set_status, "Agent timed out")
                return
            helper.join(0.1)
        if _superseded():
            return
        self._agent_chat_busy = False
        self.call_from_thread(self._set_typing, False)
        if "error" in result:
            self.call_from_thread(self._append_chat, "ERROR", str(result["error"]), LOSS_HI)
            self.call_from_thread(self._set_status, f"Agent error: {result['error']}")
            return
        answer = str(result.get("answer") or "").strip() or "Agent returned no output."
        append_external_agent_chat_message("AGENT", answer, source="tui_chat", provider=provider)
        if answer.startswith("ERROR:"):
            self.call_from_thread(self._append_chat, "AGENT", answer, LOSS_HI)
            self.call_from_thread(self._set_status, _error_line(answer))
            return
        self.call_from_thread(self._append_chat, "AGENT", answer, PURPLE)
        info = dict(result.get("info") or {})
        status = f"Agent responded │ {self._agent_header_label()}"
        if info.get("seconds"):
            status += f" │ {float(info['seconds']):.1f}s"
        if info.get("tool_calls"):
            status += f" │ {int(info['tool_calls'])} lookup{'s' if int(info['tool_calls']) != 1 else ''}"
        if info.get("notice"):
            status += f" │ {info['notice']}"
        self.call_from_thread(self._set_status, status)

    def _update_agent_status(self, text_str: str, state: str = "idle"):
        bar = self.query_one("#agent-status-bar", Static)
        a = getattr(self, '_agent_analysis', {})
        label = escape(self._agent_header_label())
        if state == "running":
            bar.update(Text.from_markup(
                f"[bold #bb88ff]◆ {label}[/]   [{YELLOW}]⧖ {text_str}[/]"))
            return
        trade = a.get("trade_today", None)
        tc = GAIN_HI if trade else LOSS_HI if trade is False else LABEL
        trade_str = "YES" if trade else "NO" if trade is False else "?"
        stocks = list(a.get("stocks", []) or a.get("shortlist", []) or [])
        n_approve = sum(1 for s in stocks if s.get("verdict", "").upper() == "APPROVE")
        n_reject = sum(1 for s in stocks if s.get("verdict", "").upper() == "REJECT")
        n_total = len(stocks)
        n_super = sum(1 for s in stocks if bool(s.get("auto_entry_candidate")))
        regime = a.get("regime", "?")
        import time as _t
        age = _t.time() - a.get("timestamp", 0)
        age_str = f"{int(age/60)}m ago" if age < 3600 else f"{int(age/3600)}h ago" if age < 86400 else "stale"
        bar.update(Text.from_markup(
            f"[bold #bb88ff]◆ {label}[/]   "
            f"[#888888]Trade today:[/] [bold {tc}]{trade_str}[/]   "
            f"[#888888]Verdicts:[/] [bold {GAIN_HI}]{n_approve}✓[/] "
            f"[bold {LOSS_HI}]{n_reject}✗[/] "
            f"[#888888]of {n_total}[/]   "
            f"[#888888]Super:[/] [bold {AMBER}]{n_super}[/]   "
            f"[#888888]Regime:[/] [{YELLOW}]{regime}[/]   "
            f"[#888888]Updated:[/] [{LABEL}]{age_str}[/]   "
            f"[#555555]A=Analyze  │  /agent  │  /model  │  /history[/]"
        ))

    def _populate_agent_meta_headers(self, analysis: dict) -> None:
        stocks = list((analysis or {}).get("stocks") or [])
        left = self.query_one("#agent-picks-subtitle", Static)
        right = self.query_one("#agent-chat-subtitle", Static)
        provider = self._agent_header_label()
        if stocks:
            buy_count = sum(1 for row in stocks if str(row.get("action_label") or "").upper() == "BUY")
            super_count = sum(1 for row in stocks if bool(row.get("auto_entry_candidate")))
            left.update(
                Text(
                    f"Ranked shortlist · {len(stocks)} names · {buy_count} buys · {super_count} super signals",
                    style=LABEL,
                )
            )
        else:
            left.update(Text("Refreshing live snapshot and building the ranked shortlist.", style=LABEL))
        archive_count = int(getattr(self, "_agent_archive_count", 0) or 0)
        right.update(
            Text(
                f"{provider} conversation stream · live session",
                style=LABEL,
            )
        )

    def _populate_agent_market_banner(self, analysis: dict) -> None:
        banner = self.query_one("#agent-market-view", Static)
        stocks = list((analysis or {}).get("stocks") or [])
        fresh_market = dict((analysis or {}).get("fresh_market") or {})
        parts: list[str] = []
        regime = str((analysis or {}).get("regime") or "unknown").upper()
        if regime and regime != "UNKNOWN":
            parts.append(f"[#888888]Regime[/] [bold {YELLOW}]{regime}[/]")
        session_date = str(fresh_market.get("session_date") or (analysis or {}).get("context_date") or "")
        if session_date:
            parts.append(f"[#888888]Session[/] [bold {WHITE}]{session_date}[/]")
        quote_count = int(fresh_market.get("quote_count") or 0)
        if quote_count > 0:
            parts.append(f"[#888888]Quotes[/] [bold {AMBER}]{quote_count}[/]")
        if stocks:
            top = stocks[0]
            parts.append(
                f"[#888888]Top[/] [bold {AMBER}]{str(top.get('symbol') or '—')}[/] "
                f"[{WHITE}]{str(top.get('action_label') or top.get('verdict') or 'REVIEW').upper()}[/] "
                f"[#888888]score[/] [bold {AMBER}]{float(top.get('signal_score') or 0.0):.2f}[/]"
            )
        elif not parts:
            parts.append("[#888888]Agent snapshot is loading[/]")
        banner.update(Text.from_markup("   [#444444]│[/]   ".join(parts)))

    def _populate_agent_detail_default(self, analysis: dict) -> None:
        try:
            self.query_one("#agent-detail-title", Static).update("FOCUS")
        except Exception:
            pass
        detail = self.query_one("#agent-detail", Static)
        stocks = list((analysis or {}).get("stocks") or [])
        fresh_market = dict((analysis or {}).get("fresh_market") or {})
        if stocks:
            top = stocks[0]
            top_symbol = str(top.get("symbol") or "—")
            top_action = str(top.get("action_label") or top.get("verdict") or "REVIEW").upper()
            top_score = float(top.get("signal_score") or 0.0)
            top_conv = float(top.get("conviction") or 0.0)
            summary = (
                f"Top setup: {top_symbol} {top_action}  "
                f"score {top_score:.2f}  conv {top_conv:.0%}\n"
                f"{str(top.get('what_matters') or top.get('reasoning') or '')[:220]}"
            )
            detail.update(Text(summary, style=WHITE))
            return
        session_date = str(fresh_market.get("session_date") or (analysis or {}).get("context_date") or "—")
        source = str(fresh_market.get("source") or "snapshot")
        quote_count = int(fresh_market.get("quote_count") or 0)
        detail.update(
            Text(
                f"Waiting for ranked picks.\n"
                f"Session: {session_date}  Source: {source}  Quotes: {quote_count}",
                style=LABEL,
            )
        )

    def _show_agent_focus_row(self, index: int) -> None:
        analysis = getattr(self, "_agent_analysis", {}) or {}
        stocks = list(analysis.get("stocks") or [])
        if not (0 <= int(index) < len(stocks)):
            self._populate_agent_detail_default(analysis)
            return

        stock = dict(stocks[int(index)] or {})
        symbol = str(stock.get("symbol") or "").upper()
        verdict = str(stock.get("verdict") or "?").upper()
        action_label = str(stock.get("action_label") or verdict or "REVIEW").upper()
        action_color = {
            "BUY": GAIN_HI,
            "SELL": LOSS_HI,
            "PASS": LOSS_HI,
            "HOLD": YELLOW,
            "REVIEW": LABEL,
        }.get(action_label, WHITE)
        conviction = float(stock.get("conviction", 0) or 0.0)
        signal_score = float(stock.get("signal_score") or 0.0)
        detail = Text.assemble(
            (f"[{symbol}] ", f"bold {AMBER}"),
            (f"{action_label} ", f"bold {action_color}"),
            (f"score {signal_score:.2f} ", YELLOW),
            (f"conv {conviction:.0%}\n", YELLOW),
            (str(stock.get("what_matters") or stock.get("reasoning") or ""), WHITE),
            ("\n", WHITE),
            (f"Bull: {str(stock.get('bull_case') or 'n/a')}\n", GAIN),
            (f"Risk: {str(stock.get('bear_case') or 'n/a')}", LOSS if stock.get("bear_case") else LABEL),
        )
        self.query_one("#agent-detail-title", Static).update(f"FOCUS · {symbol}")
        self.query_one("#agent-detail", Static).update(detail)

    def _populate_agent_tab(self):
        self._load_agent_runtime_state()
        preview = dict(getattr(self, "_agent_preview_override", {}) or {})
        live_rows = list((self._agent_analysis or {}).get("stocks") or [])
        if preview and not live_rows:
            self._agent_analysis = preview
        elif live_rows and not all(str(row.get("verdict") or "").upper() == "REVIEW" for row in live_rows):
            self._agent_preview_override = None
        a = getattr(self, '_agent_analysis', {})
        self._update_agent_status("", "idle")
        self._populate_agent_market_banner(a)
        self._populate_agent_meta_headers(a)
        self._render_agent_chat_history()

        # Verdicts table — short summary, full reasoning on row select
        dt = self.query_one("#dt-agent-verdicts", DataTable)
        dt.clear(columns=True)
        for label, key, width in [
            (" #", "n", 2),
            ("SYMBOL", "sym", 6),
            ("SIGNAL", "algo", 9),
            ("ACTION", "v", 8),
            ("CONV", "conv", 4),
            ("SCORE", "score", 5),
            ("KEY POINT", "kp", 32),
        ]:
            dt.add_column(label, key=key, width=width)

        stocks = list(a.get("stocks", []) or a.get("shortlist", []) or [])
        if stocks:
            for i, s in enumerate(stocks, 1):
                verdict = s.get("verdict", "?").upper()
                action_label = str(s.get("action_label") or verdict or "REVIEW").upper()
                vc = {
                    "BUY": f"bold {GAIN_HI}",
                    "SELL": f"bold {LOSS_HI}",
                    "PASS": f"bold {LOSS_HI}",
                    "HOLD": f"bold {YELLOW}",
                    "REVIEW": LABEL,
                }.get(action_label, LABEL)
                conv = s.get("conviction", 0)
                conv_c = GAIN_HI if conv >= 0.7 else YELLOW if conv >= 0.4 else LOSS
                # Extract first sentence as key point
                reasoning = str(s.get("what_matters") or s.get("reasoning") or "")
                first_sentence = reasoning.split(". ")[0].split(" — ")[0][:32]
                score = float(s.get("signal_score") or 0.0)
                score_style = GAIN_HI if score >= 1.0 else YELLOW if score >= 0.7 else LABEL
                dt.add_row(
                    _dim_text(f"{i:2d}"),
                    _sym_text(s.get("symbol", "")),
                    Text(str(s.get("signal_type") or s.get("algo_signal") or "")[:8], style=f"bold {AMBER}"),
                    Text(f" {action_label} ", style=vc),
                    Text(f"{conv:.0%}", style=conv_c),
                    Text(f"{score:.2f}", style=score_style),
                    Text(first_sentence, style=WHITE),
                )
        else:
            dt.add_row(_dim_text("—"), _dim_text(self._agent_empty_picks_reason(a)), *[Text("")] * 5)
        self._populate_agent_detail_default(a)

    def _agent_empty_picks_reason(self, analysis: dict | None) -> str:
        """Why the picks table is empty, in a few words (never a bare "Loading")."""
        a = dict(analysis or {})
        if getattr(self, "_agent_analysis_running", False) or not a:
            return "Analysing… (A to run again)"
        strategies = getattr(self, "_website_strategies", None)
        if strategies is not None and not strategies.assigned:
            return "No strategy assigned: 0 Strategies → ASSIGN, then A"
        raw = str(a.get("raw_response") or "")
        if raw.startswith("ERROR:"):
            return raw[6:].strip()[:70]
        if not list(a.get("shortlist") or []):
            return "No candidates today (the strategy picked nothing)"
        return "No picks yet: press A to analyse"

    def _append_chat(
        self,
        role: str,
        message: str,
        color: str,
        *,
        ts: float | None = None,
        provider: str | None = None,
        keep_lines: bool = False,
    ):
        scroll = self.query_one("#agent-chat-scroll", VerticalScroll)
        self._detach_agent_typing_widget()
        css_class = "chat-user" if role == "YOU" else "chat-agent"
        label = "YOU" if role == "YOU" else "AGENT" if role == "AGENT" else role.upper()
        ts_text = _format_nst_hm(ts)
        joiner = "\n" if keep_lines else " "  # a brief keeps its section lines
        flat_message = joiner.join(part.strip() for part in str(message or "").splitlines() if part.strip())
        if not flat_message:
            flat_message = "—"
        prefix = f"[{label}]"
        msg_text = Text.assemble(
            (prefix, f"bold {color}"),
            (" ", WHITE),
            (f"{ts_text} " if ts_text else "", LABEL),
            (": ", LABEL),
            (flat_message, WHITE),
        )
        widget = Static(msg_text, classes=f"chat-msg {css_class}")
        scroll.mount(widget)
        scroll.mount(Static("", classes="chat-gap"))
        scroll.scroll_end(animate=False)
        if bool(getattr(self, "_agent_typing_visible", False)):
            self._animate_agent_typing()

    def _handle_agent_chat_command(self, raw: str) -> bool:
        command = str(raw or "").strip().lower()
        if command in {"/history", "/archive", "/archived"}:
            self._agent_show_archived = True
            self._load_agent_runtime_state()
            self._render_agent_chat_history()
            self._set_status("Agent archive loaded")
            return True
        if command in {"/recent", "/hide", "/collapse"}:
            self._agent_show_archived = False
            self._load_agent_runtime_state()
            self._render_agent_chat_history()
            self._set_status("Agent archive hidden")
            return True
        if command == "/clear":
            self._agent_show_archived = False
            self._agent_visible_since = time.time()
            self._load_agent_runtime_state()
            self._render_agent_chat_history()
            self._set_status("Agent chat screen cleared")
            return True
        if command == "/stop":
            if self._stop_active_agent_chat():
                return True
            self._set_status("No active agent response to stop")
            return True
        if command in {"/agent", "/agent status"}:
            summary = self._agent_runtime_summary()
            self._append_chat_note(f"Active agent: {summary}. Config: {ACTIVE_AGENT_FILE}")
            self._set_status(f"Active agent: {summary}")
            return True
        if command in {"/agent list", "/agents", "/backends"}:
            help_text = self._agent_backends_help()
            self._append_chat_note(help_text)
            self._set_status(help_text)
            return True
        if command.startswith("/agent "):
            try:
                parts = shlex.split(raw)
                if len(parts) < 2:
                    raise ValueError(AGENT_SWITCH_USAGE)
                preset = parts[1].strip().lower()
                model = parts[2].strip() if len(parts) >= 3 else None
                set_active_agent(preset, model=model)
                if model and preset in REMEMBERED_MODEL_PRESETS:
                    self._remember_agent_model(preset, model)
                summary = self._agent_runtime_summary()
                self._append_chat_note(f"Agent switched: {summary}")
                self._set_status(f"Agent switched to {self._agent_header_label()}")
                self._populate_agent_tab()
                return True
            except Exception as exc:
                self._set_status(f"Agent switch failed: {exc}. {AGENT_SWITCH_USAGE}")
                return True
        if command.startswith("/model"):
            try:
                parts = shlex.split(raw)
                if len(parts) < 2:
                    summary = self._agent_runtime_summary()
                    self._append_chat_note(f"Active model: {summary}")
                    self._set_status(f"Active agent: {summary}")
                    return True
                model = " ".join(parts[1:]).strip()
                cfg = load_active_agent_config()
                preset = str(cfg.get("selected_preset") or cfg.get("backend") or "ollama")
                saved = set_active_agent(preset, model=model)
                if preset in REMEMBERED_MODEL_PRESETS:
                    self._remember_agent_model(preset, model)
                summary = self._agent_runtime_summary()
                self._append_chat_note(f"Agent model updated: {summary}")
                self._set_status(f"Agent model set to {saved.get('model')}")
                self._populate_agent_tab()
                return True
            except Exception as exc:
                self._set_status(f"Model update failed: {exc}")
                return True
        if command in {"/brief", "/brief now"}:
            self._append_chat_note("Preparing a brief: your names, news-linked movers, names worth researching.")
            self._run_agent_brief(None, manual=True)
            return True
        if command.startswith("/proactive"):
            from backend.agents import proactive

            arg = command[len("/proactive"):].strip()
            if arg in {"on", "off"}:
                proactive.set_enabled(arg == "on")
                self._agent_brief_skip_until = 0.0
            enabled = proactive.proactive_enabled()
            text = (
                "Automatic briefs are ON: once a day, and on trading days at pre-open (10:30), mid-session (13:00) "
                "and after the close (15:15 NPT). /proactive off to stop, /brief for one now."
                if enabled
                else "Automatic briefs are OFF. /proactive on to turn them back on, /brief for one now."
            )
            if arg not in {"", "on", "off", "status"}:
                text = "Usage: /proactive on|off|status"
            self._append_chat_note(text)
            self._set_status(text)
            return True
        if command == "/help":
            self._set_status(
                "Agent commands: /brief, /proactive on|off, /agent, /agent list, /agent ollama|openai|anthropic|claude|gemma4_mlx [model], "
                "/model <name>, /history, /recent, /clear, /stop"
            )
            return True
        return False

    def _remember_agent_model(self, preset: str, model: str) -> None:
        """Remember the model per provider in ai.json, so switching back keeps it."""
        try:
            update_ai_settings(model=model, model_provider=preset)
        except Exception:
            pass

    def _build_agent_auto_order_spec(self, analysis: dict) -> Optional[dict]:
        if str(getattr(self, "trade_mode", "paper")) != "paper":
            return None
        if not bool((analysis or {}).get("trade_today")):
            return None
        stocks = sorted(
            [dict(item) for item in list((analysis or {}).get("stocks") or []) if bool(item.get("auto_entry_candidate"))],
            key=lambda item: (
                -float(item.get("signal_score") or 0.0),
                -float(item.get("conviction") or 0.0),
            ),
        )
        if not stocks:
            return None
        positions = list((self._stats or {}).get("positions") or [])
        held_symbols = {str(pos.get("sym") or "").upper() for pos in positions}
        open_buy_symbols = {
            str(order.get("symbol") or "").upper()
            for order in list(getattr(self, "_paper_orders", []) or [])
            if str(order.get("status") or "").upper() == "OPEN" and str(order.get("action") or "").upper() == "BUY"
        }
        max_positions = int(
            LONG_TERM_CONFIG.get("regime_max_positions", {}).get(
                str((analysis or {}).get("regime") or "").lower(),
                LONG_TERM_CONFIG.get("max_positions", 5),
            )
        )
        if len(held_symbols | open_buy_symbols) >= max_positions:
            return None
        prices = self.md.ltps() if hasattr(self, "md") else {}
        cash = float((self._stats or {}).get("cash") or 0.0)
        nav = float((self._stats or {}).get("nav") or cash or INITIAL_CAPITAL)
        per_position_budget = min(nav / max_positions, cash * 0.95) if max_positions > 0 else 0.0
        for stock in stocks:
            symbol = str(stock.get("symbol") or "").upper()
            if symbol in held_symbols or symbol in open_buy_symbols:
                continue
            price = float(prices.get(symbol) or stock.get("last_price") or 0.0)
            if price <= 0:
                continue
            quantity = int(per_position_budget / price)
            if quantity < 10:
                continue
            order_key = f"{analysis.get('context_date') or ''}:{symbol}:{int(float(analysis.get('timestamp') or 0))}"
            if order_key == self._last_agent_auto_order_key:
                return None
            return {
                "symbol": symbol,
                "quantity": quantity,
                "price": round(price, 2),
                "order_key": order_key,
                "signal_score": float(stock.get("signal_score") or 0.0),
                "conviction": float(stock.get("conviction") or 0.0),
            }
        return None

    def _maybe_submit_agent_super_signal(self, analysis: dict) -> None:
        from backend.quant_pro.control_plane.command_service import build_tui_control_plane

        spec = self._build_agent_auto_order_spec(analysis or {})
        if not spec:
            return
        result = build_tui_control_plane(self).submit_paper_order(
            action="buy",
            symbol=spec["symbol"],
            quantity=int(spec["quantity"]),
            limit_price=float(spec["price"]),
            thesis="agent_super_signal",
            confidence=float(spec["conviction"]),
            source_signals=["agent_super_signal"],
        )
        if not result.ok:
            return
        self._last_agent_auto_order_key = str(spec["order_key"])
        self._append_chat_note(
            f"AUTO BUY queued {spec['symbol']} x{spec['quantity']} @ {spec['price']:.2f} "
            f"(score {spec['signal_score']:.2f}, conv {spec['conviction']:.0%})"
        )
        self._set_status(f"Agent auto-buy queued {spec['symbol']} x{spec['quantity']} @ {spec['price']:.2f}")
