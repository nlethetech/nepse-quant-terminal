"""Textual event handlers + key-binding action_* methods + result callbacks.

The cross-cutting dispatch surface: the giant on_button_pressed dispatcher, the
on_input_*/on_key/on_data_table_*/on_option_list_* handlers, the action_* key
bindings, the screen-result callbacks, and the signal-button/live-TMS helpers.
Every handler fans out into the other mixins via self — all targets resolve
through the NepseDashboard MRO.

action_buy/action_sell live in OrderBookMixin (beside _on_buy/_on_sell) per the
Symbol Migration Map's determinate homing; the command palette here calls them
via self.

Degrade-graceful imports (e.g. validation.quick_chart in the strategy-chart
button path) stay in-function per the lazy-fallback-import rule.
"""

from pathlib import Path

from rich.text import Text
from textual import events, work
from textual.widgets import Button, ContentSwitcher, DataTable, Input, Static

from apps.tui.theme import LABEL, LOSS_HI, WHITE, YELLOW
from apps.tui.widgets.signal_defs import _SIG_ID_PREFIX, _sig_type_from_id
from apps.tui.io.watchlist_io import _normalize_watchlist_entry, _stock_watchlist_entry
from apps.tui.screens.command_palette import CommandPalette
from apps.tui.screens.lookup import LookupScreen
from apps.tui.screens.watchlist_add import WatchlistAddScreen


class EventsActionsMixin:
    def action_tab(self, name: str) -> None:
        if name == "screener":
            name = "signals"
        if name == "news":
            name = "agents"
        self.active_tab = name
        self.query_one("#content", ContentSwitcher).current = name
        self._update_header()
        self._refresh_active_tab_view(force_watchlist_sync=True)

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_lookup(self) -> None:
        self.push_screen(LookupScreen(), callback=self._on_lookup)

    def action_run_agent(self) -> None:
        if self.active_tab == "account":
            try:
                self._set_status(self._account_activate_selected())
            except Exception as exc:
                self._set_status(f"Account activate failed: {exc}")
            return
        self.action_tab("agents")
        self._run_agent_analysis(force=True)

    def action_tf(self, tf: str) -> None:
        """Switch lookup chart timeframe: D/W/M/I."""
        if self.active_tab == "account":
            if tf == "W":
                try:
                    self._set_status(self._account_sync_watchlist())
                except Exception as exc:
                    self._set_status(f"Watchlist sync failed: {exc}")
            return
        if self.active_tab != "lookup":
            return
        self.lookup_tf = tf
        self._populate_lookup()
        tf_names = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly", "I": "Intraday"}
        self._set_status(f"Chart: {tf_names.get(tf, tf)}  │  D=Daily  W=Weekly  M=Monthly  Y=Yearly  I=Intraday")

    def action_refresh(self) -> None:
        if self.active_tab == "signals":
            self.action_force_signals_reload()
            self._do_refresh()
            return
        if self.active_tab == "watchlist" and self.trade_mode == "live" and self.tms_service:
            self._refresh_watchlist_live(force=True)
        self._do_refresh()

    def action_force_signals_reload(self) -> None:
        if self.active_tab != "signals":
            self._set_status("Force signals reload is available on the Signals tab")
            return
        self._reload_account_bindings_from_disk()
        self._signals_table_cache_key = ""
        self._signals_table_cache_payload = None
        self._set_signals_table_loading()
        self._load_signals_async(force=True)

    def _on_lookup(self, result: dict | None) -> None:
        if not result: return
        self.lookup_sym = result["symbol"]
        self.action_tab("lookup")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection across tabs."""
        # Watchlist: Enter → lookup
        if event.data_table.id == "dt-watchlist":
            idx = event.cursor_row
            stock_rows = getattr(self, "_watchlist_stock_rows", [])
            if 0 <= idx < len(stock_rows):
                entry = stock_rows[idx]
                if str(entry.get("kind") or "stock") == "stock":
                    self.lookup_sym = str(entry.get("symbol") or entry.get("label") or "").upper()
                    if self.lookup_sym:
                        self.action_tab("lookup")
                else:
                    self._rates_search_query = str(entry.get("label") or "").strip()
                    self.action_tab("kalimati")
                    try:
                        self.query_one("#kalimati-search-input", Input).value = self._rates_search_query
                    except Exception:
                        pass
            return
        if event.data_table.id == "dt-watchlist-rates":
            idx = event.cursor_row
            if 0 <= idx < len(getattr(self, "_watchlist_rates_rows", [])):
                item = self._watchlist_rates_rows[idx]
                self._rates_search_query = str(item.get("label") or "").strip()
                self.action_tab("kalimati")
                try:
                    self.query_one("#kalimati-search-input", Input).value = self._rates_search_query
                except Exception:
                    pass
            return
        if event.data_table.id == "dt-watchlist-commodities":
            idx = event.cursor_row
            if 0 <= idx < len(getattr(self, "_watchlist_commodity_rows", [])):
                item = self._watchlist_commodity_rows[idx]
                self._rates_search_query = str(item.get("label") or "").strip()
                self.action_tab("kalimati")
                try:
                    self.query_one("#kalimati-search-input", Input).value = self._rates_search_query
                except Exception:
                    pass
            return
        if event.data_table.id == "dt-account-list":
            idx = event.cursor_row
            accounts = list(getattr(self, "_paper_accounts", []) or [])
            if 0 <= idx < len(accounts):
                selected = accounts[idx]
                self._selected_account_id = str(selected.get("id") or self._current_account_id)
                try:
                    self.query_one("#profile-inp-account-name", Input).value = str(selected.get("name") or "")
                except Exception:
                    pass
                try:
                    snap = self._account_snapshot(selected)
                    self.query_one("#profile-inp-target-nav", Input).value = f"{float(snap['nav']):,.2f}".replace(",", "")
                except Exception:
                    pass
                self._populate_account_list()
                self._populate_strategies_tab()
                selected_name = str(selected.get("name") or self._selected_account_id)
                self._set_status(f"Selected {selected_name} | press ACTIVATE to switch")
            return
        if event.data_table.id == "dt-strategy-list" and self._website_strategies is not None:
            self._webstrat_select_row(event.cursor_row)
            return
        if event.data_table.id == "dt-strategy-list":
            idx = event.cursor_row
            strategies = list(getattr(self, "_strategies", []) or [])
            if 0 <= idx < len(strategies):
                selected = strategies[idx]
                self._selected_strategy_id = str(selected.get("id") or self._selected_strategy_id)
                self._set_strategy_form_from_payload(selected)
                self._populate_strategies_tab()
                self._set_status(f"Selected strategy {selected.get('name')}")
            return
        # dt-news replaced by OptionList — handled in on_option_list_option_highlighted

        # Screener: Enter → lookup
        if event.data_table.id == "dt-screener":
            try:
                row_data = event.data_table.get_row_at(event.cursor_row)
                sym = str(row_data[0]).strip()
                if sym and sym != "—":
                    self.lookup_sym = sym
                    self.action_tab("lookup")
            except Exception:
                pass
            return
        # Agent verdicts
        if event.data_table.id != "dt-agent-verdicts":
            return
        a = getattr(self, '_agent_analysis', {})
        stocks = a.get("stocks", [])
        idx = event.cursor_row
        self._show_agent_focus_row(idx)
        if 0 <= idx < len(stocks):
            symbol = str(stocks[idx].get("symbol") or "").strip().upper()
            if symbol:
                self.lookup_sym = symbol

        # Market view
        mv = self.query_one("#agent-market-view", Static)
        market_view = a.get("market_view", "")
        trade_reason = a.get("trade_today_reason", "")
        risks = a.get("risks", [])
        parts = []
        if market_view:
            parts.append(f"[bold #bb88ff]VIEW:[/] [{WHITE}]{market_view}[/]")
        if trade_reason:
            parts.append(f"[bold #bb88ff]TRADE:[/] [{YELLOW}]{trade_reason}[/]")
        if risks:
            parts.append(f"[bold {LOSS_HI}]RISKS:[/] [{LABEL}]{' | '.join(risks[:3])}[/]")
        if parts:
            mv.update(Text.from_markup("   ".join(parts)))
        else:
            mv.update(Text("Press A to run agent analysis, or type a question below", style=LABEL))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "dt-strategy-list" and self._website_strategies is not None:
            self._webstrat_select_row(event.cursor_row)
            return
        if event.data_table.id == "dt-strategy-list":
            idx = event.cursor_row
            strategies = list(getattr(self, "_strategies", []) or [])
            if 0 <= idx < len(strategies):
                selected = strategies[idx]
                self._selected_strategy_id = str(selected.get("id") or self._selected_strategy_id)
                # Selecting from the list exits new-strategy mode
                self._new_strategy_mode = False
                self._set_strategy_form_from_payload(selected)
            return
        if event.data_table.id != "dt-agent-verdicts":
            return
        self._show_agent_focus_row(event.cursor_row)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        bid = event.button.id
        if str(bid or "").startswith("webstrat-") and self._on_webstrat_button(bid):
            return
        if bid == "hedge-toggle-btn":
            self._hedge_enabled = not self._hedge_enabled
            try:
                btn = self.query_one("#hedge-toggle-btn", Button)
                if self._hedge_enabled:
                    btn.label = "● HEDGE ON"
                    btn.remove_class("hedge-btn-off")
                    btn.add_class("hedge-btn-on")
                else:
                    btn.label = "○ HEDGE OFF"
                    btn.remove_class("hedge-btn-on")
                    btn.add_class("hedge-btn-off")
            except Exception:
                pass
            # Propagate to all running engines
            for _eng in list(getattr(self, "_account_engines", {}).values()):
                try:
                    _eng.set_hedge_enabled(self._hedge_enabled)
                except Exception:
                    pass
            self._render_hedge_panel()
            return
        if bid in ("bf-filter-all", "bf-filter-circ", "bf-filter-pump", "bf-filter-smart"):
            self._broker_floor_filter = {"bf-filter-all": "all", "bf-filter-circ": "circ",
                                         "bf-filter-pump": "pump", "bf-filter-smart": "smart"}[bid]
            for btn_id in ("bf-filter-all", "bf-filter-circ", "bf-filter-pump", "bf-filter-smart"):
                try:
                    btn = self.query_one(f"#{btn_id}", Button)
                    if btn_id == bid:
                        btn.add_class("bf-active")
                    else:
                        btn.remove_class("bf-active")
                except Exception:
                    pass
            self._apply_broker_floor_filter()
            return
        if bid == "news-search-clear":
            self._news_search_query = ""
            self._vector_search_results = []
            try:
                self.query_one("#news-search-input", Input).value = ""
            except Exception:
                pass
            self._populate_news()
            return
        if bid == "kalimati-search-clear":
            self._rates_search_query = ""
            try:
                self.query_one("#kalimati-search-input", Input).value = ""
            except Exception:
                pass
            self._populate_kalimati()
            return
        if bid == "order-btn-buy":
            self._order_action = "BUY"
            self._refresh_order_action_buttons()
            self._set_status("Order action: BUY")
        elif bid == "order-btn-sell":
            self._order_action = "SELL"
            self._refresh_order_action_buttons()
            self._set_status("Order action: SELL")
        elif bid == "order-btn-submit":
            self._on_order_submit()
        elif bid == "order-btn-cancel-all":
            if self.trade_mode == "paper":
                msg = self._cancel_all_paper_orders()
                self._set_status(msg)
            else:
                self._set_status("Cancel all is only available for paper orders")
        elif bid == "profile-btn-browse-seed":
            self._browse_seed_file()
        elif bid == "profile-btn-create-account":
            try:
                self._set_status(self._account_create_from_form())
            except Exception as exc:
                self._set_status(f"Account create failed: {exc}")
        elif bid == "profile-btn-activate-account":
            try:
                self._set_status(self._account_activate_selected())
            except Exception as exc:
                self._set_status(f"Account activate failed: {exc}")
        elif bid == "profile-btn-delete-account":
            try:
                self._set_status(self._delete_paper_account(self._selected_account_id))
            except Exception as exc:
                self._set_status(f"Account delete failed: {exc}")
        elif bid == "profile-btn-set-nav":
            try:
                self._set_status(self._account_set_nav_from_form())
            except Exception as exc:
                self._set_status(f"NAV update failed: {exc}")
        elif bid == "profile-btn-sync-watchlist":
            try:
                self._set_status(self._account_sync_watchlist())
            except Exception as exc:
                self._set_status(f"Watchlist sync failed: {exc}")
        elif bid == "profile-btn-save-account":
            try:
                self._set_status(self._save_current_account_snapshot())
            except Exception as exc:
                self._set_status(f"Snapshot save failed: {exc}")
        elif bid.startswith(_SIG_ID_PREFIX):
            sig_type = _sig_type_from_id(bid)
            active = getattr(self, "_active_signals", set())
            if sig_type in active:
                active.discard(sig_type)
            else:
                active.add(sig_type)
            self._active_signals = active
            self._sync_signal_buttons()
            return
        elif bid == "strategy-btn-new":
            self._new_strategy_mode = True
            self._active_signals = set()
            self._sync_signal_buttons()
            # Pre-fill sensible defaults — user only needs to enter name, description, and signals
            _new_defaults = {
                "strategy-inp-name": "",
                "strategy-inp-description": "",
                "strategy-inp-holding-days": "40",
                "strategy-inp-rebalance": "5",
                "strategy-inp-max-positions": "5",
                "strategy-inp-stop-loss": "0.08",
                "strategy-inp-trailing-stop": "0.15",
                "strategy-inp-sector-limit": "0.35",
            }
            for _wid, _val in _new_defaults.items():
                try:
                    self.query_one(f"#{_wid}", Input).value = _val
                except Exception:
                    pass
            try:
                self.query_one("#strategy-inp-name", Input).focus()
            except Exception:
                pass
            self._set_status("New strategy — enter name, pick signals, then SAVE  (parameters pre-filled with defaults)")
            return
        elif bid == "strategy-btn-save":
            try:
                self._set_status(self._save_strategy_from_form())
            except Exception as exc:
                self._set_status(f"Strategy save failed: {exc}")
        elif bid == "strategy-btn-delete":
            try:
                self._set_status(self._delete_custom_strategy())
            except Exception as exc:
                self._set_status(f"Strategy delete failed: {exc}")
        elif bid == "strategy-btn-assign-current":
            try:
                self._set_status(self._assign_strategy_to_account(self._selected_strategy_id, self._current_account_id))
            except Exception as exc:
                self._set_status(f"Strategy assign failed: {exc}")
        elif bid == "strategy-btn-assign-selected":
            try:
                self._set_status(self._assign_strategy_to_account(self._selected_strategy_id, self._selected_account_id))
            except Exception as exc:
                self._set_status(f"Strategy assign failed: {exc}")
        elif bid == "strategy-btn-backtest":
            try:
                self._set_status(self._start_strategy_backtest())
            except Exception as exc:
                self._set_status(f"Strategy backtest failed: {exc}")
        elif bid == "strategy-btn-chart":
            selected = self._selected_strategy_payload()
            selected_id = str((selected or {}).get("id") or getattr(self, "_selected_strategy_id", "") or "").strip()
            result = self._selected_strategy_chart_result(selected_id)
            if not result:
                self._set_status("No chartable saved backtest artifact for this strategy")
            else:
                try:
                    from validation.quick_chart import generate_quick_chart
                    from backend.quant_pro.database import get_db_path as _get_db_path
                    name   = str((selected or {}).get("name") or (result.get("strategy") or {}).get("name") or "Strategy")
                    window = dict(result.get("window") or {})
                    start  = str(window.get("start") or self.query_one("#strategy-inp-backtest-start", Input).value or "").strip()
                    end    = str(window.get("end") or self.query_one("#strategy-inp-backtest-end", Input).value or "").strip()
                    path   = generate_quick_chart(
                        result,
                        strategy_name=name,
                        start_date=start or "2025-01-01",
                        end_date=end or "2026-04-11",
                        db_path=str(_get_db_path()),
                        auto_open=True,
                    )
                    self._set_status(f"Chart saved → {path}" if path else "Chart generation failed")
                except Exception as exc:
                    self._set_status(f"Chart failed: {exc}")
        elif bid == "order-btn-cancel":
            # Cancel selected order from daily order book
            try:
                dt = self.query_one("#dt-orders-daily", DataTable)
                row_key = dt.cursor_row
                if row_key is not None:
                    row_data = dt.get_row_at(row_key)
                    order_id = str(row_data[0]).strip()
                    if order_id and order_id != "—":
                        msg = self._cancel_paper_order(order_id)
                        self._set_status(msg)
            except Exception:
                pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle agent chat input + order form + news search."""
        if event.input.id == "news-search-input":
            q = event.value.strip()
            self._news_search_query = q
            if q:
                self._do_vector_search(q)
            else:
                self._vector_search_results = []
                self._populate_news()
            return
        if event.input.id == "profile-inp-portfolio":
            self._persist_profile_paths(portfolio_path=event.value.strip())
            self._set_status("Portfolio seed path updated")
            return
        if event.input.id == "profile-inp-account-name":
            self._set_status("Account name updated")
            return
        if event.input.id == "profile-inp-target-nav":
            try:
                self._set_status(self._seed_manual_nav(event.value))
                self._populate_portfolio_and_risk()
            except Exception as exc:
                self._set_status(f"NAV update failed: {exc}")
            return
        if event.input.id in ("order-inp-price", "order-inp-slippage"):
            self._on_order_submit()
            return
        if event.input.id in ("order-inp-symbol", "order-inp-qty"):
            return
        if event.input.id != "agent-input":
            return
        question = event.value.strip()
        if not question:
            return
        event.input.value = ""
        if self._refuse_secret_in_composer(question):
            return  # looked like an API key: not sent, not saved
        if question.upper() == "A":
            self._run_agent_analysis()
        elif question.startswith("/"):
            if not self._handle_agent_chat_command(question):
                self._set_status(f"Unknown agent command: {question}")
        else:
            self._agent_ask_async(question)

    def on_input_changed(self, event: Input.Changed) -> None:
        input_id = event.input.id or ""
        if input_id == "kalimati-search-input":
            self._rates_search_query = event.value.strip()
            self._populate_kalimati()
            return
        if input_id == "signal-min-score":
            try:
                self._signal_min_score = float(event.value.strip() or "0")
            except ValueError:
                self._signal_min_score = 0.0
            # Invalidate cache so next load respects new threshold
            self._signals_table_cache_key = ""
            self._signals_table_cache_payload = None
            return

    def on_key(self, event: events.Key) -> None:
        if str(self.active_tab or "") != "account":
            return
        if isinstance(self.focused, Input):
            return
        key = str(event.key or "").lower()
        try:
            if key == "n":
                self._set_status(self._account_create_from_form())
                event.stop()
            elif key == "v":
                self._set_status(self._account_set_nav_from_form())
                event.stop()
            elif key == "h":
                self._set_status(self._toggle_account_help())
                event.stop()
        except Exception as exc:
            event.stop()
            if key == "n":
                self._set_status(f"Account create failed: {exc}")
            elif key == "v":
                self._set_status(f"NAV update failed: {exc}")
            elif key == "h":
                self._set_status(f"Help failed: {exc}")

    def on_option_list_option_highlighted(self, event) -> None:
        if getattr(getattr(event, "option_list", None), "id", "") != "news-list":
            return
        highlighted = getattr(event.option_list, "highlighted", None)
        if highlighted is None:
            return
        stories = getattr(self, "_news_visible_stories", [])
        if 0 <= highlighted < len(stories):
            self._news_selected_index = highlighted

    def on_option_list_option_selected(self, event) -> None:
        if getattr(getattr(event, "option_list", None), "id", "") != "news-list":
            return
        self.on_option_list_option_highlighted(event)

    @staticmethod
    def _summarize_tms_watchlist_error(exc: Exception, action: str = "sync") -> str:
        detail = " ".join(str(exc).split())
        lowered = detail.lower()
        if "profile is already in use" in lowered or "processsingleton" in lowered or "singletonlock" in lowered:
            return f"TMS watchlist {action} paused: browser profile already in use"
        if "login required" in lowered:
            return f"TMS watchlist {action} paused: login required"
        if "member market watch page not ready" in lowered:
            return f"TMS watchlist {action} paused: member market watch not ready"
        if len(detail) > 140:
            detail = detail[:137] + "..."
        return f"TMS watchlist {action} failed: {detail}"

    @work(thread=True)
    def _browse_seed_file(self) -> None:
        """Open a native macOS file picker and populate the seed path input."""
        import subprocess as _sp
        self.call_from_thread(self._set_status, "Opening file picker…")
        try:
            script = (
                'POSIX path of (choose file with prompt '
                '"Select portfolio CSV (paper_portfolio.csv or MeroShare CSV)" '
                'of type {"csv", "CSV"})'
            )
            result = _sp.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120,
            )
            chosen = result.stdout.strip()
            if not chosen:
                self.call_from_thread(self._set_status, "Browse cancelled")
                return
            # Populate the seed input and show a preview
            self.call_from_thread(
                lambda p=chosen: (
                    setattr(self.query_one("#profile-inp-portfolio", Input), "value", p)
                    or self._set_status(f"Seed file selected: {Path(p).name}")
                )
            )
        except FileNotFoundError:
            self.call_from_thread(
                self._set_status,
                "osascript not found — type the path manually into the SEED field",
            )
        except Exception as exc:
            self.call_from_thread(self._set_status, f"Browse failed: {exc}")

    def action_command_palette(self) -> None:
        self.push_screen(CommandPalette(), callback=self._on_command)

    def _on_command(self, result: dict | None) -> None:
        if not result:
            return
        kind = result.get("kind", "")
        target = result.get("target", "")
        if kind == "tab":
            self.action_tab(target)
        elif kind == "action":
            if target == "buy":
                self.action_buy()
            elif target == "sell":
                self.action_sell()
            elif target == "refresh":
                self.action_refresh()
            elif target == "agent":
                self.action_run_agent()
        elif kind == "stock":
            self.lookup_sym = target
            self.action_tab("lookup")

    def action_watchlist_add(self) -> None:
        if self.active_tab == "kalimati":
            entry = self._watchlist_entry_from_rates_selection()
            if not entry:
                self._set_status("Select a commodity, macro price, or forex row to add")
                return
            self._add_local_watchlist_entry(entry)
            return
        self.push_screen(WatchlistAddScreen(), callback=self._on_watchlist_add)

    def _on_watchlist_add(self, item: dict | str | None) -> None:
        if not item:
            return
        if isinstance(item, dict):
            normalized = _normalize_watchlist_entry(item)
            if not normalized:
                self._set_status("Unable to add watchlist item")
                return
            if str(normalized.get("kind") or "stock") == "stock" and self.trade_mode == "live" and self.tms_service:
                sym = str(normalized.get("symbol") or normalized.get("label") or "").upper()
                if sym:
                    self._watchlist_add_live(sym)
                return
            self._add_local_watchlist_entry(normalized)
            return
        sym = str(item).upper()
        if self.trade_mode == "live" and self.tms_service:
            self._watchlist_add_live(sym)
            return
        self._add_local_watchlist_entry(_stock_watchlist_entry(sym))

    def action_watchlist_remove(self) -> None:
        """Remove the currently selected row from watchlist."""
        if self.active_tab != "watchlist":
            return
        try:
            focused = getattr(self, "focused", None)
            focused_id = getattr(focused, "id", "") if focused is not None else ""
            if focused_id == "dt-watchlist-rates":
                dt_rates = self.query_one("#dt-watchlist-rates", DataTable)
                idx = dt_rates.cursor_row
                rows = getattr(self, "_watchlist_rates_rows", [])
                if idx is not None and 0 <= idx < len(rows) and rows[idx].get("tracked"):
                    label = str(rows[idx].get("label") or "")
                    kind = str(rows[idx].get("kind") or "")
                    self._remove_local_watchlist_entry({"key": f"{kind}:{label}", "label": label})
                return
            if focused_id == "dt-watchlist-commodities":
                dt_commodities = self.query_one("#dt-watchlist-commodities", DataTable)
                idx = dt_commodities.cursor_row
                rows = getattr(self, "_watchlist_commodity_rows", [])
                if idx is not None and 0 <= idx < len(rows) and rows[idx].get("tracked"):
                    label = str(rows[idx].get("label") or "")
                    self._remove_local_watchlist_entry({"key": f"commodity:{label}", "label": label})
                return
            dt = self.query_one("#dt-watchlist", DataTable)
            row_idx = dt.cursor_row
            stock_rows = getattr(self, "_watchlist_stock_rows", [])
            if row_idx is not None and 0 <= row_idx < len(stock_rows):
                entry = stock_rows[row_idx]
                if str(entry.get("kind") or "stock") == "stock" and self.trade_mode == "live" and self.tms_service:
                    sym = str(entry.get("symbol") or entry.get("label") or "").upper()
                    if sym:
                        self._watchlist_remove_live(sym)
                    return
                self._remove_local_watchlist_entry(entry)
        except Exception:
            pass
