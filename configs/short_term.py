"""Short-term event-driven portfolio config — dividend capture only.

Data-backed by corporate action analysis:
- 128 filtered events over 6 years, mean return +1.71%, win rate 59.4%
- Optimal entry window: T-10 to T-5 before bookclose
- Exit: T-1 before bookclose (event_exit_mode)

Iteration 3: Removed settlement_pressure (15% win rate — was detecting natural
ex-dividend price drop as "selling pressure", not actual forced selling).
Event-only with corp_action signal. Also removed trailing stop — short holds
should use profit target + event exit, not trailing stop which whipsaws.
"""

SHORT_TERM_CONFIG = {
    "holding_days": 12,                         # Max hold if event exit doesn't fire
    "max_positions": 3,                         # Fewer positions = larger per-trade conviction
    "signal_types": ["corp_action"],
    "rebalance_frequency": 1,                   # Check every day (event-driven)
    "stop_loss_pct": 0.08,                      # 8% hard stop
    "trailing_stop_pct": 0.15,                  # Effectively disabled (wider than stop loss)
    "profit_target_pct": 0.06,                  # Lock in gains at 6% (below 8% to fire first)
    "event_exit_mode": True,                    # Exit on bookclose T-1
    "use_regime_filter": True,
    "sector_limit": 0.70,                       # Very relaxed (events are sector-agnostic)
    "regime_max_positions": {"bull": 3, "neutral": 3, "bear": 2},
    "bear_threshold": -0.03,
    "initial_capital": 1_000_000,
}
