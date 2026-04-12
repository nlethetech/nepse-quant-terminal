"""Public baseline strategy profile used by the live/TUI stack.

C5 is the default open-source strategy:
- six transparent signal families
- 40 trading day holds
- stock-only execution
- fixed 35% sector cap
"""

LONG_TERM_CONFIG = {
    "holding_days": 40,
    "max_positions": 5,
    "signal_types": [
        "volume",
        "quality",
        "low_vol",
        "mean_reversion",
        "quarterly_fundamental",
        "xsec_momentum",
    ],
    "rebalance_frequency": 5,
    "stop_loss_pct": 0.08,
    "trailing_stop_pct": 0.10,
    "use_trailing_stop": True,
    "use_regime_filter": True,
    "sector_limit": 0.35,
    "regime_max_positions": {"bull": 5, "neutral": 4, "bear": 2},
    "bear_threshold": -0.05,
    "initial_capital": 1_000_000,
    "profit_target_pct": None,
    "event_exit_mode": False,
}
