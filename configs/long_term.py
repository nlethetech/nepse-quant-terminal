"""Long-term portfolio config — validated GO (Sharpe 1.78, 12/12 PASS).

Config B: Vol+Quality+LowVol+MeanReversion, 40 trading day holds, regime filter.
Core config proven over 6 years. Extended with dormant signals (CGO, lead-lag, 52wk)
that were already implemented in simple_backtest.py but not wired into daily generation.
Run validation/run_all.py after any change to confirm Sharpe stays above 1.60.
"""

LONG_TERM_CONFIG = {
    "holding_days": 40,
    "max_positions": 5,
    "signal_types": [
        # CORE — validated 6yr, do not remove
        "volume", "quality", "low_vol", "mean_reversion",
        # EXTENDED — deployed 2026-04-04, pending re-validation
        "disposition",   # CGO breakout (Grinblatt & Han 2005)
        "lead_lag",      # Sector spillover (Hou 2007)
        "52wk_high",     # 52-week high proximity (George & Hwang 2004)
    ],
    "rebalance_frequency": 5,
    "stop_loss_pct": 0.08,
    "trailing_stop_pct": 0.10,
    "use_regime_filter": True,
    "sector_limit": 0.35,
    "regime_max_positions": {"bull": 5, "neutral": 4, "bear": 2},
    "bear_threshold": -0.05,
    "initial_capital": 1_000_000,
    # Long-term does NOT use event exits or profit targets
    "profit_target_pct": None,
    "event_exit_mode": False,
}
