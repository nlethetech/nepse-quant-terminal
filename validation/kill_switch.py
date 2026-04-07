"""
Automatic trading kill switch.

Safety mechanism for live/paper trading. Halts trading when risk limits
are breached. Designed to be imported into live_trader.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class KillReason(Enum):
    DAILY_LOSS = "daily_loss_limit"
    DRAWDOWN = "max_drawdown_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    STALE_DATA = "stale_data"


class KillSwitch:
    """
    Automatic trading halt when risk limits are breached.

    Usage
    -----
    ks = KillSwitch()
    should_halt, reason = ks.check(
        current_nav=950_000, peak_nav=1_000_000,
        daily_pnl=-35_000, daily_start_nav=1_000_000,
        consecutive_losses=6,
        last_data_time=datetime.now()
    )
    if should_halt:
        # Stop all trading, alert user
        ...
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,
        max_drawdown_pct: float = 0.15,
        max_consecutive_losses: int = 5,
        stale_data_minutes: int = 30,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.stale_data_minutes = stale_data_minutes
        self._triggered = False
        self._trigger_reason: Optional[str] = None
        self._trigger_time: Optional[datetime] = None

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    @property
    def trigger_reason(self) -> Optional[str]:
        return self._trigger_reason

    def reset(self) -> None:
        """Reset the kill switch (requires manual intervention)."""
        self._triggered = False
        self._trigger_reason = None
        self._trigger_time = None
        logger.info("Kill switch reset")

    def check(
        self,
        current_nav: float,
        peak_nav: float,
        daily_pnl: float,
        daily_start_nav: float,
        consecutive_losses: int = 0,
        last_data_time: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        Check all risk limits.

        Parameters
        ----------
        current_nav : Current portfolio NAV
        peak_nav : Highest NAV since inception
        daily_pnl : P&L since start of today
        daily_start_nav : NAV at start of trading day
        consecutive_losses : Number of consecutive losing trades
        last_data_time : Timestamp of most recent price data

        Returns
        -------
        (should_halt, reason) — True if any limit breached
        """
        if self._triggered:
            return True, f"Previously triggered: {self._trigger_reason}"

        # 1. Daily loss limit
        if daily_start_nav > 0:
            daily_loss_pct = -daily_pnl / daily_start_nav
            if daily_loss_pct >= self.max_daily_loss_pct:
                return self._trigger(
                    KillReason.DAILY_LOSS,
                    f"Daily loss {daily_loss_pct:.2%} exceeds limit "
                    f"{self.max_daily_loss_pct:.2%}"
                )

        # 2. Max drawdown from peak
        if peak_nav > 0:
            drawdown_pct = (peak_nav - current_nav) / peak_nav
            if drawdown_pct >= self.max_drawdown_pct:
                return self._trigger(
                    KillReason.DRAWDOWN,
                    f"Drawdown {drawdown_pct:.2%} exceeds limit "
                    f"{self.max_drawdown_pct:.2%}"
                )

        # 3. Consecutive losses
        if consecutive_losses >= self.max_consecutive_losses:
            return self._trigger(
                KillReason.CONSECUTIVE_LOSSES,
                f"Consecutive losses ({consecutive_losses}) exceeds limit "
                f"({self.max_consecutive_losses})"
            )

        # 4. Stale data
        if last_data_time is not None:
            staleness = (datetime.now() - last_data_time).total_seconds() / 60.0
            if staleness >= self.stale_data_minutes:
                return self._trigger(
                    KillReason.STALE_DATA,
                    f"Data is {staleness:.0f} min stale (limit: "
                    f"{self.stale_data_minutes} min)"
                )

        return False, "OK"

    def _trigger(self, reason: KillReason, message: str) -> Tuple[bool, str]:
        """Record the trigger and return halt signal."""
        self._triggered = True
        self._trigger_reason = message
        self._trigger_time = datetime.now()
        logger.critical("KILL SWITCH TRIGGERED: %s", message)
        return True, message
