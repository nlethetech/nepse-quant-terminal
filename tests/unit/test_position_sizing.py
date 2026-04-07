"""Unit tests for position sizing logic."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from backend.quant_pro.alpha_practical import AlphaSignal, SignalType


class TestPositionSizingLimits:
    """Test that position sizing respects risk limits."""

    @pytest.fixture
    def sample_prices(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        return pd.DataFrame(
            {"NABIL": [1000.0] * 5, "SBL": [500.0] * 5, "KBL": [300.0] * 5},
            index=dates,
        )

    def test_single_position_cap(self, sample_prices):
        """No single position should exceed 15% of capital."""
        from scripts.signals.generate_daily_signals import calculate_position_sizes, MAX_SINGLE_POSITION_PCT

        signals = [
            AlphaSignal(
                symbol="NABIL",
                signal_type=SignalType.LIQUIDITY,
                direction=1,
                strength=0.9,
                confidence=0.9,
                reasoning="Test",
            ),
        ]
        result = calculate_position_sizes(signals, 1_000_000, sample_prices, max_positions=7)
        if not result.empty:
            # Weight is formatted as string "X.X%"
            for _, row in result.iterrows():
                weight = float(row["Weight"].strip("%")) / 100
                assert weight <= MAX_SINGLE_POSITION_PCT + 0.01  # small tolerance for rounding

    def test_negative_capital_rejected(self):
        """Negative capital should not produce positions."""
        from scripts.signals.generate_daily_signals import calculate_position_sizes

        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        prices = pd.DataFrame({"NABIL": [1000.0] * 5}, index=dates)
        signals = [
            AlphaSignal(
                symbol="NABIL",
                signal_type=SignalType.LIQUIDITY,
                direction=1,
                strength=0.5,
                confidence=0.5,
                reasoning="Test",
            ),
        ]
        result = calculate_position_sizes(signals, -100000, prices, max_positions=7)
        # With negative capital, shares should be 0 or negative → empty result
        if not result.empty:
            for _, row in result.iterrows():
                shares = int(str(row["Shares"]).replace(",", ""))
                assert shares <= 0

    def test_max_positions_respected(self, sample_prices):
        """Should not exceed max positions."""
        from scripts.signals.generate_daily_signals import calculate_position_sizes

        signals = [
            AlphaSignal(
                symbol=sym,
                signal_type=SignalType.LIQUIDITY,
                direction=1,
                strength=0.5,
                confidence=0.5,
                reasoning="Test",
            )
            for sym in ["NABIL", "SBL", "KBL"]
        ]
        result = calculate_position_sizes(signals, 1_000_000, sample_prices, max_positions=2)
        assert len(result) <= 2

    def test_empty_signals_returns_empty(self, sample_prices):
        from scripts.signals.generate_daily_signals import calculate_position_sizes

        result = calculate_position_sizes([], 1_000_000, sample_prices)
        assert result.empty
