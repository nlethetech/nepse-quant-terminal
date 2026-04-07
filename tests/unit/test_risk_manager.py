"""Unit tests for risk management logic."""

import pytest
from backend.trading.paper_trade_tracker import RiskManager


class TestTrailingStop:
    def test_triggered_at_threshold(self):
        # 10% drop from high watermark
        assert RiskManager.check_trailing_stop(current_price=90.0, high_watermark=100.0) is True

    def test_not_triggered_below_threshold(self):
        assert RiskManager.check_trailing_stop(current_price=95.0, high_watermark=100.0) is False

    def test_zero_watermark(self):
        assert RiskManager.check_trailing_stop(current_price=50.0, high_watermark=0.0) is False

    def test_exact_boundary(self):
        # Exactly 10% drop
        assert RiskManager.check_trailing_stop(current_price=90.0, high_watermark=100.0) is True
        # Just above (9.9% drop = not triggered)
        assert RiskManager.check_trailing_stop(current_price=90.1, high_watermark=100.0) is False


class TestHardStop:
    def test_triggered(self):
        # 8% loss from entry
        assert RiskManager.check_hard_stop(current_price=92.0, entry_price=100.0) is True

    def test_not_triggered(self):
        assert RiskManager.check_hard_stop(current_price=95.0, entry_price=100.0) is False

    def test_zero_entry(self):
        assert RiskManager.check_hard_stop(current_price=50.0, entry_price=0.0) is False


class TestTakeProfit:
    def test_triggered(self):
        # 20% gain
        assert RiskManager.check_take_profit(current_price=120.0, entry_price=100.0) is True

    def test_not_triggered(self):
        assert RiskManager.check_take_profit(current_price=110.0, entry_price=100.0) is False

    def test_zero_entry(self):
        assert RiskManager.check_take_profit(current_price=200.0, entry_price=0.0) is False


class TestPortfolioDrawdown:
    def test_triggered(self):
        # 15% portfolio drawdown
        assert RiskManager.check_portfolio_drawdown(current_value=85000, peak_value=100000) is True

    def test_not_triggered(self):
        assert RiskManager.check_portfolio_drawdown(current_value=90000, peak_value=100000) is False

    def test_zero_peak(self):
        assert RiskManager.check_portfolio_drawdown(current_value=0, peak_value=0) is False

    def test_at_peak(self):
        assert RiskManager.check_portfolio_drawdown(current_value=100000, peak_value=100000) is False
