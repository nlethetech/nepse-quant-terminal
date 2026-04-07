"""Unit tests for NEPSE fee calculations."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from validation.transaction_costs import TransactionCostModel as NepseFees


class TestBrokerCommission:
    """Test broker commission tiers match NEPSE schedule."""

    def test_tier_below_2500(self):
        assert NepseFees.calculate_broker_commission(2000) == 10.0

    def test_tier_at_2500(self):
        assert NepseFees.calculate_broker_commission(2500) == 10.0

    def test_tier_2501_to_50000(self):
        amount = 30000
        assert NepseFees.calculate_broker_commission(amount) == pytest.approx(amount * 0.0036)

    def test_tier_50001_to_500000(self):
        amount = 200000
        assert NepseFees.calculate_broker_commission(amount) == pytest.approx(amount * 0.0033)

    def test_tier_500001_to_2000000(self):
        amount = 1000000
        assert NepseFees.calculate_broker_commission(amount) == pytest.approx(amount * 0.0031)

    def test_tier_2000001_to_10000000(self):
        amount = 5000000
        assert NepseFees.calculate_broker_commission(amount) == pytest.approx(amount * 0.0027)

    def test_tier_above_10000000(self):
        amount = 15000000
        assert NepseFees.calculate_broker_commission(amount) == pytest.approx(amount * 0.0024)


class TestTotalFees:
    """Test total fee calculation components."""

    def test_includes_commission_sebon_dp(self):
        fees = NepseFees.calculate_total_fees(qty=100, price=500.0, is_buy=True)
        amount = 100 * 500.0
        expected_commission = NepseFees.calculate_broker_commission(amount)
        expected_sebon = amount * 0.00015
        expected_nepse_fee = expected_commission * 0.20
        expected_dp = 25.0
        assert fees["commission"] == pytest.approx(expected_commission)
        assert fees["sebon"] == pytest.approx(expected_sebon)
        assert fees["nepse_fee"] == pytest.approx(expected_nepse_fee)
        assert fees["dp_charge"] == 25.0
        assert fees["dp_name_transfer"] == 0.0  # buy side
        assert fees["total"] == pytest.approx(
            expected_commission + expected_sebon + expected_nepse_fee + expected_dp
        )

    def test_small_order(self):
        fees = NepseFees.calculate_total_fees(qty=1, price=100.0)
        # Amount = 100, commission should be flat 10
        assert fees["commission"] == 10.0


class TestCapitalGainsTax:
    """Test CGT rate logic."""

    def test_short_term_gain(self):
        tax = NepseFees.calculate_tax(net_gain=10000, holding_period_days=200)
        assert tax == pytest.approx(10000 * 0.075)

    def test_long_term_gain(self):
        tax = NepseFees.calculate_tax(net_gain=10000, holding_period_days=400)
        assert tax == pytest.approx(10000 * 0.05)

    def test_no_gain(self):
        assert NepseFees.calculate_tax(net_gain=0) == 0.0

    def test_loss(self):
        assert NepseFees.calculate_tax(net_gain=-5000) == 0.0

    def test_boundary_365_days(self):
        # Exactly 365 days = long-term (5%)
        assert NepseFees.calculate_tax(10000, 365) == pytest.approx(10000 * 0.05)
        # 364 days = short-term (7.5%)
        assert NepseFees.calculate_tax(10000, 364) == pytest.approx(10000 * 0.075)
