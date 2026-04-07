"""Unit tests for the unified TransactionCostModel."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from validation.transaction_costs import TransactionCostModel, CostBreakdown, RoundTripCost


class TestBrokerCommission:
    """Verify all 6 broker commission tiers match SEBON schedule."""

    def test_tier_1_flat_below_2500(self):
        assert TransactionCostModel.broker_commission(2000) == 10.0

    def test_tier_1_flat_at_2500(self):
        assert TransactionCostModel.broker_commission(2500) == 10.0

    def test_tier_2_up_to_50000(self):
        amount = 30000
        assert TransactionCostModel.broker_commission(amount) == pytest.approx(amount * 0.0036)

    def test_tier_3_up_to_500000(self):
        amount = 200000
        assert TransactionCostModel.broker_commission(amount) == pytest.approx(amount * 0.0033)

    def test_tier_4_up_to_2000000(self):
        amount = 1000000
        assert TransactionCostModel.broker_commission(amount) == pytest.approx(amount * 0.0031)

    def test_tier_5_up_to_10000000(self):
        amount = 5000000
        assert TransactionCostModel.broker_commission(amount) == pytest.approx(amount * 0.0027)

    def test_tier_6_above_10000000(self):
        amount = 15000000
        assert TransactionCostModel.broker_commission(amount) == pytest.approx(amount * 0.0024)

    def test_tier_boundary_50000(self):
        assert TransactionCostModel.broker_commission(50000) == pytest.approx(50000 * 0.0036)

    def test_tier_boundary_500000(self):
        assert TransactionCostModel.broker_commission(500000) == pytest.approx(500000 * 0.0033)


class TestSEBONFee:
    """Verify SEBON regulatory fee."""

    def test_sebon_fee_percentage(self):
        cost = TransactionCostModel.total_buy_cost(100, 500.0)
        expected_sebon = 100 * 500.0 * 0.00015
        assert cost.sebon_fee == pytest.approx(expected_sebon)


class TestNEPSEFee:
    """Verify NEPSE fee is 20% of broker commission."""

    def test_nepse_fee_is_20pct_of_commission(self):
        cost = TransactionCostModel.total_buy_cost(100, 500.0)
        assert cost.nepse_fee == pytest.approx(cost.commission * 0.20)


class TestDPCharge:
    """Verify DP charges."""

    def test_dp_charge_fixed_25(self):
        cost = TransactionCostModel.total_buy_cost(100, 500.0)
        assert cost.dp_charge == 25.0

    def test_dp_name_transfer_on_sell(self):
        cost = TransactionCostModel.total_sell_cost(100, 500.0)
        assert cost.dp_name_transfer == 5.0

    def test_no_dp_name_transfer_on_buy(self):
        cost = TransactionCostModel.total_buy_cost(100, 500.0)
        assert cost.dp_name_transfer == 0.0


class TestCapitalGainsTax:
    """Verify CGT rates and logic."""

    def test_cgt_short_term(self):
        # < 365 days = 7.5%
        tax = TransactionCostModel.capital_gains_tax(10000, holding_days=200)
        assert tax == pytest.approx(10000 * 0.075)

    def test_cgt_long_term(self):
        # >= 365 days = 5%
        tax = TransactionCostModel.capital_gains_tax(10000, holding_days=400)
        assert tax == pytest.approx(10000 * 0.05)

    def test_cgt_boundary_365_long_term(self):
        # Exactly 365 days = long-term (5%)
        tax = TransactionCostModel.capital_gains_tax(10000, holding_days=365)
        assert tax == pytest.approx(10000 * 0.05)

    def test_cgt_boundary_364_short_term(self):
        # 364 days = short-term (7.5%)
        tax = TransactionCostModel.capital_gains_tax(10000, holding_days=364)
        assert tax == pytest.approx(10000 * 0.075)

    def test_cgt_no_tax_on_loss(self):
        assert TransactionCostModel.capital_gains_tax(-5000, 200) == 0.0

    def test_cgt_no_tax_on_zero(self):
        assert TransactionCostModel.capital_gains_tax(0, 200) == 0.0


class TestRoundTrip:
    """Verify round-trip cost calculation."""

    def test_round_trip_components(self):
        rt = TransactionCostModel.round_trip_cost(
            shares=100, entry_price=500.0, exit_price=550.0, holding_days=30
        )
        assert isinstance(rt, RoundTripCost)
        assert rt.buy.amount == 100 * 500.0
        assert rt.sell.amount == 100 * 550.0
        assert rt.gross_pnl == pytest.approx((550 - 500) * 100)
        # CGT on gain of 5000, short-term
        assert rt.cgt == pytest.approx(5000 * 0.075)
        assert rt.total_cost == pytest.approx(rt.buy.total + rt.sell.total + rt.cgt)
        assert rt.net_pnl == pytest.approx(rt.gross_pnl - rt.total_cost)
        assert rt.cost_pct == pytest.approx(rt.total_cost / rt.buy.amount)

    def test_round_trip_loss_no_cgt(self):
        rt = TransactionCostModel.round_trip_cost(
            shares=100, entry_price=500.0, exit_price=450.0, holding_days=30
        )
        assert rt.cgt == 0.0
        assert rt.gross_pnl < 0


class TestCostBreakdownStr:
    """Verify human-readable output."""

    def test_buy_cost_str(self):
        cost = TransactionCostModel.total_buy_cost(100, 500.0)
        output = str(cost)
        assert "Broker commission" in output
        assert "SEBON" in output
        assert "DP charge" in output
        assert "Total fees" in output

    def test_round_trip_str(self):
        rt = TransactionCostModel.round_trip_cost(100, 500.0, 550.0, 30)
        output = str(rt)
        assert "BUY SIDE" in output
        assert "SELL SIDE" in output
        assert "Capital gains tax" in output
        assert "Net P&L" in output


class TestKnownExample:
    """Verify against a manually calculated example."""

    def test_manual_calculation(self):
        # Buy 200 shares at NPR 1000 = NPR 200,000
        shares, entry_price = 200, 1000.0
        amount = 200000
        # Tier 3: 0.33% → commission = 660
        expected_commission = amount * 0.0033
        assert TransactionCostModel.broker_commission(amount) == pytest.approx(expected_commission)

        buy = TransactionCostModel.total_buy_cost(shares, entry_price)
        assert buy.commission == pytest.approx(660.0)
        assert buy.sebon_fee == pytest.approx(200000 * 0.00015)  # 30.0
        assert buy.nepse_fee == pytest.approx(660 * 0.20)        # 132.0
        assert buy.dp_charge == 25.0
        assert buy.total == pytest.approx(660 + 30 + 132 + 25)   # 847.0


class TestBackwardCompatibility:
    """Verify backward-compatible methods match old NepseFees."""

    def test_total_fees_matches_old(self):
        # total_fees now includes NEPSE fee (20% of commission)
        amount = 100 * 500.0
        expected_commission = amount * 0.0036  # tier 2 (<=50000)
        expected_sebon = amount * 0.00015
        expected_nepse_fee = expected_commission * 0.20
        expected_dp = 25.0
        expected = expected_commission + expected_sebon + expected_nepse_fee + expected_dp
        assert TransactionCostModel.total_fees(100, 500.0) == pytest.approx(expected)

    def test_round_trip_pct_matches_old(self):
        pct = TransactionCostModel.round_trip_pct(100, 500.0, 550.0)
        assert pct > 0
        assert pct < 0.02  # Should be under 2%

    def test_calculate_total_fees_dict(self):
        fees = TransactionCostModel.calculate_total_fees(100, 500.0)
        assert "commission" in fees
        assert "sebon" in fees
        assert "nepse_fee" in fees
        assert "dp_charge" in fees
        assert "dp_name_transfer" in fees
        assert "total" in fees
        assert fees["dp_charge"] == 25.0
        assert fees["dp_name_transfer"] == 0.0  # is_buy=True default

        # Sell side should include DP name transfer
        sell_fees = TransactionCostModel.calculate_total_fees(100, 500.0, is_buy=False)
        assert sell_fees["dp_name_transfer"] == 5.0
        assert sell_fees["total"] > fees["total"]

    def test_calculate_tax_short_term(self):
        tax = TransactionCostModel.calculate_tax(10000, 200)
        assert tax == pytest.approx(10000 * 0.075)

    def test_calculate_tax_loss(self):
        assert TransactionCostModel.calculate_tax(-5000) == 0.0


class TestDividendTax:
    """Verify dividend tax calculation."""

    def test_dividend_tax_5pct(self):
        assert TransactionCostModel.dividend_tax(10000) == pytest.approx(500.0)

    def test_dividend_tax_no_tax_on_zero(self):
        assert TransactionCostModel.dividend_tax(0) == 0.0

    def test_dividend_tax_no_tax_on_negative(self):
        assert TransactionCostModel.dividend_tax(-100) == 0.0
