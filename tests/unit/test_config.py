"""Unit tests for configuration and profile system."""

import os
import pytest


class TestConfigDefaults:
    """Verify critical config defaults are sane."""

    def test_trading_days(self):
        from backend.quant_pro.config import NEPSE_TRADING_DAYS
        assert NEPSE_TRADING_DAYS == 240

    def test_risk_free_rate(self):
        from backend.quant_pro.config import R_F_ANNUAL
        assert 0.0 < R_F_ANNUAL < 0.20

    def test_deploy_gate_thresholds(self):
        from backend.quant_pro.config import DEPLOY_GATE
        assert DEPLOY_GATE["min_cv_score"] > 0.5
        assert DEPLOY_GATE["min_psr"] > 0.8

    def test_trading_risk_params(self):
        from backend.quant_pro.config import (
            TRAILING_STOP_PCT,
            HARD_STOP_LOSS_PCT,
            TAKE_PROFIT_PCT,
            MAX_POSITIONS,
            DEFAULT_CAPITAL,
        )
        assert 0 < TRAILING_STOP_PCT < 1
        assert 0 < HARD_STOP_LOSS_PCT < 1
        assert 0 < TAKE_PROFIT_PCT < 1
        assert 1 <= MAX_POSITIONS <= 50
        assert DEFAULT_CAPITAL > 0


class TestProfileApplication:
    """Test that config profiles apply correctly."""

    def test_apply_valid_profile(self):
        from backend.quant_pro import config

        # Save original
        orig = config.LABEL_MODE
        try:
            result = config.apply_nepse_profile("nepse_research")
            assert config.LABEL_MODE == "alpha_vs_market"
            assert config.ACTIVE_NEPSE_PROFILE == "nepse_research"
        finally:
            # Restore
            config.LABEL_MODE = orig
            config.ACTIVE_NEPSE_PROFILE = None

    def test_apply_invalid_profile(self):
        from backend.quant_pro.config import apply_nepse_profile
        with pytest.raises(ValueError, match="Unknown NEPSE profile"):
            apply_nepse_profile("nonexistent_profile")

    def test_apply_empty_profile(self):
        from backend.quant_pro.config import apply_nepse_profile
        with pytest.raises(ValueError, match="Profile name is required"):
            apply_nepse_profile("")


class TestDeploymentGateCheck:
    def test_passing_gate(self):
        from backend.quant_pro.config import check_deployment_gate
        passed, tier, reasons = check_deployment_gate(
            cv_score=0.65, cv_std=0.10, prob_edge=0.015, psr=0.95
        )
        assert passed is True
        assert tier == "production"
        assert len(reasons) == 0

    def test_failing_gate(self):
        from backend.quant_pro.config import check_deployment_gate
        passed, tier, reasons = check_deployment_gate(
            cv_score=0.40, cv_std=0.25, prob_edge=0.001, psr=0.60
        )
        assert passed is False
        assert len(reasons) > 0
