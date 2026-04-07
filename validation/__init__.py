"""
NEPSE Quantitative Trading System — Production Readiness Validation Framework.

Modules:
    transaction_costs  - Unified TransactionCostModel (single source of truth)
    walk_forward       - Walk-forward cross-validation
    benchmark          - NEPSE index benchmark comparison
    statistical_tests  - PSR, DSR, t-test, minimum track record length
    sensitivity        - Parameter sensitivity analysis
    random_baseline    - Random entry baseline test
    monte_carlo        - Trade resampling + bootstrap confidence intervals
    regime_stress      - NEPSE regime stress tests
    slippage           - Slippage + liquidity model
    kill_switch        - Drawdown/loss kill switch
    run_all            - Master runner + go/no-go report
    report_generator   - PDF report with plots
"""
