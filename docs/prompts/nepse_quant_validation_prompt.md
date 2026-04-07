# NEPSE Quantitative Trading System — Production Readiness Validation Prompt

## Instructions for Claude Code

You are tasked with performing a comprehensive quantitative audit of my trading system before I deploy it with real money on the Nepal Stock Exchange (NEPSE). This is a full production readiness review. Go through every section below systematically. For each section, read my codebase, identify what exists, what is missing, and implement or flag the gaps. Ask me clarifying questions only when absolutely necessary.

---

## PHASE 1: CODEBASE DISCOVERY

First, explore the entire project structure. Map out:
- Data ingestion pipeline (where does NEPSE data come from, how is it stored)
- Signal generation logic (what indicators, models, or rules drive buy/sell decisions)
- Portfolio/position sizing logic
- Order execution logic
- Backtesting engine
- Any existing test suites

Print a summary of the architecture before proceeding.

---

## PHASE 2: SEBON/NEPSE FEE STRUCTURE VERIFICATION

This is critical. NEPSE has a specific fee structure that MUST be modeled accurately. Search the web for the latest SEBON (Securities Board of Nepal) fee schedule and NEPSE broker commission structure. Verify and implement the following:

### 2.1 Broker Commission (as of latest SEBON circular)
- Scrape or search for the latest SEBON broker commission slab rates
- The commission structure is tiered based on transaction amount:
  - Up to NPR 50,000: X%
  - NPR 50,001 to 500,000: X%
  - NPR 500,001 to 2,000,000: X%
  - NPR 2,000,001 to 10,000,000: X%
  - Above NPR 10,000,000: X%
- FIND THE EXACT CURRENT RATES. Do not hardcode old rates. Search SEBON's website (sebon.gov.np) and NEPSE (nepalstock.com) for the latest circular on broker commission rates.
- Verify if rates differ for institutional vs retail investors

### 2.2 SEBON Fee
- SEBON regulatory fee (currently believed to be 0.015% of transaction value, but VERIFY this)
- Check if this applies to both buy and sell sides

### 2.3 NEPSE Trading Fee
- NEPSE's own trading fee percentage
- Verify current rate from official sources

### 2.4 Deposit and Credit (DP) Charges
- CDSC (CDS and Clearing Limited) charges per transaction
- DP maintenance fees if relevant to modeling

### 2.5 Capital Gains Tax
- Short-term capital gains tax rate for listed securities in Nepal
- Long-term capital gains tax rate (holding period threshold)
- Check if there are different rates for individual vs institutional investors
- Verify the current rates from Inland Revenue Department or latest Finance Act

### 2.6 Dividend Tax
- Tax rate on cash dividends
- Tax treatment of stock dividends (bonus shares)

### 2.7 Implementation
After gathering all fees, create a single `TransactionCostModel` class/module that:
- Takes a transaction (buy/sell, amount, holding period) and returns the total cost
- Is used by BOTH the backtester and the live/paper trading engine
- Has unit tests verifying the fee calculations against known examples
- Has a config file or constants file where rates can be easily updated when SEBON changes them
- Prints a breakdown: broker commission + SEBON fee + NEPSE fee + DP charge + tax = total cost

---

## PHASE 3: BACKTEST INTEGRITY TESTS

### 3.1 Look-Ahead Bias Detection
- Audit every data access in the signal generation pipeline
- Ensure no future data leaks into current decisions
- Check specifically:
  - Are adjusted close prices being used? If so, are adjustments applied retroactively?
  - Is any aggregation (moving averages, etc.) using data that wouldn't be available at trade time?
  - Are corporate actions (splits, bonuses, rights) handled correctly in historical data?
  - Is there any data point that references tomorrow's open/close?
- Write an automated check that compares signals generated in walk-forward mode vs batch mode. They should be identical.

### 3.2 Survivorship Bias Check
- Does the universe of stocks include delisted companies?
- Does it include companies that were suspended for extended periods?
- If using a fixed universe, when was it defined? A universe defined today excludes past failures.
- Recommendation: Build a point-in-time universe that reflects what was tradeable on each historical date.

### 3.3 Walk-Forward Analysis
- Implement a walk-forward test framework if not already present:
  - Training window: configurable (default 2 years)
  - Test window: configurable (default 3 months)
  - Step forward: configurable (default 1 month)
  - Track performance metrics for each fold separately
  - Aggregate and report: mean, std, min, max of Sharpe across folds
  - Plot the equity curve stitched from out-of-sample segments only

### 3.4 Transaction Cost Integration in Backtest
- Verify the backtest uses the EXACT same `TransactionCostModel` from Phase 2
- Run the backtest with and without costs. Report the difference in:
  - Total return
  - Sharpe ratio
  - Number of trades
  - Average profit per trade

---

## PHASE 4: STATISTICAL ROBUSTNESS TESTS

### 4.1 Core Performance Metrics
Calculate and report all of the following for the out-of-sample period:

| Metric | Formula/Description | Target |
|--------|-------------------|--------|
| CAGR | Compound Annual Growth Rate | > NEPSE index CAGR |
| Sharpe Ratio | (Return - Rf) / Std(Return), use Nepal T-bill rate as Rf | > 1.0 |
| Sortino Ratio | (Return - Rf) / Downside Deviation | > 1.5 |
| Max Drawdown | Worst peak-to-trough decline | < 25% |
| Calmar Ratio | CAGR / Max Drawdown | > 0.5 |
| Win Rate | % of profitable trades | Report |
| Payoff Ratio | Avg Win / Avg Loss | Report |
| Profit Factor | Gross Profit / Gross Loss | > 1.5 |
| Expectancy | (Win% x Avg Win) - (Loss% x Avg Loss) | > 0 |
| Average Trade Duration | Mean holding period in days | Report |
| Skewness of Returns | Distribution shape | Positive preferred |
| Kurtosis of Returns | Tail risk measure | Report |
| VaR (95%) | Value at Risk, daily | Report |
| CVaR (95%) | Conditional VaR / Expected Shortfall | Report |

### 4.2 Benchmark Comparison
- Compare all metrics against buy-and-hold NEPSE index
- Compare against buy-and-hold of the NEPSE sensitive index (banking heavy)
- Report alpha and beta from regression against NEPSE index
- Information ratio: (Strategy Return - Benchmark Return) / Tracking Error

### 4.3 Statistical Significance
- Run a t-test on daily/weekly excess returns vs zero
- Report the p-value. If p > 0.05, the strategy may not be statistically significant
- Calculate the minimum track record length needed for significance (using Bailey & Lopez de Prado's formula)

---

## PHASE 5: OVERFITTING DETECTION

### 5.1 Parameter Sensitivity Analysis
- For every tunable parameter in the strategy:
  - Vary it across a reasonable range (at least 20 values)
  - Plot Sharpe ratio / total return as a function of that parameter
  - Flag any parameter where a small change (±10%) causes Sharpe to drop by more than 0.3
- The surface should be smooth. Jagged = overfitted.

### 5.2 Deflated Sharpe Ratio
- Implement the Deflated Sharpe Ratio (DSR) from Bailey & Lopez de Prado (2014)
- Inputs: number of strategy variants tried, best Sharpe, variance of Sharpe estimates
- If DSR p-value > 0.05, the strategy is likely overfitted

### 5.3 Combinatorial Purged Cross-Validation (CPCV)
- If feasible given data size, implement CPCV
- This tests whether performance is robust across many possible train/test splits
- Report the distribution of Sharpe ratios across all combinatorial paths

### 5.4 Random Entry Baseline
- Run 1000 simulations with random entry signals but using the SAME exit rules, position sizing, and risk management
- Compare your strategy's Sharpe against the distribution of random Sharpes
- Your strategy should be above the 95th percentile of random entries

---

## PHASE 6: MONTE CARLO SIMULATION

### 6.1 Trade Resampling
- Take the actual sequence of trades from the backtest
- Randomly resample (with replacement) the trade returns 10,000 times
- For each resampling, compute:
  - Terminal wealth
  - Max drawdown
  - Sharpe ratio
- Report:
  - 5th, 25th, 50th, 75th, 95th percentile of terminal wealth
  - 95th percentile of max drawdown (worst case you should plan for)
  - Probability of ruin (account dropping below X% of starting capital)

### 6.2 Bootstrap Confidence Intervals
- Bootstrap 95% confidence interval for Sharpe ratio
- Bootstrap 95% confidence interval for CAGR
- If the lower bound of Sharpe CI is below 0, the strategy may not be reliably profitable

---

## PHASE 7: NEPSE-SPECIFIC STRESS TESTS

### 7.1 Regime Testing
Run the strategy separately on these NEPSE periods and report metrics for each:
- Bull market: 2020-2021 (COVID recovery rally)
- Crash: 2021-2022 (post-peak decline)
- Sideways/Recovery: 2023-2024
- Recent: 2024-present
- The strategy does NOT need to profit in all regimes, but it MUST not blow up (max drawdown > 40%) in any

### 7.2 Liquidity Constraints
- For each stock in the universe, calculate average daily turnover (volume x price)
- Filter out any stock where your position size would exceed 10% of average daily volume
- Re-run the backtest with only liquid stocks. Compare results.
- Model partial fills: if you want to buy 1000 shares but average daily volume is 500, simulate filling over multiple days with price impact

### 7.3 Circuit Breaker Simulation
- NEPSE has ±10% daily price limits for most stocks (±5% for some)
- In the backtest, if a stock hits the circuit breaker:
  - You CANNOT exit at your stop loss price if the stock is locked at lower circuit
  - You CANNOT enter if the stock is locked at upper circuit
  - Model the realistic scenario: you're stuck until the circuit unlocks
- Re-run backtest with circuit breaker constraints and compare

### 7.4 Settlement Lag (T+2)
- NEPSE operates on T+2 settlement
- Model this: proceeds from a sale are not available for 2 trading days
- This affects capital allocation and position sizing
- Compare returns with and without T+2 modeling

### 7.5 Trading Calendar
- NEPSE trades Sunday through Thursday (NOT Monday through Friday)
- Market holidays follow Nepal's calendar (Dashain, Tihar, etc.)
- Verify the backtest respects the actual NEPSE trading calendar
- Check: are weekend calculations correct? Saturday and Friday are the weekend, not Saturday and Sunday.

---

## PHASE 8: RISK MANAGEMENT VALIDATION

### 8.1 Position Sizing
- What position sizing method is used? (Fixed, Kelly, fractional Kelly, risk parity, etc.)
- If Kelly: compute the Kelly fraction and verify a fractional Kelly (half or quarter Kelly) is used
- Run a simulation: what happens if you use 2x the current position size? Does it blow up?

### 8.2 Stop Loss / Take Profit
- Are stops in place? Test their effectiveness:
  - Backtest with vs without stops
  - Test different stop levels (1%, 2%, 5%, 10%)
  - Account for circuit breakers (stops may not execute at the stop price)

### 8.3 Correlation and Concentration
- Calculate pairwise correlation of returns for all stocks in the portfolio
- If average correlation > 0.6, the portfolio is effectively a concentrated bet
- Check sector concentration: what % of positions are in banking/finance? (NEPSE is heavily bank-weighted)
- Implement a sector cap (e.g., max 30% in any single sector)

### 8.4 Drawdown Recovery Analysis
- For every drawdown > 10%, calculate:
  - Duration (peak to trough)
  - Recovery time (trough back to previous peak)
  - What was happening in the market during that drawdown?

---

## PHASE 9: EXECUTION REALITY CHECK

### 9.1 Slippage Model
- Model slippage as a function of:
  - Order size relative to average daily volume
  - Bid-ask spread (estimate if not directly available)
- For illiquid NEPSE stocks, slippage can be 1-3%
- Re-run backtest with slippage model and report impact

### 9.2 Data Feed Reliability
- What is the data source? (merolagani, sharesansar, NEPSE API, etc.)
- How often does it update?
- Is there a backup data source?
- Test: introduce random data gaps and see how the system handles them

### 9.3 Order Types
- What order types does your broker support via API (if any)?
- Most NEPSE brokers only support market orders during trading hours
- Can you place limit orders? If not, market order slippage is higher

---

## PHASE 10: PAPER TRADING SETUP

### 10.1 Paper Trading Engine
- Set up a paper trading mode that:
  - Uses live/delayed NEPSE data
  - Applies the EXACT same signal generation, position sizing, and risk management as the backtest
  - Logs every signal, order, fill, and portfolio state
  - Uses the `TransactionCostModel` from Phase 2
  - Runs for a MINIMUM of 2 months before going live

### 10.2 Reconciliation Framework
- Every day, compare:
  - Paper trade signals vs what backtest would have generated
  - Actual fills vs expected fills
  - Paper P&L vs theoretical P&L
- Flag any discrepancy > 1%

### 10.3 Kill Switch
- Implement automatic shutdown if:
  - Daily loss exceeds X% (configurable)
  - Drawdown from peak exceeds Y% (configurable)
  - Data feed goes stale for more than Z minutes
  - Any system error occurs during order generation

---

## PHASE 11: FINAL REPORT

Generate a comprehensive report with:
1. Executive summary: go/no-go recommendation with reasoning
2. All metrics from Phase 4 in a clean table
3. Walk-forward equity curve plot
4. Monte Carlo distribution plots (terminal wealth and max drawdown)
5. Parameter sensitivity heatmaps
6. Regime performance comparison table
7. Fee impact analysis (gross vs net returns)
8. Top 5 risks identified and mitigation status
9. Checklist of all tests: PASS/FAIL/NOT APPLICABLE

---

## IMPORTANT NOTES

- Do NOT skip any section. If something is not applicable, explain why.
- If the codebase is missing critical infrastructure (e.g., no backtest engine), build it.
- All plots should be saved as PNG files in a `reports/` directory.
- All metrics should also be saved as a JSON file for programmatic access.
- Use Python for all analysis. Preferred libraries: pandas, numpy, scipy, matplotlib, seaborn, statsmodels.
- When searching for SEBON fees, use multiple sources to cross-verify. If sources conflict, flag it and use the most conservative (highest cost) estimate.
- The Nepal risk-free rate should use the latest 91-day Treasury Bill rate from Nepal Rastra Bank (nrb.org.np).
- All monetary values should be in NPR (Nepalese Rupees).
