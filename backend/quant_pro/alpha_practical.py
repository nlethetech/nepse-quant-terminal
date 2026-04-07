"""
Practical Alpha System for NEPSE.

Focuses on edges that actually exist in inefficient, retail-dominated markets:
1. Corporate Action Arbitrage (rights, bonus, dividends)
2. Fundamental Value (sector-relative P/E, P/B)
3. Momentum (trends persist in retail markets)
4. Liquidity Premium (illiquidity discount)
5. News/Sentiment Reaction

This replaces ML-based prediction with rule-based strategies that have
economic intuition and can be validated with limited data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of alpha signals."""
    CORPORATE_ACTION = "corporate_action"
    FUNDAMENTAL = "fundamental"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    DIVIDEND = "dividend"
    LIQUIDITY = "liquidity"
    SENTIMENT = "sentiment"
    QUARTERLY_FUNDAMENTAL = "quarterly_fundamental"
    XSEC_MOMENTUM = "xsec_momentum"
    ACCUMULATION = "accumulation"
    # Tier 1: Alpha Signals (Citadel upgrade)
    DISPOSITION = "disposition"
    RESIDUAL_MOMENTUM = "residual_momentum"
    LEAD_LAG = "lead_lag"
    ANCHORING_52WK = "anchoring_52wk"
    INFORMED_TRADING = "informed_trading"
    # Tier 4: Statistical Arbitrage
    PAIRS_TRADE = "pairs_trade"
    EARNINGS_DRIFT = "earnings_drift"
    # Tier 5: Alternative Data
    MACRO_REMITTANCE = "macro_remittance"
    SATELLITE_HYDRO = "satellite_hydro"
    NLP_SENTIMENT = "nlp_sentiment"
    SETTLEMENT_PRESSURE = "settlement_pressure"


@dataclass
class AlphaSignal:
    """A single alpha signal for a symbol."""
    symbol: str
    signal_type: SignalType
    direction: int  # +1 long, -1 short (not usable in NEPSE), 0 neutral
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    reasoning: str
    expires: Optional[datetime] = None  # When signal becomes stale
    target_exit_date: Optional[datetime] = None  # Event-driven exit date

    @property
    def score(self) -> float:
        """Combined score for ranking."""
        return self.direction * self.strength * self.confidence


@dataclass
class CorporateAction:
    """Corporate action event."""
    symbol: str
    action_type: str  # "rights", "bonus", "dividend", "agm", "merger"
    announce_date: datetime
    record_date: Optional[datetime] = None
    ex_date: Optional[datetime] = None
    details: str = ""
    ratio: Optional[float] = None  # e.g., 1:2 rights = 0.5


# =============================================================================
# 1. CORPORATE ACTION SIGNALS
# =============================================================================

class CorporateActionScanner:
    """
    Scan for corporate action opportunities.

    Edges:
    - Pre-rights rally (price often rises before rights issue)
    - Bonus share adjustment lag (market slow to adjust)
    - Dividend capture (buy before ex-date, sell after)
    - AGM run-up (speculation before AGM announcements)
    """

    # Typical patterns in NEPSE
    PATTERNS = {
        "rights": {
            "pre_event_days": 30,  # Rally starts ~30 days before
            "expected_move": 0.15,  # 15% typical pre-rights rally
            "confidence": 0.6,
        },
        "bonus": {
            "pre_event_days": 20,
            "expected_move": 0.10,
            "confidence": 0.5,
        },
        "dividend": {
            "pre_event_days": 10,
            "expected_move": 0.05,
            "confidence": 0.7,
        },
        "agm": {
            "pre_event_days": 15,
            "expected_move": 0.08,
            "confidence": 0.4,
        },
    }

    def __init__(self):
        self.actions: List[CorporateAction] = []

    def add_action(self, action: CorporateAction):
        """Add a corporate action to track."""
        self.actions.append(action)

    def scan(self, current_date: datetime) -> List[AlphaSignal]:
        """Scan for actionable corporate action signals."""
        signals = []

        for action in self.actions:
            pattern = self.PATTERNS.get(action.action_type)
            if not pattern:
                continue

            # Check if we're in the pre-event window
            target_date = action.record_date or action.ex_date or action.announce_date
            if not target_date:
                continue

            days_to_event = (target_date - current_date).days

            if 0 < days_to_event <= pattern["pre_event_days"]:
                # Scale strength by proximity to event
                proximity = 1 - (days_to_event / pattern["pre_event_days"])
                strength = pattern["expected_move"] * (0.5 + 0.5 * proximity)

                signals.append(AlphaSignal(
                    symbol=action.symbol,
                    signal_type=SignalType.CORPORATE_ACTION,
                    direction=1,  # Long
                    strength=min(strength, 1.0),
                    confidence=pattern["confidence"],
                    reasoning=f"{action.action_type.upper()} in {days_to_event} days: {action.details}",
                    expires=target_date,
                ))

        return signals


# =============================================================================
# 2. FUNDAMENTAL VALUE SIGNALS
# =============================================================================

@dataclass
class FundamentalData:
    """Fundamental data for a symbol."""
    symbol: str
    sector: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    eps: Optional[float] = None
    book_value: Optional[float] = None
    dividend_yield: Optional[float] = None
    roe: Optional[float] = None
    npm: Optional[float] = None  # Net profit margin
    market_cap: Optional[float] = None
    revenue_growth_qoq: Optional[float] = None
    profit_growth_qoq: Optional[float] = None
    eps_growth_qoq: Optional[float] = None
    book_value_growth_qoq: Optional[float] = None
    debt_to_equity: Optional[float] = None
    capital_adequacy_pct: Optional[float] = None
    npl_pct: Optional[float] = None
    cost_income_ratio: Optional[float] = None
    latest_net_profit: Optional[float] = None
    latest_revenue: Optional[float] = None
    data_source: Optional[str] = None


class FundamentalScanner:
    """
    Scan for fundamental mispricings.

    Strategy: Buy stocks trading at discount to sector peers.

    Edges:
    - Sector-relative P/E (stock P/E vs sector median P/E)
    - P/B below 1 with positive ROE
    - High dividend yield (>5%) with stable payout
    - Earnings growth not reflected in price
    """

    # Sector benchmark P/E ratios (NEPSE typical ranges)
    SECTOR_PE_BENCHMARKS = {
        "Commercial Banks": 15.0,
        "Development Banks": 12.0,
        "Finance": 10.0,
        "Life Insurance": 25.0,
        "Non-Life Insurance": 20.0,
        "Hydropower": 30.0,
        "Microfinance": 12.0,
        "Hotels & Tourism": 20.0,
        "Manufacturing & Processing": 18.0,
        "Others": 15.0,
    }

    def __init__(self):
        self.fundamentals: Dict[str, FundamentalData] = {}
        self.sector_medians: Dict[str, Dict[str, float]] = {}

    def update_fundamentals(self, data: FundamentalData):
        """Update fundamental data for a symbol."""
        self.fundamentals[data.symbol] = data
        self._recalculate_sector_medians()

    def _recalculate_sector_medians(self):
        """Recalculate sector median metrics."""
        sectors: Dict[str, List[FundamentalData]] = {}

        for fd in self.fundamentals.values():
            if fd.sector not in sectors:
                sectors[fd.sector] = []
            sectors[fd.sector].append(fd)

        for sector, stocks in sectors.items():
            pe_values = [s.pe_ratio for s in stocks if s.pe_ratio and s.pe_ratio > 0]
            pb_values = [s.pb_ratio for s in stocks if s.pb_ratio and s.pb_ratio > 0]
            dy_values = [s.dividend_yield for s in stocks if s.dividend_yield]

            self.sector_medians[sector] = {
                "pe": np.median(pe_values) if pe_values else self.SECTOR_PE_BENCHMARKS.get(sector, 15),
                "pb": np.median(pb_values) if pb_values else 1.5,
                "div_yield": np.median(dy_values) if dy_values else 0.03,
            }

    def scan(self) -> List[AlphaSignal]:
        """Scan for fundamental value signals."""
        signals = []

        for symbol, fd in self.fundamentals.items():
            sector_med = self.sector_medians.get(fd.sector, {})
            reasons = []
            total_score = 0.0
            penalty_score = 0.0
            signal_count = 0
            has_primary_catalyst = False
            sector_key = str(fd.sector or "").lower()
            is_lender = any(
                key in sector_key for key in ("bank", "finance", "microfinance", "insurance")
            )

            # 1. P/E discount to sector
            if fd.pe_ratio and fd.pe_ratio > 0:
                sector_pe = sector_med.get("pe", 15)
                pe_discount = (sector_pe - fd.pe_ratio) / sector_pe
                if pe_discount > 0.20:  # >20% discount
                    total_score += min(pe_discount, 0.5)
                    signal_count += 1
                    has_primary_catalyst = True
                    reasons.append(f"P/E {fd.pe_ratio:.1f} vs sector {sector_pe:.1f} ({pe_discount:.0%} discount)")

            # 2. P/B below 1 with positive ROE
            if fd.pb_ratio and fd.pb_ratio < 1.0 and fd.roe and fd.roe > 0:
                pb_score = (1.0 - fd.pb_ratio) * 0.5
                total_score += pb_score
                signal_count += 1
                has_primary_catalyst = True
                reasons.append(f"P/B {fd.pb_ratio:.2f} < 1.0 with ROE {fd.roe:.1%}")

            # 3. High dividend yield
            if fd.dividend_yield and fd.dividend_yield > 0.05:  # >5%
                dy_score = min((fd.dividend_yield - 0.05) * 5, 0.3)
                total_score += dy_score
                signal_count += 1
                has_primary_catalyst = True
                reasons.append(f"Dividend yield {fd.dividend_yield:.1%}")

            # 4. Quarterly growth acceleration
            if fd.eps_growth_qoq is not None and fd.eps_growth_qoq > 0.10:
                eps_score = min(fd.eps_growth_qoq, 0.60) * 0.35
                total_score += eps_score
                signal_count += 1
                has_primary_catalyst = True
                reasons.append(f"EPS QoQ +{fd.eps_growth_qoq:.0%}")

            if fd.profit_growth_qoq is not None and fd.profit_growth_qoq > 0.08:
                profit_score = min(fd.profit_growth_qoq, 0.60) * 0.25
                total_score += profit_score
                signal_count += 1
                has_primary_catalyst = True
                reasons.append(f"Net profit QoQ +{fd.profit_growth_qoq:.0%}")

            if fd.revenue_growth_qoq is not None and fd.revenue_growth_qoq > 0.05:
                revenue_score = min(fd.revenue_growth_qoq, 0.50) * 0.15
                total_score += revenue_score
                signal_count += 1
                has_primary_catalyst = True
                reasons.append(f"Revenue QoQ +{fd.revenue_growth_qoq:.0%}")

            # 5. Quality confirmation
            if fd.roe is not None:
                if fd.roe >= 0.15:
                    total_score += 0.15
                    signal_count += 1
                    reasons.append(f"ROE {fd.roe:.1%}")
                elif fd.roe < 0.07:
                    penalty_score += 0.12

            if fd.latest_net_profit is not None:
                if fd.latest_net_profit > 0:
                    total_score += 0.08
                    signal_count += 1
                else:
                    penalty_score += 0.18

            if fd.npl_pct is not None:
                if fd.npl_pct <= 2.5:
                    total_score += 0.12
                    signal_count += 1
                    reasons.append(f"NPL {fd.npl_pct:.1f}%")
                elif fd.npl_pct >= 5.0:
                    penalty_score += min((fd.npl_pct - 5.0) * 0.04, 0.25)

            if fd.capital_adequacy_pct is not None:
                if fd.capital_adequacy_pct >= 11.0:
                    total_score += 0.10
                    signal_count += 1
                    reasons.append(f"CAR {fd.capital_adequacy_pct:.1f}%")
                elif fd.capital_adequacy_pct < 8.0:
                    penalty_score += 0.15

            if fd.cost_income_ratio is not None:
                if fd.cost_income_ratio <= 55.0:
                    total_score += 0.06
                    signal_count += 1
                elif fd.cost_income_ratio >= 80.0:
                    penalty_score += 0.10

            if not is_lender and fd.debt_to_equity is not None:
                if fd.debt_to_equity <= 1.5:
                    total_score += 0.06
                    signal_count += 1
                elif fd.debt_to_equity >= 3.0:
                    penalty_score += 0.10

            net_score = total_score - penalty_score
            if signal_count > 0 and has_primary_catalyst and net_score > 0.18:
                avg_score = net_score / signal_count
                confidence = 0.50 + 0.05 * signal_count
                if fd.data_source and "quarterly" in fd.data_source:
                    confidence += 0.05
                signals.append(AlphaSignal(
                    symbol=symbol,
                    signal_type=SignalType.FUNDAMENTAL,
                    direction=1,
                    strength=min(avg_score, 1.0),
                    confidence=min(confidence, 0.85),
                    reasoning="; ".join(reasons),
                ))

        return signals


# =============================================================================
# 3. MOMENTUM SIGNALS
# =============================================================================

class MomentumScanner:
    """
    Scan for momentum signals.

    Strategy: Trend following works in retail-dominated markets.

    Edges:
    - Price momentum (past winners continue winning)
    - Volume breakouts (high volume = institutional interest)
    - Relative strength (outperforming sector)

    Rules:
    - Buy: Price > 20-day SMA > 50-day SMA, volume above average
    - Avoid: Price < 50-day SMA (downtrend)
    """

    def __init__(self, lookback_short: int = 20, lookback_long: int = 50):
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long

    def calculate_signals(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> List[AlphaSignal]:
        """
        Calculate momentum signals from price/volume data.

        Args:
            prices: DataFrame with columns as symbols, index as dates
            volumes: DataFrame with same structure as prices
        """
        signals = []

        for symbol in prices.columns:
            if symbol not in volumes.columns:
                continue

            price = prices[symbol].dropna()
            volume = volumes[symbol].dropna()

            if len(price) < self.lookback_long + 10:
                continue

            # Calculate indicators
            sma_short = price.rolling(self.lookback_short).mean()
            sma_long = price.rolling(self.lookback_long).mean()
            vol_avg = volume.rolling(self.lookback_short).mean()

            current_price = price.iloc[-1]
            current_sma_short = sma_short.iloc[-1]
            current_sma_long = sma_long.iloc[-1]
            current_volume = volume.iloc[-1]
            current_vol_avg = vol_avg.iloc[-1]

            # Momentum score components
            reasons = []
            strength = 0.0

            # 1. Trend alignment: Price > SMA20 > SMA50
            if current_price > current_sma_short > current_sma_long:
                trend_strength = (current_price / current_sma_long - 1)
                strength += min(trend_strength * 2, 0.4)
                reasons.append(f"Uptrend: Price > SMA20 > SMA50")
            elif current_price < current_sma_short < current_sma_long:
                # Downtrend - no signal (can't short)
                continue
            else:
                # Mixed - weak signal
                strength += 0.1

            # 2. Volume confirmation
            if current_volume > current_vol_avg * 1.5:
                strength += 0.2
                reasons.append(f"Volume {current_volume/current_vol_avg:.1f}x average")

            # 3. Rate of change (momentum)
            roc_20 = (current_price / price.iloc[-self.lookback_short] - 1)
            if roc_20 > 0.05:  # >5% gain in 20 days
                strength += min(roc_20, 0.3)
                reasons.append(f"ROC(20) = {roc_20:.1%}")

            if strength > 0.2 and reasons:
                signals.append(AlphaSignal(
                    symbol=symbol,
                    signal_type=SignalType.MOMENTUM,
                    direction=1,
                    strength=min(strength, 1.0),
                    confidence=0.5,  # Momentum is noisy
                    reasoning="; ".join(reasons),
                    expires=datetime.now() + timedelta(days=5),  # Short-lived signal
                ))

        return signals


# =============================================================================
# 4. LIQUIDITY PREMIUM SIGNALS
# =============================================================================

class LiquidityScanner:
    """
    Scan for liquidity premium opportunities.

    Strategy: Illiquid stocks trade at discount; buy when liquidity improves.

    Edges:
    - Illiquidity discount (low volume = lower price)
    - Volume spike detection (liquidity returning)
    - Bid-ask spread narrowing (if data available)
    """

    def __init__(self, min_volume_spike: float = 3.0):
        self.min_volume_spike = min_volume_spike

    def calculate_signals(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        market_caps: Dict[str, float],
    ) -> List[AlphaSignal]:
        """Find stocks with improving liquidity."""
        signals = []

        for symbol in prices.columns:
            if symbol not in volumes.columns:
                continue

            volume = volumes[symbol].dropna()
            price = prices[symbol].dropna()

            if len(volume) < 60:
                continue

            # Calculate metrics
            vol_60d_avg = volume.iloc[-60:].mean()
            vol_20d_avg = volume.iloc[-20:].mean()
            vol_5d_avg = volume.iloc[-5:].mean()
            current_volume = volume.iloc[-1]

            # Skip very illiquid stocks (< Rs 1M daily turnover)
            turnover = current_volume * price.iloc[-1]
            if turnover < 1_000_000:
                continue

            reasons = []
            strength = 0.0

            # 1. Volume spike (liquidity returning)
            if vol_5d_avg > vol_60d_avg * self.min_volume_spike:
                spike_ratio = vol_5d_avg / vol_60d_avg
                strength += min((spike_ratio - 1) * 0.15, 0.4)
                reasons.append(f"Volume spike: {spike_ratio:.1f}x 60-day avg")

            # 2. Gradual volume increase (institutional accumulation)
            if vol_20d_avg > vol_60d_avg * 1.5 and vol_5d_avg > vol_20d_avg:
                strength += 0.2
                reasons.append("Sustained volume increase")

            # 3. Turnover ratio (trading activity vs market cap)
            if symbol in market_caps and market_caps[symbol] > 0:
                turnover_ratio = (turnover * 252) / market_caps[symbol]  # Annualized
                if 0.3 < turnover_ratio < 2.0:  # Goldilocks zone
                    strength += 0.15
                    reasons.append(f"Healthy turnover ratio: {turnover_ratio:.1%}")

            if strength > 0.2 and reasons:
                signals.append(AlphaSignal(
                    symbol=symbol,
                    signal_type=SignalType.LIQUIDITY,
                    direction=1,
                    strength=min(strength, 1.0),
                    confidence=0.45,  # Liquidity signals are noisy
                    reasoning="; ".join(reasons),
                ))

        return signals


# =============================================================================
# 5. COMPOSITE SIGNAL AGGREGATOR
# =============================================================================

@dataclass
class CompositeSignal:
    """Aggregated signal from multiple sources."""
    symbol: str
    final_score: float  # -1 to +1
    confidence: float  # 0 to 1
    position_size: float  # 0 to 1 (fraction of capital)
    signals: List[AlphaSignal] = field(default_factory=list)
    reasoning: str = ""

    @property
    def action(self) -> str:
        """Get recommended action."""
        # Lowered thresholds based on backtest - single-signal symbols are valid
        if self.final_score >= 0.2 and self.confidence >= 0.35:
            return "BUY"
        elif self.final_score <= -0.3:
            return "AVOID"  # Can't short
        else:
            return "HOLD"


class AlphaAggregator:
    """
    Aggregate signals from multiple sources into final recommendations.

    Weighting (UPDATED based on backtest results 2024):
    - Liquidity/Volume: 0.40 (Sharpe 1.17, 4.91% avg return - BEST)
    - Corporate Action: 0.35 (untested but theoretically strong)
    - Fundamental: 0.15 (works but slow)
    - Momentum: 0.05 (NEGATIVE alpha in backtest - minimize)
    """

    WEIGHTS = {
        SignalType.LIQUIDITY: 0.40,        # BEST: Sharpe 1.17 in backtest
        SignalType.CORPORATE_ACTION: 0.35,
        SignalType.DIVIDEND: 0.30,         # Similar to corp action
        SignalType.FUNDAMENTAL: 0.15,
        SignalType.QUARTERLY_FUNDAMENTAL: 0.18,
        SignalType.EARNINGS_DRIFT: 0.18,   # PEAD overlay from quarterly results
        SignalType.MOMENTUM: 0.05,         # Negative alpha - minimize
        SignalType.SENTIMENT: 0.10,
        SignalType.DISPOSITION: 0.25,      # CGO breakout (Grinblatt & Han 2005)
        SignalType.LEAD_LAG: 0.20,         # Sector spillover (Hou 2007)
        SignalType.ANCHORING_52WK: 0.22,   # 52wk high proximity (George & Hwang 2004)
    }

    def __init__(self, db_path: Optional[str] = None, max_position_size: float = 0.10):
        self.db_path = db_path
        self.max_position_size = max_position_size
        self.momentum_scanner = MomentumScanner()
        self.liquidity_scanner = LiquidityScanner()

    def scan_all(self, symbols: List[str]) -> List[AlphaSignal]:
        """
        Scan all symbols using available scanners and return all signals.

        This convenience method fetches data from the database and runs
        momentum and liquidity scanners. Corporate action and fundamental
        scanners require external data sources.
        """
        all_signals = []

        if not self.db_path or not symbols:
            return all_signals

        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)

            # Build price and volume DataFrames
            prices_dict = {}
            volumes_dict = {}

            for symbol in symbols:
                try:
                    query = """
                        SELECT date, close, volume
                        FROM stock_prices
                        WHERE symbol = ?
                        ORDER BY date DESC
                        LIMIT 100
                    """
                    df = pd.read_sql_query(query, conn, params=(symbol,))
                    if len(df) >= 60:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date').set_index('date')
                        prices_dict[symbol] = df['close']
                        volumes_dict[symbol] = df['volume']
                except Exception:
                    continue

            conn.close()

            if not prices_dict:
                return all_signals

            prices = pd.DataFrame(prices_dict)
            volumes = pd.DataFrame(volumes_dict)

            # Run momentum scanner
            try:
                momentum_signals = self.momentum_scanner.calculate_signals(prices, volumes)
                all_signals.extend(momentum_signals)
            except Exception as e:
                logger.warning(f"Momentum scanner failed: {e}")

            # Run liquidity scanner
            try:
                # Estimate market caps from price * average volume
                market_caps = {}
                for symbol in prices.columns:
                    if symbol in volumes.columns:
                        avg_vol = volumes[symbol].mean()
                        last_price = prices[symbol].iloc[-1]
                        # Rough estimate: assume avg volume represents 0.1% of shares
                        market_caps[symbol] = last_price * avg_vol * 1000

                liquidity_signals = self.liquidity_scanner.calculate_signals(
                    prices, volumes, market_caps
                )
                all_signals.extend(liquidity_signals)
            except Exception as e:
                logger.warning(f"Liquidity scanner failed: {e}")

        except Exception as e:
            logger.error(f"scan_all failed: {e}")

        return all_signals

    def aggregate(self, signals: List[AlphaSignal]) -> Dict[str, CompositeSignal]:
        """Aggregate signals by symbol."""
        by_symbol: Dict[str, List[AlphaSignal]] = {}

        for signal in signals:
            if signal.symbol not in by_symbol:
                by_symbol[signal.symbol] = []
            by_symbol[signal.symbol].append(signal)

        results = {}

        for symbol, sym_signals in by_symbol.items():
            weighted_score = 0.0
            total_weight = 0.0
            confidence_sum = 0.0
            reasons = []

            for signal in sym_signals:
                weight = self.WEIGHTS.get(signal.signal_type, 0.1)
                weighted_score += signal.score * weight
                total_weight += weight
                confidence_sum += signal.confidence * weight
                reasons.append(f"[{signal.signal_type.value}] {signal.reasoning}")

            if total_weight > 0:
                final_score = weighted_score / total_weight
                avg_confidence = confidence_sum / total_weight
            else:
                final_score = 0.0
                avg_confidence = 0.0

            # Position sizing based on score and confidence
            raw_size = abs(final_score) * avg_confidence
            position_size = min(raw_size, self.max_position_size)

            results[symbol] = CompositeSignal(
                symbol=symbol,
                final_score=final_score,
                confidence=avg_confidence,
                position_size=position_size if final_score > 0 else 0.0,
                signals=sym_signals,
                reasoning="\n".join(reasons),
            )

        return results

    def get_top_picks(
        self,
        signals: Dict[str, CompositeSignal],
        n: int = 10,
        min_confidence: float = 0.4,
    ) -> List[CompositeSignal]:
        """Get top N actionable signals."""
        actionable = [
            s for s in signals.values()
            if s.action == "BUY" and s.confidence >= min_confidence
        ]

        # Sort by score * confidence
        actionable.sort(key=lambda x: x.final_score * x.confidence, reverse=True)

        return actionable[:n]


# =============================================================================
# VALIDATION FRAMEWORK
# =============================================================================

class SignalValidator:
    """
    Validate signal performance with proper methodology.

    Key principles:
    - Point-in-time data only (no lookahead)
    - Realistic transaction costs
    - Account for liquidity constraints
    - Track signal decay
    """

    def __init__(self, transaction_cost: float = 0.01):  # 1% round-trip
        self.transaction_cost = transaction_cost
        self.signal_history: List[Tuple[AlphaSignal, datetime, float]] = []

    def record_signal(self, signal: AlphaSignal, entry_price: float):
        """Record a signal for later validation."""
        self.signal_history.append((signal, datetime.now(), entry_price))

    def validate(
        self,
        prices: pd.DataFrame,
        holding_period: int = 20,
    ) -> pd.DataFrame:
        """
        Validate historical signals.

        Returns DataFrame with:
        - signal details
        - entry/exit prices
        - return (gross and net)
        - whether signal was correct
        """
        results = []

        for signal, entry_date, entry_price in self.signal_history:
            if signal.symbol not in prices.columns:
                continue

            symbol_prices = prices[signal.symbol]

            # Find entry and exit dates
            entry_idx = symbol_prices.index.get_indexer([entry_date], method='nearest')[0]
            exit_idx = min(entry_idx + holding_period, len(symbol_prices) - 1)

            if exit_idx <= entry_idx:
                continue

            exit_price = symbol_prices.iloc[exit_idx]
            gross_return = (exit_price / entry_price - 1) * signal.direction
            net_return = gross_return - self.transaction_cost

            results.append({
                "symbol": signal.symbol,
                "signal_type": signal.signal_type.value,
                "direction": signal.direction,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "correct": net_return > 0,
            })

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def performance_summary(self, validation_df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize signal performance."""
        if validation_df.empty:
            return {"error": "No validated signals"}

        return {
            "total_signals": len(validation_df),
            "win_rate": validation_df["correct"].mean(),
            "avg_return": validation_df["net_return"].mean(),
            "sharpe": (
                validation_df["net_return"].mean() /
                validation_df["net_return"].std() * np.sqrt(252 / 20)  # Annualized
                if validation_df["net_return"].std() > 0 else 0
            ),
            "best_signal_type": (
                validation_df.groupby("signal_type")["net_return"]
                .mean()
                .idxmax()
            ),
            "by_type": validation_df.groupby("signal_type").agg({
                "net_return": ["mean", "std", "count"],
                "correct": "mean",
            }).to_dict(),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_alpha_system() -> Tuple[
    CorporateActionScanner,
    FundamentalScanner,
    MomentumScanner,
    LiquidityScanner,
    AlphaAggregator,
    SignalValidator,
]:
    """Create all components of the alpha system."""
    return (
        CorporateActionScanner(),
        FundamentalScanner(),
        MomentumScanner(),
        LiquidityScanner(),
        AlphaAggregator(),
        SignalValidator(),
    )


__all__ = [
    "SignalType",
    "AlphaSignal",
    "CorporateAction",
    "CorporateActionScanner",
    "FundamentalData",
    "FundamentalScanner",
    "MomentumScanner",
    "LiquidityScanner",
    "CompositeSignal",
    "AlphaAggregator",
    "SignalValidator",
    "create_alpha_system",
]
