"""
Configuration constants for risk, drift, and utility settings.
UPDATED: Quarterly Institutional Settings (63 Days)
"""
import os

# Trading days
NEPSE_TRADING_DAYS = 240  
PERIODS_PER_YEAR = NEPSE_TRADING_DAYS

# Risk-free rate (91-day NRB T-bill rate, Jan 2026)
R_F_ANNUAL = 0.0235

# Drift & Utility
DRIFT_SHRINKAGE_ALPHA = 0.05
RECENT_DRIFT_WINDOW = 126
LONG_DRIFT_WINDOW = 756
RISK_AVERSION_LAMBDA = 1.5  # Conservative for competition (was 0.35)
NO_TRADE_UTILITY_THRESHOLD = 0.005

# Limits
MAX_POS_ER = 0.60
MAX_NEG_ER = -0.40
MAX_DRAWDOWN_THRESHOLD = -0.30

# System
BLOCK_BOOTSTRAP_SIZE = 30
STOCH_VOL_LOOKBACK = 252
FACTOR_CV_FOLDS = 5
AUTOCORRELATION_WINDOW = 21

# Stress
ES_PENALTY_STRESSED = 1.15
STRESS_ALLOC_CAP = 1.0

# === TRIPLE BARRIER CONFIGURATION (YIELD HUNTER MODE) ===
# Horizon: 63 Trading Days (~1 Quarter)
TRIPLE_BARRIER_HORIZON = 63

# Profit Target: tightened for NEPSE microstructure.
TRIPLE_BARRIER_PT_MULT = 0.9

# Stop Loss: tighter to avoid over-wide barriers on thin names.
TRIPLE_BARRIER_SL_MULT = 1.7

# === LABELING / CV CONTROLS ===
# Supported:
# - "triple_barrier": path-dependent TP/SL label
# - "fwd_ret_quantile": forward return quantile buckets
# - "alpha_vs_market": benchmark-relative (NEPSE) forward alpha rank label
LABEL_MODE = "triple_barrier"
QUANTILE_TOP = 0.30
QUANTILE_BOTTOM = 0.30
QUANTILE_DEADZONE = 0.10
TARGET_POSITIVE_RATE_LO = 0.35
TARGET_POSITIVE_RATE_HI = 0.65
LABEL_TUNING_PATH = "artifacts/label_tuning.json"
TUNE_LABELS = True
CLASS_FLOOR = 0.05  # minimum class fraction per fold before skipping
EMBARGO_HORIZON_MULT = 1.75  # stronger purge to reduce echo leakage

# === EMBARGO TUNING BY HORIZON ===
# Configurable purge/embargo gaps for different label horizons.
# Keys are horizon days, values are {"purge": days, "embargo": days}.
# - purge: samples removed BEFORE test set (prevent label overlap)
# - embargo: samples removed AFTER test set (prevent target leakage)
EMBARGO_BY_HORIZON = {
    5: {"purge": 5, "embargo": 8},      # Ultra-short momentum
    10: {"purge": 10, "embargo": 15},   # Short-term momentum
    21: {"purge": 21, "embargo": 30},   # Monthly rebalance
    42: {"purge": 42, "embargo": 55},   # 2-month tactical
    63: {"purge": 63, "embargo": 90},   # Quarterly institutional
}

# CPCV (Combinatorially Purged Cross-Validation) settings
CPCV_ENABLED = True
CPCV_MIN_SAMPLES_FOR_CPCV = 600  # Minimum samples to use CPCV
CPCV_N_SPLITS = 5
CPCV_PURGE_PCT = 0.01  # 1% of samples as purge window
CPCV_EMBARGO_PCT = 0.02  # 2% of samples as embargo window

# === FEATURE ENGINEERING FLAGS ===
USE_TRUE_XSEC = False  # when True, require cross-sectional panel for ranks/z-scores
LEGACY_PSEUDO_XSEC_TAG = True
USE_L1_FEATURE_SELECTION = True
PSEUDO_XSEC_WEIGHT = 0.5

# === CALIBRATION / PROBABILITY HYGIENE ===
USE_GLOBAL_CALIBRATION = True
CALIBRATION_EXPERIMENT_ENABLED = False
CALIBRATION_EXPERIMENT_OUTPUT = "artifacts/calibration_experiment_summary.json"
CALIBRATION_EXPERIMENT_ON_LABEL = "calibration_on"
CALIBRATION_EXPERIMENT_OFF_LABEL = "calibration_off"
GLOBAL_CAL_MIN_POOLED_OOS_DEFAULT = 1000
GLOBAL_CAL_MIN_POOLED_OOS_MAX = 1000
GLOBAL_CAL_MIN_SYM_OOS = 80
GLOBAL_CAL_STRICT_SYMBOL_REQUIREMENT = True
GLOBAL_CAL_FORCE_PLATT = True
GLOBAL_CAL_ISO_MIN_POOLED = 1200
GLOBAL_CAL_ISO_MIN_SYMBOL_OOS = 100
GLOBAL_CAL_PLATT_MIN_POOLED = 50
GLOBAL_CALIB_MIN_OOS = 80
GLOBAL_CALIB_MIN_OOS_RELAXED = 60
GLOBAL_CALIB_RELAXED_WEIGHT = 0.5
SHRINKER_MIN = 0.25
SHRINKER_MAX = 1.0
SHRINKER_VOL_CAP = 0.04  # optional cap for extreme CV std
SHRINKER_AUC_FLOOR = 0.55
SHRINKER_AUC_NEUTRAL = 0.50
SHRINKER_STD_LOW = 0.18
SHRINKER_STD_HIGH = 0.35
BASE_CAP_LOW = 0.05
BASE_CAP_HIGH = 0.08
MAX_CAP_SOFT = 0.12
AUC_TRUST_THRESHOLD = 0.60
RELIABILITY_MIN_CORR = 0.15
RELIABILITY_MIN_BINS_WITH_VARIATION = 3

# === LABEL BALANCE / FALLBACK ===
LABEL_POS_RATE_MIN = 0.10
LABEL_POS_RATE_MAX = 0.90
LABEL_BALANCE_ENFORCE = True
LABEL_QUANTILE_FALLBACK_ENABLED = True
# Benchmark-relative (alpha vs market) label settings
ALPHA_LABEL_LOOKBACK = 252
ALPHA_LABEL_RANK_WINDOW = 252
ALPHA_LABEL_TOP_Q = 0.80
ALPHA_LABEL_BOTTOM_Q = 0.20
ALPHA_LABEL_MIN_HISTORY = 120
ALPHA_LABEL_COST_BPS_ROUNDTRIP = 30.0
ALPHA_LABEL_CROSS_SECTIONAL = True
ALPHA_LABEL_XSEC_MIN_NAMES = 2
# NEPSE microstructure-aware alpha label cost model (clipped).
ALPHA_LABEL_COST_BASE_BPS = 20.0
ALPHA_LABEL_COST_LIQUIDITY_K = 25.0
ALPHA_LABEL_COST_ZERO_VOL_K = 80.0
ALPHA_LABEL_COST_MIN_BPS = 15.0
ALPHA_LABEL_COST_MAX_BPS = 120.0
RUN_CALIBRATION_EXPERIMENT = False
ALLOW_SIGNAL_DEPLOYMENT = False
DEPLOYMENT_HEALTH_CHECK_REQUIRED = True
DEPLOYMENT_GATE_PATH = os.environ.get("DEPLOYMENT_GATE_PATH", "artifacts/deployment_gate.json")
DEPLOYMENT_GATE_DEFAULTS = {
    # Overall batch health
    "min_accepted_symbols": 10,
    "min_accepted_pct": 0.60,
    "max_reject_rate": 0.35,
    "min_avg_cv_score": 0.53,
    "min_median_cv_score": 0.52,
    # Edge realism
    "min_avg_prob_edge": 0.004,
    "min_median_prob_edge": 0.002,
    # Calibration / reliability
    "min_reliability_corr": RELIABILITY_MIN_CORR,
    "min_reliability_bins": RELIABILITY_MIN_BINS_WITH_VARIATION,
    # Label sanity
    "max_label_outliers": 10,
    "max_label_outliers_pct": 0.25,
}
FORCE_QUANTILE_SYMBOLS = ["ADBL", "KBL", "NBL"]
SPARSE_FEATURE_MIN_NON_NULL = 0.18  # allow more features; avoid over-pruning
FEATURE_RATIO_MAX = 0.08
TREE_MAX_DEPTH = 3
TREE_MIN_SAMPLES_LEAF = 70
# ENSEMBLE_SEEDS: Configurable via NEPSE_ENSEMBLE_SEEDS env var (comma-separated ints)
# Example: export NEPSE_ENSEMBLE_SEEDS="0,42,101,202,303" for 5-seed ensemble
_env_seeds = os.environ.get("NEPSE_ENSEMBLE_SEEDS", "")
if _env_seeds.strip():
    try:
        ENSEMBLE_SEEDS = [int(s.strip()) for s in _env_seeds.split(",") if s.strip()]
    except ValueError:
        ENSEMBLE_SEEDS = [0, 101, 202]  # fallback on parse error
else:
    ENSEMBLE_SEEDS = [0, 101, 202]
SECTOR_HYPERPARAMS = {
    "Banking": {"max_depth": 3, "learning_rate": 0.03, "min_samples_leaf": 60, "l2_regularization": 6.0},
    "Hydropower": {"max_depth": 3, "learning_rate": 0.04, "min_samples_leaf": 70, "l2_regularization": 7.0},
    "default": {"max_depth": TREE_MAX_DEPTH, "learning_rate": 0.035, "min_samples_leaf": TREE_MIN_SAMPLES_LEAF, "l2_regularization": 4.0},
}

# === LIQUIDITY GATES ===
MIN_MEDIAN_SHARE_VOLUME = 1000  # shares traded
MIN_MEDIAN_TURNOVER = 200000    # NPR traded
MIN_NON_ZERO_RET_RATIO = 0.5

# === DYNAMIC BARRIERS / ROBUST CV ===
USE_DYNAMIC_BARRIERS = True
DYNAMIC_BARRIER_ATR_WINDOW = 60
DYNAMIC_BARRIER_PT_SIGMA = 1.6
DYNAMIC_BARRIER_SL_SIGMA = 2.4
DYNAMIC_BARRIER_MIN_PT = 0.01
DYNAMIC_BARRIER_MAX_PT = 0.35
DYNAMIC_BARRIER_MIN_SL = 0.02
DYNAMIC_BARRIER_MAX_SL = 0.55
DYNAMIC_BARRIER_REGIME_SCALER = 0.15
USE_GMM_REGIME_WEIGHTING = True

USE_ORTHOGONAL_FEATURES = True
ORTHOGONAL_VARIANCE = 0.95

USE_CPCV = True
CPCV_PARTITIONS = 6
CPCV_TEST_BLOCKS = 2
CPCV_MIN_SAMPLES = 180
CPCV_MAX_COMBOS = 20

USE_META_LABELING = True
META_MIN_SAMPLES = 80
META_MIN_POS_CLASS = 12
META_MIN_NEG_CLASS = 12

# === ROBUST FEATURE SELECTION ===
USE_PERMUTATION_IMPORTANCE = True
PERM_IMPORTANCE_THRESHOLD = 0.01
PERM_IMPORTANCE_FOLDS = 3
PERM_IMPORTANCE_REPEATS = 5

# === RELATIVE LIQUIDITY FILTERS ===
RELATIVE_VOLUME_TOP_PCT = 0.40       # require symbol median volume above 60th percentile
RELATIVE_LIQUIDITY_TOP_PCT = 0.30    # require turnover above 70th percentile (top 30%)
RELATIVE_ZERO_VOL_PCT = 0.70         # allow names up to 70th percentile of zero-volume days

# === PROBABILITY EDGE CONTROLS ===
EDGE_MIN_SAMPLES = 200
EDGE_MIN_SAMPLES_SYMBOL = 100
EDGE_SAMPLES_TARGET = 1200
EDGE_MAX = 0.05
EDGE_BRIER_WEIGHT = 0.65
EDGE_WARN_HIGH = 0.025
EDGE_WARN_LOW = 0.005
EDGE_BOOTSTRAP_N = 200
# EDGE_BOOTSTRAP_SEED: Configurable via NEPSE_EDGE_SEED env var
EDGE_BOOTSTRAP_SEED = int(os.environ.get("NEPSE_EDGE_SEED", "42"))
EDGE_CI_ALPHA = 0.10
EDGE_MIN_REGIME_SAMPLES = 120

# === POST-CV VALIDATION THRESHOLDS ===
POSTVAL_MIN_TRADES = 100
POSTVAL_MIN_SHARPE = 1.0
POSTVAL_MIN_WIN_RATE = 0.50
POSTVAL_MIN_CAGR = -0.02
POSTVAL_MIN_PSR = 0.90
POSTVAL_SHARPE_METRIC = "hac"  # {"hac", "standard"}
POSTVAL_SHARPE_BENCHMARK = 0.0
POSTVAL_HAC_LAGS = 5
POSTVAL_THRESHOLD_MIN = 0.55
POSTVAL_THRESHOLD_MAX = 0.70
POSTVAL_THRESHOLD_STEP = 0.02
POSTVAL_MIN_SIGNAL_STD = 0.04
POSTVAL_MAX_ADF_PVALUE = 0.15
POSTVAL_SLIPPAGE_BASE_BPS = 15.0
POSTVAL_SLIPPAGE_LIQUIDITY_K = 35.0
POSTVAL_SLIPPAGE_ZERO_VOL_K = 50.0
POSTVAL_SLIPPAGE_CAP_BPS = 120.0
POSTVAL_RETURN_KIND = "auto"  # {"simple","percent","log","auto"}

# === WALK-FORWARD PRODUCTION PIPELINE ===
# Walk-forward label mode: defaults to LABEL_MODE unless overridden.
WFV_LABEL_MODE = LABEL_MODE
WFV_LABEL_HORIZON = TRIPLE_BARRIER_HORIZON
WFV_MIN_HISTORY_DAYS = 756  # ~3 years
WFV_TRAIN_MIN_DAYS = 756
WFV_TRAIN_WINDOW_DAYS = None  # None = expanding
WFV_EVAL_WINDOW_DAYS = 126
WFV_STEP_DAYS = 126
WFV_CALIBRATION_HOLDOUT_PCT = 0.2
WFV_MIN_TRAIN_SAMPLES = 200
WFV_MIN_CALIB_SAMPLES = 80
WFV_MIN_POS_CLASS = 15
WFV_MIN_NEG_CLASS = 15
WFV_TUNE_THRESHOLD = False
WFV_THRESHOLD_DEFAULT = 0.55
WFV_THRESHOLD_GRID = [0.5, 0.55, 0.6, 0.65]
WFV_THRESHOLD_METRIC = "balanced_accuracy"
WFV_MIN_TRADES = 50
WFV_MIN_ACTIVE_DAYS = 40
WFV_MIN_COVERAGE = 0.6
WFV_MIN_SHARPE_DAILY = 0.8
WFV_MIN_PSR = 0.9
WFV_MIN_AUC = 0.55
WFV_MIN_YEARS = 2
WFV_MIN_POSITIVE_YEARS = 2
WFV_MIN_CAPACITY_TRADE_VALUE = 250000.0
WFV_SLIPPAGE_BPS_DEFAULT = 30.0
WFV_SLIPPAGE_BPS_STRESS = [15.0, 30.0, 50.0]
WFV_THRESHOLD_STRESS = [0.02, -0.02]
WFV_WINDOW_STRESS = [84, 126, 189]
# WFV_MODEL_RANDOM_STATE: Configurable via NEPSE_WFV_SEED env var
WFV_MODEL_RANDOM_STATE = int(os.environ.get("NEPSE_WFV_SEED", "42"))
WFV_CONF_BASE = 1.0
WFV_CONF_PENALTY_LEAKAGE = 0.5
WFV_CONF_PENALTY_COVERAGE = 0.4
WFV_CONF_PENALTY_CALIB = 0.3
WFV_CONF_PENALTY_STABILITY = 0.3
WFV_DISABLE_META_BY_DEFAULT = True
WFV_INTEGRATE_WITH_CV = True

# === GATING + PROMOTION FRAMEWORK ===
# Signal quality gates for deployment decisions.
# These thresholds determine when a symbol/strategy is ready for live trading.

# Deployment gate thresholds
DEPLOY_GATE = {
    # Discrimination thresholds
    "min_cv_score": 0.58,        # Minimum CV AUC/accuracy
    "max_cv_std": 0.15,          # Maximum CV fold standard deviation

    # Calibration thresholds
    "min_prob_edge": 0.008,      # Minimum probability edge
    "min_reliability_corr": 0.20, # Minimum reliability correlation

    # Stability thresholds
    "min_psr": 0.90,             # Minimum Probabilistic Sharpe Ratio
    "min_dsr": 0.85,             # Minimum Deflated Sharpe Ratio (multiple testing)

    # Capacity thresholds
    "max_annual_turnover": 24.0,  # Maximum 2x monthly turnover
    "min_median_turnover_npr": 500_000,  # Minimum median daily turnover

    # Data quality thresholds
    "min_years_history": 2.0,    # Minimum years of price history
    "max_zero_vol_pct": 0.15,    # Maximum zero volume days
}

# Promotion tiers with requirements
# Each tier has increasingly strict requirements
PROMOTION_TIERS = {
    "paper": {
        "description": "Paper trading only",
        "requirements": {
            "min_cv_score": 0.55,
            "min_prob_edge": 0.005,
            "min_psr": 0.85,
        },
    },
    "research": {
        "description": "Small live allocation (10-25% of target)",
        "requirements": {
            "min_cv_score": 0.58,
            "min_prob_edge": 0.008,
            "min_cv_std": 0.15,
            "min_psr": 0.90,
        },
    },
    "production": {
        "description": "Full allocation (requires 6mo paper track record)",
        "requirements": {
            "min_cv_score": 0.62,
            "min_prob_edge": 0.012,
            "min_cv_std": 0.12,
            "min_psr": 0.92,
            "min_dsr": 0.85,
            "paper_track_months": 6,
        },
    },
}

# Allocation multipliers by tier
TIER_ALLOCATION_MULT = {
    "paper": 0.0,       # No live allocation
    "research": 0.25,   # 25% of target
    "production": 1.0,  # Full target
}

# Sector-specific gate relaxations (some sectors have structural noise)
SECTOR_GATE_RELAXATION = {
    "Hydropower": {"min_cv_score": 0.53, "max_cv_std": 0.18},
    "Microfinance": {"min_cv_score": 0.53, "max_cv_std": 0.18},
    "Hotels": {"min_cv_score": 0.52, "max_cv_std": 0.20},
}


def get_sector_gates(sector_name: str) -> dict:
    """Get deployment gates for a specific sector, applying relaxations."""
    gates = DEPLOY_GATE.copy()
    if sector_name in SECTOR_GATE_RELAXATION:
        gates.update(SECTOR_GATE_RELAXATION[sector_name])
    return gates


def check_deployment_gate(
    cv_score: float,
    cv_std: float,
    prob_edge: float,
    psr: float,
    sector_name: str = None,
) -> tuple:
    """
    Check if a symbol passes deployment gates.

    Returns:
        (passed: bool, tier: str, reasons: list)
    """
    gates = get_sector_gates(sector_name) if sector_name else DEPLOY_GATE
    reasons = []

    # Check each gate
    if cv_score < gates.get("min_cv_score", 0.58):
        reasons.append(f"cv_score {cv_score:.3f} < {gates['min_cv_score']}")
    if cv_std > gates.get("max_cv_std", 0.15):
        reasons.append(f"cv_std {cv_std:.3f} > {gates['max_cv_std']}")
    if prob_edge < gates.get("min_prob_edge", 0.008):
        reasons.append(f"prob_edge {prob_edge:.4f} < {gates['min_prob_edge']}")
    if psr < gates.get("min_psr", 0.90):
        reasons.append(f"psr {psr:.3f} < {gates['min_psr']}")

    # Determine tier
    if not reasons:
        tier = "production"
    elif cv_score >= gates.get("min_cv_score", 0.58) * 0.95:
        tier = "research"
    elif cv_score >= gates.get("min_cv_score", 0.58) * 0.90:
        tier = "paper"
    else:
        tier = "rejected"

    passed = len(reasons) == 0
    return passed, tier, reasons


# === CROSS-SECTIONAL PORTFOLIO WALK-FORWARD ===
PORTFOLIO_TOP_K = 5
PORTFOLIO_REBALANCE_DAYS = 1
PORTFOLIO_HOLD_DAYS = 1
PORTFOLIO_LONG_ONLY = True
PORTFOLIO_MAX_WEIGHT = 0.25
PORTFOLIO_BASE_SLIPPAGE_BPS = 20.0
PORTFOLIO_SLIPPAGE_K = 50.0
PORTFOLIO_MIN_ADV_VALUE = 50000.0
PORTFOLIO_TUNE_HORIZONS = [5, 10, 20]
PORTFOLIO_TUNE_TOPK = [3, 5, 7]
PORTFOLIO_TUNE_HOLD_DAYS = [1, 5, 10]
PORTFOLIO_TUNE_MAX_DEPTH = [2, 3]

# === DYNAMIC HORIZON ADJUSTMENT ===
HORIZON_MIN = 10
HORIZON_MAX = 126
HORIZON_LIQUIDITY_MAX_MULT = 2.0
HORIZON_ZERO_VOL_WEIGHT = 0.8
HORIZON_HALF_LIFE_MIN = 5
HORIZON_HALF_LIFE_MAX = 126
HORIZON_HALFLIFE_WEIGHT = 0.7
SECTOR_HORIZON_MULT = {
    "Hydropower": 1.6,
    "Microfinance": 1.4,
}
HORIZON_SECTOR_MAX_MULT = 2.5

# === PRODUCTION UNIVERSE FILTER ===
PROD_UNIVERSE_TOP_N = 40
PROD_UNIVERSE_MIN_MEDIAN_TURNOVER = 2500000
PROD_UNIVERSE_MAX_ZERO_VOL_PCT = 0.30

# === HIGH-CONFIDENCE THRESHOLD FILTER ===
PROD_MIN_VALIDATION_THRESHOLD = 0.62

# === MODEL QUALITY GATES ===
QUALITY_MAX_CV_STD = 0.08
QUALITY_MIN_CALIBRATION = 0.05
LEAKAGE_MAX_RATIO = 1.05

# === BROKER FLOW FEATURES ===
# Enable broker flow features from floorsheet data (requires data collection)
USE_BROKER_FLOW_FEATURES = False  # Disabled - no broker data source
BROKER_FLOW_LOOKBACK_SHORT = 5   # Short-term momentum window
BROKER_FLOW_LOOKBACK_LONG = 21   # Z-score normalization window
USE_SMART_MONEY_FEATURES = False  # Requires 1+ year of historical data

# === CORPORATE ACTION / BAD PRINT FILTER (NEPSE) ===
# If your price series are not fully adjusted for bonus/right/dividend events,
# these jumps can dominate labels and create spurious "alpha". When enabled,
# labels/forward returns that span a jump window are dropped (set to NaN).
CORP_ACTION_FILTER_ENABLED = True
CORP_ACTION_ABS_RET_THRESHOLD = 0.25
CORP_ACTION_Z_THRESHOLD = 8.0
CORP_ACTION_WINDOW_DAYS = 2

# === PORTFOLIO-FIRST GATING (Option B) ===
# "symbol": use per-symbol post-validation as a hard deploy gate (legacy behavior)
# "portfolio": do not reject symbols solely on per-symbol validation; use portfolio metrics as the gate
PORTFOLIO_GATE_MODE = "symbol"

# === PORTFOLIO-FIRST GLOBAL PANEL (Option C) ===
# When enabled, train a single pooled panel model across the whole universe (all symbols),
# then evaluate/deploy using rank-based top-K selection and portfolio-level excess metrics vs NEPSE.
PORTFOLIO_FIRST_GLOBAL_PANEL = False
PORTFOLIO_FIRST_TARGET = "classification"  # classification | fwd_alpha_reg
PORTFOLIO_FIRST_TARGET_CANDIDATES = ["classification", "fwd_alpha_reg"]
PORTFOLIO_FIRST_TOP_K = 10
PORTFOLIO_FIRST_MIN_NAMES = 5
PORTFOLIO_FIRST_MAX_WEIGHT = 0.15
PORTFOLIO_FIRST_REBALANCE_DAYS = None  # None => use label horizon
PORTFOLIO_FIRST_MIN_SIGNAL_EDGE = 0.0  # rank-based; keep at 0 unless you need a confidence buffer
PORTFOLIO_FIRST_MIN_SCORE_SPREAD = 0.0  # do-not-trade filter: avg(topK) - median(universe) (score units)
PORTFOLIO_FIRST_REBALANCE_CANDIDATES = [5, 10, 21]
PORTFOLIO_FIRST_SCORE_SPREAD_CANDIDATES = [0.0, 0.001, 0.002]
PORTFOLIO_FIRST_SCORE_SPREAD_CANDIDATES_REG = [0.0, 0.002, 0.004, 0.006]
PORTFOLIO_FIRST_WEIGHT_SCHEMES = ["equal", "inv_vol"]
PORTFOLIO_FIRST_EXPOSURE_MODES = ["skip", "scale", "regime_skip", "regime_scale"]
PORTFOLIO_FIRST_SPREAD_FULL_MULT = 3.0
PORTFOLIO_FIRST_TURNOVER_PENALTY = 0.05
PORTFOLIO_FIRST_MAX_TOPK_FRAC = 0.25
PORTFOLIO_FIRST_REQUIRE_LS_POSITIVE = True
PORTFOLIO_FIRST_TRAIN_ON = "eligible"  # eligible | accepted
PORTFOLIO_FIRST_LIQUIDITY_WEIGHTING = False
PORTFOLIO_FIRST_LIQ_WEIGHT_POW = 0.5
PORTFOLIO_FIRST_LIQ_WEIGHT_ZERO_POW = 1.0
PORTFOLIO_FIRST_LIQ_WEIGHT_MIN = 0.25
PORTFOLIO_FIRST_LIQ_WEIGHT_MAX = 4.0
PORTFOLIO_FIRST_SECTOR_NEUTRAL = False
PORTFOLIO_FIRST_SECTOR_MAX_WEIGHT = 0.35
PORTFOLIO_FIRST_TRADE_TOP_N = None
PORTFOLIO_FIRST_INV_VOL_LOOKBACK = 20
PORTFOLIO_FIRST_INV_VOL_EPS = 1e-6
PORTFOLIO_FIRST_LIQ_LOOKBACK_DAYS = 90
PORTFOLIO_FIRST_MIN_MEDIAN_TURNOVER = 1_500_000.0
PORTFOLIO_FIRST_MAX_ZERO_VOL_PCT = 0.40

# Turnover / stability controls (portfolio-first execution)
PORTFOLIO_FIRST_RANK_BUFFER = 0
PORTFOLIO_FIRST_RANK_BUFFER_CANDIDATES = [0, 3]
PORTFOLIO_FIRST_MIN_HOLD_BARS = 0
PORTFOLIO_FIRST_MIN_HOLD_BARS_CANDIDATES = [0, 10]

# Benchmark regime filter (long-only): risk-off exposure when benchmark is in downtrend.
PORTFOLIO_FIRST_REGIME_MA_LONG = 200
PORTFOLIO_FIRST_REGIME_MA_LONG_CANDIDATES = [120, 200]
PORTFOLIO_FIRST_REGIME_MA_SHORT = 50
PORTFOLIO_FIRST_REGIME_MA_SHORT_CANDIDATES = [30, 50]
PORTFOLIO_FIRST_RISK_OFF_EXPOSURE = 0.0
PORTFOLIO_FIRST_RISK_OFF_EXPOSURE_CANDIDATES = [0.0, 0.25]

# Realistic cost knobs (match integrated_strategy_config_realistic.json defaults)
PORTFOLIO_FIRST_BUY_COST = 0.0
PORTFOLIO_FIRST_SELL_COST = 0.0
PORTFOLIO_FIRST_TAX_RATE = 0.0

# Portfolio deploy thresholds (excess vs NEPSE)
PORTFOLIO_FIRST_MIN_EXCESS_SHARPE = 0.30
PORTFOLIO_FIRST_MIN_EXCESS_PSR = 0.90
PORTFOLIO_FIRST_MIN_EXCESS_CAGR = 0.00
PORTFOLIO_FIRST_MIN_DAYS = 500

# === ROBUST CV SCORE SETTINGS ===
CV_SCORE_WEIGHTS = {
    "discrimination": 0.35,
    "calibration": 0.25,
    "stability": 0.15,
    "regime": 0.05,
    "leakage": 0.05,
    "label_balance": 0.10,
    "sample": 0.05,
}
CV_STABILITY_TARGET_STD = 0.12
CV_REGIME_NEUTRAL = 0.50
CV_LEAKAGE_WARN = 1.15
CV_LEAKAGE_HARD = 1.35
CV_SAMPLE_TARGET = 400
CV_MIN_OOS_SAMPLES = 120
CV_LABEL_BALANCE_MIN = 0.35
CV_MAX_WITH_LOW_SAMPLE = 0.55
CV_MAX_WITH_BAD_BALANCE = 0.55
CV_LEAKAGE_PENALTY_STRENGTH = 0.7

# === NEPSE PROFILE OVERRIDES ===
NEPSE_PROFILE_CONFIGS = {
    "nepse_research": {
        "LABEL_MODE": "alpha_vs_market",
        "TRIPLE_BARRIER_HORIZON": 21,
        "CV_STABILITY_TARGET_STD": 0.30,
        "QUALITY_MAX_CV_STD": 0.35,
        "PORTFOLIO_GATE_MODE": "portfolio",
        "WFV_LABEL_HORIZON": 21,
        "WFV_LABEL_MODE": "alpha_vs_market",
        "WFV_MIN_HISTORY_DAYS": 360,
        "WFV_TRAIN_MIN_DAYS": 252,
        "WFV_EVAL_WINDOW_DAYS": 84,
        "WFV_MIN_TRADES": 30,
        "WFV_MIN_ACTIVE_DAYS": 25,
        "WFV_MIN_SHARPE_DAILY": 0.6,
        "WFV_MIN_AUC": 0.52,
        "WFV_MIN_YEARS": 1,
        "WFV_MIN_POSITIVE_YEARS": 1,
        "POSTVAL_MIN_TRADES": 30,
        "POSTVAL_MIN_SHARPE": 0.6,
        "POSTVAL_MIN_PSR": 0.80,
        "POSTVAL_MIN_SIGNAL_STD": 0.025,
        "PROD_UNIVERSE_MIN_MEDIAN_TURNOVER": 1000000,
        "PROD_UNIVERSE_MAX_ZERO_VOL_PCT": 0.50,
        "PROD_MIN_VALIDATION_THRESHOLD": 0.58,
    },
    "nepse_prod": {
        "LABEL_MODE": "alpha_vs_market",
        "TRIPLE_BARRIER_HORIZON": 63,
        "CV_STABILITY_TARGET_STD": 0.22,
        "QUALITY_MAX_CV_STD": 0.28,
        "PORTFOLIO_GATE_MODE": "symbol",
        "WFV_LABEL_HORIZON": 63,
        "WFV_LABEL_MODE": "alpha_vs_market",
        "WFV_MIN_HISTORY_DAYS": 630,
        "WFV_TRAIN_MIN_DAYS": 504,
        "WFV_EVAL_WINDOW_DAYS": 126,
        "WFV_MIN_TRADES": 50,
        "WFV_MIN_ACTIVE_DAYS": 35,
        "WFV_MIN_SHARPE_DAILY": 0.8,
        "WFV_MIN_AUC": 0.55,
        "WFV_MIN_YEARS": 2,
        "WFV_MIN_POSITIVE_YEARS": 2,
        "POSTVAL_MIN_TRADES": 50,
        "POSTVAL_MIN_SHARPE": 0.8,
        "POSTVAL_MIN_PSR": 0.88,
        "POSTVAL_MIN_SIGNAL_STD": 0.035,
        "PROD_UNIVERSE_MIN_MEDIAN_TURNOVER": 1500000,
        "PROD_UNIVERSE_MAX_ZERO_VOL_PCT": 0.40,
        "PROD_MIN_VALIDATION_THRESHOLD": 0.62,
    },
    "nepse_portfolio_alpha": {
        # Portfolio-first, benchmark-relative alpha research/deploy profile.
        "LABEL_MODE": "alpha_vs_market",
        "TRIPLE_BARRIER_HORIZON": 21,
        "ALPHA_LABEL_CROSS_SECTIONAL": True,
        "ALPHA_LABEL_XSEC_MIN_NAMES": 5,
        "USE_TRUE_XSEC": True,

        # Expand production universe so portfolio-first has a real cross-section of liquid names.
        # (prod filter is still applied; this just increases the top-N pool.)
        "PROD_UNIVERSE_TOP_N": 120,
        "PROD_UNIVERSE_MIN_MEDIAN_TURNOVER": 1_500_000.0,
        "PROD_UNIVERSE_MAX_ZERO_VOL_PCT": 0.40,

        # Prefer portfolio gate over per-symbol validation sparsity.
        "PORTFOLIO_GATE_MODE": "portfolio",

        # Global pooled model + rank-based top-K selection.
        "PORTFOLIO_FIRST_GLOBAL_PANEL": True,
        # Train both classification and fwd-alpha regression; choose by portfolio metrics each run.
        "PORTFOLIO_FIRST_TARGET": "auto",
        "PORTFOLIO_FIRST_TARGET_CANDIDATES": ["classification", "fwd_alpha_reg"],
        "PORTFOLIO_FIRST_TOP_K": 5,
        "PORTFOLIO_FIRST_MIN_NAMES": 3,
        "PORTFOLIO_FIRST_MAX_WEIGHT": 0.25,
        # Weekly/biweekly execution tends to survive NEPSE frictions better than daily.
        "PORTFOLIO_FIRST_REBALANCE_DAYS": 10,
        "PORTFOLIO_FIRST_REBALANCE_CANDIDATES": [5, 10],
        "PORTFOLIO_FIRST_SCORE_SPREAD_CANDIDATES": [0.0, 0.001, 0.002],
        "PORTFOLIO_FIRST_SCORE_SPREAD_CANDIDATES_REG": [0.0, 0.002, 0.004, 0.006],
        "PORTFOLIO_FIRST_WEIGHT_SCHEMES": ["equal", "inv_vol"],
        "PORTFOLIO_FIRST_EXPOSURE_MODES": ["skip", "scale", "regime_skip", "regime_scale"],
        "PORTFOLIO_FIRST_SPREAD_FULL_MULT": 3.0,
        "PORTFOLIO_FIRST_TURNOVER_PENALTY": 0.12,
        "PORTFOLIO_FIRST_MAX_TOPK_FRAC": 0.20,
        "PORTFOLIO_FIRST_REQUIRE_LS_POSITIVE": True,
        "PORTFOLIO_FIRST_TRAIN_ON": "eligible",
        "PORTFOLIO_FIRST_LIQUIDITY_WEIGHTING": True,
        "PORTFOLIO_FIRST_LIQ_WEIGHT_POW": 0.6,
        "PORTFOLIO_FIRST_LIQ_WEIGHT_ZERO_POW": 1.0,
        "PORTFOLIO_FIRST_LIQ_WEIGHT_MIN": 0.25,
        "PORTFOLIO_FIRST_LIQ_WEIGHT_MAX": 4.0,
        "PORTFOLIO_FIRST_SECTOR_NEUTRAL": True,
        "PORTFOLIO_FIRST_SECTOR_MAX_WEIGHT": 0.50,
        "PORTFOLIO_FIRST_TRADE_TOP_N": 80,
        "PORTFOLIO_FIRST_MIN_SCORE_SPREAD": 0.0,
        "PORTFOLIO_FIRST_LIQ_LOOKBACK_DAYS": 90,
        # Aggressive liquid-universe tightening; CV batch will relax automatically if this collapses the panel.
        "PORTFOLIO_FIRST_MIN_MEDIAN_TURNOVER": 3_000_000.0,
        "PORTFOLIO_FIRST_MAX_ZERO_VOL_PCT": 0.25,

        # Turnover controls / hysteresis (reduce churn after costs)
        "PORTFOLIO_FIRST_RANK_BUFFER_CANDIDATES": [0, 3],
        "PORTFOLIO_FIRST_MIN_HOLD_BARS_CANDIDATES": [0, 10],

        # Regime filter candidates (tuned by portfolio metrics)
        "PORTFOLIO_FIRST_REGIME_MA_LONG_CANDIDATES": [120, 200],
        "PORTFOLIO_FIRST_REGIME_MA_SHORT_CANDIDATES": [30, 50],
        "PORTFOLIO_FIRST_RISK_OFF_EXPOSURE_CANDIDATES": [0.0, 0.25],

        # Realistic costs (approximate round-trip components; tax applied on gains at exits)
        "PORTFOLIO_FIRST_BUY_COST": 0.004,
        "PORTFOLIO_FIRST_SELL_COST": 0.004,
        "PORTFOLIO_FIRST_TAX_RATE": 0.05,

        # Deploy only if excess vs NEPSE clears.
        "PORTFOLIO_FIRST_MIN_EXCESS_SHARPE": 0.30,
        "PORTFOLIO_FIRST_MIN_EXCESS_PSR": 0.90,
        "PORTFOLIO_FIRST_MIN_EXCESS_CAGR": 0.00,
        "PORTFOLIO_FIRST_MIN_DAYS": 750,

        # Loosen symbol-level gates so the portfolio evaluation can form.
        "POSTVAL_MIN_TRADES": 10,
        "POSTVAL_MIN_SIGNAL_STD": 0.02,
        "WFV_MIN_TRADES": 10,
        "WFV_MIN_ACTIVE_DAYS": 10,
        "WFV_MIN_COVERAGE": 0.5,
    },
}
ACTIVE_NEPSE_PROFILE = None


def apply_nepse_profile(profile_name: str) -> dict:
    if not profile_name:
        raise ValueError("Profile name is required.")
    profile = NEPSE_PROFILE_CONFIGS.get(profile_name)
    if not profile:
        valid = ", ".join(sorted(NEPSE_PROFILE_CONFIGS.keys()))
        raise ValueError(f"Unknown NEPSE profile '{profile_name}'. Valid profiles: {valid}")
    for key, value in profile.items():
        if key not in globals():
            raise KeyError(f"Profile setting '{key}' not defined in config.")
        globals()[key] = value
    globals()["ACTIVE_NEPSE_PROFILE"] = profile_name
    return profile


_NEPSE_ENV_PROFILE = os.environ.get("NEPSE_PROFILE")
if _NEPSE_ENV_PROFILE:
    try:
        apply_nepse_profile(_NEPSE_ENV_PROFILE)
    except (ValueError, KeyError) as e:
        import logging
        logging.getLogger(__name__).warning("Failed to apply NEPSE profile '%s': %s", _NEPSE_ENV_PROFILE, e)

# === TRADING RISK PARAMETERS ===
# Consolidated risk parameters used by generate_daily_signals, paper_trade_tracker, and simple_backtest.
TRAILING_STOP_PCT = 0.10       # 10% trailing stop from peak
HARD_STOP_LOSS_PCT = 0.08     # 8% hard stop from entry
TAKE_PROFIT_PCT = 0.20        # 20% take profit
PORTFOLIO_DRAWDOWN_LIMIT = 0.15  # 15% portfolio-level drawdown limit
MAX_POSITIONS = 7              # Max concurrent positions
DEFAULT_CAPITAL = 1_000_000    # NPR 10 Lakhs default capital
RECOMMENDED_HOLDING_DAYS = 40  # 40 trading days (~8 NEPSE weeks)

# === REGIME DETECTION (Model 10, 11) ===
HMM_N_STATES = 3
HMM_RETRAIN_FREQUENCY = 5  # retrain every 5 trading days
HMM_LOOKBACK = 252  # 1 year of data for HMM fitting
HMM_N_INIT = 10  # multiple random restarts
BOCPD_HAZARD_LAMBDA = 200  # expected run length
BOCPD_CHANGEPOINT_THRESHOLD = 0.5

# === PORTFOLIO METHOD (Model 6, 8) ===
PORTFOLIO_METHOD = "hrp"  # "equal_weight" | "hrp" | "cvar" | "hrp_cvar" | "shrinkage_hrp"
HRP_LOOKBACK = 60  # trading days for return covariance
HRP_RISK_MEASURE = "CVaR"  # risk measure for HRP

# === CALENDAR EFFECTS (Model 16) ===
CALENDAR_WEDNESDAY_BOOST = 1.05
CALENDAR_THURSDAY_BOOST = 1.03
CALENDAR_SUNDAY_PENALTY = 0.97
CALENDAR_DASHAIN_RALLY_BOOST = 1.08
CALENDAR_PRE_HOLIDAY_PENALTY = 0.95
CALENDAR_DASHAIN_PRE_DAYS = 21  # days before Dashain to start rally boost
CALENDAR_DASHAIN_SELLOFF_DAYS = 3  # days before Dashain for pre-holiday selloff
CALENDAR_POST_DASHAIN_CORRECTION_DAYS = 7  # trading days after Dashain to apply correction penalty
CALENDAR_POST_DASHAIN_PENALTY = 0.92  # 8% confidence reduction (6 consecutive years of post-Dashain decline)

__all__ = [
    "NEPSE_TRADING_DAYS",
    "PERIODS_PER_YEAR",
    "R_F_ANNUAL",
    "DRIFT_SHRINKAGE_ALPHA",
    "RECENT_DRIFT_WINDOW",
    "LONG_DRIFT_WINDOW",
    "BLOCK_BOOTSTRAP_SIZE",
    "STOCH_VOL_LOOKBACK",
    "FACTOR_CV_FOLDS",
    "MAX_DRAWDOWN_THRESHOLD",
    "AUTOCORRELATION_WINDOW",
    "RISK_AVERSION_LAMBDA",
    "NO_TRADE_UTILITY_THRESHOLD",
    "MAX_POS_ER",
    "MAX_NEG_ER",
    "ES_PENALTY_STRESSED",
    "STRESS_ALLOC_CAP",
    "TRIPLE_BARRIER_SL_MULT",
    "TRIPLE_BARRIER_PT_MULT",
    "TRIPLE_BARRIER_HORIZON",
    "LABEL_MODE",
    "ALPHA_LABEL_LOOKBACK",
    "ALPHA_LABEL_RANK_WINDOW",
    "ALPHA_LABEL_TOP_Q",
    "ALPHA_LABEL_BOTTOM_Q",
    "ALPHA_LABEL_MIN_HISTORY",
    "ALPHA_LABEL_COST_BPS_ROUNDTRIP",
    "ALPHA_LABEL_CROSS_SECTIONAL",
    "ALPHA_LABEL_XSEC_MIN_NAMES",
    "ALPHA_LABEL_COST_BASE_BPS",
    "ALPHA_LABEL_COST_LIQUIDITY_K",
    "ALPHA_LABEL_COST_ZERO_VOL_K",
    "ALPHA_LABEL_COST_MIN_BPS",
    "ALPHA_LABEL_COST_MAX_BPS",
    "QUANTILE_TOP",
    "QUANTILE_BOTTOM",
    "QUANTILE_DEADZONE",
    "TARGET_POSITIVE_RATE_LO",
    "TARGET_POSITIVE_RATE_HI",
    "LABEL_TUNING_PATH",
    "TUNE_LABELS",
    "CLASS_FLOOR",
    "EMBARGO_HORIZON_MULT",
    "USE_TRUE_XSEC",
    "LEGACY_PSEUDO_XSEC_TAG",
    "USE_L1_FEATURE_SELECTION",
    "USE_GLOBAL_CALIBRATION",
    "CALIBRATION_EXPERIMENT_ENABLED",
    "CALIBRATION_EXPERIMENT_OUTPUT",
    "GLOBAL_CAL_MIN_POOLED_OOS_DEFAULT",
    "GLOBAL_CAL_MIN_POOLED_OOS_MAX",
    "GLOBAL_CAL_MIN_SYM_OOS",
    "GLOBAL_CAL_STRICT_SYMBOL_REQUIREMENT",
    "GLOBAL_CAL_FORCE_PLATT",
    "GLOBAL_CAL_ISO_MIN_POOLED",
    "GLOBAL_CAL_ISO_MIN_SYMBOL_OOS",
    "GLOBAL_CAL_PLATT_MIN_POOLED",
    "GLOBAL_CALIB_MIN_OOS",
    "GLOBAL_CALIB_MIN_OOS_RELAXED",
    "GLOBAL_CALIB_RELAXED_WEIGHT",
    "SHRINKER_MIN",
    "SHRINKER_MAX",
    "SHRINKER_VOL_CAP",
    "SHRINKER_AUC_FLOOR",
    "SHRINKER_AUC_NEUTRAL",
    "SHRINKER_STD_LOW",
    "SHRINKER_STD_HIGH",
    "BASE_CAP_LOW",
    "BASE_CAP_HIGH",
    "MAX_CAP_SOFT",
    "AUC_TRUST_THRESHOLD",
    "RELIABILITY_MIN_CORR",
    "RELIABILITY_MIN_BINS_WITH_VARIATION",
    "ALLOW_SIGNAL_DEPLOYMENT",
    "DEPLOYMENT_HEALTH_CHECK_REQUIRED",
    "DEPLOYMENT_GATE_PATH",
    "DEPLOYMENT_GATE_DEFAULTS",
    "FORCE_QUANTILE_SYMBOLS",
    "SPARSE_FEATURE_MIN_NON_NULL",
    "FEATURE_RATIO_MAX",
    "TREE_MAX_DEPTH",
    "TREE_MIN_SAMPLES_LEAF",
    "SECTOR_HYPERPARAMS",
    "MIN_MEDIAN_SHARE_VOLUME",
    "MIN_MEDIAN_TURNOVER",
    "MIN_NON_ZERO_RET_RATIO",
    "USE_DYNAMIC_BARRIERS",
    "DYNAMIC_BARRIER_ATR_WINDOW",
    "DYNAMIC_BARRIER_PT_SIGMA",
    "DYNAMIC_BARRIER_SL_SIGMA",
    "DYNAMIC_BARRIER_MIN_PT",
    "DYNAMIC_BARRIER_MAX_PT",
    "DYNAMIC_BARRIER_MIN_SL",
    "DYNAMIC_BARRIER_MAX_SL",
    "DYNAMIC_BARRIER_REGIME_SCALER",
    "USE_GMM_REGIME_WEIGHTING",
    "USE_ORTHOGONAL_FEATURES",
    "ORTHOGONAL_VARIANCE",
    "USE_CPCV",
    "CPCV_PARTITIONS",
    "CPCV_TEST_BLOCKS",
    "CPCV_MIN_SAMPLES",
    "CPCV_MAX_COMBOS",
    "USE_META_LABELING",
    "META_MIN_SAMPLES",
    "META_MIN_POS_CLASS",
    "META_MIN_NEG_CLASS",
    "USE_PERMUTATION_IMPORTANCE",
    "USE_BROKER_FLOW_FEATURES",
    "BROKER_FLOW_LOOKBACK_SHORT",
    "BROKER_FLOW_LOOKBACK_LONG",
    "USE_SMART_MONEY_FEATURES",
    "PERM_IMPORTANCE_THRESHOLD",
    "PERM_IMPORTANCE_FOLDS",
    "PERM_IMPORTANCE_REPEATS",
    "RELATIVE_VOLUME_TOP_PCT",
    "RELATIVE_LIQUIDITY_TOP_PCT",
    "RELATIVE_ZERO_VOL_PCT",
    "EDGE_MIN_SAMPLES",
    "EDGE_MIN_SAMPLES_SYMBOL",
    "EDGE_SAMPLES_TARGET",
    "EDGE_MAX",
    "EDGE_BRIER_WEIGHT",
    "EDGE_WARN_HIGH",
    "EDGE_WARN_LOW",
    "EDGE_BOOTSTRAP_N",
    "EDGE_BOOTSTRAP_SEED",
    "EDGE_CI_ALPHA",
    "EDGE_MIN_REGIME_SAMPLES",
    "POSTVAL_MIN_TRADES",
    "POSTVAL_MIN_SHARPE",
    "POSTVAL_MIN_WIN_RATE",
    "POSTVAL_MIN_CAGR",
    "POSTVAL_MIN_PSR",
    "POSTVAL_SHARPE_METRIC",
    "POSTVAL_SHARPE_BENCHMARK",
    "POSTVAL_HAC_LAGS",
    "POSTVAL_THRESHOLD_MIN",
    "POSTVAL_THRESHOLD_MAX",
    "POSTVAL_THRESHOLD_STEP",
    "POSTVAL_MIN_SIGNAL_STD",
    "POSTVAL_MAX_ADF_PVALUE",
    "POSTVAL_SLIPPAGE_BASE_BPS",
    "POSTVAL_SLIPPAGE_LIQUIDITY_K",
    "POSTVAL_SLIPPAGE_ZERO_VOL_K",
    "POSTVAL_SLIPPAGE_CAP_BPS",
    "POSTVAL_RETURN_KIND",
    "WFV_LABEL_HORIZON",
    "WFV_LABEL_MODE",
    "WFV_MIN_HISTORY_DAYS",
    "WFV_TRAIN_MIN_DAYS",
    "WFV_TRAIN_WINDOW_DAYS",
    "WFV_EVAL_WINDOW_DAYS",
    "WFV_STEP_DAYS",
    "WFV_CALIBRATION_HOLDOUT_PCT",
    "WFV_MIN_TRAIN_SAMPLES",
    "WFV_MIN_CALIB_SAMPLES",
    "WFV_MIN_POS_CLASS",
    "WFV_MIN_NEG_CLASS",
    "WFV_TUNE_THRESHOLD",
    "WFV_THRESHOLD_DEFAULT",
    "WFV_THRESHOLD_GRID",
    "WFV_THRESHOLD_METRIC",
    "WFV_MIN_TRADES",
    "WFV_MIN_ACTIVE_DAYS",
    "WFV_MIN_COVERAGE",
    "WFV_MIN_SHARPE_DAILY",
    "WFV_MIN_PSR",
    "WFV_MIN_AUC",
    "WFV_MIN_YEARS",
    "WFV_MIN_POSITIVE_YEARS",
    "WFV_MIN_CAPACITY_TRADE_VALUE",
    "WFV_SLIPPAGE_BPS_DEFAULT",
    "WFV_SLIPPAGE_BPS_STRESS",
    "WFV_THRESHOLD_STRESS",
    "WFV_WINDOW_STRESS",
    "WFV_MODEL_RANDOM_STATE",
    "WFV_CONF_BASE",
    "WFV_CONF_PENALTY_LEAKAGE",
    "WFV_CONF_PENALTY_COVERAGE",
    "WFV_CONF_PENALTY_CALIB",
    "WFV_CONF_PENALTY_STABILITY",
    "WFV_DISABLE_META_BY_DEFAULT",
    "WFV_INTEGRATE_WITH_CV",
    "PORTFOLIO_TOP_K",
    "PORTFOLIO_REBALANCE_DAYS",
    "PORTFOLIO_HOLD_DAYS",
    "PORTFOLIO_LONG_ONLY",
    "PORTFOLIO_MAX_WEIGHT",
    "PORTFOLIO_BASE_SLIPPAGE_BPS",
    "PORTFOLIO_SLIPPAGE_K",
    "PORTFOLIO_MIN_ADV_VALUE",
    "PORTFOLIO_TUNE_HORIZONS",
    "PORTFOLIO_TUNE_TOPK",
    "PORTFOLIO_TUNE_HOLD_DAYS",
    "PORTFOLIO_TUNE_MAX_DEPTH",
    "HORIZON_MIN",
    "HORIZON_MAX",
    "HORIZON_LIQUIDITY_MAX_MULT",
    "HORIZON_ZERO_VOL_WEIGHT",
    "HORIZON_HALF_LIFE_MIN",
    "HORIZON_HALF_LIFE_MAX",
    "HORIZON_HALFLIFE_WEIGHT",
    "SECTOR_HORIZON_MULT",
    "HORIZON_SECTOR_MAX_MULT",
    "PROD_UNIVERSE_TOP_N",
    "PROD_UNIVERSE_MIN_MEDIAN_TURNOVER",
    "PROD_UNIVERSE_MAX_ZERO_VOL_PCT",
    "CORP_ACTION_FILTER_ENABLED",
    "CORP_ACTION_ABS_RET_THRESHOLD",
    "CORP_ACTION_Z_THRESHOLD",
    "CORP_ACTION_WINDOW_DAYS",
    "PORTFOLIO_GATE_MODE",
    "PORTFOLIO_FIRST_GLOBAL_PANEL",
    "PORTFOLIO_FIRST_TARGET",
    "PORTFOLIO_FIRST_TARGET_CANDIDATES",
    "PORTFOLIO_FIRST_TOP_K",
    "PORTFOLIO_FIRST_MIN_NAMES",
    "PORTFOLIO_FIRST_MAX_WEIGHT",
    "PORTFOLIO_FIRST_REBALANCE_DAYS",
    "PORTFOLIO_FIRST_MIN_SIGNAL_EDGE",
    "PORTFOLIO_FIRST_MIN_SCORE_SPREAD",
    "PORTFOLIO_FIRST_REBALANCE_CANDIDATES",
    "PORTFOLIO_FIRST_SCORE_SPREAD_CANDIDATES",
    "PORTFOLIO_FIRST_SCORE_SPREAD_CANDIDATES_REG",
    "PORTFOLIO_FIRST_WEIGHT_SCHEMES",
    "PORTFOLIO_FIRST_EXPOSURE_MODES",
    "PORTFOLIO_FIRST_SPREAD_FULL_MULT",
    "PORTFOLIO_FIRST_TURNOVER_PENALTY",
    "PORTFOLIO_FIRST_MAX_TOPK_FRAC",
    "PORTFOLIO_FIRST_REQUIRE_LS_POSITIVE",
    "PORTFOLIO_FIRST_TRAIN_ON",
    "PORTFOLIO_FIRST_LIQUIDITY_WEIGHTING",
    "PORTFOLIO_FIRST_LIQ_WEIGHT_POW",
    "PORTFOLIO_FIRST_LIQ_WEIGHT_ZERO_POW",
    "PORTFOLIO_FIRST_LIQ_WEIGHT_MIN",
    "PORTFOLIO_FIRST_LIQ_WEIGHT_MAX",
    "PORTFOLIO_FIRST_SECTOR_NEUTRAL",
    "PORTFOLIO_FIRST_SECTOR_MAX_WEIGHT",
    "PORTFOLIO_FIRST_TRADE_TOP_N",
    "PORTFOLIO_FIRST_INV_VOL_LOOKBACK",
    "PORTFOLIO_FIRST_INV_VOL_EPS",
    "PORTFOLIO_FIRST_LIQ_LOOKBACK_DAYS",
    "PORTFOLIO_FIRST_MIN_MEDIAN_TURNOVER",
    "PORTFOLIO_FIRST_MAX_ZERO_VOL_PCT",
    "PORTFOLIO_FIRST_BUY_COST",
    "PORTFOLIO_FIRST_SELL_COST",
    "PORTFOLIO_FIRST_TAX_RATE",
    "PORTFOLIO_FIRST_MIN_EXCESS_SHARPE",
    "PORTFOLIO_FIRST_MIN_EXCESS_PSR",
    "PORTFOLIO_FIRST_MIN_EXCESS_CAGR",
    "PORTFOLIO_FIRST_MIN_DAYS",
    "PROD_MIN_VALIDATION_THRESHOLD",
    "QUALITY_MAX_CV_STD",
    "QUALITY_MIN_CALIBRATION",
    "LEAKAGE_MAX_RATIO",
    "CV_SCORE_WEIGHTS",
    "CV_STABILITY_TARGET_STD",
    "CV_REGIME_NEUTRAL",
    "CV_LEAKAGE_WARN",
    "CV_LEAKAGE_HARD",
    "CV_SAMPLE_TARGET",
    "CV_MIN_OOS_SAMPLES",
    "CV_LABEL_BALANCE_MIN",
    "CV_MAX_WITH_LOW_SAMPLE",
    "CV_MAX_WITH_BAD_BALANCE",
    "CV_LEAKAGE_PENALTY_STRENGTH",
    "NEPSE_PROFILE_CONFIGS",
    "ACTIVE_NEPSE_PROFILE",
    "apply_nepse_profile",
    "TRAILING_STOP_PCT",
    "HARD_STOP_LOSS_PCT",
    "TAKE_PROFIT_PCT",
    "PORTFOLIO_DRAWDOWN_LIMIT",
    "MAX_POSITIONS",
    "DEFAULT_CAPITAL",
    "RECOMMENDED_HOLDING_DAYS",
    # Regime Detection
    "HMM_N_STATES",
    "HMM_RETRAIN_FREQUENCY",
    "HMM_LOOKBACK",
    "HMM_N_INIT",
    "BOCPD_HAZARD_LAMBDA",
    "BOCPD_CHANGEPOINT_THRESHOLD",
    # Portfolio Method
    "PORTFOLIO_METHOD",
    "HRP_LOOKBACK",
    "HRP_RISK_MEASURE",
    # Calendar Effects
    "CALENDAR_WEDNESDAY_BOOST",
    "CALENDAR_THURSDAY_BOOST",
    "CALENDAR_SUNDAY_PENALTY",
    "CALENDAR_DASHAIN_RALLY_BOOST",
    "CALENDAR_PRE_HOLIDAY_PENALTY",
    "CALENDAR_DASHAIN_PRE_DAYS",
    "CALENDAR_DASHAIN_SELLOFF_DAYS",
    "CALENDAR_POST_DASHAIN_CORRECTION_DAYS",
    "CALENDAR_POST_DASHAIN_PENALTY",
]
