"""
FLUX ALGORITHMIC TRADING SYSTEM
================================
Configuration Module

Centralizes all system parameters, paths, and hyperparameters for the Flux
ML pipeline. Modify values here to configure training, backtesting, and execution.

Part of the Flux end-to-end algorithmic trading system.
"""

from pathlib import Path

# ============================================================================
# FILE PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Input data (CSV only)
RAW_DATA_PATH = PROJECT_ROOT / "flux_data.csv"

# Processed data (CSV for efficiency and simplicity)
TRAIN_DATA_PATH = DATA_DIR / "flux_train.csv"
VAL_DATA_PATH = DATA_DIR / "flux_val.csv"
TEST_DATA_PATH = DATA_DIR / "flux_test.csv"

# Model artifacts
MODEL_PATH = ARTIFACTS_DIR / "flux_model.txt"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "flux_feature_importance.csv"
TRAINING_LOG_PATH = ARTIFACTS_DIR / "flux_training_log.json"

# Backtest outputs
BACKTEST_RESULTS_PATH = REPORTS_DIR / "flux_backtest_results.json"
BACKTEST_PLOTS_DIR = REPORTS_DIR / "plots"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
BACKTEST_PLOTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# MARKET & SYMBOL
# ============================================================================

SYMBOL = "BTCUSDT"
EXCHANGE = "Binance Futures"

# ============================================================================
# DATA PROCESSING
# ============================================================================

# Train/Val/Test split ratios (chronological)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Features to use for modeling (exclude targets and metadata)
EXCLUDE_FEATURES = [
    "timestamp",
    "datetime",
    "return_5s",
    "return_10s",
    "return_30s",
    "forward_return_scaled",
    "price_moved",
]

# Handle missing values
FILL_METHOD = "forward"

# Feature engineering
ENABLE_LAG_FEATURES = True
LAG_PERIODS = [1, 2, 3, 5, 10]

ENABLE_ROLLING_FEATURES = True
ROLLING_WINDOWS = [5, 10, 20, 50]

# Target scaling (critical for HFT micro-returns)
# Multiply returns by this to make them workable for ML
RETURN_SCALE_FACTOR = 1_000_000  # 1e6: convert to basis points equivalent

# Filter samples with no price movement (99.6% are zero)
FILTER_ZERO_RETURNS = True
MIN_ABSOLUTE_RETURN = 1e-07  # Only train on samples with actual movement

# ============================================================================
# MODEL TRAINING
# ============================================================================

# Target variable
TARGET = "forward_return_scaled"  # Use scaled version
TARGET_TIMEFRAME = (
    "return_10s"  # Which forward return to use: return_5s, return_10s, or return_30s
)

# Quantile regression quantiles
QUANTILES = [0.1, 0.5, 0.9]

# LightGBM hyperparameters
LGBM_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

# Training parameters
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 50

# ============================================================================
# BACKTESTING
# ============================================================================

# Trading costs
COMMISSION = 0.0005  # 5 bps per trade
SLIPPAGE_BPS = 0.0002  # 2 bps slippage per trade

# Initial capital
INITIAL_CASH = 100_000

# Position sizing
POSITION_SIZE = 0.02  # 2% of capital per trade

# Strategy parameters - adaptive thresholds based on prediction distribution
STRATEGY_SCENARIOS = {
    "quantile_directional": {
        "long_threshold": 0.70,  # Enter long if Q90 > 70th percentile of Q90 predictions
        "short_threshold": 0.30,  # Enter short if Q10 < 30th percentile of Q10 predictions
        "exit_on_reversal": True,  # Exit when Q50 changes sign
    },
    "quantile_conservative": {
        "long_threshold": 0.85,
        "short_threshold": 0.15,
        "exit_on_reversal": True,
    },
    "quantile_aggressive": {
        "long_threshold": 0.60,
        "short_threshold": 0.40,
        "exit_on_reversal": True,
    },
}

# Classic strategies for comparison
BOLLINGER_PARAMS = {
    "window": 20,
    "num_std": 2.0,
}

MOMENTUM_PARAMS = {
    "short_window": 10,
    "long_window": 50,
}

# ============================================================================
# REPORTING
# ============================================================================

# Verbose logging (set False for clean output)
VERBOSE = False

# Metrics to compute
METRICS = [
    "Return [%]",
    "Sharpe Ratio",
    "Max. Drawdown [%]",
    "Win Rate [%]",
    "# Trades",
]

# Visualization settings
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_DPI = 100
