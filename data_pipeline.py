"""
FLUX ALGORITHMIC TRADING SYSTEM
================================
Data Pipeline Module

Loads and preprocesses high-frequency LOB data from CSV, engineers microstructure
features, and creates time-based train/validation/test splits for ML training.

Part of the Flux end-to-end algorithmic trading system.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


def print_banner():
    """Print Flux data pipeline banner"""
    if config.VERBOSE:
        print("\n" + "=" * 70)
        print("  FLUX ALGORITHMIC TRADING SYSTEM")
        print("  Data Pipeline - Feature Engineering")
        print("=" * 70)
        print(f"  Loading: {config.RAW_DATA_PATH.name}")
        print(f"  Feature engineering with LOB microstructure priors")
        print(f"  Output: Parquet format for efficient ML training")
        print("=" * 70 + "\n")
    else:
        print("\n[1/3] Data Preparation...")


def load_raw_data() -> pd.DataFrame:
    """Load raw market data from CSV"""
    if config.VERBOSE:
        print(f"Loading raw data from {config.RAW_DATA_PATH}...")

    df = pd.read_csv(config.RAW_DATA_PATH)

    if config.VERBOSE:
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with forward/backward fill"""
    if config.VERBOSE:
        print("\nHandling missing values...")
        missing_before = df.isnull().sum().sum()
        print(f"   Missing values before: {missing_before:,}")

    # Forward fill then backward fill
    df = df.ffill().bfill()

    # Drop any remaining rows with missing values
    df = df.dropna()

    if config.VERBOSE:
        missing_after = df.isnull().sum().sum()
        print(f"   Missing values after: {missing_after:,}")
        print(f"   Rows remaining: {len(df):,}")

    return df


def scale_and_filter_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Scale returns and mark samples with actual price movement"""
    # Scale returns to make them workable (1e6 multiplier)
    df["forward_return_scaled"] = (
        df[config.TARGET_TIMEFRAME] * config.RETURN_SCALE_FACTOR
    )

    # Mark samples with actual price movement
    df["price_moved"] = (
        df[config.TARGET_TIMEFRAME].abs() > config.MIN_ABSOLUTE_RETURN
    ).astype(int)

    moved_pct = df["price_moved"].mean()

    if not config.VERBOSE:
        print(f"   - Target scaled (1e6x) - {moved_pct:.1%} have price movement")
    else:
        print(f"\nScaling targets...")
        print(f"   Using target: {config.TARGET_TIMEFRAME}")
        print(f"   Scaled returns by {config.RETURN_SCALE_FACTOR:,}x")
        print(f"   Samples with movement: {moved_pct:.2%}")
        print(f"   Original mean: {df[config.TARGET_TIMEFRAME].mean():.2e}")
        print(f"   Scaled mean: {df['forward_return_scaled'].mean():.4f}")

    return df


def engineer_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged features for key metrics"""
    if not config.ENABLE_LAG_FEATURES:
        return df

    if config.VERBOSE:
        print("\nEngineering lag features...")

    lag_features = [
        "ofi",
        "ofi_velocity",
        "vpin_toxicity",
        "depth_imbalance",
        "tick_momentum",
        "spread_bps",
        "volatility",
        "microprice_edge",
    ]

    for feature in lag_features:
        if feature in df.columns:
            for lag in config.LAG_PERIODS:
                df[f"{feature}_lag{lag}"] = df[feature].shift(lag)

    if config.VERBOSE:
        print(
            f"   Created lags for {len(lag_features)} features across {len(config.LAG_PERIODS)} periods"
        )

    return df


def engineer_rolling_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Create rolling statistics features per-split to prevent data leakage"""
    if not config.ENABLE_ROLLING_FEATURES:
        return df

    if config.VERBOSE:
        print("\nEngineering rolling features...")

    rolling_features = [
        "ofi",
        "vpin_toxicity",
        "volatility",
        "spread_bps",
        "tick_momentum",
    ]

    for feature in rolling_features:
        if feature in df.columns:
            for window in config.ROLLING_WINDOWS:
                # Use regular rolling for all splits
                # The key is that this happens AFTER splitting
                df[f"{feature}_rolling_mean_{window}"] = (
                    df[feature].rolling(window, min_periods=1).mean()
                )
                df[f"{feature}_rolling_std_{window}"] = (
                    df[feature].rolling(window, min_periods=1).std()
                )

    if config.VERBOSE:
        print(
            f"   Created rolling stats for {len(rolling_features)} features across {len(config.ROLLING_WINDOWS)} windows"
        )

    return df


def engineer_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer advanced microstructure features based on LOB research"""
    if config.VERBOSE:
        print("\nEngineering microstructure features...")

    # OFI acceleration ratio (velocity/acceleration relationship)
    df["ofi_acc_ratio"] = df["ofi_acceleration"] / (df["ofi_velocity"].abs() + 1e-8)

    # Spread-volatility ratio (liquidity vs volatility)
    df["spread_vol_ratio"] = df["spread_bps"] / (df["volatility"] + 1e-8)

    # Microprice momentum - with min_periods to prevent lookahead
    df["microprice_momentum"] = df["microprice_edge"].rolling(5, min_periods=1).mean()

    # VPIN-OFI interaction (toxicity + order flow)
    df["vpin_ofi_interaction"] = df["vpin_toxicity"] * df["ofi_zscore"]

    # Quote churn momentum (cancellation patterns) - with min_periods
    df["quote_churn_momentum"] = df["quote_churn"].rolling(10, min_periods=1).mean()

    # Tick pressure (directional momentum strength)
    df["tick_pressure"] = df["tick_direction"] * df["tick_momentum"]

    # Depth-weighted OFI (imbalance + order flow)
    df["depth_weighted_ofi"] = df["ofi"] * df["depth_imbalance"]

    # Toxicity-volatility interaction
    df["toxicity_vol_interaction"] = df["vpin_toxicity"] * df["volatility"]

    if config.VERBOSE:
        print(f"   Created 8 advanced microstructure features")

    return df


def create_time_based_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically with gap to prevent autocorrelation leakage"""
    if config.VERBOSE:
        print("\nCreating time-based splits...")

    n = len(df)

    # Calculate gap size (1 hour of ~100ms ticks = ~36000 samples)
    gap_size = min(36000, int(n * 0.01))  # 1% of data or 1 hour, whichever is smaller

    # Adjust ratios to account for gaps
    usable_n = n - (2 * gap_size)  # Remove 2 gaps from total
    train_end = int(usable_n * config.TRAIN_RATIO)
    val_end = train_end + gap_size + int(usable_n * config.VAL_RATIO)
    test_start = val_end + gap_size

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end + gap_size : val_end].copy()
    test_df = df.iloc[test_start:].copy()

    if config.VERBOSE:
        print(f"   Gap size: {gap_size:,} samples (~1 hour)")
        print(f"   Train: {len(train_df):,} rows ({len(train_df)/n:.1%})")
        print(f"   Val:   {len(val_df):,} rows ({len(val_df)/n:.1%})")
        print(f"   Test:  {len(test_df):,} rows ({len(test_df)/n:.1%})")
        print(
            f"   Total: {len(train_df) + len(val_df) + len(test_df):,} / {n:,} ({(len(train_df) + len(val_df) + len(test_df))/n:.1%})"
        )

    return train_df, val_df, test_df


def validate_feature_scaling(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
):
    """Validate feature scaling consistency across splits"""
    if config.VERBOSE:
        print("\nValidating feature scaling...")

    feature_cols = [
        col for col in train_df.columns if col not in config.EXCLUDE_FEATURES
    ]

    issues_found = []

    for col in feature_cols:
        train_std = train_df[col].std()
        val_std = val_df[col].std()
        test_std = test_df[col].std()

        # Check for extreme differences in scale between splits
        if train_std > 0:
            val_ratio = val_std / train_std if train_std > 0 else 0
            test_ratio = test_std / train_std if train_std > 0 else 0

            # Flag if validation/test std is >10x or <0.1x training std
            if val_ratio > 10 or val_ratio < 0.1:
                issues_found.append(f"{col}: val/train std ratio = {val_ratio:.2f}")
            if test_ratio > 10 or test_ratio < 0.1:
                issues_found.append(f"{col}: test/train std ratio = {test_ratio:.2f}")

        # Check for infinite or extremely large values
        if train_df[col].max() > 1e10 or train_df[col].min() < -1e10:
            issues_found.append(f"{col}: extreme values detected")

    if issues_found and config.VERBOSE:
        print(f"   WARNING: {len(issues_found)} potential scaling issues found:")
        for issue in issues_found[:5]:  # Show first 5
            print(f"     - {issue}")
    elif config.VERBOSE:
        print(f"   Feature scaling validation passed")


def save_processed_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
):
    """Save processed datasets to CSV format"""
    if config.VERBOSE:
        print("\nSaving processed data...")

    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    val_df.to_csv(config.VAL_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

    if config.VERBOSE:
        print(f"   Train: {config.TRAIN_DATA_PATH}")
        print(f"   Val:   {config.VAL_DATA_PATH}")
        print(f"   Test:  {config.TEST_DATA_PATH}")

    # Save metadata
    metadata = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "total_rows": len(train_df) + len(val_df) + len(test_df),
        "num_features": len(train_df.columns),
        "features": list(train_df.columns),
    }

    metadata_path = config.DATA_DIR / "flux_data_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if config.VERBOSE:
        print(f"   Metadata: {metadata_path}")


def run_pipeline():
    """Execute the complete data pipeline"""
    print_banner()

    # Load raw data
    df = load_raw_data()

    # Handle missing values
    df = handle_missing_values(df)

    # Scale targets and mark movement
    df = scale_and_filter_targets(df)

    # Apply lag features (safe - no forward-looking information)
    if not config.VERBOSE:
        print(f"   - Engineering lag features...")
    df = engineer_lag_features(df)

    # Drop NaNs from lag features only
    df = df.dropna()
    if not config.VERBOSE:
        print(f"   - Dataset after lags: {len(df):,} samples")

    if not config.VERBOSE:
        print(f"   - Creating train/val/test splits...")
    train_df, val_df, test_df = create_time_based_splits(df)
    if not config.VERBOSE:
        print(f"   - Engineering rolling + microstructure features per split...")

    train_df = engineer_rolling_features(train_df, is_train=True)
    train_df = engineer_microstructure_features(train_df)
    train_df = train_df.dropna()

    val_df = engineer_rolling_features(val_df, is_train=False)
    val_df = engineer_microstructure_features(val_df)
    val_df = val_df.dropna()

    test_df = engineer_rolling_features(test_df, is_train=False)
    test_df = engineer_microstructure_features(test_df)
    test_df = test_df.dropna()

    if not config.VERBOSE:
        print(
            f"   - Final: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}"
        )
    else:
        print(f"\nFinal cleaned splits:")
        print(f"   Train: {len(train_df):,} rows")
        print(f"   Val: {len(val_df):,} rows")
        print(f"   Test: {len(test_df):,} rows")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raw_count = len(load_raw_data())
        min_required = max(config.ROLLING_WINDOWS) + max(config.LAG_PERIODS)
        raise ValueError(
            f"Insufficient data for feature engineering after splitting.\n"
            f"   Raw samples: {raw_count:,}\n"
            f"   Required minimum: {min_required:,} (largest window + lags)\n"
            f"   Solution: Collect more data using data_collector.py"
        )

    # Validate feature scaling
    if not config.VERBOSE:
        print(f"   - Validating feature scaling...")
    validate_feature_scaling(train_df, val_df, test_df)

    # Save processed data
    save_processed_data(train_df, val_df, test_df)

    if not config.VERBOSE:
        print("   [OK] Data preparation complete\n")
    else:
        print("\n" + "=" * 70)
        print("FLUX DATA PIPELINE COMPLETE")
        print("=" * 70)
        print("Data ready for model training")
        print(f"   Next: Run `python model_trainer.py` to train quantile models")
        print("=" * 70 + "\n")

    return train_df, val_df, test_df


if __name__ == "__main__":
    run_pipeline()
