"""
FLUX ALGORITHMIC TRADING SYSTEM
================================
Model Training Module

Trains quantile regression models using LightGBM to predict forward returns.
Optimizes for pinball loss and saves model artifacts for backtesting.

Part of the Flux end-to-end algorithmic trading system.
"""

import json
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config


def print_banner():
    """Print Flux model training banner"""
    if config.VERBOSE:
        print("\n" + "=" * 70)
        print("  FLUX ALGORITHMIC TRADING SYSTEM")
        print("  Model Trainer - Quantile Regression")
        print("=" * 70)
        print(f"  Algorithm: LightGBM Gradient Boosting")
        print(f"  Quantiles: {config.QUANTILES}")
        print(f"  Target: {config.TARGET}")
        print("=" * 70 + "\n")
    else:
        print("[2/3] Model Training...")


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train/val/test data"""
    if config.VERBOSE:
        print("Loading processed data...")

    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(config.VAL_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    # Filter to samples with actual price movement if configured
    if config.FILTER_ZERO_RETURNS and "price_moved" in train_df.columns:
        train_orig = len(train_df)
        train_df = train_df[train_df["price_moved"] == 1].copy()

        if train_orig == 0:
            raise ValueError(
                "Training dataset is empty after filtering.\n"
                "   This typically indicates insufficient raw data.\n"
                "   Solution: Run data collection to gather more samples."
            )

        if not config.VERBOSE:
            print(
                f"   - Training on {len(train_df):,} samples with price movement ({len(train_df)/train_orig:.1%})"
            )
        else:
            print(
                f"   Filtered train: {len(train_df):,} / {train_orig:,} ({len(train_df)/train_orig:.1%} with movement)"
            )

    if config.VERBOSE:
        print(f"   Train: {len(train_df):,} rows")
        print(f"   Val:   {len(val_df):,} rows")
        print(f"   Test:  {len(test_df):,} rows")

    return train_df, val_df, test_df


def prepare_features_and_targets(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple:
    """Separate features and targets"""
    if config.VERBOSE:
        print("\nPreparing features and targets...")

    # Identify feature columns
    feature_cols = [
        col for col in train_df.columns if col not in config.EXCLUDE_FEATURES
    ]

    if config.VERBOSE:
        print(f"   Features: {len(feature_cols)}")
        print(f"   Target: {config.TARGET}")

    X_train = train_df[feature_cols]
    y_train = train_df[config.TARGET]

    X_val = val_df[feature_cols]
    y_val = val_df[config.TARGET]

    X_test = test_df[feature_cols]
    y_test = test_df[config.TARGET]

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_quantile_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    quantile: float,
) -> lgb.Booster:
    """Train a single quantile regression model"""
    if config.VERBOSE:
        print(f"\nTraining quantile={quantile} model...")

    # Update params for this quantile
    params = config.LGBM_PARAMS.copy()
    params["alpha"] = quantile

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train model
    start_time = time.time()

    callbacks = [lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS)]
    if config.VERBOSE:
        callbacks.append(lgb.log_evaluation(period=50))

    model = lgb.train(
        params,
        train_data,
        num_boost_round=config.NUM_BOOST_ROUND,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    elapsed = time.time() - start_time

    if not config.VERBOSE:
        print(
            f"   - Q{int(quantile*100)} trained ({elapsed:.1f}s, {model.best_iteration} rounds)"
        )
    else:
        print(f"   Training complete in {elapsed:.2f}s")
        print(f"   Best iteration: {model.best_iteration}")
        print(f"   Best score: {model.best_score['val']['quantile']:.6f}")

    return model


def save_models(models: dict[float, lgb.Booster]):
    """Save trained models to disk"""
    if config.VERBOSE:
        print("\nSaving models...")

    for quantile, model in models.items():
        model_path = config.ARTIFACTS_DIR / f"flux_model_q{int(quantile*100)}.txt"
        model.save_model(str(model_path))
        if config.VERBOSE:
            print(f"   Q{quantile}: {model_path}")


def compute_feature_importance(
    models: dict[float, lgb.Booster], feature_cols: list
) -> pd.DataFrame:
    """Compute and save feature importance"""
    print("\nComputing feature importance...")

    # Average importance across all quantiles
    importance_dict = {}

    for quantile, model in models.items():
        importance = model.feature_importance(importance_type="gain")
        for i, col in enumerate(feature_cols):
            if col not in importance_dict:
                importance_dict[col] = []
            importance_dict[col].append(importance[i])

    # Create dataframe
    importance_df = pd.DataFrame(
        {
            "feature": list(importance_dict.keys()),
            "importance": [np.mean(vals) for vals in importance_dict.values()],
            "importance_std": [np.std(vals) for vals in importance_dict.values()],
        }
    )

    importance_df = importance_df.sort_values("importance", ascending=False)

    # Save to CSV
    importance_df.to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)
    print(f"   Saved to {config.FEATURE_IMPORTANCE_PATH}")

    # Print top 10
    print("\n   Top 10 Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"      {row['feature']:<30} {row['importance']:>10.2f}")

    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame):
    """Plot feature importance"""
    print("\nPlotting feature importance...")

    plt.figure(figsize=(10, 8))

    top_features = importance_df.head(20)

    sns.barplot(data=top_features, y="feature", x="importance", palette="viridis")
    plt.title(
        "Top 20 Feature Importance (Avg across Quantiles)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Importance (Gain)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    plot_path = config.ARTIFACTS_DIR / "flux_feature_importance.png"
    plt.savefig(plot_path, dpi=config.FIGURE_DPI)
    plt.close()

    print(f"   Saved plot to {plot_path}")


def evaluate_models(
    models: dict[float, lgb.Booster], X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """Evaluate model performance on test set"""
    print("\nEvaluating models on test set...")

    results = {}

    for quantile, model in models.items():
        y_pred = model.predict(X_test)

        # Compute pinball loss
        errors = y_test - y_pred
        pinball_loss = np.mean(
            np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
        )

        # Compute MAE for reference
        mae = np.mean(np.abs(errors))

        results[f"Q{quantile}"] = {
            "pinball_loss": pinball_loss,
            "mae": mae,
            "mean_prediction": np.mean(y_pred),
            "std_prediction": np.std(y_pred),
        }

        print(f"   Q{quantile}: Pinball Loss = {pinball_loss:.6f}, MAE = {mae:.6f}")

    return results


def save_training_log(results: dict, feature_cols: list):
    """Save training metadata and results"""
    print("\nSaving training log...")

    log = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "config": {
            "symbol": config.SYMBOL,
            "target": config.TARGET,
            "quantiles": config.QUANTILES,
            "num_features": len(feature_cols),
            "lgbm_params": config.LGBM_PARAMS,
        },
        "results": results,
    }

    with open(config.TRAINING_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    print(f"   Saved to {config.TRAINING_LOG_PATH}")


def run_training():
    """Execute the complete training pipeline"""
    print_banner()

    # Load data
    train_df, val_df, test_df = load_processed_data()

    # Prepare features and targets
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = (
        prepare_features_and_targets(train_df, val_df, test_df)
    )

    # Train models for each quantile
    models = {}
    for quantile in config.QUANTILES:
        models[quantile] = train_quantile_model(
            X_train, y_train, X_val, y_val, quantile
        )

    # Save models
    save_models(models)

    # Feature importance (suppress verbose parts)
    if config.VERBOSE:
        importance_df = compute_feature_importance(models, feature_cols)
        plot_feature_importance(importance_df)
    else:
        # Silent compute
        importance_dict = {}
        for quantile, model in models.items():
            importance = model.feature_importance(importance_type="gain")
            for i, col in enumerate(feature_cols):
                if col not in importance_dict:
                    importance_dict[col] = []
                importance_dict[col].append(importance[i])

        importance_df = pd.DataFrame(
            {
                "feature": list(importance_dict.keys()),
                "importance": [np.mean(vals) for vals in importance_dict.values()],
            }
        ).sort_values("importance", ascending=False)

        importance_df.to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)

    # Evaluate on test set
    if config.VERBOSE:
        results = evaluate_models(models, X_test, y_test)
        save_training_log(results, feature_cols)
    else:
        # Silent evaluation
        results = {}
        for quantile, model in models.items():
            y_pred = model.predict(X_test)
            errors = y_test - y_pred
            pinball_loss = np.mean(
                np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
            )
            results[f"Q{quantile}"] = {"pinball_loss": pinball_loss}

        log = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": {
                "symbol": config.SYMBOL,
                "target": config.TARGET,
                "quantiles": config.QUANTILES,
            },
            "results": results,
        }
        with open(config.TRAINING_LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)

    if not config.VERBOSE:
        print("   [OK] Model training complete\n")
    else:
        print("\n" + "=" * 70)
        print("FLUX MODEL TRAINING COMPLETE")
        print("=" * 70)
        print("Models trained and saved successfully")
        print(f"   Next: Run `python backtester.py` to evaluate trading performance")
        print("=" * 70 + "\n")

    return models


if __name__ == "__main__":
    run_training()
