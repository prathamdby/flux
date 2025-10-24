"""
FLUX ALGORITHMIC TRADING SYSTEM
================================
Backtesting Module

Evaluates trained quantile models using trading strategies with realistic costs.
Generates intuitive reports for all users.

Part of the Flux end-to-end algorithmic trading system.
"""

import json
import warnings

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

warnings.filterwarnings("ignore")


def print_banner():
    """Print Flux backtesting banner"""
    if config.VERBOSE:
        print("\n" + "=" * 70)
        print("  FLUX ALGORITHMIC TRADING SYSTEM")
        print("  Backtester - Strategy Evaluation")
        print("=" * 70)
        print(f"  Initial Capital: ${config.INITIAL_CASH:,}")
        print(f"  Commission: {config.COMMISSION*100:.2f}% per trade")
        print("=" * 70 + "\n")
    else:
        print("[3/3] Backtesting...")


def load_models() -> dict[float, lgb.Booster]:
    """Load trained quantile models"""
    if config.VERBOSE:
        print("Loading trained models...")

    models = {}
    for quantile in config.QUANTILES:
        model_path = config.ARTIFACTS_DIR / f"flux_model_q{int(quantile*100)}.txt"
        models[quantile] = lgb.Booster(model_file=str(model_path))

    if config.VERBOSE:
        for quantile in config.QUANTILES:
            print(f"   Loaded Q{quantile}")

    return models


def generate_predictions(
    models: dict[float, lgb.Booster], test_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate quantile predictions on test data"""
    if config.VERBOSE:
        print("\nGenerating predictions...")

    feature_cols = [
        col for col in test_df.columns if col not in config.EXCLUDE_FEATURES
    ]
    X_test = test_df[feature_cols]

    predictions = pd.DataFrame()
    predictions["mid"] = test_df["mid"].values
    predictions["actual_return"] = test_df["forward_return_scaled"].values

    for quantile, model in models.items():
        predictions[f"Q{int(quantile*100)}"] = model.predict(X_test)

    if config.VERBOSE:
        print(f"   Generated {len(predictions):,} predictions")

    return predictions


def simple_vectorized_backtest(
    predictions: pd.DataFrame, strategy_name: str, **params
) -> dict:
    """Simple vectorized backtest - more reliable than backtesting.py library"""

    if strategy_name == "quantile_directional":
        # Use percentile thresholds on predictions
        long_thresh = params.get("long_threshold", 0.70)
        short_thresh = params.get("short_threshold", 0.30)

        q90_pct = predictions["Q90"].rank(pct=True)
        q10_pct = predictions["Q10"].rank(pct=True)

        # Generate signals
        signals = pd.Series(0, index=predictions.index)
        signals[q90_pct > long_thresh] = 1  # Long signal
        signals[q10_pct < short_thresh] = -1  # Short signal

    elif strategy_name == "simple_momentum":
        # Simple: long if Q50 positive, short if negative
        signals = pd.Series(0, index=predictions.index)
        signals[predictions["Q50"] > 0] = 1
        signals[predictions["Q50"] < 0] = -1

    elif strategy_name == "bollinger_simple":
        # Simplified Bollinger on mid price
        window = 20
        sma = predictions["mid"].rolling(window).mean()
        std = predictions["mid"].rolling(window).std()
        upper = sma + 2 * std
        lower = sma - 2 * std

        signals = pd.Series(0, index=predictions.index)
        signals[predictions["mid"] > upper] = 1  # Breakout long
        signals[predictions["mid"] < lower] = -1  # Breakout short

    else:
        signals = pd.Series(0, index=predictions.index)

    # Calculate returns
    position = signals.shift(1).fillna(0)  # Enter next period

    # Forward returns (already scaled)
    forward_ret = (
        predictions["actual_return"].fillna(0) / config.RETURN_SCALE_FACTOR
    )  # Unscale to actual

    # Strategy returns
    strat_returns = position * forward_ret

    # Apply transaction costs
    trades = (position != position.shift(1)).astype(int)
    costs = trades * config.COMMISSION
    net_returns = strat_returns - costs

    # Calculate metrics
    num_trades = trades.sum()

    if num_trades == 0:
        return {
            "strategy": strategy_name,
            "return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "num_trades": 0,
            "final_equity": config.INITIAL_CASH,
        }

    # Equity curve
    equity = (1 + net_returns).cumprod() * config.INITIAL_CASH
    final_equity = equity.iloc[-1]
    total_return = (final_equity / config.INITIAL_CASH - 1) * 100

    # Sharpe ratio
    if net_returns.std() > 0:
        sharpe = (
            net_returns.mean() / net_returns.std() * np.sqrt(252 * 24 * 60)
        )  # Annualized for 1-min data
    else:
        sharpe = 0.0

    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax * 100
    max_dd = drawdown.min()

    # Win rate - calculate on position returns, not just trade points
    position_returns = strat_returns[position != 0]
    if len(position_returns) > 0:
        win_rate = (position_returns > 0).sum() / len(position_returns) * 100
    else:
        win_rate = 0.0

    return {
        "strategy": strategy_name,
        "return_pct": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "win_rate_pct": float(win_rate),
        "num_trades": int(num_trades),
        "final_equity": float(final_equity),
    }


def run_all_backtests(predictions: pd.DataFrame) -> list[dict]:
    """Run all strategy backtests"""
    if not config.VERBOSE:
        print("   - Testing strategies...")

    results = []

    # Quantile-based strategies
    for scenario_name, params in config.STRATEGY_SCENARIOS.items():
        if config.VERBOSE:
            print(f"\nTesting {scenario_name}...")

        result = simple_vectorized_backtest(
            predictions, "quantile_directional", **params
        )
        result["strategy"] = f"Quantile: {scenario_name}"
        results.append(result)

        if config.VERBOSE:
            print(
                f"   Return: {result['return_pct']:.2f}%, Trades: {result['num_trades']}"
            )

    # Simple strategies
    result = simple_vectorized_backtest(predictions, "simple_momentum")
    result["strategy"] = "Simple Momentum (Q50)"
    results.append(result)

    result = simple_vectorized_backtest(predictions, "bollinger_simple")
    result["strategy"] = "Bollinger Breakout"
    results.append(result)

    if not config.VERBOSE:
        total_trades = sum(r["num_trades"] for r in results)
        print(f"   - Strategies tested: {len(results)}, Total trades: {total_trades:,}")

    return results


def generate_summary_report(results: list[dict]) -> str:
    """Generate plain-language summary"""

    successful_results = [r for r in results if r["num_trades"] > 0]

    if not successful_results:
        return """
No strategies generated any trades. This likely means:
1. Prediction confidence thresholds too strict
2. Market had very little movement in test period
3. Strategy parameters need tuning

Try adjusting config.py parameters:
- Lower STRATEGY_SCENARIOS thresholds
- Check MIN_ABSOLUTE_RETURN setting
- Ensure enough samples with price movement
"""

    best_strategy = max(successful_results, key=lambda x: x["return_pct"])
    avg_return = np.mean([r["return_pct"] for r in successful_results])

    summary = f"""
{'='*60}
BACKTEST RESULTS
{'='*60}

Starting Capital: ${config.INITIAL_CASH:,}
Transaction Cost: {config.COMMISSION*100:.2f}% per trade

BEST STRATEGY: {best_strategy['strategy']}
  Final Value:  ${best_strategy['final_equity']:,.2f}
  Return:       {best_strategy['return_pct']:+.2f}%
  Trades:       {best_strategy['num_trades']:,}
  Win Rate:     {best_strategy['win_rate_pct']:.1f}%
  Sharpe Ratio: {best_strategy['sharpe_ratio']:.2f}
  Max Drawdown: {best_strategy['max_drawdown_pct']:.2f}%

INTERPRETATION:
  {('You made' if best_strategy['return_pct'] > 0 else 'You lost')} ${abs(best_strategy['final_equity'] - config.INITIAL_CASH):,.2f}

  Sharpe Ratio {best_strategy['sharpe_ratio']:.2f}:
    {'Good! Above 1.0 = decent risk-adjusted returns' if best_strategy['sharpe_ratio'] > 1.0 else 'Below 1.0 = returns not worth the risk'}
  
  Win Rate {best_strategy['win_rate_pct']:.1f}%:
    {'Solid - over half your trades profitable' if best_strategy['win_rate_pct'] > 50 else 'Most trades lost money - review strategy'}

OVERALL:
  Strategies tested: {len(successful_results)}
  Average return: {avg_return:+.2f}%

{'='*60}
"""

    return summary


def save_results(results: list[dict], summary: str):
    """Save backtest results"""
    if config.VERBOSE:
        print("\nSaving results...")

    results_data = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "config": {
            "initial_cash": config.INITIAL_CASH,
            "commission": config.COMMISSION,
        },
        "results": results,
        "summary": summary,
    }

    with open(config.BACKTEST_RESULTS_PATH, "w") as f:
        json.dump(results_data, f, indent=2)

    summary_path = config.REPORTS_DIR / "flux_backtest_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    if config.VERBOSE:
        print(f"   Saved to {config.REPORTS_DIR}")

    plot_results_comparison(results)


def plot_results_comparison(results: list[dict]):
    """Plot strategy comparison"""
    if config.VERBOSE:
        print("\nGenerating plots...")

    successful_results = [r for r in results if r["num_trades"] > 0]

    if not successful_results:
        if config.VERBOSE:
            print("   No trades to plot")
        return

    df = pd.DataFrame(successful_results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Flux Strategy Comparison", fontsize=16, fontweight="bold")

    # Returns
    ax = axes[0, 0]
    colors = ["g" if x > 0 else "r" for x in df["return_pct"]]
    ax.barh(df["strategy"], df["return_pct"], color=colors, alpha=0.7)
    ax.set_xlabel("Return (%)")
    ax.set_title("Returns", fontweight="bold")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

    # Sharpe
    ax = axes[0, 1]
    ax.barh(df["strategy"], df["sharpe_ratio"], color="steelblue", alpha=0.7)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio (higher = better risk-adj)", fontweight="bold")
    ax.axvline(1.0, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)

    # Drawdown
    ax = axes[1, 0]
    ax.barh(df["strategy"], df["max_drawdown_pct"], color="crimson", alpha=0.7)
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_title("Max Drawdown (lower = better)", fontweight="bold")

    # Trades
    ax = axes[1, 1]
    ax.barh(df["strategy"], df["num_trades"], color="purple", alpha=0.7)
    ax.set_xlabel("# Trades")
    ax.set_title("Trade Count", fontweight="bold")

    plt.tight_layout()

    plot_path = config.BACKTEST_PLOTS_DIR / "flux_strategy_comparison.png"
    plt.savefig(plot_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()

    if config.VERBOSE:
        print(f"   Saved plot to {plot_path}")


def run_audit():
    """Run audit analysis on backtest results"""
    import subprocess
    import sys

    audit_path = config.PROJECT_ROOT / "audit_results.py"
    if audit_path.exists():
        if config.VERBOSE:
            print("\nRunning audit analysis...")

        try:
            subprocess.run([sys.executable, str(audit_path)], check=True)
        except subprocess.CalledProcessError:
            if config.VERBOSE:
                print("   Audit analysis failed")
    else:
        if config.VERBOSE:
            print("\n   Audit script not found, skipping")


def run_backtesting():
    """Execute the complete backtesting pipeline"""
    print_banner()

    # Load models
    models = load_models()

    # Load test data
    if config.VERBOSE:
        print("\nLoading test data...")
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    if config.VERBOSE:
        print(f"   {len(test_df):,} test samples")

    # Generate predictions
    predictions = generate_predictions(models, test_df)

    # Run backtests
    results = run_all_backtests(predictions)

    # Generate summary
    summary = generate_summary_report(results)

    # Print summary
    print(summary)

    # Save results
    save_results(results, summary)

    if not config.VERBOSE:
        print("   [OK] Backtesting complete\n")
    else:
        print("\n" + "=" * 70)
        print("FLUX BACKTESTING COMPLETE")
        print("=" * 70)
        print(f"Results: {config.REPORTS_DIR}")
        print("=" * 70 + "\n")

    # Run audit analysis
    run_audit()


if __name__ == "__main__":
    run_backtesting()
