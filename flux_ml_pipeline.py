"""
FLUX ALGORITHMIC TRADING SYSTEM
================================
ML Pipeline Orchestrator

Unified CLI entrypoint for running the complete Flux machine learning pipeline:
data preparation, model training, and backtesting with quantile regression.

Part of the Flux end-to-end algorithmic trading system.
"""

import argparse
import sys
import time

import data_pipeline
import model_trainer
import backtester


def print_main_banner():
    """Print main Flux pipeline banner"""
    print("\n" + "=" * 60)
    print("  FLUX ML PIPELINE")
    print("=" * 60)


def run_full_pipeline():
    """Run the complete pipeline: data prep → training → backtesting"""
    start_time = time.time()

    print_main_banner()

    # Step 1: Data Pipeline
    data_pipeline.run_pipeline()

    # Step 2: Model Training
    model_trainer.run_training()

    # Step 3: Backtesting
    backtester.run_backtesting()

    # Final summary
    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"[OK] Pipeline complete ({elapsed/60:.1f} minutes)")
    print("=" * 60)
    print("\nResults:")
    print("  - Backtest summary: reports/flux_backtest_summary.txt")
    print("  - Strategy plots: reports/plots/")
    print("  - Feature importance: artifacts/flux_feature_importance.csv")
    print("\n")


def main():
    """Main entrypoint with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Flux ML Pipeline - Quantile Regression for Algorithmic Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flux_ml_pipeline.py all           # Run full pipeline
  python flux_ml_pipeline.py data          # Data prep only
  python flux_ml_pipeline.py train         # Training only
  python flux_ml_pipeline.py backtest      # Backtesting only
        """,
    )

    parser.add_argument(
        "command",
        choices=["all", "data", "train", "backtest"],
        help="Pipeline stage to run",
    )

    args = parser.parse_args()

    try:
        if args.command == "all":
            run_full_pipeline()

        elif args.command == "data":
            print_main_banner()
            data_pipeline.run_pipeline()

        elif args.command == "train":
            print_main_banner()
            model_trainer.run_training()

        elif args.command == "backtest":
            print_main_banner()
            backtester.run_backtesting()

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
