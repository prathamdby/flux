# Flux

🤖 An algorithmic trading bot for cryptocurrency markets.

## Overview

Flux is a comprehensive algorithmic trading system designed to collect, analyze, and trade cryptocurrency markets using advanced high-frequency trading (HFT) metrics and machine learning models.

## Project Goals

The end goal is to build a fully autonomous algorithmic trading system that can:

Collect real-time market data from cryptocurrency exchanges, process it into meaningful features, train machine learning models to predict price movements, generate trading signals, and execute trades automatically with proper risk management.

The system will continuously learn from market patterns, adapt to changing conditions, and operate independently to generate consistent returns while managing risk through position sizing, stop losses, and portfolio diversification.

**Current Progress:** ✅ Data collection complete | ✅ ML pipeline implemented

## Current Status

**Phase 1 - Data Collection:** ✅ Complete

- Real-time BTC/USDT market data collection via WebSocket (~100ms tick-by-tick)
- Multiple forward return timeframes (5s, 10s, 30s) for short-term predictions
- Enhanced microstructure features (microprice edge, VPIN toxicity, quote churn)
- All matured samples saved (filtering done in ML pipeline)
- Output: CSV format with automatic forward return backfilling
- Real-time quality statistics and periodic status updates
- Robust Binance depth stream synchronization with auto-reconnection

**Phase 2 - ML Pipeline:** ✅ Complete

- Quantile regression models (LightGBM)
- Multi-strategy backtesting framework
- Cost-aware PnL evaluation
- Feature engineering with LOB priors

## Architecture

```
Data Collection → Feature Engineering → Model Training → Strategy Engine → Execution
     ↓                    ↓                    ↓              ↓            ↓
  WebSocket         Feature Pipeline      ML Models      Signal Gen    Order Mgmt
  Streams           & Selection          & Validation    & Risk Mgmt   & Portfolio
```

## Getting Started

### Prerequisites

- Python 3.13+
- UV package manager

### Installation

```bash
# Install all dependencies
uv sync
```

### Phase 1: Data Collection

```bash
# Run data collection (streams live market data)
python data_collector.py
```

**Configuration Options** (edit top of `data_collector.py`):

- `OFI_DEPTH_LEVELS` - Order book depth for OFI calculation (default: 15 levels)
- `RETURN_TIMEFRAMES` - Forward return horizons (default: [5, 10, 30] seconds)
- `OFI_ZSCORE_WINDOW` - Lookback window for OFI normalization (default: 200)
- `SAVE_INTERVAL_SECONDS` - How often to save data (default: 60s)

**Target Configuration** (edit `config.py`):

- `TARGET_TIMEFRAME` - Which return to use for training (default: "return_10s")

**Output:** `flux_data.csv` with enhanced HFT metrics:

- **Price Features:** Mid price, microprice, microprice edge, spread (raw & bps)
- **OFI Features:** Multi-level weighted OFI, velocity, acceleration, z-score
- **Depth Features:** Depth imbalance, total volume
- **Microstructure Features:** Quote churn, VPIN toxicity, tick direction & momentum
- **Market Regime:** Realized volatility
- **ML Targets:** Forward returns at 5s, 10s, 30s horizons

### Phase 2: ML Pipeline

```bash
# Run complete ML pipeline (data prep → train → backtest)
python flux_ml_pipeline.py all

# Or run individual stages:
python flux_ml_pipeline.py data      # Data preparation & feature engineering
python flux_ml_pipeline.py train     # Train quantile models
python flux_ml_pipeline.py backtest  # Evaluate with multiple strategies
python flux_ml_pipeline.py cleanup   # Remove all generated files
```

**Output:**

- `data/flux_train.csv` - Training data with engineered features
- `artifacts/flux_model_q*.txt` - Trained quantile models
- `reports/flux_backtest_summary.txt` - Plain-language results
- `reports/plots/` - Strategy comparison visualizations

### Configuration

Edit `config.py` to customize:

- Train/val/test split ratios
- Target timeframe selection (return_5s, return_10s, or return_30s)
- LightGBM hyperparameters
- Quantile levels (default: 0.1, 0.5, 0.9)
- Trading costs (default: 5 bps)
- Strategy parameters

## Project Structure

```
flux/
├── data_collector.py          # ✅ Market data collection (WebSocket streams)
├── data_pipeline.py           # ✅ Feature engineering & preprocessing
├── model_trainer.py           # ✅ Quantile regression training
├── backtester.py              # ✅ Multi-strategy backtesting
├── flux_ml_pipeline.py        # ✅ Unified CLI orchestrator
├── config.py                  # ✅ System configuration
├── FLUX_BRANDING.md           # Branding guidelines
├── flux_data.csv              # Raw market data (CSV format)
├── data/                      # Processed datasets (CSV)
├── artifacts/                 # Trained models & diagnostics
└── reports/                   # Backtest results & visualizations
```

## ML Approach

### Quantile Regression

Flux uses **quantile regression** instead of traditional point predictions because:

1. **Captures Uncertainty:** Predicts distribution (Q10, Q50, Q90) rather than single values
2. **Risk-Aware:** Separates upside potential from downside risk
3. **Robust to Outliers:** Pinball loss is more robust than MSE in financial data
4. **Actionable:** Quantiles directly inform position sizing and entry/exit thresholds

### Feature Engineering

Based on LOB microstructure research ([Briola et al., 2024](https://arxiv.org/html/2403.09267v3)):

- **Order Flow Imbalance (OFI):** Multi-level weighted depth changes with velocity & acceleration
- **VPIN Toxicity:** Volume-synchronized informed trading probability
- **Quote Churn:** Order cancellation rate proxy for market toxicity
- **Microprice Features:** Volume-weighted mid price and edge calculations
- **Tick Dynamics:** Direction and momentum from microprice movements
- **Microstructure Interactions:** OFI-VPIN, depth-weighted OFI, toxicity-volatility
- **Temporal Features:** Lag features and rolling statistics for pattern detection

### Backtesting Strategies

**Quantile-Based:**

- Momentum: Trade on extreme quantile predictions
- Conservative: Tight thresholds, fewer trades
- Aggressive: Wider thresholds, more trades

**Classic Baselines:**

- Bollinger Breakout: Mean reversion on band touches
- VWAP Mean Reversion: Trade deviations from volume-weighted average
- Momentum Crossover: Simple moving average crossovers

All strategies include realistic transaction costs (5 bps) and clear, non-technical reporting.

## Key Features

✅ **Tick-by-Tick Data:** High-frequency order book updates (~100ms resolution)
✅ **Multiple Prediction Horizons:** Train on 5s, 10s, or 30s forward returns
✅ **Enhanced Microstructure Features:** Microprice edge, quote churn, VPIN toxicity
✅ **Robust Data Collection:** Automatic backfilling, periodic saves, minimal data loss
✅ **Advanced Metrics:** Multi-level OFI, tick direction momentum, depth imbalance
✅ **Production-Ready:** Auto-reconnection, proper Binance sync, error handling
✅ **Cost-Aware:** Transaction costs baked into backtesting
✅ **Interpretable:** Feature importance analysis and plain-language reports
✅ **Multi-Strategy:** Test multiple approaches simultaneously
✅ **Real-time Monitoring:** Status updates during collection

## Contributing

This is a personal exploration project for algorithmic trading. The system is designed to be modular and extensible, with each component building upon the previous ones.

## License

MIT License - See LICENSE file for details.
