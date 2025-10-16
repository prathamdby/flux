# Flux

🤖 An algorithmic trading bot for cryptocurrency markets.

## Overview

Flux is a comprehensive algorithmic trading system designed to collect, analyze, and trade cryptocurrency markets using advanced high-frequency trading (HFT) metrics and machine learning models.

## Project Goals

The end goal is to build a fully autonomous algorithmic trading system that can:

Collect real-time market data from cryptocurrency exchanges, process it into meaningful features, train machine learning models to predict price movements, generate trading signals, and execute trades automatically with proper risk management.

The system will continuously learn from market patterns, adapt to changing conditions, and operate independently to generate consistent returns while managing risk through position sizing, stop losses, and portfolio diversification.

**Current Progress:** ✅ Data collection system is complete and running

## Current Status

**Active Module:** Data Collection

- Collecting real-time BTC/USDT market data
- Computing 1-second HFT metrics
- Generating 5-second forward return labels
- Output: `flux_data.csv`

## Architecture

```
Data Collection → Feature Engineering → Model Training → Strategy Engine → Execution
     ↓                    ↓                    ↓              ↓            ↓
  WebSocket         Feature Pipeline      ML Models      Signal Gen    Order Mgmt
  Streams           & Selection          & Validation    & Risk Mgmt   & Portfolio
```

## Getting Started

### Prerequisites

- Python 3.8+
- UV package manager

### Installation

```bash
# Install dependencies
uv add aiofiles numpy requests websockets sortedcontainers

# Run data collection
python data_collector.py
```

### Output

The system generates `flux_data.csv` with the following metrics:

- **Market Data:** Mid price, spread, bid/ask depth
- **HFT Metrics:** OFI, VPIN, microprice, tick momentum
- **Flow Metrics:** Trade volume, direction, churn, cancel rates
- **ML Targets:** Forward returns, directional labels

## Project Structure

```
flux/
├── data_collector.py          # Market data collection
├── feature_engineering.py     # Feature creation
├── model_trainer.py           # ML model training
├── strategy_engine.py         # Trading strategy
├── order_manager.py           # Order execution
├── backtester.py              # Strategy backtesting
├── config.py                  # System configuration
├── utils.py                   # Shared utilities
├── FLUX_BRANDING.md           # Branding guidelines
└── flux_data.csv             # Generated market data
```

## Contributing

This is a personal exploration project for algorithmic trading. The system is designed to be modular and extensible, with each component building upon the previous ones.

## License

MIT License - See LICENSE file for details.
