# Flux Branding Guidelines

This document outlines the branding standards for the Flux algorithmic trading system. Reference this when creating new scripts or modules to maintain consistent identity across the codebase.

## System Name

**Primary:** Flux Algorithmic Trading System  
**Short form:** Flux

## Module Headers

Every Python script should include a module docstring following this format:

```python
"""
FLUX ALGORITHMIC TRADING SYSTEM
================================
[Module Name/Purpose]

[2-3 sentence description of what this module does]

Part of the Flux end-to-end algorithmic trading system.
"""
```

**Example:**

```python
"""
FLUX ALGORITHMIC TRADING SYSTEM
================================
Market Data Collection Module

Connects to Binance futures WebSocket streams to collect real-time order book
and trade data. Computes high-frequency trading metrics for ML model training.

Part of the Flux end-to-end algorithmic trading system.
"""
```

## Startup Banners

Use consistent banner formatting for CLI output:

```python
def print_banner():
    """Print Flux startup banner"""
    print("\n" + "=" * 70)
    print("  FLUX ALGORITHMIC TRADING SYSTEM")
    print("  [Module Name] - [Context/Symbol]")
    print("=" * 70)
    print(f"  [Key features/info line 1]")
    print(f"  [Key features/info line 2]")
    print(f"  [Key features/info line 3]")
    print("=" * 70 + "\n")
```

## Runtime Messages

### Success Messages

- Use "FLUX" prefix for major status updates
- Examples:
  - `"🚀 FLUX DATA COLLECTION ACTIVE"`
  - `"✅ FLUX DATA SAVED SUCCESSFULLY!"`
  - `"✅ FLUX MODEL TRAINING COMPLETE"`

### Shutdown Messages

- Always include "FLUX" in shutdown notifications
- Example: `"🛑 FLUX SHUTDOWN REQUESTED - Saving data..."`

### Status Updates

- Use "FLUX -" prefix for significant state changes
- Example: `"💾 FLUX - FINALIZING DATA:"`

### Completion Messages

- Reference next Flux module or stage
- Example: `"👋 Flux data collection complete. System ready for next stage."`

## File Naming

### Output Files

Prefix data and model artifacts with `flux_`:

- `flux_data.csv`
- `flux_model.pkl`
- `flux_features.parquet`
- `flux_predictions.csv`

### Script Files

Use descriptive names without prefix:

- `data_collector.py`
- `model_trainer.py`
- `strategy_engine.py`
- `backtester.py`

## Comment Style

### Inline Comments

Use standard Python style, mention "Flux" only when referencing system-specific behavior:

```python
# Calculate OFI using Flux's multi-level weighting
weighted_ofi = calculate_weighted_ofi(bids, asks)
```

### Section Comments

Standard descriptive comments, no branding needed:

```python
# Window accumulators
trade_vol_window = 0.0
trade_count_window = 0
```

## User-Facing Text

### Next Steps

Always guide users to the next Flux module:

- "Ready for Flux ML model training"
- "Next: Run feature engineering and model training modules"
- "Data ready for Flux strategy backtesting"

### Completion

Keep it professional and system-focused:

- ✅ Good: "Flux data collection complete. System ready for next stage."
- ❌ Avoid: "Thanks for using our amazing tool!"

## Consistency Checklist

When adding a new script, ensure:

- [ ] Module docstring includes Flux header
- [ ] Startup banner uses Flux branding
- [ ] Output files prefixed with `flux_`
- [ ] Major status messages include "FLUX"
- [ ] Completion text references next Flux stage
- [ ] Professional, system-focused tone throughout

## Brand Tone

**Professional** - This is enterprise-grade trading software  
**Systematic** - Emphasize the interconnected pipeline  
**Technical** - Use proper terminology (OFI, VPIN, etc.)  
**Actionable** - Guide users to next steps

Avoid: Cutesy language, excessive emojis (1-2 per message max), marketing speak
