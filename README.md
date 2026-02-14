# KNN QQQ Trading Model

**Daily EOD signal generator for TQQQ/SQQQ trading using K-Nearest Neighbors on QQQ data.**

## Objective

Build a KNN machine learning model that:
1. Trains **exclusively on QQQ** data (plus auxiliary market data like VIX, SPY, etc.)
2. Predicts next-day QQQ direction and confidence
3. Translates predictions into TQQQ or SQQQ share recommendations
4. Targets consistent profit on a **$50,000 account**

> **Critical constraint:** TQQQ and SQQQ data are **never** used in model training.
> They are only used in the final position sizing phase.

## How It Works

```
[EOD Data] → [Feature Engineering] → [KNN Model] → [Direction + Confidence]
                                                           ↓
                                              [Position Sizer]
                                                           ↓
                                      "Buy 150 shares TQQQ" or
                                      "Buy 200 shares SQQQ" or
                                      "Stay in cash"
```

### Why QQQ for Training?

TQQQ is a 3x leveraged ETF that rebalances daily. Its returns are a function of QQQ's daily returns (roughly 3× QQQ daily return minus fees and decay). By predicting QQQ direction accurately, we can exploit TQQQ's leverage on up days and SQQQ's inverse leverage on down days — without contaminating our training data with the noise of leveraged ETF mechanics.

### Why KNN?

- **No distributional assumptions** — financial returns are non-normal
- **Captures non-linear regime patterns** — similar market conditions tend to produce similar outcomes
- **Naturally adapts** via rolling training window — "forgets" stale regimes
- **Interpretable** — you can inspect which historical days are the nearest neighbors
- **Simple to implement and debug** — critical for a production trading system

## Project Structure

```
knn-qqq-trading/
├── feature_list.json          # Harness: tracked feature list (28 features, 6 phases)
├── claude-progress.txt        # Harness: session-by-session progress log
├── init.sh                    # Harness: session bootstrap script
├── requirements.txt           # Python dependencies
├── data/
│   ├── raw/                   # Downloaded OHLCV CSVs
│   └── processed/             # Merged, cleaned, feature-enriched datasets
├── features/
│   ├── price_features.csv
│   ├── momentum_features.csv
│   ├── volatility_features.csv
│   ├── volume_features.csv
│   ├── cross_asset_features.csv
│   ├── calendar_features.csv
│   └── feature_registry.json  # Feature names, descriptions, and metadata
├── models/
│   ├── walk_forward.py        # Walk-forward splitter
│   ├── knn_model.py           # KNN classifier wrapper
│   ├── best_config.json       # Tuned hyperparameters
│   ├── selected_features.json # Optimal feature subset
│   └── trained/               # Serialized model artifacts
├── backtesting/
│   ├── engine.py              # Backtesting simulator
│   └── results/               # Metrics, plots, stress tests
├── signals/
│   ├── generate_daily_signal.py   # Daily signal generator
│   ├── position_sizer.py         # TQQQ/SQQQ position sizing
│   ├── sizing_config.json        # Optimized sizing parameters
│   └── daily_recommendations/    # Daily JSON signals
├── scripts/
│   └── retrain_model.py      # Model retraining pipeline
├── tests/                     # Unit tests
└── notebooks/                 # Exploratory analysis (optional)
```

## Phases

| Phase | Name | Features | Status |
|-------|------|----------|--------|
| 1 | Data Pipeline | F001-F004 | 🔴 Not started |
| 2 | Feature Engineering | F005-F011 | 🔴 Not started |
| 3 | KNN Model Development | F012-F017 | 🔴 Not started |
| 4 | Backtesting Engine | F018-F021 | 🔴 Not started |
| 5 | Position Sizing (TQQQ/SQQQ) | F022-F025 | 🔴 Not started |
| 6 | Daily Signal Generator | F026-F028 | 🔴 Not started |

## Development Methodology

This project uses the **Anthropic long-running agent harness** pattern:
- `feature_list.json` — Structured feature tracker (never delete/edit descriptions, only flip `passes` to `true`)
- `claude-progress.txt` — Session log for continuity across context windows
- `init.sh` — Bootstrap script to orient each new session
- **Git commits** after each feature completion with descriptive messages
- **Incremental progress** — one feature at a time, always leave code in a clean state

## Quick Start

```bash
# Bootstrap environment
chmod +x init.sh && ./init.sh

# Generate today's signal (once model is trained)
python3 signals/generate_daily_signal.py

# Run full backtest
python3 backtesting/engine.py
```

## Account Parameters

| Parameter | Value |
|-----------|-------|
| Account size | $50,000 |
| Max position | 50% of account |
| Instruments | TQQQ (3x bull) / SQQQ (3x bear) |
| Holding period | 1 day (EOD to EOD) |
| Confidence dead zone | ~0.45–0.55 (stay in cash) |
