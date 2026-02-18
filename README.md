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
| 1 | Data Pipeline | F001-F004 | Complete |
| 2 | Feature Engineering | F005-F011 | Complete |
| 3 | KNN Model Development | F012-F017 | Complete |
| 4 | Backtesting Engine | F018-F021 | Complete |
| 5 | Position Sizing (TQQQ/SQQQ) | F022-F025 | Complete |
| 6 | Daily Signal Generator | F026-F028 | Complete |

## Backtest Results (2020-2025)

| Metric | Strategy (1.0x vol) | QQQ Buy & Hold | TQQQ Buy & Hold |
|--------|---------------------|----------------|-----------------|
| Total Return | 478% | 184% | 364% |
| Annual Return | 34.0% | 19.0% | — |
| Sharpe Ratio | 1.18 | — | — |
| Sortino Ratio | 1.85 | — | — |
| Max Drawdown | -31.4% | -35.6% | — |
| Win Rate | 53.1% | — | — |
| Starting Capital | $50,000 | — | — |
| Ending Capital | $290,836 | — | — |

### Known Limitation: Bullish Bias in Bear Markets

The model has a structural bullish bias — it predicts QQQ "up" ~62% of days regardless of regime. During the 2022 bear market (QQQ -29%), it was still LONG 62% of the time with LONG accuracy at only 45.8%. This is the primary driver of drawdowns. Short trades actually work well (52% win rate, profitable), but the model doesn't go short often enough.

Root cause: KNN learns the historical base rate (QQQ up 55% of days) and the selected features (RSI, MACD, stochastics) are mean-reverting oscillators that signal "buy" after selloffs.

Mitigated by 1.0x vol targeting which caps leverage at +/-100% during volatile periods, reducing max DD from -55% to -31%. Regime overlay (trend-following rules) tested but provided no incremental benefit over vol targeting alone.

See [backtesting/results/BACKTEST_FINDINGS.md](backtesting/results/BACKTEST_FINDINGS.md) for full analysis.

### Risk Management: Vol Targeting vs Stop Losses

We tested professional stop loss strategies (previous day low/high, 2-day low/high, pivot S1/R1) using hourly intraday data. **Stops do not improve this strategy:**

| Risk Management | Sharpe | Annual | Max DD |
|---|---|---|---|
| Vol targeting 1.0x (production) | 1.18 | 34.0% | -31.4% |
| + 2-Day Low/High stops | 0.80 | 21.1% | -28.8% |
| + Prev Day Low/High stops | 0.64 | 16.1% | -26.8% |
| No risk management | 1.18 | 55.2% | -55.2% |

Vol targeting is fundamentally better for a mean-reversion KNN strategy because:
1. **Preventive**: reduces position size before bad days (vs reactive stops that exit after damage)
2. **Gap-proof**: 69% of stop triggers happen at the open (overnight gaps) — too late for stops to help
3. **Preserves edge**: 53% of stopped trades would have recovered — stops cut winning mean-reversion trades
4. **Sharpe-preserving**: vol targeting halves max DD while maintaining the same Sharpe ratio

Stop loss infrastructure is available in the backtest engine (`stop_type` parameter) but not used in production.

### Feature Engineering Experiments

Tested additional features beyond the optimal 8; all degraded performance:
- Overnight gap / intraday return: Sharpe 0.91 (from 1.51), bear DD improved but overall worse
- VIX term structure (VIX/VIX3M ratio): Sharpe 0.26-0.67, worse across all metrics
- Regime features (SMA200, golden cross, ADX): curse of dimensionality, negative Sharpe

The KNN model's 8 momentum/oscillator features are tightly co-optimized. Adding fundamentally different feature types degrades the neighbor search.

## Development Methodology

This project uses the **Anthropic long-running agent harness** pattern:
- `feature_list.json` — Structured feature tracker (never delete/edit descriptions, only flip `passes` to `true`)
- `claude-progress.txt` — Session log for continuity across context windows
- `init.sh` — Bootstrap script to orient each new session
- **Git commits** after each feature completion with descriptive messages
- **Incremental progress** — one feature at a time, always leave code in a clean state

## Model Versions

Two model versions are available for daily signal generation:

| | V1 (Ensemble) | V2 (KNN-only) |
|---|---|---|
| Method | KNN 70% + Logistic Regression 30% blend | KNN only |
| Sharpe Ratio | 1.30 | 1.18 |
| Total Return (2020-2025) | 478% | 478% |
| Max Drawdown | -33.1% | -31.4% |
| Bear Market (2022) | +15.4% | +30.7% |
| Best for | Smoother returns, higher Sharpe | Bear market resilience, lower drawdown |

V1 blends KNN neighbor-vote probabilities with a logistic regression probability estimate. The two models have low error correlation (0.278), so the blend smooths predictions. V2 is the original KNN-only model which performs better during bear markets.

## Quick Start

```bash
# Bootstrap environment
chmod +x init.sh && ./init.sh

# Generate today's signal (default: V1 ensemble)
python3 signals/generate_daily_signal.py

# Generate V1 (ensemble) or V2 (KNN-only) signal
python3 signals/generate_daily_signal.py --v1
python3 signals/generate_daily_signal.py --v2

# Run both models and compare
python3 signals/generate_daily_signal.py --both

# Custom account balance
python3 signals/generate_daily_signal.py --v1 --balance 100000

# Run full backtest
python3 backtesting/engine.py
```

## Execution Timing

The model is trained on close-to-close returns: entry at Day T close, exit at Day T+1 close. To match this assumption, run the signal generator near market close and execute before 4:00 PM ET.

**Recommended daily workflow:**

| Time (ET) | Action |
|---|---|
| 3:55 PM | Run `python3 signals/generate_daily_signal.py --both` |
| 3:56 PM | Review signal, decide V1 or V2 |
| 3:58-3:59 PM | Place market order (buy TQQQ, SQQQ, or close position) |
| 4:00 PM | Market closes, hold overnight |
| Next day 3:55 PM | Run new signal, adjust position |

**Why not run next morning?**
- The backtest's Sharpe 1.18-1.30 assumes close-to-close execution. Entering at the next-day open introduces overnight gap risk that the model never trained on.
- TQQQ/SQQQ regularly gap 1-3% at the open. If the model predicts UP and you wait until morning, you've already missed the overnight move.
- At 3:55 PM, QQQ's price is within ~0.05% of the actual close — features (RSI, MACD, stochastics) are virtually identical.

## Account Parameters

| Parameter | Value |
|-----------|-------|
| Account size | $50,000 |
| Leverage levels | +300%, +200%, +100%, 0% (cash), -100%, -200%, -300% |
| Instruments | TQQQ (3x bull) / SQQQ (3x bear) + cash |
| Holding period | 1 day (EOD to EOD) |
| Confidence dead zone | 0.45–0.55 (stay in cash) |
| Vol targeting | 1.0x QQQ realized vol (caps leverage in high-vol regimes) |
