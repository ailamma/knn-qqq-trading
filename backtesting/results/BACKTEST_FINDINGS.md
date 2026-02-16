# Backtest Findings & Analysis

## Overview

Walk-forward backtest of KNN QQQ model → TQQQ/SQQQ discrete leverage strategy.
Period: 2020-01-02 to 2025-12-31 (1,508 trading days, ~6 years).
Starting capital: $50,000.

## Model Configuration

- K=7, manhattan distance, uniform weights
- 750-day rolling training window
- 8 features: macd_hist, close_position, return_5d, williams_r, daily_return, rsi_14, stoch_d, stoch_k
- 7 discrete leverage levels: +300%, +200%, +100%, 0% (cash), -100%, -200%, -300%
- Leverage implemented via TQQQ/SQQQ + cash allocation
- Volatility targeting: caps leverage when realized vol x leverage > target

## Data Leakage Verification

Verified clean — no leakage:
- 0 violations across all 1,508 walk-forward splits
- Every test date strictly after all training dates
- StandardScaler fit on training data only, transformed on test
- Target (next_day_return) uses shift(-1) correctly — verified at 5 sample points
- Features use only data available at close on day T — verified feat_return_5d at 3 sample points
- TQQQ/SQQQ data never touches model training

## Headline Results (default config: 10pt dead zone, 2.0x vol target)

| Metric | Strategy | QQQ Buy & Hold | TQQQ Buy & Hold |
|--------|----------|----------------|-----------------|
| Total Return | 1,003% | 184% | 364% |
| Annual Return | 49.2% | 19.0% | — |
| Sharpe Ratio | 1.16 | — | — |
| Sortino Ratio | 1.64 | — | — |
| Max Drawdown | -51.0% | -35.6% | — |
| Win Rate | 53.1% | — | — |
| Final Equity | $559K | — | — |

## Dead Zone & Vol Targeting Optimization

Tested 36 combinations (6 dead zone widths x 6 vol target multiples).

### Key Configs

| Config | Sharpe | Annual | Max DD | Cash % | $50K → |
|--------|--------|--------|--------|--------|--------|
| 10pt DZ, no vol cap | 1.18 | 55.2% | -55.2% | 0% | $708K |
| 10pt DZ, 2.0x vol (default) | 1.16 | 49.2% | -51.0% | 0% | $559K |
| **10pt DZ, 1.0x vol** | **1.18** | **34.0%** | **-31.4%** | **0%** | **$291K** |
| 20pt+ DZ, no vol cap | 0.83 | 21.7% | -34.6% | 53% | $176K |
| 20pt+ DZ, 1.0x vol | 0.67 | 14.3% | -27.9% | 53% | $112K |

**Finding:** Dead zone wider than 20pt all produce identical results (~53% cash) because the model's prob_up distribution is concentrated near 0.57. The dead zone width beyond 20pt doesn't matter.

**Recommended config:** 10pt dead zone + 1.0x vol targeting (Sharpe 1.18, -31% max DD).

## Max Drawdown Analysis

All configs share the same drawdown period: the 2021-2022 bear market.

| Config | DD Start | DD Trough | Recovery | Duration |
|--------|----------|-----------|----------|----------|
| Default (2.0x vol) | 2021-11-18 | 2022-02-16 | 2022-07-26 | ~8 months |
| 1.0x vol | 2021-11-18 | 2022-04-12 | 2022-06-17 | ~7 months |
| QQQ Buy & Hold | 2021-11-18 | 2022-12-27 | — | >12 months |

## Critical Finding: Structural Bullish Bias

### The Problem

During the 2022 bear market (QQQ down -29%), the model was **LONG 62% of the time** instead of shifting bearish. This is the primary driver of drawdowns.

### Evidence

**Prob_up distribution is regime-invariant:**

| Regime | Mean prob_up | % days LONG (>0.55) |
|--------|-------------|---------------------|
| Post-COVID bull (May 2020-Oct 2021) | 0.587 | 69% |
| 2022 Bear (Nov 2021-Sep 2022) | 0.563 | 62% |
| 2023 Recovery | 0.557 | 62% |
| 2024 Bull | 0.543 | 60% |

The model outputs similar probabilities regardless of market regime.

**Accuracy collapses during bear market:**

| Regime | Overall Accuracy | Long Accuracy | Short Accuracy |
|--------|-----------------|---------------|----------------|
| COVID crash (Feb-Apr 2020) | 69.4% | 66.7% | 73.9% |
| Post-COVID bull (2020-2021) | 59.4% | 65.0% | 46.6% |
| **2022 Bear** | **48.1%** | **45.8%** | **51.7%** |
| 2023 Recovery | 52.8% | 57.4% | 45.3% |
| 2024 Bull | 55.2% | 61.6% | 45.5% |

During the bear market, long accuracy drops to 45.8% (worse than coin flip) while the model is long 62% of days at +100% to +200% leverage.

**Worst days are all confident longs during selloffs:**

8 of the 10 worst P&L days are LONG at +200% leverage with prob_up = 0.714 while QQQ drops 2-5%.

### Root Cause

1. **Historical base rate:** QQQ goes up ~55% of days. KNN learns this prior and doesn't deviate enough.
2. **Mean-reverting features:** RSI, stochastics, MACD histogram are contrarian — after selloffs they signal "oversold, buy." In a sustained bear market, this is systematically wrong.
3. **No trend/regime features in top 8:** The selected features are all oscillators/momentum. No trend-following features (e.g., SMA cross, price vs 200-day MA) made the cut.

### When Shorts Work

Short trades during the bear market actually have positive edge:
- Short win rate: 52% (vs 46% for longs)
- Short total P&L during bear: +$36K
- Long total P&L during bear: -$47K

The model is profitable when it does go short — it just doesn't go short often enough.

## Performance by Regime

| Regime | Days | Long % | Short % | Win Rate | P&L | QQQ Return |
|--------|------|--------|---------|----------|-----|------------|
| COVID crash (Feb-Apr 2020) | 62 | 63% | 37% | 65% | +$47.7K | -4.3% |
| Post-COVID bull (2020-2021) | 379 | 69% | 31% | 58% | +$173.1K | +82.1% |
| 2022 Bear (Nov 2021-Sep 2022) | 231 | 62% | 38% | 48% | +$21.2K | -29.4% |
| 2023 Recovery | 250 | 62% | 38% | 53% | +$90.3K | +52.2% |
| 2024 Bull | 252 | 60% | 40% | 53% | +$39.9K | +26.7% |
| 2025 | 250 | 67% | 33% | 49% | +$111.2K | +20.2% |

Note: Strategy is profitable in ALL regimes including the bear market (+$21K), but underperforms during bears due to being overleveraged long.

## Next Steps

1. **Regime detection:** Add trend-following features (e.g., price vs 200-day SMA, SMA cross signals, trend strength ADX) to help the model detect bear markets.
2. **Asymmetric thresholds:** Shift leverage thresholds bearish when a regime filter (e.g., price below 200-day SMA) is active.
3. **Dynamic dead zone:** Widen the dead zone during high-uncertainty regimes (high VIX, mixed signals).
4. **Ensemble approach:** Combine KNN with a simple trend-following overlay — if both agree, full leverage; if they disagree, reduce to cash.
5. **Feature engineering for bears:** Add features that capture sustained trends rather than mean-reversion (e.g., 50/200 SMA cross, consecutive down days, breadth indicators).
