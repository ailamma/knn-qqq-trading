# Specification Change Analysis: Original vs. Refined

## Key Changes (Old → New)

### 1. TRAINING DATA WINDOW
- **Old:** 2011–present (~15 years)
- **New:** 20 years of historical data (~2006–present)
- **Impact:** F001, F002 — change `start` date from `2011-01-01` to `2006-01-01`
- **Rework:** MINIMAL — one parameter change per download call

### 2. MODEL OPTIMIZATION TARGET
- **Old:** Maximize accuracy / win rate; Sharpe as secondary metric
- **New:** Maximize DAILY Sharpe ratio with volatility target of 2x Nasdaq-100
- **Impact:** F014 (hyperparameter tuning), F015 (feature selection), F016 (window optimization) — optimization objective function changes from accuracy to daily Sharpe
- **Impact:** F017 (confidence calibration) — thresholds now tuned for Sharpe, not just win rate
- **Rework:** MODERATE — objective function in tuning loops changes, but KNN model itself is unchanged

### 3. LEVERAGE MODEL (BIGGEST CHANGE)
- **Old:** Binary TQQQ or SQQQ, scaled by confidence (0–50% of account)
- **New:** Discrete leverage levels: -300%, -200%, -100%, 0% (cash), +100%, +200%, +300%
  - Implemented via TQQQ/SQQQ allocation + cash
  - +300% = 100% in TQQQ
  - +200% = 67% in TQQQ, 33% cash
  - +100% = 33% in TQQQ, 67% cash
  - 0% = 100% cash
  - -100% = 33% in SQQQ, 67% cash
  - -200% = 67% in SQQQ, 33% cash
  - -300% = 100% in SQQQ
- **Impact:** F022 (position sizer) — COMPLETE REWRITE. No longer confidence-scaled continuous sizing; now maps to 7 discrete leverage levels
- **Impact:** F024, F025 (backtest + optimize) — must simulate discrete leverage switching
- **Rework:** SIGNIFICANT for Phase 5, but Phases 1–3 untouched

### 4. SIGNAL OUTPUT
- **Old:** {action: BUY, ticker: TQQQ, shares: N, confidence: X}
- **New:** {leverage_level: +200%, tqqq_allocation: 67%, sqqq_allocation: 0%, cash: 33%, action: "increase leverage"}
- **Impact:** F022, F026 — output format changes
- **Rework:** MODERATE

### 5. INTRADAY SELL STOPS
- **Old:** No intraday risk management mentioned
- **New:** Sell stops on all open day positions to limit intraday downside
- **Impact:** NEW FEATURE needed — not in current feature_list.json
- **Note:** This is an execution-layer feature, not a model feature. The KNN model still predicts close-to-close. The stops are applied to live positions.
- **Rework:** NEW addition (F029)

### 6. SIGNAL DESCRIPTION
- **Old:** "Buy TQQQ or SQQQ or stay in cash"
- **New:** "Increase, decrease, or hold leverage based on projected close-to-close change"
- **Impact:** F026 — signal generation logic. Model predicts return magnitude, then maps to leverage change relative to CURRENT position
- **Key difference:** The system considers CURRENT leverage state when deciding next action
- **Rework:** MODERATE — adds state tracking (what is our current leverage?)

### 7. VOLATILITY TARGETING
- **Old:** No explicit vol target
- **New:** Target 2x Nasdaq-100 daily volatility
- **Impact:** F022 — position sizer must incorporate realized vol scaling
- **Example:** If NDX realized vol is 15% annualized, target portfolio vol = 30%. If current leverage would produce 45% vol, reduce leverage.
- **Rework:** MODERATE — new module in position sizer

## Impact Matrix by Feature

| Feature | Status | Change Needed | Rework Level |
|---------|--------|---------------|-------------|
| F001 | Completed | Change start date to 2006 | 🟢 Trivial |
| F002 | Completed | Change start date to 2006 | 🟢 Trivial |
| F003 | Completed | Re-run with new date range | 🟢 Trivial |
| F004 | Completed | No change (still predicts QQQ next-day return) | ⚪ None |
| F005-F010 | Completed | No change (features are the same) | ⚪ None |
| F011 | Completed | Re-run to incorporate longer history | 🟢 Trivial |
| F012 | Completed | No change (walk-forward infrastructure) | ⚪ None |
| F013 | Completed | Change optimization target to daily Sharpe | 🟡 Moderate |
| F014 | Completed | Optimize for daily Sharpe, not accuracy | 🟡 Moderate |
| F015 | Completed | Re-run feature selection with Sharpe objective | 🟡 Moderate |
| F016 | Completed | Re-run window optimization with Sharpe objective | 🟡 Moderate |
| F017 | Completed | Recalibrate for leverage levels, not binary | 🟡 Moderate |
| F018 | Completed | Update to simulate 7 leverage levels | 🔴 Significant |
| F019 | Completed | Add vol-adjusted metrics | 🟡 Moderate |
| F020 | Completed | Update plots for leverage visualization | 🟡 Moderate |
| F021 | Completed | Re-run with new strategy logic | 🟢 Trivial |
| F022 | Completed | REWRITE — discrete leverage + vol targeting | 🔴 Significant |
| F023 | Completed | Change start date to 2006 | 🟢 Trivial |
| F024 | Completed | REWRITE — simulate discrete leverage switching | 🔴 Significant |
| F025 | Completed | Optimize leverage thresholds, not sizing % | 🔴 Significant |
| F026 | Completed | Update output format + state tracking | 🟡 Moderate |
| F027 | Completed | Minor updates to match new model config | 🟢 Trivial |
| F028 | Completed | Update journal format for leverage levels | 🟢 Trivial |
| NEW F029 | — | Intraday sell stop logic | 🆕 New |

## Summary
- **No change:** 8 features (F004, F005-F010, F012)
- **Trivial:** 7 features (F001-F003, F011, F021, F023, F027-F028)  
- **Moderate:** 7 features (F013-F017, F019-F020, F026)
- **Significant rewrite:** 4 features (F018, F022, F024, F025)
- **Brand new:** 1 feature (F029 — sell stops)

## Recommended Execution Order
1. F001-F003: Re-download data with 2006 start (10 minutes)
2. F013-F016: Update optimization objective to daily Sharpe (core model change)
3. F017: Recalibrate confidence → leverage level mapping
4. F022: Rewrite position sizer for discrete leverage + vol targeting
5. F018: Update backtest engine for leverage levels
6. F024-F025: Re-run TQQQ/SQQQ backtest with new logic
7. F019-F020: Update metrics and plots
8. F026: Update signal generator output
9. F029 (new): Add sell stop logic
10. F027-F028: Minor cleanup
