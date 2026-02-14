# KNN QQQ Trading Model

## What This Is
A KNN machine learning model that predicts next-day QQQ direction, then translates predictions into TQQQ/SQQQ share recommendations for a $50K account. Generates daily EOD signals.

## Critical Constraint
**NEVER use TQQQ or SQQQ data for model training (Phases 1-4).** Only QQQ + auxiliary data (VIX, SPY, IWM, TLT, GLD) for training. TQQQ/SQQQ prices are used ONLY in Phase 5 (position sizing backtest) and Phase 6 (daily signal generation).

## Tech Stack
- Python 3.12+, scikit-learn, pandas, numpy, yfinance, ta (technical analysis), matplotlib, seaborn
- No deep learning frameworks — this is a pure KNN project

## Commands
- `bash init.sh` — Bootstrap session (show progress, git log, data status)
- `pip install -r requirements.txt --break-system-packages` — Install deps
- `python3 signals/generate_daily_signal.py` — Generate today's recommendation (Phase 6)
- `python3 backtesting/engine.py` — Run full backtest (Phase 4)
- `python3 scripts/retrain_model.py` — Retrain model on latest data (Phase 6)

## Project Structure
- `data/raw/` — Downloaded OHLCV CSVs (gitignored)
- `data/processed/` — Merged, cleaned datasets (gitignored)
- `features/` — Feature computation modules and CSVs
- `models/` — KNN model code, walk-forward splitter, tuning results
- `models/trained/` — Serialized model artifacts (gitignored)
- `backtesting/` — Backtest engine and results
- `signals/` — Daily signal generator and position sizer
- `scripts/` — Retraining pipeline
- `tests/` — Unit tests

## Development Workflow (Harness Pattern)
This project uses the Anthropic long-running agent harness:
1. **Read `claude-progress.txt`** at session start to see what was done last
2. **Read `feature_list.json`** to find the next incomplete feature (`passes: false`)
3. **Work on ONE feature** per session
4. **Commit with descriptive message** after completing the feature
5. **Update `feature_list.json`** — flip `passes` to `true` (never edit descriptions)
6. **Update `claude-progress.txt`** — add session entry with what was done and what's next

## Code Style
- Use type hints on all function signatures
- Docstrings on all public functions (Google style)
- No wildcard imports
- Pandas: prefer `.loc[]` over chained indexing
- Always verify no look-ahead bias in feature calculations (shift(-1) for targets, never use future data)

## Testing
- Run `python3 -m pytest tests/` for unit tests
- Walk-forward backtest is the primary validation — no random train/test splits for time series
- Always check class balance when evaluating classification metrics

## Important Notes
- Features must be computed using ONLY data available at market close on day T to predict day T+1
- StandardScaler must be fit on training data only, then transform test data
- Position sizing caps max position at 50% of $50K account
- Confidence dead zone (0.45–0.55): model says "stay in cash" when not confident
