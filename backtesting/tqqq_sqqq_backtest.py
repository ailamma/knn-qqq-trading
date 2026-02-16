"""Backtest full TQQQ/SQQQ strategy with position sizing on $50K account."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.knn_model import run_walk_forward_backtest
from signals.position_sizer import PositionSizer
from backtesting.metrics import compute_full_metrics, print_full_metrics


def load_leveraged_prices(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load TQQQ and SQQQ daily price data.

    Args:
        raw_dir: Directory with raw CSV files.

    Returns:
        Tuple of (TQQQ DataFrame, SQQQ DataFrame).
    """
    tqqq = pd.read_csv(raw_dir / "tqqq_daily.csv", index_col=0, parse_dates=True)
    sqqq = pd.read_csv(raw_dir / "sqqq_daily.csv", index_col=0, parse_dates=True)
    return tqqq, sqqq


def run_tqqq_sqqq_backtest(
    predictions: pd.DataFrame,
    tqqq: pd.DataFrame,
    sqqq: pd.DataFrame,
    sizer: PositionSizer,
    initial_capital: float = 50000.0,
) -> pd.DataFrame:
    """Simulate TQQQ/SQQQ trading with position sizing.

    Args:
        predictions: Walk-forward predictions with prob_up and actual_return.
        tqqq: TQQQ daily price data.
        sqqq: SQQQ daily price data.
        sizer: PositionSizer instance.
        initial_capital: Starting account balance.

    Returns:
        Trade log DataFrame with equity curve.
    """
    cash = initial_capital
    records = []

    for date, row in predictions.iterrows():
        # Get leveraged ETF prices for this date
        if date not in tqqq.index or date not in sqqq.index:
            continue

        tqqq_price = tqqq.loc[date, "Close"]
        sqqq_price = sqqq.loc[date, "Close"]

        # Get next-day prices for P&L calculation
        tqqq_idx = tqqq.index.get_loc(date)
        sqqq_idx = sqqq.index.get_loc(date)

        if tqqq_idx + 1 >= len(tqqq) or sqqq_idx + 1 >= len(sqqq):
            continue

        tqqq_next = tqqq.iloc[tqqq_idx + 1]["Close"]
        sqqq_next = sqqq.iloc[sqqq_idx + 1]["Close"]

        # Size position
        total_equity = cash
        rec = sizer.size(row["prob_up"], total_equity, tqqq_price, sqqq_price)

        # Calculate P&L
        if rec["action"] == "BUY" and rec["shares"] > 0:
            ticker = rec["ticker"]
            shares = rec["shares"]
            if ticker == "TQQQ":
                entry_price = tqqq_price
                exit_price = tqqq_next
            else:
                entry_price = sqqq_price
                exit_price = sqqq_next

            # Slippage: $0.01/share each way
            cost_basis = shares * (entry_price + 0.01)
            proceeds = shares * (exit_price - 0.01)
            daily_pnl = proceeds - cost_basis
        else:
            ticker = None
            shares = 0
            daily_pnl = 0.0

        cash += daily_pnl
        strategy_return = daily_pnl / total_equity if total_equity > 0 else 0

        # Map action to LONG/SHORT/CASH for consistent reporting
        if rec["action"] == "BUY" and rec.get("ticker") == "TQQQ":
            signal_label = "LONG"
        elif rec["action"] == "BUY" and rec.get("ticker") == "SQQQ":
            signal_label = "SHORT"
        else:
            signal_label = "CASH"

        records.append({
            "date": date,
            "signal": signal_label,
            "ticker": ticker,
            "shares": shares,
            "prob_up": round(row["prob_up"], 4),
            "allocation_tier": rec["allocation_tier"],
            "allocation_pct": rec["allocation_pct"],
            "actual_return": row["actual_return"],
            "strategy_return": round(strategy_return, 6),
            "daily_pnl": round(daily_pnl, 2),
            "equity": round(cash, 2),
            "prediction": int(row["prediction"]),
            "actual": int(row["actual"]),
            "correct": int(row["prediction"] == row["actual"]),
        })

    return pd.DataFrame(records).set_index("date")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    # Load config
    with open(project_root / "models" / "best_config.json") as f:
        config = json.load(f)

    # Load data
    features_path = project_root / "data" / "processed" / "features_master.csv"
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    tqqq, sqqq = load_leveraged_prices(project_root / "data" / "raw")

    print(f"Features: {len(df)} rows")
    print(f"TQQQ: {len(tqqq)} rows, SQQQ: {len(sqqq)} rows")
    print(f"Config: K={config['k']}, metric={config['metric']}, window={config['training_window']}")

    # Run walk-forward predictions
    print("\nRunning walk-forward predictions...")
    predictions = run_walk_forward_backtest(
        df, feature_cols=config["features"],
        k=config["k"], metric=config["metric"], weights=config["weights"],
        training_window=config["training_window"],
        start_date="2020-01-01", end_date="2025-12-31",
    )

    # Run TQQQ/SQQQ backtest with tiered sizing (10/20/30%)
    print("\nRunning TQQQ/SQQQ backtest (tiered: 10/20/30%)...")
    with open(project_root / "signals" / "sizing_config.json") as f:
        sizing_config = json.load(f)
    sizer = PositionSizer(
        bull_threshold=sizing_config["bull_threshold"],
        bear_threshold=sizing_config["bear_threshold"],
    )

    trade_log = run_tqqq_sqqq_backtest(predictions, tqqq, sqqq, sizer)

    # Compute metrics
    metrics = compute_full_metrics(trade_log)
    print_full_metrics(metrics)

    # Compare with buy-and-hold
    overlap_dates = trade_log.index
    tqqq_bah = tqqq.loc[overlap_dates[0]:overlap_dates[-1], "Close"]
    tqqq_bah_return = tqqq_bah.iloc[-1] / tqqq_bah.iloc[0] - 1

    qqq_bah = (1 + trade_log["actual_return"]).prod() - 1

    print(f"\n=== Strategy Comparison ===")
    print(f"  KNN TQQQ/SQQQ Strategy: {metrics['total_return']:.2%}")
    print(f"  Buy & Hold TQQQ:        {tqqq_bah_return:.2%}")
    print(f"  Buy & Hold QQQ:          {qqq_bah:.2%}")

    # Save results
    results_dir = project_root / "backtesting" / "results" / "tqqq_sqqq_backtest"
    results_dir.mkdir(parents=True, exist_ok=True)

    trade_log.to_csv(results_dir / "trade_log.csv")
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {results_dir}")
