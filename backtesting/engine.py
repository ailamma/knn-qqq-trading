"""Backtesting engine: simulates daily EOD trading on QQQ returns."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.knn_model import run_walk_forward_backtest


def load_config(project_root: Path) -> dict:
    """Load best model configuration.

    Args:
        project_root: Project root directory.

    Returns:
        Configuration dictionary.
    """
    with open(project_root / "models" / "best_config.json") as f:
        return json.load(f)


class Backtester:
    """Simulates daily EOD trading based on KNN model predictions.

    Tracks daily P&L, cumulative return, equity curve, and full trade log.

    Args:
        confidence_threshold: Minimum probability to enter a trade.
        slippage_per_share: Estimated slippage cost per share in dollars.
        initial_capital: Starting account value for return calculations.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.58,
        slippage_per_share: float = 0.01,
        initial_capital: float = 50000.0,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.slippage_per_share = slippage_per_share
        self.initial_capital = initial_capital

    def run(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Run backtest on walk-forward predictions.

        Args:
            predictions: DataFrame from run_walk_forward_backtest with columns:
                prediction, actual, prob_up, actual_return.

        Returns:
            Trade log DataFrame with daily P&L and equity curve.
        """
        records = []
        equity = self.initial_capital

        for date, row in predictions.iterrows():
            prob_up = row["prob_up"]
            actual_return = row["actual_return"]

            # Determine signal
            if prob_up >= self.confidence_threshold:
                signal = "LONG"
                strategy_return = actual_return
            elif prob_up <= (1 - self.confidence_threshold):
                signal = "SHORT"
                strategy_return = -actual_return
            else:
                signal = "CASH"
                strategy_return = 0.0

            # Apply slippage (as fraction of return, approximate)
            if signal != "CASH":
                # Approximate slippage: $0.01/share on ~$400 QQQ ≈ 0.0025%
                slippage_cost = self.slippage_per_share / 400.0  # rough QQQ price
                strategy_return -= slippage_cost

            daily_pnl = equity * strategy_return
            equity += daily_pnl

            records.append({
                "date": date,
                "signal": signal,
                "prob_up": round(prob_up, 4),
                "actual_return": round(actual_return, 6),
                "strategy_return": round(strategy_return, 6),
                "daily_pnl": round(daily_pnl, 2),
                "equity": round(equity, 2),
                "prediction": int(row["prediction"]),
                "actual": int(row["actual"]),
                "correct": int(row["prediction"] == row["actual"]),
            })

        trade_log = pd.DataFrame(records).set_index("date")
        return trade_log


def run_full_backtest(
    project_root: Path,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run complete walk-forward backtest with best config.

    Args:
        project_root: Project root directory.
        start_date: Backtest start date.
        end_date: Backtest end date.

    Returns:
        Tuple of (predictions DataFrame, trade log DataFrame).
    """
    config = load_config(project_root)
    features_path = project_root / "data" / "processed" / "features_master.csv"
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)

    print(f"Config: K={config['k']}, metric={config['metric']}, "
          f"weights={config['weights']}, window={config['training_window']}")
    print(f"Features: {config['features']}")
    print(f"Confidence threshold: {config.get('confidence_threshold', 0.58)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Data: {len(df)} rows\n")

    # Run walk-forward predictions
    print("Running walk-forward backtest...")
    predictions = run_walk_forward_backtest(
        df,
        feature_cols=config["features"],
        k=config["k"],
        metric=config["metric"],
        weights=config["weights"],
        training_window=config["training_window"],
        start_date=start_date,
        end_date=end_date,
    )

    # Run backtester
    threshold = config.get("confidence_threshold", 0.58)
    backtester = Backtester(confidence_threshold=threshold)
    trade_log = backtester.run(predictions)

    return predictions, trade_log


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    predictions, trade_log = run_full_backtest(project_root)

    # Save trade log
    results_dir = project_root / "backtesting" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trade_log.to_csv(results_dir / "trade_log.csv")
    predictions.to_csv(results_dir / "predictions.csv")

    # Print summary
    n_long = (trade_log["signal"] == "LONG").sum()
    n_short = (trade_log["signal"] == "SHORT").sum()
    n_cash = (trade_log["signal"] == "CASH").sum()
    total = len(trade_log)

    print(f"\n=== Trade Log Summary ===")
    print(f"Total days: {total}")
    print(f"  LONG:  {n_long} ({n_long/total:.1%})")
    print(f"  SHORT: {n_short} ({n_short/total:.1%})")
    print(f"  CASH:  {n_cash} ({n_cash/total:.1%})")
    print(f"\nStarting equity: $50,000.00")
    print(f"Final equity:    ${trade_log['equity'].iloc[-1]:,.2f}")
    print(f"Total P&L:       ${trade_log['daily_pnl'].sum():,.2f}")
    print(f"\nTrade log saved to {results_dir / 'trade_log.csv'}")
