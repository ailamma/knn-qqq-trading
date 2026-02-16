"""Optimize position sizing parameters for discrete leverage TQQQ/SQQQ strategy."""

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from models.knn_model import run_walk_forward_backtest
from signals.position_sizer import PositionSizer, DEFAULT_THRESHOLDS
from backtesting.tqqq_sqqq_backtest import load_leveraged_prices, run_tqqq_sqqq_backtest
from backtesting.metrics import compute_full_metrics


# Threshold sets to test: each is a list of (min_prob, leverage) pairs
THRESHOLD_SETS = {
    "wide": [
        (0.85, 300), (0.70, 200), (0.55, 100),
        (0.45, 0),
        (0.30, -100), (0.15, -200), (0.00, -300),
    ],
    "narrow": [
        (0.80, 300), (0.65, 200), (0.55, 100),
        (0.45, 0),
        (0.35, -100), (0.20, -200), (0.00, -300),
    ],
    "aggressive": [
        (0.75, 300), (0.60, 200), (0.52, 100),
        (0.48, 0),
        (0.40, -100), (0.25, -200), (0.00, -300),
    ],
    "conservative": [
        (0.90, 300), (0.75, 200), (0.60, 100),
        (0.40, 0),
        (0.25, -100), (0.10, -200), (0.00, -300),
    ],
}

VOL_TARGETS = [None, 1.5, 2.0, 2.5, 3.0]


def run_sizing_optimization(
    predictions: pd.DataFrame,
    tqqq: pd.DataFrame,
    sqqq: pd.DataFrame,
    features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Grid search over leverage threshold and vol-targeting parameters.

    Args:
        predictions: Walk-forward predictions.
        tqqq: TQQQ price data.
        sqqq: SQQQ price data.
        features_df: Features master for realized vol lookup.

    Returns:
        DataFrame with metrics for each parameter combination.
    """
    combos = list(itertools.product(THRESHOLD_SETS.keys(), VOL_TARGETS))
    total = len(combos)
    all_results = []

    for i, (thresh_name, vol_target) in enumerate(combos):
        thresholds = THRESHOLD_SETS[thresh_name]
        sizer = PositionSizer(
            thresholds=thresholds,
            vol_target_multiple=vol_target,
        )

        trade_log = run_tqqq_sqqq_backtest(
            predictions, tqqq, sqqq, sizer, features_df=features_df,
        )

        if len(trade_log) == 0:
            continue

        metrics = compute_full_metrics(trade_log)

        result = {
            "threshold_set": thresh_name,
            "vol_target": vol_target if vol_target else "none",
            "sharpe_ratio": metrics["sharpe_ratio"],
            "annual_return": metrics["annual_return"],
            "total_return": metrics["total_return"],
            "max_drawdown": metrics["max_drawdown"],
            "sortino_ratio": metrics["sortino_ratio"],
            "win_rate": metrics["win_rate_traded_days"],
            "profit_factor": metrics["profit_factor"],
            "trade_frequency": metrics["trade_frequency"],
        }
        all_results.append(result)

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i + 1}/{total}")

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    with open(project_root / "models" / "best_config.json") as f:
        config = json.load(f)

    features_path = project_root / "data" / "processed" / "features_master.csv"
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    tqqq, sqqq = load_leveraged_prices(project_root / "data" / "raw")

    print(f"Running walk-forward predictions...")
    predictions = run_walk_forward_backtest(
        df, feature_cols=config["features"],
        k=config["k"], metric=config["metric"], weights=config["weights"],
        training_window=config["training_window"],
        start_date="2020-01-01", end_date="2025-12-31",
        verbose=False,
    )

    n_combos = len(THRESHOLD_SETS) * len(VOL_TARGETS)
    print(f"Optimizing leverage sizing ({len(THRESHOLD_SETS)} threshold sets × "
          f"{len(VOL_TARGETS)} vol targets = {n_combos} combos)...\n")

    results_df = run_sizing_optimization(predictions, tqqq, sqqq, features_df=df)

    # Sort by Sharpe
    results_df = results_df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)

    print(f"\n=== All Results (sorted by Sharpe) ===")
    print(results_df[["threshold_set", "vol_target", "sharpe_ratio",
                       "annual_return", "max_drawdown",
                       "trade_frequency"]].to_string(index=False))

    # Save all results
    results_df.to_csv(project_root / "signals" / "sizing_optimization.csv", index=False)
    print(f"\nResults saved to signals/sizing_optimization.csv")
