"""Optimize position sizing parameters for TQQQ/SQQQ strategy."""

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from models.knn_model import run_walk_forward_backtest
from signals.position_sizer import PositionSizer
from backtesting.tqqq_sqqq_backtest import load_leveraged_prices, run_tqqq_sqqq_backtest
from backtesting.metrics import compute_full_metrics


PARAM_GRID = {
    "bull_threshold": [0.52, 0.55, 0.58, 0.60, 0.65],
    "max_position_pct": [0.25, 0.33, 0.50, 0.75, 1.00],
    "scaling": ["linear", "quadratic", "step"],
}


def run_sizing_optimization(
    predictions: pd.DataFrame,
    tqqq: pd.DataFrame,
    sqqq: pd.DataFrame,
) -> pd.DataFrame:
    """Grid search over position sizing parameters.

    Args:
        predictions: Walk-forward predictions.
        tqqq: TQQQ price data.
        sqqq: SQQQ price data.

    Returns:
        DataFrame with metrics for each parameter combination.
    """
    combos = list(itertools.product(
        PARAM_GRID["bull_threshold"],
        PARAM_GRID["max_position_pct"],
        PARAM_GRID["scaling"],
    ))
    total = len(combos)
    all_results = []

    for i, (bull_thresh, max_pos, scaling) in enumerate(combos):
        bear_thresh = 1 - bull_thresh

        sizer = PositionSizer(
            bull_threshold=bull_thresh,
            bear_threshold=bear_thresh,
            max_position_pct=max_pos,
            scaling=scaling,
        )

        trade_log = run_tqqq_sqqq_backtest(predictions, tqqq, sqqq, sizer)

        if len(trade_log) == 0:
            continue

        metrics = compute_full_metrics(trade_log)

        result = {
            "bull_threshold": bull_thresh,
            "bear_threshold": bear_thresh,
            "max_position_pct": max_pos,
            "scaling": scaling,
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

        if (i + 1) % 15 == 0:
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

    print(f"Optimizing position sizing ({len(PARAM_GRID['bull_threshold'])} × "
          f"{len(PARAM_GRID['max_position_pct'])} × {len(PARAM_GRID['scaling'])} "
          f"= {len(PARAM_GRID['bull_threshold']) * len(PARAM_GRID['max_position_pct']) * len(PARAM_GRID['scaling'])} combos)...\n")

    results_df = run_sizing_optimization(predictions, tqqq, sqqq)

    # Filter: max drawdown must be < 25%
    feasible = results_df[results_df["max_drawdown"] > -0.25].copy()
    feasible = feasible.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)

    print(f"\n=== Top 10 (Sharpe, maxDD > -25%) ===")
    print(feasible[["bull_threshold", "max_position_pct", "scaling",
                     "sharpe_ratio", "annual_return", "max_drawdown",
                     "trade_frequency"]].head(10).to_string(index=False))

    # If no feasible configs, relax constraint
    if len(feasible) == 0:
        print("\nNo configs with maxDD > -25%. Showing best overall:")
        feasible = results_df.sort_values("sharpe_ratio", ascending=False)
        print(feasible.head(5).to_string(index=False))

    # Save optimal config
    best = feasible.iloc[0] if len(feasible) > 0 else results_df.sort_values("sharpe_ratio", ascending=False).iloc[0]
    sizing_config = {
        "bull_threshold": float(best["bull_threshold"]),
        "bear_threshold": float(best["bear_threshold"]),
        "max_position_pct": float(best["max_position_pct"]),
        "scaling": best["scaling"],
        "sharpe_ratio": float(best["sharpe_ratio"]),
        "annual_return": float(best["annual_return"]),
        "max_drawdown": float(best["max_drawdown"]),
    }

    config_path = project_root / "signals" / "sizing_config.json"
    with open(config_path, "w") as f:
        json.dump(sizing_config, f, indent=2)
    print(f"\nOptimal sizing config saved to {config_path}")

    # Save all results
    results_df.to_csv(project_root / "signals" / "sizing_optimization.csv", index=False)
