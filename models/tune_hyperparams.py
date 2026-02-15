"""Hyperparameter tuning: optimize K, distance metric, weighting scheme."""

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from models.knn_model import run_walk_forward_backtest, compute_metrics, BASELINE_FEATURES


# Grid search space
PARAM_GRID = {
    "k": [3, 5, 7, 10, 15, 20, 30, 50],
    "metric": ["euclidean", "manhattan", "minkowski"],
    "weights": ["uniform", "distance"],
}


def run_grid_search(
    df: pd.DataFrame,
    feature_cols: list[str],
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    training_window: int = 500,
) -> pd.DataFrame:
    """Run grid search over KNN hyperparameters.

    Evaluates each combination via walk-forward backtest and ranks by Sharpe ratio.

    Args:
        df: Features master DataFrame.
        feature_cols: Feature columns to use.
        start_date: Backtest start date.
        end_date: Backtest end date.
        training_window: Training window size.

    Returns:
        DataFrame with all parameter combinations and their metrics.
    """
    combos = list(itertools.product(
        PARAM_GRID["k"], PARAM_GRID["metric"], PARAM_GRID["weights"]
    ))
    total = len(combos)
    all_results = []

    for i, (k, metric, weights) in enumerate(combos):
        print(f"[{i + 1}/{total}] K={k}, metric={metric}, weights={weights}...")

        try:
            results = run_walk_forward_backtest(
                df,
                feature_cols=feature_cols,
                k=k,
                metric=metric,
                weights=weights,
                training_window=training_window,
                start_date=start_date,
                end_date=end_date,
                verbose=False,
            )
            metrics = compute_metrics(results)
            metrics["k"] = k
            metrics["metric"] = metric
            metrics["weights"] = weights
            all_results.append(metrics)

            print(f"  → Sharpe: {metrics['sharpe_ratio']:.3f}, "
                  f"Accuracy: {metrics['accuracy']:.3f}, "
                  f"Annual: {metrics['annual_return']:.3f}")
        except Exception as e:
            print(f"  → ERROR: {e}")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    return results_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    features_path = project_root / "data" / "processed" / "features_master.csv"

    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    print(f"Loaded features: {len(df)} rows")
    print(f"Features: {BASELINE_FEATURES}")
    print(f"Grid: {len(PARAM_GRID['k'])} K × {len(PARAM_GRID['metric'])} metrics × "
          f"{len(PARAM_GRID['weights'])} weights = "
          f"{len(PARAM_GRID['k']) * len(PARAM_GRID['metric']) * len(PARAM_GRID['weights'])} combos\n")

    results_df = run_grid_search(df, BASELINE_FEATURES)

    # Save all results
    results_path = project_root / "models" / "tuning_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nAll results saved to {results_path}")

    # Save best config
    best = results_df.iloc[0]
    best_config = {
        "k": int(best["k"]),
        "metric": best["metric"],
        "weights": best["weights"],
        "sharpe_ratio": float(best["sharpe_ratio"]),
        "accuracy": float(best["accuracy"]),
        "annual_return": float(best["annual_return"]),
        "max_drawdown": float(best["max_drawdown"]),
    }
    config_path = project_root / "models" / "best_config.json"
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Best config saved to {config_path}")

    print(f"\n=== Top 5 Configurations (by Sharpe) ===")
    print(results_df[["k", "metric", "weights", "sharpe_ratio", "accuracy",
                       "annual_return", "max_drawdown"]].head(10).to_string(index=False))
