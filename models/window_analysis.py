"""Rolling window optimization: find optimal training window size."""

import json
from pathlib import Path

import pandas as pd

from models.knn_model import run_walk_forward_backtest, compute_metrics


# Load best feature set
def load_best_features(project_root: Path) -> list[str]:
    """Load selected features from feature selection results.

    Args:
        project_root: Project root directory.

    Returns:
        List of top-8 feature names.
    """
    path = project_root / "models" / "selected_features.json"
    with open(path) as f:
        data = json.load(f)
    return data["top_8"]


WINDOW_SIZES = [250, 500, 750, 1000, 1500]


def run_window_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_sizes: list[int],
    k: int = 7,
    metric: str = "manhattan",
    weights: str = "uniform",
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """Test different training window sizes and compare Sharpe ratios.

    Args:
        df: Features master DataFrame.
        feature_cols: Feature columns.
        window_sizes: List of training window sizes to test.
        k: KNN neighbors.
        metric: Distance metric.
        weights: Weight function.
        start_date: Backtest start.
        end_date: Backtest end.

    Returns:
        DataFrame with metrics for each window size.
    """
    all_results = []

    for window in window_sizes:
        print(f"\nTesting window={window} days...")
        results = run_walk_forward_backtest(
            df, feature_cols=feature_cols, k=k, metric=metric, weights=weights,
            training_window=window, start_date=start_date, end_date=end_date,
            verbose=False,
        )
        metrics = compute_metrics(results)
        metrics["training_window"] = window
        all_results.append(metrics)

        print(f"  → Sharpe: {metrics['sharpe_ratio']:.3f}, "
              f"Accuracy: {metrics['accuracy']:.3f}, "
              f"Annual: {metrics['annual_return']:.3f}, "
              f"MaxDD: {metrics['max_drawdown']:.3f}")

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    features_path = project_root / "data" / "processed" / "features_master.csv"

    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    features = load_best_features(project_root)
    print(f"Loaded features: {len(df)} rows")
    print(f"Using top-8 features: {features}")
    print(f"Testing windows: {WINDOW_SIZES}")

    results_df = run_window_analysis(df, features, WINDOW_SIZES)

    # Save
    output_path = project_root / "models" / "window_analysis.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Print comparison
    print("\n=== Window Size Comparison ===")
    print(results_df[["training_window", "sharpe_ratio", "accuracy",
                       "annual_return", "max_drawdown", "profit_factor"]].to_string(index=False))

    # Update best config with optimal window
    best_idx = results_df["sharpe_ratio"].idxmax()
    best_window = int(results_df.loc[best_idx, "training_window"])
    print(f"\nBest window: {best_window} days (Sharpe: {results_df.loc[best_idx, 'sharpe_ratio']:.3f})")

    config_path = project_root / "models" / "best_config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["training_window"] = best_window
    config["features"] = features
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Updated best_config.json with window={best_window}")
