"""KNN classifier for QQQ next-day direction prediction."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.walk_forward import WalkForwardSplitter


# Default baseline features
BASELINE_FEATURES = [
    "feat_daily_return",
    "feat_rsi_14",
    "feat_vix_close",
    "feat_volume_ratio",
    "feat_dist_sma20",
]


ENSEMBLE_KS = [11, 15, 21]
RECENCY_DECAY_DAYS = 180


def _recency_weighted_proba(knn, X_test, y_train, train_dates, decay_days=RECENCY_DECAY_DAYS):
    """Compute recency-weighted P(up) from KNN neighbors.

    Args:
        knn: Fitted KNeighborsClassifier.
        X_test: Test feature vector (1, n_features).
        y_train: Training labels.
        train_dates: DatetimeIndex of training rows.
        decay_days: Exponential decay half-life in days.

    Returns:
        Recency-weighted probability of UP.
    """
    distances, indices = knn.kneighbors(X_test)
    indices = indices[0]

    neighbor_dates = train_dates[indices]
    last_train_date = train_dates[-1]
    ages = (last_train_date - neighbor_dates).days.astype(float)

    weights = np.exp(-ages / decay_days)
    neighbor_labels = y_train[indices]

    return float(np.sum(weights * neighbor_labels) / np.sum(weights))


def run_walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    k: int = 10,
    metric: str = "minkowski",
    weights: str = "distance",
    training_window: int = 500,
    start_date: str | None = None,
    end_date: str | None = None,
    verbose: bool = True,
    ensemble_ks: list[int] | None = None,
    recency_decay_days: int = RECENCY_DECAY_DAYS,
) -> pd.DataFrame:
    """Run walk-forward backtest with KNN classifier.

    For each test day, fits a fresh KNN on the trailing training window,
    scales features using only training data, and predicts the test day.

    When ensemble_ks is provided, fits multiple KNNs and averages their
    recency-weighted probabilities.

    Args:
        df: Features master DataFrame with target column.
        feature_cols: List of feature column names. Defaults to BASELINE_FEATURES.
        k: Number of neighbors (used when ensemble_ks is None).
        metric: Distance metric.
        weights: Weight function (uniform or distance).
        training_window: Number of training days.
        start_date: Start date for test predictions.
        end_date: End date for test predictions.
        verbose: Print progress updates.
        ensemble_ks: List of K values for multi-K ensemble. None = single K.
        recency_decay_days: Decay for recency weighting. 0 = no recency weighting.

    Returns:
        DataFrame with columns: date, prediction, actual, probability, correct.
    """
    if feature_cols is None:
        feature_cols = BASELINE_FEATURES

    use_ensemble = ensemble_ks is not None and len(ensemble_ks) > 1
    use_recency = recency_decay_days > 0
    k_values = ensemble_ks if use_ensemble else [k]

    X = df[feature_cols].values
    y = df["target"].values
    dates = df.index

    splitter = WalkForwardSplitter(training_window=training_window, test_window=1)
    results = []

    splits = list(splitter.split_by_date(df, start_date=start_date, end_date=end_date))
    total = len(splits)

    for i, (train_idx, test_idx) in enumerate(splits):
        # Scale: fit on train only, transform both
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        y_train = y[train_idx]
        y_test = y[test_idx]
        train_dates = dates[train_idx]

        # Multi-K ensemble with optional recency weighting
        k_probs = []
        for kv in k_values:
            knn = KNeighborsClassifier(n_neighbors=kv, metric=metric, weights="uniform")
            knn.fit(X_train, y_train)

            if use_recency:
                prob_up = _recency_weighted_proba(
                    knn, X_test, y_train, train_dates, recency_decay_days
                )
            else:
                proba = knn.predict_proba(X_test)[0]
                prob_up = float(proba[1] if len(proba) > 1 else proba[0])
            k_probs.append(prob_up)

        prob_up = float(np.mean(k_probs))
        pred = 1 if prob_up > 0.5 else 0

        results.append({
            "date": dates[test_idx[0]],
            "prediction": int(pred),
            "actual": int(y_test[0]),
            "prob_up": prob_up,
            "correct": int(pred == y_test[0]),
            "actual_return": float(df["next_day_return"].iloc[test_idx[0]]),
        })

        if verbose and (i + 1) % 500 == 0:
            acc = np.mean([r["correct"] for r in results])
            print(f"  Progress: {i + 1}/{total} days, running accuracy: {acc:.3f}")

    results_df = pd.DataFrame(results).set_index("date")
    return results_df


def compute_metrics(results: pd.DataFrame) -> dict[str, Any]:
    """Compute classification and trading metrics from backtest results.

    Args:
        results: DataFrame from run_walk_forward_backtest.

    Returns:
        Dictionary of performance metrics.
    """
    y_true = results["actual"]
    y_pred = results["prediction"]
    returns = results["actual_return"]

    # Strategy return: go long on predict=1, short on predict=0
    strategy_returns = returns * np.where(y_pred == 1, 1, -1)

    # Classification metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Trading metrics
    total_return = (1 + strategy_returns).prod() - 1
    n_years = len(results) / 252
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # Sharpe ratio (annualized, risk-free = 4%)
    excess = strategy_returns - 0.04 / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    # Profit factor
    gross_profit = strategy_returns[strategy_returns > 0].sum()
    gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "total_return": round(total_return, 4),
        "annual_return": round(annual_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(results["correct"].mean(), 4),
        "n_days": len(results),
        "n_years": round(n_years, 1),
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    """Pretty-print backtest metrics.

    Args:
        metrics: Dictionary from compute_metrics.
    """
    print("\n=== Backtest Results ===")
    print(f"Period: {metrics['n_days']} days ({metrics['n_years']} years)")
    print(f"\nClassification:")
    print(f"  Accuracy:     {metrics['accuracy']:.2%}")
    print(f"  Precision:    {metrics['precision']:.2%}")
    print(f"  Recall:       {metrics['recall']:.2%}")
    print(f"  F1 Score:     {metrics['f1_score']:.2%}")
    print(f"\nTrading:")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Annual Return:{metrics['annual_return']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Profit Factor:{metrics['profit_factor']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Win Rate:     {metrics['win_rate']:.2%}")
    print("========================\n")


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    features_path = project_root / "data" / "processed" / "features_master.csv"

    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    print(f"Loaded features: {len(df)} rows, {len(df.columns)} columns")
    print(f"Using baseline features: {BASELINE_FEATURES}")

    print(f"\nRunning walk-forward backtest (2020-01-01 to 2025-12-31)...")
    results = run_walk_forward_backtest(
        df,
        feature_cols=BASELINE_FEATURES,
        k=10,
        weights="distance",
        training_window=500,
        start_date="2020-01-01",
        end_date="2025-12-31",
    )

    metrics = compute_metrics(results)
    print_metrics(metrics)

    # Save results
    results_path = project_root / "models" / "baseline_results.csv"
    results.to_csv(results_path)
    print(f"Results saved to {results_path}")
