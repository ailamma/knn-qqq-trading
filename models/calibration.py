"""Confidence calibration: convert predict_proba to actionable confidence scores."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.knn_model import run_walk_forward_backtest, compute_metrics


def load_config(project_root: Path) -> dict:
    """Load best model configuration.

    Args:
        project_root: Project root directory.

    Returns:
        Configuration dictionary.
    """
    with open(project_root / "models" / "best_config.json") as f:
        return json.load(f)


def analyze_confidence(results: pd.DataFrame) -> dict:
    """Analyze predict_proba distribution and calibration.

    Args:
        results: Walk-forward backtest results with prob_up column.

    Returns:
        Calibration analysis dictionary.
    """
    probs = results["prob_up"]
    actual = results["actual"]

    # Bin probabilities and compute actual frequency
    bins = np.arange(0.3, 0.75, 0.05)
    bin_labels = [f"{b:.2f}-{b + 0.05:.2f}" for b in bins[:-1]]

    calibration_data = []
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            calibration_data.append({
                "bin": bin_labels[i],
                "bin_center": (bins[i] + bins[i + 1]) / 2,
                "predicted_prob": probs[mask].mean(),
                "actual_freq": actual[mask].mean(),
                "count": int(mask.sum()),
            })

    return {
        "prob_mean": float(probs.mean()),
        "prob_std": float(probs.std()),
        "prob_min": float(probs.min()),
        "prob_max": float(probs.max()),
        "calibration_bins": calibration_data,
    }


def test_thresholds(
    results: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    """Test different confidence thresholds for trade/no-trade decisions.

    For each threshold, only trades when prob_up > threshold (long) or
    prob_up < (1-threshold) (short). Otherwise stays in cash.

    Args:
        results: Walk-forward backtest results.
        thresholds: List of probability thresholds to test.

    Returns:
        DataFrame with metrics at each threshold.
    """
    all_metrics = []

    for threshold in thresholds:
        r = results.copy()
        # Trade only when confident: prob_up > threshold (long) or < 1-threshold (short)
        long_mask = r["prob_up"] >= threshold
        short_mask = r["prob_up"] <= (1 - threshold)
        trade_mask = long_mask | short_mask

        if trade_mask.sum() == 0:
            continue

        traded = r[trade_mask].copy()
        strategy_returns = traded["actual_return"] * np.where(
            traded["prob_up"] >= threshold, 1, -1
        )

        # Compute metrics on traded days only
        n_traded = len(traded)
        trade_freq = n_traded / len(results)
        win_rate = (strategy_returns > 0).mean()

        total_return = (1 + strategy_returns).prod() - 1
        n_years = len(results) / 252  # Use full period for annualization
        annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        excess = strategy_returns - 0.04 / 252
        sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        gross_profit = strategy_returns[strategy_returns > 0].sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        all_metrics.append({
            "threshold": threshold,
            "n_trades": n_traded,
            "trade_frequency": round(trade_freq, 4),
            "win_rate": round(win_rate, 4),
            "sharpe_ratio": round(sharpe, 4),
            "annual_return": round(annual_return, 4),
            "total_return": round(total_return, 4),
            "max_drawdown": round(max_dd, 4),
            "profit_factor": round(profit_factor, 4),
        })

    return pd.DataFrame(all_metrics)


def plot_calibration(
    calibration_data: list[dict],
    output_path: Path,
) -> None:
    """Plot calibration curve.

    Args:
        calibration_data: List of calibration bin dicts.
        output_path: Path to save plot.
    """
    if not calibration_data:
        return

    predicted = [d["predicted_prob"] for d in calibration_data]
    actual = [d["actual_freq"] for d in calibration_data]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0.3, 0.7], [0.3, 0.7], "k--", alpha=0.5, label="Perfect calibration")
    ax.scatter(predicted, actual, s=80, zorder=5)
    ax.plot(predicted, actual, "b-", alpha=0.7, label="Model calibration")

    ax.set_xlabel("Predicted probability (up)")
    ax.set_ylabel("Actual frequency (up)")
    ax.set_title("KNN Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    features_path = project_root / "data" / "processed" / "features_master.csv"

    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    config = load_config(project_root)
    features = config["features"]

    print(f"Loaded: {len(df)} rows")
    print(f"Config: K={config['k']}, metric={config['metric']}, "
          f"weights={config['weights']}, window={config['training_window']}")
    print(f"Features: {features}")

    # Run walk-forward with best config
    print("\nRunning walk-forward backtest...")
    results = run_walk_forward_backtest(
        df, feature_cols=features,
        k=config["k"], metric=config["metric"], weights=config["weights"],
        training_window=config["training_window"],
        start_date="2020-01-01", end_date="2025-12-31",
    )

    # Analyze confidence distribution
    print("\n=== Confidence Analysis ===")
    cal_analysis = analyze_confidence(results)
    print(f"Prob_up distribution: mean={cal_analysis['prob_mean']:.3f}, "
          f"std={cal_analysis['prob_std']:.3f}, "
          f"range=[{cal_analysis['prob_min']:.3f}, {cal_analysis['prob_max']:.3f}]")

    # Plot calibration curve
    plot_path = project_root / "models" / "calibration_curve.png"
    plot_calibration(cal_analysis["calibration_bins"], plot_path)
    print(f"Calibration curve saved to {plot_path}")

    # Test thresholds
    print("\n=== Threshold Analysis ===")
    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60]
    threshold_results = test_thresholds(results, thresholds)
    print(threshold_results.to_string(index=False))

    # Save calibration data
    cal_output = {
        "confidence_analysis": cal_analysis,
        "threshold_results": threshold_results.to_dict(orient="records"),
        "optimal_threshold": float(
            threshold_results.loc[threshold_results["sharpe_ratio"].idxmax(), "threshold"]
        ),
    }

    cal_path = project_root / "models" / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(cal_output, f, indent=2)
    print(f"\nCalibration data saved to {cal_path}")

    # Update best config
    config["confidence_threshold"] = cal_output["optimal_threshold"]
    with open(project_root / "models" / "best_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Updated best_config.json with threshold={cal_output['optimal_threshold']}")
