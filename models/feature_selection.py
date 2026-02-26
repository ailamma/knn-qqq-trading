"""Feature selection: forward selection to find optimal feature subset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from models.knn_model import run_walk_forward_backtest, compute_metrics, ENSEMBLE_KS


ALL_FEATURES = [
    # Price features
    "feat_daily_return", "feat_return_5d", "feat_return_10d", "feat_return_20d",
    "feat_dist_sma20", "feat_dist_sma50", "feat_dist_sma200",
    "feat_hl_range_pct", "feat_close_position",
    "feat_overnight_gap", "feat_intraday_return",
    # Momentum features
    "feat_rsi_14", "feat_macd_line", "feat_macd_signal", "feat_macd_hist",
    "feat_roc_10", "feat_roc_20", "feat_stoch_k", "feat_stoch_d", "feat_williams_r",
    # Volatility features
    "feat_vix_close", "feat_vix_change_5d", "feat_vix_change_10d",
    "feat_realized_vol_10d", "feat_realized_vol_20d", "feat_iv_rv_ratio",
    "feat_bb_position", "feat_bb_width",
    "feat_vix_term_structure", "feat_vix_ts_change_5d",
    # Volume features
    "feat_volume_ratio", "feat_volume_trend_5d", "feat_obv_roc_10d",
    "feat_vol_price_divergence",
    # Cross-asset features
    "feat_qqq_spy_rs_10d", "feat_qqq_iwm_rs_10d",
    "feat_tlt_return_5d", "feat_gld_return_5d", "feat_spy_qqq_spread_5d",
    # Macro proxy features
    "feat_smh_qqq_rs_10d", "feat_smh_return_5d",
    "feat_hyg_tlt_spread_5d",
    "feat_uup_return_5d", "feat_uup_return_10d",
    # Calendar features
    "feat_day_of_week", "feat_month", "feat_is_opex_week", "feat_days_since_fomc",
    # Regime features
    "feat_above_sma200", "feat_golden_cross", "feat_adx_14", "feat_days_below_sma200",
]

# Manual baseline (from F013)
MANUAL_5 = [
    "feat_daily_return", "feat_rsi_14", "feat_vix_close",
    "feat_volume_ratio", "feat_dist_sma20",
]


def mutual_info_ranking(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Rank features by mutual information with the target.

    Args:
        df: Features master DataFrame.
        features: List of feature column names.

    Returns:
        DataFrame with feature names and MI scores, sorted descending.
    """
    X = df[features].values
    y = df["target"].values

    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=7)

    ranking = pd.DataFrame({
        "feature": features,
        "mi_score": mi_scores,
    }).sort_values("mi_score", ascending=False).reset_index(drop=True)

    return ranking


def forward_feature_selection(
    df: pd.DataFrame,
    candidate_features: list[str],
    max_features: int = 12,
    k: int = 7,
    metric: str = "manhattan",
    weights: str = "uniform",
    training_window: int = 500,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
) -> tuple[list[str], pd.DataFrame]:
    """Greedy forward feature selection maximizing walk-forward Sharpe ratio.

    Args:
        df: Features master DataFrame.
        candidate_features: Pool of candidate features.
        max_features: Maximum number of features to select.
        k: KNN neighbors.
        metric: Distance metric.
        weights: Weight function.
        training_window: Training window size.
        start_date: Backtest start.
        end_date: Backtest end.

    Returns:
        Tuple of (selected feature list, selection log DataFrame).
    """
    selected: list[str] = []
    remaining = list(candidate_features)
    log_entries = []

    for step in range(1, max_features + 1):
        best_sharpe = -999.0
        best_feature = ""
        best_metrics: dict = {}

        print(f"\n--- Step {step}: testing {len(remaining)} candidates ---")

        for feat in remaining:
            trial = selected + [feat]
            try:
                results = run_walk_forward_backtest(
                    df, feature_cols=trial, k=k, metric=metric, weights=weights,
                    training_window=training_window, start_date=start_date,
                    end_date=end_date, verbose=False,
                    ensemble_ks=ENSEMBLE_KS, recency_decay_days=0,
                )
                metrics = compute_metrics(results)
                sharpe = metrics["sharpe_ratio"]

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_feature = feat
                    best_metrics = metrics
            except Exception as e:
                print(f"  ERROR with {feat}: {e}")

        if not best_feature:
            print("No improvement found, stopping.")
            break

        selected.append(best_feature)
        remaining.remove(best_feature)

        log_entries.append({
            "step": step,
            "feature_added": best_feature,
            "n_features": len(selected),
            "sharpe_ratio": best_metrics.get("sharpe_ratio", 0),
            "accuracy": best_metrics.get("accuracy", 0),
            "annual_return": best_metrics.get("annual_return", 0),
            "max_drawdown": best_metrics.get("max_drawdown", 0),
        })

        print(f"  → Added: {best_feature}")
        print(f"    Sharpe: {best_sharpe:.3f}, "
              f"Accuracy: {best_metrics.get('accuracy', 0):.3f}, "
              f"Annual: {best_metrics.get('annual_return', 0):.3f}")

    log_df = pd.DataFrame(log_entries)
    return selected, log_df


def compare_feature_sets(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    k: int = 7,
    metric: str = "manhattan",
    weights: str = "uniform",
) -> pd.DataFrame:
    """Compare multiple feature sets via walk-forward backtest.

    Args:
        df: Features master DataFrame.
        feature_sets: Dict mapping set name to feature list.
        k: KNN neighbors.
        metric: Distance metric.
        weights: Weight function.

    Returns:
        DataFrame comparing all feature sets.
    """
    comparisons = []
    for name, features in feature_sets.items():
        print(f"\nEvaluating '{name}' ({len(features)} features)...")
        results = run_walk_forward_backtest(
            df, feature_cols=features, k=k, metric=metric, weights=weights,
            training_window=500, start_date="2020-01-01", end_date="2025-12-31",
            verbose=False, ensemble_ks=ENSEMBLE_KS, recency_decay_days=0,
        )
        metrics = compute_metrics(results)
        metrics["set_name"] = name
        metrics["n_features"] = len(features)
        comparisons.append(metrics)

    return pd.DataFrame(comparisons)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    features_path = project_root / "data" / "processed" / "features_master.csv"

    df = pd.read_csv(features_path, index_col=0, parse_dates=True)

    # Filter ALL_FEATURES to only those present in the data
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"Skipping {len(missing)} features not in data: {missing}")
    print(f"Loaded features: {len(df)} rows, {len(available_features)} candidate features")

    # Step 1: Mutual information ranking as pre-filter
    print("\n=== Mutual Information Ranking ===")
    mi_ranking = mutual_info_ranking(df, available_features)
    print(mi_ranking.to_string(index=False))

    # Use top 20 by MI as candidates for forward selection (reduces search space)
    top_mi_features = mi_ranking.head(20)["feature"].tolist()
    print(f"\nTop 20 MI features selected as candidates for forward selection")

    # Step 2: Forward feature selection (greedy, max 12)
    print("\n=== Forward Feature Selection ===")
    selected, selection_log = forward_feature_selection(
        df, candidate_features=top_mi_features, max_features=12,
    )
    print(f"\nSelected features ({len(selected)}):")
    for i, f in enumerate(selected, 1):
        print(f"  {i}. {f}")

    # Step 3: Compare feature sets
    print("\n=== Feature Set Comparison ===")
    top_8 = selected[:8]
    top_12 = selected[:12]

    comparison = compare_feature_sets(df, {
        "manual_5": MANUAL_5,
        "top_8": top_8,
        "top_12": top_12,
        "all_available": available_features,
    })

    print("\n" + comparison[["set_name", "n_features", "sharpe_ratio", "accuracy",
                             "annual_return", "max_drawdown"]].to_string(index=False))

    # Save results
    selected_path = project_root / "models" / "selected_features.json"
    with open(selected_path, "w") as f:
        json.dump({
            "selected_features": selected,
            "top_8": top_8,
            "top_12": top_12,
            "selection_log": selection_log.to_dict(orient="records"),
            "mi_ranking": mi_ranking.to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nSaved to {selected_path}")
