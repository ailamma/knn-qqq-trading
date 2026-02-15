"""Merge all feature sets into single training-ready dataframe with no NaN rows."""

import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


FEATURE_FILES = [
    "price_features.csv",
    "momentum_features.csv",
    "volatility_features.csv",
    "volume_features.csv",
    "cross_asset_features.csv",
    "calendar_features.csv",
]


def merge_features(features_dir: Path, master_path: Path) -> pd.DataFrame:
    """Merge all feature CSVs with the master dataset.

    Args:
        features_dir: Directory containing feature CSV files.
        master_path: Path to master_daily.csv (for target columns).

    Returns:
        Merged DataFrame with all features and targets, no NaN rows.
    """
    # Load master for target columns
    master = pd.read_csv(master_path, index_col=0, parse_dates=True)
    target_cols = ["daily_return", "next_day_return", "target"]
    targets = master[target_cols]

    # Load and merge all feature files
    all_features = []
    for filename in FEATURE_FILES:
        path = features_dir / filename
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  Loaded {filename}: {len(df.columns)} features, {len(df)} rows")
        all_features.append(df)

    # Join all features on date
    merged = all_features[0]
    for df in all_features[1:]:
        merged = merged.join(df, how="inner")

    # Join with targets
    merged = merged.join(targets, how="inner")

    print(f"\nBefore NaN drop: {len(merged)} rows, {len(merged.columns)} columns")

    # Drop rows with any NaN
    nan_before = merged.isna().sum().sum()
    merged = merged.dropna()

    print(f"Dropped {nan_before} NaN values → {len(merged)} rows remain")

    return merged


def generate_reports(
    df: pd.DataFrame,
    features_dir: Path,
    output_path: Path,
) -> None:
    """Generate feature reports: registry, correlation heatmap.

    Args:
        df: Merged features DataFrame.
        features_dir: Directory to save feature registry.
        output_path: Path where merged CSV was saved.
    """
    feat_cols = [c for c in df.columns if c.startswith("feat_")]

    # Feature registry
    registry = {
        "total_features": len(feat_cols),
        "total_rows": len(df),
        "date_range": {
            "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()),
        },
        "features": [
            {
                "name": col,
                "mean": round(float(df[col].mean()), 6),
                "std": round(float(df[col].std()), 6),
                "min": round(float(df[col].min()), 6),
                "max": round(float(df[col].max()), 6),
            }
            for col in feat_cols
        ],
    }

    registry_path = features_dir / "feature_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\nFeature registry saved to {registry_path}")

    # Correlation heatmap
    corr = df[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr, annot=False, cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, ax=ax, square=True,
        xticklabels=[c.replace("feat_", "") for c in feat_cols],
        yticklabels=[c.replace("feat_", "") for c in feat_cols],
    )
    ax.set_title("Feature Correlation Matrix")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    plot_path = features_dir / "correlation_heatmap.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Correlation heatmap saved to {plot_path}")

    # Print summary
    print(f"\n--- Feature Summary ---")
    print(f"Total features: {len(feat_cols)}")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Target class balance: {df['target'].mean():.1%} positive")

    # Highly correlated pairs (|r| > 0.9)
    high_corr = []
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.9:
                high_corr.append((feat_cols[i], feat_cols[j], round(r, 3)))

    if high_corr:
        print(f"\nHighly correlated pairs (|r| > 0.9):")
        for a, b, r in sorted(high_corr, key=lambda x: -abs(x[2])):
            print(f"  {a} <-> {b}: {r}")
    print(f"--- Summary complete ---")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    features_dir = project_root / "features"
    master_path = project_root / "data" / "processed" / "master_daily.csv"
    output_path = project_root / "data" / "processed" / "features_master.csv"

    print("Merging all feature sets...\n")
    df = merge_features(features_dir, master_path)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"\nSaved to {output_path}")

    generate_reports(df, features_dir, output_path)
