"""Price-based features: returns, cumulative returns, distance from moving averages."""

from pathlib import Path

import pandas as pd


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price-based features from QQQ OHLCV data.

    All features use only data available at market close on day T.
    No look-ahead bias — no future data is used.

    Args:
        df: Master dataset with QQQ OHLCV columns.

    Returns:
        DataFrame with price feature columns added.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Daily return (close-to-close %)
    df["feat_daily_return"] = close.pct_change()

    # Multi-day cumulative returns
    df["feat_return_5d"] = close.pct_change(5)
    df["feat_return_10d"] = close.pct_change(10)
    df["feat_return_20d"] = close.pct_change(20)

    # Moving averages
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    # Distance from moving averages (as %)
    df["feat_dist_sma20"] = (close - sma20) / sma20
    df["feat_dist_sma50"] = (close - sma50) / sma50
    df["feat_dist_sma200"] = (close - sma200) / sma200

    # High-Low range as % of close
    df["feat_hl_range_pct"] = (high - low) / close

    # Close position within day's range: (C-L)/(H-L)
    hl_diff = high - low
    df["feat_close_position"] = (close - low) / hl_diff.replace(0, float("nan"))

    return df


def save_price_features(df: pd.DataFrame, output_path: Path) -> None:
    """Extract and save only the price feature columns.

    Args:
        df: DataFrame with price features computed.
        output_path: Path to save CSV.
    """
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    feat_df = df[feat_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_path)

    print(f"Price features saved: {len(feat_cols)} features, {len(feat_df)} rows")
    print(f"Features: {feat_cols}")

    # Report NaN counts (expected in first rows due to rolling windows)
    nan_counts = feat_df.isna().sum()
    print(f"\nNaN counts per feature (expected for rolling windows):")
    for col in feat_cols:
        print(f"  {col}: {nan_counts[col]}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    master_path = project_root / "data" / "processed" / "master_daily.csv"
    output_path = project_root / "features" / "price_features.csv"

    df = pd.read_csv(master_path, index_col=0, parse_dates=True)
    print(f"Loaded master dataset: {len(df)} rows\n")

    df = compute_price_features(df)
    save_price_features(df, output_path)
