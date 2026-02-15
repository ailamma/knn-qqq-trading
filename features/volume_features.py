"""Volume features: volume ratio, OBV trend, volume-price divergence."""

from pathlib import Path

import numpy as np
import pandas as pd


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based features from QQQ data.

    Args:
        df: Master dataset with QQQ OHLCV columns.

    Returns:
        DataFrame with volume feature columns added.
    """
    close = df["Close"]
    volume = df["Volume"]

    # Volume ratio: today's volume / 20-day average volume
    vol_ma20 = volume.rolling(20).mean()
    df["feat_volume_ratio"] = volume / vol_ma20

    # 5-day volume trend (slope of linear regression on volume)
    def rolling_slope(s: pd.Series, window: int) -> pd.Series:
        x = np.arange(window, dtype=float)
        x -= x.mean()
        results = s.rolling(window).apply(
            lambda y: np.polyfit(x, y, 1)[0] if len(y) == window else np.nan,
            raw=True,
        )
        return results

    df["feat_volume_trend_5d"] = rolling_slope(volume, 5)

    # On-Balance Volume (OBV) 10-day rate of change
    price_direction = np.sign(close.diff())
    obv = (price_direction * volume).cumsum()
    obv_10d_ago = obv.shift(10)
    df["feat_obv_roc_10d"] = (obv - obv_10d_ago) / obv_10d_ago.abs().replace(0, float("nan"))

    # Volume-price divergence: sign(price_change) != sign(volume_change)
    price_change_sign = np.sign(close.diff())
    volume_change_sign = np.sign(volume.diff())
    df["feat_vol_price_divergence"] = (price_change_sign != volume_change_sign).astype(int)

    return df


def save_volume_features(df: pd.DataFrame, output_path: Path) -> None:
    """Extract and save only the volume feature columns.

    Args:
        df: DataFrame with volume features computed.
        output_path: Path to save CSV.
    """
    vol_cols = [
        "feat_volume_ratio", "feat_volume_trend_5d",
        "feat_obv_roc_10d", "feat_vol_price_divergence",
    ]
    feat_df = df[vol_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_path)

    print(f"Volume features saved: {len(vol_cols)} features, {len(feat_df)} rows")
    print(f"Features: {vol_cols}")

    nan_counts = feat_df.isna().sum()
    print(f"\nNaN counts per feature:")
    for col in vol_cols:
        print(f"  {col}: {nan_counts[col]}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    master_path = project_root / "data" / "processed" / "master_daily.csv"
    output_path = project_root / "features" / "volume_features.csv"

    df = pd.read_csv(master_path, index_col=0, parse_dates=True)
    print(f"Loaded master dataset: {len(df)} rows\n")

    df = compute_volume_features(df)
    save_volume_features(df, output_path)
