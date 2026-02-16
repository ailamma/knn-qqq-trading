"""Regime detection features: trend state, SMA cross, ADX, bear persistence."""

from pathlib import Path

import pandas as pd
import ta


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend/regime features from QQQ OHLCV data.

    These features help the model distinguish bull from bear markets.
    All features use only data available at market close on day T.

    Args:
        df: Master dataset with QQQ OHLCV columns.

    Returns:
        DataFrame with regime feature columns added.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # 1. Price above/below 200-day SMA (binary: 1=above, 0=below)
    sma200 = close.rolling(200).mean()
    df["feat_above_sma200"] = (close > sma200).astype(float)

    # 2. 50/200 SMA cross state (binary: 1=golden cross, 0=death cross)
    sma50 = close.rolling(50).mean()
    df["feat_golden_cross"] = (sma50 > sma200).astype(float)

    # 3. ADX(14) — trend strength regardless of direction
    adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
    df["feat_adx_14"] = adx_indicator.adx()

    # 4. Days below 200-day SMA (rolling count, capped at 60)
    below_sma200 = (close < sma200).astype(int)
    # Count consecutive days below: use cumsum trick
    # Reset count when price crosses above SMA
    above_mask = close >= sma200
    # Group consecutive below-periods
    group_id = above_mask.cumsum()
    days_below = below_sma200.groupby(group_id).cumsum()
    df["feat_days_below_sma200"] = days_below.clip(upper=60).astype(float)

    return df


def save_regime_features(df: pd.DataFrame, output_path: Path) -> None:
    """Extract and save only the regime feature columns.

    Args:
        df: DataFrame with regime features computed.
        output_path: Path to save CSV.
    """
    regime_cols = [
        "feat_above_sma200",
        "feat_golden_cross",
        "feat_adx_14",
        "feat_days_below_sma200",
    ]
    feat_df = df[regime_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_path)

    print(f"Regime features saved: {len(regime_cols)} features, {len(feat_df)} rows")
    print(f"Features: {regime_cols}")

    nan_counts = feat_df.isna().sum()
    print(f"\nNaN counts per feature:")
    for col in regime_cols:
        print(f"  {col}: {nan_counts[col]}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    master_path = project_root / "data" / "processed" / "master_daily.csv"
    output_path = project_root / "features" / "regime_features.csv"

    df = pd.read_csv(master_path, index_col=0, parse_dates=True)
    print(f"Loaded master dataset: {len(df)} rows\n")

    df = compute_regime_features(df)
    save_regime_features(df, output_path)
