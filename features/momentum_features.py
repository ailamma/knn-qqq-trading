"""Momentum and oscillator features: RSI, MACD, ROC, stochastic, Williams %R."""

from pathlib import Path

import pandas as pd
import ta


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum and oscillator features from QQQ OHLCV data.

    Uses the `ta` library for standard technical indicators.
    All features use only data available at market close on day T.

    Args:
        df: Master dataset with QQQ OHLCV columns.

    Returns:
        DataFrame with momentum feature columns added.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # RSI(14)
    df["feat_rsi_14"] = ta.momentum.rsi(close, window=14)

    # MACD(12, 26, 9)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["feat_macd_line"] = macd.macd()
    df["feat_macd_signal"] = macd.macd_signal()
    df["feat_macd_hist"] = macd.macd_diff()

    # Rate of Change (ROC) — 10-day and 20-day
    df["feat_roc_10"] = ta.momentum.roc(close, window=10)
    df["feat_roc_20"] = ta.momentum.roc(close, window=20)

    # Stochastic %K(14) and %D(3)
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["feat_stoch_k"] = stoch.stoch()
    df["feat_stoch_d"] = stoch.stoch_signal()

    # Williams %R(14)
    df["feat_williams_r"] = ta.momentum.williams_r(high, low, close, lbp=14)

    return df


def save_momentum_features(df: pd.DataFrame, output_path: Path) -> None:
    """Extract and save only the momentum feature columns.

    Args:
        df: DataFrame with momentum features computed.
        output_path: Path to save CSV.
    """
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    # Only keep momentum features (exclude price features if present)
    momentum_cols = [
        "feat_rsi_14", "feat_macd_line", "feat_macd_signal", "feat_macd_hist",
        "feat_roc_10", "feat_roc_20", "feat_stoch_k", "feat_stoch_d", "feat_williams_r",
    ]
    feat_df = df[momentum_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_path)

    print(f"Momentum features saved: {len(momentum_cols)} features, {len(feat_df)} rows")
    print(f"Features: {momentum_cols}")

    nan_counts = feat_df.isna().sum()
    print(f"\nNaN counts per feature:")
    for col in momentum_cols:
        print(f"  {col}: {nan_counts[col]}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    master_path = project_root / "data" / "processed" / "master_daily.csv"
    output_path = project_root / "features" / "momentum_features.csv"

    df = pd.read_csv(master_path, index_col=0, parse_dates=True)
    print(f"Loaded master dataset: {len(df)} rows\n")

    df = compute_momentum_features(df)
    save_momentum_features(df, output_path)
