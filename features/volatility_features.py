"""Volatility features: VIX levels, realized vol, Bollinger Band metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
import ta


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility features from QQQ and VIX data.

    Args:
        df: Master dataset with QQQ OHLCV and VIX columns.

    Returns:
        DataFrame with volatility feature columns added.
    """
    close = df["Close"]
    vix_close = df["VIX_Close"]

    # VIX level
    df["feat_vix_close"] = vix_close

    # VIX changes
    df["feat_vix_change_5d"] = vix_close.pct_change(5)
    df["feat_vix_change_10d"] = vix_close.pct_change(10)

    # Realized volatility (annualized) — using log returns
    log_returns = np.log(close / close.shift(1))
    df["feat_realized_vol_10d"] = log_returns.rolling(10).std() * np.sqrt(252)
    df["feat_realized_vol_20d"] = log_returns.rolling(20).std() * np.sqrt(252)

    # IV/RV ratio: VIX (implied vol) / 20-day realized vol
    # VIX is already annualized and in % terms, realized vol is in decimal
    df["feat_iv_rv_ratio"] = (vix_close / 100) / df["feat_realized_vol_20d"]

    # Bollinger Bands (20-day, 2 std)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()

    # Bollinger Band position: (Close - lower) / (upper - lower)
    bb_width_raw = bb_upper - bb_lower
    df["feat_bb_position"] = (close - bb_lower) / bb_width_raw.replace(0, float("nan"))

    # Bollinger Band width: (upper - lower) / mid
    df["feat_bb_width"] = bb_width_raw / bb_mid

    # VIX term structure: VIX / VIX3M ratio
    # > 1 = backwardation (fear/stress), < 1 = contango (normal/complacent)
    if "VIX3M_Close" in df.columns:
        vix3m_close = df["VIX3M_Close"]
        df["feat_vix_term_structure"] = vix_close / vix3m_close.replace(0, float("nan"))
        # 5-day change in term structure
        df["feat_vix_ts_change_5d"] = df["feat_vix_term_structure"].pct_change(5)

    return df


def save_volatility_features(df: pd.DataFrame, output_path: Path) -> None:
    """Extract and save only the volatility feature columns.

    Args:
        df: DataFrame with volatility features computed.
        output_path: Path to save CSV.
    """
    vol_cols = [
        "feat_vix_close", "feat_vix_change_5d", "feat_vix_change_10d",
        "feat_realized_vol_10d", "feat_realized_vol_20d", "feat_iv_rv_ratio",
        "feat_bb_position", "feat_bb_width",
    ]
    # Add term structure features if present
    if "feat_vix_term_structure" in df.columns:
        vol_cols.extend(["feat_vix_term_structure", "feat_vix_ts_change_5d"])
    feat_df = df[vol_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_path)

    print(f"Volatility features saved: {len(vol_cols)} features, {len(feat_df)} rows")
    print(f"Features: {vol_cols}")

    nan_counts = feat_df.isna().sum()
    print(f"\nNaN counts per feature:")
    for col in vol_cols:
        print(f"  {col}: {nan_counts[col]}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    master_path = project_root / "data" / "processed" / "master_daily.csv"
    output_path = project_root / "features" / "volatility_features.csv"

    df = pd.read_csv(master_path, index_col=0, parse_dates=True)
    print(f"Loaded master dataset: {len(df)} rows\n")

    df = compute_volatility_features(df)
    save_volatility_features(df, output_path)
