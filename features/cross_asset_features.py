"""Cross-asset features: relative strength vs SPY/IWM, bond/gold signals."""

from pathlib import Path

import pandas as pd


def compute_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-asset features from auxiliary ticker data.

    Args:
        df: Master dataset with QQQ, SPY, IWM, TLT, GLD, SMH, HYG, UUP columns.

    Returns:
        DataFrame with cross-asset feature columns added.
    """
    qqq_close = df["Close"]
    spy_close = df["SPY_Close"]
    iwm_close = df["IWM_Close"]
    tlt_close = df["TLT_Close"]
    gld_close = df["GLD_Close"]

    # QQQ/SPY relative strength (10-day ratio change)
    qqq_spy_ratio = qqq_close / spy_close
    df["feat_qqq_spy_rs_10d"] = qqq_spy_ratio.pct_change(10)

    # QQQ/IWM relative strength (10-day ratio change)
    qqq_iwm_ratio = qqq_close / iwm_close
    df["feat_qqq_iwm_rs_10d"] = qqq_iwm_ratio.pct_change(10)

    # TLT 5-day return (bond market signal)
    df["feat_tlt_return_5d"] = tlt_close.pct_change(5)

    # GLD 5-day return (risk-off signal)
    df["feat_gld_return_5d"] = gld_close.pct_change(5)

    # SPY-QQQ return spread (5-day)
    spy_ret_5d = spy_close.pct_change(5)
    qqq_ret_5d = qqq_close.pct_change(5)
    df["feat_spy_qqq_spread_5d"] = spy_ret_5d - qqq_ret_5d

    # --- Macro proxy features ---

    # SMH/QQQ relative strength (AI/semiconductor momentum)
    if "SMH_Close" in df.columns:
        smh_close = df["SMH_Close"]
        smh_qqq_ratio = smh_close / qqq_close
        df["feat_smh_qqq_rs_10d"] = smh_qqq_ratio.pct_change(10)
        df["feat_smh_return_5d"] = smh_close.pct_change(5)

    # HYG/TLT spread (credit risk / risk appetite)
    if "HYG_Close" in df.columns:
        hyg_close = df["HYG_Close"]
        hyg_ret_5d = hyg_close.pct_change(5)
        tlt_ret_5d = tlt_close.pct_change(5)
        df["feat_hyg_tlt_spread_5d"] = hyg_ret_5d - tlt_ret_5d

    # UUP returns (dollar strength / tariff proxy)
    if "UUP_Close" in df.columns:
        uup_close = df["UUP_Close"]
        df["feat_uup_return_5d"] = uup_close.pct_change(5)
        df["feat_uup_return_10d"] = uup_close.pct_change(10)

    return df


def save_cross_asset_features(df: pd.DataFrame, output_path: Path) -> None:
    """Extract and save only the cross-asset feature columns.

    Args:
        df: DataFrame with cross-asset features computed.
        output_path: Path to save CSV.
    """
    ca_cols = [
        "feat_qqq_spy_rs_10d", "feat_qqq_iwm_rs_10d",
        "feat_tlt_return_5d", "feat_gld_return_5d", "feat_spy_qqq_spread_5d",
        "feat_smh_qqq_rs_10d", "feat_smh_return_5d",
        "feat_hyg_tlt_spread_5d",
        "feat_uup_return_5d", "feat_uup_return_10d",
    ]
    # Only include columns that exist (in case tickers aren't downloaded yet)
    ca_cols = [c for c in ca_cols if c in df.columns]
    feat_df = df[ca_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_path)

    print(f"Cross-asset features saved: {len(ca_cols)} features, {len(feat_df)} rows")
    print(f"Features: {ca_cols}")

    nan_counts = feat_df.isna().sum()
    print(f"\nNaN counts per feature:")
    for col in ca_cols:
        print(f"  {col}: {nan_counts[col]}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    master_path = project_root / "data" / "processed" / "master_daily.csv"
    output_path = project_root / "features" / "cross_asset_features.csv"

    df = pd.read_csv(master_path, index_col=0, parse_dates=True)
    print(f"Loaded master dataset: {len(df)} rows\n")

    df = compute_cross_asset_features(df)
    save_cross_asset_features(df, output_path)
