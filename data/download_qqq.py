"""Download QQQ daily OHLCV data from 2011-01-01 to present using yfinance."""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf


def download_qqq_data(
    start_date: str = "1999-01-01",
    output_path: str = "data/raw/qqq_daily.csv",
) -> pd.DataFrame:
    """Download QQQ daily OHLCV data and save to CSV.

    Args:
        start_date: Start date for download in YYYY-MM-DD format.
        output_path: Path to save the CSV file.

    Returns:
        DataFrame with QQQ daily OHLCV data.
    """
    print(f"Downloading QQQ daily data from {start_date} to present...")
    ticker = yf.Ticker("QQQ")
    df = ticker.history(start=start_date, auto_adjust=False)

    if df.empty:
        print("ERROR: No data returned from yfinance.")
        sys.exit(1)

    # Keep standard OHLCV columns
    cols_to_keep = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in cols_to_keep if c in df.columns]]

    # Ensure index is named Date and is a date (not datetime)
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).date

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output)
    print(f"Saved to {output}")

    return df


def verify_data(df: pd.DataFrame) -> None:
    """Verify data completeness and log summary statistics.

    Args:
        df: QQQ daily OHLCV DataFrame.
    """
    print("\n--- Data Verification ---")
    print(f"Row count: {len(df)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")

    # Check for NaN values
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print(f"WARNING: NaN values found:\n{nan_counts[nan_counts > 0]}")
    else:
        print("No NaN values found.")

    # Check for gaps > 3 business days
    dates = pd.to_datetime(pd.Series(df.index))
    diffs = dates.diff().dt.days
    large_gaps = diffs[diffs > 5]  # 5 calendar days ≈ 3 business days + weekend
    if len(large_gaps) > 0:
        print(f"\nGaps > 3 business days ({len(large_gaps)} found):")
        for idx in large_gaps.index:
            gap_start = dates.iloc[idx - 1].date()
            gap_end = dates.iloc[idx].date()
            print(f"  {gap_start} → {gap_end} ({int(diffs.iloc[idx])} calendar days)")
    else:
        print("No gaps > 3 business days found.")

    print("--- Verification complete ---\n")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    output = project_root / "data" / "raw" / "qqq_daily.csv"
    df = download_qqq_data(output_path=str(output))
    verify_data(df)
