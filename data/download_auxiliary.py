"""Download auxiliary data (VIX, SPY, IWM, TLT, GLD) for cross-asset features."""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf


TICKERS = {
    "^VIX": "vix_daily.csv",
    "SPY": "spy_daily.csv",
    "IWM": "iwm_daily.csv",
    "TLT": "tlt_daily.csv",
    "GLD": "gld_daily.csv",
}

START_DATE = "1999-01-01"


def download_ticker(ticker: str, output_path: Path) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker.

    Args:
        ticker: Yahoo Finance ticker symbol.
        output_path: Path to save the CSV file.

    Returns:
        DataFrame with daily OHLCV data.
    """
    print(f"Downloading {ticker}...")
    t = yf.Ticker(ticker)
    df = t.history(start=START_DATE, auto_adjust=False)

    if df.empty:
        print(f"ERROR: No data returned for {ticker}.")
        sys.exit(1)

    # Keep standard columns (VIX only has Close/Open/High/Low, no Volume sometimes)
    cols_to_keep = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in cols_to_keep if c in df.columns]]

    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).date

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)

    return df


def verify_alignment(raw_dir: Path) -> None:
    """Verify date alignment across all downloaded tickers and QQQ.

    Args:
        raw_dir: Directory containing raw CSV files.
    """
    print("\n--- Date Alignment Check ---")

    qqq_path = raw_dir / "qqq_daily.csv"
    if not qqq_path.exists():
        print("WARNING: qqq_daily.csv not found, skipping alignment check.")
        return

    qqq = pd.read_csv(qqq_path, index_col=0, parse_dates=True)
    qqq_dates = set(qqq.index)

    for ticker, filename in TICKERS.items():
        path = raw_dir / filename
        if not path.exists():
            print(f"  {ticker}: FILE MISSING")
            continue

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        aux_dates = set(df.index)

        overlap = qqq_dates & aux_dates
        only_qqq = qqq_dates - aux_dates
        only_aux = aux_dates - qqq_dates

        print(f"  {ticker:6s}: {len(df)} rows | "
              f"range {df.index[0].date()} to {df.index[-1].date()} | "
              f"overlap with QQQ: {len(overlap)} | "
              f"QQQ-only: {len(only_qqq)} | {ticker}-only: {len(only_aux)}")

    print("--- Alignment check complete ---\n")


def check_missing_data(raw_dir: Path) -> None:
    """Check for NaN values and apply forward-fill (max 3 days) if needed.

    Args:
        raw_dir: Directory containing raw CSV files.
    """
    print("--- Missing Data Check ---")
    for ticker, filename in TICKERS.items():
        path = raw_dir / filename
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"  {ticker}: {nan_count} NaN values found, forward-filling (max 3 days)...")
            df = df.ffill(limit=3)
            remaining = df.isna().sum().sum()
            if remaining > 0:
                print(f"    WARNING: {remaining} NaN values remain after ffill(3)")
            else:
                print(f"    All NaN values filled.")
            df.to_csv(path)
        else:
            print(f"  {ticker}: No NaN values.")
    print("--- Missing data check complete ---\n")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"

    # Download all auxiliary tickers
    for ticker, filename in TICKERS.items():
        output = raw_dir / filename
        df = download_ticker(ticker, output)
        print(f"  → {len(df)} rows, {df.index[0]} to {df.index[-1]}")

    # Verify and clean
    check_missing_data(raw_dir)
    verify_alignment(raw_dir)
