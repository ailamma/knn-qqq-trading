"""Create merged master dataset with aligned dates across all tickers."""

from pathlib import Path

import pandas as pd


TICKER_FILES = {
    "QQQ": "qqq_daily.csv",
    "VIX": "vix_daily.csv",
    "SPY": "spy_daily.csv",
    "IWM": "iwm_daily.csv",
    "TLT": "tlt_daily.csv",
    "GLD": "gld_daily.csv",
}


def load_ticker(raw_dir: Path, ticker: str, filename: str) -> pd.DataFrame:
    """Load a ticker CSV and prefix columns with ticker name.

    Args:
        raw_dir: Directory containing raw CSV files.
        ticker: Ticker symbol used as column prefix.
        filename: CSV filename.

    Returns:
        DataFrame with prefixed columns and Date index.
    """
    df = pd.read_csv(raw_dir / filename, index_col=0, parse_dates=True)
    # Prefix all columns except for QQQ (keep as-is for primary ticker)
    if ticker == "QQQ":
        return df
    df.columns = [f"{ticker}_{col}" for col in df.columns]
    return df


def merge_all(raw_dir: Path) -> pd.DataFrame:
    """Inner-join all ticker datasets on date.

    Args:
        raw_dir: Directory containing raw CSV files.

    Returns:
        Merged DataFrame with no NaN values.
    """
    dfs = []
    for ticker, filename in TICKER_FILES.items():
        df = load_ticker(raw_dir, ticker, filename)
        dfs.append(df)
        print(f"  Loaded {ticker}: {len(df)} rows, {len(df.columns)} cols")

    # Inner join on date — only keeps dates present in ALL datasets
    master = dfs[0]
    for df in dfs[1:]:
        master = master.join(df, how="inner")

    return master


def validate_and_save(master: pd.DataFrame, output_path: Path) -> None:
    """Validate merged dataset and save to CSV.

    Args:
        master: Merged DataFrame.
        output_path: Path to save the CSV.
    """
    print(f"\n--- Validation Report ---")
    print(f"Row count: {len(master)}")
    print(f"Column count: {len(master.columns)}")
    print(f"Date range: {master.index[0].date()} to {master.index[-1].date()}")

    nan_total = master.isna().sum().sum()
    if nan_total > 0:
        nan_by_col = master.isna().sum()
        print(f"WARNING: {nan_total} NaN values found:")
        print(nan_by_col[nan_by_col > 0])
    else:
        print("No NaN values — dataset is clean.")

    print(f"\nColumns:\n  {list(master.columns)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path)
    print(f"\nSaved to {output_path}")
    print(f"--- Validation complete ---")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    output = project_root / "data" / "processed" / "master_daily.csv"

    print("Merging all tickers (inner join on date)...\n")
    master = merge_all(raw_dir)
    validate_and_save(master, output)
