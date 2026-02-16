"""Download TQQQ/SQQQ historical prices for position sizing backtest ONLY.

NOTE: This data is NEVER used in model training. It is used only for
position sizing simulation in Phase 5 and daily signal generation in Phase 6.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf


TICKERS = {"TQQQ": "tqqq_daily.csv", "SQQQ": "sqqq_daily.csv"}
START_DATE = "2010-01-01"


def download_leveraged_etfs(raw_dir: Path) -> None:
    """Download TQQQ and SQQQ daily OHLCV data.

    Args:
        raw_dir: Directory to save CSV files.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    for ticker, filename in TICKERS.items():
        print(f"Downloading {ticker}...")
        t = yf.Ticker(ticker)
        df = t.history(start=START_DATE, auto_adjust=False)

        if df.empty:
            print(f"  ERROR: No data for {ticker}")
            continue

        cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        df = df[[c for c in cols if c in df.columns]]
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index).date

        output = raw_dir / filename
        df.to_csv(output)
        print(f"  → {len(df)} rows, {df.index[0]} to {df.index[-1]}")

    # Verify alignment with QQQ
    qqq_path = raw_dir / "qqq_daily.csv"
    if qqq_path.exists():
        qqq = pd.read_csv(qqq_path, index_col=0, parse_dates=True)
        qqq_dates = set(qqq.index)
        print(f"\nAlignment with QQQ ({len(qqq)} rows):")
        for ticker, filename in TICKERS.items():
            path = raw_dir / filename
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            overlap = len(qqq_dates & set(df.index))
            print(f"  {ticker}: {overlap} overlapping dates")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    download_leveraged_etfs(raw_dir)
