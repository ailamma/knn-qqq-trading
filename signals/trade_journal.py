"""Trade journal: track recommendations and actual results."""

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOURNAL_DIR = PROJECT_ROOT / "signals" / "journal"
JOURNAL_PATH = JOURNAL_DIR / "trade_journal.csv"

COLUMNS = [
    "date", "signal", "ticker", "shares", "confidence",
    "entry_price", "exit_price", "actual_return", "pnl", "cumulative_pnl",
]


def init_journal() -> None:
    """Initialize trade journal CSV if it doesn't exist."""
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    if not JOURNAL_PATH.exists():
        with open(JOURNAL_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
        print(f"Journal initialized at {JOURNAL_PATH}")


def log_signal(signal: dict) -> None:
    """Log a daily signal recommendation to the journal.

    Args:
        signal: Signal dictionary from generate_daily_signal.
    """
    init_journal()
    rec = signal["recommendation"]

    row = {
        "date": signal["date"],
        "signal": rec["action"],
        "ticker": rec.get("ticker", ""),
        "shares": rec.get("shares", 0),
        "confidence": signal["prob_up"],
        "entry_price": "",
        "exit_price": "",
        "actual_return": "",
        "pnl": "",
        "cumulative_pnl": "",
    }

    with open(JOURNAL_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(row)

    print(f"Signal logged to journal: {row['date']} {row['signal']} {row['ticker']}")


def update_results(date: str, actual_return: float, exit_price: float) -> None:
    """Update journal entry with actual results after market close.

    Args:
        date: Trade date to update.
        actual_return: Actual return on the trade.
        exit_price: Exit price.
    """
    if not JOURNAL_PATH.exists():
        print("No journal found.")
        return

    df = pd.read_csv(JOURNAL_PATH)
    mask = df["date"] == date

    if mask.sum() == 0:
        print(f"No entry found for {date}")
        return

    idx = df[mask].index[0]
    df.loc[idx, "exit_price"] = exit_price
    df.loc[idx, "actual_return"] = round(actual_return, 6)

    shares = df.loc[idx, "shares"]
    entry = df.loc[idx, "entry_price"]
    if pd.notna(entry) and entry != "" and shares > 0:
        pnl = shares * (exit_price - float(entry))
        df.loc[idx, "pnl"] = round(pnl, 2)
    else:
        df.loc[idx, "pnl"] = 0

    # Recalculate cumulative P&L
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)
    df["cumulative_pnl"] = df["pnl"].cumsum()

    df.to_csv(JOURNAL_PATH, index=False)
    print(f"Results updated for {date}: return={actual_return:.4%}")


def weekly_summary() -> None:
    """Print weekly performance summary."""
    if not JOURNAL_PATH.exists():
        print("No journal found.")
        return

    df = pd.read_csv(JOURNAL_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)

    if len(df) == 0:
        print("No entries in journal.")
        return

    # Last 5 trading days
    recent = df.tail(5)
    print("\n=== Last 5 Trading Days ===")
    for _, row in recent.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        signal = row["signal"]
        ticker = row.get("ticker", "")
        pnl = row.get("pnl", 0)
        print(f"  {date_str}: {signal:5s} {ticker:4s} | P&L: ${pnl:>8.2f}")

    # Summary stats
    total_pnl = df["pnl"].sum()
    n_trades = (df["signal"] != "CASH").sum()
    if n_trades > 0:
        traded = df[df["signal"] != "CASH"]
        win_rate = (traded["pnl"] > 0).mean()
        print(f"\n  Total trades: {n_trades}")
        print(f"  Win rate:     {win_rate:.1%}")
    print(f"  Cumulative P&L: ${total_pnl:,.2f}")


if __name__ == "__main__":
    init_journal()
    weekly_summary()
