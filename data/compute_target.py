"""Compute target variable: QQQ next-day close-to-close return and binary direction."""

from pathlib import Path

import pandas as pd


def compute_target(master_path: Path, output_path: Path) -> pd.DataFrame:
    """Add next-day return and binary target columns to master dataset.

    Target is shifted so that row T contains tomorrow's return (T+1),
    allowing the model to predict using features available at close on day T.

    Args:
        master_path: Path to master_daily.csv.
        output_path: Path to save updated CSV.

    Returns:
        DataFrame with target columns added.
    """
    df = pd.read_csv(master_path, index_col=0, parse_dates=True)
    print(f"Loaded master dataset: {len(df)} rows")

    # Daily return (close-to-close, same day — useful as a feature later)
    df["daily_return"] = df["Close"].pct_change()

    # Next-day return: shift(-1) so row T holds the return from T to T+1
    df["next_day_return"] = df["Close"].pct_change().shift(-1)

    # Binary target: 1 if next day is positive, 0 otherwise
    df["target"] = (df["next_day_return"] > 0).astype(int)

    # Drop the last row (no next-day return available)
    df = df.iloc[:-1]

    # Drop the first row (no daily_return available)
    df = df.iloc[1:]

    # Verify no NaN in target columns
    nan_count = df[["next_day_return", "target"]].isna().sum().sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values in target columns!")
    else:
        print("Target columns clean — no NaN values.")

    # Class balance
    pos_pct = df["target"].mean() * 100
    neg_pct = 100 - pos_pct
    print(f"\nClass balance:")
    print(f"  Positive days (target=1): {df['target'].sum()} ({pos_pct:.1f}%)")
    print(f"  Negative days (target=0): {(df['target'] == 0).sum()} ({neg_pct:.1f}%)")

    # Summary stats on next-day return
    print(f"\nNext-day return stats:")
    print(f"  Mean:   {df['next_day_return'].mean():.5f}")
    print(f"  Std:    {df['next_day_return'].std():.5f}")
    print(f"  Min:    {df['next_day_return'].min():.5f}")
    print(f"  Max:    {df['next_day_return'].max():.5f}")

    print(f"\nFinal dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    master_path = project_root / "data" / "processed" / "master_daily.csv"
    output_path = project_root / "data" / "processed" / "master_daily.csv"
    compute_target(master_path, output_path)
