"""Calendar features: day of week, month, options expiration, FOMC proximity."""

from pathlib import Path

import numpy as np
import pandas as pd


# Approximate FOMC meeting dates (month, day) — 8 meetings per year, typically
# on Wednesdays. We use a simplified approach: known meeting months.
# For exact dates we'd need a lookup table, but approximate days-since is sufficient.
FOMC_MONTHS_DAYS = [
    (1, 28), (3, 18), (5, 6), (6, 17), (7, 29), (9, 16), (11, 4), (12, 16)
]


def get_fomc_dates(years: list[int]) -> list[pd.Timestamp]:
    """Generate approximate FOMC meeting dates for given years.

    Args:
        years: List of years to generate dates for.

    Returns:
        Sorted list of approximate FOMC meeting timestamps.
    """
    dates = []
    for year in years:
        for month, day in FOMC_MONTHS_DAYS:
            try:
                dates.append(pd.Timestamp(year=year, month=month, day=day))
            except ValueError:
                # Handle invalid dates by using last day of month
                dates.append(pd.Timestamp(year=year, month=month, day=28))
    return sorted(dates)


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute calendar-based features.

    Args:
        df: Master dataset with DatetimeIndex.

    Returns:
        DataFrame with calendar feature columns added.
    """
    dates = pd.to_datetime(df.index)

    # Day of week (0=Monday, 4=Friday)
    df["feat_day_of_week"] = dates.dayofweek

    # Month (1-12)
    df["feat_month"] = dates.month

    # Is monthly options expiration week (3rd Friday of month)
    # Third Friday = first day that is Friday with day between 15-21
    def is_opex_week(dt: pd.Timestamp) -> bool:
        """Check if date falls in options expiration week (week of 3rd Friday)."""
        # Find 3rd Friday of the month
        first_day = dt.replace(day=1)
        # Days until Friday from first day (Friday=4)
        days_to_friday = (4 - first_day.dayofweek) % 7
        first_friday = first_day.day + days_to_friday
        third_friday = first_friday + 14
        # OpEx week: Monday through Friday of that week
        opex_monday = third_friday - 4
        return opex_monday <= dt.day <= third_friday

    df["feat_is_opex_week"] = [int(is_opex_week(d)) for d in dates]

    # Days since last FOMC meeting (approximate)
    years = sorted(set(dates.year))
    fomc_dates = get_fomc_dates(list(range(min(years) - 1, max(years) + 2)))

    def days_since_fomc(dt: pd.Timestamp) -> int:
        """Calculate days since the most recent FOMC meeting."""
        past = [f for f in fomc_dates if f <= dt]
        if not past:
            return 90  # Default if no prior meeting
        return (dt - past[-1]).days

    df["feat_days_since_fomc"] = [days_since_fomc(d) for d in dates]

    return df


def save_calendar_features(df: pd.DataFrame, output_path: Path) -> None:
    """Extract and save only the calendar feature columns.

    Args:
        df: DataFrame with calendar features computed.
        output_path: Path to save CSV.
    """
    cal_cols = [
        "feat_day_of_week", "feat_month",
        "feat_is_opex_week", "feat_days_since_fomc",
    ]
    feat_df = df[cal_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_path)

    print(f"Calendar features saved: {len(cal_cols)} features, {len(feat_df)} rows")
    print(f"Features: {cal_cols}")

    nan_counts = feat_df.isna().sum()
    print(f"\nNaN counts per feature:")
    for col in cal_cols:
        print(f"  {col}: {nan_counts[col]}")

    # Show distribution
    print(f"\nDay of week distribution:")
    print(df["feat_day_of_week"].value_counts().sort_index().to_string())
    print(f"\nOpEx week frequency: {df['feat_is_opex_week'].mean():.1%}")
    print(f"Mean days since FOMC: {df['feat_days_since_fomc'].mean():.1f}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    master_path = project_root / "data" / "processed" / "master_daily.csv"
    output_path = project_root / "features" / "calendar_features.csv"

    df = pd.read_csv(master_path, index_col=0, parse_dates=True)
    print(f"Loaded master dataset: {len(df)} rows\n")

    df = compute_calendar_features(df)
    save_calendar_features(df, output_path)
