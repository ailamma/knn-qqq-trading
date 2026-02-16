"""Backtest full TQQQ/SQQQ strategy with discrete leverage levels on $50K account."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.knn_model import run_walk_forward_backtest
from signals.position_sizer import PositionSizer, LEVERAGE_LEVELS
from backtesting.metrics import compute_full_metrics, print_full_metrics


def load_leveraged_prices(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load TQQQ and SQQQ daily price data.

    Args:
        raw_dir: Directory with raw CSV files.

    Returns:
        Tuple of (TQQQ DataFrame, SQQQ DataFrame).
    """
    tqqq = pd.read_csv(raw_dir / "tqqq_daily.csv", index_col=0, parse_dates=True)
    sqqq = pd.read_csv(raw_dir / "sqqq_daily.csv", index_col=0, parse_dates=True)
    return tqqq, sqqq


def run_tqqq_sqqq_backtest(
    predictions: pd.DataFrame,
    tqqq: pd.DataFrame,
    sqqq: pd.DataFrame,
    sizer: PositionSizer,
    features_df: pd.DataFrame | None = None,
    initial_capital: float = 50000.0,
) -> pd.DataFrame:
    """Simulate TQQQ/SQQQ trading with discrete leverage levels.

    Args:
        predictions: Walk-forward predictions with prob_up and actual_return.
        tqqq: TQQQ daily price data.
        sqqq: SQQQ daily price data.
        sizer: PositionSizer instance.
        features_df: Features master for realized vol lookup. Optional.
        initial_capital: Starting account balance.

    Returns:
        Trade log DataFrame with equity curve.
    """
    cash = initial_capital
    current_leverage = 0
    records = []

    for date, row in predictions.iterrows():
        if date not in tqqq.index or date not in sqqq.index:
            continue

        tqqq_price = tqqq.loc[date, "Close"]
        sqqq_price = sqqq.loc[date, "Close"]

        # Get next-day prices for P&L
        tqqq_idx = tqqq.index.get_loc(date)
        sqqq_idx = sqqq.index.get_loc(date)
        if tqqq_idx + 1 >= len(tqqq) or sqqq_idx + 1 >= len(sqqq):
            continue

        tqqq_next = tqqq.iloc[tqqq_idx + 1]["Close"]
        sqqq_next = sqqq.iloc[sqqq_idx + 1]["Close"]

        # Get realized vol if available
        realized_vol = None
        if features_df is not None and date in features_df.index:
            vol_col = "feat_realized_vol_20d"
            if vol_col in features_df.columns:
                realized_vol = features_df.loc[date, vol_col]

        # Size position with state tracking
        total_equity = cash
        rec = sizer.size(
            row["prob_up"], total_equity, tqqq_price, sqqq_price,
            realized_vol_annual=realized_vol,
            current_leverage=current_leverage,
        )

        leverage = rec["leverage_level"]
        ticker = rec["ticker"]
        shares = rec["shares"]

        # Calculate P&L based on allocation
        if ticker == "TQQQ" and shares > 0:
            cost_basis = shares * (tqqq_price + 0.01)
            proceeds = shares * (tqqq_next - 0.01)
            daily_pnl = proceeds - cost_basis
        elif ticker == "SQQQ" and shares > 0:
            cost_basis = shares * (sqqq_price + 0.01)
            proceeds = shares * (sqqq_next - 0.01)
            daily_pnl = proceeds - cost_basis
        else:
            daily_pnl = 0.0

        cash += daily_pnl
        strategy_return = daily_pnl / total_equity if total_equity > 0 else 0

        # Signal label
        if leverage > 0:
            signal_label = "LONG"
        elif leverage < 0:
            signal_label = "SHORT"
        else:
            signal_label = "CASH"

        records.append({
            "date": date,
            "signal": signal_label,
            "leverage_level": leverage,
            "ticker": ticker,
            "shares": shares,
            "prob_up": round(row["prob_up"], 4),
            "tqqq_alloc": rec["tqqq_allocation"],
            "sqqq_alloc": rec["sqqq_allocation"],
            "cash_alloc": rec["cash_allocation"],
            "vol_adjusted": rec["vol_adjusted"],
            "actual_return": row["actual_return"],
            "strategy_return": round(strategy_return, 6),
            "daily_pnl": round(daily_pnl, 2),
            "equity": round(cash, 2),
            "prediction": int(row["prediction"]),
            "actual": int(row["actual"]),
            "correct": int(row["prediction"] == row["actual"]),
        })

        current_leverage = leverage

    return pd.DataFrame(records).set_index("date")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    with open(project_root / "models" / "best_config.json") as f:
        config = json.load(f)

    features_path = project_root / "data" / "processed" / "features_master.csv"
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    tqqq, sqqq = load_leveraged_prices(project_root / "data" / "raw")

    print(f"Features: {len(df)} rows")
    print(f"TQQQ: {len(tqqq)} rows, SQQQ: {len(sqqq)} rows")
    print(f"Config: K={config['k']}, metric={config['metric']}, window={config['training_window']}")

    # Run walk-forward predictions
    print("\nRunning walk-forward predictions...")
    predictions = run_walk_forward_backtest(
        df, feature_cols=config["features"],
        k=config["k"], metric=config["metric"], weights=config["weights"],
        training_window=config["training_window"],
        start_date="2020-01-01", end_date="2025-12-31",
    )

    # Run TQQQ/SQQQ backtest with discrete leverage + vol targeting
    print("\nRunning TQQQ/SQQQ backtest (7 leverage levels, vol targeting)...")
    sizer = PositionSizer(vol_target_multiple=1.0)
    trade_log = run_tqqq_sqqq_backtest(predictions, tqqq, sqqq, sizer, features_df=df)

    # Compute metrics
    metrics = compute_full_metrics(trade_log)
    print_full_metrics(metrics)

    # Leverage distribution
    if "leverage_level" in trade_log.columns:
        print("\n=== Leverage Distribution ===")
        lev_dist = trade_log["leverage_level"].value_counts().sort_index()
        for lev, count in lev_dist.items():
            pct = count / len(trade_log) * 100
            print(f"  {lev:+4d}%: {count:5d} days ({pct:5.1f}%)")

        vol_adj = trade_log["vol_adjusted"].sum()
        print(f"\n  Vol-adjusted days: {vol_adj} ({vol_adj/len(trade_log)*100:.1f}%)")

    # Compare with buy-and-hold
    overlap_dates = trade_log.index
    tqqq_bah = tqqq.loc[overlap_dates[0]:overlap_dates[-1], "Close"]
    tqqq_bah_return = tqqq_bah.iloc[-1] / tqqq_bah.iloc[0] - 1
    qqq_bah = (1 + trade_log["actual_return"]).prod() - 1

    print(f"\n=== Strategy Comparison ===")
    print(f"  KNN Leverage Strategy:  {metrics['total_return']:.2%}")
    print(f"  Buy & Hold TQQQ:        {tqqq_bah_return:.2%}")
    print(f"  Buy & Hold QQQ:         {qqq_bah:.2%}")

    # Save results
    results_dir = project_root / "backtesting" / "results" / "tqqq_sqqq_backtest"
    results_dir.mkdir(parents=True, exist_ok=True)
    trade_log.to_csv(results_dir / "trade_log.csv")
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_dir}")
