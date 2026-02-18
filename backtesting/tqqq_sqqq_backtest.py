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


def _compute_stop_levels(qqq: pd.DataFrame, stop_type: str = "two_day") -> pd.DataFrame:
    """Compute structural stop levels from QQQ daily data.

    Args:
        qqq: QQQ daily OHLCV data.
        stop_type: "two_day" (2-day low/high), "prev_day" (previous day low/high),
                   or "pivot" (classic pivot points S1/R1).

    Returns:
        DataFrame with long_stop and short_stop columns indexed by date.
    """
    stops = pd.DataFrame(index=qqq.index)

    if stop_type == "two_day":
        # 2-day low/high including signal day (known at signal day close)
        stops["long_stop"] = qqq["Low"].rolling(2).min()
        stops["short_stop"] = qqq["High"].rolling(2).max()
    elif stop_type == "prev_day":
        # Previous day's low/high (known at signal day close)
        stops["long_stop"] = qqq["Low"].shift(1)
        stops["short_stop"] = qqq["High"].shift(1)
    elif stop_type == "pivot":
        pivot = (qqq["High"] + qqq["Low"] + qqq["Close"]) / 3
        stops["long_stop"] = 2 * pivot - qqq["High"]   # S1 based on signal day
        stops["short_stop"] = 2 * pivot - qqq["Low"]    # R1 based on signal day
    else:
        raise ValueError(f"Unknown stop_type: {stop_type}")

    return stops


def _estimate_stop_exit_price(
    entry_price: float,
    qqq_close: float,
    qqq_open_next: float,
    qqq_stop_level: float,
    leveraged_open_next: float,
    leveraged_low_next: float,
    leveraged_high_next: float,
    is_long: bool,
) -> float:
    """Estimate leveraged ETF exit price when QQQ hits a stop level.

    Two scenarios:
    1. Gap through stop (QQQ opens beyond stop): exit at leveraged ETF open.
    2. Intraday breach: proportional estimate with slippage penalty.

    Validated against hourly data: proportional estimate alone is ~0.8% too
    optimistic on average, largely because most stops trigger at the open bar.

    Args:
        entry_price: Entry price (leveraged ETF close on signal day).
        qqq_close: QQQ close on signal day.
        qqq_open_next: QQQ open on the next day.
        qqq_stop_level: QQQ stop level.
        leveraged_open_next: Leveraged ETF open on next day.
        leveraged_low_next: Leveraged ETF low on next day.
        leveraged_high_next: Leveraged ETF high on next day.
        is_long: True for TQQQ long, False for SQQQ short.

    Returns:
        Estimated exit price.
    """
    if is_long:
        # Scenario 1: QQQ gaps below stop at open
        if qqq_open_next <= qqq_stop_level:
            return leveraged_open_next

        # Scenario 2: Intraday breach — use proportional estimate + slippage
        qqq_pct = (qqq_stop_level - qqq_close) / qqq_close
        estimated_exit = entry_price * (1 + 3 * qqq_pct)
        # Apply 0.5% slippage penalty for stop execution
        estimated_exit *= 0.995
        # Bound by actual day's low
        estimated_exit = max(estimated_exit, leveraged_low_next)
        return estimated_exit

    else:
        # Short side: SQQQ stops when QQQ rises above stop level
        # Scenario 1: QQQ gaps above stop at open
        if qqq_open_next >= qqq_stop_level:
            return leveraged_open_next

        # Scenario 2: Intraday breach
        qqq_pct = (qqq_stop_level - qqq_close) / qqq_close
        estimated_exit = entry_price * (1 - 3 * qqq_pct)
        # Apply 0.5% slippage penalty
        estimated_exit *= 0.995
        # Bound by actual day's low
        estimated_exit = max(estimated_exit, leveraged_low_next)
        return estimated_exit


def run_tqqq_sqqq_backtest(
    predictions: pd.DataFrame,
    tqqq: pd.DataFrame,
    sqqq: pd.DataFrame,
    sizer: PositionSizer,
    features_df: pd.DataFrame | None = None,
    qqq: pd.DataFrame | None = None,
    stop_type: str | None = None,
    initial_capital: float = 50000.0,
) -> pd.DataFrame:
    """Simulate TQQQ/SQQQ trading with discrete leverage levels.

    Args:
        predictions: Walk-forward predictions with prob_up and actual_return.
        tqqq: TQQQ daily price data.
        sqqq: SQQQ daily price data.
        sizer: PositionSizer instance.
        features_df: Features master for realized vol lookup. Optional.
        qqq: QQQ daily price data. Required if stop_type is set.
        stop_type: Stop loss type: "two_day", "prev_day", "pivot", or None.
        initial_capital: Starting account balance.

    Returns:
        Trade log DataFrame with equity curve.
    """
    # Precompute stop levels if using stops
    stop_levels = None
    if stop_type is not None:
        if qqq is None:
            raise ValueError("QQQ daily data required for stop loss simulation")
        stop_levels = _compute_stop_levels(qqq, stop_type)

    cash = initial_capital
    current_leverage = 0
    records = []
    stop_count = 0

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

        tqqq_next_close = tqqq.iloc[tqqq_idx + 1]["Close"]
        sqqq_next_close = sqqq.iloc[sqqq_idx + 1]["Close"]

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

        # Check stop loss on next day
        was_stopped = False
        if stop_levels is not None and shares > 0:
            next_date = tqqq.index[tqqq_idx + 1]
            if next_date in stop_levels.index and next_date in qqq.index:
                qqq_next = qqq.loc[next_date]

                if ticker == "TQQQ" and leverage > 0:
                    long_stop = stop_levels.loc[date, "long_stop"]
                    if not pd.isna(long_stop) and qqq_next["Low"] <= long_stop:
                        # Stop triggered — estimate exit price
                        exit_price = _estimate_stop_exit_price(
                            entry_price=tqqq_price,
                            qqq_close=qqq.loc[date, "Close"],
                            qqq_open_next=qqq_next["Open"],
                            qqq_stop_level=long_stop,
                            leveraged_open_next=tqqq.iloc[tqqq_idx + 1]["Open"],
                            leveraged_low_next=tqqq.iloc[tqqq_idx + 1]["Low"],
                            leveraged_high_next=tqqq.iloc[tqqq_idx + 1]["High"],
                            is_long=True,
                        )
                        tqqq_next_close = exit_price
                        was_stopped = True
                        stop_count += 1

                elif ticker == "SQQQ" and leverage < 0:
                    short_stop = stop_levels.loc[date, "short_stop"]
                    if not pd.isna(short_stop) and qqq_next["High"] >= short_stop:
                        exit_price = _estimate_stop_exit_price(
                            entry_price=sqqq_price,
                            qqq_close=qqq.loc[date, "Close"],
                            qqq_open_next=qqq_next["Open"],
                            qqq_stop_level=short_stop,
                            leveraged_open_next=sqqq.iloc[sqqq_idx + 1]["Open"],
                            leveraged_low_next=sqqq.iloc[sqqq_idx + 1]["Low"],
                            leveraged_high_next=sqqq.iloc[sqqq_idx + 1]["High"],
                            is_long=False,
                        )
                        sqqq_next_close = exit_price
                        was_stopped = True
                        stop_count += 1

        # Calculate P&L based on allocation
        if ticker == "TQQQ" and shares > 0:
            cost_basis = shares * (tqqq_price + 0.01)
            proceeds = shares * (tqqq_next_close - 0.01)
            daily_pnl = proceeds - cost_basis
        elif ticker == "SQQQ" and shares > 0:
            cost_basis = shares * (sqqq_price + 0.01)
            proceeds = shares * (sqqq_next_close - 0.01)
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
            "was_stopped": was_stopped,
        })

        current_leverage = leverage

    result = pd.DataFrame(records).set_index("date")
    if stop_type is not None:
        total = len(result[result["signal"] != "CASH"])
        print(f"  Stop triggers: {stop_count}/{total} traded days "
              f"({stop_count/max(total,1):.1%})")
    return result


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
