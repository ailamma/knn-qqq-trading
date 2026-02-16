"""Stress test: evaluate model during known crisis periods."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


CRISIS_PERIODS = {
    "COVID Crash (Feb-Apr 2020)": ("2020-02-19", "2020-04-30"),
    "2022 Bear Market (Jan-Oct 2022)": ("2022-01-03", "2022-10-31"),
    "2023 Banking Crisis (Mar 2023)": ("2023-03-01", "2023-03-31"),
    "2024 Aug VIX Spike": ("2024-07-15", "2024-08-15"),
}


def stress_test_period(
    trade_log: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    """Compute metrics for a specific crisis period.

    Args:
        trade_log: Full trade log DataFrame.
        start: Period start date.
        end: Period end date.

    Returns:
        Dictionary of crisis period metrics.
    """
    mask = (trade_log.index >= start) & (trade_log.index <= end)
    period = trade_log[mask]

    if len(period) == 0:
        return {"error": "No data in period"}

    traded = period[period["signal"] != "CASH"]
    strategy_returns = period["strategy_return"]

    # Total return in period
    total_return = (1 + strategy_returns).prod() - 1

    # QQQ buy-and-hold return in period
    bah_return = (1 + period["actual_return"]).prod() - 1

    # Max single-day loss
    worst_day = strategy_returns.min()
    worst_day_date = strategy_returns.idxmin()

    # Max drawdown in period
    equity = (1 + strategy_returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate on traded days
    if len(traded) > 0:
        traded_returns = traded["strategy_return"]
        win_rate = (traded_returns > 0).mean()
    else:
        win_rate = 0.0

    n_traded = len(traded)
    n_long = (period["signal"] == "LONG").sum()
    n_short = (period["signal"] == "SHORT").sum()
    n_cash = (period["signal"] == "CASH").sum()

    return {
        "trading_days": len(period),
        "strategy_return": round(float(total_return), 4),
        "qqq_buy_hold_return": round(float(bah_return), 4),
        "outperformance": round(float(total_return - bah_return), 4),
        "max_drawdown": round(float(max_dd), 4),
        "worst_day_return": round(float(worst_day), 4),
        "worst_day_date": str(worst_day_date.date()) if hasattr(worst_day_date, "date") else str(worst_day_date),
        "win_rate": round(float(win_rate), 4),
        "n_traded": n_traded,
        "n_long": int(n_long),
        "n_short": int(n_short),
        "n_cash": int(n_cash),
    }


def run_stress_tests(trade_log: pd.DataFrame) -> dict:
    """Run stress tests for all crisis periods.

    Args:
        trade_log: Full trade log DataFrame.

    Returns:
        Dictionary with results for each crisis period.
    """
    results = {}
    for name, (start, end) in CRISIS_PERIODS.items():
        print(f"\n--- {name} ---")
        metrics = stress_test_period(trade_log, start, end)
        results[name] = metrics

        if "error" not in metrics:
            print(f"  Days: {metrics['trading_days']} ({metrics['n_traded']} traded)")
            print(f"  Strategy Return: {metrics['strategy_return']:.2%}")
            print(f"  QQQ Buy&Hold:    {metrics['qqq_buy_hold_return']:.2%}")
            print(f"  Outperformance:  {metrics['outperformance']:.2%}")
            print(f"  Max Drawdown:    {metrics['max_drawdown']:.2%}")
            print(f"  Worst Day:       {metrics['worst_day_return']:.2%} ({metrics['worst_day_date']})")
            print(f"  Win Rate:        {metrics['win_rate']:.2%}")
            print(f"  Signals: L={metrics['n_long']} S={metrics['n_short']} C={metrics['n_cash']}")
        else:
            print(f"  {metrics['error']}")

    return results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "backtesting" / "results"

    trade_log = pd.read_csv(results_dir / "trade_log.csv", index_col=0, parse_dates=True)
    print(f"Loaded trade log: {len(trade_log)} days")

    results = run_stress_tests(trade_log)

    # Save
    output_path = results_dir / "stress_tests.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nStress test results saved to {output_path}")
