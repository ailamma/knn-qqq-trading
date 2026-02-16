"""Compute comprehensive performance metrics from backtest trade log."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compute_full_metrics(trade_log: pd.DataFrame) -> dict[str, Any]:
    """Compute comprehensive performance metrics from trade log.

    Args:
        trade_log: DataFrame from Backtester.run() with equity curve.

    Returns:
        Dictionary of all performance metrics.
    """
    traded = trade_log[trade_log["signal"] != "CASH"]
    strategy_returns = trade_log["strategy_return"]
    traded_returns = traded["strategy_return"]

    # Total and annualized return
    total_return = trade_log["equity"].iloc[-1] / trade_log["equity"].iloc[0] - 1
    n_calendar_days = (trade_log.index[-1] - trade_log.index[0]).days
    n_years = n_calendar_days / 365.25
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # Sharpe ratio (annualized, risk-free = 4.0%)
    daily_rf = 0.04 / 252
    excess = strategy_returns - daily_rf
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    # Sortino ratio (downside deviation only)
    downside = excess[excess < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-10
    sortino = excess.mean() / downside_std * np.sqrt(252)

    # Max drawdown (% and duration)
    equity = trade_log["equity"]
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Drawdown duration
    in_drawdown = drawdown < 0
    dd_groups = (~in_drawdown).cumsum()
    dd_durations = in_drawdown.groupby(dd_groups).sum()
    max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    # Win rate
    win_rate_all = (strategy_returns > 0).sum() / max(len(strategy_returns), 1)
    win_rate_traded = (traded_returns > 0).sum() / max(len(traded_returns), 1)

    # Profit factor
    gross_profit = traded_returns[traded_returns > 0].sum()
    gross_loss = abs(traded_returns[traded_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average win / average loss
    avg_win = traded_returns[traded_returns > 0].mean() if (traded_returns > 0).any() else 0
    avg_loss = abs(traded_returns[traded_returns < 0].mean()) if (traded_returns < 0).any() else 1e-10
    win_loss_ratio = avg_win / avg_loss

    # Trade stats
    n_total = len(trade_log)
    n_traded = len(traded)
    n_long = (trade_log["signal"] == "LONG").sum()
    n_short = (trade_log["signal"] == "SHORT").sum()
    n_cash = (trade_log["signal"] == "CASH").sum()

    # Monthly returns
    trade_log_copy = trade_log.copy()
    trade_log_copy["month"] = trade_log_copy.index.to_period("M")
    monthly = trade_log_copy.groupby("month")["strategy_return"].apply(
        lambda x: (1 + x).prod() - 1
    )

    return {
        "total_return": round(float(total_return), 4),
        "annual_return": round(float(annual_return), 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "sortino_ratio": round(float(sortino), 4),
        "max_drawdown": round(float(max_dd), 4),
        "max_drawdown_duration_days": max_dd_duration,
        "win_rate_all_days": round(float(win_rate_all), 4),
        "win_rate_traded_days": round(float(win_rate_traded), 4),
        "profit_factor": round(float(profit_factor), 4),
        "avg_win_loss_ratio": round(float(win_loss_ratio), 4),
        "avg_win": round(float(avg_win), 6),
        "avg_loss": round(float(avg_loss if avg_loss != 1e-10 else 0), 6),
        "n_total_days": n_total,
        "n_traded_days": n_traded,
        "n_long": int(n_long),
        "n_short": int(n_short),
        "n_cash": int(n_cash),
        "trade_frequency": round(n_traded / n_total, 4),
        "starting_equity": round(float(trade_log["equity"].iloc[0]), 2),
        "ending_equity": round(float(trade_log["equity"].iloc[-1]), 2),
        "total_pnl": round(float(trade_log["daily_pnl"].sum()), 2),
        "best_day": round(float(strategy_returns.max()), 6),
        "worst_day": round(float(strategy_returns.min()), 6),
        "monthly_returns": {
            str(k): round(float(v), 4) for k, v in monthly.items()
        },
    }


def print_full_metrics(metrics: dict[str, Any]) -> None:
    """Pretty-print comprehensive backtest metrics.

    Args:
        metrics: Dictionary from compute_full_metrics.
    """
    print("\n" + "=" * 50)
    print("       BACKTEST PERFORMANCE REPORT")
    print("=" * 50)

    print(f"\n--- Returns ---")
    print(f"  Total Return:      {metrics['total_return']:.2%}")
    print(f"  Annual Return:     {metrics['annual_return']:.2%}")
    print(f"  Starting Equity:   ${metrics['starting_equity']:,.2f}")
    print(f"  Ending Equity:     ${metrics['ending_equity']:,.2f}")
    print(f"  Total P&L:         ${metrics['total_pnl']:,.2f}")

    print(f"\n--- Risk Metrics ---")
    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:     {metrics['sortino_ratio']:.2f}")
    print(f"  Max Drawdown:      {metrics['max_drawdown']:.2%}")
    print(f"  Max DD Duration:   {metrics['max_drawdown_duration_days']} trading days")
    print(f"  Best Day:          {metrics['best_day']:.4%}")
    print(f"  Worst Day:         {metrics['worst_day']:.4%}")

    print(f"\n--- Trade Statistics ---")
    print(f"  Win Rate (all):    {metrics['win_rate_all_days']:.2%}")
    print(f"  Win Rate (traded): {metrics['win_rate_traded_days']:.2%}")
    print(f"  Profit Factor:     {metrics['profit_factor']:.2f}")
    print(f"  Avg Win/Loss:      {metrics['avg_win_loss_ratio']:.2f}")
    print(f"  Trade Frequency:   {metrics['trade_frequency']:.2%}")

    print(f"\n--- Signal Distribution ---")
    print(f"  Total Days:  {metrics['n_total_days']}")
    print(f"  LONG:        {metrics['n_long']} ({metrics['n_long']/metrics['n_total_days']:.1%})")
    print(f"  SHORT:       {metrics['n_short']} ({metrics['n_short']/metrics['n_total_days']:.1%})")
    print(f"  CASH:        {metrics['n_cash']} ({metrics['n_cash']/metrics['n_total_days']:.1%})")
    print("=" * 50)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "backtesting" / "results"

    trade_log = pd.read_csv(results_dir / "trade_log.csv", index_col=0, parse_dates=True)
    print(f"Loaded trade log: {len(trade_log)} days")

    metrics = compute_full_metrics(trade_log)
    print_full_metrics(metrics)

    # Save metrics
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
