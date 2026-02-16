"""Generate backtest visualization plots."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def plot_equity_curve(trade_log: pd.DataFrame, output_path: Path) -> None:
    """Plot equity curve: strategy vs buy-and-hold QQQ.

    Args:
        trade_log: Trade log DataFrame with equity and actual_return columns.
        output_path: Path to save plot.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Strategy equity
    ax.plot(trade_log.index, trade_log["equity"], "b-", linewidth=1.5, label="KNN Strategy")

    # Buy-and-hold QQQ
    bah = trade_log["equity"].iloc[0] * (1 + trade_log["actual_return"]).cumprod()
    ax.plot(trade_log.index, bah, "gray", linewidth=1, alpha=0.7, label="Buy & Hold QQQ")

    ax.set_title("Equity Curve: KNN Strategy vs Buy & Hold QQQ", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_monthly_heatmap(trade_log: pd.DataFrame, output_path: Path) -> None:
    """Plot monthly returns heatmap.

    Args:
        trade_log: Trade log DataFrame.
        output_path: Path to save plot.
    """
    tl = trade_log.copy()
    tl["year"] = tl.index.year
    tl["month"] = tl.index.month

    monthly = tl.groupby(["year", "month"])["strategy_return"].apply(
        lambda x: (1 + x).prod() - 1
    ).unstack()

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly.columns = [month_names[m - 1] for m in monthly.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        monthly * 100, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
        ax=ax, linewidths=0.5, cbar_kws={"label": "Return (%)"},
    )
    ax.set_title("Monthly Returns (%)", fontsize=14)
    ax.set_ylabel("Year")
    plt.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_drawdown(trade_log: pd.DataFrame, output_path: Path) -> None:
    """Plot drawdown chart over time.

    Args:
        trade_log: Trade log DataFrame with equity column.
        output_path: Path to save plot.
    """
    equity = trade_log["equity"]
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    ax.plot(drawdown.index, drawdown.values, "r-", linewidth=0.8)
    ax.set_title("Drawdown Over Time", fontsize=14)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_rolling_sharpe(trade_log: pd.DataFrame, output_path: Path, window: int = 63) -> None:
    """Plot rolling Sharpe ratio.

    Args:
        trade_log: Trade log DataFrame.
        output_path: Path to save plot.
        window: Rolling window in trading days (63 ≈ 3 months).
    """
    returns = trade_log["strategy_return"]
    daily_rf = 0.04 / 252
    excess = returns - daily_rf

    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, "b-", linewidth=1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.3, label="Sharpe = 1")
    ax.axhline(y=-1, color="red", linestyle="--", alpha=0.3, label="Sharpe = -1")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontsize=14)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_signal_distribution(trade_log: pd.DataFrame, output_path: Path) -> None:
    """Plot signal confidence score distribution.

    Args:
        trade_log: Trade log DataFrame with prob_up column.
        output_path: Path to save plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(trade_log["prob_up"], bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(x=0.58, color="green", linestyle="--", linewidth=2, label="Long threshold (0.58)")
    ax.axvline(x=0.42, color="red", linestyle="--", linewidth=2, label="Short threshold (0.42)")
    ax.set_title("Prediction Confidence Distribution", fontsize=14)
    ax.set_xlabel("P(Up)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_win_rate_by_regime(trade_log: pd.DataFrame, output_path: Path) -> None:
    """Plot win rate by month and VIX regime (approximated by volatility).

    Args:
        trade_log: Trade log DataFrame.
        output_path: Path to save plot.
    """
    traded = trade_log[trade_log["signal"] != "CASH"].copy()
    traded["win"] = (traded["strategy_return"] > 0).astype(int)
    traded["month"] = traded.index.month

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Win rate by month
    monthly_wr = traded.groupby("month")["win"].mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    axes[0].bar(range(1, 13), monthly_wr.reindex(range(1, 13), fill_value=0),
                color="steelblue", edgecolor="black")
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(month_names, rotation=45)
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title("Win Rate by Month")
    axes[0].set_ylabel("Win Rate")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Win rate by volatility regime (use rolling 20d vol of strategy returns)
    all_returns = trade_log["strategy_return"]
    rolling_vol = all_returns.rolling(20).std() * np.sqrt(252)
    traded["vol_regime"] = pd.cut(
        rolling_vol.reindex(traded.index),
        bins=[0, 0.10, 0.20, 0.30, float("inf")],
        labels=["Low (<10%)", "Med (10-20%)", "High (20-30%)", "Very High (>30%)"],
    )
    regime_wr = traded.groupby("vol_regime", observed=True)["win"].mean()
    regime_wr.plot(kind="bar", ax=axes[1], color="coral", edgecolor="black")
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("Win Rate by Volatility Regime")
    axes[1].set_ylabel("Win Rate")
    axes[1].set_xlabel("")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_html_report(metrics_path: Path, plots_dir: Path, output_path: Path) -> None:
    """Generate HTML summary report.

    Args:
        metrics_path: Path to metrics.json.
        plots_dir: Directory containing plot images.
        output_path: Path to save HTML report.
    """
    with open(metrics_path) as f:
        m = json.load(f)

    plot_files = sorted(plots_dir.glob("*.png"))
    plots_html = "\n".join(
        f'<div class="plot"><h3>{p.stem.replace("_", " ").title()}</h3>'
        f'<img src="plots/{p.name}" width="100%"></div>'
        for p in plot_files
    )

    html = f"""<!DOCTYPE html>
<html><head><title>KNN QQQ Backtest Report</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #333; }} h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
.metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
.metric {{ background: #f5f5f5; padding: 15px; border-radius: 8px; }}
.metric .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
.metric .label {{ font-size: 12px; color: #666; }}
.plot {{ margin: 20px 0; }}
img {{ border: 1px solid #ddd; border-radius: 4px; }}
</style></head><body>
<h1>KNN QQQ Trading Model — Backtest Report</h1>
<h2>Performance Summary</h2>
<div class="metrics">
<div class="metric"><div class="value">{m['total_return']:.1%}</div><div class="label">Total Return</div></div>
<div class="metric"><div class="value">{m['annual_return']:.1%}</div><div class="label">Annual Return</div></div>
<div class="metric"><div class="value">{m['sharpe_ratio']:.2f}</div><div class="label">Sharpe Ratio</div></div>
<div class="metric"><div class="value">{m['sortino_ratio']:.2f}</div><div class="label">Sortino Ratio</div></div>
<div class="metric"><div class="value">{m['max_drawdown']:.1%}</div><div class="label">Max Drawdown</div></div>
<div class="metric"><div class="value">{m['profit_factor']:.2f}</div><div class="label">Profit Factor</div></div>
<div class="metric"><div class="value">{m['win_rate_traded_days']:.1%}</div><div class="label">Win Rate (Traded)</div></div>
<div class="metric"><div class="value">{m['trade_frequency']:.1%}</div><div class="label">Trade Frequency</div></div>
<div class="metric"><div class="value">${m['total_pnl']:,.0f}</div><div class="label">Total P&L</div></div>
</div>
<h2>Plots</h2>
{plots_html}
</body></html>"""

    with open(output_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "backtesting" / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    trade_log = pd.read_csv(results_dir / "trade_log.csv", index_col=0, parse_dates=True)
    print(f"Loaded trade log: {len(trade_log)} days\n")

    print("Generating plots...")
    plot_equity_curve(trade_log, plots_dir / "equity_curve.png")
    print("  equity_curve.png")

    plot_monthly_heatmap(trade_log, plots_dir / "monthly_returns.png")
    print("  monthly_returns.png")

    plot_drawdown(trade_log, plots_dir / "drawdown.png")
    print("  drawdown.png")

    plot_rolling_sharpe(trade_log, plots_dir / "rolling_sharpe.png")
    print("  rolling_sharpe.png")

    plot_signal_distribution(trade_log, plots_dir / "signal_distribution.png")
    print("  signal_distribution.png")

    plot_win_rate_by_regime(trade_log, plots_dir / "win_rate_regimes.png")
    print("  win_rate_regimes.png")

    # Generate HTML report
    generate_html_report(
        results_dir / "metrics.json",
        plots_dir,
        results_dir / "report.html",
    )
    print("\n  report.html generated")
    print(f"\nAll outputs saved to {results_dir}")
