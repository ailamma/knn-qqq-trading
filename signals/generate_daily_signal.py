"""Daily signal generation: run at EOD to generate next-day TQQQ/SQQQ recommendation.

Two model versions available:
  V1 (backtested): Multi-K ensemble [11,15,21], uniform weighting — backtested +160%, Sharpe 0.66
  V2 (recency):    Multi-K ensemble [11,15,21] + recency weighting (180d decay) — regime-adaptive

Usage:
  python signals/generate_daily_signal.py          # default (--both)
  python signals/generate_daily_signal.py --v1     # V1 backtested only
  python signals/generate_daily_signal.py --v2     # V2 recency only
  python signals/generate_daily_signal.py --both   # run both, show comparison
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.price_features import compute_price_features
from features.momentum_features import compute_momentum_features
from features.volatility_features import compute_volatility_features
from features.volume_features import compute_volume_features
from features.cross_asset_features import compute_cross_asset_features
from features.calendar_features import compute_calendar_features
from features.regime_features import compute_regime_features
from signals.position_sizer import PositionSizer, LEVERAGE_LEVELS


def load_config() -> dict:
    """Load model configuration.

    Returns:
        Model configuration dictionary.
    """
    with open(PROJECT_ROOT / "models" / "best_config.json") as f:
        return json.load(f)


# Mapping: yfinance ticker → (column prefix, local CSV filename)
TICKER_MAP = {
    "QQQ": (None, "qqq_daily.csv"),
    "^VIX": ("VIX", "vix_daily.csv"),
    "SPY": ("SPY", "spy_daily.csv"),
    "IWM": ("IWM", "iwm_daily.csv"),
    "TLT": ("TLT", "tlt_daily.csv"),
    "GLD": ("GLD", "gld_daily.csv"),
    "SMH": ("SMH", "smh_daily.csv"),
    "HYG": ("HYG", "hyg_daily.csv"),
    "UUP": ("UUP", "uup_daily.csv"),
}

KEEP_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _load_or_download_ticker(
    yf_ticker: str, prefix: str | None, csv_path: Path,
) -> pd.DataFrame:
    """Load ticker from local cache and download only missing days.

    Args:
        yf_ticker: Yahoo Finance ticker symbol.
        prefix: Column prefix (None for QQQ).
        csv_path: Path to local CSV cache file.

    Returns:
        Full DataFrame with all available history.
    """
    from datetime import timedelta

    cached = None
    if csv_path.exists():
        cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        last_date = cached.index[-1]
        # Download from 2 days before last cached date to handle corrections
        start = (last_date - timedelta(days=2)).strftime("%Y-%m-%d")
        new = yf.Ticker(yf_ticker).history(start=start, auto_adjust=False)
    else:
        new = yf.Ticker(yf_ticker).history(start="2003-01-01", auto_adjust=False)

    new = new[[c for c in KEEP_COLS if c in new.columns]]
    new.index = pd.to_datetime(new.index).tz_localize(None)

    if cached is not None:
        # Overwrite overlapping dates with fresh data, append new dates
        combined = cached.copy()
        for date in new.index:
            combined.loc[date] = new.loc[date]
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        df = combined
    else:
        df = new

    # Save updated cache
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)

    if prefix:
        df.columns = [f"{prefix}_{c}" for c in df.columns]

    return df


def download_fresh_data() -> pd.DataFrame:
    """Load cached data and download only incremental updates, then merge.

    Returns:
        Merged DataFrame with all tickers aligned.
    """
    raw_dir = PROJECT_ROOT / "data" / "raw"
    dfs = {}

    for yf_ticker, (prefix, filename) in TICKER_MAP.items():
        csv_path = raw_dir / filename
        cached_rows = 0
        if csv_path.exists():
            cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            cached_rows = len(cached)
            last_date = cached.index[-1].date()
            label = f"cached to {last_date}, updating"
        else:
            label = "full download"
        df = _load_or_download_ticker(yf_ticker, prefix, csv_path)
        new_rows = len(df) - cached_rows
        ticker_label = prefix or "QQQ"
        if new_rows > 0:
            print(f"    {ticker_label}: +{new_rows} new rows ({label})")
        dfs[yf_ticker] = df

    # If today's QQQ data is missing (market still open), fetch real-time quote
    today = pd.Timestamp(datetime.now().date())
    qqq_df = dfs["QQQ"]
    if today not in qqq_df.index:
        try:
            ticker_info = yf.Ticker("QQQ")
            fast_info = ticker_info.fast_info
            current_price = fast_info.get("lastPrice") or fast_info.get("previousClose")
            if current_price:
                # Create approximate row using current price
                approx_row = pd.Series({
                    "Open": current_price,
                    "High": current_price,
                    "Low": current_price,
                    "Close": current_price,
                    "Adj Close": current_price,
                    "Volume": 0,
                }, name=today)
                qqq_df = pd.concat([qqq_df, approx_row.to_frame().T])
                dfs["QQQ"] = qqq_df
                print(f"    QQQ: using real-time price ${current_price:.2f} as approx close for {today.date()}")

                # Also update the cached CSV so resolve can find it
                raw_dir = PROJECT_ROOT / "data" / "raw"
                cached_qqq = pd.read_csv(raw_dir / "qqq_daily.csv", index_col=0, parse_dates=True)
                cached_qqq.loc[today] = approx_row
                cached_qqq.to_csv(raw_dir / "qqq_daily.csv")
        except Exception as e:
            print(f"    QQQ: could not fetch real-time price ({e})")

    # Inner join
    master = dfs["QQQ"]
    for key in ["^VIX", "SPY", "IWM", "TLT", "GLD", "SMH", "HYG", "UUP"]:
        master = master.join(dfs[key], how="inner")

    return master


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features on the merged dataset.

    Args:
        df: Merged master DataFrame.

    Returns:
        DataFrame with all features computed.
    """
    df = compute_price_features(df)
    df = compute_momentum_features(df)
    df = compute_volatility_features(df)
    df = compute_volume_features(df)
    df = compute_cross_asset_features(df)
    df = compute_calendar_features(df)
    df = compute_regime_features(df)
    return df


def _prepare_training_data(model_config: dict) -> tuple:
    """Download data, compute features, prepare training arrays.

    Args:
        model_config: Model configuration dictionary.

    Returns:
        Tuple of (train_X_scaled, train_y, today_X_scaled, today_row, today_date, scaler, train_dates).
    """
    print("Downloading latest market data...")
    master = download_fresh_data()
    print(f"  Data: {len(master)} rows, latest date: {master.index[-1].date()}")

    print("Computing features...")
    master = compute_all_features(master)

    feat_cols = [c for c in master.columns if c.startswith("feat_")]
    master = master.dropna(subset=feat_cols)
    print(f"  Clean data: {len(master)} rows")

    features = model_config["features"]
    training_window = model_config["training_window"]

    # Save today's row before dropping
    today_row = master.iloc[-1]
    today_X = today_row[features].values.reshape(1, -1)
    today_date = master.index[-1].date()

    # Target: next-day direction (drop last row — no future target)
    master["_target"] = (master["Close"].pct_change().shift(-1) > 0).astype(int)
    master = master.iloc[:-1]
    X = master[features].values
    y = master["_target"].values

    train_X = X[-training_window:]
    train_y = y[-training_window:]
    train_dates = master.index[-training_window:]

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    today_X_scaled = scaler.transform(today_X)

    return train_X_scaled, train_y, today_X_scaled, today_row, today_date, scaler, train_dates


ENSEMBLE_KS = [11, 15, 21]


def _recency_weighted_knn_proba(
    knn: KNeighborsClassifier,
    X_today: np.ndarray,
    train_y: np.ndarray,
    train_dates: pd.DatetimeIndex,
    decay_days: int = 180,
) -> float:
    """Compute recency-weighted probability from KNN neighbors.

    Recent neighbors get higher weight via exponential decay.

    Args:
        knn: Fitted KNeighborsClassifier.
        X_today: Today's feature vector (1, n_features).
        train_y: Training labels array.
        train_dates: DatetimeIndex of training rows.
        decay_days: Half-life for exponential decay in days.

    Returns:
        Weighted probability of UP (class 1).
    """
    distances, indices = knn.kneighbors(X_today)
    indices = indices[0]

    neighbor_dates = train_dates[indices]
    last_train_date = train_dates[-1]
    ages = (last_train_date - neighbor_dates).days.astype(float)

    weights = np.exp(-ages / decay_days)
    neighbor_labels = train_y[indices]

    prob_up = float(np.sum(weights * neighbor_labels) / np.sum(weights))
    return prob_up


RECENCY_DECAY_DAYS = 180


def generate_signal(
    model_config: dict,
    account_balance: float = 50000.0,
    model_version: str = "v1",
    prepared_data: tuple | None = None,
) -> dict:
    """Generate today's trading signal.

    Args:
        model_config: KNN model configuration.
        account_balance: Current account balance.
        model_version: "v1" (multi-K ensemble, no recency) or "v2" (multi-K + recency).
        prepared_data: Pre-computed training data tuple to avoid re-downloading.

    Returns:
        Signal recommendation dictionary.
    """
    if prepared_data is not None:
        train_X_scaled, train_y, today_X_scaled, today_row, today_date, scaler, train_dates = prepared_data
    else:
        train_X_scaled, train_y, today_X_scaled, today_row, today_date, scaler, train_dates = \
            _prepare_training_data(model_config)

    features = model_config["features"]
    use_recency = model_version == "v2"

    # Multi-K ensemble
    per_k_probs = {}
    for k in ENSEMBLE_KS:
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric=model_config["metric"],
            weights="uniform",
        )
        knn.fit(train_X_scaled, train_y)

        if use_recency:
            prob = _recency_weighted_knn_proba(
                knn, today_X_scaled, train_y, train_dates, RECENCY_DECAY_DAYS
            )
        else:
            proba = knn.predict_proba(today_X_scaled)[0]
            prob = float(proba[1] if len(proba) > 1 else proba[0])

        per_k_probs[k] = prob

    prob_up = float(np.mean(list(per_k_probs.values())))

    if model_version == "v1":
        version_label = "V1 (multi-K ensemble, backtested)"
    else:
        version_label = "V2 (multi-K ensemble + recency 180d)"

    prediction = 1 if prob_up > 0.5 else 0

    # Get current TQQQ/SQQQ prices
    tqqq_price = yf.Ticker("TQQQ").history(period="1d", auto_adjust=False)["Close"].iloc[-1]
    sqqq_price = yf.Ticker("SQQQ").history(period="1d", auto_adjust=False)["Close"].iloc[-1]

    # Realized vol for vol targeting
    realized_vol = None
    vol_col = "feat_realized_vol_20d"
    if vol_col in today_row.index:
        realized_vol = float(today_row[vol_col])

    # Position sizing
    sizer = PositionSizer(vol_target_multiple=1.0)
    recommendation = sizer.size(
        prob_up, account_balance, tqqq_price, sqqq_price,
        realized_vol_annual=realized_vol,
    )

    signal = {
        "date": str(today_date),
        "generated_at": datetime.now().isoformat(),
        "model_version": model_version,
        "model_version_label": version_label,
        "prediction": "UP" if prediction == 1 else "DOWN",
        "prob_up": round(float(prob_up), 4),
        "knn_prob_up": round(float(prob_up), 4),
        "lr_prob_up": None,
        "recommendation": recommendation,
        "model_details": {
            "ensemble_k_values": ENSEMBLE_KS,
            "per_k_probs": {str(k): round(p, 4) for k, p in per_k_probs.items()},
            "recency_decay_days": RECENCY_DECAY_DAYS if use_recency else None,
            "metric": model_config["metric"],
            "training_window": model_config["training_window"],
            "features_used": features,
        },
        "prices": {
            "qqq_close": round(float(today_row["Close"]), 2),
            "tqqq_close": round(float(tqqq_price), 2),
            "sqqq_close": round(float(sqqq_price), 2),
        },
        "account_balance": account_balance,
    }

    return signal


def print_signal(signal: dict) -> None:
    """Print human-readable signal summary.

    Args:
        signal: Signal dictionary from generate_signal.
    """
    rec = signal["recommendation"]
    prob = signal["prob_up"]
    leverage = rec["leverage_level"]
    version = signal.get("model_version_label", "unknown")

    print("\n" + "=" * 60)
    print("       KNN QQQ TRADING MODEL — DAILY SIGNAL")
    print("=" * 60)
    print(f"  Model:          {version}")
    print(f"  Date:           {signal['date']}")
    print(f"  QQQ Close:      ${signal['prices']['qqq_close']:.2f}")
    print(f"  QQQ Prediction: {signal['prediction']} (P(up) = {prob:.1%})")
    # Show per-K ensemble breakdown
    per_k = signal.get("model_details", {}).get("per_k_probs", {})
    if per_k:
        k_str = "  ".join(f"K={k}:{float(p):.1%}" for k, p in per_k.items())
        print(f"    KNN ensemble: {k_str}")
    recency = signal.get("model_details", {}).get("recency_decay_days")
    if recency:
        print(f"    Recency:      decay={recency}d")
    print()
    print(f"  ┌─────────────────────────────────────────────────┐")

    if leverage > 0:
        ticker = rec["ticker"]
        alloc = rec["tqqq_allocation"]
        price_key = "tqqq_close"
        price = signal["prices"].get(price_key, 0)
        print(f"  │  ACTION:     LONG {ticker} (leverage {leverage:+d}%)         │")
        print(f"  │  Allocation: {alloc:.0%} {ticker}, {rec['cash_allocation']:.0%} cash{' ' * 16}│")
        print(f"  │  Shares:     {rec['shares']:<6d} @ ${price:.2f}{' ' * (21 - len(f'{price:.2f}'))}│")
        print(f"  │  Dollar amt: ${rec['dollar_amount']:>10,.2f}                    │")
    elif leverage < 0:
        ticker = rec["ticker"]
        alloc = rec["sqqq_allocation"]
        price_key = "sqqq_close"
        price = signal["prices"].get(price_key, 0)
        print(f"  │  ACTION:     SHORT via {ticker} (leverage {leverage:+d}%)    │")
        print(f"  │  Allocation: {alloc:.0%} {ticker}, {rec['cash_allocation']:.0%} cash{' ' * 16}│")
        print(f"  │  Shares:     {rec['shares']:<6d} @ ${price:.2f}{' ' * (21 - len(f'{price:.2f}'))}│")
        print(f"  │  Dollar amt: ${rec['dollar_amount']:>10,.2f}                    │")
    else:
        print(f"  │  ACTION:     NO TRADE — STAY IN CASH              │")
        print(f"  │  Leverage:   0% (confidence in dead zone)         │")
        print(f"  │  P(up) = {prob:.1%} — not enough edge either way   │")

    if rec["vol_adjusted"]:
        print(f"  │  ⚠ Vol-adjusted: raw {rec['raw_leverage']:+d}% → {leverage:+d}%           │")

    print(f"  └─────────────────────────────────────────────────┘")
    print()
    print(f"  Leverage Levels (via TQQQ/SQQQ + cash):")
    print(f"    +300% = 100% TQQQ    +200% = 67% TQQQ")
    print(f"    +100% = 33% TQQQ        0% = cash")
    print(f"    -100% = 33% SQQQ    -200% = 67% SQQQ")
    print(f"    -300% = 100% SQQQ")
    print(f"  Account: ${signal['account_balance']:,.2f}")
    print()
    print(f"  Execution: Run at ~3:55 PM ET, place order by 3:59 PM")
    print(f"  Hold overnight, repeat next day at 3:55 PM")
    print("=" * 60)


PREDICTION_LOG_PATH = PROJECT_ROOT / "signals" / "prediction_log.csv"

PREDICTION_LOG_COLUMNS = [
    "signal_date", "predict_date", "model_version", "prediction",
    "prob_up", "knn_prob_up", "lr_prob_up", "leverage_level",
    "ticker", "shares", "qqq_close",
    "actual_direction", "actual_return", "correct", "pnl",
]


def _get_next_trading_day(signal_date: str, qqq_data: pd.DataFrame) -> str | None:
    """Find the next trading day after signal_date in QQQ data.

    Args:
        signal_date: Date string (YYYY-MM-DD).
        qqq_data: QQQ DataFrame with DatetimeIndex.

    Returns:
        Next trading day as string, or None if not found.
    """
    sig_dt = pd.Timestamp(signal_date)
    future = qqq_data.index[qqq_data.index > sig_dt]
    if len(future) > 0:
        return str(future[0].date())
    return None


def _resolve_pending_predictions(qqq_data: pd.DataFrame) -> None:
    """Resolve pending predictions by filling in actual outcomes.

    Reads prediction_log.csv, finds rows without actual_direction,
    looks up actual QQQ returns, and updates the CSV.

    Args:
        qqq_data: QQQ DataFrame with DatetimeIndex and Close column.
    """
    if not PREDICTION_LOG_PATH.exists():
        return

    df = pd.read_csv(PREDICTION_LOG_PATH, dtype=str)
    if df.empty:
        return

    pending = df["actual_direction"].isna() | (df["actual_direction"] == "")
    if not pending.any():
        return

    resolved_count = 0
    for idx in df.index[pending]:
        signal_date = df.at[idx, "signal_date"]
        predict_date = df.at[idx, "predict_date"]

        sig_dt = pd.Timestamp(signal_date)
        pred_dt = pd.Timestamp(predict_date)

        # Need both signal_date and predict_date closes
        if sig_dt not in qqq_data.index or pred_dt not in qqq_data.index:
            continue

        close_signal = qqq_data.loc[sig_dt, "Close"]
        close_predict = qqq_data.loc[pred_dt, "Close"]
        actual_return = (close_predict - close_signal) / close_signal
        actual_direction = "UP" if actual_return > 0 else "DOWN"
        prediction = df.at[idx, "prediction"]
        correct = 1 if actual_direction == prediction else 0

        # Estimate P&L: leverage_level% * actual_return * qqq_close
        leverage = int(df.at[idx, "leverage_level"])
        qqq_close = float(df.at[idx, "qqq_close"])
        # leverage is effective: +300 means 3x long, -300 means 3x short
        # P&L = (leverage/100) * actual_return * account_value_proxy
        # Use qqq_close * shares as dollar exposure for a simpler estimate
        shares = int(df.at[idx, "shares"]) if df.at[idx, "shares"] not in ("", "0") else 0
        ticker = df.at[idx, "ticker"]
        if ticker == "TQQQ":
            # TQQQ moves ~3x QQQ, so P&L ≈ shares * tqqq_price * 3 * actual_return
            # Simpler: use leverage_level/100 * actual_return * qqq_close as proxy
            pnl = (leverage / 100) * actual_return * qqq_close
        elif ticker == "SQQQ":
            pnl = (leverage / 100) * actual_return * qqq_close
        else:
            pnl = 0.0

        df.at[idx, "actual_direction"] = actual_direction
        df.at[idx, "actual_return"] = f"{actual_return:.6f}"
        df.at[idx, "correct"] = str(correct)
        df.at[idx, "pnl"] = f"{pnl:.2f}"
        resolved_count += 1

    if resolved_count > 0:
        df.to_csv(PREDICTION_LOG_PATH, index=False)
        print(f"\n  Resolved {resolved_count} pending prediction(s):")
        for idx in df.index[pending]:
            if df.at[idx, "actual_direction"] != "" and pd.notna(df.at[idx, "actual_direction"]):
                pred = df.at[idx, "prediction"]
                actual = df.at[idx, "actual_direction"]
                ret = df.at[idx, "actual_return"]
                mark = "Y" if df.at[idx, "correct"] == "1" else "X"
                print(f"    [{mark}] {df.at[idx, 'signal_date']} {df.at[idx, 'model_version']}: "
                      f"predicted {pred}, actual {actual} ({float(ret):+.2%})")


def _log_prediction(signal: dict, qqq_data: pd.DataFrame) -> None:
    """Append a new prediction to prediction_log.csv.

    Args:
        signal: Signal dictionary from generate_signal.
        qqq_data: QQQ DataFrame for finding next trading day.
    """
    signal_date = signal["date"]
    predict_date = _get_next_trading_day(signal_date, qqq_data)
    if predict_date is None:
        # Estimate next trading day as signal_date + 1 business day
        predict_date = str((pd.Timestamp(signal_date) + pd.offsets.BDay(1)).date())

    rec = signal["recommendation"]
    row = {
        "signal_date": signal_date,
        "predict_date": predict_date,
        "model_version": signal["model_version"],
        "prediction": signal["prediction"],
        "prob_up": signal["prob_up"],
        "knn_prob_up": signal["knn_prob_up"],
        "lr_prob_up": signal.get("lr_prob_up", ""),
        "leverage_level": rec["leverage_level"],
        "ticker": rec["ticker"],
        "shares": rec["shares"],
        "qqq_close": signal["prices"]["qqq_close"],
        "actual_direction": "",
        "actual_return": "",
        "correct": "",
        "pnl": "",
    }

    file_exists = PREDICTION_LOG_PATH.exists()

    # Check for duplicate (same signal_date + model_version)
    if file_exists:
        existing = pd.read_csv(PREDICTION_LOG_PATH, dtype=str)
        mask = (existing["signal_date"] == signal_date) & \
               (existing["model_version"] == signal["model_version"])
        if mask.any():
            # Update existing row instead of duplicating
            for col in ["prediction", "prob_up", "knn_prob_up", "lr_prob_up",
                        "leverage_level", "ticker", "shares", "qqq_close"]:
                existing.loc[mask, col] = str(row[col])
            existing.to_csv(PREDICTION_LOG_PATH, index=False)
            return

    with open(PREDICTION_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _print_scorecard() -> None:
    """Print a running accuracy scorecard from prediction_log.csv."""
    if not PREDICTION_LOG_PATH.exists():
        return

    df = pd.read_csv(PREDICTION_LOG_PATH, dtype=str)
    resolved = df[df["correct"].notna() & (df["correct"] != "")]
    if resolved.empty:
        return

    resolved = resolved.copy()
    resolved["correct"] = resolved["correct"].astype(int)
    resolved["pnl"] = resolved["pnl"].astype(float)

    total = len(resolved)
    accuracy = resolved["correct"].mean()
    cum_pnl = resolved["pnl"].sum()

    # Recent streak
    recent = resolved.sort_values("signal_date")["correct"].values
    streak = 0
    if len(recent) > 0:
        last_val = recent[-1]
        for v in reversed(recent):
            if v == last_val:
                streak += 1
            else:
                break
        streak_label = f"{streak}W" if last_val == 1 else f"{streak}L"
    else:
        streak_label = "-"

    print(f"\n{'=' * 60}")
    print(f"       PREDICTION SCORECARD")
    print(f"{'=' * 60}")
    print(f"  Total predictions resolved: {total}")
    print(f"  Accuracy:                   {accuracy:.1%} ({resolved['correct'].sum()}/{total})")
    print(f"  Current streak:             {streak_label}")
    print(f"  Cumulative P&L (proxy):     ${cum_pnl:+,.2f}")

    # V1 vs V2 breakdown
    for ver in ["v1", "v2"]:
        ver_df = resolved[resolved["model_version"] == ver]
        if not ver_df.empty:
            ver_acc = ver_df["correct"].mean()
            ver_pnl = ver_df["pnl"].sum()
            print(f"  {ver.upper()}: {ver_acc:.1%} accuracy ({len(ver_df)} predictions), P&L ${ver_pnl:+,.2f}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN QQQ Trading Model — Daily Signal Generator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--v1", action="store_true", help="V1: multi-K ensemble (backtested)")
    group.add_argument("--v2", action="store_true", help="V2: multi-K + recency weighting")
    group.add_argument("--both", action="store_true", help="Run both models and compare (default)")
    parser.add_argument("--balance", type=float, default=50000.0, help="Account balance (default: 50000)")
    args = parser.parse_args()

    model_config = load_config()

    if args.v1:
        versions = ["v1"]
    elif args.v2:
        versions = ["v2"]
    else:
        versions = ["v1", "v2"]  # default: run both

    print("KNN QQQ Trading Model — Daily Signal Generator")
    print(f"Model: K={ENSEMBLE_KS} ensemble, {model_config['metric']}, "
          f"window={model_config['training_window']}")
    print(f"  V1: uniform weighting (backtested +160%, Sharpe 0.66)")
    print(f"  V2: recency weighting (decay={RECENCY_DECAY_DAYS}d, regime-adaptive)")
    print(f"Running: {', '.join(v.upper() for v in versions)}\n")

    signals_dir = PROJECT_ROOT / "signals" / "daily_recommendations"
    signals_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data once (avoid re-downloading for --both)
    prepared = _prepare_training_data(model_config)

    # Resolve any pending predictions from previous runs using fresh QQQ data
    qqq_csv = PROJECT_ROOT / "data" / "raw" / "qqq_daily.csv"
    qqq_data = pd.read_csv(qqq_csv, index_col=0, parse_dates=True)
    _resolve_pending_predictions(qqq_data)

    for version in versions:
        signal = generate_signal(model_config, account_balance=args.balance,
                                 model_version=version, prepared_data=prepared)

        signal_path = signals_dir / f"{signal['date']}_{version}.json"
        with open(signal_path, "w") as f:
            json.dump(signal, f, indent=2)

        # Log prediction to CSV tracker
        _log_prediction(signal, qqq_data)

        print_signal(signal)
        print(f"  Signal saved to {signal_path}")

    if len(versions) == 2:
        print("\n" + "=" * 60)
        print("       MODEL COMPARISON")
        print("=" * 60)
        # Re-read saved signals for comparison
        v1_path = signals_dir / f"{signal['date']}_v1.json"
        v2_path = signals_dir / f"{signal['date']}_v2.json"
        if v1_path.exists() and v2_path.exists():
            with open(v1_path) as f:
                v1 = json.load(f)
            with open(v2_path) as f:
                v2 = json.load(f)
            v1r = v1["recommendation"]
            v2r = v2["recommendation"]
            agree = v1["prediction"] == v2["prediction"]
            print(f"  V1 (backtested): {v1['prediction']} P(up)={v1['prob_up']:.1%} → {v1r['leverage_level']:+d}% leverage")
            print(f"  V2 (recency):    {v2['prediction']} P(up)={v2['prob_up']:.1%} → {v2r['leverage_level']:+d}% leverage")
            print(f"  Models {'AGREE' if agree else 'DISAGREE'}")
        print("=" * 60)

    # Print running scorecard if we have resolved predictions
    _print_scorecard()

    print(f"\n  Prediction log: {PREDICTION_LOG_PATH}")
