"""Daily signal generation: run at EOD to generate next-day TQQQ/SQQQ recommendation.

Two model versions available:
  V1 (ensemble): KNN 70% + Logistic Regression 30% blend — higher Sharpe (1.30), smoother
  V2 (knn-only): KNN only — higher total return, better in bear markets

Usage:
  python signals/generate_daily_signal.py          # default (V1 ensemble)
  python signals/generate_daily_signal.py --v1     # V1 ensemble
  python signals/generate_daily_signal.py --v2     # V2 KNN-only
  python signals/generate_daily_signal.py --both   # run both, show comparison
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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


def download_fresh_data() -> pd.DataFrame:
    """Download latest data for all tickers and merge.

    Returns:
        Merged DataFrame with all tickers aligned.
    """
    tickers = {
        "QQQ": None,
        "^VIX": "VIX",
        "SPY": "SPY",
        "IWM": "IWM",
        "TLT": "TLT",
        "GLD": "GLD",
    }

    dfs = {}
    for yf_ticker, prefix in tickers.items():
        t = yf.Ticker(yf_ticker)
        df = t.history(start="2003-01-01", auto_adjust=False)
        cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        df = df[[c for c in cols if c in df.columns]]
        df.index = pd.to_datetime(df.index).tz_localize(None)

        if prefix:
            df.columns = [f"{prefix}_{c}" for c in df.columns]

        dfs[yf_ticker] = df

    # Inner join
    master = dfs["QQQ"]
    for key in ["^VIX", "SPY", "IWM", "TLT", "GLD"]:
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
        Tuple of (train_X_scaled, train_y, today_X_scaled, today_row, today_date, scaler).
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

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    today_X_scaled = scaler.transform(today_X)

    return train_X_scaled, train_y, today_X_scaled, today_row, today_date, scaler


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
        model_version: "v1" (KNN+LR ensemble) or "v2" (KNN only).
        prepared_data: Pre-computed training data tuple to avoid re-downloading.

    Returns:
        Signal recommendation dictionary.
    """
    if prepared_data is not None:
        train_X_scaled, train_y, today_X_scaled, today_row, today_date, scaler = prepared_data
    else:
        train_X_scaled, train_y, today_X_scaled, today_row, today_date, scaler = \
            _prepare_training_data(model_config)

    features = model_config["features"]

    # Fit KNN
    knn = KNeighborsClassifier(
        n_neighbors=model_config["k"],
        metric=model_config["metric"],
        weights=model_config["weights"],
    )
    knn.fit(train_X_scaled, train_y)
    knn_proba = knn.predict_proba(today_X_scaled)[0]
    knn_prob_up = float(knn_proba[1] if len(knn_proba) > 1 else knn_proba[0])

    if model_version == "v1":
        # V1: KNN 70% + LR 30% blend
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr.fit(train_X_scaled, train_y)
        lr_proba = lr.predict_proba(today_X_scaled)[0]
        lr_prob_up = float(lr_proba[1] if len(lr_proba) > 1 else lr_proba[0])
        prob_up = 0.7 * knn_prob_up + 0.3 * lr_prob_up
        version_label = "V1 (KNN 70% + LR 30%)"
    else:
        # V2: KNN only
        prob_up = knn_prob_up
        lr_prob_up = None
        version_label = "V2 (KNN only)"

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
        "knn_prob_up": round(float(knn_prob_up), 4),
        "lr_prob_up": round(float(lr_prob_up), 4) if lr_prob_up is not None else None,
        "recommendation": recommendation,
        "model_details": {
            "k": model_config["k"],
            "metric": model_config["metric"],
            "weights": model_config["weights"],
            "training_window": model_config["training_window"],
            "features_used": features,
            "blend_weights": {"knn": 0.7, "lr": 0.3} if model_version == "v1" else {"knn": 1.0},
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
    if signal.get("lr_prob_up") is not None:
        print(f"    KNN P(up):    {signal['knn_prob_up']:.1%}")
        print(f"    LR P(up):     {signal['lr_prob_up']:.1%}")
        print(f"    Blended:      {prob:.1%}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN QQQ Trading Model — Daily Signal Generator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--v1", action="store_true", help="V1: KNN+LR ensemble (default)")
    group.add_argument("--v2", action="store_true", help="V2: KNN only")
    group.add_argument("--both", action="store_true", help="Run both models and compare")
    parser.add_argument("--balance", type=float, default=50000.0, help="Account balance (default: 50000)")
    args = parser.parse_args()

    model_config = load_config()

    if args.both:
        # Run both versions
        versions = ["v1", "v2"]
    elif args.v2:
        versions = ["v2"]
    else:
        versions = ["v1"]  # default

    print("KNN QQQ Trading Model — Daily Signal Generator")
    print(f"Model: K={model_config['k']}, {model_config['metric']}, "
          f"window={model_config['training_window']}")
    print(f"Running: {', '.join(v.upper() for v in versions)}\n")

    signals_dir = PROJECT_ROOT / "signals" / "daily_recommendations"
    signals_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data once (avoid re-downloading for --both)
    prepared = _prepare_training_data(model_config)

    for version in versions:
        signal = generate_signal(model_config, account_balance=args.balance,
                                 model_version=version, prepared_data=prepared)

        signal_path = signals_dir / f"{signal['date']}_{version}.json"
        with open(signal_path, "w") as f:
            json.dump(signal, f, indent=2)

        print_signal(signal)
        print(f"  Signal saved to {signal_path}")

    if args.both and len(versions) == 2:
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
            print(f"  V1 (ensemble): {v1['prediction']} P(up)={v1['prob_up']:.1%} → {v1r['leverage_level']:+d}% leverage")
            print(f"  V2 (KNN-only): {v2['prediction']} P(up)={v2['prob_up']:.1%} → {v2r['leverage_level']:+d}% leverage")
            print(f"  Models {'AGREE' if agree else 'DISAGREE'}")
        print("=" * 60)
