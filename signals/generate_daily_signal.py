"""Daily signal generation: run at EOD to generate next-day TQQQ/SQQQ recommendation."""

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


def generate_signal(
    model_config: dict,
    account_balance: float = 50000.0,
) -> dict:
    """Generate today's trading signal.

    Args:
        model_config: KNN model configuration.
        account_balance: Current account balance.

    Returns:
        Signal recommendation dictionary.
    """
    print("Downloading latest market data...")
    master = download_fresh_data()
    print(f"  Data: {len(master)} rows, latest date: {master.index[-1].date()}")

    print("Computing features...")
    master = compute_all_features(master)

    # Drop NaN rows
    feat_cols = [c for c in master.columns if c.startswith("feat_")]
    master = master.dropna(subset=feat_cols)
    print(f"  Clean data: {len(master)} rows")

    features = model_config["features"]
    training_window = model_config["training_window"]

    # Train on the most recent window, predict for today
    X = master[features].values
    # We need a target for training — use daily return direction
    master["_target"] = (master["Close"].pct_change().shift(-1) > 0).astype(int)
    # Drop last row (no target)
    master = master.iloc[:-1]
    X = master[features].values
    y = master["_target"].values

    # Training data: last `training_window` rows before today
    train_X = X[-training_window:]
    train_y = y[-training_window:]

    # Today's features (last row after training)
    # Re-download to get today's row
    master_full = download_fresh_data()
    master_full = compute_all_features(master_full)
    master_full = master_full.dropna(subset=feat_cols)
    today_row = master_full.iloc[-1]
    today_X = today_row[features].values.reshape(1, -1)
    today_date = master_full.index[-1].date()

    # Scale
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    today_X_scaled = scaler.transform(today_X)

    # Fit and predict
    knn = KNeighborsClassifier(
        n_neighbors=model_config["k"],
        metric=model_config["metric"],
        weights=model_config["weights"],
    )
    knn.fit(train_X_scaled, train_y)

    prediction = knn.predict(today_X_scaled)[0]
    proba = knn.predict_proba(today_X_scaled)[0]
    prob_up = proba[1] if len(proba) > 1 else proba[0]

    # Get current TQQQ/SQQQ prices
    tqqq_price = yf.Ticker("TQQQ").history(period="1d", auto_adjust=False)["Close"].iloc[-1]
    sqqq_price = yf.Ticker("SQQQ").history(period="1d", auto_adjust=False)["Close"].iloc[-1]

    # Get realized vol for vol targeting
    realized_vol = None
    vol_col = "feat_realized_vol_20d"
    if vol_col in today_row.index:
        realized_vol = float(today_row[vol_col])

    # Position sizing with discrete leverage levels + vol targeting
    sizer = PositionSizer(vol_target_multiple=1.0)
    recommendation = sizer.size(
        prob_up, account_balance, tqqq_price, sqqq_price,
        realized_vol_annual=realized_vol,
    )

    signal = {
        "date": str(today_date),
        "generated_at": datetime.now().isoformat(),
        "prediction": "UP" if prediction == 1 else "DOWN",
        "prob_up": round(float(prob_up), 4),
        "recommendation": recommendation,
        "model_details": {
            "k": model_config["k"],
            "metric": model_config["metric"],
            "weights": model_config["weights"],
            "training_window": training_window,
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

    print("\n" + "=" * 60)
    print("       KNN QQQ TRADING MODEL — DAILY SIGNAL")
    print("=" * 60)
    print(f"  Date:           {signal['date']}")
    print(f"  QQQ Close:      ${signal['prices']['qqq_close']:.2f}")
    print(f"  QQQ Prediction: {signal['prediction']} (P(up) = {prob:.1%})")
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
    print("=" * 60)


if __name__ == "__main__":
    model_config = load_config()

    print("KNN QQQ Trading Model — Daily Signal Generator")
    print(f"Model: K={model_config['k']}, {model_config['metric']}, "
          f"window={model_config['training_window']}\n")

    signal = generate_signal(model_config)

    # Save signal
    signals_dir = PROJECT_ROOT / "signals" / "daily_recommendations"
    signals_dir.mkdir(parents=True, exist_ok=True)
    signal_path = signals_dir / f"{signal['date']}.json"
    with open(signal_path, "w") as f:
        json.dump(signal, f, indent=2)

    print_signal(signal)
    print(f"\n  Signal saved to {signal_path}")
