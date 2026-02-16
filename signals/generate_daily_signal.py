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
from signals.position_sizer import PositionSizer


def load_configs() -> tuple[dict, dict]:
    """Load model and sizing configurations.

    Returns:
        Tuple of (model config, sizing config).
    """
    with open(PROJECT_ROOT / "models" / "best_config.json") as f:
        model_config = json.load(f)
    with open(PROJECT_ROOT / "signals" / "sizing_config.json") as f:
        sizing_config = json.load(f)
    return model_config, sizing_config


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
    return df


def generate_signal(
    model_config: dict,
    sizing_config: dict,
    account_balance: float = 50000.0,
) -> dict:
    """Generate today's trading signal.

    Args:
        model_config: KNN model configuration.
        sizing_config: Position sizing configuration.
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

    # Position sizing
    sizer = PositionSizer(
        bull_threshold=sizing_config["bull_threshold"],
        bear_threshold=sizing_config["bear_threshold"],
        max_position_pct=sizing_config["max_position_pct"],
        scaling=sizing_config["scaling"],
    )
    recommendation = sizer.size(prob_up, account_balance, tqqq_price, sqqq_price)

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
    print("\n" + "=" * 55)
    print("  KNN QQQ TRADING MODEL — DAILY SIGNAL")
    print("=" * 55)
    print(f"  Date:        {signal['date']}")
    print(f"  Prediction:  QQQ {signal['prediction']} (P(up) = {signal['prob_up']:.1%})")
    print(f"  QQQ Close:   ${signal['prices']['qqq_close']:.2f}")
    print(f"")

    if rec["action"] == "BUY":
        print(f"  >>> RECOMMENDATION: BUY {rec['shares']} shares of {rec['ticker']} <<<")
        print(f"  Dollar Amount: ${rec['dollar_amount']:,.2f}")
        print(f"  Position Size: {rec['position_pct']:.1%} of account")
        print(f"  {rec['ticker']} Price: ${signal['prices'].get(rec['ticker'].lower() + '_close', 'N/A')}")
    else:
        print(f"  >>> RECOMMENDATION: STAY IN CASH <<<")
        print(f"  Model confidence too low to trade.")

    print(f"\n  Confidence:  {rec['confidence']:.1%}")
    print(f"  Account:     ${signal['account_balance']:,.2f}")
    print("=" * 55)


if __name__ == "__main__":
    model_config, sizing_config = load_configs()

    print("KNN QQQ Trading Model — Daily Signal Generator")
    print(f"Model: K={model_config['k']}, {model_config['metric']}, "
          f"window={model_config['training_window']}\n")

    signal = generate_signal(model_config, sizing_config)

    # Save signal
    signals_dir = PROJECT_ROOT / "signals" / "daily_recommendations"
    signals_dir.mkdir(parents=True, exist_ok=True)
    signal_path = signals_dir / f"{signal['date']}.json"
    with open(signal_path, "w") as f:
        json.dump(signal, f, indent=2)

    print_signal(signal)
    print(f"\n  Signal saved to {signal_path}")
