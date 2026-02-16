"""Model retraining pipeline: downloads fresh data, recomputes features, retrains KNN."""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from signals.generate_daily_signal import download_fresh_data, compute_all_features


def load_model_config() -> dict:
    """Load model configuration.

    Returns:
        Model configuration dictionary.
    """
    with open(PROJECT_ROOT / "models" / "best_config.json") as f:
        return json.load(f)


def retrain(config: dict) -> dict:
    """Retrain KNN model on latest data.

    Args:
        config: Model configuration.

    Returns:
        Training metrics dictionary.
    """
    print("Downloading fresh data...")
    master = download_fresh_data()
    print(f"  Data: {len(master)} rows, latest: {master.index[-1].date()}")

    print("Computing features...")
    master = compute_all_features(master)

    feat_cols = [c for c in master.columns if c.startswith("feat_")]
    master = master.dropna(subset=feat_cols)

    # Create target
    master["target"] = (master["Close"].pct_change().shift(-1) > 0).astype(int)
    master = master.iloc[:-1]  # Drop last row (no target)

    features = config["features"]
    window = config["training_window"]

    # Train on last `window` days
    train_data = master.iloc[-window:]
    X_train = train_data[features].values
    y_train = train_data["target"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    knn = KNeighborsClassifier(
        n_neighbors=config["k"],
        metric=config["metric"],
        weights=config["weights"],
    )
    knn.fit(X_train_scaled, y_train)

    # In-sample accuracy
    in_sample_acc = knn.score(X_train_scaled, y_train)

    # Save model artifact
    today = datetime.now().strftime("%Y-%m-%d")
    model_dir = PROJECT_ROOT / "models" / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": knn,
        "scaler": scaler,
        "config": config,
        "train_date_range": {
            "start": str(train_data.index[0].date()),
            "end": str(train_data.index[-1].date()),
        },
        "trained_at": datetime.now().isoformat(),
    }

    model_path = model_dir / f"model_{today}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    metrics = {
        "trained_at": datetime.now().isoformat(),
        "training_window": window,
        "n_features": len(features),
        "n_training_samples": len(X_train),
        "in_sample_accuracy": round(float(in_sample_acc), 4),
        "class_balance": round(float(y_train.mean()), 4),
        "train_date_range": {
            "start": str(train_data.index[0].date()),
            "end": str(train_data.index[-1].date()),
        },
        "model_path": str(model_path),
    }

    print(f"\n=== Retraining Complete ===")
    print(f"  Window: {metrics['train_date_range']['start']} to {metrics['train_date_range']['end']}")
    print(f"  Samples: {metrics['n_training_samples']}")
    print(f"  Features: {metrics['n_features']}")
    print(f"  In-sample accuracy: {metrics['in_sample_accuracy']:.2%}")
    print(f"  Class balance: {metrics['class_balance']:.2%} positive")
    print(f"  Model saved to {model_path}")

    return metrics


if __name__ == "__main__":
    config = load_model_config()
    print("KNN QQQ Trading Model — Retraining Pipeline")
    print(f"Config: K={config['k']}, {config['metric']}, window={config['training_window']}\n")

    metrics = retrain(config)

    # Save training log
    log_path = PROJECT_ROOT / "models" / "trained" / "training_log.json"
    log = []
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
    log.append(metrics)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log updated: {log_path}")
