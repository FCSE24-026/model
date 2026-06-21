import json
import os
import pickle
from datetime import datetime, timezone

import yfinance as yf

from features import calculate_features


def generate_signal(model, scaler):
    raw = yf.download("EURUSD=X", period="5d", interval="5m", progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError("No data returned for EURUSD=X")

    df = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    if "volume" not in df.columns:
        df["volume"] = 0.0

    feat = calculate_features(df)
    if feat.empty:
        raise RuntimeError("Feature frame is empty")

    latest = feat.iloc[-1]
    x = latest[:-1].values.reshape(1, -1)
    x_scaled = scaler.transform(x)

    proba = model.predict_proba(x_scaled)[0]
    direction = "UP" if proba[1] > 0.5 else "DOWN"
    confidence = float(max(proba))

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direction": direction,
        "confidence": confidence,
        "rsi": float(latest["rsi_14"]),
        "bb_percent_b": float(latest["bb_percent_b"]),
        "momentum_10": float(latest["momentum_10"]),
        "volatility_20": float(latest["volatility_20"]),
    }


if __name__ == "__main__":
    with open("models/eurusd_model.pkl", "rb") as f:
        model, scaler, _ = pickle.load(f)

    signal = generate_signal(model, scaler)
    print(json.dumps(signal, indent=2))

    os.makedirs("signals", exist_ok=True)
    out_path = "signals/latest_signal.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(signal, f, indent=2)

    print(f"✓ Saved {out_path}")
