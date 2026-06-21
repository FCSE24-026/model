import pickle
import numpy as np
import pandas as pd
import yfinance as yf

from features import FEATURE_COLUMNS, calculate_features


def download_5m_data(ticker: str, periods=("60d", "30d")) -> pd.DataFrame:
    for period in periods:
        df = yf.download(ticker, period=period, interval="5m", progress=False, auto_adjust=False)
        if not df.empty:
            return df
    return pd.DataFrame()


if __name__ == "__main__":
    with open("models/eurusd_model.pkl", "rb") as f:
        model, scaler_eurusd, feature_names = pickle.load(f)

    pairs = {
        "GBPUSD=X": "GBP/USD",
        "^GDAXI": "DAX40",
        "GC=F": "Gold",
    }

    print("Testing EUR/USD-trained model on other instruments...")
    print("=" * 60)

    results = {}

    for ticker, name in pairs.items():
        print(f"\n{name} ({ticker})")
        raw = download_5m_data(ticker)
        if raw.empty:
            print("  ✗ No data")
            continue

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

        df_feat = calculate_features(df)
        if df_feat.empty:
            print("  ✗ Not enough processed rows")
            continue

        split_idx = int(len(df_feat) * 0.8)
        x_test = df_feat.iloc[split_idx:][FEATURE_COLUMNS].values
        y_test = df_feat.iloc[split_idx:]["label"].values

        x_test_scaled = scaler_eurusd.transform(x_test)
        y_pred = model.predict(x_test_scaled)
        acc = (y_pred == y_test).mean()

        results[name] = {"accuracy": acc, "n_trades": len(y_test)}
        print(f"  ✓ Accuracy: {acc:.1%} ({len(y_test)} trades)")

    print("\n" + "=" * 60)
    print("MULTI-PAIR SUMMARY")
    print("=" * 60)
    if not results:
        print("No successful pair tests.")
    else:
        accuracies = [v["accuracy"] for v in results.values()]
        for pair_name, row in results.items():
            print(f"{pair_name:<10} accuracy={row['accuracy']:.1%} trades={row['n_trades']}")
        print(f"Average: {np.mean(accuracies):.1%}")
        print(f"Min/Max: {np.min(accuracies):.1%} / {np.max(accuracies):.1%}")
