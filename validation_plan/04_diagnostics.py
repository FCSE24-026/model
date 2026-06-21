import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from features import FEATURE_COLUMNS


if __name__ == "__main__":
    df = pd.read_csv("data/eurusd_features.csv", index_col=0, parse_dates=True)
    with open("models/eurusd_model.pkl", "rb") as f:
        model, scaler, _ = pickle.load(f)

    split_idx = int(len(df) * 0.8)
    x_test = df.iloc[split_idx:][FEATURE_COLUMNS].values
    y_test = df.iloc[split_idx:]["label"].values

    x_test_scaled = scaler.transform(x_test)
    y_pred = model.predict(x_test_scaled)
    y_proba = model.predict_proba(x_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    accuracy = (y_pred == y_test).mean()

    print("=" * 60)
    print("DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    print(f"AUC: {auc:.3f} (random=0.500)")
    print(f"Accuracy: {accuracy:.1%}")

    rolling_window = 50
    roll = []
    if len(y_test) > rolling_window:
        for i in range(0, len(y_test) - rolling_window + 1):
            roll.append((y_pred[i : i + rolling_window] == y_test[i : i + rolling_window]).mean())

    if roll:
        std = float(np.std(roll))
        print(f"Rolling {rolling_window}-trade accuracy std: {std:.1%}")
    else:
        std = 1.0
        print("Not enough data for rolling stability window.")

    print("\nConfidence bins")
    bins = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    names = ["45-50%", "50-55%", "55-60%", "60-65%", "65-70%", "70%+"]

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (y_proba >= lo) & (y_proba < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc_bin = (y_pred[mask] == y_test[mask]).mean()
        exp_mid = (lo + hi) / 2
        mark = "✓" if abs(acc_bin - exp_mid) < 0.08 else "✗"
        print(f"{names[i]:<10} trades={n:<4} actual={acc_bin:.1%} expected~{exp_mid:.1%} {mark}")

    checks = [
        (auc > 0.62, "AUC > 0.62"),
        (accuracy > 0.60, "Accuracy > 60%"),
        (std < 0.07, "Rolling std < 7%"),
    ]

    print("\nFINAL VERDICT")
    for ok, label in checks:
        print(f"{'✓' if ok else '✗'} {label}")

    if all(ok for ok, _ in checks):
        print("\n✅ READY FOR WEEK 2")
    else:
        print("\n⚠️ NEEDS WORK")
