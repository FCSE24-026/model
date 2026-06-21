import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def split_xy(df: pd.DataFrame):
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return x, y


if __name__ == "__main__":
    df = pd.read_csv("data/eurusd_features.csv", index_col=0, parse_dates=True)
    print(f"Total candles: {len(df)}")

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    x_train, y_train = split_xy(train_df)
    x_test, y_test = split_xy(test_df)

    print(f"Train set: {len(x_train)}")
    print(f"Test set: {len(x_test)}")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        subsample=0.8,
    )

    print("Training GradientBoosting baseline model...")
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    y_proba = model.predict_proba(x_test_scaled)[:, 1]

    accuracy = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.1%}")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Sensitivity (UP recall): {tp / max(tp + fn, 1):.1%}")
    print(f"Specificity (DOWN recall): {tn / max(tn + fp, 1):.1%}")

    feature_names = df.columns[:-1]
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("\nTop 10 features:")
    cumsum = 0.0
    for i in range(min(10, len(sorted_idx))):
        idx = sorted_idx[i]
        cumsum += importances[idx]
        print(f"{feature_names[idx]:<20} {importances[idx]:>7.2%}   cumulative={cumsum:>7.2%}")

    os.makedirs("models", exist_ok=True)
    with open("models/eurusd_model.pkl", "wb") as f:
        pickle.dump((model, scaler, list(feature_names)), f)

    print("\n✓ Saved model to models/eurusd_model.pkl")
