import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    df = pd.read_csv("data/eurusd_features.csv", index_col=0, parse_dates=True)

    print("Running walk-forward validation...")

    window_size = 500
    step_size = 50
    rows = []

    for i in range(window_size, len(df) - step_size + 1, step_size):
        x_train = df.iloc[i - window_size : i, :-1].values
        y_train = df.iloc[i - window_size : i, -1].values
        x_test = df.iloc[i : i + step_size, :-1].values
        y_test = df.iloc[i : i + step_size, -1].values

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
        model.fit(x_train_scaled, y_train)

        y_pred = model.predict(x_test_scaled)
        acc = (y_pred == y_test).mean()

        rows.append(
            {
                "period": len(rows) + 1,
                "test_date": str(df.index[i]),
                "accuracy": acc,
                "n_trades": len(y_test),
            }
        )
        print(f"Period {len(rows):02d}: {df.index[i]} | accuracy={acc:.1%} | trades={len(y_test)}")

    results = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out_path = "results/walk_forward_results.csv"
    results.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("WALK-FORWARD SUMMARY")
    print("=" * 60)
    if results.empty:
        print("No windows available; increase dataset size or reduce window parameters.")
    else:
        print(f"Periods: {len(results)}")
        print(f"Total trades: {results['n_trades'].sum()}")
        print(f"Average accuracy: {results['accuracy'].mean():.1%}")
        print(f"Std dev: {results['accuracy'].std(ddof=0):.1%}")
        print(f"Min: {results['accuracy'].min():.1%}")
        print(f"Max: {results['accuracy'].max():.1%}")

    print(f"\n✓ Saved {out_path}")
