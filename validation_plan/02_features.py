import pandas as pd

from features import calculate_features


if __name__ == "__main__":
    in_path = "data/eurusd_5m.csv"
    out_path = "data/eurusd_features.csv"

    df = pd.read_csv(in_path, index_col=0, parse_dates=True)
    df_feat = calculate_features(df)
    df_feat.to_csv(out_path)

    print(f"✓ Calculated features for {len(df_feat)} candles")
    print(f"Saved: {out_path}")
    print(f"Columns: {list(df_feat.columns)}")
