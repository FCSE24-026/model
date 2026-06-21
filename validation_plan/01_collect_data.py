import os
import pandas as pd
import yfinance as yf


def download_5m_data(ticker: str, periods=("90d", "60d", "30d")) -> pd.DataFrame:
    for period in periods:
        df = yf.download(ticker, period=period, interval="5m", progress=False, auto_adjust=False)
        if not df.empty:
            print(f"Downloaded {ticker} with period={period}")
            return df
    return pd.DataFrame()


if __name__ == "__main__":
    print("Downloading EUR/USD 5m data...")
    raw = download_5m_data("EURUSD=X")

    if raw.empty:
        raise SystemExit("No data returned from Yahoo Finance.")

    df = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    keep_cols = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    if "volume" not in df.columns:
        df["volume"] = 0.0

    os.makedirs("data", exist_ok=True)
    out_path = "data/eurusd_5m.csv"
    df.to_csv(out_path)

    print(f"✓ Saved {len(df)} candles to {out_path}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
