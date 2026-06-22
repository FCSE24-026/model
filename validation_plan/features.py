import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange


# Canonical feature order used for model training/inference.
FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_upper",
    "bb_lower",
    "bb_percent_b",
    "atr_14",
    "stoch_k",
    "stoch_d",
    "momentum_10",
    "momentum_5",
    "volatility_20",
    "sma_20",
    "sma_50",
]


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical features and next-candle label."""
    data = df.copy()

    if data.empty:
        return data

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rsi = RSIIndicator(close=data["close"], window=14)
    data["rsi_14"] = rsi.rsi() / 100.0

    macd = MACD(close=data["close"], window_slow=26, window_fast=12, window_sign=9)
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_histogram"] = macd.macd_diff()

    bb = BollingerBands(close=data["close"], window=20, window_dev=2)
    data["bb_upper"] = bb.bollinger_hband()
    data["bb_lower"] = bb.bollinger_lband()
    data["bb_percent_b"] = bb.bollinger_pband()

    atr = AverageTrueRange(high=data["high"], low=data["low"], close=data["close"], window=14)
    data["atr_14"] = atr.average_true_range()

    stoch = StochasticOscillator(high=data["high"], low=data["low"], close=data["close"], window=14, smooth_window=3)
    data["stoch_k"] = stoch.stoch() / 100.0
    data["stoch_d"] = stoch.stoch_signal() / 100.0

    data["momentum_10"] = data["close"].pct_change(10)
    data["momentum_5"] = data["close"].pct_change(5)
    data["volatility_20"] = data["close"].pct_change().rolling(20).std()
    data["sma_20"] = data["close"].rolling(20).mean()
    data["sma_50"] = data["close"].rolling(50).mean()

    data["label"] = (data["close"].shift(-1) > data["close"]).astype(int)

    data = data.dropna().copy()
    return data
