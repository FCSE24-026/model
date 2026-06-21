"""EUR/USD 5m binary options modeling system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import exp
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote
from urllib.request import urlopen
from urllib.error import URLError
import csv


CALL = "CALL"
PUT = "PUT"
HOLD = "HOLD"
SIGMOID_CLAMP = 60.0


FEATURE_NAMES: Tuple[str, ...] = (
    "rsi_14",
    "bb_percent_b",
    "momentum_10",
    "macd_hist",
    "stoch_k",
    "atr_14",
    "sma20_sma50_delta",
    "volatility_regime",
    "price_zone_20",
    "volume_ratio_20",
    "cci_20",
    "williams_r_14",
    "spread_to_atr",
    "candle_body_to_atr",
    "return_1",
    "range_position_10",
)


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class TradingSystemConfig:
    lookback_candles: int = 100
    history_days: int = 90
    train_split: float = 0.8
    call_threshold: float = 0.55
    put_threshold: float = 0.45
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    subsample: float = 0.8


@dataclass(frozen=True)
class TradeLog:
    timestamp: datetime
    signal: str
    entry_price: float
    exit_price: float
    confidence: float
    stake: float = 1.0
    payout_ratio: float = 0.85

    @property
    def outcome(self) -> str:
        if self.signal == CALL:
            return "WIN" if self.exit_price > self.entry_price else "LOSS"
        if self.signal == PUT:
            return "WIN" if self.exit_price < self.entry_price else "LOSS"
        return "HOLD"

    @property
    def pnl(self) -> float:
        if self.outcome == "WIN":
            return self.stake * self.payout_ratio
        if self.outcome == "LOSS":
            return -self.stake
        return 0.0


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def _sma(values: Sequence[float], window: int, idx: int) -> float:
    if idx + 1 < window:
        return values[idx]
    segment = values[idx - window + 1 : idx + 1]
    return sum(segment) / window


def _std(values: Sequence[float], window: int, idx: int) -> float:
    if idx + 1 < window:
        return 0.0
    segment = values[idx - window + 1 : idx + 1]
    m = sum(segment) / window
    return (sum((x - m) ** 2 for x in segment) / window) ** 0.5


def _ema(values: Sequence[float], period: int) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    out: List[float] = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1 - alpha) * out[-1])
    return out


def _rsi(closes: Sequence[float], period: int = 14) -> List[float]:
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    out: List[float] = []
    for i in range(len(closes)):
        avg_gain = _sma(gains, period, i)
        avg_loss = _sma(losses, period, i)
        rs = _safe_div(avg_gain, avg_loss, default=10.0)
        rsi_raw = 100.0 - (100.0 / (1.0 + rs))
        out.append(_clamp(rsi_raw / 100.0, 0.0, 1.0))
    return out


def _atr(candles: Sequence[Candle], period: int = 14) -> List[float]:
    tr_values: List[float] = []
    for i, candle in enumerate(candles):
        prev_close = candles[i - 1].close if i > 0 else candle.close
        tr = max(
            candle.high - candle.low,
            abs(candle.high - prev_close),
            abs(candle.low - prev_close),
        )
        tr_values.append(tr)
    return [_sma(tr_values, period, i) for i in range(len(tr_values))]


def compute_features(candles: Sequence[Candle]) -> List[Dict[str, float]]:
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]
    rsis = _rsi(closes, 14)
    atrs = _atr(candles, 14)

    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    macd_signal = _ema(macd_line, 9)
    macd_hist = [a - b for a, b in zip(macd_line, macd_signal)]

    features: List[Dict[str, float]] = []
    for i, candle in enumerate(candles):
        sma20 = _sma(closes, 20, i)
        sma50 = _sma(closes, 50, i)
        std20 = _std(closes, 20, i)
        bb_upper = sma20 + (2 * std20)
        bb_lower = sma20 - (2 * std20)
        bb_percent_b = _safe_div(candle.close - bb_lower, bb_upper - bb_lower, 0.5)
        bb_percent_b = _clamp(bb_percent_b, 0.0, 1.0)

        momentum10 = _safe_div(
            candle.close - closes[max(i - 10, 0)], closes[max(i - 10, 0)], 0.0
        )
        momentum10 = _clamp(momentum10, -1.0, 1.0)

        low14 = min(lows[max(i - 13, 0) : i + 1])
        high14 = max(highs[max(i - 13, 0) : i + 1])
        stoch_k = _safe_div(candle.close - low14, high14 - low14, 0.5)
        stoch_k = _clamp(stoch_k, 0.0, 1.0)

        spread = candle.high - candle.low
        body = abs(candle.close - candle.open)

        low20 = min(lows[max(i - 19, 0) : i + 1])
        high20 = max(highs[max(i - 19, 0) : i + 1])
        price_zone = _safe_div(candle.close - low20, high20 - low20, 0.5)
        price_zone = _clamp(price_zone, 0.0, 1.0)

        volume_mean20 = _sma(volumes, 20, i)
        volume_ratio = _safe_div(candle.volume, volume_mean20, 1.0)

        tp = (candle.high + candle.low + candle.close) / 3.0
        tp_window = [
            (candles[j].high + candles[j].low + candles[j].close) / 3.0
            for j in range(max(0, i - 19), i + 1)
        ]
        tp_mean = sum(tp_window) / len(tp_window)
        tp_dev = sum(abs(x - tp_mean) for x in tp_window) / len(tp_window)
        cci20 = _safe_div(tp - tp_mean, 0.015 * tp_dev, 0.0)
        cci20 = _clamp(cci20 / 200.0, -1.0, 1.0)

        williams_r = -100.0 * _safe_div(high14 - candle.close, high14 - low14, 0.5)
        williams_r = _clamp(williams_r / 100.0, -1.0, 0.0)

        recent_returns = [
            _safe_div(closes[j] - closes[j - 1], closes[j - 1], 0.0)
            for j in range(max(1, i - 19), i + 1)
        ]
        vol = (sum(r * r for r in recent_returns) / max(len(recent_returns), 1)) ** 0.5
        volatility_regime = _clamp(_safe_div(vol, 0.0025, 0.0), 0.0, 2.0) - 1.0

        close_window10 = closes[max(i - 9, 0) : i + 1]
        min10, max10 = min(close_window10), max(close_window10)
        range_position10 = _safe_div(candle.close - min10, max10 - min10, 0.5)

        features.append(
            {
                "rsi_14": rsis[i],
                "bb_percent_b": bb_percent_b,
                "momentum_10": momentum10,
                "macd_hist": macd_hist[i],
                "stoch_k": stoch_k,
                "atr_14": atrs[i],
                "sma20_sma50_delta": _safe_div(sma20 - sma50, candle.close, 0.0),
                "volatility_regime": volatility_regime,
                "price_zone_20": price_zone,
                "volume_ratio_20": volume_ratio,
                "cci_20": cci20,
                "williams_r_14": williams_r,
                "spread_to_atr": _safe_div(spread, atrs[i], 0.0),
                "candle_body_to_atr": _safe_div(body, atrs[i], 0.0),
                "return_1": _safe_div(candle.close - closes[i - 1], closes[i - 1], 0.0)
                if i
                else 0.0,
                "range_position_10": _clamp(range_position10, 0.0, 1.0),
            }
        )
    return features


def build_labeled_dataset(candles: Sequence[Candle]) -> Tuple[List[List[float]], List[int]]:
    features = compute_features(candles)
    x: List[List[float]] = []
    y: List[int] = []
    for i in range(len(candles) - 1):
        x.append([features[i][name] for name in FEATURE_NAMES])
        y.append(1 if candles[i + 1].close > candles[i].close else 0)
    return x, y


def train_test_split_no_overlap(
    x: Sequence[List[float]], y: Sequence[int], train_ratio: float = 0.8
) -> Tuple[List[List[float]], List[List[float]], List[int], List[int]]:
    split = int(len(x) * train_ratio)
    return list(x[:split]), list(x[split:]), list(y[:split]), list(y[split:])


def fit_standard_scaler(rows: Sequence[Sequence[float]]) -> Dict[str, List[float]]:
    cols = len(rows[0]) if rows else len(FEATURE_NAMES)
    means = []
    stds = []
    for i in range(cols):
        values = [row[i] for row in rows] if rows else [0.0]
        m = mean(values)
        s = pstdev(values) if len(values) > 1 else 1.0
        means.append(m)
        stds.append(s if s > 0 else 1.0)
    return {"means": means, "stds": stds}


def apply_standard_scaler(
    rows: Sequence[Sequence[float]], scaler: Dict[str, Sequence[float]]
) -> List[List[float]]:
    means = scaler["means"]
    stds = scaler["stds"]
    return [[(v - means[i]) / stds[i] for i, v in enumerate(row)] for row in rows]


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def precision_recall(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[float, float]:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if b == 1 and a == 0)
    fn = sum(1 for a, b in zip(y_true, y_pred) if b == 0 and a == 1)
    precision = _safe_div(tp, tp + fp, 0.0)
    recall = _safe_div(tp, tp + fn, 0.0)
    return precision, recall


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def roc_auc_score(y_true: Sequence[int], y_prob_up: Sequence[float]) -> float:
    pos = [(p, y) for p, y in zip(y_prob_up, y_true) if y == 1]
    neg = [(p, y) for p, y in zip(y_prob_up, y_true) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p_pos, _ in pos:
        for p_neg, _ in neg:
            if p_pos > p_neg:
                wins += 1.0
            elif p_pos == p_neg:
                wins += 0.5
    return wins / (len(pos) * len(neg))


class _FallbackBinaryModel:
    def __init__(self) -> None:
        self.weights: List[float] = [0.0 for _ in FEATURE_NAMES]
        self.bias = 0.0

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        if not x:
            return
        pos = [row for row, label in zip(x, y) if label == 1]
        neg = [row for row, label in zip(x, y) if label == 0]
        for i in range(len(self.weights)):
            pos_m = mean([row[i] for row in pos]) if pos else 0.0
            neg_m = mean([row[i] for row in neg]) if neg else 0.0
            self.weights[i] = pos_m - neg_m
        self.bias = _safe_div(sum(y), len(y), 0.5) - 0.5

    def predict_proba(self, x: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
        out = []
        for row in x:
            score = self.bias + sum(w * v for w, v in zip(self.weights, row))
            # exp(±60) is already extreme, while staying far from overflow.
            score = _clamp(score, -SIGMOID_CLAMP, SIGMOID_CLAMP)
            p_up = 1 / (1 + exp(-score))
            out.append((1 - p_up, p_up))
        return out

    @property
    def feature_importances_(self) -> List[float]:
        values = [abs(w) for w in self.weights]
        total = sum(values)
        if total == 0:
            return [0.0 for _ in values]
        return [v / total for v in values]


class BinaryOptionsTradingSystem:
    def __init__(self, config: Optional[TradingSystemConfig] = None) -> None:
        self.config = config or TradingSystemConfig()
        self.scaler: Optional[Dict[str, List[float]]] = None
        self.model = None
        self._using_xgboost = False

    def _create_model(self) -> Union[object, _FallbackBinaryModel]:
        try:
            from xgboost import XGBClassifier  # type: ignore

            self._using_xgboost = True
            return XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )
        except Exception:
            self._using_xgboost = False
            return _FallbackBinaryModel()

    def fit(self, candles: Sequence[Candle]) -> Dict[str, float]:
        x, y = build_labeled_dataset(candles)
        x_train, x_test, y_train, y_test = train_test_split_no_overlap(
            x, y, self.config.train_split
        )

        self.scaler = fit_standard_scaler(x_train)
        x_train_scaled = apply_standard_scaler(x_train, self.scaler)
        x_test_scaled = apply_standard_scaler(x_test, self.scaler)

        self.model = self._create_model()
        self.model.fit(x_train_scaled, y_train)
        proba = [p[1] for p in self.model.predict_proba(x_test_scaled)]
        y_pred = [1 if p >= 0.5 else 0 for p in proba]

        train_proba = [p[1] for p in self.model.predict_proba(x_train_scaled)]
        train_pred = [1 if p >= 0.5 else 0 for p in train_proba]

        cm = confusion_matrix(y_test, y_pred)
        precision, recall = precision_recall(y_test, y_pred)

        return {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "precision": precision,
            "recall": recall,
            "sensitivity": _safe_div(cm["tp"], cm["tp"] + cm["fn"], 0.0),
            "specificity": _safe_div(cm["tn"], cm["tn"] + cm["fp"], 0.0),
            "overfitting_gap": accuracy_score(y_train, train_pred)
            - accuracy_score(y_test, y_pred),
        }

    def predict_signal_from_features(
        self, feature_row: Sequence[float]
    ) -> Dict[str, Union[float, str]]:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model must be fit before prediction.")
        scaled = apply_standard_scaler([feature_row], self.scaler)
        p_up = self.model.predict_proba(scaled)[0][1]
        return signal_from_probability(p_up, self.config.call_threshold, self.config.put_threshold)

    def predict_signal(self, candles: Sequence[Candle]) -> Dict[str, Union[float, str]]:
        if len(candles) < 2:
            raise ValueError("At least 2 candles are required.")
        row = build_labeled_dataset(candles)[0][-1]
        return self.predict_signal_from_features(row)

    def feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model must be fit first.")
        raw = list(self.model.feature_importances_)
        total = sum(raw) or 1.0
        return {name: (value / total) for name, value in zip(FEATURE_NAMES, raw)}

    def walk_forward_validation(
        self, candles: Sequence[Candle], train_window: int = 500, test_window: int = 50
    ) -> Dict[str, Union[float, List[float]]]:
        x, y = build_labeled_dataset(candles)
        accuracies: List[float] = []
        for start in range(0, max(len(x) - train_window - test_window + 1, 0), test_window):
            x_train = x[start : start + train_window]
            y_train = y[start : start + train_window]
            x_test = x[start + train_window : start + train_window + test_window]
            y_test = y[start + train_window : start + train_window + test_window]
            scaler = fit_standard_scaler(x_train)
            x_train_s = apply_standard_scaler(x_train, scaler)
            x_test_s = apply_standard_scaler(x_test, scaler)
            model = self._create_model()
            model.fit(x_train_s, y_train)
            p = [r[1] for r in model.predict_proba(x_test_s)]
            pred = [1 if v >= 0.5 else 0 for v in p]
            accuracies.append(accuracy_score(y_test, pred))

        if not accuracies:
            return {"period_accuracies": [], "average": 0.0, "std_dev": 0.0, "min": 0.0, "max": 0.0}
        return {
            "period_accuracies": accuracies,
            "average": sum(accuracies) / len(accuracies),
            "std_dev": pstdev(accuracies) if len(accuracies) > 1 else 0.0,
            "min": min(accuracies),
            "max": max(accuracies),
        }


def signal_from_probability(
    probability_up: float, call_threshold: float = 0.55, put_threshold: float = 0.45
) -> Dict[str, Union[float, str]]:
    if probability_up > call_threshold:
        signal = CALL
    elif probability_up < put_threshold:
        signal = PUT
    else:
        signal = HOLD
    confidence = max(probability_up, 1.0 - probability_up)
    return {"signal": signal, "probability_up": probability_up, "confidence": confidence}


def expected_roi_per_trade(win_rate: float, payout_ratio: float = 0.85, loss_ratio: float = 1.0) -> float:
    return (win_rate * payout_ratio) - ((1.0 - win_rate) * loss_ratio)


def confidence_calibration(
    y_true: Sequence[int], y_prob_up: Sequence[float], bin_size: float = 0.05
) -> List[Dict[str, float]]:
    bins: Dict[int, List[Tuple[int, float]]] = {}
    for actual, prob_up in zip(y_true, y_prob_up):
        confidence = max(prob_up, 1.0 - prob_up)
        idx = min(int(confidence / bin_size), int(1.0 / bin_size) - 1)
        bins.setdefault(idx, []).append((actual, prob_up))

    rows: List[Dict[str, float]] = []
    for idx in sorted(bins):
        items = bins[idx]
        predicted = [1 if p >= 0.5 else 0 for _, p in items]
        matches = sum(1 for (a, _), pred in zip(items, predicted) if a == pred)
        lower = idx * bin_size
        upper = min((idx + 1) * bin_size, 1.0)
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "sample_count": float(len(items)),
                "predicted_confidence": sum(max(p, 1.0 - p) for _, p in items) / len(items),
                "actual_win_rate": matches / len(items),
            }
        )
    return rows


def classify_operating_status(
    rolling_7d_accuracy: float, high_confidence_win_rate: float, rsi_importance: float
) -> Dict[str, str]:
    if rolling_7d_accuracy < 0.58 or high_confidence_win_rate < 0.55 or rsi_importance < 0.15:
        return {"status": "RED", "action": "Stop trading and retrain full model."}
    if (
        rolling_7d_accuracy < 0.62
        or high_confidence_win_rate < 0.65
        or rsi_importance < 0.25
    ):
        return {"status": "YELLOW", "action": "Reduce size and retrain on latest candles."}
    return {"status": "GREEN", "action": "Keep trading with normal size."}


def fetch_eurusd_5m_history(history_days: int = 90) -> List[Candle]:
    """
    Download EUR/USD 5m candles from Yahoo Finance chart endpoint.
    Yahoo may throttle frequent calls, so use caller-side scheduling/caching.
    """

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=history_days)
    query_template = (
        "period1={period1}&period2={period2}&interval=5m"
        "&events=history&includeAdjustedClose=true"
    )
    params = query_template.format(
        period1=int(start.timestamp()),
        period2=int(now.timestamp()),
    )
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{quote('EURUSD=X')}?{params}"
    try:
        with urlopen(url, timeout=30) as response:
            rows = response.read().decode("utf-8").splitlines()
    except (URLError, OSError) as exc:
        raise RuntimeError("Unable to fetch EUR/USD data from Yahoo Finance.") from exc
    reader = csv.DictReader(rows)
    candles: List[Candle] = []
    for row in reader:
        close = (row.get("Close") or "").strip()
        if not close or close.lower() == "null":
            continue
        candles.append(
            Candle(
                timestamp=datetime.fromisoformat(row["Date"]).replace(tzinfo=timezone.utc),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(close),
                volume=float(row["Volume"] or 0),
            )
        )
    return candles
