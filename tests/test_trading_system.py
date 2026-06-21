from datetime import datetime, timedelta, timezone
import unittest

from trading_system import (
    FEATURE_NAMES,
    BinaryOptionsTradingSystem,
    Candle,
    HOLD,
    PUT,
    CALL,
    build_labeled_dataset,
    compute_features,
    signal_from_probability,
    train_test_split_no_overlap,
)


def _make_candles(count: int = 220):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 1.1000
    for i in range(count):
        drift = 0.00015 if (i % 12) < 7 else -0.00008
        price += drift
        low = price - 0.0004
        high = price + 0.0004
        close = price + (0.00008 if i % 2 == 0 else -0.00006)
        candles.append(
            Candle(
                timestamp=base + timedelta(minutes=5 * i),
                open=price,
                high=high,
                low=low,
                close=close,
                volume=100 + (i % 20) * 3,
            )
        )
    return candles


class TradingSystemTests(unittest.TestCase):
    def test_compute_features_has_16_indicators(self):
        rows = compute_features(_make_candles(80))
        self.assertTrue(rows)
        self.assertEqual(set(rows[0].keys()), set(FEATURE_NAMES))

    def test_signal_thresholds(self):
        self.assertEqual(signal_from_probability(0.60)["signal"], CALL)
        self.assertEqual(signal_from_probability(0.40)["signal"], PUT)
        self.assertEqual(signal_from_probability(0.50)["signal"], HOLD)

    def test_dataset_split_is_no_overlap(self):
        x, y = build_labeled_dataset(_make_candles(150))
        x_train, x_test, y_train, y_test = train_test_split_no_overlap(x, y, 0.8)
        self.assertEqual(len(x_train) + len(x_test), len(x))
        self.assertEqual(len(y_train) + len(y_test), len(y))
        self.assertNotEqual(id(x_train), id(x_test))

    def test_system_fit_and_walk_forward(self):
        candles = _make_candles(260)
        system = BinaryOptionsTradingSystem()
        metrics = system.fit(candles)
        for key in ("train_accuracy", "test_accuracy", "roc_auc", "precision", "recall"):
            self.assertIn(key, metrics)
        wf = system.walk_forward_validation(candles, train_window=100, test_window=20)
        self.assertIn("average", wf)
        self.assertIn("period_accuracies", wf)


if __name__ == "__main__":
    unittest.main()
