from datetime import datetime, timedelta, timezone
import unittest

from trading_system import (
    FEATURE_NAMES,
    BinaryOptionsTradingSystem,
    Candle,
    TradeLog,
    HOLD,
    PUT,
    CALL,
    build_labeled_dataset,
    classify_operating_status,
    confidence_calibration,
    compute_features,
    expected_roi_per_trade,
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

    def test_execution_pnl_and_roi(self):
        win = TradeLog(
            timestamp=datetime.now(timezone.utc),
            signal=CALL,
            entry_price=1.1,
            exit_price=1.11,
            confidence=0.7,
        )
        loss = TradeLog(
            timestamp=datetime.now(timezone.utc),
            signal=PUT,
            entry_price=1.1,
            exit_price=1.12,
            confidence=0.68,
        )
        self.assertEqual(win.outcome, "WIN")
        self.assertEqual(loss.outcome, "LOSS")
        self.assertEqual(round(win.pnl, 2), 0.85)
        self.assertEqual(round(loss.pnl, 2), -1.00)
        self.assertAlmostEqual(expected_roi_per_trade(0.61), 0.1285, places=6)

    def test_calibration_and_operating_status(self):
        y_true = [1, 1, 0, 0, 1]
        y_prob = [0.8, 0.7, 0.2, 0.4, 0.6]
        bins = confidence_calibration(y_true, y_prob, 0.1)
        self.assertTrue(len(bins) >= 1)
        self.assertIn("actual_win_rate", bins[0])
        self.assertEqual(
            classify_operating_status(0.63, 0.71, 0.30)["status"],
            "GREEN",
        )
        self.assertEqual(
            classify_operating_status(0.60, 0.64, 0.20)["status"],
            "YELLOW",
        )
        self.assertEqual(
            classify_operating_status(0.55, 0.54, 0.14)["status"],
            "RED",
        )


if __name__ == "__main__":
    unittest.main()
