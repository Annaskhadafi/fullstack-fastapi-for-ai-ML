"""Run: python -m unittest discover -s tests -p test_forecast_history.py"""
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from app.services import market_forecast_service as service


class ForecastHistoryTest(unittest.TestCase):
    def test_immutable_predictions_and_market_isolation(self):
        rates = np.array([(1000 + i * 60, 100.0 + i) for i in range(31)],
                         dtype=[("time", "i8"), ("close", "f8")])
        forecast = {"current_price": 100.0, "projected_price": 120.0,
                    "projection": [120.0] * 30, "updated_at": "2026-10-03"}
        original_connect = service.sqlite3.connect
        with tempfile.TemporaryDirectory() as directory:
            db = os.path.join(directory, "history.db")
            with patch.object(service.sqlite3, "connect", side_effect=lambda *args, **kwargs: original_connect(db, timeout=30)):
                service._record_and_evaluate("BTCUSD", "M1", rates[:1], forecast)
                service._record_and_evaluate("BTCUSD", "M1", rates[:1], {**forecast, "projected_price": 999.0})
                self.assertEqual(service.get_history("BTCUSD", "M1")["items"][0]["predicted_price"], 120.0)
                service._record_and_evaluate("EURUSD", "M1", rates, forecast)
                self.assertEqual(service.get_history("BTCUSD", "M1")["completed"], 0)
                service._record_and_evaluate("BTCUSD", "M1", rates[:30], forecast)
                self.assertEqual(service.get_history("BTCUSD", "M1")["completed"], 0)
                service._record_and_evaluate("BTCUSD", "M1", rates, forecast)
                history = service.get_history("BTCUSD", "M1")
                self.assertEqual(history["completed"], 1)
                self.assertEqual(history["direction_accuracy"], 100.0)
                self.assertAlmostEqual(history["average_error_percent"], 10 / 130 * 100, places=4)
                self.assertEqual(history["items"][-1]["actual_price"], 130.0)


if __name__ == "__main__":
    unittest.main()
