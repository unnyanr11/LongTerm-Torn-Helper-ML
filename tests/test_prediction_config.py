import unittest

import numpy as np

from long_term.predict_daily import (
    ANALYSIS_RANGES,
    FORECAST_HORIZONS,
    MIN_TRAIN_POINTS,
    _fit_predict,
    _validate,
)


class PredictionConfigurationTests(unittest.TestCase):
    def test_expected_range_and_horizon_counts(self):
        self.assertEqual(len(ANALYSIS_RANGES), 13)
        self.assertEqual(len(FORECAST_HORIZONS), 7)
        self.assertEqual(len(ANALYSIS_RANGES) * len(FORECAST_HORIZONS), 91)

    def test_two_year_six_months_is_30_months(self):
        ranges = dict(ANALYSIS_RANGES)
        self.assertEqual(ranges["2_years_6_months"], 913)

    def test_horizons_match_requested_periods(self):
        horizons = dict(FORECAST_HORIZONS)
        self.assertEqual(horizons, {
            "1_week": 7,
            "15_days": 15,
            "1_month": 30,
            "2_months": 60,
            "3_months": 90,
            "6_months": 180,
            "1_year": 365,
        })

    def test_forecast_is_above_origin_for_positive_trend(self):
        prices = [100.0 * (1.01 ** i) for i in range(30)]
        self.assertGreater(_fit_predict(prices, 30), prices[-1])

    def test_sparse_window_is_not_marked_validated(self):
        prices = [100.0 + i for i in range(MIN_TRAIN_POINTS)]
        self.assertEqual(_validate(prices)["status"], "insufficient_validation")

    def test_stable_trend_validation_is_walk_forward(self):
        prices = [100.0 * (1.01 ** i) for i in range(40)]
        result = _validate(prices)
        self.assertEqual(result["status"], "validated")


if __name__ == "__main__":
    unittest.main()
