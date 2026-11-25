import unittest
from long_term.predict_daily import ANALYSIS_RANGES, FORECAST_HORIZONS, MIN_TRAIN_POINTS, _fit_predict
from long_term.evaluation import horizon_weeks


class PredictionConfigurationTests(unittest.TestCase):
    def test_expected_range_and_horizon_counts(self):
        self.assertEqual(len(ANALYSIS_RANGES), 13)
        self.assertEqual(len(FORECAST_HORIZONS), 7)
        self.assertEqual(len(ANALYSIS_RANGES) * len(FORECAST_HORIZONS), 91)

    def test_range_and_horizon_values(self):
        self.assertEqual(dict(ANALYSIS_RANGES)["2_years_6_months"], 913)
        self.assertEqual(dict(FORECAST_HORIZONS), {
            "1_week": 7, "15_days": 15, "1_month": 30, "2_months": 60,
            "3_months": 90, "6_months": 180, "1_year": 365,
        })

    def test_weekly_projection_rounds_to_weekly_observation(self):
        prices = [100.0 * (1.01 ** i) for i in range(MIN_TRAIN_POINTS)]
        self.assertGreater(_fit_predict(prices, 15), prices[-1])
        self.assertEqual(horizon_weeks(15), 3)

    def test_insufficient_training_is_rejected(self):
        with self.assertRaises(ValueError):
            _fit_predict([100.0] * (MIN_TRAIN_POINTS - 1), 7)


if __name__ == "__main__":
    unittest.main()
