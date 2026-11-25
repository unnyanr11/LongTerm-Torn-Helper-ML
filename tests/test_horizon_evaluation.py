import unittest
from long_term.evaluation import evaluate_horizon, horizon_weeks, fit_log_linear


class HorizonEvaluationTests(unittest.TestCase):
    def test_day_horizons_map_to_weekly_observations(self):
        self.assertEqual(horizon_weeks(7), 1)
        self.assertEqual(horizon_weeks(15), 3)
        self.assertEqual(horizon_weeks(30), 5)
        self.assertEqual(horizon_weeks(365), 53)

    def test_requires_six_months_training(self):
        with self.assertRaises(ValueError):
            fit_log_linear([100.0] * 25, 1)

    def test_perfect_exponential_trend_beats_last_close(self):
        prices = [100.0 * (1.01 ** i) for i in range(261)]
        result = evaluate_horizon(prices, 30)
        self.assertEqual(result["status"], "validated")
        self.assertGreaterEqual(result["folds"], 3)
        self.assertGreater(result["mae_improvement_vs_last_close"], 0.05)

    def test_insufficient_history_is_not_promoted(self):
        result = evaluate_horizon([100.0 + i for i in range(35)], 365)
        self.assertEqual(result["status"], "insufficient_backtest")


if __name__ == "__main__":
    unittest.main()
