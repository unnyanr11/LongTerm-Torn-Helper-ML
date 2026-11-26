import unittest
import numpy as np

from long_term.ml_forecaster import (
    build_supervised,
    evaluate_ml_horizon,
    _features,
    MIN_FEATURE_HISTORY,
    MIN_EXPECTED_NET_RETURN,
    ROUND_TRIP_COST,
)


class MultiModelForecastTests(unittest.TestCase):
    def test_long_term_defaults_are_point_two_percent_net(self):
        self.assertAlmostEqual(MIN_EXPECTED_NET_RETURN, 0.002)
        self.assertAlmostEqual(ROUND_TRIP_COST, 0.002)

    def test_live_signal_cost_defaults_remain_configured_separately(self):
        self.assertAlmostEqual(MIN_EXPECTED_NET_RETURN, 0.002)
        self.assertAlmostEqual(ROUND_TRIP_COST, 0.002)
    def test_feature_vector_uses_only_history_through_origin(self):
        prices = [100.0 * (1.002 ** i) for i in range(80)]
        original = _features(prices, 60)
        changed_future = prices[:61] + [999999.0] * 19
        changed = _features(changed_future, 60)
        np.testing.assert_allclose(original, changed)

    def test_supervised_targets_are_future_log_returns(self):
        prices = [100.0 * (1.01 ** i) for i in range(100)]
        X, y, origins, targets = build_supervised(prices, 15)
        self.assertGreater(len(y), 0)
        self.assertTrue(np.all(targets > origins))
        self.assertTrue(np.allclose(y, np.log(1.01) * 3))

    def test_ml_forecaster_selects_and_tests_models_chronologically(self):
        prices = [100.0 * (1.01 ** i) for i in range(261)]
        result = evaluate_ml_horizon(prices, 7)
        self.assertEqual(result["status"], "validated")
        self.assertIn(result["selected_model"], {
            "ridge", "random_forest", "extra_trees", "hist_gradient_boosting"
        })
        self.assertGreaterEqual(result["holdout_folds"], 3)
        self.assertGreaterEqual(result["holdout"]["observations"], 3)



    def test_long_horizon_without_enough_holdout_is_not_validated(self):
        prices = [100.0 * (1.002 ** i) for i in range(261)]
        result = evaluate_ml_horizon(prices, 365)
        self.assertIn(result["status"], {
            "insufficient_ml_history",
            "insufficient_holdout_folds",
        })

    def test_forecast_signal_uses_accuracy_validation_and_separate_live_cost_gate(self):
        from long_term.predict_daily import _forecast_signal
        validated = {
            "status": "validated",

        }
        self.assertEqual(_forecast_signal(100.6, 100.0, validated), "buy")
        self.assertEqual(_forecast_signal(100.3, 100.0, validated), "wait")
        self.assertEqual(_forecast_signal(99.7, 100.0, validated), "wait")
        self.assertEqual(_forecast_signal(99.4, 100.0, validated), "sell")
        self.assertEqual(_forecast_signal(110.0, 100.0, {**validated, "status": "rejected"}), "wait")

    def test_forecast_signal_rejects_invalid_prices(self):
        from long_term.predict_daily import _forecast_signal
        validation = {"status": "validated", "holdout_trading": {"eligible": True}}
        self.assertEqual(_forecast_signal(float("nan"), 100.0, validation), "wait")
        self.assertEqual(_forecast_signal(100.0, 0.0, validation), "wait")


if __name__ == "__main__":
    unittest.main()
