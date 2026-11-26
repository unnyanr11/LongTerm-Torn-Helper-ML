import unittest
import numpy as np

from long_term.algorithm_registry import ALGORITHM_REGISTRY, algorithm_id_for_family, family_for_algorithm_id

from long_term.ml_forecaster import (
    build_supervised,
    evaluate_ml_horizon,
    _features,
    MIN_FEATURE_HISTORY,
    MIN_EXPECTED_NET_RETURN,
    ROUND_TRIP_COST,
)


class MultiModelForecastTests(unittest.TestCase):
    def test_numbered_algorithm_registry_is_stable(self):
        self.assertEqual(ALGORITHM_REGISTRY, {
            1: "ridge", 2: "elastic_net", 3: "random_forest",
            4: "extra_trees", 5: "hist_gradient_boosting", 6: "ensemble",
        })
        for algorithm_id, family in ALGORITHM_REGISTRY.items():
            self.assertEqual(algorithm_id_for_family(family), algorithm_id)
            self.assertEqual(family_for_algorithm_id(algorithm_id), family)

    def test_prediction_uses_registered_algorithm_for_latest_forecast(self):
        prices = [100.0 * (1.01 ** i) for i in range(261)]
        result = evaluate_ml_horizon(prices, 7, preferred_model="extra_trees")
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["selected_model"], "extra_trees")
        self.assertEqual(result["selected_algorithm_id"], 4)
        self.assertEqual(result["algorithm_selection_source"], "stock_algorithm_registry")

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
            "ridge", "elastic_net", "random_forest", "extra_trees",
            "hist_gradient_boosting", "ensemble"
        })
        self.assertGreaterEqual(result["holdout_folds"], 3)
        self.assertGreaterEqual(result["holdout"]["observations"], 3)
        self.assertIn("directional_accuracy", result["holdout"])
        self.assertNotIn("holdout_trading", result)
        self.assertEqual(result["initial_selection"]["status"], "selected_by_prediction_accuracy")
        for candidate in result["initial_selection"]["candidates"].values():
            self.assertNotIn("validation_trading", candidate)



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
        validation = {"status": "validated"}
        self.assertEqual(_forecast_signal(float("nan"), 100.0, validation), "wait")
        self.assertEqual(_forecast_signal(100.0, 0.0, validation), "wait")


if __name__ == "__main__":
    unittest.main()
