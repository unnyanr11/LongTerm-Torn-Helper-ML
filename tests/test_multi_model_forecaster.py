import unittest
import numpy as np

from long_term.ml_forecaster import (
    build_supervised,
    evaluate_ml_horizon,
    _features,
    MIN_FEATURE_HISTORY,
    _strategy_metrics,
    MIN_EXPECTED_NET_RETURN,
    ROUND_TRIP_COST,
)


class MultiModelForecastTests(unittest.TestCase):
    def test_long_term_defaults_are_point_two_percent_net(self):
        self.assertAlmostEqual(MIN_EXPECTED_NET_RETURN, 0.002)
        self.assertAlmostEqual(ROUND_TRIP_COST, 0.002)

    def test_strategy_threshold_is_configurable_and_costs_are_deducted(self):
        result = _strategy_metrics(
            np.log1p([0.006, 0.006]),
            np.log1p([0.006, 0.006]),
            round_trip_cost=0.003,
            minimum_net_return=0.002,
        )
        self.assertEqual(result['minimum_expected_net_return'], 0.002)
        self.assertAlmostEqual(result['minimum_predicted_gross_return'], 0.005)
        self.assertEqual(result['trades'], 2)
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


    def test_strategy_metrics_deduct_costs_and_track_drawdown(self):
        result = _strategy_metrics(
            np.log1p([0.02, -0.01, 0.03]),
            np.log1p([0.02, 0.02, 0.03]),
        )
        self.assertEqual(result["trades"], 3)
        self.assertEqual(result["wins"], 2)
        self.assertGreater(result["max_drawdown"], -0.02)
        self.assertGreater(result["total_return"], 0)
        self.assertTrue(result["eligible"])

    def test_strategy_metrics_rejects_unprofitable_or_too_few_trades(self):
        result = _strategy_metrics(
            np.log1p([-0.02, -0.01, -0.03]),
            np.log1p([0.02, 0.02, 0.03]),
        )
        self.assertFalse(result["eligible"])
        self.assertLessEqual(result["total_return"], 0)

    def test_long_horizon_without_enough_holdout_is_not_validated(self):
        prices = [100.0 * (1.002 ** i) for i in range(261)]
        result = evaluate_ml_horizon(prices, 365)
        self.assertIn(result["status"], {
            "insufficient_ml_history",
            "insufficient_holdout_folds",
            "insufficient_non_overlapping_validation_trades",
        })

    def test_forecast_signal_requires_validated_profitable_holdout(self):
        from long_term.predict_daily import _forecast_signal
        validated = {
            "status": "validated",
            "holdout_trading": {
                "eligible": True,
                "round_trip_cost": 0.002,
                "minimum_expected_net_return": 0.002,
            },
        }
        self.assertEqual(_forecast_signal(100.6, 100.0, validated), "buy")
        self.assertEqual(_forecast_signal(100.3, 100.0, validated), "wait")
        self.assertEqual(_forecast_signal(99.7, 100.0, validated), "wait")
        self.assertEqual(_forecast_signal(99.4, 100.0, validated), "sell")
        self.assertEqual(_forecast_signal(110.0, 100.0, {**validated, "status": "rejected"}), "wait")
        self.assertEqual(_forecast_signal(110.0, 100.0, {"status": "validated", "holdout_trading": {"eligible": False}}), "wait")

    def test_forecast_signal_rejects_invalid_prices(self):
        from long_term.predict_daily import _forecast_signal
        validation = {"status": "validated", "holdout_trading": {"eligible": True}}
        self.assertEqual(_forecast_signal(float("nan"), 100.0, validation), "wait")
        self.assertEqual(_forecast_signal(100.0, 0.0, validation), "wait")


if __name__ == "__main__":
    unittest.main()
