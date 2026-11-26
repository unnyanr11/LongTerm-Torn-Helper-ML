import unittest
import numpy as np

from long_term.portfolio import optimize_portfolio, _project_capped_simplex


class PortfolioOptimizerTests(unittest.TestCase):
    def test_weights_are_long_only_capped_and_fully_invested_when_feasible(self):
        result = optimize_portfolio(
            [0.03, 0.02, 0.01, 0.025, 0.015, 0.018, 0.012, 0.022],
            np.eye(8) * 0.01,
            max_weight=0.20,
        )
        weights = np.asarray(result["weights"])
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 0.20 + 1e-8))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=7)
        self.assertAlmostEqual(result["cash_weight"], 0.0, places=7)

    def test_optimizer_prefers_higher_return_when_risk_is_equal(self):
        result = optimize_portfolio(
            [0.04, 0.01, 0.02, 0.015, 0.012, 0.018, 0.011],
            np.eye(7) * 0.01,
            max_weight=0.25,
        )
        weights = result["weights"]
        self.assertGreater(weights[0], weights[1])

    def test_infeasible_full_investment_leaves_cash_instead_of_failing(self):
        result = optimize_portfolio(
            [0.03, 0.02, 0.01],
            np.eye(3) * 0.01,
            max_weight=0.20,
        )
        self.assertAlmostEqual(result["invested_weight"], 0.60, places=7)
        self.assertAlmostEqual(result["cash_weight"], 0.40, places=7)

    def test_rejects_bad_covariance_shape_and_non_finite_inputs(self):
        with self.assertRaises(ValueError):
            optimize_portfolio([0.01, 0.02], np.eye(3))
        with self.assertRaises(ValueError):
            optimize_portfolio([0.01, float("nan")], np.eye(2))

    def test_capped_simplex_projection(self):
        weights = _project_capped_simplex([10, 2, 1, 0, -1], total=1, cap=0.4)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=7)
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 0.4 + 1e-8))


if __name__ == "__main__":
    unittest.main()
