import unittest
import numpy as np

from long_term.calibration import fit_calibration, apply_calibration


class ForecastCalibrationTests(unittest.TestCase):
    def test_requires_minimum_matured_samples(self):
        result = fit_calibration([0.01] * 10, [0.012] * 10)
        self.assertEqual(result["status"], "insufficient_samples")

    def test_accepts_calibration_that_improves_later_holdout(self):
        predicted = np.linspace(-0.08, 0.08, 40)
        noise = np.sin(np.arange(40)) * 0.0005
        actual = 1.12 * predicted + 0.003 + noise
        result = fit_calibration(predicted, actual)
        self.assertEqual(result["status"], "accepted")
        self.assertGreaterEqual(result["holdout_mae_improvement"], 0.02)
        self.assertGreaterEqual(result["holdout_samples"], 6)

    def test_rejects_unhelpful_calibration(self):
        predicted = np.linspace(-0.05, 0.05, 40)
        actual = predicted + np.sin(np.arange(40) * 2.3) * 0.04
        result = fit_calibration(predicted, actual)
        self.assertNotEqual(result["status"], "accepted")

    def test_apply_calibration_adjusts_log_return(self):
        calibration = {"scale": 1.0, "bias": 0.01}
        adjusted = apply_calibration(110.0, 100.0, calibration)
        self.assertAlmostEqual(adjusted, 110.0 * np.exp(0.01), places=6)


if __name__ == "__main__":
    unittest.main()
