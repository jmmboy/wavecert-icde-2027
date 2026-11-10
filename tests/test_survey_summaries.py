"""Regression checks for the aggregate survey statistics used in Section IV."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from survey_summaries import observation_ablations, observed_window_calibration

import numpy as np
import pandas as pd


class SurveySummaryTests(unittest.TestCase):
    def test_calibration_resamples_whole_households_and_keeps_sparse_bins(self):
        # One household contributes all five cases in the 0.10 bin.
        rows = [("A", "00", 1, 0.10)] * 5
        rows.extend(("A", f"{i:02d}", 0, 0.10) for i in range(1, 10))
        rows.extend([("A", "01", 1, 1.0), ("A", "02", 0, 0.05)])
        frame = pd.DataFrame(rows, columns=["cohort", "household_id", "y", "PartialOrder"])
        result = observed_window_calibration(frame).set_index("bin")
        self.assertEqual(result.index.tolist(), [1, 2, 7])
        self.assertEqual(result.loc[2, "households"], 10)
        self.assertAlmostEqual(result.loc[2, "observed_fraction"], 5 / 14)
        self.assertEqual(result.loc[2, "ci_low"], 0.0)
        self.assertAlmostEqual(result.loc[2, "ci_high"], 15 / 22)
        self.assertEqual(result.loc[2, "bootstrap_valid_repetitions"], 1000)
        self.assertTrue(result.loc[[1, 7], ["ci_low", "ci_high"]].isna().all().all())
        self.assertEqual(result.loc[7, "mean_prediction"], 1.0)

    def test_ablation_summary_uses_equal_cohort_weights_and_sample_sd(self):
        rows = []
        for experiment in ["no_observation_covariates", "delete_intermediate_history_visits_rebuild"]:
            for cohort, delta, n in [("CHARLS", -0.01, 20), ("ELSA", 0.02, 100), ("HRS", 0.05, 150)]:
                rows.append((experiment, cohort, "completed", delta, n))
        frame = pd.DataFrame(rows, columns=["experiment", "cohort", "status", "brier_change", "test_n"])
        result = observation_ablations(frame)
        np.testing.assert_allclose(result["brier_change_mean"], 0.02)
        np.testing.assert_allclose(result["brier_change_cohort_sd"], 0.03)
        with self.assertRaises(ValueError):
            observation_ablations(frame.iloc[1:])


if __name__ == "__main__":
    unittest.main()
