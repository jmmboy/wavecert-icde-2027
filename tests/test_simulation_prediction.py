"""Checks that oracle data never enter the prediction design or fit labels."""
import sys
from pathlib import Path
import unittest
import copy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bootstrap_env
import numpy as np
from simulation_prediction import _observed_domain, _run_one, _train_chain_selector, _selected_design, METHODS
from sim_data import simulate_domain


class TestObservedPredictionDesign(unittest.TestCase):
    def test_latent_features_are_not_used(self):
        sim = simulate_domain(50, 4701, "chain", domain=0)
        ordinary = _observed_domain(sim, 77)
        altered = copy.deepcopy(sim)
        altered["exact_times"][:] = 9999
        altered["latent_paths"] = []
        altered["truth_chain"][:] = 1-altered["truth_chain"]
        altered["truth_state3"][:] = 2
        for c in ["latent_frailty", "level", "slope", "current_health"]:
            altered["persons"][c] = 1e8
        rederived = _observed_domain(altered, 77)
        for method in METHODS:
            np.testing.assert_allclose(ordinary["design"][method], rederived["design"][method], equal_nan=True)
        for key in ordinary["panel"]:
            np.testing.assert_array_equal(ordinary["panel"][key], rederived["panel"][key])

    def test_observed_panel_integrity_and_equivalence(self):
        sim = simulate_domain(60, 4702, "hidden_death", domain=1)
        data = _observed_domain(sim, 78)
        panel = data["panel"]
        self.assertTrue(np.all(panel["delta"] > 0))
        self.assertTrue(np.all(np.isin(panel["start"], [0, 1])))
        self.assertTrue(np.all(panel["end"][panel["exact_death"]] == 2))
        self.assertEqual(panel["exact_death"].sum(), np.isfinite(sim["outcomes"].death_time).sum())
        np.testing.assert_allclose(data["design"]["wave_order"], data["design"]["right_endpoint"], equal_nan=True)
        keep = _train_chain_selector([data, data])
        self.assertEqual(_selected_design(data, "partial_order", keep).shape[1],
                         data["bag_ncols"]+2*keep.sum()+3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
