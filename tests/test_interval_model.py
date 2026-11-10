"""Deterministic mathematical checks plus a known-generator simulation check.

These tests do not represent real-cohort experiments.
"""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.linalg import expm
from interval_model import (ReversiblePanelCTMC, transition_matrix, _transient_exp,
                            _as_panel, _row_likelihood, oracle_state_metrics,
                            partial_state_brier_bounds)


class TestPanelCTMC(unittest.TestCase):
    def test_expm_and_probability_conservation(self):
        rng = np.random.default_rng(120)
        rates = np.exp(rng.uniform(-5, 1.5, size=(100, 4)))
        t = rng.uniform(1e-6, 12, size=100)
        P = transition_matrix(rates, t)
        for j, (a, b, c, d) in enumerate(rates):
            Q = np.array([[-a-b, a, b], [c, -c-d, d], [0, 0, 0]])
            np.testing.assert_allclose(P[j], expm(Q*t[j]), atol=2e-13, rtol=2e-12)
        np.testing.assert_allclose(P.sum(axis=-1), 1, atol=1e-13)

    def test_analytic_rate_gradient(self):
        rng = np.random.default_rng(121)
        rates = np.exp(rng.uniform(-4, 0, size=(30, 4)))
        t = rng.uniform(.001, 8, size=30)
        _, D = _transient_exp(rates, t, True)
        for k in range(4):
            eps = 1e-6 * rates[:, k]
            upper, lower = rates.copy(), rates.copy()
            upper[:, k] += eps
            lower[:, k] -= eps
            numerical = (_transient_exp(upper, t)-_transient_exp(lower, t))/(2*eps[:, None, None])
            np.testing.assert_allclose(D[..., k], numerical, rtol=8e-5, atol=3e-8)

    def test_exact_death_integrates_hidden_paths(self):
        rates = np.array([[.4, .01, .15, .6]])
        t = 4.
        P = transition_matrix(rates, t)[0]
        obs = {"person": [0], "delta": [t], "start": [0], "end": [2]}
        panel = _as_panel(obs, 1)
        v = _row_likelihood(rates, panel)[0]
        direct_only = rates[0, 1]/rates[0, :2].sum()*(1-np.exp(-rates[0, :2].sum()*t))
        self.assertGreater(v, direct_only + .2)
        self.assertAlmostEqual(v, P[0, 2], places=14)
        obs["exact_death"] = [True]
        panel = _as_panel(obs, 1)
        value, grad = _row_likelihood(rates, panel, True)
        self.assertAlmostEqual(value[0], P[0, 0]*rates[0, 1]+P[0, 1]*rates[0, 3], places=14)
        for k in range(4):
            eps = rates[0, k]*1e-5
            upper, lower = rates.copy(), rates.copy()
            upper[0, k] += eps
            lower[0, k] -= eps
            num = (_row_likelihood(upper, panel)-_row_likelihood(lower, panel))/(2*eps)
            np.testing.assert_allclose(grad[:, k], num, rtol=1e-6, atol=1e-8)

    def test_known_generator_recovery(self):
        rng = np.random.default_rng(122)
        truth = np.array([.16, .035, .23, .075])
        n = 20000
        X = np.empty((n, 0))
        current = rng.binomial(1, .4, size=n)
        obs = {k: [] for k in ["person", "delta", "start", "end"]}
        for _ in range(4):
            alive = np.flatnonzero(current < 2)
            delta = rng.uniform(.7, 2.6, size=len(alive))
            P = transition_matrix(np.tile(truth, (len(alive), 1)), delta)
            prob = P[np.arange(len(alive)), current[alive]]
            nxt = (rng.uniform(size=len(alive))[:, None] > prob.cumsum(axis=1)).sum(axis=1)
            obs["person"].extend(alive)
            obs["delta"].extend(delta)
            obs["start"].extend(current[alive])
            obs["end"].extend(nxt)
            current[alive] = nxt
        obs = {k: np.asarray(v) for k, v in obs.items()}
        model = ReversiblePanelCTMC(penalty=0, max_iter=100).fit(X, obs)
        estimated = model.predict_rates(X[:1])[0]
        np.testing.assert_allclose(estimated, truth, rtol=.055, atol=.003)
        self.assertTrue(model.converged_)
        print("known-generator rates:", estimated.tolist(), "truth:", truth.tolist(),
              "fit_seconds:", model.fit_seconds_)

    def test_scoring_bounds_do_not_impute_truth(self):
        p = np.array([[.4, .4, .2], [.1, .6, .3], [.8, .1, .1]])
        y = np.array([1, 2, 0])
        oracle = oracle_state_metrics(p, y)
        mask = np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=bool)
        bound = partial_state_brier_bounds(p, mask)
        self.assertLessEqual(bound["brier_lower"], oracle["multiclass_brier"])
        self.assertGreaterEqual(bound["brier_upper"], oracle["multiclass_brier"])
        exact = partial_state_brier_bounds(p, np.eye(3)[y])
        self.assertAlmostEqual(exact["brier_lower"], oracle["multiclass_brier"])
        self.assertAlmostEqual(exact["brier_upper"], oracle["multiclass_brier"])

    def test_cif_differs_from_reversible_occupation(self):
        model = ReversiblePanelCTMC()
        model.median_ = model.mean_ = np.empty(0)
        model.scale_ = np.empty(0)
        model.params_ = np.log(np.array([.2, .03, .5, .06]))[:, None]
        X = np.empty((1, 0))
        cif = model.predict_cif_first(X)[0]
        state1 = model.predict_state(X)[0, 1]
        self.assertGreater(cif, state1)

    def test_fixed_death_zero_mode(self):
        rng = np.random.default_rng(123)
        n = 6000
        rates = np.array([.18, 0., .3, 0.])
        initial = rng.binomial(1, .5, size=n)
        dt = rng.uniform(.7, 4., size=n)
        P = transition_matrix(np.tile(rates, (n, 1)), dt)
        nxt = (rng.random(n) > P[np.arange(n), initial, 0]).astype(int)
        obs = {"person": np.arange(n), "delta": dt, "start": initial, "end": nxt}
        model = ReversiblePanelCTMC(penalty=0, include_death=False).fit(np.empty((n, 0)), obs)
        est = model.predict_rates(np.empty((1, 0)))[0]
        np.testing.assert_allclose(est[[0, 2]], rates[[0, 2]], rtol=.09)
        np.testing.assert_array_equal(est[[1, 3]], 0.)
        np.testing.assert_array_equal(model.predict_state(np.empty((1, 0)))[:, 2], 0.)
        obs["end"][0] = 2
        with self.assertRaises(ValueError):
            model.fit(np.empty((n, 0)), obs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
