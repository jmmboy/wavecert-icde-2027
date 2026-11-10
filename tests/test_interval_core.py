"""Semantic tests including exhaustive possible-world and hidden-path checks."""
import sys
import unittest
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from interval_core import (chain_bounds, batch_chain_features, interval_logs_from_visits,
                           support_bounds, _feasible, chain_truth)
from sim_data import simulate_domain


def logs(intervals):
    return pd.DataFrame([{"person_id": "p", "event_type": e, "L": l, "R": r}
                         for e, (l, r) in intervals.items()])


class IntervalSemantics(unittest.TestCase):
    def test_open_left_closed_right(self):
        self.assertEqual(chain_bounds(logs({"A": (0, 1), "B": (1, 2)}), ("A", "B"))["lower"], 1)
        self.assertEqual(chain_bounds(logs({"A": (1, 2), "B": (0, 1)}), ("A", "B"))["upper"], 0)
        answer = chain_bounds(logs({"A": (0, 1), "B": (0, 1)}), ("A", "B"))
        self.assertEqual((answer["lower"], answer["upper"]), (0, 1))

    def test_pairwise_possible_is_not_three_chain_possible(self):
        x = logs({"A": (2, 3), "B": (0, 4), "C": (0, 1)})
        self.assertEqual(chain_bounds(x, ("A", "B"))["upper"], 1)
        self.assertEqual(chain_bounds(x, ("B", "C"))["upper"], 1)
        self.assertEqual(chain_bounds(x, ("A", "B", "C"))["upper"], 0)

    def test_global_constraints_and_span(self):
        x = logs({"A": (0, 1), "B": (1, 2), "C": (2, 3)})
        self.assertEqual(chain_bounds(x, ("A", "B", "C"), max_span=1)["upper"], 0)
        self.assertEqual(chain_bounds(x, ("A", "B", "C"), max_span=1.1)["upper"], 1)
        self.assertEqual(chain_bounds(x, ("A", "B", "C"), max_span=3)["lower"], 1)
        y = logs({"A": (0, 4), "B": (0, 4), "C": (0, 4)})
        constraints = (("A", "B", 0, True), ("B", "C", 0, True))
        self.assertEqual(chain_bounds(y, ("A", "C"), constraints=constraints)["lower"], 1)
        with self.assertRaises(ValueError):
            chain_bounds(y, ("A", "B"), constraints=constraints + (("C", "A", 0, True),))

    def test_duplicate_recurrence_rejected(self):
        x = logs({"A": (0, 1), "B": (1, 2)})
        x = pd.concat([x, x.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            chain_bounds(x, ("A", "B"))
        with self.assertRaises(ValueError):
            batch_chain_features(x, ["p"], [("A", "B")])

    def test_unknown_not_negative(self):
        x = logs({"A": (0, 1)})
        self.assertTrue(np.isnan(chain_bounds(x, ("A", "B"), history_complete=False)["upper"]))
        self.assertEqual(chain_bounds(x, ("A", "B"), history_complete=True)["upper"], 0)
        f = batch_chain_features(x, ["p", "q"], [("A", "B")], completeness={"p": False, "q": True})
        self.assertTrue(np.isnan(f.loc["p", "A>B__lower"]))
        self.assertEqual(f.loc["p", "A>B__unknown"], 1)
        self.assertEqual(f.loc["q", "A>B__upper"], 0)
        summary = support_bounds(f).iloc[0]
        self.assertEqual((summary.n_eligible, summary.n_unknown), (1, 1))

    def test_missing_wave_does_not_create_negative(self):
        v = pd.DataFrame({"person_id": ["p"] * 4, "time": [0, 1, 2, 3],
                          "observed": [True, False, True, True], "A": [0, np.nan, 1, 0],
                          "B": [1, np.nan, 1, 1]})
        ll, audit = interval_logs_from_visits(v, ["A", "B"], history=(0, 3))
        self.assertEqual(len(ll), 1)
        self.assertEqual((ll.iloc[0].L, ll.iloc[0].R), (0, 2))
        self.assertEqual(audit.loc[audit.event_type == "A", "reversals"].iloc[0], 1)
        self.assertTrue(audit.loc[audit.event_type == "B", "baseline_positive"].iloc[0])

    def test_exhaustive_rational_worlds(self):
        # All 216 integer-endpoint configurations. Quarter-grid completions
        # witness each order cell for three events in these unit-width boxes.
        intervals = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for box in product(intervals, repeat=3):
            rec = dict(zip("ABC", box))
            values = [np.arange(l + .25, r + .01, .25) for l, r in box]
            worlds = np.array(list(product(*values)))
            match = (worlds[:, 0] < worlds[:, 1]) & (worlds[:, 1] < worlds[:, 2])
            b = chain_bounds(logs(rec), tuple("ABC"))
            self.assertEqual((b["lower"], b["upper"]), (int(match.all()), int(match.any())), msg=str(rec))
            f = batch_chain_features(logs(rec), ["p"], [tuple("ABC")]).iloc[0]
            self.assertEqual((f["A>B>C__lower"], f["A>B>C__upper"]), (b["lower"], b["upper"]))

    def test_sampled_truth_inside_logical_bounds(self):
        rng = np.random.default_rng(23)
        for _ in range(150):
            l = rng.uniform(-5, 0, 3)
            r = l + rng.uniform(.01, 3, 3)
            x = logs(dict(zip("ABC", zip(l, r))))
            for span in [None, 0.0, .2, 1.5, 5.0]:
                b = chain_bounds(x, tuple("ABC"), max_span=span)
                t = rng.uniform(l, r, (80, 3))
                good = np.all(np.diff(t, axis=1) > 0, axis=1)
                if span is not None:
                    good &= t[:, -1] - t[:, 0] <= span
                self.assertTrue(np.all(good >= b["lower"]))
                self.assertTrue(np.all(good <= b["upper"]))


class ContinuousTimeSimulation(unittest.TestCase):
    def test_seed_reproduction_and_no_landmark_leak(self):
        a = simulate_domain(100, 18, "chain", 0)
        b = simulate_domain(100, 18, "chain", 0)
        pd.testing.assert_frame_equal(a["persons"], b["persons"])
        np.testing.assert_equal(a["truth_state3"], b["truth_state3"])
        self.assertTrue((a["interval_logs"].R <= 0).all())
        self.assertTrue(a["meta"]["synthetic"])
        # Exact times are genuinely continuous, not nominal wave numbers.
        self.assertGreater(np.std(a["exact_times"][:, 0] % 1), .1)

    def test_recovery_and_observed_endpoint_sets(self):
        data = simulate_domain(700, 338, "reversible", 2)
        transitions = [list(zip(p[:-1], p[1:])) for p in data["latent_paths"]]
        self.assertTrue(any(a[1] == 1 and b[1] == 0 for pairs in transitions for a, b in pairs))
        o = data["outcomes"]
        allowed = o[["state0_possible", "state1_possible", "state2_possible"]].to_numpy(bool)
        self.assertTrue(allowed[np.arange(len(o)), data["truth_state3"]].all())
        self.assertTrue(((o.observed_y_lower <= o.truth_yA) & (o.truth_yA <= o.observed_y_upper)).all())
        alive = ~o.death_by_horizon.astype(bool)
        self.assertTrue(o.loc[alive, "observed_state3"].isna().all())

    def test_hidden_illness_before_death_is_not_healthy_negative(self):
        data = simulate_domain(1200, 918, "hidden_death", 1)
        o = data["outcomes"]
        hidden = o.hidden_illness_before_death
        self.assertGreater(hidden.sum(), 0)
        self.assertTrue((o.loc[hidden, "observed_y_upper"] == 1).all())
        self.assertTrue((o.loc[hidden, "truth_state3"] == 2).all())
        self.assertTrue((o.loc[hidden, "truth_yA"] == 1).all())

    def test_recorded_log_bounds_cover_recorded_exact_events_without_error(self):
        d = simulate_domain(200, 42, {"name": "chain", "event_reversal_rate": 0, "report_error": 0}, 0)
        logs_ = d["interval_logs"]
        self.assertTrue(((logs_.truth_time > logs_.L) & (logs_.truth_time <= logs_.R)).all())
        # Target is the recorded-event subset, not all hidden true events.
        for _, personlogs in logs_.groupby("person_id"):
            if len(personlogs) >= 2:
                p = tuple(personlogs.event_type.iloc[:2])
                b = chain_bounds(personlogs, p)
                truth = int(personlogs.set_index("event_type").loc[p[0], "truth_time"] <
                            personlogs.set_index("event_type").loc[p[1], "truth_time"])
                self.assertLessEqual(b["lower"], truth)
                self.assertLessEqual(truth, b["upper"])


if __name__ == "__main__":
    unittest.main()
