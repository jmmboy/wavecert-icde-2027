"""Additional frozen-main-independent informative historical observation test.

Only the intermediate HISTORICAL interview is thinned.  Attendance depends on
events already present at that interview and contemporaneous latent severity.
The comparator has the same expected attendance but state-independent thinning.
All exact events, health/death paths and prospective visits remain paired.
"""
import bootstrap_env  # noqa: F401
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import psutil
from sim_data import simulate_domain, EVENT_TYPES
from simulation_runner import metrics_for_domain, summarize

BASE = Path(__file__).resolve().parents[1]


def _rebuild_first_logs(data, visits):
    """Vectorized equivalent of interval_logs_from_visits for these three waves."""
    persons = data["persons"]
    ids = persons.person_id.to_numpy()
    historical = visits[visits.time <= 0].sort_values(["wave", "person_id"])
    # Reindex by simulation order explicitly, so lexical identifiers are safe.
    waves = [historical[historical.wave == w].set_index("person_id").reindex(ids) for w in range(3)]
    times = np.column_stack([v.time.to_numpy() for v in waves])
    rows = []
    for event in EVENT_TYPES:
        values = np.column_stack([v[event].to_numpy(float) for v in waves])
        negative = np.full(len(ids), np.nan)
        seen, baseline_positive, emitted = np.zeros((3, len(ids)), bool)
        for wave in range(3):
            value = values[:, wave]
            finite = np.isfinite(value)
            first = finite & ~seen
            baseline_positive |= first & (value == 1)
            match = (value == 1) & np.isfinite(negative) & ~baseline_positive & ~emitted
            if match.any():
                e = EVENT_TYPES.index(event)
                rows.append(pd.DataFrame({"person_id": ids[match], "cohort": data["meta"]["label"],
                                          "event_type": event, "L": negative[match], "R": times[match, wave],
                                          "prior_state": 0, "new_state": 1, "precision": "interview_interval",
                                          "observation_mask": "explicit_negative_to_positive",
                                          "truth_time": data["exact_times"][match, e]}))
                emitted |= match
            isnegative = value == 0
            negative[isnegative] = times[isnegative, wave]
            seen |= finite
    return pd.concat(rows, ignore_index=True) if rows else data["interval_logs"].iloc[:0].copy()


def simulate_history_pair(n, seed, domain=0):
    """Same latent participants under matched noninformative/informative history."""
    data = simulate_domain(n, seed, "chain", domain)
    rng = np.random.default_rng(seed + 100_000_000)
    mid = data["visits"][data["visits"].wave == 1].copy()
    current_events = data["exact_times"] <= mid.time.to_numpy()[:, None]
    # This process depends on state at the historical visit, never a future
    # outcome or a later event.  Latent severity is generative, not a predictor.
    severity = data["persons"].level.to_numpy() + mid.time.to_numpy() * data["persons"].slope.to_numpy()
    pinfo = 1 / (1 + np.exp(-(1.6 - .65 * current_events.sum(axis=1) - .5 * severity)))
    uniform = rng.random(n)
    missingreport = rng.random((n, 4)) < .015
    results = {}
    for mechanism, probability in [("matched_noninformative_history", np.full(n, pinfo.mean())),
                                   ("informative_history", pinfo)]:
        observed = uniform < probability
        visits = data["visits"].copy()
        ix = visits.wave == 1
        visits.loc[ix, "observed"] = observed
        values = current_events.astype(float)
        values[missingreport] = np.nan
        values[~observed] = np.nan
        for e, event in enumerate(EVENT_TYPES):
            visits.loc[ix, event] = values[:, e]
        visits.loc[ix, "state"] = np.where(observed, mid.true_state.to_numpy(float), np.nan)
        paired = dict(data)
        paired["visits"] = visits
        paired["interval_logs"] = _rebuild_first_logs(data, visits)
        paired["persons"] = data["persons"].copy()
        paired["persons"]["n_history_visits"] = 2 + observed.astype(int)
        paired["persons"]["history_missing"] = (~observed).astype(int)
        paired["persons"]["mean_history_interval"] = np.where(observed, 2.0, 4.0)
        paired["persons"]["max_history_interval"] = np.where(observed, data["persons"].max_history_interval, 4.0)
        paired["meta"] = dict(data["meta"], historical_observation_mechanism=mechanism,
                              expected_midattendance=float(probability.mean()),
                              actual_midattendance=float(observed.mean()))
        results[mechanism] = paired
    return results


def one(task):
    repeat, n = task
    started, rows, resources = time.perf_counter(), [], []
    for domain in range(3):
        seed = 4_000_000 + repeat * 10 + domain
        pair = simulate_history_pair(n, seed, domain)
        for mechanism, data in pair.items():
            rows.extend(metrics_for_domain(data, mechanism, repeat, "historical_observation_sensitivity", seed))
            resources.append({"repeat": repeat, "domain": domain, "seed": seed, "mechanism": mechanism,
                              "expected_midattendance": data["meta"]["expected_midattendance"],
                              "actual_midattendance": data["meta"]["actual_midattendance"],
                              "recorded_events": len(data["interval_logs"]), "n": n,
                              "rss_bytes": psutil.Process().memory_info().rss})
    return rows, resources, time.perf_counter() - started


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repetitions", type=int, default=50)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    out = BASE / "simulation" / "historical_observation_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    started, rows, resource_rows = time.perf_counter(), [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, (repeat, args.n)) for repeat in range(args.repetitions)]
        for i, future in enumerate(as_completed(futures), 1):
            rr, resources, elapsed = future.result()
            rows.extend(rr)
            resource_rows.extend(resources)
            if i % 10 == 0:
                print(json.dumps({"completed": i, "total": len(futures), "seconds": time.perf_counter() - started}), flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "replicate_long.csv", index=False)
    pd.DataFrame(resource_rows).to_csv(out / "resources.csv", index=False)
    summarize(raw).to_csv(out / "scenario_summary_mcse.csv", index=False)
    numeric = [c for c in raw.select_dtypes("number") if c not in ["repeat", "seed", "chain_length"]]
    pooled = raw.groupby(["stage", "scenario", "repeat", "domain", "chain_length", "method"])[numeric].mean().reset_index()
    pooled["pattern"] = "all_patterns_macro"
    summarize(pooled).to_csv(out / "scenario_macro_summary_mcse.csv", index=False)
    selected = ["ambiguity_among_recorded", "support_width", "precision", "recall", "f1", "recorded_complete_pattern_n"]
    difference_rows = []
    for method, g in pooled.groupby("method"):
        for metric in selected:
            r = g.groupby(["repeat", "scenario"])[metric].mean().unstack("scenario")
            diff = r.informative_history - r.matched_noninformative_history
            difference_rows.append({"method": method, "metric": metric, "difference": "informative-minus-matched",
                                    "mean": diff.mean(), "sd_replications": diff.std(ddof=1),
                                    "mcse": diff.std(ddof=1) / np.sqrt(len(diff)), "n_replications": len(diff)})
    pd.DataFrame(difference_rows).to_csv(out / "paired_differences.csv", index=False)
    manifest = {"synthetic_only": True, "repetitions": args.repetitions, "domains": 3, "n_per_domain": args.n,
                "elapsed_seconds": time.perf_counter() - started, "failures": 0,
                "seed_formula": "4000000 + repeat*10 + domain", "main_simulator_modified": False,
                "mechanism": "Intermediate historical attendance p=expit(1.6-.65*number_events_already_present-.5*contemporaneous_severity)",
                "comparator": "same random attendance U, same mean p; state-independent observation",
                "paired_latent_truth": True, "prospective_information_used": False,
                "scope": "Additional sensitivity only; original main informative_dropout/visits affect prospective ascertainment"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
