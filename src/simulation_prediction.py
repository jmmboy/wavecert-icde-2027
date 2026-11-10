"""Frozen synthetic-domain prediction benchmark, independent oracle scoring.

All fitting uses observed interviews plus completely observed simulated death
times. Latent states at year 3 are used *only* to score frozen predictions.
Three synthetic domains are not real cohorts. Run pilot before main:
  python simulation_prediction.py --phase pilot --reps 20 --workers 3
  python simulation_prediction.py --phase main --reps 200 --workers 3
"""
from __future__ import annotations
import bootstrap_env  # paths and one-thread environment before numpy/scipy
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations, combinations
import json
import hashlib
import os
import platform
from pathlib import Path
import threading
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import scipy
from scipy.stats import norm, t as student_t
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from sim_data import simulate_domain, EVENT_TYPES
from interval_core import batch_chain_features
from interval_model import ReversiblePanelCTMC, oracle_state_metrics, partial_state_brier_bounds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "simulation_prediction"
SCENARIOS = ("null", "chain", "overlap", "irregular", "informative_dropout",
             "informative_visits", "hidden_impairment_death", "recovery_shift")
METHODS = ("strong_history", "unordered", "midpoint", "right_endpoint", "wave_order",
           "uniform_sampling", "partial_order", "observation_only")
PATTERNS = tuple(permutations(EVENT_TYPES, 2))


class _MemorySampler:
    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop.wait(.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def reset(self):
        self.peak = self.process.memory_info().rss

    def read_mb(self):
        self.peak = max(self.peak, self.process.memory_info().rss)
        return self.peak / 2**20

    def close(self):
        self.stop.set()
        self.thread.join()


def _scenario(name):
    if name == "hidden_impairment_death":
        return "hidden_death"
    if name == "recovery_shift":
        return {"name": "custom", "recovery_rate": .8,
                "event_reversal_rate": .35, "domain_shift": True}
    return name


def _observed_domain(sim, sampling_seed):
    """Construct features and likelihood without access to latent truth paths."""
    p = sim["persons"].copy()
    ids = p.person_id.tolist()
    ix = {person: j for j, person in enumerate(ids)}
    visits = sim["visits"]
    history = visits[(visits.time <= 0) & visits.observed].copy().sort_values(["person_id", "time"])
    base = p[["age", "sex", "education"]].to_numpy(float)
    history["time_level"] = history.time*history.level
    history["time_square"] = history.time**2
    history["phone"] = (history["mode"] == "phone").astype(float)
    grouped = history.groupby("person_id", sort=False)
    history["gap"] = grouped.time.diff()
    sums = grouped[["time", "level", "time_level", "time_square"]].sum().reindex(ids)
    counts = grouped.size().reindex(ids).to_numpy(float)
    last = history.drop_duplicates("person_id", keep="last").set_index("person_id").reindex(ids)
    first = history.drop_duplicates("person_id", keep="first").set_index("person_id").reindex(ids)
    numer = sums.time_level.to_numpy()-sums.time.to_numpy()*sums.level.to_numpy()/counts
    denom = sums.time_square.to_numpy()-sums.time.to_numpy()**2/counts
    slope = np.divide(numer, denom, out=np.full(len(p), np.nan), where=denom > 0)
    means = grouped[["state", "proxy", "phone", "gap"]].mean().reindex(ids)
    maxgaps = grouped.gap.max().reindex(ids).fillna(4).to_numpy(float)
    hist = np.column_stack((last.level, slope, last.level-first.level, means.state, counts))
    obsfeat = np.column_stack((counts, means.gap.fillna(4), maxgaps, 3-counts,
                               last.proxy, last.phone, means.proxy, means.phone))
    start = last.state.to_numpy(int)
    current = last[list(EVENT_TYPES)].to_numpy(float)
    prev = first[list(EVENT_TYPES)].to_numpy(float)
    complete_mat = (grouped[list(EVENT_TYPES)].count().reindex(ids).to_numpy() == 3) & (counts[:, None] == 3)
    completeness = {person: dict(zip(EVENT_TYPES, complete_mat[j])) for j, person in enumerate(ids)}
    logs = sim["interval_logs"]
    lower_upper = batch_chain_features(logs, ids, PATTERNS, completeness=completeness)
    left = logs.pivot(index="person_id", columns="event_type", values="L").reindex(index=ids, columns=EVENT_TYPES).to_numpy(float)
    right = logs.pivot(index="person_id", columns="event_type", values="R").reindex(index=ids, columns=EVENT_TYPES).to_numpy(float)
    recorded = np.isfinite(left).astype(float)
    # An absent record in an incomplete event history remains explicitly unknown.
    unknown_event = (~np.isfinite(left)) & ~complete_mat
    bag = recorded.copy()
    bag[unknown_event] = np.nan
    unordered_pairs = np.column_stack([bag[:, a]*bag[:, b] for a, b in combinations(range(4), 2)])
    generic = np.column_stack((base, hist, current, prev, start, obsfeat,
                               np.isnan(current).astype(float), unknown_event.astype(float)))
    strong_unordered = np.column_stack((generic, bag, unordered_pairs))
    mid, endpoint, wave, uniform, partial = [], [], [], [], []
    midpoint = .5*(left+right)
    rng = np.random.default_rng(sampling_seed)
    random_time = left[None] + rng.random((64, len(p), 4))*(right-left)[None]
    for pattern in PATTERNS:
        a, b = [EVENT_TYPES.index(e) for e in pattern]
        name = ">".join(pattern)
        unknown = lower_upper[name+"__unknown"].to_numpy(bool)
        for container, value in [(mid, (midpoint[:, a] < midpoint[:, b]).astype(float)),
                                 (endpoint, (right[:, a] < right[:, b]).astype(float)),
                                 (wave, (right[:, a] < right[:, b]).astype(float)),
                                 (uniform, (random_time[:, :, a] < random_time[:, :, b]).mean(axis=0))]:
            value[unknown] = np.nan
            container.append(value)
        partial.extend([lower_upper[name+"__lower"].to_numpy(float),
                        lower_upper[name+"__ambiguous"].to_numpy(float)])
    # Wave order and right-endpoint order coincide under these subject-level
    # schedules (ties are rejected in both). Report the equivalence explicitly.
    reps = {"midpoint": np.column_stack(mid), "right_endpoint": np.column_stack(endpoint),
            "wave_order": np.column_stack(wave), "uniform_sampling": np.column_stack(uniform),
            "partial_order": np.column_stack(partial)}
    widths = right-left
    count = np.isfinite(widths).sum(axis=1)
    meanwidth = np.divide(np.nansum(widths, axis=1), count, out=np.zeros(len(p)), where=count > 0)
    width_features = np.column_stack((meanwidth, count, unknown_event.mean(axis=1)))
    # All augmented representations share the strong unordered covariates.
    design = {"strong_history": generic, "unordered": strong_unordered,
              "observation_only": obsfeat}
    for method, features in reps.items():
        design[method] = np.column_stack((strong_unordered, features, width_features))
    # Only recorded health observations determine each living transition.
    future = visits[(visits.time > 0) & visits.observed & visits.state.notna()].copy().sort_values(["person_id", "time"])
    fj = future.person_id.map(ix).to_numpy(int)
    fg = future.groupby("person_id", sort=False)
    prior_time = fg.time.shift().fillna(0).to_numpy(float)
    prior_state = fg.state.shift().to_numpy(float)
    prior_state = np.where(np.isfinite(prior_state), prior_state, start[fj]).astype(int)
    last_future = future.drop_duplicates("person_id", keep="last").set_index("person_id").reindex(ids)
    finaltime = last_future.time.fillna(0).to_numpy(float)
    finalstate = last_future.state.to_numpy(float)
    finalstate = np.where(np.isfinite(finalstate), finalstate, start).astype(int)
    death_t = sim["outcomes"].set_index("person_id").reindex(ids).death_time.to_numpy(float)
    deceased = np.isfinite(death_t) & (death_t > finaltime)
    panel = {"person": np.r_[fj, np.flatnonzero(deceased)],
             "delta": np.r_[future.time.to_numpy()-prior_time, death_t[deceased]-finaltime[deceased]],
             "start": np.r_[prior_state, finalstate[deceased]],
             "end": np.r_[future.state.to_numpy(int), np.full(deceased.sum(), 2)],
             "exact_death": np.r_[np.zeros(len(future), bool), np.ones(deceased.sum(), bool)]}
    return {"ids": ids, "design": design, "panel": panel, "start": start,
            "chain_bounds": lower_upper, "bag_ncols": strong_unordered.shape[1],
            "truth": sim["truth_state3"],
            "mask": sim["outcomes"][["state0_possible", "state1_possible", "state2_possible"]].to_numpy(bool),
            "hidden_death": float(sim["outcomes"].hidden_illness_before_death.mean()),
            "n_observed": len(np.unique(panel["person"]))}


def _join_panel(domains):
    result = {key: [] for key in domains[0]["panel"]}
    offset = 0
    for data in domains:
        for key, values in data["panel"].items():
            result[key].append(values+offset if key == "person" else values)
        offset += len(data["ids"])
    return {key: np.concatenate(value) for key, value in result.items()}


def _train_chain_selector(train):
    keep = []
    for pattern in PATTERNS:
        col = ">".join(pattern)+"__upper"
        support = [float(data["chain_bounds"][col].mean()) for data in train]
        keep.append(all(np.isfinite(s) and s >= .02 for s in support))
    return np.asarray(keep, dtype=bool)


def _selected_design(data, method, keep):
    X = data["design"][method]
    if method in ("strong_history", "unordered", "observation_only"):
        return X
    nbase = data["bag_ncols"]
    pattern_keep = np.repeat(keep, 2) if method == "partial_order" else keep
    return np.column_stack((X[:, :nbase], X[:, nbase:-3][:, pattern_keep], X[:, -3:]))


def _run_one(job):
    scenario, rep, n, phase = job
    threadpool_limits(1)
    start_time = time.perf_counter()
    seedbase = (21000000 if phase == "pilot" else 31000000) + SCENARIOS.index(scenario)*100000 + rep*10
    memory = _MemorySampler()
    output = []
    try:
        domains = [_observed_domain(simulate_domain(n, seedbase+d, _scenario(scenario), domain=d), seedbase+d+777)
                   for d in range(3)]
        heldout = rep % 3
        test = domains[heldout]
        train = [data for d, data in enumerate(domains) if d != heldout]
        panel = _join_panel(train)
        keep = _train_chain_selector(train)
        for method in METHODS:
            memory.reset()
            cpu0 = sum(psutil.Process().cpu_times()[:2])
            tick = time.perf_counter()
            row = {"phase": phase, "scenario": scenario, "rep": rep, "heldout_domain": heldout,
                   "seed": seedbase, "n_per_domain": n, "method": method, "penalty": .03,
                   "selected_pattern_count": int(keep.sum()), "status": "ok"}
            try:
                train_X = np.vstack([_selected_design(data, method, keep) for data in train])
                test_X = _selected_design(test, method, keep)
                model = ReversiblePanelCTMC(penalty=.03, max_iter=130, tol=1e-7).fit(train_X, panel)
                predtick = time.perf_counter()
                predicted = model.predict_state(test_X, t=3., start_state=test["start"])
                inference = time.perf_counter()-predtick
                metrics = oracle_state_metrics(predicted, test["truth"])
                bound = partial_state_brier_bounds(predicted, test["mask"])
                auc = []
                ece = []
                calibration_bins = {}
                for state in range(3):
                    binary = (test["truth"] == state).astype(int)
                    auc.append(float(roc_auc_score(binary, predicted[:, state])) if len(np.unique(binary)) == 2 else np.nan)
                    bins = np.minimum((predicted[:, state]*10).astype(int), 9)
                    count = np.bincount(bins, minlength=10)
                    risk_sum = np.bincount(bins, weights=predicted[:, state], minlength=10)
                    outcome_sum = np.bincount(bins, weights=binary, minlength=10)
                    ece.append(float(np.abs(risk_sum-outcome_sum).sum()/len(binary)))
                    calibration_bins[str(state)] = {"n": count.tolist(), "risk_sum": risk_sum.tolist(),
                                                     "outcome_sum": outcome_sum.tolist()}
                row.update({"oracle_brier": metrics["multiclass_brier"], "oracle_log_loss": metrics["log_loss"],
                            "auc_macro": float(np.nanmean(auc)), "auc_impairment": auc[1], "auc_death": auc[2],
                            "ece_macro_10_fixed_bins": float(np.mean(ece)),
                            "calibration_bins": calibration_bins,
                            "calibration_error_impairment": float(predicted[:, 1].mean()-(test["truth"] == 1).mean()),
                            "calibration_error_death": float(predicted[:, 2].mean()-(test["truth"] == 2).mean()),
                            "brier_identified_lower": bound["brier_lower"], "brier_identified_upper": bound["brier_upper"],
                            "identified_state_fraction": bound["exact_horizon_state_fraction"],
                            "observed_panel_nll": model.observed_nll(test_X, test["panel"]),
                            "fit_seconds": model.fit_seconds_, "inference_ms_per_1000": inference*1e6/len(test_X),
                            "converged": model.converged_, "iterations": model.n_iter_,
                            "gradient_max": model.gradient_max_, "rate_clipping_fraction": model.rate_clipping_fraction_,
                            "feature_count": test_X.shape[1], "hidden_impairment_death_fraction": test["hidden_death"]})
                row["oracle_within_observed_brier_bounds"] = bool(bound["brier_lower"]-1e-10 <= metrics["multiclass_brier"] <= bound["brier_upper"]+1e-10)
                if not model.converged_:
                    row["optimizer_message"] = model.optimizer_message_
            except Exception as exc:
                row.update({"status": "failed", "exception": repr(exc), "traceback": traceback.format_exc()})
            row.update({"method_seconds": time.perf_counter()-tick,
                        "cpu_seconds": sum(psutil.Process().cpu_times()[:2])-cpu0,
                        "rss_peak_sampled_mb": memory.read_mb()})
            output.append(row)
    except Exception as exc:
        for method in METHODS:
            output.append({"phase": phase, "scenario": scenario, "rep": rep, "method": method,
                           "status": "failed", "exception": repr(exc), "traceback": traceback.format_exc()})
    finally:
        memory.close()
    return output, time.perf_counter()-start_time


def _holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    previous = 0.
    for rank, ix in enumerate(order):
        previous = max(previous, (len(pvalues)-rank)*pvalues[ix])
        adjusted[ix] = min(previous, 1.)
    return adjusted


def _summarize(raw, phase):
    frame = pd.DataFrame(raw)
    frame.to_csv(OUT/f"{phase}_raw.csv", index=False)
    good = frame[frame.status == "ok"].copy()
    columns = ["oracle_brier", "auc_macro", "auc_impairment", "auc_death", "ece_macro_10_fixed_bins",
               "calibration_error_impairment", "calibration_error_death", "observed_panel_nll",
               "brier_identified_lower", "brier_identified_upper", "identified_state_fraction",
               "fit_seconds", "method_seconds", "cpu_seconds", "rss_peak_sampled_mb",
               "inference_ms_per_1000", "converged"]
    if len(good):
        summary = good.groupby(["scenario", "method"])[columns].agg(["mean", "std", "count"])
        summary.columns = ["__".join(c) for c in summary.columns]
        summary.reset_index().to_csv(OUT/f"{phase}_summary.csv", index=False)
    comparisons = []
    # The independent unit is the independently generated replicate, never a
    # seed duplicated on the same people or a three-domain pseudo sample.
    for scenario in SCENARIOS:
        sf = good[good.scenario == scenario]
        wide = sf.pivot(index="rep", columns="method", values="oracle_brier")
        for baseline in ["unordered", "strong_history"]:
            if baseline not in wide or "partial_order" not in wide:
                continue
            difference = (wide[baseline]-wide.partial_order).dropna().to_numpy(float)
            nr = len(difference)
            if nr < 2:
                continue
            mean = float(difference.mean())
            sd = float(difference.std(ddof=1))
            mcse = sd/np.sqrt(nr)
            # Monte Carlo CI for the expected paired performance difference
            # under this fixed DGP. It is NOT a real-cohort population CI.
            critical = student_t.ppf(.975, nr-1)
            # A paired sign-flip null test across independently simulated
            # replicates tests zero symmetric mean difference under its stated
            # exchangeability condition. It is not claimed as a clinical test.
            rng = np.random.default_rng(701003+SCENARIOS.index(scenario)*7+(baseline == "unordered"))
            flips = rng.choice([-1., 1.], size=(19999, nr))
            null_mean = (flips*difference).mean(axis=1)
            p = (1+int((np.abs(null_mean) >= abs(mean)).sum()))/20000
            comparisons.append({"scenario": scenario, "baseline": baseline, "comparison": "baseline_brier_minus_partial_order",
                                "n_paired_replicates": nr, "mean_improvement": mean, "replicate_sd": sd,
                                "monte_carlo_se": mcse, "mean_ci95_lower": mean-critical*mcse,
                                "mean_ci95_upper": mean+critical*mcse, "sign_flip_p": p,
                                "test_assumption": "independent replicates; symmetric paired-difference null"})
    contrasts = pd.DataFrame(comparisons)
    if len(contrasts):
        contrasts["holm_p_all_prespecified_contrasts"] = _holm(contrasts.sign_flip_p)
        contrasts.to_csv(OUT/f"{phase}_paired_comparisons.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pilot", "main"], required=True)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--n", type=int, default=400)
    args = parser.parse_args()
    reps = args.reps or (20 if args.phase == "pilot" else 200)
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {"phase": args.phase, "n_per_domain": args.n, "repetitions_per_scenario": reps,
              "scenarios": SCENARIOS, "methods": METHODS, "synthetic_domains_per_repetition": 3,
              "heldout_rule": "rep modulo 3", "penalty": .03, "max_iterations": 130,
              "train_domain_support_threshold": .02, "uniform_interval_draws": 64,
              "features": "observed demographics and historical visit-derived levels/slopes only",
              "complete_simulated_death_ascertainment": True, "max_workers": min(3, args.workers),
              "threads_per_worker": 1, "Brier_definition": "three-class squared-error sum",
              "equivalence": "wave_order and right_endpoint coincide for this simulator schedule",
              "uncertainty": "repeat SD and MCSE; CI targets mean paired synthetic-DGP difference",
              "observed_brier": "model-independent horizon-state masks yield identified bounds, never midpoint labels"}
    frozen["code_sha256"] = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                              for name in ["simulation_prediction.py", "interval_model.py", "sim_data.py", "interval_core.py"]}
    frozen["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "pandas": pd.__version__,
                              "platform": platform.platform(), "processor": platform.processor(),
                              "logical_cpus": psutil.cpu_count(), "ram_gb": psutil.virtual_memory().total/2**30}
    (OUT/f"{args.phase}_frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    ledger = OUT/f"{args.phase}_ledger.jsonl"
    existing = []
    completed = set()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            existing.extend(record["rows"])
            completed.add((record["scenario"], record["rep"]))
    jobs = [(scenario, rep, args.n, args.phase) for scenario in SCENARIOS for rep in range(reps)
            if (scenario, rep) not in completed]
    start = time.perf_counter()
    all_rows = existing
    with ProcessPoolExecutor(max_workers=min(3, args.workers)) as pool, ledger.open("a", encoding="utf-8") as handle:
        futures = {pool.submit(_run_one, job): job for job in jobs}
        done = 0
        for future in as_completed(futures):
            job = futures[future]
            rows, elapsed = future.result()
            record = {"scenario": job[0], "rep": job[1], "seconds": elapsed, "rows": rows}
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=True)+"\n")
            handle.flush()
            all_rows.extend(rows)
            done += 1
            if done % 5 == 0 or done == len(jobs):
                failed = sum(r["status"] != "ok" for r in all_rows)
                print(json.dumps({"phase": args.phase, "completed_new": done, "total_new": len(jobs),
                                  "elapsed_seconds": round(time.perf_counter()-start, 2), "failed_method_rows": failed}), flush=True)
    _summarize(all_rows, args.phase)
    print(json.dumps({"phase": args.phase, "complete": True, "method_rows": len(all_rows),
                      "wall_seconds_this_invocation": time.perf_counter()-start}), flush=True)


if __name__ == "__main__":
    main()
