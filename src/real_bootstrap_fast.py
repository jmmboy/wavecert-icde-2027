"""Full household bootstrap with exact repeated rows and analytic cold fits.

Every replicate resamples cohort x discovery/confirmation households ONCE,
shared by all LOCO folds. Source support selection, preprocessing, all three
penalty candidates, final fits, and target household scoring are rerun.
Warm starts were rejected after reaching different numerical stationary points.
The original cold initial point, tolerance, iteration cap and objective are
preserved. The equivalent two-state expression avoids general matrix tensors.
"""
import bootstrap_env
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from scipy.optimize import minimize

from interval_model import ReversiblePanelCTMC
from real_benchmark import load_features, design, select_patterns, predict, metrics, ROOT
from real_statistics import family_resample_weights

OUT = ROOT / "results" / "statistics"
G = {}
MAX_ITER = 120
FIT_TOL = 1e-7


def _brier(y, probability, weights=None):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(probability, float), 1e-7, 1-1e-7)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    good = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    return float(np.average((p[good]-y[good])**2, weights=w[good]))


def _init():
    threadpool_limits(1)
    e, logs, patterns, f, representations, width = load_features()
    models = {p.stem: pickle.load(p.open("rb")) for p in (ROOT/"private"/"models").glob("*.pkl")}
    if len(models) < 3:
        raise RuntimeError("Wait for all three frozen main LOCO model files")
    strata = []
    for cohort in sorted(e.cohort.unique()):
        for split in [0, 1]:
            ix = np.flatnonzero((e.cohort == cohort) & (e.confirmation == split) & (e.response == 1))
            if len(ix):
                strata.append((ix, e.iloc[ix].household_id.to_numpy(str)))
    inner = np.array([int(hashlib.sha256((str(v)+"|inner").encode()).hexdigest()[:8], 16) % 5
                      for v in e.household_id])
    lower = np.column_stack([f[">".join(p)+"__lower"].to_numpy(float) for p in patterns])
    upper = np.column_stack([f[">".join(p)+"__upper"].to_numpy(float) for p in patterns])
    G.update(e=e, patterns=patterns, f=f, representations=representations, width=width,
             models=models, strata=strata, inner=inner, lower=lower, upper=upper, cache=OrderedDict())


def _draw(rng):
    weights = np.zeros(len(G["e"]), dtype=float)
    for ix, families in G["strata"]:
        weights[ix] = family_resample_weights(families, rng)
    return weights


def _states(R):
    rng = np.random.default_rng(550031)
    states = []
    for _ in range(R):
        states.append(copy.deepcopy(rng.bit_generator.state))
        _draw(rng)
    return states


def _selected(train, weights):
    """Exact weighted equivalent of selecting on np.repeat(train, counts)."""
    e, lower, upper = G["e"], G["lower"], G["upper"]
    passes = np.zeros(len(G["patterns"]), dtype=int)
    totals = np.zeros(len(G["patterns"]))
    for cohort in sorted(e.iloc[train].cohort.unique()):
        ix = train[e.iloc[train].cohort.to_numpy() == cohort]
        w = weights[ix]
        if w.sum() == 0:
            continue
        l, u = lower[ix], upper[ix]
        certain = (np.nan_to_num(l)*w[:, None]).sum(axis=0)
        eligible = (np.isfinite(l) & np.isfinite(u))*w[:, None]
        eligible = eligible.sum(axis=0)
        passes += certain >= np.maximum(10, .0025*eligible)
        totals += certain
    selected = np.flatnonzero(passes >= 2).tolist()
    return sorted(selected, key=lambda k: (-totals[k], G["patterns"][k]))[:12]


def _X(selected, method):
    if not selected and method == "PartialOrder":
        return _X(selected, "Unordered")
    key = (tuple(selected), method)
    cache = G["cache"]
    if key not in cache:
        cache[key] = design(G["e"], G["f"], G["representations"], G["width"],
                            G["patterns"], selected, method)
        while len(cache) > 12:
            cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]


def _warm(old, oldlabels, X, labels, idx):
    """Map old centered predictors to raw, then to new training scale.

    Recomputed imputation medians need a compensating coefficient on each
    existing missingness indicator. Without it, merely rescaling coefficients
    changes the starting linear predictor for missing rows.
    """
    scratch = ReversiblePanelCTMC(include_death=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        scratch._features(X[idx], fit=True)
    oldmap = {label: j for j, label in enumerate(oldlabels)}
    newmap = {label: j for j, label in enumerate(labels)}
    raw = np.zeros((4, len(labels)))
    raw_intercept = old.params_[:, 0].copy()
    for label, newj in newmap.items():
        if label in oldmap:
            oldj = oldmap[label]
            beta = old.params_[:, 1+oldj] / old.scale_[oldj]
            raw[:, newj] = beta
            raw_intercept -= beta*old.mean_[oldj]
    for label, newj in newmap.items():
        if label in oldmap and not label.endswith("__missing") and label+"__missing" in newmap:
            oldj = oldmap[label]
            beta = old.params_[:, 1+oldj] / old.scale_[oldj]
            raw[:, newmap[label+"__missing"]] += beta*(old.median_[oldj]-scratch.median_[newj])
    initial = np.column_stack((raw_intercept + raw @ scratch.mean_, raw*scratch.scale_[None, :]))
    return initial


def _fit(X, labels, idx, penalty, cache, method, warm=False, reference=False):
    e = G["e"]
    old = cache[method]
    initial = _warm(old["model"], old["labels"], X, labels, idx) if warm else None
    obs = {"person": np.arange(len(idx)), "delta": e.iloc[idx].delta.to_numpy(float),
           "start": e.iloc[idx].baseline_state.to_numpy(int), "end": e.iloc[idx].y.to_numpy(int)}
    model = ReversiblePanelCTMC(penalty=penalty, max_iter=MAX_ITER, tol=FIT_TOL, include_death=False)
    if reference:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            model.fit(X[idx], obs, initial_params=initial)
        return model
    tick = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        Z = model._features(X[idx], fit=True)
    dt = obs["delta"]
    s, y = obs["start"], obs["end"]
    shape = (2, Z.shape[1])
    if initial is None:
        initial = np.zeros(shape)
        for j, origin in enumerate([0, 1]):
            exposure = dt[s == origin].sum()
            rate = ((s == origin) & (y == 1-origin)).sum()/max(exposure, 1)
            initial[j, 0] = np.log(np.clip(rate, .005, 1))
    else:
        initial = initial[[0, 2]]
    sign = 2*y-1

    def objective(flat):
        theta = flat.reshape(shape)
        eta = Z @ theta.T
        active = (eta > model.lograte_bounds[0]) & (eta < model.lograte_bounds[1])
        rate = np.exp(np.clip(eta, *model.lograte_bounds))
        a, c = rate.T
        k = a+c
        ex = np.exp(-k*dt)
        h = -np.expm1(-k*dt)
        imp0 = a/k*h
        imp1 = 1-c/k*h
        p = np.where(s == 0, imp0, imp1)
        da = np.where(s == 0, c/k**2*h+a/k*dt*ex, c/k**2*h-c/k*dt*ex)
        dc = np.where(s == 0, -a/k**2*h+a/k*dt*ex, -a/k**2*h-c/k*dt*ex)
        lik = np.maximum(np.where(y == 1, p, 1-p), 1e-300)
        nll = -np.log(lik).mean()+.5*penalty*np.square(theta[:, 1:]).sum()
        d_eta = (-sign/lik/len(idx))[:, None]*np.column_stack((da, dc))*rate*active
        gradient = d_eta.T @ Z
        gradient[:, 1:] += penalty*theta[:, 1:]
        return float(nll), gradient.ravel()

    result = minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B",
                      options={"maxiter": MAX_ITER, "ftol": FIT_TOL, "gtol": max(FIT_TOL, 1e-6), "maxls": 30})
    model.params_ = np.zeros((4, Z.shape[1]))
    model.params_[[0, 2]] = result.x.reshape(shape)
    model.params_[[1, 3], 0] = np.log(.005)
    model.intercept_ = model.params_[:, 0].copy()
    model.coef_ = model.params_[:, 1:].copy()
    model.converged_ = bool(result.success)
    model.n_iter_ = int(result.nit)
    model.gradient_max_ = float(np.abs(result.jac).max())
    model.optimizer_message_ = str(result.message)
    model.objective_ = float(result.fun)
    model.fit_seconds_ = time.perf_counter()-tick
    model.rate_clipping_fraction_ = float((~model._rates(Z)[1][:, [0, 2]]).mean())
    return model


def _one(job):
    rep, state, cold_check = job
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    shared = _draw(rng)
    e = G["e"]
    tick = time.perf_counter()
    process = psutil.Process()
    cpu0 = sum(process.cpu_times()[:2])
    rows, fits, checks, diffs, converged = [], [], [], [], []
    for target, cache in sorted(G["models"].items()):
        train, test = cache["train"], cache["test"]
        idx = np.repeat(train, shared[train].astype(int))
        selected = _selected(train, shared)
        if cold_check:
            expected, _ = select_patterns(e, G["f"], G["patterns"], idx)
            if selected != expected:
                raise AssertionError("Weighted mining differs from exact repeated-row mining")
        itrain, ival = idx[G["inner"][idx] != 0], idx[G["inner"][idx] == 0]
        U, ulabels = _X(selected, "Unordered")
        P, plabels = _X(selected, "PartialOrder")

        def fit_and_record(X, labels, subset, penalty, method, stage):
            model = _fit(X, labels, subset, penalty, cache, method)
            item = {"rep": rep, "cohort": target, "stage": stage, "method": method, "penalty": penalty,
                    "n_rows": len(subset), "features": X.shape[1], "selected": len(selected),
                    "fit_seconds": model.fit_seconds_, "iterations": model.n_iter_, "converged": model.converged_,
                    "gradient_max": model.gradient_max_, "objective": model.objective_,
                    "rate_clipping_fraction": model.rate_clipping_fraction_, "optimizer_message": model.optimizer_message_}
            if cold_check:
                cold = _fit(X, labels, subset, penalty, cache, method, warm=False, reference=True)
                pp = predict(model, X, e, test)
                cp = predict(cold, X, e, test)
                check = {"rep": rep, "cohort": target, "stage": stage, "method": method, "penalty": penalty,
                         "warm_objective": model.objective_, "cold_objective": cold.objective_,
                         "abs_objective_difference": abs(model.objective_-cold.objective_),
                         "max_abs_probability_difference": float(np.max(np.abs(pp-cp))),
                         "mean_abs_probability_difference": float(np.mean(np.abs(pp-cp))),
                         "warm_converged": model.converged_, "cold_converged": cold.converged_,
                         "warm_gradient": model.gradient_max_, "cold_gradient": cold.gradient_max_,
                         "warm_iterations": model.n_iter_, "cold_iterations": cold.n_iter_,
                         "objective_agrees_at_1e_minus_7": abs(model.objective_-cold.objective_) <= 1e-7,
                         "cold_fit_seconds": cold.fit_seconds_}
                checks.append(check)
                if not check["objective_agrees_at_1e_minus_7"]:
                    (OUT/("warm_cold_failure_rep_"+str(rep)+".json")).write_text(json.dumps(check, indent=2), encoding="utf-8")
                    raise AssertionError("Analytic-cold/reference-cold likelihood optima differ beyond 1e-7: "+json.dumps(check))
            fits.append(item)
            converged.append(model.converged_)
            return model

        tuning = []
        for penalty in [.003, .03, .3]:
            candidate = fit_and_record(U, ulabels, itrain, penalty, "Unordered", "inner_tuning")
            score = _brier(e.iloc[ival].y, predict(candidate, U, e, ival))
            tuning.append((score, penalty))
        penalty = min(tuning)[1]
        unordered = fit_and_record(U, ulabels, idx, penalty, "Unordered", "full_source")
        equal = U is P or (U.shape == P.shape and np.array_equal(U, P, equal_nan=True))
        partial = unordered if equal else fit_and_record(P, plabels, idx, penalty, "PartialOrder", "full_source")
        uscore = _brier(e.iloc[test].y, predict(unordered, U, e, test), shared[test])
        pscore = uscore if equal else _brier(e.iloc[test].y, predict(partial, P, e, test), shared[test])
        difference = uscore-pscore
        diffs.append(difference)
        rows.append({"rep": rep, "cohort": target, "improvement": difference,
                     "unordered_brier": uscore, "partial_brier": pscore, "selected": len(selected),
                     "selected_patterns": ";".join(">".join(G["patterns"][k]) for k in selected),
                     "penalty": penalty, "converged": bool(unordered.converged_ and partial.converged_),
                     "all_tuning_converged": bool(all(f["converged"] for f in fits if f["cohort"] == target)),
                     "exact_design_reuse": bool(equal), "unordered_gradient_max": unordered.gradient_max_,
                     "partial_gradient_max": partial.gradient_max_, "unordered_objective": unordered.objective_,
                     "partial_objective": partial.objective_, "tuning_brier_scores": json.dumps(tuning),
                     "bootstrap_global_weights_sha256": hashlib.sha256(shared.astype(np.int32).tobytes()).hexdigest()})
    rows.append({"rep": rep, "cohort": "macro", "improvement": float(np.mean(diffs)), "converged": bool(all(converged))})
    return {"rep": rep, "rows": rows, "fits": fits, "checks": checks, "seconds": time.perf_counter()-tick,
            "cpu_seconds": sum(process.cpu_times()[:2])-cpu0,
            "process_peak_rss_mb": getattr(process.memory_info(), "peak_wset", process.memory_info().rss)/2**20}


def _write(records, mode):
    frame = pd.DataFrame([r for record in records for r in record["rows"]])
    prefix = "full_pipeline" if mode == "main" else "bootstrap_fast_pilot"
    frame.to_csv(OUT/(prefix+"_bootstrap.csv"), index=False)
    fits = pd.DataFrame([r for record in records for r in record["fits"]])
    fits.to_csv(OUT/(prefix+"_fits.csv"), index=False)
    checks = [r for record in records for r in record["checks"]]
    if checks:
        pd.DataFrame(checks).to_csv(OUT/(prefix+"_warm_cold_checks.csv"), index=False)
    rows = []
    for cohort, group in frame.groupby("cohort"):
        rows.append({"cohort": cohort, "mean_improvement": group.improvement.mean(),
                     "bootstrap_se": group.improvement.std(ddof=1), "ci_low": group.improvement.quantile(.025),
                     "ci_high": group.improvement.quantile(.975), "R": len(group),
                     "all_converged": bool(group.converged.all()),
                     "scope": "source/target household resampling; repeated source rows; source preprocessing, support mining, three-penalty tuning and final models rerun; shared source resample across LOCO folds"})
    pd.DataFrame(rows).to_csv(OUT/(prefix+"_intervals.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "main"], required=True)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    R = args.reps or (3 if args.mode == "pilot" else 200)
    OUT.mkdir(exist_ok=True)
    _init()
    states = _states(R)
    config = {"repetitions": R, "mode": args.mode, "workers": min(3, args.workers),
              "source": "same scientific bootstrap as real_statistics.full_pipeline",
              "rng": "exact original PCG64 seed550031 serial stream; cohort x DISC/CONF draws shared across folds",
              "penalty_grid": [.003, .03, .3], "max_iter": MAX_ITER, "fit_tolerance": FIT_TOL,
              "initialization": "ORIGINAL COLD; warm rejected after observed different converged stationary points",
              "equivalent_backend": "analytic two-state zero-death transition likelihood and gradient",
              "warm_mapping_available_but_not_used": "old raw coefficients to new scale, label mapping and missing-imputation indicator compensation",
              "identical_design_reuse": True,
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                for name in ["real_bootstrap_fast.py", "real_benchmark.py", "real_statistics.py", "interval_model.py"]}}
    (OUT/("bootstrap_fast_"+args.mode+"_frozen.json")).write_text(json.dumps(config, indent=2), encoding="utf-8")
    ledger = OUT/("bootstrap_fast_cold_"+args.mode+"_ledger.jsonl")
    records = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    completed = {r["rep"] for r in records}
    jobs = [(r, states[r], args.mode == "pilot" and r == 0) for r in range(R) if r not in completed]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(3, args.workers), initializer=_init) as pool, ledger.open("a", encoding="utf-8") as handle:
        futures = {pool.submit(_one, job): job[0] for job in jobs}
        for j, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            handle.write(json.dumps(record, allow_nan=True)+"\n")
            handle.flush()
            _write(records, args.mode)
            print(json.dumps({"mode": args.mode, "completed": len(records), "R": R,
                              "wall_seconds": time.perf_counter()-start, "rep_seconds": record["seconds"],
                              "rep_peak_rss_mb": record["process_peak_rss_mb"]}), flush=True)
    audit = {"R": len(records), "wall_seconds": time.perf_counter()-start,
             "sum_rep_cpu_seconds": sum(r["cpu_seconds"] for r in records),
             "peak_worker_rss_mb": max(r["process_peak_rss_mb"] for r in records),
             "actual_fit_count": sum(len(r["fits"]) for r in records),
             "cold_check_count": sum(len(r["checks"]) for r in records),
             "all_fits_converged": all(f["converged"] for r in records for f in r["fits"])}
    (OUT/("bootstrap_fast_"+args.mode+"_audit.json")).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
