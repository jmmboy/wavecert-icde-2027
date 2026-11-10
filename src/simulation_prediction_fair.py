"""Prespecified additional direction-only control; original main is untouched.

Two core scenarios (null and chain), 200 repetitions each, n=400 per domain,
same seeds/splits as the original main prediction benchmark. Both arms share
the same event bag, unordered pairs, widths, overlap and unknownness. The only
added inputs are signed *certain* directions. No ambiguity/co-occurrence
features are added exclusively to the direction arm.
"""
import bootstrap_env
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import simulation_prediction as bench
from sim_data import simulate_domain, EVENT_TYPES
from interval_model import ReversiblePanelCTMC, oracle_state_metrics, partial_state_brier_bounds

OUT = bench.ROOT / "results" / "simulation_prediction_fair"
ARMS = ("fair_unordered_overlap_width", "fair_plus_certain_direction")


def _design(data, keep):
    shared = [data["design"]["unordered"], data["design"]["partial_order"][:, -3:]]
    direction = []
    bounds = data["chain_bounds"]
    for a, b in combinations(EVENT_TYPES, 2):
        j = bench.PATTERNS.index((a, b))
        k = bench.PATTERNS.index((b, a))
        if not (keep[j] or keep[k]):
            continue
        name, reverse = a+">"+b, b+">"+a
        overlap = bounds[name+"__ambiguous"].to_numpy(float)
        unknown = bounds[name+"__unknown"].to_numpy(float)
        shared.append(np.column_stack((overlap, unknown)))
        signed = bounds[name+"__lower"].to_numpy(float)-bounds[reverse+"__lower"].to_numpy(float)
        direction.append(signed)
    X = np.column_stack(shared)
    D = np.column_stack(direction) if direction else np.empty((len(X), 0))
    return {ARMS[0]: X, ARMS[1]: np.column_stack((X, D))}, D.shape[1]


def _one(job):
    scenario, rep, n = job
    threadpool_limits(1)
    seed = 31000000+bench.SCENARIOS.index(scenario)*100000+rep*10
    tick = time.perf_counter()
    sampler = bench._MemorySampler()
    rows = []
    try:
        domains = [bench._observed_domain(simulate_domain(n, seed+d, scenario, d), seed+d+777)
                   for d in range(3)]
        testdomain = rep % 3
        train = [data for j, data in enumerate(domains) if j != testdomain]
        test = domains[testdomain]
        keep = bench._train_chain_selector(train)
        designs = [_design(data, keep) for data in domains]
        panel = bench._join_panel(train)
        for arm in ARMS:
            sampler.reset()
            cpu0 = sum(psutil.Process().cpu_times()[:2])
            Xtrain = np.vstack([designs[j][0][arm] for j in range(3) if j != testdomain])
            Xtest = designs[testdomain][0][arm]
            row = {"scenario": scenario, "rep": rep, "seed": seed, "heldout_domain": testdomain,
                   "n_per_domain": n, "method": arm, "status": "ok", "direction_pair_count": designs[0][1]}
            try:
                model = ReversiblePanelCTMC(penalty=.03, max_iter=400, tol=1e-7).fit(Xtrain, panel)
                pred = model.predict_state(Xtest, t=3, start_state=test["start"])
                metric = oracle_state_metrics(pred, test["truth"])
                bound = partial_state_brier_bounds(pred, test["mask"])
                auc = [roc_auc_score(test["truth"] == s, pred[:, s]) if len(np.unique(test["truth"] == s)) > 1 else np.nan for s in range(3)]
                row.update({"oracle_brier": metric["multiclass_brier"], "oracle_log_loss": metric["log_loss"],
                            "auc_macro": float(np.nanmean(auc)), "auc_impairment": float(auc[1]), "auc_death": float(auc[2]),
                            "calibration_bias_impairment": float(pred[:, 1].mean()-(test["truth"] == 1).mean()),
                            "calibration_bias_death": float(pred[:, 2].mean()-(test["truth"] == 2).mean()),
                            "observed_panel_nll": model.observed_nll(Xtest, test["panel"]),
                            "brier_identified_lower": bound["brier_lower"], "brier_identified_upper": bound["brier_upper"],
                            "identified_state_fraction": bound["exact_horizon_state_fraction"],
                            "fit_seconds": model.fit_seconds_, "converged": model.converged_,
                            "iterations": model.n_iter_, "gradient_max": model.gradient_max_,
                            "rate_clipping_fraction": model.rate_clipping_fraction_, "feature_count": Xtest.shape[1]})
            except Exception as exc:
                row.update({"status": "failed", "exception": repr(exc)})
            row.update({"rss_peak_sampled_mb": sampler.read_mb(), "cpu_seconds": sum(psutil.Process().cpu_times()[:2])-cpu0})
            rows.append(row)
    finally:
        sampler.close()
    return {"scenario": scenario, "rep": rep, "seconds": time.perf_counter()-tick, "rows": rows}


def _summarize(records):
    frame = pd.DataFrame([row for record in records for row in record["rows"]])
    frame.to_csv(OUT/"raw.csv", index=False)
    good = frame[frame.status == "ok"].copy()
    metrics = ["oracle_brier", "auc_macro", "auc_impairment", "auc_death", "fit_seconds", "converged", "rss_peak_sampled_mb"]
    summary = good.groupby(["scenario", "method"])[metrics].agg(["mean", "std", "count"])
    summary.columns = ["__".join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT/"summary.csv", index=False)
    comparison = []
    for scenario, sf in good.groupby("scenario"):
        wide = sf.pivot(index=["rep", "heldout_domain"], columns="method", values="oracle_brier")
        delta = (wide[ARMS[0]]-wide[ARMS[1]]).dropna().rename("improvement").reset_index()
        grouped = delta.groupby("heldout_domain").improvement.agg(["mean", "std", "count"])
        mu = float(grouped["mean"].mean())
        v = grouped["std"].to_numpy()**2/grouped["count"].to_numpy()/9
        se = float(np.sqrt(v.sum()))
        df = float(v.sum()**2/np.sum(v**2/(grouped["count"].to_numpy()-1)))
        crit = student_t.ppf(.975, df)
        comparison.append({"scenario": scenario, "n_paired_replicates": len(delta), "mean_direction_improvement": mu,
                           "stratified_mcse": se, "mean_ci95_lower": mu-crit*se, "mean_ci95_upper": mu+crit*se,
                           "within_domain_difference_sd": float(np.sqrt(np.mean(grouped["std"].to_numpy()**2))),
                           "between_domain_difference_sd": float(grouped["mean"].std(ddof=1)),
                           "mc_mean_zero_p": float(2*student_t.sf(abs(mu/se), df)),
                           "inference_target": "mean direction increment across 3 fixed synthetic domains; independent generated repetitions"})
    comparison = pd.DataFrame(comparison)
    comparison["holm_p_2_prespecified_scenarios"] = bench._holm(comparison.mc_mean_zero_p)
    comparison.to_csv(OUT/"paired_comparisons.csv", index=False)
    audit = {"independent_repetitions": len(records), "fits": len(frame),
             "exceptions": int((frame.status != "ok").sum()), "not_converged": int((~good.converged).sum()),
             "sum_rep_seconds": float(sum(r["seconds"] for r in records)),
             "max_rss_mb": float(good.rss_peak_sampled_mb.max()),
             "purpose": "additional fair direction-only control; does not overwrite original main"}
    (OUT/"run_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--n", type=int, default=400)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {"scenarios": ["null", "chain"], "repetitions_per_scenario": args.reps, "n_per_domain": args.n,
              "seed_policy": "same datasets and rotating heldout domains as original main",
              "arms": ARMS, "ridge_penalty": .03, "max_iter_each_arm": 400,
              "common_features": "observed strong history, event bag, all unordered pairs, interval width, overlap, unknownness",
              "only_added_features": "signed certainty of each training-selected pair direction",
              "purpose": "isolate direction after shared co-occurrence and observation uncertainty; independent additional analysis",
              "sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                         for name in ["simulation_prediction_fair.py", "simulation_prediction.py", "sim_data.py", "interval_core.py", "interval_model.py"]}}
    (OUT/"frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    ledger = OUT/"ledger.jsonl"
    records = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    finished = {(r["scenario"], r["rep"]) for r in records}
    jobs = [(scenario, rep, args.n) for scenario in ["null", "chain"] for rep in range(args.reps) if (scenario, rep) not in finished]
    tick = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(3, args.workers)) as pool, ledger.open("a", encoding="utf-8") as handle:
        futures = [pool.submit(_one, job) for job in jobs]
        for j, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            handle.write(json.dumps(record, allow_nan=True)+"\n")
            handle.flush()
            if j % 20 == 0:
                print(json.dumps({"finished_new": j, "total_new": len(jobs), "wall_seconds": time.perf_counter()-tick}), flush=True)
    _summarize(records)
    (OUT/"wall_time.json").write_text(json.dumps({"wall_seconds": time.perf_counter()-tick}), encoding="utf-8")


if __name__ == "__main__":
    main()
