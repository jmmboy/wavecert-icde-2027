"""Separate iteration-cap sensitivity; never replaces the 200 primary draws.

Repeat every affected bootstrap population with maxiter=400, otherwise the
same cold-start pipeline. Retry all failed fits and report objective/probability
changes. Successful original fits stop by the same convergence rule, so only
the raised cap is changed. All source tuning and final fitting rerun.
"""
import bootstrap_env
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import real_bootstrap_fast as base
from real_benchmark import predict

OUT = base.OUT


def _init():
    base._init()
    base.MAX_ITER = 400


def _one(job):
    rep, state, failed = job
    base.MAX_ITER = 400
    full = base._one((rep, state, False))
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    weights = base._draw(rng)
    checks = []
    for original in failed:
        cache = base.G["models"][original["cohort"]]
        train, test = cache["train"], cache["test"]
        idx = np.repeat(train, weights[train].astype(int))
        selected = base._selected(train, weights)
        X, labels = base._X(selected, original["method"])
        subset = idx[base.G["inner"][idx] != 0] if original["stage"] == "inner_tuning" else idx
        base.MAX_ITER = 120
        old = base._fit(X, labels, subset, original["penalty"], cache, original["method"])
        base.MAX_ITER = 400
        new = base._fit(X, labels, subset, original["penalty"], cache, original["method"])
        pp = predict(old, X, base.G["e"], test)
        qp = predict(new, X, base.G["e"], test)
        check = {"rep": rep, "cohort": original["cohort"], "stage": original["stage"],
                 "method": original["method"], "penalty": original["penalty"],
                 "old_max_iter": 120, "new_max_iter": 400,
                 "old_converged": old.converged_, "new_converged": new.converged_,
                 "old_iterations": old.n_iter_, "new_iterations": new.n_iter_,
                 "old_objective": old.objective_, "new_objective": new.objective_,
                 "objective_improvement": old.objective_-new.objective_,
                 "old_gradient_max": old.gradient_max_, "new_gradient_max": new.gradient_max_,
                 "max_abs_prediction_change": float(np.max(np.abs(pp-qp))),
                 "mean_abs_prediction_change": float(np.mean(np.abs(pp-qp))),
                 "repeat_120_matches_primary_objective": abs(old.objective_-original["objective"]) <= 1e-10,
                 "target_brier_120": base._brier(base.G["e"].iloc[test].y, pp, weights[test]),
                 "target_brier_400": base._brier(base.G["e"].iloc[test].y, qp, weights[test])}
        checks.append(check)
    return {"rep": rep, "full_pipeline": full, "failed_fit_checks": checks}


def main():
    start = time.perf_counter()
    original = pd.read_csv(OUT/"full_pipeline_fits.csv")
    failed = original[~original.converged]
    old_boot = pd.read_csv(OUT/"full_pipeline_bootstrap.csv")
    reps = sorted(int(value) for value in failed.rep.unique())
    base._init()
    states = base._states(max(reps)+1)
    config = {"affected_repetitions": reps, "original_max_iter": 120, "sensitivity_max_iter": 400,
              "unchanged": ["cold initialization", "tol1e-7", "household draws", "source-only mining",
                            "all three penalties retuned", "shared resampling across folds", "P/U exact reuse"],
              "primary_200_results_overwritten": False,
              "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/"bootstrap_convergence_sensitivity_frozen.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    jobs = [(rep, states[rep], failed[failed.rep == rep].to_dict("records")) for rep in reps]
    records = []
    with ProcessPoolExecutor(max_workers=3, initializer=_init) as pool:
        futures = [pool.submit(_one, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({"completed_affected_reps": len(records), "total": len(reps),
                              "wall_seconds": time.perf_counter()-start}), flush=True)
    checks = pd.DataFrame([r for record in records for r in record["failed_fit_checks"]])
    full = pd.DataFrame([r for record in records for r in record["full_pipeline"]["rows"]])
    fit = pd.DataFrame([r for record in records for r in record["full_pipeline"]["fits"]])
    checks.to_csv(OUT/"bootstrap_convergence_sensitivity_fits.csv", index=False)
    full.to_csv(OUT/"bootstrap_convergence_sensitivity_pipeline.csv", index=False)
    fit.to_csv(OUT/"bootstrap_convergence_sensitivity_all_fits.csv", index=False)
    (OUT/"bootstrap_convergence_sensitivity_ledger.jsonl").write_text("\n".join(json.dumps(r) for r in records)+"\n", encoding="utf-8")
    merged = full[full.cohort != "macro"].merge(old_boot[old_boot.cohort != "macro"], on=["rep", "cohort"], suffixes=("_400", "_120"))
    summary = {"primary_repetitions": int(old_boot.rep.nunique()), "primary_actual_fit_count": len(original),
               "primary_nonconverged_fit_count": len(failed), "primary_nonconverged_fit_fraction": len(failed)/len(original),
               "primary_affected_repetition_count": len(reps),
               "primary_nonconverged_tuning_fit_count": int((failed.stage == "inner_tuning").sum()),
               "primary_nonconverged_full_source_fit_count": int((failed.stage == "full_source").sum()),
               "retry_fit_count": len(checks), "retry_converged_fit_count": int(checks.new_converged.sum()),
               "retry_nonconverged_fit_count": int((~checks.new_converged).sum()),
               "sensitivity_pipeline_fit_count": len(fit), "sensitivity_pipeline_nonconverged_fit_count": int((~fit.converged).sum()),
               "reproduced_original_120_objectives": bool(checks.repeat_120_matches_primary_objective.all()),
               "max_objective_improvement": float(checks.objective_improvement.max()),
               "max_abs_prediction_change": float(checks.max_abs_prediction_change.max()),
               "max_mean_abs_prediction_change": float(checks.mean_abs_prediction_change.max()),
               "retuned_penalty_changed_fold_count": int((merged.penalty_400 != merged.penalty_120).sum()),
               "reselected_catalog_changed_fold_count": int((merged.selected_400 != merged.selected_120).sum()),
               "max_abs_primary_paired_difference_change": float((merged.improvement_400-merged.improvement_120).abs().max()),
               "max_abs_sensitivity_P_minus_U_brier": float(full.improvement.abs().max()),
               "wall_seconds": time.perf_counter()-start, "primary_results_replaced": False}
    (OUT/"bootstrap_convergence_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
