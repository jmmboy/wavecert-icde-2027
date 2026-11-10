"""Stratified Monte Carlo inference for the fixed rotating-domain benchmark.

The sampling units are independently simulated repetitions. Three specified
synthetic domains are fixed strata, not independently sampled human cohorts.
Equal domain macro means and within-domain Monte Carlo errors are separated
from between-domain heterogeneity. This does not change fitted models.
"""
import bootstrap_env
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "simulation_prediction"


def _holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    result = np.empty_like(p)
    previous = 0.
    for rank, j in enumerate(order):
        previous = max(previous, (len(p)-rank)*p[j])
        result[j] = min(1., previous)
    return result


def main():
    records = [json.loads(s) for s in (OUT/"main_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for record in records for r in record["rows"]]
    frame = pd.DataFrame(rows)
    good = frame[frame.status == "ok"].copy()
    numeric = ["oracle_brier", "auc_macro", "auc_impairment", "auc_death",
               "calibration_error_impairment", "calibration_error_death", "ece_macro_10_fixed_bins",
               "observed_panel_nll", "brier_identified_lower", "brier_identified_upper", "identified_state_fraction",
               "fit_seconds", "cpu_seconds", "rss_peak_sampled_mb", "inference_ms_per_1000", "converged"]
    bydomain = good.groupby(["scenario", "method", "heldout_domain"])[numeric].agg(["mean", "std", "count"])
    bydomain.columns = ["__".join(c) for c in bydomain.columns]
    bydomain = bydomain.reset_index()
    bydomain.to_csv(OUT/"main_by_domain.csv", index=False)
    macro = []
    for (scenario, method), table in bydomain.groupby(["scenario", "method"]):
        row = {"scenario": scenario, "method": method, "domains": len(table)}
        for metric in numeric:
            means = table[metric+"__mean"].to_numpy(float)
            sds = table[metric+"__std"].to_numpy(float)
            counts = table[metric+"__count"].to_numpy(float)
            row[metric+"__domain_macro_mean"] = float(np.mean(means))
            row[metric+"__between_domain_sd"] = float(np.std(means, ddof=1))
            row[metric+"__root_mean_within_domain_variance"] = float(np.sqrt(np.nanmean(sds**2)))
            row[metric+"__stratified_mcse"] = float(np.sqrt(np.nansum(sds**2/counts))/len(table))
        macro.append(row)
    pd.DataFrame(macro).to_csv(OUT/"main_stratified_summary.csv", index=False)
    comparisons = []
    for analysis in ["all_finite", "both_converged"]:
        for scenario, sf in good.groupby("scenario", dropna=False):
            for baseline in ["unordered", "strong_history"]:
                sub = sf[sf.method.isin([baseline, "partial_order"])]
                wide = sub.pivot(index=["rep", "heldout_domain"], columns="method", values="oracle_brier")
                if analysis == "both_converged":
                    conv = sub.pivot(index=["rep", "heldout_domain"], columns="method", values="converged")
                    wide = wide[conv.fillna(False).all(axis=1)]
                if baseline not in wide or "partial_order" not in wide:
                    continue
                differences = (wide[baseline]-wide.partial_order).dropna().rename("difference").reset_index()
                strata = differences.groupby("heldout_domain").difference.agg(["mean", "std", "count"])
                if len(strata) != 3 or strata["count"].min() < 2:
                    continue
                mean = float(strata["mean"].mean())
                varparts = strata["std"].to_numpy()**2/strata["count"].to_numpy()/9.
                variance = float(varparts.sum())
                se = np.sqrt(variance)
                denominator = (varparts**2/(strata["count"].to_numpy()-1)).sum()
                df = variance**2/denominator if denominator > 0 else np.inf
                critical = student_t.ppf(.975, df)
                pvalue = float(2*student_t.sf(abs(mean/se), df)) if se > 0 else (1. if mean == 0 else 0.)
                row = {"analysis": analysis, "scenario": scenario, "baseline": baseline,
                       "contrast": "baseline_brier_minus_partial_order", "n_paired_replicates": len(differences),
                       "equal_domain_mean_improvement": mean,
                       "between_domain_improvement_sd": float(strata["mean"].std(ddof=1)),
                       "within_domain_improvement_sd": float(np.sqrt(np.mean(strata["std"].to_numpy()**2))),
                       "stratified_monte_carlo_se": float(se), "welch_degrees_of_freedom": float(df),
                       "mean_ci95_lower": float(mean-critical*se), "mean_ci95_upper": float(mean+critical*se),
                       "mc_mean_zero_p": pvalue,
                       "inference_target": "expected paired algorithm performance, equal mixture of 3 fixed synthetic domains",
                       "assumption": "independently generated repetitions; finite variances; asymptotic within-stratum mean normality"}
                comparisons.append(row)
    contrast = pd.DataFrame(comparisons)
    if len(contrast):
        contrast["holm_p_16_prespecified_contrasts"] = contrast.groupby("analysis").mc_mean_zero_p.transform(_holm)
        contrast.to_csv(OUT/"main_stratified_comparisons.csv", index=False)
    calrows = []
    for row in rows:
        if row["status"] != "ok" or "calibration_bins" not in row:
            continue
        for state, bins in row["calibration_bins"].items():
            for b in range(10):
                calrows.append({"scenario": row["scenario"], "method": row["method"],
                                "heldout_domain": row["heldout_domain"], "state": int(state), "risk_bin": b,
                                "n": bins["n"][b], "predicted_sum": bins["risk_sum"][b],
                                "observed_sum": bins["outcome_sum"][b]})
    cal = pd.DataFrame(calrows).groupby(["scenario", "method", "heldout_domain", "state", "risk_bin"])[["n", "predicted_sum", "observed_sum"]].sum().reset_index()
    cal["mean_prediction"] = cal.predicted_sum/cal.n.replace(0, np.nan)
    cal["observed_frequency"] = cal.observed_sum/cal.n.replace(0, np.nan)
    cal.to_csv(OUT/"main_calibration_bins.csv", index=False)
    frozen = json.loads((OUT/"main_frozen.json").read_text(encoding="utf-8"))
    current_hashes = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                      for name in frozen["code_sha256"]}
    identity = frame[["scenario", "rep", "method"]].duplicated().sum()
    shape_counts = frame.groupby("scenario").size().to_dict()
    checks = {"source": "main_ledger.jsonl", "independent_simulation_repetitions": len(records),
              "model_fits": len(frame), "expected_model_fits": 8*200*8,
              "duplicate_scenario_rep_method": int(identity), "rows_by_scenario": shape_counts,
              "exception_rows": int((frame.status != "ok").sum()),
              "nonconverged_finite_rows": int((~good.converged).sum()),
              "oracle_outside_logical_observed_bounds": int((~good.oracle_within_observed_brier_bounds).sum()),
              "peak_single_worker_sampled_rss_mb": float(good.rss_peak_sampled_mb.max()),
              "total_method_cpu_seconds": float(good.cpu_seconds.sum()),
              "sum_repetition_wall_seconds": float(sum(r["seconds"] for r in records)),
              "max_abs_log_rate_clipping_fraction": float(good.rate_clipping_fraction.max()),
              "all_prediction_sources_match_frozen_sha256": current_hashes == frozen["code_sha256"],
              "stratified_postprocessor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "notes": ["Synthetic domains never count as real cohorts", "MC CIs describe fixed-DGP algorithm performance, not human-population uncertainty",
                        "Brier logical bounds identify partial horizon-state scores; no candidate-specific pseudo-labels"]}
    wave = good[good.method.isin(["wave_order", "right_endpoint"])].pivot(index=["scenario", "rep"], columns="method", values="oracle_brier")
    checks["max_abs_wave_vs_right_endpoint_brier_difference"] = float((wave.wave_order-wave.right_endpoint).abs().max())
    (OUT/"main_run_audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
