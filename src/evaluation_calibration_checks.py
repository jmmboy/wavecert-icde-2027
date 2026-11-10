"""Additional scoped MC-sampling and fixed-prediction binary-CI calibration.

These checks do not change primary results, repair model nonconvergence, or
validate latent three-state/interval-censored/full-refitting confidence limits.
Uniform interval probabilities below are assumption-dependent sensitivity
quantities, never identified probabilities of actual event order.
"""
import bootstrap_env  # noqa: F401
import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path
import platform
import sys
import time
import tracemalloc
import numpy as np
import pandas as pd
import psutil
from sim_data import simulate_domain, EVENT_TYPES
from interval_core import chain_truth
from real_statistics import clustered_mean_test, family_resample_weights

ROOT = Path(__file__).resolve().parents[1]
SAMPLING = ROOT / "results" / "sampling_scaling"
CALIBRATION = ROOT / "results" / "evaluation_calibration"


def analytic_uniform_order(left_a, right_a, left_b, right_b):
    """P(U_a<U_b), independent continuous uniforms within the two intervals.

    Integrate 1, (R_b-t)/(R_b-L_b), and 0 over the relevant portions of a.
    Missing recorded events are absent from the *recorded-log* target only.
    """
    la, ra, lb, rb = map(lambda x: np.asarray(x, float), (left_a, right_a, left_b, right_b))
    have = np.isfinite(la + ra + lb + rb)
    answer = np.zeros(len(la))
    wa, wb = ra[have] - la[have], rb[have] - lb[have]
    assert np.all(wa > 0) and np.all(wb > 0)
    full = np.maximum(0, np.minimum(ra[have], lb[have]) - la[have])
    lo, hi = np.maximum(la[have], lb[have]), np.minimum(ra[have], rb[have])
    length = np.maximum(0, hi - lo)
    partial = length * (rb[have] - (hi + lo) / 2) / wb
    answer[have] = np.clip((full + partial) / wa, 0, 1)
    return answer, have


def sampled_features(left, right, patterns, draws, seed, chunk=64):
    """Chunked possible-world features with exact prefixes in N and draw count.

    A separate reproducible stream indexed by world ensures that changing N
    cannot reassign other persons' draws; increasing M retains all earlier worlds.
    """
    n, m = left.shape
    total = np.zeros((n, len(patterns)), float)
    max_draw_bytes, prefix_hash = 0, None
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        random = np.stack([np.random.default_rng(np.random.SeedSequence([seed, world])).random((n, m))
                           for world in range(start, stop)], axis=0)
        if start == 0:
            prefix_hash = hashlib.sha256(random[:16, :min(n, 1000)].tobytes()).hexdigest()
        worlds = left[None, :, :] + (right - left)[None, :, :] * random
        max_draw_bytes = max(max_draw_bytes, random.nbytes + worlds.nbytes)
        for k, (a, b) in enumerate(patterns):
            total[:, k] += np.sum(worlds[:, :, a] < worlds[:, :, b], axis=0)
    return total / draws, max_draw_bytes, prefix_hash


def sampling_scaling():
    SAMPLING.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    budgets, patterns = [16, 64, 256, 1024], list(permutations(range(4), 2))
    rows, resources, data_manifest = [], [], []
    for si, scenario in enumerate(["null", "chain", "overlap"]):
        data_seed, world_seed = 930031 + si, 940031
        data = simulate_domain(10000, data_seed, scenario, domain=0)
        allids = data["persons"].person_id.to_numpy()
        logs = data["interval_logs"]
        lf = logs.pivot(index="person_id", columns="event_type", values="L").reindex(index=allids, columns=EVENT_TYPES).to_numpy(float)
        rt = logs.pivot(index="person_id", columns="event_type", values="R").reindex(index=allids, columns=EVENT_TYPES).to_numpy(float)
        for n in [1000, 10000]:
            left, right, exact = lf[:n], rt[:n], data["exact_times"][:n]
            uniform, recorded_truth, complete_truth, present = [], [], [], []
            for a, b in patterns:
                probability, have = analytic_uniform_order(left[:, a], right[:, a], left[:, b], right[:, b])
                truth = chain_truth(exact, (EVENT_TYPES[a], EVENT_TYPES[b]), EVENT_TYPES)
                uniform.append(probability)
                recorded_truth.append(truth * have)
                complete_truth.append(truth)
                present.append(have)
            uniform = np.column_stack(uniform)
            recorded_truth = np.column_stack(recorded_truth)
            complete_truth = np.column_stack(complete_truth)
            present = np.column_stack(present)
            data_manifest.append({"scenario": scenario, "n": n, "data_seed": data_seed, "world_seed": world_seed,
                                  "exact_time_sha256": hashlib.sha256(exact.tobytes()).hexdigest(),
                                  "interval_sha256": hashlib.sha256(left.tobytes() + right.tobytes()).hexdigest(),
                                  "candidate_count": len(patterns), "persons": "same nested fixed synthetic participants across all budgets"})
            prefix_hashes = []
            for budget in budgets:
                tracemalloc.start()
                tick, cpu = time.perf_counter(), time.process_time()
                feature, buffer_bytes, prefix_hash = sampled_features(left, right, patterns, budget, world_seed)
                seconds, cpu_seconds = time.perf_counter() - tick, time.process_time() - cpu
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                prefix_hashes.append(prefix_hash)
                support = feature.mean(axis=0)
                oracle_uniform = uniform.mean(axis=0)
                oracle_recorded = recorded_truth.mean(axis=0)
                oracle_complete = complete_truth.mean(axis=0)
                resources.append({"scenario": scenario, "n": n, "draws": budget, "world_seed": world_seed,
                                  "seconds": seconds, "cpu_seconds": cpu_seconds,
                                  "feature_bytes": feature.nbytes, "max_live_draw_arrays_bytes": buffer_bytes,
                                  "python_tracemalloc_peak_bytes": peak, "rss_bytes_after": psutil.Process().memory_info().rss,
                                  "prefix_uniform_sha256": prefix_hash, "chunk_worlds": 64,
                                  "MAE_support_uniform_oracle": float(np.mean(np.abs(support - oracle_uniform))),
                                  "max_abs_support_uniform_oracle": float(np.max(np.abs(support - oracle_uniform))),
                                  "RMSE_features_uniform_oracle_recorded_pairs": float(np.sqrt(np.mean((feature[present] - uniform[present])**2))),
                                  "MAE_support_recorded_actual_truth": float(np.mean(np.abs(support - oracle_recorded))),
                                  "MAE_support_all_actual_truth": float(np.mean(np.abs(support - oracle_complete)))})
                for k, (a, b) in enumerate(patterns):
                    rows.append({"scenario": scenario, "n": n, "draws": budget, "pattern": EVENT_TYPES[a] + ">" + EVENT_TYPES[b],
                                 "data_seed": data_seed, "world_seed": world_seed,
                                 "recorded_pairs": int(present[:, k].sum()),
                                 "MC_uniform_support": support[k], "analytic_uniform_support": oracle_uniform[k],
                                 "actual_recorded_chain_support": oracle_recorded[k], "actual_all_chain_support": oracle_complete[k],
                                 "MC_error_vs_analytic_uniform": support[k] - oracle_uniform[k],
                                 "error_vs_actual_recorded_truth": support[k] - oracle_recorded[k],
                                 "uniform_model_discrepancy_vs_actual_truth": oracle_uniform[k] - oracle_recorded[k]})
                print(json.dumps({"sampling_scenario": scenario, "n": n, "draws": budget, "seconds": round(seconds, 3)}), flush=True)
            assert len(set(prefix_hashes)) == 1, "MC budgets did not share identical first 16 worlds"
    # Prefix hashes must also agree across N and scenarios; latent data differ,
    # but the common random uniform worlds have the same first-1000 prefix.
    assert len(set(r["prefix_uniform_sha256"] for r in resources)) == 1
    pd.DataFrame(rows).to_csv(SAMPLING / "support_error_long.csv", index=False)
    pd.DataFrame(resources).to_csv(SAMPLING / "sampling_budget_resources.csv", index=False)
    (SAMPLING / "fixed_data_manifest.json").write_text(json.dumps(data_manifest, indent=2), encoding="utf-8")
    manifest = {"scope": "Additional fixed-data two-event sensitivity; primary experiments unchanged", "synthetic_only": True,
                "scenarios": ["null", "chain", "overlap"], "N": [1000, 10000], "draw_budgets": budgets,
                "patterns": 12, "same_random_prefixes_verified": True,
                "assumption": "Independent uniforms conditional on recorded (L,R] intervals; not identified actual event-order probabilities",
                "oracle_distinction": "analytic uniform probability checks numerical MC integration; actual continuous recorded/all-chain supports separately check reconstruction discrepancy",
                "feature_memory": "Exact output-array bytes; chunked draw buffers and tracemalloc reported separately; output need not grow with M",
                "no_repetition_SD": "One frozen random-world realization per scenario/N/M; no invented across-seed SD or CI",
                "elapsed_seconds": time.perf_counter() - started}
    (SAMPLING / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def evaluation_calibration(repetitions=200, bootstrap=1000, households=1000):
    CALIBRATION.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    p, rho, m = .3, .25, 2
    concentration = 1 / rho - 1
    alpha, beta = p * concentration, (1 - p) * concentration
    analytic_se = .2 * np.sqrt(p * (1 - p) * (1 + (m - 1) * rho) / (m * households))
    families = np.repeat([f"HH{j:04d}" for j in range(households)], m)
    rows, agreement = [], []
    for repeat in range(repetitions):
        seed = 950031 + repeat
        rng = np.random.default_rng(seed)
        latent = rng.beta(alpha, beta, households)
        y = rng.binomial(1, latent[:, None], size=(households, m))
        household_y = y.mean(axis=1)
        # Actual household resampling, not person-level or seed/fold resampling.
        boot_rng = np.random.default_rng(960031 + repeat)
        draws = boot_rng.integers(0, households, size=(bootstrap, households))
        boot_mean_y = household_y[draws].mean(axis=1)
        ylo, yhi = np.quantile(boot_mean_y, [.025, .975])
        for label, unordered, partial, truth in [("fixed_0.3_vs_0.4", .3, .4, -.01), ("null_0.25_vs_0.35", .25, .35, 0.)]:
            person_d = (unordered - y.ravel())**2 - (partial - y.ravel())**2
            delta, cluster_se, normal_p, G = clustered_mean_test(person_d, families)
            assert G == households
            intercept, gradient = unordered**2 - partial**2, 2 * (partial - unordered)
            boot_delta = intercept + gradient * boot_mean_y
            # Both contrasts are the same increasing affine transform of the
            # household mean. Decide closed-endpoint coverage on that common
            # scale, avoiding contradictory results from round-off at -0.01/0.
            # Twelve decimals remove only floating noise: this fixed H=1000,
            # m=2, B=1000 design has score endpoints on a 0.0000025 grid.
            lo, hi = np.round(intercept + gradient * np.array([ylo, yhi]), 12)
            boot_se = boot_delta.std(ddof=1)
            if repeat == 0:
                # Same sampled family draws through the existing production
                # utility must equal the direct cluster-mean implementation.
                ref_rng = np.random.default_rng(960031)
                for b in range(10):
                    w = family_resample_weights(families, ref_rng)
                    reference = np.average(person_d, weights=w)
                    agreement.append({"scenario": label, "draw": b, "production_weighted_delta": reference,
                                      "direct_cluster_delta": boot_delta[b], "absolute_difference": abs(reference - boot_delta[b])})
                    assert abs(reference - boot_delta[b]) < 1e-12
            rows.append({"scenario": label, "repeat": repeat, "seed": seed, "households": households, "persons": households * m,
                         "members_per_household": m, "bootstrap_repetitions": bootstrap, "population_mean_Y": p,
                         "beta_alpha": alpha, "beta_beta": beta, "population_ICC": rho,
                         "empirical_mean_Y": float(y.mean()), "empirical_within_household_correlation": float(np.corrcoef(y[:, 0], y[:, 1])[0, 1]),
                         "prediction_unordered": unordered, "prediction_partial": partial, "true_population_improvement": truth,
                         "estimated_improvement": delta, "analytic_sampling_SE": analytic_se, "cluster_normal_SE": cluster_se,
                         "household_bootstrap_SE": boot_se, "CI_low": lo, "CI_high": hi,
                         "mean_Y_CI_low": ylo, "mean_Y_CI_high": yhi,
                         "CI_covers_truth": bool(ylo <= p <= yhi),
                         "cluster_normal_p": normal_p, "null_typeI_reject_0.05": bool(normal_p < .05) if truth == 0 else np.nan})
        if (repeat + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(CALIBRATION / "replicate_long.csv", index=False)
            print(json.dumps({"fixed_prediction_calibration": repeat + 1, "of": repetitions,
                              "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(CALIBRATION / "replicate_long.csv", index=False)
    pd.DataFrame(agreement).to_csv(CALIBRATION / "bootstrap_implementation_agreement.csv", index=False)
    summary = []
    for scenario, g in raw.groupby("scenario"):
        coverage = float(g.CI_covers_truth.mean())
        null = float(g.true_population_improvement.iloc[0]) == 0
        typeI = float(g["null_typeI_reject_0.05"].mean()) if null else np.nan
        summary.append({"scenario": scenario, "independent_simulation_repetitions": len(g), "families_per_repetition": households,
                        "persons_per_repetition": households * m, "bootstrap_draws_per_repetition": bootstrap,
                        "true_population_improvement": g.true_population_improvement.iloc[0],
                        "mean_estimated_improvement": g.estimated_improvement.mean(),
                        "bias": g.estimated_improvement.mean() - g.true_population_improvement.iloc[0],
                        "replication_SD_of_estimate": g.estimated_improvement.std(ddof=1),
                        "MCSE_of_mean_estimate": g.estimated_improvement.std(ddof=1) / np.sqrt(len(g)),
                        "analytic_sampling_SE": analytic_se, "mean_cluster_normal_SE": g.cluster_normal_SE.mean(),
                        "mean_household_bootstrap_SE": g.household_bootstrap_SE.mean(),
                        "mean_CI_width": (g.CI_high - g.CI_low).mean(),
                        "CI_coverage": coverage, "coverage_MCSE": np.sqrt(coverage * (1 - coverage) / len(g)),
                        "null_cluster_normal_typeI_0.05": typeI,
                        "typeI_MCSE": np.sqrt(typeI * (1 - typeI) / len(g)) if null else np.nan,
                        "mean_empirical_Y": g.empirical_mean_Y.mean(),
                        "mean_empirical_within_household_correlation": g.empirical_within_household_correlation.mean()})
    pd.DataFrame(summary).to_csv(CALIBRATION / "calibration_summary.csv", index=False)
    manifest = {"scope": "Exact observed binary Y, fixed predictions, independent equal-size households; conditional-score CI/test calibration only",
                "not_validated": ["latent three-state Brier", "interval-censored endpoint evaluation", "informative observation correction", "full-pipeline refitting CI", "unequal-size or survey-PSU clusters"],
                "data": "Beta-Bernoulli: per household probability Beta(0.9,2.1), two conditional Bernoulli members; E[Y]=0.3, ICC=0.25",
                "independent_repetitions": repetitions, "households": households, "members_per_household": m, "bootstrap": bootstrap,
                "CI": "95% empirical household-bootstrap percentile interval for mean Brier improvement U-P",
                "numerical_boundary_rule": "Coverage decided on common Bernoulli mean scale; score endpoints rounded to 12 decimals to remove only floating-point noise. Initial uncorrected run retained in first_run_fp_boundary; identical seeds/data, no selection.",
                "analytic_true_improvement": {"fixed_0.3_vs_0.4": -.01, "null_0.25_vs_0.35": 0},
                "analytic_SE": analytic_se, "production_utility_equivalence_verified": True,
                "test": "Existing household cluster-normal two-sided test at alpha=.05 under the null contrast only",
                "multiplicity": "One prespecified null test in each independent synthetic dataset. Repetitions estimate operating characteristics, not a family of real-data discoveries; no incompatible Holm/BH calculation added.",
                "pairing": "Both fixed contrasts evaluated on the SAME generated families and bootstrap draws; their coverage results are not independent evidence",
                "uncertainty": "Replication SD, within-dataset bootstrap/cluster SE and Monte Carlo SE reported separately",
                "primary_results_changed": False, "nonconvergence_repaired": False,
                "elapsed_seconds": time.perf_counter() - started}
    (CALIBRATION / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "sampling", "calibration"], default="all")
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--households", type=int, default=1000)
    args = parser.parse_args()
    started, cpu = time.perf_counter(), time.process_time()
    results = {}
    if args.mode in ["all", "sampling"]:
        results["sampling"] = sampling_scaling()
    if args.mode in ["all", "calibration"]:
        results["calibration"] = evaluation_calibration(args.repetitions, args.bootstrap, args.households)
    mem = psutil.Process().memory_info()
    resource = {"command_args": vars(args), "elapsed_seconds": time.perf_counter() - started,
                "cpu_seconds": time.process_time() - cpu, "rss_bytes_after": mem.rss,
                "peak_working_set_bytes": getattr(mem, "peak_wset", mem.rss),
                "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(),
                "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(), "RAM_bytes": psutil.virtual_memory().total,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "primary_main_and_QA_unchanged": True, "results": results}
    for out in ([SAMPLING] if args.mode == "sampling" else [CALIBRATION] if args.mode == "calibration" else [SAMPLING, CALIBRATION]):
        (out / "run_resources.json").write_text(json.dumps(resource, indent=2), encoding="utf-8")
    print(json.dumps({"mode": args.mode, "elapsed_seconds": resource["elapsed_seconds"],
                      "peak_working_set_MB": resource["peak_working_set_bytes"] / 2**20}), flush=True)


if __name__ == "__main__":
    main()
