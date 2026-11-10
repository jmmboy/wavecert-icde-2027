"""Reproducible structural validation; every domain in this file is synthetic.

The long tables compare observed-event completions with true continuous event
times.  True-life-chain recovery is a different target from validity of logical
bounds for recorded events.  No predictive model or association FDR is claimed.
"""
import bootstrap_env  # noqa: F401
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from itertools import permutations
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
import pandas as pd
import psutil
from interval_core import batch_chain_features, chain_truth, chain_bounds
from sim_data import simulate_domain, EVENT_TYPES

BASE = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = ["null", "chain", "overlap", "irregular", "informative_dropout",
                     "informative_visits", "hidden_impairment_death", "recovery_shift"]


def ratio(a, b):
    return float(a / b) if b else np.nan


def metrics_for_domain(data, scenario, repeat, stage, seed, chain_length=2):
    ids = data["persons"].person_id
    patterns = list(permutations(EVENT_TYPES, chain_length))
    logs = data["interval_logs"]
    f = batch_chain_features(logs, ids, patterns)
    left = logs.pivot(index="person_id", columns="event_type", values="L").reindex(index=ids, columns=EVENT_TYPES).to_numpy(float)
    right = logs.pivot(index="person_id", columns="event_type", values="R").reindex(index=ids, columns=EVENT_TYPES).to_numpy(float)
    exact = data["exact_times"]
    names = {e: j for j, e in enumerate(EVENT_TYPES)}
    allrows = []
    for pattern in patterns:
        key = ">".join(pattern)
        jj = [names[e] for e in pattern]
        la, ra, ta = left[:, jj], right[:, jj], exact[:, jj]
        present = np.isfinite(la).all(axis=1)
        intervals_contain_truth = np.all((ta > la) & (ta <= ra), axis=1) & present
        truth = chain_truth(exact, pattern, EVENT_TYPES)
        truth_recorded = truth.astype(bool) & present
        lower = f[key + "__lower"].to_numpy()
        upper = f[key + "__upper"].to_numpy()
        ambiguous = (upper - lower).astype(bool) & present
        recgood = (lower <= truth_recorded) & (truth_recorded <= upper)
        row = {"stage": stage, "scenario": scenario, "repeat": repeat, "seed": seed,
               "domain": data["meta"]["label"], "pattern": key, "chain_length": chain_length,
               "n": len(ids), "recorded_complete_pattern_n": int(present.sum()),
               "valid_recorded_pattern_n": int(intervals_contain_truth.sum()),
               "true_chain_n": int(truth.sum()), "recorded_truth_chain_n": int(truth_recorded.sum())}
        common = {
            "bounds_coverage_recorded_valid": ratio(recgood[intervals_contain_truth].sum(), intervals_contain_truth.sum()),
            "bounds_coverage_recorded_all": ratio(recgood[present].sum(), present.sum()),
            "recorded_interval_truth_containment": ratio(intervals_contain_truth.sum(), present.sum()),
            "support_bound_coverage_recorded": float(lower.mean() <= truth_recorded.mean() <= upper.mean()),
            "support_bound_coverage_all_true_diagnostic": float(lower.mean() <= truth.mean() <= upper.mean()),
            "support_lower": float(lower.mean()), "support_upper": float(upper.mean()),
            "support_width": float((upper - lower).mean()),
            "ambiguity_among_recorded": ratio(ambiguous.sum(), present.sum()),
            "true_support": float(truth.mean()),
            "hidden_illness_death_fraction": float(data["outcomes"].hidden_illness_before_death.mean()),
            "known_state3_fraction": float(data["outcomes"].observed_state3.notna().mean()),
            "observed_A_interval_width": float((data["outcomes"].observed_y_upper - data["outcomes"].observed_y_lower).mean()),
        }
        methods = {"partial_certain": lower.astype(bool), "partial_possible": upper.astype(bool)}
        for method, points in {"midpoint": (la + ra) / 2, "right_endpoint": ra, "strict_wave": ra}.items():
            # Same endpoint / same interview-wave ties are abstentions.  No
            # alphabetical tie-breaking is passed off as an observed order.
            methods[method] = present & np.all(np.diff(points, axis=1) > 0, axis=1)
        for method, pred in methods.items():
            tp, fp = int((pred & truth.astype(bool)).sum()), int((pred & ~truth.astype(bool)).sum())
            calls, actual = int(pred.sum()), int(truth.sum())
            precision, recall = ratio(tp, calls), ratio(tp, actual)
            f1 = ratio(2 * tp, calls + actual)
            # An order asserted in a logically ambiguous completion is a
            # pseudocertainty, even if the simulated truth happens to agree.
            unsupported = int((pred & ambiguous).sum())
            record = dict(row, method=method, **common,
                          called_n=calls, true_positive_n=tp, false_positive_n=fp,
                          precision=precision, recall=recall, f1=f1,
                          error_among_calls=ratio(fp, calls),
                          false_order_fraction_recorded=ratio(fp, present.sum()),
                          unsupported_order_fraction_calls=ratio(unsupported, calls),
                          unsupported_order_fraction_recorded=ratio(unsupported, present.sum()))
            # Rejection is a relation-level metric; partial_possible is not a
            # point-order rule and is explicitly excluded from this quantity.
            if method == "partial_possible":
                record["order_reject_fraction_recorded"] = np.nan
            else:
                if method == "partial_certain":
                    decided = np.any([f[">".join(p) + "__lower"].to_numpy().astype(bool)
                                      for p in permutations(pattern)], axis=0)
                else:
                    points = (la + ra) / 2 if method == "midpoint" else ra
                    decided = present & np.all(np.diff(np.sort(points, axis=1), axis=1) > 0, axis=1)
                record["order_reject_fraction_recorded"] = ratio((present & ~decided).sum(), present.sum())
            allrows.append(record)
    return allrows


def run_one(task):
    scenario, repeat, stage, n, chain_length, seedbase = task
    started = time.perf_counter()
    process = psutil.Process()
    rows, records = [], []
    try:
        for domain in range(3):
            seed = int(seedbase + repeat * 10 + domain)
            t0 = time.perf_counter()
            data = simulate_domain(n, seed, scenario=scenario, domain=domain)
            afterdata = time.perf_counter()
            rows += metrics_for_domain(data, scenario, repeat, stage, seed, chain_length)
            records.append({"stage": stage, "scenario": scenario, "repeat": repeat,
                            "domain": data["meta"]["label"], "seed": seed, "n": n,
                            "simulation_seconds": afterdata - t0,
                            "matching_seconds": time.perf_counter() - afterdata,
                            "process_rss_bytes_after": process.memory_info().rss,
                            "status": "ok", "chain_length": chain_length})
        return {"rows": rows, "resources": records, "seconds": time.perf_counter() - started, "failure": None}
    except Exception as exc:
        return {"rows": [], "resources": records, "seconds": time.perf_counter() - started,
                "failure": {"stage": stage, "scenario": scenario, "repeat": repeat,
                            "seedbase": seedbase, "error": repr(exc), "traceback": traceback.format_exc()}}


def summarize(raw):
    # First combine the three observation domains within the SAME replication.
    # Monte Carlo SD/SE then uses independent simulation replications, never
    # treats patterns, domains or seeds as independent real participants.
    idcols = ["stage", "scenario", "repeat", "pattern", "chain_length", "method"]
    metrics = [c for c in raw.select_dtypes("number") if c not in ["repeat", "seed", "chain_length"]]
    grouped = raw.groupby(idcols, dropna=False)[metrics].mean().reset_index()
    rows = []
    for keys, g in grouped.groupby(["stage", "scenario", "pattern", "chain_length", "method"]):
        base = dict(zip(["stage", "scenario", "pattern", "chain_length", "method"], keys))
        for metric in metrics:
            x = g[metric].dropna().to_numpy(float)
            sd = float(x.std(ddof=1)) if len(x) > 1 else np.nan
            rows.append(dict(base, metric=metric, mean=float(x.mean()) if len(x) else np.nan,
                             sd_replications=sd, mcse=sd / np.sqrt(len(x)) if len(x) else np.nan,
                             n_replications=len(x)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--stage", default="pilot")
    parser.add_argument("--chain-length", type=int, choices=[2, 3], default=2)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    args = parser.parse_args()
    out = BASE / "simulation" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    rawpath, resourcepath, failpath = out / "replicate_long.csv", out / "resources.csv", out / "failures.jsonl"
    tasks = []
    existing = pd.read_csv(resourcepath, keep_default_na=False, na_values=[""]) if resourcepath.exists() else pd.DataFrame()
    done = set(zip(existing.scenario, existing["repeat"])) if len(existing) else set()
    seedoffset = 1_000_000 if args.stage == "pilot" else (3_000_000 if args.chain_length == 3 else 2_000_000)
    for si, scenario in enumerate(args.scenarios):
        for repeat in range(args.repetitions):
            if (scenario, repeat) not in done:
                tasks.append((scenario, repeat, args.stage, args.n, args.chain_length, seedoffset + si * 10000))
    manifest = {"args": vars(args), "synthetic_only": True, "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                "platform": platform.platform(), "cpu": platform.processor(),
                "logical_cpu_count": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
                "threads": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
                "seed_formula": f"{seedoffset} + scenario_index*10000 + repeat*10 + domain_index",
                "wave_endpoint_alias": "strict_wave and right_endpoint identical for this first-report interview simulator",
                "three_chain_rejection": "a triple is decided if one of its six full permutations is certain; all 24 type-ordered patterns evaluated",
                "scope": "Structural simulation; no empirical cohort performance and no association significance/FDR"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started, peak_rss, lastprint, completed, failures = time.perf_counter(), 0, time.perf_counter(), 0, 0
    parent = psutil.Process()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_one, task): task for task in tasks}
        while pending:
            finished, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
            rss = parent.memory_info().rss
            for child in parent.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    pass
            peak_rss = max(peak_rss, rss)
            elapsed = time.perf_counter() - started
            if elapsed > args.timeout_seconds:
                for f in pending:
                    f.cancel()
                raise TimeoutError(f"Simulation exceeded {args.timeout_seconds}s; completed files are resumable")
            for future in finished:
                task = pending.pop(future)
                result = future.result()
                if result["failure"]:
                    failures += 1
                    with failpath.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result["failure"]) + "\n")
                else:
                    df = pd.DataFrame(result["rows"])
                    df.to_csv(rawpath, index=False, mode="a", header=not rawpath.exists())
                    rr = pd.DataFrame(result["resources"])
                    rr.to_csv(resourcepath, index=False, mode="a", header=not resourcepath.exists())
                completed += 1
            if time.perf_counter() - lastprint >= 15 or not pending:
                print(json.dumps({"stage": args.stage, "completed_tasks": completed, "total_new_tasks": len(tasks),
                                  "failures": failures, "elapsed_seconds": round(elapsed, 2),
                                  "peak_parent_children_rss_mb": round(peak_rss / 1024**2, 1)}), flush=True)
                lastprint = time.perf_counter()
    if rawpath.exists():
        raw = pd.read_csv(rawpath, keep_default_na=False, na_values=[""])
        summarize(raw).to_csv(out / "scenario_summary_mcse.csv", index=False)
        # Pooled pattern averages aid concise tables without hiding individual
        # pattern outputs in the saved long summary.
        numeric = [c for c in raw.select_dtypes("number") if c not in ["repeat", "seed", "chain_length"]]
        pooled = raw.groupby(["stage", "scenario", "repeat", "domain", "chain_length", "method"])[numeric].mean().reset_index()
        pooled["pattern"] = "all_patterns_macro"
        summarize(pooled).to_csv(out / "scenario_macro_summary_mcse.csv", index=False)
    manifest.update(elapsed_seconds=time.perf_counter() - started, peak_parent_children_rss_bytes=peak_rss,
                    new_tasks_completed=completed, failures=failures,
                    completed_utc=pd.Timestamp.now(tz="UTC").isoformat())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"result_directory": str(out), "elapsed_seconds": manifest["elapsed_seconds"], "failures": failures}), flush=True)


if __name__ == "__main__":
    main()
