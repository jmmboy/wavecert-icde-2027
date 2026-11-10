"""Source-frozen chain interpretation and explicitly exploratory CONF analyses.

Writes only results/interpretation and private/interpretation_cases.json.
All coefficients/importances are model dependence or prediction associations,
never causal effects.  Stratified outcome shuffling is a leakage diagnostic,
not a valid association p-value or a guarantee that AUC becomes exactly .5.
"""
import bootstrap_env  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import psutil
from scipy.optimize import minimize
from scipy.linalg import qr
from scipy.special import expit
from scipy.stats import norm, chi2
import real_benchmark as rb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "interpretation"
EVENTS = ["hypertension", "diabetes", "stroke", "widowhood"]


def save(frame, filename):
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame).to_csv(OUT / filename, index=False)


def canonical(pattern):
    return tuple(sorted(pattern))


def family_key(e, ix):
    return (e.iloc[ix].cohort.astype(str) + "|" + e.iloc[ix].household_id.astype(str)).to_numpy()


def age_group(age):
    a = np.asarray(age, float)
    return np.where(~np.isfinite(a), "unknown", np.where(a < 70, "60s", np.where(a < 80, "70s", "80plus")))


def strata(e, ix, include_bag=False, include_cohort=True):
    x = e.iloc[ix]
    values = [x.baseline_state.astype(str).to_numpy(), age_group(x.age)]
    if include_cohort:
        values.insert(0, x.cohort.astype(str).to_numpy())
    if include_bag:
        for event in EVENTS:
            a = x[event + "_event"].to_numpy(float)
            values.append(np.where(np.isnan(a), "unknown", a.astype(str)))
    key = np.array(["|".join(parts) for parts in zip(*values)])
    return [np.flatnonzero(key == k) for k in np.unique(key)]


def load():
    e, logs, patterns, features, reps, width = rb.load_features()
    paths = sorted((ROOT / "private" / "models").glob("*.pkl"))
    if len(paths) < 3 or not (ROOT / "private" / "real_predictions.csv.gz").exists():
        raise FileNotFoundError("Three real model caches and real_predictions.csv.gz must exist before execution")
    models = {p.stem: pickle.load(p.open("rb")) for p in paths}
    pred = pd.read_csv(ROOT / "private" / "real_predictions.csv.gz", dtype={"person_id": str, "household_id": str})
    for target, bundle in models.items():
        train, test = bundle["train"], bundle["test"]
        assert np.all(e.iloc[train].cohort != target)
        assert np.all(e.iloc[train].confirmation == 0)
        assert np.all(e.iloc[test].cohort == target)
        assert np.all(e.iloc[test].confirmation == 1)
        assert not set(family_key(e, train)) & set(family_key(e, test))
        X, labels = rb.design(e, features, reps, width, patterns, bundle["selected"], "PartialOrder")
        assert labels == bundle["PartialOrder"]["labels"], "Model/source feature version differs; do not reinterpret stale caches"
        cached = pred[pred.cohort == target].set_index("person_id").reindex(e.iloc[test].person_id)
        rebuilt = rb.predict(bundle["PartialOrder"]["model"], X, e, test)
        if not np.allclose(rebuilt, cached.PartialOrder.to_numpy(), atol=1e-10, rtol=1e-8):
            raise ValueError(f"Rebuilt predictions differ from frozen cache for {target}")
    return e, logs, patterns, features, reps, width, models


def support_table(e, patterns, features, models):
    rows, catalog = [], set()
    for target, bundle in models.items():
        selected = set(bundle["selected"])
        catalog.update(selected)
        groups = [(c, "source_DISC", bundle["train"][e.iloc[bundle["train"]].cohort.to_numpy() == c])
                  for c in sorted(e.iloc[bundle["train"]].cohort.unique())]
        groups.append((target, "target_CONF", bundle["test"]))
        for k, pattern in enumerate(patterns):
            name = ">".join(pattern)
            for cohort, split, ix in groups:
                lower = features.iloc[ix][name + "__lower"].to_numpy(float)
                upper = features.iloc[ix][name + "__upper"].to_numpy(float)
                valid = np.isfinite(lower) & np.isfinite(upper)
                n = int(valid.sum())
                certain, possible = float(np.nansum(lower)), float(np.nansum(upper))
                threshold = max(10, .0025 * n)
                rows.append({"heldout_cohort": target, "cohort": cohort, "split": split,
                             "pattern": name, "canonical_pair": "&".join(canonical(pattern)),
                             "selected_in_source": k in selected, "participants": len(ix),
                             "eligible": n, "unknown_histories": int((~valid).sum()),
                             "certain_count": certain, "possible_count": possible,
                             "support_lower": certain / n if n else np.nan,
                             "support_upper": possible / n if n else np.nan,
                             "ambiguity_width": (possible - certain) / n if n else np.nan,
                             "certain_support_threshold": threshold,
                             "passes_prespecified_support": certain >= threshold,
                             "interpretation": "recorded-event support; low support does not prove cohort-specific absence"})
    save(rows, "chain_support_source_and_confirmation.csv")
    directory = [{"pattern_index": k, "pattern": ">".join(patterns[k]),
                  "selected_for_targets": ";".join(t for t, b in models.items() if k in b["selected"]),
                  "discovery": "source DISC only; union fixed before CONF association analysis"} for k in sorted(catalog)]
    save(directory if directory else [{"status": "not_estimable", "reason": "No pattern passed prespecified source support in two source cohorts; no formal chain directory"}], "frozen_discovery_catalog.csv")
    return sorted(catalog)


def coefficient_table(models):
    rows = []
    for target, bundle in models.items():
        model, labels = bundle["PartialOrder"]["model"], bundle["PartialOrder"]["labels"]
        for j, label in enumerate(labels):
            if not label.endswith("__certain_direction"):
                continue
            for rate, transition in [(0, "healthy_to_impaired"), (2, "impaired_to_healthy")]:
                value = float(model.coef_[rate, j])
                raw = value / float(model.scale_[j])
                rows.append({"heldout_cohort": target, "feature": label, "transition": transition,
                             "standardized_lograte_coefficient": value, "per_raw_unit_lograte_coefficient": raw,
                             "working_model_rate_ratio_per_raw_unit": float(np.exp(np.clip(raw, -50, 50))),
                             "source_only": True, "penalty": bundle["penalty"],
                             "meaning": "ridge CTMC model dependence; conditional working-model rate, not causal effect"})
    save(rows if rows else [{"status": "not_estimable", "reason": "No source-selected canonical direction; primary model contains no chain coefficient"}], "source_chain_coefficients.csv")


def conditional_permutation(e, patterns, f, reps, width, models, repeats=20):
    rows = []
    for target, bundle in models.items():
        test = bundle["test"]
        X, labels = rb.design(e, f, reps, width, patterns, bundle["selected"], "PartialOrder")
        model = bundle["PartialOrder"]["model"]
        p0 = rb.predict(model, X, e, test)
        score0 = rb.metrics(e.iloc[test].y, p0)["brier"]
        groups = strata(e, test, include_bag=True)
        for j, label in enumerate(labels):
            if not label.endswith("__certain_direction"):
                continue
            values = X[test, j]
            usable = []
            for group in groups:
                # Zero = absent/ambiguous/unknown and remains fixed.  Only swap
                # opposing identified directions with the same baseline/bag/age.
                group = group[np.isfinite(values[group]) & (values[group] != 0)]
                if len(group) > 1 and len(np.unique(values[group])) > 1:
                    usable.append(group)
            exchangeable = int(sum(len(g) for g in usable))
            rng = np.random.default_rng(670031 + j)
            for repeat in range(repeats):
                xp = X.copy()
                changed = 0
                for group in usable:
                    donor = rng.permutation(group)
                    xp[test[group], j] = values[donor]
                    changed += int((values[donor] != values[group]).sum())
                p = rb.predict(model, xp, e, test)
                score = rb.metrics(e.iloc[test].y, p)["brier"]
                rows.append({"cohort": target, "feature": label, "repeat": repeat, "n": len(test),
                             "exchangeable_people": exchangeable, "exchangeable_strata": len(usable),
                             "changed_people": changed, "original_brier": score0, "permuted_brier": score,
                             "brier_increase": score - score0,
                             "status": "estimable" if exchangeable else "no_mixed_direction_strata",
                             "scope": "conditional feature importance; baseline ADL, age group and single-event bag fixed; zero direction fixed; no p-value"})
    empty_reason = [{"status": "not_estimable", "reason": "No source-selected direction feature; zero added direction is not a measured zero importance"}]
    save(rows if rows else empty_reason, "conditional_permutation_long.csv")
    if rows:
        summary = pd.DataFrame(rows).groupby(["cohort", "feature", "status"], dropna=False).agg(
            brier_increase_mean=("brier_increase", "mean"), permutation_sd=("brier_increase", "std"),
            repetitions=("repeat", "count"), exchangeable_people=("exchangeable_people", "max"),
            changed_people_mean=("changed_people", "mean")).reset_index()
        save(summary, "conditional_permutation_summary.csv")
    else:
        save(empty_reason, "conditional_permutation_summary.csv")


def case_sample(e, logs, patterns, f, reps, width, models):
    rng = np.random.default_rng(790031)
    private, summary = [], []
    for target, bundle in models.items():
        test = bundle["test"]
        X, _ = rb.design(e, f, reps, width, patterns, bundle["selected"], "PartialOrder")
        p = rb.predict(bundle["PartialOrder"]["model"], X, e, test)
        y = e.iloc[test].y.to_numpy()
        error = (p >= .5).astype(int) != y
        keys = [">".join(patterns[k]) + "__ambiguous" for k in bundle["selected"]]
        amb = np.any(f.iloc[test][keys].fillna(0).to_numpy() > 0, axis=1) if keys else np.zeros(len(test), bool)
        for category, mask in [("correct", ~error), ("incorrect", error), ("ambiguous", amb)]:
            options = np.flatnonzero(mask)
            chosen = rng.choice(options, min(3, len(options)), replace=False) if len(options) else []
            summary.append({"cohort": target, "category": category, "available": len(options),
                            "sampled": len(chosen), "seed": 790031,
                            "public_identifiers": False, "diagnostic_threshold": .5})
            for number, loc in enumerate(chosen, 1):
                idx = int(test[loc])
                person = e.iloc[idx].person_id
                private.append({"case_code": f"CASE_{target}_{category}_{number}", "person_id": str(person),
                                "cohort": target, "category": category, "t0": float(e.iloc[idx].t0),
                                "baseline_state": int(e.iloc[idx].baseline_state), "observed_y": int(y[loc]),
                                "predicted_probability": float(p[loc]), "delta": float(e.iloc[idx].delta),
                                "ambiguous_selected_chain": bool(amb[loc]),
                                "intervals": logs[logs.person_id == person][["event_type", "L", "R"]].to_dict("records")})
    (ROOT / "private" / "interpretation_cases.json").write_text(json.dumps(private, ensure_ascii=False, indent=2), encoding="utf-8")
    save(summary, "case_sampling_summary.csv")


def adjustment(e):
    # Association adjustment is prespecified and uses no outcome to choose
    # controls. Event unknowns remain separate; unknown events are not negatives.
    names = ["age_per_10y", "female", "baseline_state"] + [d + "_event" for d in EVENTS] + [d + "_unknown" for d in EVENTS]
    a = e[["age", "female", "baseline_state"] + [d + "_event" for d in EVENTS] + [d + "_unknown" for d in EVENTS]].to_numpy(float)
    source = e.confirmation.to_numpy() == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(a[source], axis=0)
    med = np.nan_to_num(med, nan=0)
    # Fill unknown single events with 0 AND retain their explicit unknown flag.
    med[3:7] = 0
    missing = ~np.isfinite(a[:, :2])
    a = np.where(np.isfinite(a), a, med)
    a[:, 0] /= 10
    for j, label in enumerate(["age_missing", "female_missing"]):
        if missing[:, j].any():
            a = np.c_[a, missing[:, j].astype(float)]
            names.append(label)
    return a, names


def logistic(X, y, names, families):
    """Unpenalized logistic MLE, household sandwich covariance, explicit failure."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if len(y) < 100 or len(np.unique(y)) < 2:
        return {"status": "not_estimable", "reason": "insufficient sample or outcome classes"}
    # Constant adjustment columns can be removed without using outcome values.
    keep = np.array([name == "intercept" or np.ptp(X[:, j]) > 1e-10 for j, name in enumerate(names)])
    names = [name for name, yes in zip(names, keep) if yes]
    X = X[:, keep]
    if "chain" not in names:
        return {"status": "not_estimable", "reason": "constant exposure"}
    protected = [j for j, name in enumerate(names) if name == "chain" or name.startswith("chain_x_")]
    controls = [j for j in range(len(names)) if j not in protected]
    rank_controls = np.linalg.matrix_rank(X[:, controls])
    if np.linalg.matrix_rank(X) - rank_controls < len(protected):
        return {"status": "not_estimable", "reason": "chain/interaction not identifiable conditional on adjustment span"}
    dropped = []
    if rank_controls < len(controls):
        # Keep the SAME adjustment column space while discarding exact duplicate
        # controls, e.g. a missingness flag identical to a cohort dummy.  Choice
        # uses X only and never removes/regularizes the tested exposure.
        intercept = names.index("intercept")
        other = [j for j in controls if j != intercept]
        centered = X[:, other] - X[:, other].mean(axis=0)
        _, R, piv = qr(centered, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(R))
        rank = int(np.sum(diagonal > (diagonal.max() if len(diagonal) else 0) * 1e-10))
        chosen = [intercept] + [other[j] for j in piv[:rank]] + protected
        dropped = [name for j, name in enumerate(names) if j not in chosen]
        X = X[:, chosen]
        names = [names[j] for j in chosen]
    beta0 = np.zeros(X.shape[1])
    beta0[names.index("intercept")] = np.log(y.mean() / (1 - y.mean()))
    def fun(beta):
        eta = X @ beta
        return float(np.sum(np.logaddexp(0, eta) - y * eta)), X.T @ (expit(eta) - y)
    result = minimize(fun, beta0, jac=True, method="BFGS", options={"maxiter": 300, "gtol": 1e-5})
    p = expit(X @ result.x)
    H = X.T @ ((p * (1 - p))[:, None] * X)
    if (not result.success and np.max(np.abs(result.jac)) > .01) or np.linalg.cond(H) > 1e12 or np.max(np.abs(result.x)) > 40:
        return {"status": "not_estimable", "reason": "MLE nonconvergence/separation/ill-conditioning", "optimizer": str(result.message)}
    unique, inv = np.unique(families, return_inverse=True)
    score = np.zeros((len(unique), X.shape[1]))
    np.add.at(score, inv, X * (y - p)[:, None])
    bread = np.linalg.inv(H)
    correction = len(unique) / max(len(unique) - 1, 1) * (len(y) - 1) / max(len(y) - X.shape[1], 1)
    cov = bread @ (score.T @ score) @ bread * correction
    return {"status": "estimable", "names": names, "beta": result.x, "cov": cov,
            "households": len(unique), "n": len(y), "optimizer": str(result.message),
            "redundant_adjustment_columns_removed": ";".join(dropped)}


def coefficient(fit, name="chain"):
    j = fit["names"].index(name)
    beta, se = float(fit["beta"][j]), float(np.sqrt(max(fit["cov"][j, j], 0)))
    return {"log_odds": beta, "cluster_robust_se": se, "odds_ratio": float(np.exp(np.clip(beta, -50, 50))),
            "ci_low_OR": float(np.exp(np.clip(beta - 1.96 * se, -50, 50))),
            "ci_high_OR": float(np.exp(np.clip(beta + 1.96 * se, -50, 50))),
            "p_value": float(2 * norm.sf(abs(beta / se))) if se > 0 else np.nan}


def adjust_fdr(values, by=False):
    p = np.asarray(values, float)
    valid = np.isfinite(p)
    out = np.full(len(p), np.nan)
    order = np.argsort(p[valid])
    q = p[valid][order] * valid.sum() / np.arange(1, valid.sum() + 1)
    if by:
        q *= np.sum(1 / np.arange(1, valid.sum() + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty(len(q))
    restored[order] = np.minimum(q, 1)
    out[valid] = restored
    return out


def association_catalog(e, patterns, f, models, catalog):
    if not catalog:
        save([{"status": "not_estimable", "reason": "Source DISC selected no patterns; CONF formal association/interaction/FDR family is empty; no p/q values fabricated"}],
             "confirmation_associations_and_interactions.csv")
        return
    adjust, adj_names = adjustment(e)
    cohorts = sorted(models)
    conf = np.flatnonzero((e.confirmation == 1) & (e.response == 1) & e.cohort.isin(cohorts))
    rows = []
    for k in catalog:
        pattern = ">".join(patterns[k])
        exposure = f[pattern + "__lower"].to_numpy(float)
        valid = np.isfinite(exposure[conf])
        ixall = conf[valid]
        for cohort in cohorts + ["pooled"]:
            ix = ixall if cohort == "pooled" else ixall[e.iloc[ixall].cohort.to_numpy() == cohort]
            z, y = exposure[ix], e.iloc[ix].y.to_numpy(float)
            exposed = int((z == 1).sum())
            exposed_events = int(y[z == 1].sum())
            base = {"pattern": pattern, "cohort": cohort, "analysis": "adjusted_chain_association",
                    "n": len(ix), "certain_exposed": exposed, "exposed_outcomes": exposed_events,
                    "scope": "CONF-only exploratory association; recorded certain chain vs other eligible histories; not causal"}
            if exposed < 10 or exposed_events < 5 or exposed - exposed_events < 5:
                rows.append(dict(base, status="not_estimable", reason="fewer than 10 exposed or <5 outcomes/nonoutcomes in exposed", p_value=np.nan))
                if cohort == "pooled":
                    rows.append(dict(base, analysis="cohort_by_chain_interaction", status="not_estimable",
                                     reason="pooled chain itself has insufficient exposure/outcome support", p_value=np.nan))
                continue
            X = np.c_[np.ones(len(ix)), z, adjust[ix]]
            names = ["intercept", "chain"] + adj_names
            if cohort == "pooled":
                for c in cohorts[1:]:
                    X = np.c_[X, (e.iloc[ix].cohort.to_numpy() == c).astype(float)]
                    names.append("cohort_" + c)
            fit = logistic(X, y, names, family_key(e, ix))
            if fit["status"] != "estimable":
                rows.append(dict(base, **{k: v for k, v in fit.items() if k in ["status", "reason", "optimizer"]}, p_value=np.nan))
                if cohort == "pooled":
                    rows.append(dict(base, analysis="cohort_by_chain_interaction", status="not_estimable",
                                     reason="pooled adjusted association is not estimable", p_value=np.nan))
                continue
            rows.append(dict(base, status="estimable", households=fit["households"],
                             redundant_adjustment_columns_removed=fit["redundant_adjustment_columns_removed"], **coefficient(fit)))
            if cohort == "pooled":
                # Interaction assessed jointly, not by comparing significance
                # of separate cohort coefficients. Household-robust Wald test.
                inter = []
                for c in cohorts[1:]:
                    label = "chain_x_" + c
                    X = np.c_[X, z * (e.iloc[ix].cohort.to_numpy() == c)]
                    names.append(label)
                    inter.append(label)
                full = logistic(X, y, names, family_key(e, ix))
                ibase = dict(base, analysis="cohort_by_chain_interaction", certain_exposed=exposed)
                if full["status"] != "estimable" or any(v not in full.get("names", []) for v in inter):
                    rows.append(dict(ibase, status="not_estimable", reason="sparse/nonidentifiable interaction design", p_value=np.nan))
                else:
                    j = [full["names"].index(v) for v in inter]
                    b, cov = full["beta"][j], full["cov"][np.ix_(j, j)]
                    rank = int(np.linalg.matrix_rank(cov))
                    if rank < len(j):
                        rows.append(dict(ibase, status="not_estimable", reason="rank-deficient interaction covariance", p_value=np.nan))
                    else:
                        wald = float(b @ np.linalg.solve(cov, b))
                        rows.append(dict(ibase, status="estimable", wald_chi2=wald, df=len(j),
                                         p_value=float(chi2.sf(wald, len(j))),
                                         reason="limited power with only three cohorts; non-significant interaction is not equivalence"))
    result = pd.DataFrame(rows)
    if len(result):
        result["fdr_family"] = np.where(result.cohort == "pooled", "pooled_association_and_interaction", "within_cohort_associations")
        result["q_BH"] = np.nan
        result["q_BY_conservative"] = np.nan
        result["planned_family_size"] = 0
        for family, ix in result.groupby("fdr_family").groups.items():
            original = result.loc[ix, "p_value"].to_numpy(float)
            # Keep the DISC-defined full test family. A non-estimable planned
            # hypothesis occupies a conservative p=1 slot for correction only;
            # its reported p/q remain missing, never invented inferential data.
            correction_values = np.where(np.isfinite(original), original, 1.)
            qbh = adjust_fdr(correction_values)
            qby = adjust_fdr(correction_values, by=True)
            qbh[~np.isfinite(original)] = np.nan
            qby[~np.isfinite(original)] = np.nan
            result.loc[ix, "q_BH"] = qbh
            result.loc[ix, "q_BY_conservative"] = qby
            result.loc[ix, "planned_family_size"] = len(ix)
        result["BH_assumption"] = "independence or appropriate positive dependence not established; BY also reported; exploratory only"
    save(result, "confirmation_associations_and_interactions.csv")


def source_outcome_shuffle(e, patterns, f, reps, width, models, repeats=20):
    rows = []
    for target, bundle in models.items():
        train, test = bundle["train"], bundle["test"]
        groups = strata(e, train, include_bag=False, include_cohort=True)
        rng = np.random.default_rng(830031)
        actual_target = e.iloc[test].y.to_numpy(float).copy()
        Xp, _ = rb.design(e, f, reps, width, patterns, bundle["selected"], "PartialOrder")
        Xu, _ = rb.design(e, f, reps, width, patterns, bundle["selected"], "Unordered")
        original = rb.metrics(actual_target, rb.predict(bundle["PartialOrder"]["model"], Xp, e, test))
        # Baseline control retains age and the CTMC's observed starting state.
        Xbase = rb.attach_missing(e[["age"]].to_numpy(float))
        baseline = rb.fit_model(Xbase, e, train, bundle["penalty"])
        base_score = rb.metrics(actual_target, rb.predict(baseline, Xbase, e, test))
        for repeat in range(repeats):
            shuffled = e.copy()
            ys = e.iloc[train].y.to_numpy(float).copy()
            for group in groups:
                ys[group] = ys[rng.permutation(group)]
            shuffled.loc[train, "y"] = ys
            assert np.array_equal(shuffled.iloc[test].y.to_numpy(), actual_target)
            started = time.perf_counter()
            scores, converged = {}, {}
            identical_design = Xp.shape == Xu.shape and np.array_equal(Xp, Xu, equal_nan=True)
            for name, X in [("Unordered", Xu), ("PartialOrder", Xp)]:
                if name == "PartialOrder" and identical_design:
                    scores[name] = dict(scores["Unordered"])
                    converged[name] = converged["Unordered"]
                    continue
                model = rb.fit_model(X, shuffled, train, bundle["penalty"])
                prediction = rb.predict(model, X, shuffled, test)
                scores[name] = rb.metrics(actual_target, prediction)
                converged[name] = model.converged_
            rows.append({"cohort": target, "repeat": repeat, "seed": 830031,
                         "source_n": len(train), "target_n": len(test),
                         "strata": len(groups), "source_label_changed_fraction": float((ys != e.iloc[train].y.to_numpy()).mean()),
                         "original_partial_brier": original["brier"], "baseline_age_state_brier": base_score["brier"],
                         "permuted_partial_brier": scores["PartialOrder"]["brier"],
                         "permuted_unordered_brier": scores["Unordered"]["brier"],
                         "permuted_partial_auc": scores["PartialOrder"]["auc"],
                         "permuted_chain_improvement": scores["Unordered"]["brier"] - scores["PartialOrder"]["brier"],
                         "partial_degradation": scores["PartialOrder"]["brier"] - original["brier"],
                         "partial_excess_over_age_state_baseline": scores["PartialOrder"]["brier"] - base_score["brier"],
                         "identical_design_fit_reused": identical_design,
                         "converged": all(converged.values()), "seconds": time.perf_counter() - started,
                         "scope": "leakage diagnostic only; shuffle source outcomes within cohort/baseline state/age group; target fixed; no p-value; frozen source penalty"})
            save(rows, "source_outcome_shuffle_long.csv")
            print(json.dumps({"diagnostic": "source_outcome_shuffle", "cohort": target, "repeat": repeat + 1, "of": repeats}), flush=True)
    if rows:
        d = pd.DataFrame(rows)
        cols = ["permuted_partial_brier", "permuted_unordered_brier", "permuted_chain_improvement", "partial_degradation", "partial_excess_over_age_state_baseline"]
        summary = []
        for c, g in d.groupby("cohort"):
            for col in cols:
                summary.append({"cohort": c, "metric": col, "mean": g[col].mean(), "permutation_sd": g[col].std(ddof=1),
                                "repetitions": len(g), "all_converged": bool(g.converged.all()), "formal_p_value": False})
        save(summary, "source_outcome_shuffle_summary.csv")


def selection_frequency(patterns):
    path = ROOT / "results" / "statistics" / "full_pipeline_bootstrap.csv"
    if not path.exists():
        save([{"status": "pending", "reason": "full-pipeline bootstrap file not yet available; rerun --mode selection later"}], "bootstrap_selection_frequency.csv")
        return
    raw = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if "selected_patterns" not in raw:
        save([{"status": "not_estimable", "reason": "bootstrap records contain chain counts but not chain identifiers"}], "bootstrap_selection_frequency.csv")
        return
    rows = []
    for cohort, g in raw[raw.cohort != "macro"].groupby("cohort"):
        for pattern in patterns:
            name = ">".join(pattern)
            selected = [name in (str(value).split(";") if pd.notna(value) else []) for value in g.selected_patterns]
            rows.append({"cohort": cohort, "pattern": name, "selected_repetitions": int(sum(selected)),
                         "available_repetitions": len(selected), "selection_frequency": float(np.mean(selected)),
                         "converged_repetitions": int(g.converged.sum()),
                         "status": "complete" if len(g) >= 200 else "partial_bootstrap_file"})
    save(rows, "bootstrap_selection_frequency.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "support", "negative", "association", "selection"], default="all")
    parser.add_argument("--permutations", type=int, default=20)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    e, logs, patterns, f, reps, width, models = load()
    catalog = support_table(e, patterns, f, models)
    if args.mode in ["all", "support"]:
        coefficient_table(models)
        conditional_permutation(e, patterns, f, reps, width, models, args.permutations)
        case_sample(e, logs, patterns, f, reps, width, models)
    if args.mode in ["all", "association"]:
        association_catalog(e, patterns, f, models, catalog)
    if args.mode in ["all", "negative"]:
        source_outcome_shuffle(e, patterns, f, reps, width, models, args.permutations)
    selection_frequency(patterns)
    mem = psutil.Process().memory_info()
    manifest = {"mode": args.mode, "permutations": args.permutations, "elapsed_seconds": time.perf_counter() - started,
                "rss_MB": mem.rss / 2**20, "peak_working_set_MB": getattr(mem, "peak_wset", mem.rss) / 2**20,
                "cpu_seconds": time.process_time(), "cohorts": sorted(models), "catalog_size": len(catalog),
                "discovery": "source DISC only; CONF never selected patterns", "individual_cases": "private only; no IDs in results",
                "conditional_permutation": "within baseline ADL, single-event bag and age groups; zero/unknown/ambiguous directions held fixed",
                "association": "CONF only, adjustment age/sex/baseline state/single events+unknown; household sandwich; BH with BY sensitivity",
                "negative_control": "source labels permuted within cohort/baseline state/age; target fixed; diagnostic, not p-value",
                "causal_claim": False,
                "model_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "private" / "models").glob("*.pkl")},
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / ("manifest_" + args.mode + ".json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
