"""Read-only panel construction for the explicitly declared observed-window estimand."""
import bootstrap_env
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANELS = bootstrap_env.PANELS
ADL = ["adl_dress", "adl_bath", "adl_eat", "adl_toilet"]
DISEASES = ["hypertension", "diabetes", "stroke"]
COHORTS = ["HRS", "ELSA", "CHARLS", "MHAS"]

def functional_state(frame):
    a = frame[ADL].to_numpy(float)
    return np.where(np.any(a == 1, axis=1), 1., np.where(np.all(a == 0, axis=1), 0., np.nan))

def stable_split(identifier):
    digest = hashlib.sha256((str(identifier) + "|interval_chain_101").encode()).hexdigest()
    return int(digest[:10], 16) % 4 == 0

def robust_slope(t, x):
    ok = np.isfinite(t) & np.isfinite(x)
    if ok.sum() < 2 or np.ptp(t[ok]) < 1e-6:
        return np.nan
    return float(np.polyfit(t[ok] - t[ok].mean(), x[ok], 1)[0])

def prepare():
    started = time.perf_counter()
    (ROOT / "private").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    rows, logs, visits, flow = [], [], [], []
    for cohort in COHORTS:
        path = PANELS / (cohort + ".csv.gz")
        p = pd.read_csv(path, dtype={"person_id": str, "household_id": str})
        # The source's combined bed/walk item in CHARLS 2011 is not silently
        # equated with a bed-only item. A common four-item outcome is frozen.
        p["state"] = functional_state(p)
        p = p[np.isfinite(p.date_year)].sort_values(["person_id", "date_year", "wave"])
        if p.duplicated(["person_id", "date_year"]).any():
            raise ValueError(f"Duplicate person-date rows in {cohort}")
        extra_path = ROOT / "private" / "event_status" / f"{cohort}_event_status.csv.gz"
        if extra_path.exists():
            ex = pd.read_csv(extra_path, dtype={"person_id": str})
            p = p.merge(ex, on=["person_id", "wave"], how="left", validate="one_to_one", suffixes=("", "_event"))
            # Restore visit-time reports where harmonized ever-disease fields
            # had been retrospectively edited using later interviews.
            for disease in DISEASES:
                at_visit = disease + "_at_visit"
                if at_visit in p:
                    p[disease] = p[at_visit]
        counts = dict(cohort=cohort, source_rows=len(p), source_people=p.person_id.nunique(),
                      age_history_eligible=0, baseline_state_unknown=0, no_followup_window=0,
                      followup_state_unknown=0, assessed=0, confirmation=0,
                      complete_event_history=0)
        for pid, g in p.groupby("person_id", sort=False):
            g = g.reset_index(drop=True)
            times = g.date_year.to_numpy(float)
            # t0 is chosen without using future attendance or health values.
            chosen = None
            age_array = g.age.to_numpy(float)
            earliest = np.searchsorted(times, times-4.25, side="left")
            eligible = np.flatnonzero((age_array >= 60) & ((np.arange(len(g))-earliest+1)>=3) & ((times-times[earliest])>=3.75))
            if len(eligible):
                j = int(eligible[0])
                chosen = (j, g.iloc[int(earliest[j]):j+1])
            if chosen is None:
                continue
            counts["age_history_eligible"] += 1
            j, hist = chosen
            b = g.iloc[j]
            t0 = float(b.date_year)
            if not np.isfinite(b.state):
                counts["baseline_state_unknown"] += 1
                continue
            follow = g[(g.date_year >= t0 + 2.0 - 1e-8) & (g.date_year <= t0 + 4.0 + 1e-8)]
            assessed = len(follow) > 0
            end = follow.iloc[0] if assessed else None
            if not assessed:
                counts["no_followup_window"] += 1
            elif not np.isfinite(end.state):
                counts["followup_state_unknown"] += 1
            known = bool(assessed and np.isfinite(end.state))
            family = b.get("household_id", pid)
            if pd.isna(family):
                family = pid
            confirm = stable_split(family)
            hs = hist.state.to_numpy(float)
            record = {
                "person_id": pid, "household_id": str(family), "cohort": cohort,
                "t0": t0, "baseline_state": int(b.state), "confirmation": int(confirm),
                "response": int(known), "y": float(end.state) if known else np.nan,
                "delta": float(end.date_year - t0) if assessed else np.nan,
                "followup_year": float(end.date_year) if assessed else np.nan,
                "age": b.age, "female": b.get("female", np.nan),
                "education_years": b.get("education_years", np.nan),
                "married": b.get("married", np.nan), "smoking": b.get("smoking", np.nan),
                "self_rated_health": b.get("self_rated_health", np.nan),
                "adl_level": b.state, "adl_mean": np.nanmean(hs) if np.isfinite(hs).any() else np.nan,
                "adl_slope": robust_slope(hist.date_year.to_numpy(float), hs),
                "adl_changes": float(np.sum(np.diff(hs[np.isfinite(hs)]) != 0)),
                "history_visits": len(hist), "history_span": t0 - float(hist.date_year.min()),
                "gap_mean": float(np.diff(hist.date_year).mean()),
                "gap_max": float(np.diff(hist.date_year).max()),
                "proxy_latest": b.get("proxy", np.nan),
                "proxy_mean": hist["proxy"].mean() if "proxy" in hist else np.nan,
                "known_death_year": g.death_year.dropna().iloc[0] if "death_year" in g and g.death_year.notna().any() else np.nan,
                "survey_weight": b.get("survey_weight", np.nan),
            }
            history_complete = True
            for disease in DISEASES:
                a = hist[disease].to_numpy(float)
                record[disease] = float(b[disease])
                record[disease + "_mean"] = float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
                record[disease + "_slope"] = robust_slope(hist.date_year.to_numpy(float), a)
                record[disease + "_unknown"] = int(np.any(~np.isfinite(a)))
                history_complete &= bool(np.all(np.isfinite(a)))
                record[disease + "_event"] = 0. if np.all(np.isfinite(a)) else np.nan
                # Only the first observed negative -> positive transition;
                # prevalent disease and positive -> negative -> positive are not new onset.
                full_before = g.iloc[:j+1]
                values = full_before[disease].to_numpy(float)
                yes = np.flatnonzero(values == 1)
                if len(yes):
                    iy = int(yes[0])
                    neg = np.flatnonzero(values[:iy] == 0)
                    if len(neg):
                        lastneg = int(neg[-1])
                        left, right = float(times[lastneg]), float(times[iy])
                        if left >= t0 - 4.25 and right <= t0 and right > left:
                            logs.append({"person_id": pid, "cohort": cohort, "event_type": disease,
                                         "L": left - t0, "R": right - t0,
                                         "prior_state": 0, "new_state": 1,
                                         "precision": "observed_interview_interval", "observation_mask": "first_positive_after_negative"})
                            record[disease + "_event"] = 1.
            # Social events are optional until source-level coding is validated.
            if "widowed" in hist:
                w = hist.widowed.to_numpy(float)
                m = hist.married.to_numpy(float)
                record["widowhood_unknown"] = int(np.any(~np.isfinite(w)) or np.any(~np.isfinite(m)))
                record["widowhood_event"] = np.nan if record["widowhood_unknown"] else 0.
                for q in range(1, len(hist)):
                    if w[q] == 1 and m[q-1] == 1 and w[q-1] == 0:
                        logs.append({"person_id": pid, "cohort": cohort, "event_type": "widowhood",
                                     "L": float(hist.date_year.iloc[q-1]-t0), "R": float(hist.date_year.iloc[q]-t0),
                                     "prior_state": 0, "new_state": 1, "precision": "observed_interview_interval",
                                     "observation_mask": "validated_widowhood"})
                        record["widowhood_event"] = 1.
                        break
            else:
                record["widowhood_unknown"] = 1
                record["widowhood_event"] = np.nan
            record["event_history_complete"] = int(history_complete)
            counts["complete_event_history"] += int(history_complete)
            rows.append(record)
            for _, h in hist.iterrows():
                visits.append({"person_id": pid, "cohort": cohort, "time": float(h.date_year-t0),
                               "state": h.state, "proxy": h.get("proxy", np.nan)})
            if known:
                counts["assessed"] += 1
                counts["confirmation"] += int(confirm)
        flow.append(counts)
        print(json.dumps({"prepared_cohort":cohort,**counts},ensure_ascii=False),flush=True)
    episodes = pd.DataFrame(rows)
    event_logs = pd.DataFrame(logs)
    history_visits = pd.DataFrame(visits)
    episodes.to_csv(ROOT / "private" / "episodes.csv.gz", index=False)
    event_logs.to_csv(ROOT / "private" / "interval_logs.csv.gz", index=False)
    history_visits.to_csv(ROOT / "private" / "history_visits.csv.gz", index=False)
    pd.DataFrame(flow).to_csv(ROOT / "results" / "riskset_flow.csv", index=False)
    summary = episodes.groupby("cohort").agg(n=("person_id", "size"), assessed=("response", "sum"),
                                               outcome_rate=("y", "mean"), mean_delta=("delta", "mean"),
                                               confirmation=("confirmation", "sum"),
                                               complete_event_history=("event_history_complete", "sum")).reset_index()
    summary.to_csv(ROOT / "results" / "cohort_summary.csv", index=False)
    report = {"status": "completed", "elapsed_seconds": time.perf_counter()-started,
              "primary_outcome_items": ADL, "excluded_item": "bed-only not equated to CHARLS2011 combined bed/walk",
              "rows": len(episodes), "event_rows": len(event_logs), "no_future_t0_selection": True,
              "source_panels_sha256": {c: hashlib.sha256((PANELS/(c+".csv.gz")).read_bytes()).hexdigest() for c in COHORTS},
              "social_extensions_loaded": {c: (ROOT/"private"/"event_status"/f"{c}_event_status.csv.gz").exists() for c in COHORTS}}
    (ROOT/"results"/"prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"], "rows": len(episodes)}, ensure_ascii=False))

if __name__ == "__main__":
    prepare()
