"""Continuous-time synthetic domains, never actual cohorts.

Exact event times and reversible illness/death paths are generated before
interviews.  Reported event logs and uncertain observed endpoints are then
derived from interviews; latent truth is retained solely for validation.
"""
import numpy as np
import pandas as pd
try:
    from .interval_core import chain_truth
except ImportError:
    from interval_core import chain_truth

EVENT_TYPES = ("retirement", "widowhood", "chronic", "alone")
SCENARIOS = ("null", "chain", "triple", "overlap", "irregular", "informative_dropout",
             "informative_visits", "hidden_death", "reversible", "measurement_shift")


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def _scenario(scenario):
    if isinstance(scenario, dict):
        values = dict(scenario)
        name = values.pop("name", "custom")
    else:
        name, values = str(scenario), {}
    settings = {"name": name, "order_effect": 0.9, "triple_effect": 0.0,
                "history": 4.0, "horizon": 3.0, "max_followup": 5.5,
                "overlap": False, "irregular": False, "dropout": 0.06,
                "informative_dropout": 0.0, "informative_visits": 0.0,
                "death_rate": 0.035, "illness_death_multiplier": 2.5,
                "recovery_rate": 0.22, "report_error": 0.0,
                "event_reversal_rate": 0.0, "missing_report": 0.015,
                "domain_shift": True}
    if name in ("null", "zero_order", "null_order"):
        settings["order_effect"] = 0.0
    elif name in ("triple", "true_triple"):
        settings.update(order_effect=0.3, triple_effect=1.25)
    elif name in ("overlap", "same_interval"):
        settings["overlap"] = True
    elif name in ("irregular", "long_interval"):
        settings["irregular"] = True
    elif name in ("informative_dropout", "dropout"):
        settings.update(dropout=0.13, informative_dropout=1.1)
    elif name in ("informative_visits", "health_visits"):
        settings["informative_visits"] = 1.2
    elif name in ("hidden_death", "death_hidden_deterioration", "hidden_impairment_death"):
        settings.update(death_rate=0.07, illness_death_multiplier=4.0)
    elif name in ("reversible", "recovery", "recovery_shift"):
        settings.update(recovery_rate=0.8, event_reversal_rate=0.35)
        if name == "recovery_shift":
            settings["report_error"] = 0.04
    elif name in ("measurement_shift", "report_error"):
        settings.update(report_error=0.07, event_reversal_rate=0.15)
    elif name not in ("chain", "base", "custom"):
        raise ValueError(f"Unknown simulation scenario {name!r}")
    settings.update(values)
    return settings


def simulate_domain(n, seed, scenario="chain", domain=0):
    """Return one synthetic observation domain with exact and observed data.

    Keys: persons, interval_logs, visits, outcomes, exact_times, event_types,
    truth_y (=truth_yA), truth_state3, truth_chain, latent_paths, meta.
    persons includes level/slope/current-state and observation-process features.
    visits includes all planned waves, observed flag, true/reported outcome state,
    event reports, proxy and mode.  Dead is state 2.  outcomes contains first
    illness endpoint bounds AND set-valued state-at-horizon observations.

    ``scenario`` may be a named scenario or a dictionary overriding settings.
    Main target B permits recovery.  For state-at-horizon, an interview on
    either side cannot identify an unobserved transition/recovery; alive cases
    therefore remain set-valued unless interviewed exactly at the horizon.
    """
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    s = _scenario(scenario)
    rng = np.random.default_rng(seed)
    d = int(domain) if isinstance(domain, (int, np.integer)) else sum(map(ord, str(domain))) % 3
    d %= 3
    label = f"synthetic_domain_{d + 1}"
    h, horizon, maxt = float(s["history"]), float(s["horizon"]), float(s["max_followup"])
    ids = np.array([f"S{d + 1}_{seed}_{j:06d}" for j in range(n)], dtype=object)
    age = np.clip(rng.normal(70 + 1.5 * d, 6.0, n), 60, 92)
    sex = rng.binomial(1, 0.49 + 0.02 * d, n)
    education = np.clip(rng.normal(11.5 - 1.2 * d, 3.0, n), 0, 22)
    frailty = rng.normal(0, 1, n)
    level = 0.45 * frailty + 0.025 * (age - 70) - 0.035 * (education - 10) + rng.normal(0, .5, n)
    slope = 0.08 * frailty + 0.018 * (age - 70) + rng.normal(0, .13, n)
    rate = np.array([.15, .095, .17, .12])
    domain_scale = np.array([1.0, 1.2, .82])[d] if s["domain_shift"] else 1.0
    hazards = rate[None, :] * domain_scale * np.exp(.18 * frailty[:, None] + .012 * (age[:, None] - 70))
    exact = -h + rng.exponential(1 / hazards)
    prevalent = rng.random((n, 4)) < np.array([.12, .05, .14, .08])
    exact[prevalent] = -h - rng.uniform(.1, 8, prevalent.sum())
    if s["overlap"]:
        center = rng.uniform(-h + .4, -.5, n)
        cluster = rng.random(n) < .65
        exact[cluster] = center[cluster, None] + rng.normal(0, .08, (cluster.sum(), 4))
    true_chain2 = chain_truth(exact, ("retirement", "chronic"), EVENT_TYPES, (-h, 0))
    true_chain3 = chain_truth(exact, ("widowhood", "alone", "chronic"), EVENT_TYPES, (-h, 0))
    event_bag = np.sum((exact > -h) & (exact <= 0), axis=1)
    # Under the null, events can be correlated and prognostic; their *order*
    # carries no further effect after the generated covariates and event bag.
    logill = -2.45 + .026 * (age - 70) + .32 * level + .55 * slope + .11 * event_bag
    logill += s["order_effect"] * true_chain2 + s["triple_effect"] * true_chain3
    logill += np.array([0.0, .08, -.08])[d] if s["domain_shift"] else 0.0
    illness_rate = np.exp(logill)
    death_rate = float(s["death_rate"]) * np.exp(.035 * (age - 70) + .18 * frailty)
    recovery_rate = float(s["recovery_rate"]) * np.exp(-.1 * frailty)

    latent_paths, death, first_illness = [], np.full(n, np.inf), np.full(n, np.inf)
    # Continuous-time Markov paths are generated without referring to visits.
    for j in range(n):
        t, state, path = 0.0, 0, [(0.0, 0)]
        while t < maxt and state != 2:
            health_rate = illness_rate[j] if state == 0 else recovery_rate[j]
            mortality = death_rate[j] * (float(s["illness_death_multiplier"]) if state == 1 else 1.0)
            total = health_rate + mortality
            t += rng.exponential(1 / total)
            if t > maxt:
                break
            if rng.random() < mortality / total:
                state, death[j] = 2, t
            else:
                state = 1 - state
                if state == 1 and not np.isfinite(first_illness[j]):
                    first_illness[j] = t
            path.append((float(t), int(state)))
        latent_paths.append(path)

    def states_at(times):
        result = np.zeros(n, int)
        for j, path in enumerate(latent_paths):
            if times[j] >= 0:
                tt = np.array([v[0] for v in path])
                result[j] = path[np.searchsorted(tt, times[j], side="right") - 1][1]
        return result

    truth_state3 = states_at(np.full(n, horizon))
    truth_yA = ((first_illness <= horizon) & (first_illness < death)).astype(int)
    reversaltime = np.full_like(exact, np.inf)
    if s["event_reversal_rate"] > 0:
        reversaltime[:, [0, 3]] = exact[:, [0, 3]] + rng.exponential(1 / float(s["event_reversal_rate"]), (n, 2))
    if d == 0:
        schedule = np.array([-h, -h / 2, 0, 1.8, 3.7, maxt])
    elif d == 1:
        schedule = np.array([-h, -h * .55, 0, 2.2, 4.3, maxt])
    else:
        schedule = np.array([-h, -h * .35, 0, 2.7, 4.8, maxt])
    if s["irregular"]:
        schedule[1] = -h * .72
        schedule[3] = 3.2
        schedule[4] = 4.9
    all_rows, report_cube, visit_times, observe_cube = [], [], [], []
    dropout = np.zeros(n, bool)
    for wave, scheduled in enumerate(schedule):
        time = np.full(n, scheduled, float)
        if scheduled not in (-h, 0):
            jitter = .22 if not s["irregular"] else .5
            time += rng.uniform(-jitter, jitter, n)
            if scheduled > 0:
                time = np.minimum(time, maxt)
        state = states_at(time)
        if scheduled > 0 and s["informative_visits"]:
            time = np.maximum(.1, time - float(s["informative_visits"]) * .35 * (state == 1))
            state = states_at(time)
        death_before = death <= time
        if scheduled > 0:
            pdrop = _expit(-3.3 + float(s["informative_dropout"]) * (.5 * frailty + (state == 1)))
            dropout |= rng.random(n) < (pdrop + float(s["dropout"]) * .2)
        missprob = .055 + .025 * d + .025 * (scheduled < 0)
        observed = (rng.random(n) > missprob) & ~death_before & ~dropout
        if scheduled in (-h, 0):
            observed[:] = True   # chosen landmark participants and baseline record
        proxy = (rng.random(n) < _expit(-3.2 + .7 * frailty + .8 * (state == 1))).astype(int)
        mode = np.where(rng.random(n) < .15 + .15 * d, "phone", "in_person")
        reports = ((exact <= time[:, None]) & (time[:, None] < reversaltime)).astype(float)
        err = float(s["report_error"]) * (1 + .3 * d + .4 * proxy[:, None])
        flip = rng.random((n, 4)) < err
        reports[flip] = 1 - reports[flip]
        reports[rng.random((n, 4)) < float(s["missing_report"])] = np.nan
        reports[~observed] = np.nan
        reported_state = state.astype(float)
        state_flip = (rng.random(n) < float(s["report_error"]) * (1 + proxy * .4)) & (state != 2)
        reported_state[state_flip] = 1 - reported_state[state_flip]
        reported_state[~observed] = np.nan
        frame = pd.DataFrame({"person_id": ids, "cohort": label, "wave": wave, "time": time,
                              "observed": observed, "dropout": dropout.copy(), "death_before_visit": death_before,
                              "proxy": proxy, "mode": mode, "true_state": state,
                              "state": reported_state, "state_report_error": state_flip,
                              "level": level + slope * time + .6 * (state == 1) + rng.normal(0, .15, n)})
        for e, event in enumerate(EVENT_TYPES):
            frame[event] = reports[:, e]
        all_rows.append(frame)
        report_cube.append(reports)
        visit_times.append(time)
        observe_cube.append(observed)
    visits = pd.concat(all_rows, ignore_index=True)
    reports = np.stack(report_cube, axis=1)
    vt = np.stack(visit_times, axis=1)
    obs = np.stack(observe_cube, axis=1)
    states = np.stack([f["state"].to_numpy() for f in all_rows], axis=1)
    logs, baseline_prevalent, reversal_counts = [], np.zeros((n, 4), bool), np.zeros((n, 4), int)
    for j in range(n):
        for e, event in enumerate(EVENT_TYPES):
            lastnegative, laststate, emitted, first = None, None, False, True
            for w in range(3):
                value = reports[j, w, e]
                if np.isnan(value):
                    continue
                if first:
                    baseline_prevalent[j, e] = value == 1
                    first = False
                reversal_counts[j, e] += int(laststate == 1 and value == 0)
                if value == 1 and lastnegative is not None and not emitted and not baseline_prevalent[j, e]:
                    logs.append({"person_id": ids[j], "cohort": label, "event_type": event,
                                 "L": float(lastnegative), "R": float(vt[j, w]), "prior_state": 0, "new_state": 1,
                                 "precision": "interview_interval", "observation_mask": "explicit_negative_to_positive",
                                 "truth_time": float(exact[j, e])})
                    emitted = True
                if value == 0:
                    lastnegative = vt[j, w]
                laststate = value
    logcols = ["person_id", "cohort", "event_type", "L", "R", "prior_state", "new_state", "precision", "observation_mask", "truth_time"]
    interval_logs = pd.DataFrame(logs, columns=logcols)
    outcome_rows = []
    for j in range(n):
        valid = obs[j] & (vt[j] >= 0) & np.isfinite(states[j])
        positives = valid & (states[j] == 1)
        negatives = valid & (states[j] == 0)
        positive_t = float(np.min(vt[j, positives])) if positives.any() else np.inf
        beforepositive = negatives & (vt[j] < positive_t)
        lastnegative = float(np.max(vt[j, beforepositive])) if beforepositive.any() else 0.0
        # First-event bounds need irreversibility and error-free reporting.  For
        # recovery scenarios, later negative visits cannot rule out prior illness.
        lower = int(positive_t <= horizon)
        if lower:
            upper, status = 1, "observed_positive_by_horizon"
        elif s["recovery_rate"] == 0 and (negatives & (vt[j] >= horizon)).any() and death[j] > horizon:
            upper, status = 0, "observed_negative_after_horizon_absorbing"
        else:
            upper = 1
            status = "death_hidden_path" if death[j] <= horizon else "interval_or_unobserved_recovery"
        allowed = [True, True, False]
        observed_state = np.nan
        if death[j] <= horizon:
            allowed, observed_state = [False, False, True], 2.0
        else:
            exactvisit = valid & (vt[j] == horizon)
            if exactvisit.any():
                observed_state = float(states[j, np.flatnonzero(exactvisit)[0]])
                allowed = [observed_state == 0, observed_state == 1, False]
        future_obs = valid & (vt[j] > 0)
        lastvisit = float(np.max(vt[j, future_obs])) if future_obs.any() else 0.0
        hidden = bool(first_illness[j] < death[j] <= horizon and not np.any(positives & (vt[j] < death[j])))
        outcome_rows.append({"person_id": ids[j], "cohort": label, "truth_yA": int(truth_yA[j]),
                             "truth_state3": int(truth_state3[j]), "observed_y_lower": lower,
                             "observed_y_upper": upper, "status": status,
                             "outcome_L": lastnegative, "outcome_R": positive_t,
                             "death_time": float(death[j]), "death_by_horizon": int(death[j] <= horizon),
                             "first_illness_time": float(first_illness[j]), "last_observed_time": lastvisit,
                             "observed_state3": observed_state, "state0_possible": allowed[0],
                             "state1_possible": allowed[1], "state2_possible": allowed[2],
                             "hidden_illness_before_death": hidden,
                             "A_bounds_assume_error_free_reports": True,
                             "A_negative_requires_absorption": True})
    outcomes = pd.DataFrame(outcome_rows)
    historical_obs = obs[:, :3].sum(axis=1)
    hist_intervals = vt[:, 1:3] - vt[:, :2]
    persons = pd.DataFrame({"person_id": ids, "cohort": label, "domain": d, "age": age, "sex": sex,
                            "education": education, "level": level, "slope": slope,
                            "current_health": level, "current_state": np.zeros(n, int),
                            "n_history_visits": historical_obs,
                            "mean_history_interval": hist_intervals.mean(axis=1),
                            "max_history_interval": hist_intervals.max(axis=1),
                            "history_missing": 3 - historical_obs, "history_complete": historical_obs >= 2,
                            "proxy_t0": all_rows[2]["proxy"].to_numpy(),
                            "mode_phone_t0": (all_rows[2]["mode"].to_numpy() == "phone").astype(int),
                            "baseline_prevalent_events": baseline_prevalent.sum(axis=1),
                            "report_reversals": reversal_counts.sum(axis=1),
                            "latent_frailty": frailty})
    for e, event in enumerate(EVENT_TYPES):
        persons[event + "_current"] = reports[:, 2, e]
        persons[event + "_prevalent"] = baseline_prevalent[:, e].astype(int)
        persons[event + "_at_risk"] = (reports[:, 0, e] == 0).astype(int)
        persons[event + "_history_complete"] = np.isfinite(reports[:, 0, e]) & np.isfinite(reports[:, 2, e])
    history_completeness = {
        person: {event: bool(np.isfinite(reports[j, 0, e]) and np.isfinite(reports[j, 2, e]))
                 for e, event in enumerate(EVENT_TYPES)}
        for j, person in enumerate(ids)}
    return {"persons": persons, "interval_logs": interval_logs, "visits": visits, "outcomes": outcomes,
            "exact_times": exact, "event_types": EVENT_TYPES, "truth_y": truth_yA,
            "truth_yA": truth_yA, "truth_state3": truth_state3,
            "truth_chain": true_chain2, "truth_chain3": true_chain3,
            "latent_paths": latent_paths, "history_completeness": history_completeness,
            "meta": {"synthetic": True, "label": label, "n": n, "seed": int(seed),
                     "domain": d, "scenario": s, "t0": 0.0, "history": (-h, 0.0), "horizon": horizon,
                     "truth_use": "Simulation validation only; never predictor inputs",
                     "target_B": "state-at-horizon: healthy/impaired/dead; recovery allowed",
                     "target_A": "first illness before death by horizon; negatives unidentified with recovery",
                     "bounds_scope": "recorded transitions; reporting error can invalidate true-event coverage"}}
