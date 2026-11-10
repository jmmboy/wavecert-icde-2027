"""Exact logical matching of first recorded events with times in (L, R].

Bounds describe completions of *recorded* events.  They do not cover unrecorded
transitions or turn incomplete histories into observed negatives.  Recurrence
is intentionally not implemented: duplicate event types raise ValueError.
"""
from itertools import permutations
from collections.abc import Mapping
import numpy as np
import pandas as pd


def _records(logs):
    rows = logs.to_dict("records") if isinstance(logs, pd.DataFrame) else list(logs)
    ans = {}
    for row in rows:
        event = str(row["event_type"])
        if event in ans:
            raise ValueError(f"Repeated event type {event!r}; first-event-only semantics required")
        lo, hi = float(row["L"]), float(row["R"])
        if not np.isfinite([lo, hi]).all() or not lo < hi:
            raise ValueError(f"Invalid (L,R] for {event}: ({lo},{hi}]")
        ans[event] = (lo, hi)
    return ans


def _feasible(intervals, constraints=()):
    """Difference constraints, preserving strictness symbolically, without epsilon.

    A constraint (a,b,c,strict) means t_a - t_b <= c, or < c when
    strict=True.  Floyd--Warshall propagates (bound, strict) pairs.  A zero
    diagonal with strict=True is infeasible, as is a negative diagonal.
    """
    names = list(intervals)
    index = {e: j + 1 for j, e in enumerate(names)}
    n = len(names) + 1
    d = np.full((n, n), np.inf)
    strict = np.zeros((n, n), bool)
    np.fill_diagonal(d, 0.0)

    def edge(a, b, c, is_strict):
        if c < d[a, b] or (c == d[a, b] and is_strict and not strict[a, b]):
            d[a, b], strict[a, b] = c, bool(is_strict)

    for e, (lo, hi) in intervals.items():
        j = index[e]
        edge(0, j, hi, False)    # t_j <= R
        edge(j, 0, -lo, True)    # 0 - t_j < -L
    for a, b, c, s in constraints:
        if a not in index or b not in index:
            raise ValueError("Constraint refers to an unrecorded event")
        edge(index[b], index[a], float(c), s)
    for k in range(n):
        candidate = d[:, [k]] + d[[k], :]
        candidate_strict = strict[:, [k]] | strict[[k], :]
        improve = (candidate < d) | ((candidate == d) & candidate_strict & ~strict)
        d[improve] = candidate[improve]
        strict[improve] = candidate_strict[improve]
    diagonal = np.diag(d)
    return not np.any((diagonal < 0) | ((diagonal == 0) & np.diag(strict)))


def chain_bounds(logs, chain, max_span=None, history_complete=True, constraints=()):
    """Return lower/upper logical indicators for one chain in one person's log.

    Strict order t1<t2<... and optional total span t_last-t_first<=max_span.
    Additional constraints use the convention in ``_feasible``.  If a chain
    event is absent and its history is incomplete, both bounds are NaN.
    ``history_complete`` can instead map event type to a completeness flag.
    """
    chain = tuple(map(str, chain))
    if len(chain) < 2 or len(set(chain)) != len(chain):
        raise ValueError("Use a chain of at least two distinct first-event types")
    if max_span is not None and (not np.isfinite(max_span) or max_span < 0):
        raise ValueError("max_span must be finite and nonnegative")
    rec = _records(logs)
    missing = [e for e in chain if e not in rec]
    complete = lambda e: bool(history_complete.get(e, False)) if isinstance(history_complete, Mapping) else bool(history_complete)
    if missing:
        # One known absent event already rules the chain out, despite other missingness.
        if any(complete(e) for e in missing):
            return {"lower": 0, "upper": 0, "status": "absent_recorded_event"}
        return {"lower": np.nan, "upper": np.nan, "status": "unknown_history"}
    base = tuple(constraints)
    if not _feasible(rec, base):
        raise ValueError("Observed intervals and declared constraints have no feasible completion")
    order = [(a, b, 0.0, True) for a, b in zip(chain[:-1], chain[1:])]
    requested = order + ([] if max_span is None else [(chain[-1], chain[0], float(max_span), False)])
    possible = _feasible(rec, base + tuple(requested))
    if not possible:
        return {"lower": 0, "upper": 0, "status": "impossible"}
    # Complement of conjunction: at least one constraint can fail.  We check
    # every violation against the same complete feasible-world constraints.
    necessary = True
    for a, b in zip(chain[:-1], chain[1:]):
        if _feasible(rec, base + ((b, a, 0.0, False),)):
            necessary = False
            break
    if necessary and max_span is not None:
        necessary = not _feasible(rec, base + ((chain[0], chain[-1], -float(max_span), True),))
    return {"lower": int(necessary), "upper": 1,
            "status": "certain" if necessary else "ambiguous"}


def enumerate_patterns(event_types, lengths=(2,)):
    return [p for k in lengths for p in permutations(tuple(event_types), k)]


def chain_truth(exact_times, chain, event_types=None, history=(-4.0, 0.0), max_span=None):
    """Exact first-event-chain indicator, for simulation diagnostics only."""
    if isinstance(exact_times, pd.DataFrame):
        t = exact_times.loc[:, list(chain)].to_numpy(float)
    else:
        if event_types is None:
            raise ValueError("event_types is required for an array")
        idx = {v: j for j, v in enumerate(event_types)}
        t = np.asarray(exact_times)[:, [idx[e] for e in chain]]
    good = np.all(np.isfinite(t) & (t > history[0]) & (t <= history[1]), axis=1)
    good &= np.all(np.diff(t, axis=1) > 0, axis=1)
    if max_span is not None:
        good &= t[:, -1] - t[:, 0] <= max_span
    return good.astype(int)


def batch_chain_features(logs, person_ids, patterns, completeness=None, max_span=None):
    """Vectorized first-event features; optional three-event/max-span fallback.

    ``completeness``: None (complete recorded-history risk set), bool, mapping
    person_id -> bool or event-completeness mapping, or Series indexed by ID.
    Unknown histories have NaN lower/upper/ambiguous and unknown=1.  Known
    absence means no *recorded* event within an eligible observed history.
    """
    ids = pd.Index(person_ids, name="person_id")
    if ids.has_duplicates:
        raise ValueError("person_ids must be unique")
    patterns = [tuple(map(str, p)) for p in patterns]
    if len(logs) and logs.duplicated(["person_id", "event_type"]).any():
        raise ValueError("Repeated person/event_type; first-event-only semantics required")
    if len(logs) and (not np.isfinite(logs[["L", "R"]].to_numpy(float)).all() or
                     np.any(logs.L.to_numpy() >= logs.R.to_numpy())):
        raise ValueError("Invalid event interval")
    if completeness is None:
        comp = {i: True for i in ids}
    elif isinstance(completeness, (bool, np.bool_)):
        comp = {i: bool(completeness) for i in ids}
    else:
        comp = completeness
    events = sorted(set(e for p in patterns for e in p))
    if len(logs):
        left = logs.pivot(index="person_id", columns="event_type", values="L").reindex(index=ids, columns=events)
        right = logs.pivot(index="person_id", columns="event_type", values="R").reindex(index=ids, columns=events)
    else:
        left = pd.DataFrame(np.nan, index=ids, columns=events)
        right = left.copy()
    la, ra = left.to_numpy(float), right.to_numpy(float)
    ei = {e: j for j, e in enumerate(events)}
    complete_by_event = np.empty((len(ids), len(events)), bool)
    for j, e in enumerate(events):
        for k, person in enumerate(ids):
            c = comp.get(person, False)
            complete_by_event[k, j] = bool(c.get(e, False)) if isinstance(c, Mapping) else bool(c)
    result = {}
    records_by_person = None
    for pattern in patterns:
        if len(set(pattern)) != len(pattern) or len(pattern) < 2:
            raise ValueError("Patterns must have distinct first-event types")
        jj = [ei[e] for e in pattern]
        l, r = la[:, jj], ra[:, jj]
        missing = np.isnan(l)
        known_absent = np.any(missing & complete_by_event[:, jj], axis=1)
        unknown = np.any(missing, axis=1) & ~known_absent
        # For strict ordered (L,R] events, feasibility is exactly that each
        # prefix's greatest lower endpoint is below the current right endpoint.
        possible = np.all(np.maximum.accumulate(l, axis=1) < r, axis=1)
        certain = np.all(r[:, :-1] <= l[:, 1:], axis=1)
        if max_span is not None:
            if records_by_person is None:
                records_by_person = {key: value for key, value in logs.groupby("person_id", sort=False)}
            for k, person in enumerate(ids):
                if not missing[k].any():
                    b = chain_bounds(records_by_person[person], pattern, max_span=max_span)
                    possible[k], certain[k] = b["upper"], b["lower"]
        lower, upper = certain.astype(float), possible.astype(float)
        lower[known_absent] = upper[known_absent] = 0.0
        lower[unknown] = upper[unknown] = np.nan
        name = ">".join(pattern)
        result[name + "__lower"] = lower
        result[name + "__upper"] = upper
        result[name + "__ambiguous"] = upper - lower
        result[name + "__unknown"] = unknown.astype(float)
    return pd.DataFrame(result, index=ids)


def support_bounds(features, patterns=None):
    """Support only among known eligible histories; missingness is explicit."""
    if patterns is None:
        names = [c[:-7] for c in features if c.endswith("__lower")]
    else:
        names = [">".join(p) for p in patterns]
    result = []
    for name in names:
        l, u = features[name + "__lower"], features[name + "__upper"]
        good = l.notna() & u.notna()
        n = int(good.sum())
        result.append({"pattern": name, "n_eligible": n, "n_unknown": int((~good).sum()),
                       "certain_count": int(l[good].sum()), "possible_count": int(u[good].sum()),
                       "support_lower": float(l[good].mean()) if n else np.nan,
                       "support_upper": float(u[good].mean()) if n else np.nan,
                       "ambiguity_width": float((u[good] - l[good]).mean()) if n else np.nan})
    return pd.DataFrame(result)


def interval_logs_from_visits(visits, event_types, person_col="person_id", time_col="time",
                              observed_col="observed", history=(-4.0, 0.0)):
    """Build first recorded 0->1 transitions; do not bridge through a fake zero.

    Input has an observed flag and one 0/1/NaN column per event.  A missing
    wave widens L to the last explicit negative.  Baseline-positive entries
    never become incident events.  Reversals are counted separately in audit.
    """
    rows, audit = [], []
    use = visits[(visits[time_col] >= history[0]) & (visits[time_col] <= history[1])]
    for person, frame in use.groupby(person_col, sort=False):
        frame = frame.sort_values(time_col)
        for event in event_types:
            prev, negative, emitted, baseline_positive, reversals = None, None, False, False, 0
            for _, row in frame.iterrows():
                if observed_col in frame and not bool(row[observed_col]):
                    continue
                value = row[event]
                if pd.isna(value):
                    continue
                value = int(value)
                if value not in (0, 1):
                    raise ValueError("Visit event states must be 0,1,NaN")
                if prev is None and value == 1:
                    baseline_positive = True
                if prev == 1 and value == 0:
                    reversals += 1
                if value == 1 and negative is not None and not emitted and not baseline_positive:
                    rows.append({"person_id": person, "cohort": row.get("cohort", "unknown"),
                                 "event_type": event, "L": float(negative), "R": float(row[time_col]),
                                 "prior_state": 0, "new_state": 1, "precision": "interview_interval",
                                 "observation_mask": "explicit_negative_to_positive"})
                    emitted = True
                if value == 0:
                    negative = float(row[time_col])
                prev = value
            audit.append({"person_id": person, "event_type": event, "baseline_positive": baseline_positive,
                          "reversals": reversals, "recorded_event": emitted,
                          "has_explicit_negative": negative is not None})
    columns = ["person_id", "cohort", "event_type", "L", "R", "prior_state", "new_state", "precision", "observation_mask"]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(audit)
