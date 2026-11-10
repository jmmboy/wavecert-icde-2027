"""Measured scaling and independently sampled possible-world stress checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bootstrap_env  # noqa: F401
from itertools import permutations
import json
import time
import numpy as np
import pandas as pd
import psutil
from interval_core import batch_chain_features, chain_bounds, support_bounds

OUT = Path(__file__).resolve().parents[1] / "simulation" / "core_scaling"
OUT.mkdir(parents=True, exist_ok=True)


def generate(n, m, overlap, seed):
    rng = np.random.default_rng(seed)
    types = [f"event{j}" for j in range(m)]
    ids = np.arange(n)
    center = np.sort(rng.uniform(-4, 0, (n, m)), axis=1)
    # Random type assignments ensure no single universal ordered pattern.
    center = np.take_along_axis(center, np.argsort(rng.random((n, m)), axis=1), axis=1)
    halfwidth = .03 + 1.7 * overlap
    l = np.maximum(-4, center - halfwidth)
    r = np.minimum(0, center + halfwidth)
    have = rng.random((n, m)) < .8
    frame = pd.DataFrame({"person_id": np.repeat(ids, m), "event_type": np.tile(types, n),
                          "L": l.ravel(), "R": r.ravel()})
    return frame.loc[have.ravel()].copy(), ids, types


configs = set()
for n in [1000, 10000, 50000]:
    for k in [2, 3]:
        configs.add((n, 4, k, .5))
for m in [2, 4, 6]:
    for k in [2, 3]:
        if m >= k:
            configs.add((5000, m, k, .5))
for o in [.1, .5, .9]:
    for k in [2, 3]:
        configs.add((5000, 4, k, o))
rows = []
for seed, (n, m, k, overlap) in enumerate(sorted(configs), 20261023):
    logs, ids, types = generate(n, m, overlap, seed)
    patterns = list(permutations(types, k))
    for run in ["first_run", "repeat_same_input"]:
        t0 = time.perf_counter()
        features = batch_chain_features(logs, ids, patterns)
        seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        bounds = support_bounds(features)
        filtering_seconds = time.perf_counter() - t1
        for threshold in [.01, .05, .1]:
            rows.append({"n": n, "m": m, "k": k, "overlap_parameter": overlap,
                         "seed": seed, "run": run, "candidate_patterns": len(patterns),
                         "minimum_possible_support": threshold,
                         "retained_patterns": int((bounds.support_upper >= threshold).sum()),
                         "matching_seconds": seconds, "support_filter_seconds": filtering_seconds,
                         "feature_bytes": int(features.memory_usage(index=True, deep=True).sum()),
                         "process_rss_bytes_after": psutil.Process().memory_info().rss,
                         "pruning": "none; support filtering after full computation"})
pd.DataFrame(rows).to_csv(OUT / "scaling.csv", index=False)

# Continuous-time randomized completions, independent of the solver.  Include
# strict max-span constraints and wide/narrow/touching interval configurations.
rng = np.random.default_rng(89117)
violations, comparisons = 0, 0
started = time.perf_counter()
for j in range(500):
    left = rng.uniform(-5, 1, 3)
    right = left + rng.uniform(.0001, 5, 3)
    logs = pd.DataFrame({"person_id": 0, "event_type": list("ABC"), "L": left, "R": right})
    span = [None, 0.0, .25, 1.0, 3.0][j % 5]
    b = chain_bounds(logs, tuple("ABC"), max_span=span)
    worlds = left + rng.random((200, 3)) * (right - left)
    good = np.all(np.diff(worlds, axis=1) > 0, axis=1)
    if span is not None:
        good &= worlds[:, -1] - worlds[:, 0] <= span
    violations += int(np.sum((good < b["lower"]) | (good > b["upper"])))
    comparisons += len(worlds)
stress = {"seed": 89117, "random_interval_boxes": 500, "sampled_completions": comparisons,
          "logical_bound_violations": violations, "seconds": time.perf_counter() - started,
          "caveat": "Sampled stress complements, and does not replace, exhaustive tests and difference-constraint proof."}
(OUT / "possible_world_stress.json").write_text(json.dumps(stress, indent=2), encoding="utf-8")
print(json.dumps({"scaling_rows": len(rows), **stress}))
assert violations == 0
