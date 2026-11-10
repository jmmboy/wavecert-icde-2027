"""Postprocessing repair: pandas' default NA vocabulary includes literal 'null'."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bootstrap_env  # noqa: F401
import json
import time
import pandas as pd
from simulation_runner import summarize

started = time.perf_counter()
base = Path(__file__).resolve().parents[1] / "simulation"
fixed = []
for stage in ["pilot", "main", "three_event_extension"]:
    out = base / stage
    raw = pd.read_csv(out / "replicate_long.csv", keep_default_na=False, na_values=[""])
    summarize(raw).to_csv(out / "scenario_summary_mcse.csv", index=False)
    numeric = [c for c in raw.select_dtypes("number") if c not in ["repeat", "seed", "chain_length"]]
    pooled = raw.groupby(["stage", "scenario", "repeat", "domain", "chain_length", "method"])[numeric].mean().reset_index()
    pooled["pattern"] = "all_patterns_macro"
    summarize(pooled).to_csv(out / "scenario_macro_summary_mcse.csv", index=False)
    manifestpath = out / "manifest.json"
    manifest = json.loads(manifestpath.read_text(encoding="utf-8"))
    manifest["observation_scope_note"] = "Main informative_dropout and informative_visits modify prospective ascertainment only; historical informative observation tested separately."
    manifest["summary_postprocessing_repair"] = "Preserve literal scenario 'null' with keep_default_na=False, na_values=['']; raw simulation rows unchanged; summaries recomputed."
    manifestpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fixed.append({"stage": stage, "scenarios": sorted(raw.scenario.unique()), "raw_rows": len(raw)})
result = {"reason": "pandas default NA parsing converted literal scenario null to missing; original raw CSV is intact",
          "fix": "Recompute summaries with keep_default_na=False and explicit blank NA; no resimulation, no result selection",
          "affected": fixed, "seconds": time.perf_counter() - started,
          "command": "python tests/repair_null_summary.py",
          "subsequent_read_rule": "pd.read_csv(path, keep_default_na=False, na_values=[''])"}
(base / "summary_repair_log.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result), flush=True)
