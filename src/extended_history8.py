"""Exploratory data-support sensitivity, preserves the complete four-year primary run.

Motivation is the zero eligible source chains under the frozen four-year design.
This changes landmark population as well as available history, so it is NOT a
paired causal estimate of increasing history length or a replacement primary test.
"""
import bootstrap_env
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/"extended_history8"

def run():
    started=time.perf_counter()
    (EXT/"private"/"event_status").mkdir(parents=True,exist_ok=True)
    (EXT/"src").mkdir(exist_ok=True)
    for source in (ROOT/"private"/"event_status").glob("*_event_status.csv.gz"):
        shutil.copy2(source,EXT/"private"/"event_status"/source.name)
    base=(ROOT/"src"/"real_prepare.py").read_text(encoding="utf-8")
    source=base.replace('ROOT = Path(__file__).resolve().parents[1]',"ROOT = Path("+repr(str(EXT))+")")
    source=source.replace("4.25","8.25")
    old="eligible = np.flatnonzero((age_array >= 60) & ((np.arange(len(g))-earliest+1)>=3) & ((times-times[earliest])>=3.75))"
    new="eligible = np.flatnonzero((age_array >= 60) & ((np.arange(len(g))-earliest+1)>=3) & ((times-times[earliest])>=3.75) & (times+2.0 <= cohort_admin_end+1e-8))"
    if old not in source:raise RuntimeError("Base preparer changed; review sensitivity transformation")
    source=source.replace(old,new).replace("j = int(eligible[0])","j = int(eligible[-1])")
    source=source.replace('for pid, g in p.groupby("person_id", sort=False):','cohort_admin_end = float(p.date_year.max())\n        for pid, g in p.groupby("person_id", sort=False):')
    snapshot=EXT/"src"/"prepare_extended_snapshot.py";snapshot.write_text(source,encoding="utf-8")
    amendment={"scope":"exploratory only; primary four-year result retained unchanged", "reason":"zero source-stable certain chains in every primary fold",
               "history_max_years":8.25,"history_min_span":3.75,"landmark":"latest administratively eligible interview, age>=60, >=3 history interviews",
               "future_target":"same first alive-observed assessment in [2,4] years; four common ADL items",
               "change_limitation":"landmark calendar, survival selection and population also change; not an isolated history-length effect",
               "primary_result_replacement":False,"support_threshold_unchanged":True,
               "snapshot_sha256":hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    (EXT/"exploratory_amendment.json").write_text(json.dumps(amendment,indent=2),encoding="utf-8")
    env=dict(os.environ);env["PYTHONPATH"]=str(ROOT/"src")
    result=subprocess.run([sys.executable,str(snapshot)],env=env)
    if result.returncode:raise RuntimeError("Extended data preparation failed")
    import finalize_real_inputs as fi
    fi.ROOT=EXT;fi.run()
    import real_benchmark as rb
    rb.ROOT=EXT;rb.run()
    import real_statistics as rs
    rs.ROOT=EXT;rs.conditional()
    (EXT/"execution_summary.json").write_text(json.dumps({"status":"completed","elapsed_seconds":time.perf_counter()-started,"primary_untouched":True}),encoding="utf-8")

if __name__=="__main__":run()
