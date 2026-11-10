import bootstrap_env
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import platform
import time
import numpy as np
import pandas as pd
import psutil
from real_benchmark import ROOT,load_features,design,fit_model,predict,metrics

SEEDS=[101,211,307,401,503,601,701,809,907,1009]

def run():
    out=ROOT/"results"/"cost_and_randomness";out.mkdir(exist_ok=True)
    rows=[];tick=time.perf_counter()
    caches={p.stem:pickle.load(open(p,"rb")) for p in (ROOT/"private"/"models").glob("*.pkl")}
    for seed in SEEDS:
        e,logs,patterns,f,reps,width=load_features(seed)
        for c,cache in sorted(caches.items()):
            X,labels=design(e,f,reps,width,patterns,cache["selected"],"UniformInterval")
            t=time.perf_counter();m=fit_model(X,e,cache["train"],cache["penalty"])
            p=predict(m,X,e,cache["test"])
            row={"cohort":c,"seed":seed,"method":"UniformInterval","fit_predict_seconds":time.perf_counter()-t,
                 "converged":m.converged_,**metrics(e.iloc[cache["test"]].y,p)}
            rows.append(row)
        pd.DataFrame(rows).to_csv(out/"seed_metrics.csv",index=False)
        print("completed seed",seed,flush=True)
    pd.DataFrame(rows).groupby(["cohort","method"]).agg(brier_mean=("brier","mean"),brier_seed_sd=("brier","std"),
                    auc_mean=("auc","mean"),auc_seed_sd=("auc","std"),seeds=("seed","size")).reset_index().to_csv(out/"seed_summary.csv",index=False)
    # Same-data/same-code deterministic rerun for one frozen primary fold.
    c=sorted(caches)[0];cache=caches[c]
    e,logs,patterns,f,reps,width=load_features(101)
    X,labels=design(e,f,reps,width,patterns,cache["selected"],"PartialOrder")
    original=cache["PartialOrder"]["model"]
    rerun=fit_model(X,e,cache["train"],cache["penalty"])
    p1,p2=predict(original,X,e,cache["test"]),predict(rerun,X,e,cache["test"])
    reproducibility={"cohort":c,"max_abs_prediction_difference":float(np.max(np.abs(p1-p2))),
                     "exact_prediction_match":bool(np.array_equal(p1,p2)),"timing_excluded_from_match":True,
                     "scope":"one deterministic real LOCO primary fold re-fit; not all simulations rerun"}
    (out/"deterministic_rerun.json").write_text(json.dumps(reproducibility,indent=2),encoding="utf-8")
    # Full-data, fixed-cohort setup scalability. Nested population prefixes
    # are explicitly a cost experiment, not the primary effectiveness result.
    scaling=[]
    c=sorted(caches)[-1];cache=caches[c];train=cache["train"];test=cache["test"]
    X,labels=design(e,f,reps,width,patterns,cache["selected"],"PartialOrder")
    rng=np.random.default_rng(810031);order=rng.permutation(train)
    for fraction in [.25,.5,1.0]:
        use=order[:max(100,int(len(order)*fraction))]
        t=time.perf_counter();cpu=time.process_time();m=fit_model(X,e,use,cache["penalty"])
        training=time.perf_counter()-t;cpu_s=time.process_time()-cpu
        t=time.perf_counter()
        for _ in range(30):p=predict(m,X,e,test)
        infer=(time.perf_counter()-t)/30
        memory=psutil.Process().memory_info()
        scaling.append({"train_n":len(use),"fraction":fraction,"features":X.shape[1],"fit_seconds":training,
                        "cpu_seconds":cpu_s,"inference_ms_per_1000":infer/max(len(test),1)*1e6,
                        "sampled_rss_MiB":memory.rss/2**20,"process_peak_rss_MiB":getattr(memory,"peak_wset",memory.rss)/2**20,
                        "converged":m.converged_,"purpose":"nested sample sizes for computational scaling only"})
    pd.DataFrame(scaling).to_csv(out/"model_scaling.csv",index=False)
    hw={"platform":platform.platform(),"machine":platform.machine(),"processor":platform.processor(),
        "physical_cores":psutil.cpu_count(logical=False),"logical_cores":psutil.cpu_count(),
        "ram_GiB":psutil.virtual_memory().total/2**30,"python":platform.python_version(),
        "numpy":np.__version__,"pandas":pd.__version__,"seeds":SEEDS,"elapsed_seconds":time.perf_counter()-tick,
        "deterministic_methods_seed_sd":"not applicable: exact deterministic optimizer; do not count identical reruns as independent observations",
        "model_bytes":{p.name:p.stat().st_size for p in (ROOT/"private"/"models").glob("*.pkl")}}
    (out/"runtime_manifest.json").write_text(json.dumps(hw,indent=2),encoding="utf-8")

if __name__=="__main__":run()
