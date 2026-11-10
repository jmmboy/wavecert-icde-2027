import bootstrap_env
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from real_benchmark import load_features,design,select_patterns,fit_model,predict,metrics,ROOT

def holm(p):
    p=np.asarray(p,float);valid=np.isfinite(p);work=np.where(valid,p,1.);order=np.argsort(work);out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate(work[order]*(len(p)-np.arange(len(p)))));out[~valid]=np.nan
    return out

def family_resample_weights(families,rng):
    unique,inv=np.unique(np.asarray(families,str),return_inverse=True)
    counts=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
    return counts[inv].astype(float)

def clustered_mean_test(values,families):
    d=pd.DataFrame({"v":values,"family":families})
    mean=float(d.v.mean());g=d.groupby("family").agg(s=("v","sum"),n=("v","size"))
    influence=g.s.to_numpy()-mean*g.n.to_numpy();G=len(g);n=len(d)
    se=float(np.sqrt(G/(G-1)*np.sum(influence**2))/n) if G>1 else np.nan
    z=mean/se if se>0 else np.nan
    p=float(2*norm.sf(abs(z))) if np.isfinite(z) else (1. if se==0 and mean==0 else np.nan)
    return mean,se,p,G

def conditional():
    out=ROOT/"results"/"statistics";out.mkdir(exist_ok=True)
    pred=pd.read_csv(ROOT/"private"/"real_predictions.csv.gz",dtype={"person_id":str,"household_id":str})
    methods=[c for c in pred if c in ["Static","History","Unordered","Wave","Midpoint","RightEndpoint","UniformInterval","PartialOrder","StateInterval","ObservationOnly"]]
    rng=np.random.default_rng(130031);rows=[];intervals=[];boot_macro={m:[] for m in methods}
    cohort_groups=list(pred.groupby("cohort",sort=True));R=1000
    brier_boot={c:{m:[] for m in methods} for c,_ in cohort_groups}
    auc_boot={c:{m:[] for m in ["History","Unordered","PartialOrder"]} for c,_ in cohort_groups}
    for r in range(R):
        totals={m:[] for m in methods}
        for c,g in cohort_groups:
            w=family_resample_weights(g.household_id,rng)
            y=g.y.to_numpy(float)
            for m in methods:
                score=float(np.average((g[m].to_numpy()-y)**2,weights=w))
                brier_boot[c][m].append(score);totals[m].append(score)
                if m in auc_boot[c]:
                    auc_boot[c][m].append(metrics(y,g[m],w)["auc"])
        for m in methods:boot_macro[m].append(np.mean(totals[m]))
    for c,g in cohort_groups:
        for m in methods:
            v=np.array(brier_boot[c][m]);base=metrics(g.y,g[m])
            rows.append({"cohort":c,"method":m,"metric":"brier","estimate":base["brier"],"bootstrap_se":v.std(ddof=1),"ci_low":np.quantile(v,.025),"ci_high":np.quantile(v,.975),"repetitions":R,"uncertainty":"conditional on frozen trained model, household bootstrap"})
            if m in auc_boot[c]:
                v=np.array(auc_boot[c][m]);rows.append({"cohort":c,"method":m,"metric":"auc","estimate":base["auc"],"bootstrap_se":v.std(ddof=1),"ci_low":np.nanquantile(v,.025),"ci_high":np.nanquantile(v,.975),"repetitions":R,"uncertainty":"conditional on frozen trained model, household bootstrap"})
        for base in ["History","Unordered"]:
            diff=(g[base].to_numpy()-g.y.to_numpy())**2-(g.PartialOrder.to_numpy()-g.y.to_numpy())**2
            mean,se,p,G=clustered_mean_test(diff,g.household_id)
            v=np.array(brier_boot[c][base])-np.array(brier_boot[c]["PartialOrder"])
            intervals.append({"cohort":c,"comparison":"PartialOrder_vs_"+base,"brier_improvement":mean,"cluster_robust_se":se,"ci_low":np.quantile(v,.025),"ci_high":np.quantile(v,.975),"p_cluster_normal":p,"households":G,"n":len(g),"bootstrap_R":R})
    for m in methods:
        v=np.array(boot_macro[m]);rows.append({"cohort":"macro","method":m,"metric":"brier","estimate":float(np.mean([metrics(g.y,g[m])["brier"] for _,g in cohort_groups])),"bootstrap_se":v.std(ddof=1),"ci_low":np.quantile(v,.025),"ci_high":np.quantile(v,.975),"repetitions":R,"uncertainty":"equal-cohort macro; independently resampled target households"})
    tests=pd.DataFrame(intervals);tests["p_holm"]=holm(tests.p_cluster_normal)
    tests.to_csv(out/"primary_paired_tests.csv",index=False);pd.DataFrame(rows).to_csv(out/"conditional_intervals.csv",index=False)
    np.savez_compressed(out/"conditional_bootstrap_arrays.npz",**{m:np.array(v) for m,v in boot_macro.items()})
    print("conditional bootstrap completed",R,flush=True)

def full_pipeline(R=200):
    out=ROOT/"results"/"statistics";out.mkdir(exist_ok=True)
    e,logs,patterns,f,reps,width=load_features()
    models={p.stem:pickle.load(open(p,"rb")) for p in (ROOT/"private"/"models").glob("*.pkl")}
    rng=np.random.default_rng(550031);records=[];tick=time.perf_counter()
    for r in range(R):
        diffs=[];rep_convergence=[]
        # One common bootstrap population per source cohort/split is reused
        # by every LOCO fold, preserving shared-training covariance.
        shared_weights=np.zeros(len(e))
        for c in sorted(e.cohort.unique()):
            for split in [0,1]:
                ix=np.flatnonzero((e.cohort==c)&(e.confirmation==split)&(e.response==1))
                if len(ix):shared_weights[ix]=family_resample_weights(e.iloc[ix].household_id,rng)
        for target,cache in sorted(models.items()):
            train,test=cache["train"],cache["test"]
            wtrain=shared_weights[train]
            wtest=shared_weights[test]
            # Re-mine source support in the bootstrap population, including
            # duplicated families. All preprocessing and fitting rerun.
            idx=np.repeat(train,wtrain.astype(int))
            selected,_=select_patterns(e,f,patterns,idx)
            # Retune the source-only main penalty and recompute preprocessing
            # on the actually resampled people, not unweighted original rows.
            inner=np.array([int(hashlib.sha256((str(v)+"|inner").encode()).hexdigest()[:8],16)%5 for v in e.iloc[idx].household_id])
            itrain,ival=idx[inner!=0],idx[inner==0]
            baseX,_=design(e,f,reps,width,patterns,selected,"Unordered")
            tuning=[]
            for penalty in [.003,.03,.3]:
                candidate=fit_model(baseX,e,itrain,penalty)
                rep_convergence.append(candidate.converged_)
                tuning.append((metrics(e.iloc[ival].y,predict(candidate,baseX,e,ival))["brier"],penalty))
            selected_penalty=min(tuning)[1]
            scores={};convergence={}
            for method in ["Unordered","PartialOrder"]:
                X,labels=design(e,f,reps,width,patterns,selected,method)
                m=fit_model(X,e,idx,selected_penalty)
                p=predict(m,X,e,test);scores[method]=metrics(e.iloc[test].y,p,wtest)["brier"]
                convergence[method]=m.converged_
            d=scores["Unordered"]-scores["PartialOrder"];diffs.append(d)
            rep_convergence.extend(convergence.values())
            records.append({"rep":r,"cohort":target,"improvement":d,"unordered_brier":scores["Unordered"],"partial_brier":scores["PartialOrder"],"selected":len(selected),"selected_patterns":";".join(">".join(patterns[k]) for k in selected),"penalty":selected_penalty,"converged":all(convergence.values())})
        records.append({"rep":r,"cohort":"macro","improvement":float(np.mean(diffs)),"converged":bool(all(rep_convergence))})
        if r%10==0 or r==R-1:
            pd.DataFrame(records).to_csv(out/"full_pipeline_bootstrap.csv",index=False)
            print(json.dumps({"full_pipeline_bootstrap":r+1,"of":R,"elapsed_seconds":time.perf_counter()-tick}),flush=True)
    table=[]
    for c,g in pd.DataFrame(records).groupby("cohort"):
        table.append({"cohort":c,"mean_improvement":g.improvement.mean(),"bootstrap_se":g.improvement.std(ddof=1),"ci_low":g.improvement.quantile(.025),"ci_high":g.improvement.quantile(.975),"R":len(g),"all_converged":bool(g.converged.all()),"scope":"source/target household resampling; source preprocessing, support mining, penalty tuning and model fitting all rerun"})
    pd.DataFrame(table).to_csv(out/"full_pipeline_intervals.csv",index=False)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["conditional","full"]);p.add_argument("--reps",type=int,default=200);a=p.parse_args()
    conditional() if a.mode=="conditional" else full_pipeline(a.reps)
