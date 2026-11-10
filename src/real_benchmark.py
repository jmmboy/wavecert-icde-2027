"""All transforms/mining fitted within source discovery cohorts; targets stay held out."""
import bootstrap_env
from pathlib import Path
from itertools import permutations
import argparse
import hashlib
import json
import pickle
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score
from interval_core import batch_chain_features
from interval_model import ReversiblePanelCTMC

ROOT = Path(__file__).resolve().parents[1]
STATIC = ["age", "female", "education_years", "married", "smoking", "self_rated_health", "adl_level",
          "hypertension", "diabetes", "stroke"]
HISTORY = STATIC + ["adl_mean", "adl_slope", "adl_changes"] + [f"{d}_{s}" for d in ["hypertension", "diabetes", "stroke"] for s in ["mean", "slope"]]
OBS = ["history_visits", "history_span", "gap_mean", "gap_max", "proxy_latest", "proxy_mean"]
BAG = [f"{d}_{s}" for d in ["hypertension", "diabetes", "stroke"] for s in ["event", "unknown"]]
BAG += ["widowhood_event", "widowhood_unknown"]
METHODS = ["Static", "History", "Unordered", "Wave", "Midpoint", "RightEndpoint", "UniformInterval", "PartialOrder", "StateInterval", "ObservationOnly"]

def metrics(y, p, weights=None):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-7, 1-1e-7)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    good = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    y, p, w = y[good], p[good], w[good]
    if not len(y):
        return {"n": 0}
    a = float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) == 2 else np.nan
    b = np.array([0., 1.])
    Z = np.c_[np.ones(len(y)), logit(p)]
    for _ in range(35):
        q = expit(Z @ b)
        H = Z.T @ ((w*q*(1-q))[:,None]*Z) + np.diag([1e-7, 1e-6])
        step = np.clip(np.linalg.solve(H, Z.T @ (w*(y-q))), -2, 2)
        b += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return {"n": len(y), "events": int(y.sum()), "prevalence": float(np.average(y, weights=w)),
            "brier": float(np.average((p-y)**2, weights=w)), "auc": a,
            "auprc": float(average_precision_score(y, p, sample_weight=w)),
            "calibration_intercept": float(b[0]), "calibration_slope": float(b[1]),
            "mean_probability": float(np.average(p, weights=w)),
            "logloss": float(np.average(-y*np.log(p)-(1-y)*np.log(1-p), weights=w))}

def attach_missing(X):
    X = np.asarray(X, float)
    return np.c_[X, (~np.isfinite(X)).astype(float)]

def load_features(seed=101):
    e = pd.read_csv(ROOT/"private"/"episodes.csv.gz", dtype={"person_id": str, "household_id": str})
    logs = pd.read_csv(ROOT/"private"/"interval_logs.csv.gz", dtype={"person_id": str})
    e = e.sort_values(["cohort", "person_id"]).reset_index(drop=True)
    # Fixed from original-question/variable audit, before target data mining.
    # Counts and supports are selected ONLY within each outer training set.
    eligible_events = ["diabetes", "hypertension", "stroke", "widowhood"]
    logs = logs[logs.event_type.isin(eligible_events)]
    patterns = list(permutations(eligible_events, 2))
    completeness = {r.person_id:{"diabetes":not bool(r.diabetes_unknown),"hypertension":not bool(r.hypertension_unknown),
                               "stroke":not bool(r.stroke_unknown),"widowhood":not bool(r.widowhood_unknown)} for r in e.itertuples()}
    features = batch_chain_features(logs, e.person_id.tolist(), patterns, completeness=completeness)
    left = logs.pivot(index="person_id", columns="event_type", values="L").reindex(index=e.person_id, columns=eligible_events).to_numpy(float)
    right = logs.pivot(index="person_id", columns="event_type", values="R").reindex(index=e.person_id, columns=eligible_events).to_numpy(float)
    rng = np.random.default_rng(seed)
    representation = {key: [] for key in ["Wave", "Midpoint", "RightEndpoint", "UniformInterval", "StateInterval", "StateOverlap", "PairCooccurrence", "PairOverlap", "PairUnknown"]}
    widths = []
    for a,b in patterns:
        i,j = eligible_events.index(a), eligible_events.index(b)
        name = a+">"+b
        known = features[name+"__lower"].notna().to_numpy()
        have = np.isfinite(left[:,i]) & np.isfinite(left[:,j])
        mid_a, mid_b = (left[:,i]+right[:,i])/2, (left[:,j]+right[:,j])/2
        def mask(value):
            return np.where(known, np.where(have, value, 0.), np.nan)
        representation["Midpoint"].append(mask(mid_a < mid_b))
        representation["RightEndpoint"].append(mask(right[:,i] < right[:,j]))
        # Wave order uses first-positive visit endpoints and refuses same-wave ties.
        representation["Wave"].append(mask(right[:,i] < right[:,j]))
        ua = left[:,i,None] + (right[:,i]-left[:,i])[:,None]*rng.random((len(e),64))
        ub = left[:,j,None] + (right[:,j]-left[:,j])[:,None]*rng.random((len(e),64))
        representation["UniformInterval"].append(mask((ua < ub).mean(axis=1)))
        # State-interval adaptation: event-positive state persists from first report to t0;
        # compare its known starts and overlap duration. NOT original C-TPM reproduction.
        representation["StateInterval"].append(mask(right[:,i] < right[:,j]))
        representation["StateOverlap"].append(mask(np.minimum(-right[:,i],-right[:,j])))
        representation["PairCooccurrence"].append(mask(have))
        representation["PairOverlap"].append(mask((right[:,i]>left[:,j])&(right[:,j]>left[:,i])))
        representation["PairUnknown"].append((~known).astype(float))
        widths.append(np.where(have, np.minimum(right[:,i]-left[:,i],right[:,j]-left[:,j]), np.nan))
    representations = {k: np.column_stack(v) if v else np.zeros((len(e),0)) for k,v in representation.items()}
    width = np.column_stack(widths) if widths else np.zeros((len(e),0))
    return e, logs, patterns, features.reset_index(drop=True), representations, width

def select_patterns(e, features, patterns, train, disable_stability=False):
    chosen, audit = [], []
    for k,p in enumerate(patterns):
        name = ">".join(p)
        supports=[]
        for c in sorted(e.loc[train,"cohort"].unique()):
            ix = train[e.iloc[train].cohort.to_numpy()==c]
            l = features.iloc[ix][name+"__lower"].to_numpy(float)
            u = features.iloc[ix][name+"__upper"].to_numpy(float)
            good = np.isfinite(l)&np.isfinite(u)
            certain = float(np.nansum(l))
            possible = float(np.nansum(u))
            eligible = int(good.sum())
            ok = certain >= max(10, .0025*eligible)
            supports.append(ok)
            audit.append({"pattern": name,"cohort":c,"certain_count":certain,"possible_count":possible,
                          "eligible":eligible,"support_lower":certain/eligible if eligible else np.nan,
                          "support_upper":possible/eligible if eligible else np.nan,"passes":ok})
        if (sum(supports)>=2) or (disable_stability and sum(supports)>=1):
            chosen.append(k)
    # Fixed ordering and cap depend only on source support; never on target/outcome values.
    scores={k: float(np.nansum(features.iloc[train][">".join(patterns[k])+"__lower"])) for k in chosen}
    chosen=sorted(chosen,key=lambda k:(-scores[k],patterns[k]))[:12]
    return chosen, audit

def design(e, features, representations, width, patterns, selected, method, ablation=None):
    columns = STATIC if method=="Static" else HISTORY+OBS if method not in ["ObservationOnly"] else OBS
    if method=="Unordered" or method in ["Wave","Midpoint","RightEndpoint","UniformInterval","PartialOrder","StateInterval"]:
        columns = HISTORY+OBS+BAG
    if ablation=="no_observation": columns=[c for c in columns if c not in OBS]
    X=e[columns].to_numpy(float)
    labels=list(columns)
    # Every order-based representation and the strong unordered control gets
    # the SAME pair presence, direction-free ambiguity, width and missingness.
    # Thus the primary comparison does not confound order with an interaction.
    if method in ["Unordered","Wave","Midpoint","RightEndpoint","UniformInterval","PartialOrder","StateInterval"] and selected:
        common=[];common_labels=[];seen=set()
        for k in selected:
            pair=tuple(sorted(patterns[k]))
            if pair in seen:continue
            seen.add(pair)
            for rep,suffix in [("PairCooccurrence","pair_present"),("PairOverlap","pair_overlap"),("PairUnknown","pair_unknown")]:
                if ablation=="no_ambiguity" and rep=="PairOverlap":continue
                common.append(representations[rep][:,k]);common_labels.append("&".join(pair)+"__"+suffix)
            if ablation!="no_width":
                common.append(width[:,k]);common_labels.append("&".join(pair)+"__pair_width")
        X=np.c_[X,np.column_stack(common)];labels+=common_labels
    if method=="PartialOrder":
        if ablation=="no_order":
            selected=[]
        if selected:
            directions=[];direction_labels=[];seen=set()
            for k in selected:
                pair=tuple(sorted(patterns[k]))
                if pair in seen:continue
                seen.add(pair)
                a,b=pair
                # Shared cooccurrence/overlap/unknown encodes the order-free
                # uncertainty; one signed column adds only identified direction.
                directions.append(np.nan_to_num(features[a+">"+b+"__lower"].to_numpy(float)-features[b+">"+a+"__lower"].to_numpy(float),nan=0.))
                direction_labels.append(a+">"+b+"__certain_direction")
            X=np.c_[X,np.column_stack(directions)];labels+=direction_labels
    elif method in representations and selected:
        X=np.c_[X,np.nan_to_num(representations[method][:,selected],nan=0.)];labels += [">".join(patterns[k])+"__"+method for k in selected]
        if method=="StateInterval":
            # Persistent post-report intervals end at t0; their overlap is the
            # shorter duration since the two reports. Recoverable disease-report
            # states are a simplifying assumption of this adaptation only.
            duration=representations["StateOverlap"][:,selected]
            X=np.c_[X,np.nan_to_num(duration,nan=0.)];labels += [">".join(patterns[k])+"__state_overlap_duration" for k in selected]
    return attach_missing(X), labels+[x+"__missing" for x in labels]

def fit_model(X,e,train,penalty=.03,weights=None,warm=None):
    y=e.iloc[train].y.to_numpy(int)
    obs={"person":np.arange(len(train)),"delta":e.iloc[train].delta.to_numpy(float),
         "start":e.iloc[train].baseline_state.to_numpy(int),"end":y}
    m=ReversiblePanelCTMC(penalty=penalty,max_iter=120,tol=1e-7,include_death=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="All-NaN slice encountered",category=RuntimeWarning)
        m.fit(X[train],obs,sample_weight=weights,initial_params=warm)
    return m

def predict(m,X,e,test):
    return m.predict_state(X[test],t=e.iloc[test].delta.to_numpy(float),start_state=e.iloc[test].baseline_state.to_numpy(int))[:,1]

def run():
    import psutil
    start=time.perf_counter();cpu=time.process_time()
    results=ROOT/"results"/"real"; results.mkdir(parents=True,exist_ok=True)
    private=ROOT/"private"/"models";private.mkdir(exist_ok=True)
    e,logs,patterns,f,reps,width=load_features()
    available=e[e.response==1].groupby("cohort").size()
    cohorts=available[available>=100].index.tolist()
    if len(cohorts)<3:
        raise ValueError("Fewer than three eligible real cohorts; cross-cohort primary analysis cannot run")
    all_results=[];prediction_frames=[];selection=[];resource=[];all_models={}
    for target in cohorts:
        train=np.flatnonzero((e.cohort!=target)&(e.confirmation==0)&(e.response==1)&e.cohort.isin(cohorts))
        test=np.flatnonzero((e.cohort==target)&(e.confirmation==1)&(e.response==1))
        selected,audit=select_patterns(e,f,patterns,train)
        for row in audit:row["heldout_cohort"]=target;row["selected"]=row["pattern"] in [">".join(patterns[k]) for k in selected]
        selection.extend(audit)
        predictions=e.iloc[test][["person_id","household_id","cohort","y","delta","baseline_state","t0"]].copy()
        predictions["source_index"]=test
        # Small penalty selection only within source discovery cohorts.
        family=e.iloc[train].household_id.map(lambda x:int(hashlib.sha256((str(x)+"|inner").encode()).hexdigest()[:8],16)%5).to_numpy()
        itrain=train[family!=0];ival=train[family==0]
        baseX,_=design(e,f,reps,width,patterns,selected,"Unordered")
        tuning=[]
        for penalty in [.003,.03,.3]:
            m=fit_model(baseX,e,itrain,penalty)
            score=metrics(e.iloc[ival].y,predict(m,baseX,e,ival))["brier"]
            tuning.append((score,penalty))
        penalty=min(tuning)[1]
        all_models[target]={"train":train,"test":test,"selected":selected,"penalty":penalty,"tuning":tuning}
        for method in METHODS:
            X,labels=design(e,f,reps,width,patterns,selected,method)
            ts=time.perf_counter();m=fit_model(X,e,train,penalty);fit_seconds=time.perf_counter()-ts
            ts=time.perf_counter();p=predict(m,X,e,test);infer_seconds=time.perf_counter()-ts
            row={"cohort":target,"method":method,"train_n":len(train),"test_n":len(test),"selected_chains":len(selected),
                 "features":X.shape[1],"penalty":penalty,"fit_seconds":fit_seconds,"predict_seconds":infer_seconds,
                 "inference_ms_per_1000":infer_seconds/max(len(test),1)*1e6,"rss_MB":psutil.Process().memory_info().rss/2**20,
                 "process_peak_rss_MB":getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)/2**20,
                 "iterations":m.n_iter_,"gradient_max":m.gradient_max_,"optimizer_message":m.optimizer_message_,
                 "converged":bool(m.result_.success) if hasattr(m,"result_") else bool(getattr(m,"converged_",True))}
            row.update(metrics(e.iloc[test].y,p));all_results.append(row);resource.append(row.copy())
            predictions[method]=p
            all_models[target][method]={"model":m,"labels":labels}
            pd.DataFrame(all_results).to_csv(results/"metrics.csv",index=False)
            print(json.dumps({"target":target,"method":method,"n_train":len(train),"n_test":len(test),"chains":len(selected),"brier":row["brier"],"fit_seconds":fit_seconds}),flush=True)
        prediction_frames.append(predictions)
        with open(private/(target+".pkl"),"wb") as fh:pickle.dump(all_models[target],fh)
    pd.concat(prediction_frames).to_csv(ROOT/"private"/"real_predictions.csv.gz",index=False)
    pd.DataFrame(selection).to_csv(results/"training_chain_selection.csv",index=False)
    met=pd.DataFrame(all_results)
    summary=met.groupby("method").agg(cohorts=("cohort","nunique"),auc_mean=("auc","mean"),auc_cohort_sd=("auc","std"),
                    brier_mean=("brier","mean"),brier_cohort_sd=("brier","std"),calibration_slope_mean=("calibration_slope","mean"),
                    fit_seconds_total=("fit_seconds","sum"),features_mean=("features","mean")).reset_index()
    summary.to_csv(results/"method_summary.csv",index=False)
    pd.DataFrame(resource).to_csv(results/"resources.csv",index=False)
    (results/"run_summary.json").write_text(json.dumps({"status":"completed","cohorts":cohorts,"elapsed_seconds":time.perf_counter()-start,
                     "cpu_seconds":time.process_time()-cpu,"estimand":"alive-interviewed functional status in 2–4y at actual assessment time",
                     "primary_items":["dress","bath","eat","toilet"],"source_participant_data_exported":False,
                     "algorithm_adaptations":"UniformInterval is independent-uniform possible-world representation, StateInterval is simplified state-interval adaptation, neither is a full original-software reproduction.",
                     "private_files":["real_predictions.csv.gz","models/*.pkl"]},ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--run",action="store_true");args=parser.parse_args()
    if args.run:run()
