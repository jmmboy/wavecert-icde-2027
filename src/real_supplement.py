"""Prespecified real-data supplements for the alive-observed interview estimand.

This module writes only its own result directory. It never modifies primary
preparation, primary benchmarks, source panels, or their fitted models.
"""
import bootstrap_env  # noqa: F401
from pathlib import Path
import argparse
import json
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import real_benchmark as rb
from real_prepare import PANELS, ADL, DISEASES, functional_state, robust_slope
from interval_core import batch_chain_features

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'real_supplement'
ROWS=[]
PROCESS=psutil.Process()


def resource():
    mem=PROCESS.memory_info()
    return {'rss_MB':mem.rss/2**20,'peak_working_set_MB':getattr(mem,'peak_wset',mem.rss)/2**20,
            'cpu_seconds_process':time.process_time()}


def save():
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(ROWS).to_csv(OUT/'experiment_long.csv',index=False)


def unavailable(name,cohort,reason,**kwargs):
    ROWS.append({'experiment':name,'cohort':cohort,'status':'not_estimable','reason':reason,**kwargs})
    save()


def paired_ci(e,ix,y,p,p0,weights=None,repetitions=1000):
    family=e.iloc[ix].household_id.astype(str).to_numpy()
    _,unit=np.unique(family,return_inverse=True)
    w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    numer=np.bincount(unit,weights=w*((np.asarray(p)-y)**2-(np.asarray(p0)-y)**2))
    denom=np.bincount(unit,weights=w)
    rng=np.random.default_rng(61023)
    # Same household draws for both predictions, conditional on fitted models.
    draws=rng.integers(0,len(numer),size=(repetitions,len(numer)))
    values=numer[draws].sum(axis=1)/denom[draws].sum(axis=1)
    return {'brier_change_ci_low':float(np.quantile(values,.025)),
            'brier_change_ci_high':float(np.quantile(values,.975)),
            'brier_change_bootstrap_se':float(values.std(ddof=1)),
            'bootstrap_repetitions':repetitions,'bootstrap_scope':'paired household validation bootstrap, conditional on fitted models'}


def record(name,c,e,train,test,p,p0,seconds,features,notes='',weights=None,model=None,reference='main PartialOrder',**kwargs):
    y=e.iloc[test].y.to_numpy(float)
    v=rb.metrics(y,p,weights);b=rb.metrics(y,p0,weights)
    row={'experiment':name,'cohort':c,'status':'completed','train_n':len(train),'test_n':len(test),
         'features':features,'reference':reference,'elapsed_seconds':seconds,'notes':notes,
         'brier_reference':b.get('brier',np.nan),'auc_reference':b.get('auc',np.nan),**v,**resource(),**kwargs}
    row['brier_change']=v.get('brier',np.nan)-b.get('brier',np.nan)
    row['auc_change']=v.get('auc',np.nan)-b.get('auc',np.nan)
    if len(test)>1:
        row.update(paired_ci(e,test,y,p,p0,weights))
    if model is not None:
        row['converged']=bool(model.result_.success) if hasattr(model,'result_') else bool(getattr(model,'converged_',True))
    if weights is not None:
        weights=np.asarray(weights,float)
        row['weighted_ess']=float(weights.sum()**2/np.sum(weights**2))
    ROWS.append(row);save()
    print(json.dumps({'experiment':name,'cohort':c,'n':len(test),'brier':row.get('brier'),'change':row.get('brier_change'),'seconds':seconds}),flush=True)


def can_fit(e,train,test):
    return len(train)>=100 and len(test)>=30 and e.iloc[train].y.nunique()==2 and e.iloc[test].y.nunique()==2


def fit_compare(name,c,e,X,train,test,penalty,p0,labels,notes='',**kwargs):
    if not can_fit(e,train,test):
        unavailable(name,c,'Insufficient observed train/test size or outcome classes under frozen restriction',train_n=len(train),test_n=len(test),notes=notes)
        return
    ts=time.perf_counter();m=rb.fit_model(X,e,train,penalty);p=rb.predict(m,X,e,test)
    record(name,c,e,train,test,p,p0,time.perf_counter()-ts,X.shape[1],notes,model=m,**kwargs)


def ablations(e,f,reps,width,patterns,models):
    for c,bundle in models.items():
        train,test,selected,penalty=(bundle[x] for x in ['train','test','selected','penalty'])
        fullX,_=rb.design(e,f,reps,width,patterns,selected,'PartialOrder')
        p0=rb.predict(bundle['PartialOrder']['model'],fullX,e,test)
        for name,ab in [('no_order_keep_event_bag','no_order'),('no_direction_free_interval_overlap','no_ambiguity'),('no_interval_width','no_width'),('no_observation_covariates','no_observation')]:
            X,labels=rb.design(e,f,reps,width,patterns,selected,'PartialOrder',ablation=ab)
            fit_compare(name,c,e,X,train,test,penalty,p0,labels,
                notes='Same participants, common reversible CTMC, same source-selected penalty; variant-minus-matched-full Brier.')
        fc=f.copy();rc={k:v.copy() for k,v in reps.items()}
        for k,pattern in enumerate(patterns):
            n='>'.join(pattern);fc[n+'__lower']=reps['Midpoint'][:,k];fc[n+'__upper']=reps['Midpoint'][:,k]
            fc[n+'__ambiguous']=np.where(fc[n+'__lower'].notna(),0.,np.nan)
        rc['PairOverlap']=np.where(np.isfinite(reps['PairOverlap']),0.,np.nan)
        collapsed_width=np.where(np.isfinite(width),0.,np.nan)
        X,labels=rb.design(e,fc,rc,collapsed_width,patterns,selected,'PartialOrder')
        fit_compare('collapse_event_intervals_to_midpoints',c,e,X,train,test,penalty,p0,labels,
                    notes='Collapse each event interval to its midpoint: signed direction recomputed, interval overlap and width become0, unknown stays unknown. Keep original training-selected patterns to isolate representation loss.')
        relaxed,_=rb.select_patterns(e,f,patterns,train,disable_stability=True)
        X,labels=rb.design(e,f,reps,width,patterns,relaxed,'PartialOrder')
        fit_compare('no_cross_cohort_stability_filter',c,e,X,train,test,penalty,p0,labels,
                    notes='Require source support in one rather than two cohorts; same maximum chain cap.',selected_chains=len(relaxed))
        for family in ['hypertension','diabetes','stroke','widowhood']:
            keep=[k for k in selected if family not in patterns[k]]
            X,labels=rb.design(e,f,reps,width,patterns,keep,'PartialOrder')
            good=np.array([family not in label and not (family=='widowhood' and label in ['married','married__missing']) for label in labels])
            X=X[:,good];lab=[v for v,g in zip(labels,good) if g]
            fit_compare('delete_event_family_'+family,c,e,X,train,test,penalty,p0,lab,
                        notes='Delete this event family from current state, trend, bag, missingness flags and every associated chain, where present.',selected_chains=len(keep))
    for family in ['retirement','living_arrangement','functional_exposure_events']:
        unavailable('delete_event_family_'+family,'cross_cohort',
                    'This event family is not admitted to the common primary exposure dictionary; an ablation of an absent module cannot be performed. Functional state/trend are required baseline outcome-history controls, not a functional-event exposure.')


def temporal(e,f,reps,width,patterns,models):
    audit=[]
    for c,bundle in models.items():
        discovery=np.flatnonzero((e.cohort==c)&(e.confirmation==0)&(e.response==1))
        if not len(discovery):
            unavailable('early_train_late_test',c,'No source-cohort discovery participants')
            continue
        cutoff=float(np.quantile(e.iloc[discovery].t0,.7))
        train=np.flatnonzero((e.cohort==c)&(e.confirmation==0)&(e.response==1)&(e.t0<=cutoff))
        test=np.flatnonzero((e.cohort==c)&(e.confirmation==1)&(e.response==1)&(e.t0>cutoff))
        audit.append({'cohort':c,'cutoff_t0':cutoff,'train_n':len(train),'test_n':len(test),
                      'households_overlap':len(set(e.iloc[train].household_id)&set(e.iloc[test].household_id))})
        if not can_fit(e,train,test):
            unavailable('early_train_late_test',c,'Frozen 70th-percentile calendar cut has insufficient outcomes or size',train_n=len(train),test_n=len(test),cutoff=cutoff)
            continue
        selected=bundle['selected'];penalty=bundle['penalty']
        # Chain templates come from the main OTHER-cohort discovery fold; all
        # temporal coefficients/transformations are fitted in early local data.
        X0,_=rb.design(e,f,reps,width,patterns,selected,'Unordered')
        ts=time.perf_counter();m0=rb.fit_model(X0,e,train,penalty);p0=rb.predict(m0,X0,e,test)
        X,labels=rb.design(e,f,reps,width,patterns,selected,'PartialOrder')
        m=rb.fit_model(X,e,train,penalty);p=rb.predict(m,X,e,test)
        record('early_train_late_test',c,e,train,test,p,p0,time.perf_counter()-ts,X.shape[1],
               notes='70th-percentile t0 of local discovery fixes early/late cut. Early discovery fits coefficients, late confirmation tests. Chain templates frozen from other-cohort LOCO training.',
               reference='temporally fitted Unordered CTMC',model=m,cutoff_t0=cutoff)
    pd.DataFrame(audit).to_csv(OUT/'temporal_split_audit.csv',index=False)


def read_panels():
    out={}
    for c in ['HRS','ELSA','CHARLS','MHAS']:
        p=pd.read_csv(PANELS/f'{c}.csv.gz',dtype={'person_id':str,'household_id':str})
        ex=pd.read_csv(ROOT/'private'/'event_status'/f'{c}_event_status.csv.gz',dtype={'person_id':str})
        p=p.merge(ex,on=['person_id','wave'],validate='one_to_one',suffixes=('','_event'))
        for d in DISEASES:p[d]=p[d+'_at_visit']
        p['state']=functional_state(p)
        out[c]=p.sort_values(['person_id','date_year'])
    return out


def rebuild_thinned(e,panels):
    thin=e.copy();logs=[];audit=[];reversal=np.zeros(len(e),bool)
    indexlookup=dict(zip(e.person_id,e.index))
    for c,p in panels.items():
        ix=e.index[e.cohort.eq(c)]
        emap=e.loc[ix].set_index('person_id')
        removed=original=0
        for pid,g in p[p.person_id.isin(emap.index)].groupby('person_id',sort=False):
            row=emap.loc[pid];i=int(indexlookup[pid]);t0=float(row.t0)
            full=g[g.date_year<=t0+1e-8].copy();hist=full[full.date_year>=t0-4.25-1e-8].copy()
            if len(hist)<2:continue
            original+=len(hist);drop=hist.index[1:-1];removed+=len(drop)
            retained=full.drop(index=drop);hh=hist.drop(index=drop)
            ts=hh.date_year.to_numpy(float);hs=hh.state.to_numpy(float)
            thin.loc[i,'history_visits']=len(hh);thin.loc[i,'history_span']=t0-ts.min()
            thin.loc[i,'gap_mean']=np.diff(ts).mean();thin.loc[i,'gap_max']=np.diff(ts).max()
            thin.loc[i,'proxy_mean']=hh.proxy.mean()
            thin.loc[i,'adl_mean']=np.nanmean(hs) if np.isfinite(hs).any() else np.nan
            thin.loc[i,'adl_slope']=robust_slope(ts,hs)
            thin.loc[i,'adl_changes']=np.sum(np.diff(hs[np.isfinite(hs)])!=0)
            complete=True
            for disease in DISEASES:
                values=full[disease].to_numpy(float);v=values[np.isfinite(values)]
                reversal[i]|=bool(np.any((v[:-1]==1)&(v[1:]==0)))
                arr=hh[disease].to_numpy(float)
                thin.loc[i,disease+'_mean']=np.nanmean(arr) if np.isfinite(arr).any() else np.nan
                thin.loc[i,disease+'_slope']=robust_slope(ts,arr)
                thin.loc[i,disease+'_unknown']=int(np.any(~np.isfinite(arr)))
                complete &= bool(np.all(np.isfinite(arr)))
                thin.loc[i,disease+'_event']=0. if np.all(np.isfinite(arr)) else np.nan
                vals=retained[disease].to_numpy(float);times=retained.date_year.to_numpy(float)
                yes=np.flatnonzero(vals==1)
                if len(yes):
                    j=int(yes[0]);neg=np.flatnonzero(vals[:j]==0)
                    if len(neg):
                        left,right=float(times[neg[-1]]),float(times[j])
                        if left>=t0-4.25-1e-8 and right>left:
                            logs.append({'person_id':pid,'cohort':c,'event_type':disease,'L':left-t0,'R':right-t0})
                            thin.loc[i,disease+'_event']=1.
            w=hh.widowed.to_numpy(float);m=hh.married.to_numpy(float)
            unknown=np.any(~np.isfinite(w)) or np.any(~np.isfinite(m))
            thin.loc[i,'widowhood_unknown']=int(unknown);thin.loc[i,'widowhood_event']=np.nan if unknown else 0.
            for j in range(1,len(hh)):
                if w[j]==1 and m[j-1]==1 and w[j-1]==0:
                    logs.append({'person_id':pid,'cohort':c,'event_type':'widowhood','L':ts[j-1]-t0,'R':ts[j]-t0})
                    thin.loc[i,'widowhood_event']=1.;break
            thin.loc[i,'event_history_complete']=int(complete)
        audit.append({'cohort':c,'people':len(ix),'original_historical_visits':original,'deleted_intermediate_visits':removed,
                      'retained_endpoints':True,'t0_and_target_unchanged':True,'recomputed_event_intervals':True})
        print(json.dumps({'thinned_cohort':c,'removed':removed,'people':len(ix)}),flush=True)
    return thin,pd.DataFrame(logs,columns=['person_id','cohort','event_type','L','R']),reversal,pd.DataFrame(audit)


def feature_frame(e,logs,patterns):
    # Event dictionary/patterns are frozen by the primary schema; no target support
    # decision is reintroduced after thinning. Completeness remains person/event specific.
    comp={}
    for _,r in e.iterrows():
        comp[r.person_id]={d:not bool(r[d+'_unknown']) for d in DISEASES}
        comp[r.person_id]['widowhood']=not bool(r.widowhood_unknown)
    f=batch_chain_features(logs,e.person_id.tolist(),patterns,completeness=comp).reset_index(drop=True)
    events=sorted({x for pat in patterns for x in pat})
    left=logs.pivot(index='person_id',columns='event_type',values='L').reindex(index=e.person_id,columns=events).to_numpy(float)
    right=logs.pivot(index='person_id',columns='event_type',values='R').reindex(index=e.person_id,columns=events).to_numpy(float)
    widths=[];reps={k:[] for k in ['PairCooccurrence','PairOverlap','PairUnknown']}
    for a,b in patterns:
        i,j=events.index(a),events.index(b);have=np.isfinite(left[:,i])&np.isfinite(left[:,j])
        known=f[a+'>'+b+'__lower'].notna().to_numpy()
        def mask(v):return np.where(known,np.where(have,v,0.),np.nan)
        reps['PairCooccurrence'].append(mask(have))
        reps['PairOverlap'].append(mask((right[:,i]>left[:,j])&(right[:,j]>left[:,i])))
        reps['PairUnknown'].append((~known).astype(float))
        widths.append(np.where(have,np.minimum(right[:,i]-left[:,i],right[:,j]-left[:,j]),np.nan))
    return f,{k:np.column_stack(v) for k,v in reps.items()},np.column_stack(widths) if widths else np.zeros((len(e),0))


def thinning(e,f,reps,width,patterns,models,panels):
    ts=time.perf_counter();et,lt,reversal,audit=rebuild_thinned(e,panels)
    ft,rt,wt=feature_frame(et,lt,patterns)
    audit['rebuild_seconds_total']=time.perf_counter()-ts
    audit.to_csv(OUT/'historical_thinning_audit.csv',index=False)
    for c,b in models.items():
        train,test,selected,penalty=(b[x] for x in ['train','test','selected','penalty'])
        X0,_=rb.design(e,f,reps,width,patterns,selected,'PartialOrder')
        p0=rb.predict(b['PartialOrder']['model'],X0,e,test)
        selected_t,_=rb.select_patterns(et,ft,patterns,train)
        X,lab=rb.design(et,ft,rt,wt,patterns,selected_t,'PartialOrder')
        fit_compare('delete_intermediate_history_visits_rebuild',c,et,X,train,test,penalty,p0,lab,
                    notes='Delete all intermediate historical interviews; preserve first/history endpoint t0 and identical future outcome/person. Rebuild first-positive and widowhood logs, trends, observation covariates and training chain selection from retained visits.',selected_chains=len(selected_t),rebuild_seconds=time.perf_counter()-ts)
        Xbag,lab=rb.design(et,ft,rt,wt,patterns,selected_t,'Unordered')
        t=time.perf_counter();mb=rb.fit_model(Xbag,et,train,penalty);pb=rb.predict(mb,Xbag,et,test)
        Xfull,_=rb.design(et,ft,rt,wt,patterns,selected_t,'PartialOrder');mf=rb.fit_model(Xfull,et,train,penalty);pf=rb.predict(mf,Xfull,et,test)
        record('thinned_history_chain_vs_unordered',c,et,train,test,pf,pb,time.perf_counter()-t,Xfull.shape[1],
              notes='Both models fitted/evaluated on same actually thinned interviews; same original persons/targets retained.',reference='thinned-history Unordered CTMC',model=mf,selected_chains=len(selected_t))
    return reversal


def subgroup_sensitivity(e,f,reps,width,patterns,models,reversal):
    for c,b in models.items():
        train,test,selected,penalty=(b[x] for x in ['train','test','selected','penalty'])
        X,labels=rb.design(e,f,reps,width,patterns,selected,'PartialOrder')
        main=b['PartialOrder']['model'];p0all=rb.predict(main,X,e,test)
        masks={
            'exclude_documented_proxy':~((e.proxy_latest==1)|(e.proxy_mean>0)),
            'exclude_pre_t0_disease_reversal':~reversal,
            'matched_gap_1p5_to_2p5_years':e.gap_mean.between(1.5,2.5)&e.gap_max.le(2.75),
            'exactly_three_historical_visits':e.history_visits.eq(3),
        }
        for name,mask in masks.items():
            tr=train[np.asarray(mask)[train]];te=test[np.asarray(mask)[test]]
            base=rb.predict(main,X,e,te) if len(te) else np.array([])
            fit_compare(name,c,e,X,tr,te,penalty,base,labels,
                        notes='Refit after same pre-t0 restriction in source discovery and held-out target. Reference is frozen main model on precisely the same restricted test participants; unknown proxy status is retained.')
        for name,mask in {
            'followup_before_2020':e.followup_year.lt(2020),
            'followup_2020_or_later':e.followup_year.ge(2020),
            'death_year_recorded_diagnostic':e.known_death_year.notna(),
            'death_year_unrecorded_diagnostic':e.known_death_year.isna(),
            'baseline_adl_normal':e.baseline_state.eq(0),
            'baseline_adl_impaired':e.baseline_state.eq(1),
        }.items():
            te=test[np.asarray(mask)[test]]
            if len(te)<30 or e.iloc[te].y.nunique()<2:
                unavailable(name,c,'Subgroup too small or only one observed outcome class',test_n=len(te));continue
            ts=time.perf_counter()
            p=rb.predict(main,X,e,te)
            xb,_=rb.design(e,f,reps,width,patterns,selected,'Unordered');pb=rb.predict(b['Unordered']['model'],xb,e,te)
            record(name,c,e,train,te,p,pb,time.perf_counter()-ts,X.shape[1],reference='frozen Unordered on matched subgroup',
                   notes='Frozen-model diagnostic subgroup evaluation. Death-year strata reflect retrospective end-of-record ascertainment, are not deployment filters and do not establish survival; calendar split uses actual future interview year.')


def response_ipw(e,f,reps,width,patterns,models):
    audits=[]
    for c,b in models.items():
        train,test,selected,penalty=(b[x] for x in ['train','test','selected','penalty'])
        cohorts=set(e.iloc[train].cohort)
        administrative=e.get('administrative_window_available',pd.Series(1,index=e.index)).eq(1)
        rp=np.flatnonzero((e.cohort!=c)&e.cohort.isin(cohorts)&(e.confirmation==0)&administrative)
        observed=e.iloc[rp].response.to_numpy(int)
        if len(np.unique(observed))<2:
            unavailable('source_response_ipw',c,'Response model has only one class');continue
        ts=time.perf_counter();XX=e[rb.HISTORY+rb.OBS].to_numpy(float)
        response_model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=500,solver='lbfgs'))
        response_model.fit(XX[rp],observed)
        propensity=response_model.predict_proba(XX)[:,1]
        prevalence=float(observed.mean())
        weights=prevalence/np.clip(propensity,.05,.95)
        X,_=rb.design(e,f,reps,width,patterns,selected,'PartialOrder')
        fit=rb.fit_model(X,e,train,penalty,weights=weights[train]);p=rb.predict(fit,X,e,test)
        p0=rb.predict(b['PartialOrder']['model'],X,e,test)
        record('source_response_ipw',c,e,train,test,p,p0,time.perf_counter()-ts,X.shape[1],weights=weights[test],model=fit,
               notes='Exploratory response-transport sensitivity: source-only logistic response model among administrative-window-available risk set, stabilized weights with propensity clipped[.05,.95]. Both predictions scored with same target weights. Requires MAR given measured history, response-model transport and positivity; cannot recover unobserved deaths or MNAR outcomes.')
        for tag,ii in [('source_observed',train),('target_observed',test),('source_response_riskset',rp)]:
            w=weights[ii]
            audits.append({'cohort':c,'sample':tag,'n':len(ii),'propensity_min':float(propensity[ii].min()),'propensity_p01':float(np.quantile(propensity[ii],.01)),
                           'propensity_p50':float(np.median(propensity[ii])),'fraction_below_p05':float(np.mean(propensity[ii]<.05)),
                           'fraction_above_p95':float(np.mean(propensity[ii]>.95)),'weight_max':float(w.max()),'ess':float(w.sum()**2/(w@w))})
    pd.DataFrame(audits).to_csv(OUT/'response_weight_audit.csv',index=False)
    for c in models:
        unavailable('remove_dropout_adjustment',c,'Primary model is the explicitly conditional alive-observed model without an attrition-weight module. There is no adjustment to ablate; source_response_ipw is a separate assumption-dependent sensitivity.')


def auxiliary_audits(panels):
    cognitive=[];mhas=[]
    for c,p in panels.items():
        score=p.get('memory_score',pd.Series(np.nan,index=p.index))
        cognitive.append({'cohort':c,'rows':len(p),'memory_nonmissing':int(score.notna().sum()),'memory_missing_fraction':float(score.isna().mean()),
                          'score_min':float(score.min()) if score.notna().any() else np.nan,'score_max':float(score.max()) if score.notna().any() else np.nan,
                          'task':('10-word immediate+delayed recall; original language/list/mode differ' if c in ['HRS','ELSA'] else '8-word immediate+delayed, list/language and scale differ' if c=='MHAS' else 'Not validated/extracted in current panel'),
                          'cross_cohort_equivalence':'not_established','model_status':'not_estimable',
                          'reason':'No frozen >=3-cohort item/task/administration-equivalent cognitive endpoint; do not pool cohort z-scores as proof of equivalence.'})
    mh=panels['MHAS']
    eligible=followknown=followany=0
    for pid,g in mh.groupby('person_id',sort=False):
        t=g.date_year.to_numpy(float);age=g.age.to_numpy(float)
        earliest=np.searchsorted(t,t-6.25,side='left')
        ok=np.flatnonzero((age>=60)&((np.arange(len(g))-earliest+1)>=3)&((t-t[earliest])>=5.75))
        if not len(ok):continue
        j=int(ok[0]);eligible+=1;t0=t[j]
        future=g[(g.date_year>=t0+2-1e-8)&(g.date_year<=t0+4+1e-8)]
        if len(future):followany+=1;followknown+=int(np.isfinite(future.state.iloc[0]))
    mhas.append({'cohort':'MHAS','history_years':6.,'history_bounds':'[t0-6.25,t0] with >=3 visits and >=5.75y span',
                 'eligible_people':eligible,'future_interview_2to4years':followany,'known_target_2to4years':followknown,
                 'status':'not_estimable' if followknown<100 else 'ready',
                 'reason':'C.2 ends in2018; the earliest feasible dense six-year three-wave history is2012/2015/2018 and lacks2020–2022 target.'})
    pd.DataFrame(cognitive).to_csv(OUT/'cognitive_endpoint_audit.csv',index=False)
    pd.DataFrame(mhas).to_csv(OUT/'mhas_sixyear_history_audit.csv',index=False)
    unavailable('three_cohort_memory_endpoint','cross_cohort','No >=3-cohort comparable endpoint is frozen; cognitive task/item/mode differences remain unresolved.')
    unavailable('mhas_sixyear_history_validation','MHAS','No evaluable [2,4]y future target after feasible six-year history in available C.2',eligible_people=eligible,test_n=followknown)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--phase',choices=['all','ablations','temporal','sensitivity'],default='all');args=parser.parse_args()
    if not args.run:return
    started=time.perf_counter();cpu=time.process_time();OUT.mkdir(parents=True,exist_ok=True)
    e,logs,patterns,f,reps,width=rb.load_features()
    models={p.stem:pickle.loads(p.read_bytes()) for p in sorted((ROOT/'private'/'models').glob('*.pkl'))}
    models={c:m for c,m in models.items() if 'PartialOrder' in m and 'Unordered' in m}
    if len(models)<3:raise RuntimeError('Wait for at least3 completed main LOCO fold model files before supplementary execution')
    if args.phase in ['all','ablations']:ablations(e,f,reps,width,patterns,models)
    if args.phase in ['all','temporal']:temporal(e,f,reps,width,patterns,models)
    if args.phase in ['all','sensitivity']:
        panels=read_panels();reversal=thinning(e,f,reps,width,patterns,models,panels)
        subgroup_sensitivity(e,f,reps,width,patterns,models,reversal)
        response_ipw(e,f,reps,width,patterns,models)
        auxiliary_audits(panels)
    manifest={'status':'completed','phase':args.phase,'cohorts':list(models),'elapsed_seconds':time.perf_counter()-started,
              'cpu_seconds':time.process_time()-cpu,'rows':len(ROWS),'completed':sum(r['status']=='completed' for r in ROWS),
              'not_estimable':sum(r['status']=='not_estimable' for r in ROWS),'estimand':'Actual-delta reversible ADL state among alive-observed interviews in[2,4]y; not unconditional first incidence or death risk',
              'brier_change_sign':'variant-minus-matched-reference; negative improves',
              'input_models_modified':False,'source_panels_modified':False,'participant_outputs_exported':False,
              'conditional_bootstrap':'1000 paired household validation draws per result; does not include training uncertainty',**resource()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
