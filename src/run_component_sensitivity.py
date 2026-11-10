"""Conditional paired resampling by historical co-residence component.

Does not refit models, change DISC/CONF membership, or replace primary tables.
Only aggregate graph and interval results are written in audit/.
"""
import json,time
from bootstrap_env import ROOT, PANELS
import numpy as np
import pandas as pd

def main():
    (ROOT/"audit").mkdir(parents=True,exist_ok=True)
    tick=time.perf_counter()
    e=pd.read_csv(ROOT/'private/episodes.csv.gz',dtype={'person_id':str,'household_id':str})
    pred=pd.read_csv(ROOT/'private/real_predictions.csv.gz',dtype={'person_id':str,'household_id':str})
    original=pd.read_csv(ROOT/'results/statistics/primary_paired_tests.csv')
    rows=[];graphs=[];bootmacro={k:[] for k in ['PartialOrder_vs_Unordered','PartialOrder_vs_History']}
    rng=np.random.default_rng(9023101)
    for c in ['HRS','ELSA','CHARLS']:
        p=pd.read_csv(PANELS/f'{c}.csv.gz',usecols=['person_id','household_id','wave'],dtype={'person_id':str,'household_id':str}).dropna()
        parent={v:v for v in p.person_id.unique()}
        def find(a):
            while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
            return a
        # Include wave to avoid treating recycled serial household codes at
        # different waves as proof of co-residence.
        for _,g in p.groupby(['wave','household_id'],sort=False):
            values=g.person_id.to_numpy();first=find(values[0])
            for value in values[1:]:parent[find(value)]=first
        ee=e[e.cohort.eq(c)].copy();ee['component']=ee.person_id.map(find)
        parts=ee.groupby('component').confirmation.nunique();cross=set(parts[parts>1].index)
        current=ee.groupby('household_id').confirmation.nunique()
        g=pred[pred.cohort.eq(c)].copy();g['component']=g.person_id.map(find)
        graphs.append({'cohort':c,'eligible_people':len(ee),'landmark_household_groups':ee.household_id.nunique(),
                       'landmark_household_cross_split':int(current.gt(1).sum()),'historical_components':ee.component.nunique(),
                       'historical_components_cross_split':len(cross),'participants_in_cross_split_components':int(ee.component.isin(cross).sum()),
                       'confirmation_test_people':len(g),'confirmation_t0_groups':g.household_id.nunique(),
                       'confirmation_historical_components':g.component.nunique()})
        _,ix=np.unique(g.component,return_inverse=True);count=np.bincount(ix)
        draws=rng.integers(0,len(count),size=(1000,len(count)))
        for base in ['Unordered','History']:
            comp='PartialOrder_vs_'+base
            diff=(g[base].to_numpy()-g.y.to_numpy())**2-(g.PartialOrder.to_numpy()-g.y.to_numpy())**2
            sums=np.bincount(ix,weights=diff)
            v=sums[draws].sum(axis=1)/count[draws].sum(axis=1)
            bootmacro[comp].append(v)
            old=original[(original.cohort==c)&(original.comparison==comp)].iloc[0]
            rows.append({'cohort':c,'comparison':comp,'estimate':float(diff.mean()),'component_bootstrap_se':float(v.std(ddof=1)),
                         'component_ci_low':float(np.quantile(v,.025)),'component_ci_high':float(np.quantile(v,.975)),
                         'original_t0_ci_low':float(old.ci_low),'original_t0_ci_high':float(old.ci_high),'repetitions':1000,
                         'cluster_units':len(count),'n':len(g),'model_refitted':False,'split_changed':False,
                         'scope':'conditional on frozen models; available historical co-residence graph; not a full repair of family/PSU/selection dependence'})
    for comp,arrays in bootmacro.items():
        v=np.mean(arrays,axis=0)
        rows.append({'cohort':'macro','comparison':comp,'estimate':float(np.mean([r['estimate'] for r in rows if r['cohort']!='macro' and r['comparison']==comp])),
                     'component_bootstrap_se':float(v.std(ddof=1)),'component_ci_low':float(np.quantile(v,.025)),
                     'component_ci_high':float(np.quantile(v,.975)),'repetitions':1000,'model_refitted':False,'split_changed':False,
                     'scope':'equal-cohort macro; paired conditional graph-cluster resampling independently within target cohorts'})
    pd.DataFrame(rows).to_csv(ROOT/'audit/component_cluster_sensitivity.csv',index=False)
    pd.DataFrame(graphs).to_csv(ROOT/'audit/household_component_split_audit.csv',index=False)
    (ROOT/'audit/component_cluster_sensitivity_manifest.json').write_text(json.dumps({'status':'completed','repetitions':1000,
          'seed':9023101,'elapsed_seconds':time.perf_counter()-tick,'raw_data_changed':False,'models_refitted':False,
          'discovery_confirmation_split_changed':False,'primary_tables_replaced':False,'person_outputs_exported':False},indent=2),encoding='utf-8')
    print(pd.DataFrame(graphs).to_string(index=False))
    print(pd.DataFrame(rows)[['cohort','comparison','estimate','component_ci_low','component_ci_high']].to_string(index=False))

if __name__=='__main__':main()
