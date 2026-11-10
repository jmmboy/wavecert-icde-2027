"""Recover documented follow-up diagnosis routing omitted by the legacy ETL."""
import json,hashlib
from bootstrap_env import ROOT, RAW_DATA
import numpy as np
import pandas as pd
import pyreadstat
BASE = ROOT / "audit"
RAW = RAW_DATA / "CHARLS"
DEST = ROOT / "private" / "event_status" / "CHARLS_event_status.csv.gz"

def ident(s):
    return s.astype(str).str.strip().str.replace(r'\.0$','',regex=True).map(lambda v:v[:9]+'0'+v[9:] if len(v)==11 else v)
def yn(v):return v.map({1:1.,2:0.})
def main():
    BASE.mkdir(parents=True,exist_ok=True)
    ext=pd.read_csv(DEST,dtype={'person_id':str})
    original_sha=hashlib.sha256(DEST.read_bytes()).hexdigest()
    rows=[];audits=[]
    for wave,year in enumerate([2011,2013,2015,2018,2020],1):
        file=next(p for p in (RAW/str(year)).glob('*.dta') if 'health_status' in p.name.lower())
        _,meta=pyreadstat.read_dta(file,metadataonly=True);labels=dict(zip(meta.column_names,meta.column_labels))
        wanted=['ID']
        for j in [1,3,4,8]:
            wanted += [f'{s}{j}_' for s in ['da007_','zda007_','da007_w2_1_','da007_w2_2_','zdiagnosed_','zdisease_','da009_','da010_w2_2_','da002_','da003_']]
        wanted=[c for c in wanted if c in labels]
        d,_=pyreadstat.read_dta(file,usecols=wanted,apply_value_formats=False)
        ids=ident(d.ID);keep=ids.str.fullmatch(r'\d{12}',na=False);d=d.loc[keep].copy();ids=ids.loc[keep]
        out=pd.DataFrame({'person_id':'CHARLS:'+ids,'wave':wave})
        def col(c):return pd.to_numeric(d[c],errors='coerce') if c in d else pd.Series(np.nan,index=d.index)
        for name,j in [('hypertension',1),('diabetes',3),('cancer',4),('stroke',8)]:
            if year==2011:
                v=yn(col(f'da007_{j}_'));rule='Baseline doctor diagnosis DA0071yes/2no. Unknown remains unknown.'
            elif year in [2013,2015]:
                baseline=yn(col(f'da007_{j}_'));new=yn(col(f'da007_w2_2_{j}_'))
                v=baseline.fillna(new)
                confirmed=col(f'zda007_{j}_').eq(1)&col(f'da007_w2_1_{j}_').eq(1)
                v=v.mask(confirmed,1.)
                rule='New interview DA007 doctor1/2; follow-up DA007_W2_2 doctor1/2; preloaded doctor ZDA0071 confirmed by DA007_W2_11 supplies1 because current-new-diagnosis question is legitimately skipped. Missing new diagnosis is not0; prior-negative disagreement alone is not enough for doctor diagnosis; self-known DA008 never promoted to doctor diagnosis.'
            elif year==2018:
                v=yn(col(f'da007_{j}_'))
                prior=col(f'zdiagnosed_{j}_').eq(1)  # physician diagnosis only; not broader Zdisease self-known.
                dispute=col(f'da009_{j}_').eq(99)|col(f'da010_w2_2_{j}_').eq(99)
                v=v.mask(v.isna()&prior&~dispute,1.)
                v=v.mask(dispute,0.)
                rule='Current DA0071/2 first; where structurally skipped use explicit physician-diagnosis preload Zdiagnosed1 (not broader Zdisease); no future carry. Contemporaneous DA009 or DA010_W2_299 explicitly never had->0. Unknown unpreloaded reports stay unknown.'
            else:
                v=yn(col(f'da003_{j}_'));prior=col(f'zdisease_{j}_').eq(1);compare=col(f'da002_{j}_')
                # Flowchart p28: prior physician diagnosis routes to DA002;99
                # disputes prior illness and routes into DA003 for current status.
                v=v.mask(v.isna()&prior&compare.isin([1,2,3]),1.)
                rule='Current DA003 physician1/2 first; explicit preloaded physician diagnosis ZDisease1 plus DA002 comparison1/2/3 supplies1 when DA003 skipped. DA00299 means not affected at PREVIOUS visit, not an automatic current0; require DA003 after that route, otherwise unknown.'
            out[name+'_at_visit']=v
            old=ext.loc[ext.wave.eq(wave),['person_id',name+'_at_visit']].set_index('person_id')[name+'_at_visit']
            before=out.person_id.map(old)
            audits.append({'cohort':'CHARLS','year':year,'wave':wave,'field':name,'source':str(file),'rule':rule,
                           'rows':len(out),'known_before':int(before.notna().sum()),'known_after':int(v.notna().sum()),
                           'new_known_from_legitimate_routing':int((before.isna()&v.notna()).sum()),
                           'changed_known_value':int((before.notna()&v.notna()&before.ne(v)).sum()),
                           'source_dictionary':{c:{'label':labels[c],'codes':meta.variable_value_labels.get(c,{})} for c in wanted if c!='ID' and str(j)+'_' in c}})
        rows.append(out)
    repaired=pd.concat(rows,ignore_index=True)
    assert not repaired.duplicated(['person_id','wave']).any()
    keys=['person_id','wave'];diseasecols=[c for c in repaired if c.endswith('_at_visit')]
    final=ext.drop(columns=diseasecols).merge(repaired,on=keys,how='left',validate='one_to_one')
    assert len(final)==len(ext)
    final.to_csv(DEST,index=False)
    pd.DataFrame([{k:v for k,v in a.items() if k!='source_dictionary'} for a in audits]).to_csv(BASE/'charls_diagnosis_routing_repair.csv',index=False,encoding='utf-8-sig')
    report={'source_panel_modified':False,'old_event_status_sha256':original_sha,'new_event_status_sha256':hashlib.sha256(DEST.read_bytes()).hexdigest(),
            'documentation':{'2013':'家户问卷PDF75–76, DA007_W2_1/2','2015':'家户问卷PDF60–62, DA007_W2_1/2','2018':'家户问卷PDF71–75, Zdiagnosed/DA007/never-had routing','2020':'问卷PDF32 and 问卷流程图PDF28'},
            'no_blanket_forward_fill':True,'no_missing_to_zero':True,'current_or_past_information_only':True,'audits':audits}
    (BASE/'charls_diagnosis_routing_repair.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(audits)[['year','field','known_before','known_after','new_known_from_legitimate_routing','changed_known_value']].to_string(index=False))
if __name__=='__main__':main()
