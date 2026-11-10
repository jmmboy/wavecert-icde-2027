"""Create current-wave event status sidecars; never modify source/old panels."""
import json, time
from bootstrap_env import PANELS, RAW_DATA as SOURCE, ROOT
import pandas as pd
import numpy as np
import pyreadstat

BASE = ROOT / "audit"
DEST = ROOT / "private" / "event_status"
AUDIT={'created':'2026-10-23','individual_data_scope':'local private/event_status only','dictionaries':{},'cohorts':{},'issues':[]}

def clean(s,valid=None):
    s=pd.to_numeric(s,errors='coerce').astype(float)
    return s.where(s.isin(valid)) if valid is not None else s

def pid_h(d):
    return 'HRS_'+d.HHID.astype(str).str.strip().str.replace(r'\.0$','',regex=True).str.zfill(6)+'_'+d.PN.astype(str).str.strip().str.replace(r'\.0$','',regex=True).str.zfill(3)

def put(c,rows,statuses):
    p=pd.concat(rows,ignore_index=True)
    diseases=['hypertension','diabetes','stroke','cancer']
    panel=pd.read_csv(PANELS/f'{c}.csv.gz',usecols=['person_id','wave']+diseases,dtype={'person_id':str})
    assert not p.duplicated(['person_id','wave']).any()
    merge=panel.merge(p,on=['person_id','wave'],how='left',validate='one_to_one',indicator=True)
    for name in diseases:
        if name+'_at_visit' not in merge:merge[name+'_at_visit']=merge[name]
    merge=merge.drop(columns=diseases)
    AUDIT['cohorts'][c]={'panel_rows':len(panel),'extension_rows':len(p),'exact_join_matches':int(merge._merge.eq('both').sum()),
                         'field_nonmissing':merge.drop(columns='_merge').notna().sum().to_dict(),
                         'event_value_counts':{col:merge[col].value_counts(dropna=False).to_dict() for col in ['widowed','retired_any','retired_full']}}
    merge.drop(columns='_merge').to_csv(DEST/f'{c}_event_status.csv.gz',index=False)
    if statuses:
        st=pd.concat(statuses,ignore_index=True)
        st.to_csv(DEST/f'{c}_interview_status.csv.gz',index=False)
        AUDIT['cohorts'][c]['interview_status_by_wave']={str(w):s.interview_status.value_counts(dropna=False).to_dict() for w,s in st.groupby('wave')}
    print(c,'sidecar',len(merge),'joined',int(merge._merge.eq('both').sum()),flush=True)

def hrs():
    cache=ROOT/'private'/'HRS_cache'
    source=cache/'randhrs1992_2022v1.sas7bdat'
    cols=['HHID','PN']+[f'R{w}{s}' for w in range(4,17) for s in ['MSTAT','SAYRET','LBRF','IWSTAT']]
    t=time.perf_counter();d,metadata=pyreadstat.read_sas7bdat(source,usecols=cols)
    labels=metadata.column_names_to_labels
    print('HRS selected raw SAS read seconds',time.perf_counter()-t,flush=True)
    d['person_id']=pid_h(d); rows=[];statuses=[]
    for w in range(4,17):
        p=pd.DataFrame({'person_id':d.person_id,'wave':w})
        p['marital_status']=clean(d[f'R{w}MSTAT'],[1,2,3,4,5,6,7,8])
        p['widowed']=p.marital_status.eq(7).astype(float).where(p.marital_status.notna())
        raw=clean(d[f'R{w}SAYRET'],[0,1,2])
        p['retirement_status_raw']=raw;p['retired_any']=raw.map({0:0.,1:1.,2:1.});p['retired_full']=raw.map({0:0.,1:1.,2:0.})
        p['labor_status_raw']=clean(d[f'R{w}LBRF'],[1,2,3,4,5,6,7])
        rows.append(p)
        statuses.append(pd.DataFrame({'person_id':d.person_id,'wave':w,'interview_status':clean(d[f'R{w}IWSTAT'])}))
        AUDIT['dictionaries'][f'HRS:{w}']={'source':str(source),'marital_status':{'column':f'R{w}MSTAT','label':labels[f'R{w}MSTAT'],'codes':'1 married;2 married spouse absent;3 partnered;4 separated;5 divorced;6 separated/divorced;7 widowed;8 never married'},
             'retired_any':{'column':f'R{w}SAYRET','label':labels[f'R{w}SAYRET'],'rule':'0 not retired->0;1 completely/2 partly retired->1;3 irrelevant and SAS special missing->unknown','question':'At this time do you consider yourself completely retired, partly retired, or not retired at all?','skip':'Proxy interview .S; DK/RF/other specials remain unknown; no nonemployment recode'},
             'codebook':str(cache/'randhrs1992_2022v1.pdf'),'pages_printed':[1778,1779,1780,1781]}
    put('HRS',rows,statuses)

def harmonized(c):
    source=SOURCE/'ELSA'/'h_elsa_g3.dta' if c=='ELSA' else SOURCE/'MHAS'/'H_MHAS_c2.dta'
    _,m=pyreadstat.read_dta(source,metadataonly=True);labels=dict(zip(m.column_names,m.column_labels)); names=set(m.column_names)
    idcol='idauniqc' if c=='ELSA' else 'rahhidnp';last=9 if c=='ELSA' else 5;rsuffix='lbrf_e' if c=='ELSA' else 'lbrf_m'
    suffixes=['mstat','mstath',rsuffix,'iwstat']
    if c=='ELSA':suffixes+=['hibpe','diabe','stroke','cancre','hibpf','diabf','strokf','cancrf']
    cols=[idcol]+[f'r{w}{s}' for w in range(1,last+1) for s in suffixes if f'r{w}{s}' in names]
    d,_=pyreadstat.read_dta(source,usecols=cols,apply_value_formats=False,user_missing=False)
    d['person_id']=c+'_'+d[idcol].astype(str).str.strip().str.replace(r'\.0$','',regex=True)
    rows=[];statuses=[]
    for w in range(1,last+1):
        p=pd.DataFrame({'person_id':d.person_id,'wave':w})
        p['marital_status']=clean(d[f'r{w}mstat'],[1,2,3,4,5,6,7,8])
        p['marital_status_legal']=clean(d[f'r{w}mstath'],[1,2,3,4,5,6,7,8]) if f'r{w}mstath' in d else np.nan
        if c=='ELSA':
            restored={}
            for name,es,fs in [('hypertension','hibpe','hibpf'),('diabetes','diabe','diabf'),('stroke','stroke','strokf'),('cancer','cancre','cancrf')]:
                ev=clean(d[f'r{w}{es}'],[0,1]);flag=clean(d[f'r{w}{fs}'])
                # Code6 explicitly means later dispute rewrote a YES in this wave to NO.
                # Undo vendor use of future information; contemporaneous code3 dispute remains NO.
                p[name+'_at_visit']=ev.where(~flag.eq(6),1.)
                restored[name]=int(flag.eq(6).sum())
            AUDIT['dictionaries'][f'ELSA:{w}:asof_disease_reconstruction']={'source':str(source),'n_restored_yes':restored,
              'rule':'RwconditionF6: vendor later dispute changed current-wave yes->no; restore1. Flag3 contemporaneous dispute remains0. Flag0 unchanged. No future report is used to revise a current-wave clinical belief.',
              'source_pdf_pages':[274,275,276],
              'selected':{f'r{w}{s}':{'label':labels[f'r{w}{s}'],'codes':m.variable_value_labels.get(f'r{w}{s}',{})} for s in ['hibpe','diabe','stroke','cancre','hibpf','diabf','strokf','cancrf']}}
        # Current implied partnership can override legal widowed status; use legal self-report where provided.
        legal=p.marital_status_legal.fillna(p.marital_status)
        p['widowed']=legal.eq(7).astype(float).where(legal.notna())
        raw=clean(d[f'r{w}{rsuffix}'],[1,2,3,4,5,6,7] if c=='ELSA' else [1,2,3,4,5,6])
        p['retirement_status_raw']=raw;p['labor_status_raw']=raw
        if c=='ELSA':
            p['retired_any']=raw.map({1:0.,2:0.,3:0.,4:1.,5:1.,6:0.,7:0.});p['retired_full']=raw.map({1:0.,2:0.,3:0.,4:0.,5:1.,6:0.,7:0.})
        else:
            p['retired_any']=raw.map({1:0.,2:0.,3:1.,4:0.,5:0.});p['retired_full']=np.nan
            # W1 has no question separating retired/unemployed/disabled; no transition can start from W1 code6.
            if w==1:p['retired_any']=np.nan
        rows.append(p);statuses.append(pd.DataFrame({'person_id':d.person_id,'wave':w,'interview_status':clean(d[f'r{w}iwstat'])}))
        AUDIT['dictionaries'][f'{c}:{w}']={'source':str(source),'selected':{x:{'label':labels[x],'value_labels':m.variable_value_labels.get(x,{})} for x in [f'r{w}{s}' for s in suffixes] if x in labels},
          'marital_rule':'widowed from current-wave legal mstath where available; otherwise current mstat; code7=widowed. Current mstat includes partnership and remains separate.',
          'retirement_rule':('lbrf_e4 partly retired/5 retired->retired_any1; valid other states->0; full only5.' if c=='ELSA' else 'lbrf_m3 explicit retired->retired_any1;1/2/4/5->0;6 mixed unknown; entire W1 excluded as retirement risk-set anchor. No retired_full assumption.'),
          'skip':'All negative/tagged/skip/proxy/refusal/DK/missing codes remain unknown. Nonresponse . remains missing; interview_status exported separately.',
          'retirement_question':('Best description of current situation (WPDES).' if c=='ELSA' else 'Current work status plus reason for not working; questions differ significantly from HRS; employed respondents can miss reason-for-not-working retirement question.'),
          'evidence_pdf_pages':([98,99,104,105,106,724,725,726] if c=='ELSA' else [465,466,467])}
    put(c,rows,statuses)

def charls():
    root=SOURCE/'CHARLS';rows=[]
    for w,y in enumerate([2011,2013,2015,2018,2020],1):
        source=next(p for p in (root/str(y)).glob('*.dta') if 'demographic' in p.name.lower());col='ba011' if y==2020 else 'be001'
        d,m=pyreadstat.read_dta(source,usecols=['ID',col],apply_value_formats=False)
        ids=d.ID.astype(str).str.strip().str.replace(r'\.0$','',regex=True).map(lambda v:v[:9]+'0'+v[9:] if len(v)==11 else v)
        valid=ids.str.fullmatch(r'\d{12}',na=False); d=d.loc[valid];ids=ids.loc[valid]
        p=pd.DataFrame({'person_id':'CHARLS:'+ids,'wave':w})
        raw=clean(d[col],[1,2,3,4,5,6,7]);p['marital_status']=raw.map({1:1.,2:2.,3:4.,4:5.,5:7.,6:8.,7:3.})
        p['marital_status_legal']=p.marital_status;p['widowed']=raw.eq(5).astype(float).where(raw.notna())
        for k in ['retirement_status_raw','retired_any','retired_full','labor_status_raw']:p[k]=np.nan
        rows.append(p)
        AUDIT['dictionaries'][f'CHARLS:{w}']={'source':str(source),'year':y,'selected':{col:{'label':dict(zip(m.column_names,m.column_labels))[col],'value_labels':m.variable_value_labels.get(col,{})}},
          'marital_status_recode':{'1':1,'2':2,'3':4,'4':5,'5':7,'6':8,'7':3},'widowed_rule':'raw5 widowed; valid other known codes0; missing unknown',
          'retirement_rule':'Not extracted: legal retirement procedure and self-reported current retirement are not automatically equivalent.'}
    put('CHARLS',rows,[])

def jsonsafe(x):
    if isinstance(x,dict):return {str(k):jsonsafe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonsafe(v) for v in x]
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    return x

if __name__=='__main__':
    BASE.mkdir(parents=True,exist_ok=True)
    DEST.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    for c in ['ELSA','MHAS']:harmonized(c)
    charls();hrs()
    AUDIT['issues']=['Self-identified HRS retirement, ELSA best-status retirement, and MHAS retirement reason are not measurement-equivalent. Treat broad retired_any as sensitivity/exploratory unless a narrower common estimand is justified.',
        'ELSA/mHAS legal marital status can differ from current implied partnership status; widowed does not mean living alone.',
        'Death-year missing does not mean alive; full interview status is local only and retains unknown9.']
    AUDIT['runtime_seconds']=time.perf_counter()-start
    (BASE/'event_status_audit.json').write_text(json.dumps(jsonsafe(AUDIT),ensure_ascii=False,indent=2),encoding='utf-8')
    # The legacy common panel omitted documented CHARLS follow-up routing.
    # Keep this reconstruction in the reproducible extension entry point.
    from recover_charls_diagnoses import main as recover_charls_diagnoses
    recover_charls_diagnoses()
