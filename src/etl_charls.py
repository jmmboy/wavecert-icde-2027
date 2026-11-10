"""CHARLS common-item panel; conservative missingness and historical-only filling."""
from bootstrap_env import PANELS, RAW_DATA, ROOT as BASE
import pandas as pd,numpy as np,json,re
ROOT = RAW_DATA / "CHARLS"
AUDIT={'cohort':'CHARLS','sources':[],'dictionary':{},'issues':[],'joins':[]}
REQ=['cohort','person_id','household_id','wave','year','month','date_year','date_quality','age','female','education_years','married','bmi','smoking','hypertension','diabetes','stroke','heart','lung','cancer','self_rated_health','depression_score','memory_score','adl_dress','adl_bath','adl_eat','adl_bed','adl_toilet','proxy','survey_weight','death_year','interview_alive']
def numeric(s):return pd.to_numeric(s,errors='coerce')
def key(s):
 s=s.astype(str).str.strip().str.replace(r'\.0$','',regex=True)
 return s.map(lambda v:v[:9]+'0'+v[9:] if len(v)==11 else v)
def read(p,columns):
 with pd.io.stata.StataReader(p) as rr:
  labels=rr.variable_labels(); vl=rr.value_labels()
 cols=[x for x in dict.fromkeys(['ID']+columns) if x in labels]
 d=pd.read_stata(p,columns=cols,convert_categoricals=False)
 d['ID']=key(d['ID'])
 invalid=~d.ID.str.fullmatch(r'\d{12}',na=False)
 if invalid.any():
  AUDIT['issues'].append(str(p)+': excluded '+str(int(invalid.sum()))+' rows without valid individual ID')
  d=d.loc[~invalid].copy()
 assert not d.ID.duplicated().any(),str(p)
 AUDIT['sources'].append({'path':str(p),'rows':len(d),'columns':{x:labels[x] for x in cols},'value_labels':{k:{str(a):b for a,b in v.items()} for k,v in vl.items() if k in cols}})
 return d,labels
def series(d,c):return numeric(d[c]) if c in d else pd.Series(np.nan,index=d.index)
def first(d,cs):
 o=pd.Series(np.nan,index=d.index)
 for c in cs:o=o.fillna(series(d,c))
 return o
def yn(s):return s.map({1:1.,2:0.})
def find(year,part):return next((p for p in (ROOT/str(year)).glob('*.dta') if part.lower() in p.name.lower()),None)
panels=[]; deaths=[]
for wi,year in enumerate([2011,2013,2015,2018,2020],1):
 hp=find(year,'health_status');dp=find(year,'demographic')
 with pd.io.stata.StataReader(hp) as rr:hl=rr.variable_labels()
 with pd.io.stata.StataReader(dp) as rr:dl=rr.variable_labels()
 adln=['db001','db003','db005','db007','db009'] if year==2020 else ['db010','db011','db012','db013','db014']
 healthcols=adln+['proxy','proxy_5','da001','da002','da059','da061','da067','da068']
 for j in [1,3,4,5,7,8]:healthcols+= [f'da007_{j}_',f'da002_{j}_',f'da003_{j}_',f'zdisease_{j}_']
 # Keep validated one-question local predictors rather than silently equating scales.
 extras={}
 for dest,pattern in [('extra_depressed',r'^(Felt|Feel).*Depressed|^Depressed$'),('extra_lonely',r'^(Felt|Feel).*Lonely'),('extra_restless_sleep',r'Sleep.*Restless|Restless.*Sleep')]:
  matches=[k for k,v in hl.items() if re.search(pattern,v,re.I)]
  if matches:extras[dest]=matches[0];healthcols+=matches[:1]
 h,_=read(hp,healthcols)
 demo_cols=['householdID','rgender','xrgender','ba000_w2_3','ba002_1','ba002_2','ba004','ba004_w3_1','ba004_w3_2','ba003_1','ba003_2','zrbirthyear','ba001','bd001','zbd001','bd001_w2_4','ba010','be001','ba011']
 d,_=read(dp,demo_cols)
 m=h.merge(d,on='ID',how='left',validate='one_to_one',suffixes=('','_demo'))
 AUDIT['joins'].append({'year':year,'health_n':len(h),'demographic_n':len(d),'matched':int(h.ID.isin(d.ID).sum())})
 p=pd.DataFrame(index=m.index)
 p['cohort']='CHARLS';p['person_id']='CHARLS:'+m.ID;p['household_id']='CHARLS:'+m.ID.str[:-2];p['wave']=wi
 p['year']=float(year);p['month']=np.nan
 # Interview year/month are in sample/weights files, not presumed from wave label.
 for part in ['sample','weight']:
  sp=find(year,part)
  if sp is None:continue
  s,_=read(sp,['iyear','imonth','INDV_weight_ad2','INDV_weight','died'])
  s=s.set_index('ID')
  if 'iyear' in s:p['year']=numeric(m.ID.map(s.iyear)).where(lambda a:a.between(year-1,year+2)).fillna(p['year'])
  if 'imonth' in s:p['month']=p['month'].fillna(numeric(m.ID.map(s.imonth)).where(lambda a:a.between(1,12)))
  wc='INDV_weight_ad2' if 'INDV_weight_ad2' in s else 'INDV_weight'
  if wc in s:p['survey_weight']=numeric(m.ID.map(s[wc])).where(lambda a:a>0)
 p['date_quality']=np.where(p.month.notna(),'month','year')
 p['date_year']=p.year+(p.month.fillna(6.5)-.5)/12
 bycols=['ba003_1','zrbirthyear'] if year==2020 else ['ba002_1','ba004_w3_1']
 p['_birth_year']=first(m,bycols).where(lambda a:a.between(1880,2005))
 sexcols=['xrgender','rgender','ba000_w2_3']+(['ba001'] if year==2020 else [])
 p['female']=first(m,sexcols).map({1:0.,2:1.})
 p['_raw_age']=series(m,'ba004').where(lambda a:a.between(30,115)) if year in [2011,2013] else np.nan
 # Education categories are retained locally; no invented exact schooling years.
 edu=first(m,['ba010'] if year==2020 else ['bd001','zbd001','bd001_w2_4'])
 p['extra_education_category']=edu.where(edu.between(1,11))
 marital=first(m,['ba011'] if year==2020 else ['be001'])
 p['married']=marital.map({1:1.,2:1.,3:0.,4:0.,5:0.,6:0.,7:1.})
 for target,col in zip(['adl_dress','adl_bath','adl_eat','adl_bed','adl_toilet'],adln):
  p[target]=series(m,col).map({1:0.,2:1.,3:1.,4:1.})
  AUDIT['dictionary'][f'{year}:{target}']={'source':col,'label':hl.get(col),'codes':{'1':0,'2':1,'3':1,'4':1},'missing':'unresolved remains NaN'}
 if year==2011:
  p['extra_bed_or_walk_difficulty']=p.adl_bed;p['adl_bed']=np.nan
  AUDIT['issues'].append('2011 db013 combines getting out of bed and walking across a room; not used as strict bed-only ADL. Current/future five-item main landmarks are from 2013 onward.')
 for name,j in [('hypertension',1),('diabetes',3),('cancer',4),('lung',5),('heart',7),('stroke',8)]:
  if year==2020:
   # New diagnosis plus preloaded prior doctor diagnosis. Do not interpret missing as no disease.
   old=yn(series(m,f'zdisease_{j}_'));new=yn(series(m,f'da003_{j}_'))
   p[name]=new.where(old.ne(1),1.)
   p.loc[(old==0)&(new==0),name]=0.
  else:p[name]=yn(series(m,f'da007_{j}_'))
  AUDIT['dictionary'][f'{year}:{name}']={'source':[f'da007_{j}_'] if year<2020 else [f'zdisease_{j}_',f'da003_{j}_'],'rule':'1 yes/2 no; prior confirmed yes retained in 2020; unresolved stays missing'}
 if year<2020:
  ever=yn(series(m,'da059'));still=yn(series(m,'da061'))
  p['smoking']=still.where(ever.ne(0),0.)
 # Self-rated health order same poor->good, but alternate 2011 question scales unresolved.
 if year>=2013:p['self_rated_health']=series(m,'da001').where(lambda a:a.between(1,5)).map(lambda a:(5-a)/4 if pd.notna(a) else np.nan)
 if year==2011:p['proxy']=series(m,'proxy').where(lambda a:a.isin([0,1]))
 elif year==2015:p['proxy']=yn(series(m,'proxy'))
 elif year==2020:
  # proxy field has 1=Yes only: absence is unknown, not an inferred No.
  p['proxy']=series(m,'proxy_5').map({1:1.,2:0.})
 for dest,source in extras.items():p[dest]=series(m,source).where(lambda a:a.between(1,4))
 p['interview_alive']=1.;p['death_year']=np.nan
 ep=find(year,'exit')
 if ep:
  e,_=read(ep,['exb001_1','exb001_2'])
  dy=series(e,'exb001_1').where(lambda a:a.between(2000,year+2))
  deaths.append(pd.DataFrame({'person_id':'CHARLS:'+e.ID,'death_year':dy}))
 panels.append(p)
 print('CHARLS wave',year,'rows',len(p),'five observed',int(p[[c for c in p if c.startswith('adl_')]].notna().all(axis=1).sum()),flush=True)
panel=pd.concat(panels,ignore_index=True).sort_values(['person_id','date_year'])
# Only baseline/past recorded stable information is propagated forward. No bfill.
for c in ['_birth_year','female','extra_education_category']:
 panel[c]=panel.groupby('person_id',sort=False)[c].ffill()
panel['age']=(panel.date_year-panel._birth_year).where(lambda a:a.between(30,115)).fillna(panel._raw_age)
if deaths:
 death=pd.concat(deaths).groupby('person_id').death_year.min()
 panel['death_year']=panel.person_id.map(death)
for c in REQ:
 if c not in panel:panel[c]=np.nan
panel=panel[REQ+[c for c in panel if c.startswith('extra_')]]
assert not panel.duplicated(['person_id','wave']).any()
assert all(set(panel[c].dropna().unique()).issubset({0.,1.}) for c in panel if c.startswith('adl_'))
AUDIT.update({'n_rows':len(panel),'n_people':int(panel.person_id.nunique()),'n_households':int(panel.household_id.nunique()),'waves':panel.groupby('wave').size().to_dict(),'date_quality':panel.date_quality.value_counts().to_dict(),'missing_fraction':panel.isna().mean().to_dict(),'age_range':[float(panel.age.min()),float(panel.age.max())],'ID_rule':'2011 11-character IDs insert 0 after ninth character; other 12-character IDs preserved; household=ID minus final 2 chars','education':'Raw validated education category is local extra; exact education years unavailable, left NaN.','BMI':'Not available from selected health modules, left NaN.','death_rule':'Earliest death year from available 2013 and 2020 exit files; complete death follow-up not guaranteed.','date_rule':'Use local sample/weights iyear/imonth when available; otherwise nominal survey year midpoint and date_quality=year.','self_health_rule':'2013+ da001 1..5 normalized (5-code)/4, 2011 alternate scales left unresolved.','privacy':'No person records uploaded; panel remains local private directory.'})
PANELS.mkdir(parents=True,exist_ok=True)
(BASE/'audit').mkdir(parents=True,exist_ok=True)
panel.to_csv(PANELS/'CHARLS.csv.gz',index=False)
(BASE/'audit'/'CHARLS_data_audit.json').write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2,default=lambda a:int(a) if isinstance(a,np.integer) else float(a)),encoding='utf-8')
print('CHARLS complete',len(panel),panel.person_id.nunique(),flush=True)
