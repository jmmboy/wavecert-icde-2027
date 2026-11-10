"""Complete social masks and administrative follow-up flags without imputing health."""
import bootstrap_env
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def run(history_years=None):
    if history_years is None:
        amendment = ROOT/"exploratory_amendment.json"
        history_years = float(json.loads(amendment.read_text(encoding="utf-8"))["history_max_years"]) if amendment.exists() else 4.25
    e=pd.read_csv(ROOT/"private"/"episodes.csv.gz",dtype={"person_id":str,"household_id":str})
    logs=pd.read_csv(ROOT/"private"/"interval_logs.csv.gz",dtype={"person_id":str})
    all_complete={};max_dates={}
    for c in sorted(e.cohort.unique()):
        p=pd.read_csv(bootstrap_env.PANELS/(c+".csv.gz"),
                      usecols=["person_id","wave","date_year","married"],dtype={"person_id":str})
        side=pd.read_csv(ROOT/"private"/"event_status"/(c+"_event_status.csv.gz"),usecols=["person_id","wave","widowed"],dtype={"person_id":str})
        p=p.merge(side,on=["person_id","wave"],how="left",validate="one_to_one")
        p=p.merge(e.loc[e.cohort==c,["person_id","t0"]],on="person_id",how="inner",validate="many_to_one")
        p=p[(p.date_year>=p.t0-history_years)&(p.date_year<=p.t0)]
        p["marital_complete"]=p.widowed.notna()&p.married.notna()
        all_complete.update(p.groupby("person_id").marital_complete.all().to_dict())
        max_dates[c]=float(pd.read_csv(bootstrap_env.PANELS/(c+".csv.gz"),usecols=["date_year"]).date_year.max())
    e["widowhood_unknown"]=(~e.person_id.map(all_complete).fillna(False)).astype(int)
    event_ids=set(logs.loc[logs.event_type=="widowhood","person_id"])
    e["widowhood_event"]=np.where(e.person_id.isin(event_ids),1.,np.where(e.widowhood_unknown==0,0.,np.nan))
    e["administrative_window_available"]=(e.t0+2<=e.cohort.map(max_dates)+1e-8).astype(int)
    e.to_csv(ROOT/"private"/"episodes.csv.gz",index=False)
    (ROOT/"results"/"input_finalization.json").write_text(json.dumps({"status":"completed","rows":len(e),
              "cohort_observed_end":max_dates,"history_max_years":history_years,"social_mask_unknown":int(e.widowhood_unknown.sum()),
              "administrative_ineligible":int((e.administrative_window_available==0).sum()),
              "explanation":"Administration is separated from missed attendance for response-weight sensitivity. No unknown death/health is set to zero."},indent=2),encoding="utf-8")
    print("finalized",len(e),"episodes",flush=True)
if __name__=="__main__":run()
