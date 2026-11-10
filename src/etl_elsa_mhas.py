"""Local, metadata-validated ELSA/MHAS person-wave extraction. No model fitting.

Run with the bundled Python; all individual data stay under private/.
"""
from __future__ import annotations
import json
import re
import sys
import time

from bootstrap_env import PANELS, RAW_DATA as ROOT, ROOT as BASE
import numpy as np
import pandas as pd
import pyreadstat

SOURCES = {
    "ELSA": ROOT / "ELSA/h_elsa_g3.dta",
    "MHAS": ROOT / "MHAS/H_MHAS_c2.dta",
}
ADL = {"adl_dress": "dressa", "adl_bath": "batha", "adl_eat": "eata", "adl_bed": "beda", "adl_toilet": "toilta"}
COMMON = {"age": "agey", "married": "mstat", "smoking": "smoken",
          "hypertension": "hibpe", "diabetes": "diabe", "stroke": "stroke", "cancer": "cancre",
          "self_rated_health": "shlt", "proxy": "proxy"}


def clean(s, low=None, high=None):
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if low is not None:
        x = x.where(x.ge(low))
    if high is not None:
        x = x.where(x.le(high))
    return x


def ident(s, cohort, width=0):
    if pd.api.types.is_numeric_dtype(s):
        x = s.map(lambda z: str(int(z)) if pd.notna(z) and z >= 0 else None).astype("string")
    else:
        x = s.astype("string").str.strip().replace("", pd.NA)
    if width:
        x = x.str.zfill(width)
    return cohort + "_" + x


def ifs_file(filename):
    path = ROOT / "ELSA" / "ifs" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing ELSA wave-specific IFS file: {path}")
    return path


def inspect_ifs():
    for w in [1, 2, 3, 4, 5, 6, 7, 9]:
        suffix = f"wave_{w}_ifs_derived_variables{'_v2' if w == 9 else ''}.dta"
        p = ifs_file(suffix)
        _, m = pyreadstat.read_dta(p, metadataonly=True)
        print(json.dumps({"wave": w, "file_bytes": p.stat().st_size, "rows": m.number_rows,
                          "selected": {k: {"label": v, "codes": m.variable_value_labels.get(k)}
                                       for k, v in zip(m.column_names, m.column_labels)
                                       if re.search(r"^idauniq$|edqual|edend", k, re.I)}}, ensure_ascii=False), flush=True)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def validate_binary_meta(m, col):
    codes = m.variable_value_labels.get(col, {})
    assert 0 in codes and 1 in codes, (col, codes)
    assert any(word in codes[0].lower() for word in ["no", "not"]), (col, codes)
    assert any(word in codes[1].lower() for word in ["yes", "proxy"]), (col, codes)


def run(cohort):
    started = time.perf_counter()
    p = SOURCES[cohort]
    _, m = pyreadstat.read_dta(p, metadataonly=True)
    names = set(m.column_names)
    labels = dict(zip(m.column_names, m.column_labels))
    maxwave = 9 if cohort == "ELSA" else 5
    idcol = "idauniqc" if cohort == "ELSA" else "rahhidnp"
    static = [idcol, "ragender", "radyear", "raedyrs", "raeducl", "raedyrs_e"]
    suffixes = [*ADL.values(), *COMMON.values(), "iwstat", "cwtresp", "wtresp", "iwy", "iwm", "iwindy", "iwindm", "iwindf", "iwf",
                "hearte", "lunge", "hrtatte", "respe", "bmi", "mbmi", "iwnrsy", "iwnrsm", "cesd", "cesdm", "cesd_m", "cesdm_m",
                "imrc", "dlrc", "imrc8", "dlrc8", "fimrc8", "fdlrc8", "tr20", "tr16", "mobilb", "mobila", "mobilbm", "mobilam",
                "iadla", "iadlam", "arthre", "orient", "slfmem"]
    selected = [x for x in static if x in names]
    for w in range(1, maxwave + 1):
        selected += [f"r{w}{s}" for s in suffixes if f"r{w}{s}" in names]
        selected += [f"inw{w}"]
        selected += [x for x in [f"hh{w}hhidc", f"h{w}hhidc"] if x in names]
    selected = list(dict.fromkeys(selected))
    d, _ = pyreadstat.read_dta(p, usecols=selected, apply_value_formats=False, user_missing=False)
    raw_adl_cols = [f"r{w}{s}" for w in range(1, maxwave + 1) for s in ADL.values()]
    raw_adl, _ = pyreadstat.read_dta(p, usecols=raw_adl_cols, apply_value_formats=False, user_missing=True)
    assert not d[idcol].duplicated().any(), "Person ID is not unique in wide source"
    audit = {
        "cohort": cohort,
        "source_version": "Harmonized ELSA G.3 (2002-2019), June 2023" if cohort == "ELSA" else "Harmonized MHAS C.2 (2001-2018)",
        "source_paths": [str(p)], "source_bytes": p.stat().st_size, "source_wide_rows": len(d),
        "dictionary": {c: {"label": labels[c], "value_labels": m.variable_value_labels.get(c, {})} for c in selected},
        "id_rule": f"{idcol} preserved as character identifier, prefixed {cohort}_; household IDs are wave-specific and prefixed.",
        "date_rule": "actual individual interview year and month; date_year=year+(month-0.5)/12, missing month=year+0.5; no nominal wave years filled",
        "five_item_rules": "Individual dressing, bathing, eating, getting in/out of bed, toilet items: 0=no difficulty, 1=some difficulty. All tagged/negative/skip/proxy/unresolved codes are missing. No summary ADL score is substituted; no missing is recoded to 0.",
        "row_rule": "Retain every alive interview (inw=1), including interviews with all ADL missing. Nonresponse/death status counts retained in audit, death_year copied as an analysis status field, never a predictor.",
        "interview_proxy_death_weight": {}, "wave_status_counts": {}, "wave_summaries": [], "transformations": {}, "join_checks": {}, "unresolved_issues": [],
        "adl_raw_code_counts_interviewed": {},
        "codebook_validation": {
            "file": str(ROOT / ("ELSA/Harmonized ELSA Documentation, Version G.3 (2002-2019), June 2023.pdf" if cohort == "ELSA" else "MHAS/Harmonized_MHAS_C.2_2001_2018.pdf")),
            "pdf_pages": [155, 193] if cohort == "ELSA" else [80, 88, 128],
            "validated_rules": ("Individual Rw* A codes are 0 no problem/1 difficulty; wave-1 proxy ADL skips remain missing. ADLA contains walking, ADLFIVE contains toilet, so use individual common-five items instead of exchanging these summaries."
                                if cohort == "ELSA" else "Individual Rw* A codes include cannot-do as 1 and don't-do with help as 1. Official A-variables already assign 0 for a skip that explicitly follows no difficulty on H1-H13; this documented skip is accepted as 0. Don't-do without help stays .x missing, and dressing is .p missing in proxy interviews. Summary indices may sum incomplete items; individual items retained instead.")}
    }
    person_id = ident(d[idcol], cohort)
    female = clean(d["ragender"]).map({1: 0., 2: 1.})
    death = clean(d["radyear"], 1900, 2026)
    if cohort == "MHAS":
        eolp = ROOT / "MHAS/H_MHAS_EOL_b.dta"
        ed, em = pyreadstat.read_dta(eolp, usecols=["rahhidnp", "raxyear"], apply_value_formats=False)
        assert not ed.rahhidnp.duplicated().any()
        eol = ed.set_index("rahhidnp")["raxyear"]
        extra_death = clean(d[idcol].map(eol), 1900, 2026)
        conflict = death.notna() & extra_death.notna() & death.ne(extra_death)
        audit["join_checks"]["eol"] = {"source": str(eolp), "id": idcol, "eol_rows": len(ed), "eol_duplicate_ids": 0,
                                          "matched_persons": int(d[idcol].isin(ed.rahhidnp).sum()), "death_year_conflicts": int(conflict.sum()),
                                          "new_death_years": int((death.isna() & extra_death.notna()).sum()),
                                          "rule": "many-to-one exact character-ID join; main radyear preferred on conflict, fill only missing death-year status with raxyear"}
        audit["source_paths"].append(str(eolp))
        death = death.fillna(extra_death)
        audit["unresolved_issues"] += ["MHAS respiratory disease includes asthma and differs from common chronic lung disease; lung left missing and extra_respiratory_disease retained.",
                                       "MHAS waves 1-3 only measure heart attack, narrower than general heart disease; heart is structurally unavailable in waves 1-3, genuine hearte retained only in waves 4-5, and extra_heart_attack retained in all waves.",
                                       "MHAS wave 2-to-3 interval is about nine years; retain raw panel and exclude from routine next-wave 2/3-year primary landmarks downstream."]
    else:
        audit["unresolved_issues"] += ["ELSA age is top-coded at 90 in public data; retained as released.",
                                       "ELSA radyear is derived from available end-of-life interviews only; missing death year does not establish survival.",
                                       "ELSA harmonized G.3 covers waves 1-9. Raw wave 10 is present in archive but is not appended without a separate item/date/eligibility validation.",
                                       "ELSA raedyrs_e is an education-leaving-age category, not completed years; education_years is unavailable."]
    panels = []
    for w in range(1, maxwave + 1):
        prefix = f"r{w}"
        def val(s, low=None, high=None):
            c = prefix + s
            return clean(d[c], low, high) if c in d else pd.Series(np.nan, index=d.index, dtype=float)
        def binary(s):
            c = prefix + s
            if c in names:
                validate_binary_meta(m, c)
            return val(s).where(lambda x: x.isin([0, 1]))
        inw = clean(d[f"inw{w}"])
        use = inw.eq(1)
        audit["adl_raw_code_counts_interviewed"][w] = {prefix+s: {str(k): int(v) for k, v in raw_adl.loc[use, prefix+s].value_counts(dropna=False).items()} for s in ADL.values()}
        audit["wave_status_counts"][w] = {str(k): int(v) for k, v in d[prefix + "iwstat"].value_counts(dropna=False).items()}
        z = pd.DataFrame(index=d.index)
        z["cohort"], z["person_id"], z["wave"] = cohort, person_id, w
        hh = f"hh{w}hhidc" if cohort == "ELSA" else f"h{w}hhidc"
        z["household_id"] = ident(d[hh], cohort) if hh in d else pd.NA
        z["year"] = val("iwindy" if cohort == "ELSA" else "iwy", 1900, 2026)
        z["month"] = val("iwindm" if cohort == "ELSA" else "iwm", 1, 12)
        z["date_year"] = z.year + (z.month - 0.5) / 12
        z.loc[z.month.isna(), "date_year"] = z.loc[z.month.isna(), "year"] + 0.5
        z["date_quality"] = np.where(z.year.notna() & z.month.notna(), "month", np.where(z.year.notna(), "year", "missing"))
        z["age"], z["female"] = val("agey", 0, 120), female
        z["education_years"] = clean(d["raedyrs"], 0, 30) if cohort == "MHAS" else np.nan
        # ELSA has retrospective merged static education; handled below from wave-specific IFS files.
        z["extra_education_category"] = clean(d["raeducl"]).where(lambda x: x.isin([1, 2, 3])) if cohort == "MHAS" else np.nan
        z["married"] = val("mstat").map({1: 1., 2: 1., 3: 1., 4: 0., 5: 0., 7: 0., 8: 0.})
        for out in ["smoking", "hypertension", "diabetes", "stroke", "cancer", "proxy"]:
            z[out] = binary(COMMON[out])
        for out, s in ADL.items():
            z[out] = binary(s)
        z["heart"], z["lung"] = binary("hearte"), binary("lunge")
        z["self_rated_health"] = (5 - val("shlt", 1, 5)) / 4
        z["extra_arthritis"], z["extra_heart_attack"] = binary("arthre"), binary("hrtatte")
        z["extra_respiratory_disease"] = binary("respe")
        z["bmi"] = val("bmi", 10, 80)
        if cohort == "ELSA":
            # Nurse assessments may postdate the core predictor landmark, especially waves 8/9.
            nurse_date = val("iwnrsy", 1900, 2026) + (val("iwnrsm", 1, 12) - 0.5) / 12
            measured = val("mbmi", 10, 80)
            available_at_landmark = nurse_date.notna() & z.date_year.notna() & nurse_date.le(z.date_year)
            z["bmi"] = measured.where(available_at_landmark)
            audit["transformations"][f"wave_{w}_bmi"] = {"measured_nonmissing": int((measured.notna() & use).sum()),
                                                          "retained_prelandmark": int((z.bmi.notna() & use).sum()),
                                                          "later_or_unknown_nurse_date_excluded": int((measured.notna() & z.bmi.isna() & use).sum())}
        dep_suffix = "cesd" if cohort == "ELSA" else "cesd_m"
        dep_miss = "cesdm" if cohort == "ELSA" else "cesdm_m"
        z["depression_score"] = val(dep_suffix, 0, 8 if cohort == "ELSA" else 9).where(val(dep_miss).eq(0))
        imm_suffix, del_suffix = ("imrc", "dlrc") if cohort == "ELSA" else ("imrc8", "dlrc8")
        z["extra_memory_immediate"] = val(imm_suffix, 0, 10 if cohort == "ELSA" else 8)
        z["extra_memory_delayed"] = val(del_suffix, 0, 10 if cohort == "ELSA" else 8)
        if cohort == "MHAS":
            # Keep observed recall only: prevent harmonized donor imputation from introducing unverifiable timing.
            for out, flag in [("extra_memory_immediate", "fimrc8"), ("extra_memory_delayed", "fdlrc8")]:
                n_before = int((z[out].notna() & use).sum())
                z[out] = z[out].where(val(flag).eq(0))
                audit["transformations"][f"wave_{w}_{out}"] = {"before": n_before, "after_observed_only": int((z[out].notna() & use).sum()), "flag": prefix+flag}
        z["memory_score"] = z[["extra_memory_immediate", "extra_memory_delayed"]].sum(axis=1, min_count=2)
        mobility_suffix = "mobilb" if cohort == "ELSA" else "mobila"
        mobility_miss = "mobilbm" if cohort == "ELSA" else "mobilam"
        z["extra_mobility_score"] = val(mobility_suffix, 0, 4 if cohort == "ELSA" else 5).where(val(mobility_miss).eq(0))
        z["extra_memory_orientation"] = val("orient", 0, 4)
        z["survey_weight"] = val("cwtresp" if cohort == "ELSA" else "wtresp", 0)
        z["death_year"], z["interview_alive"] = death, 1.
        # Analysis descriptors, not predictors. Preserve proxy and item absence for attrition audits.
        z["extra_interview_status"] = val("iwstat")
        z = z.loc[use].copy()
        assert z.person_id.notna().all()
        panels.append(z)
    panel = pd.concat(panels, ignore_index=True)
    if cohort == "ELSA":
        panel, education_audit = add_elsa_education(panel)
        audit["join_checks"]["wave_ifs_education"] = education_audit
    panel = panel.sort_values(["person_id", "date_year", "wave"], kind="stable").reset_index(drop=True)
    last_alive_year = panel.groupby("person_id").year.transform("max")
    invalid_death = panel.death_year.notna() & panel.death_year.lt(last_alive_year)
    audit["transformations"]["death_year_logical_check"] = {"source_death_year_before_later_alive_interview_persons": int(panel.loc[invalid_death, "person_id"].nunique()),
                                                             "affected_panel_rows": int(invalid_death.sum()),
                                                             "rule": "If recorded death year precedes a later alive interview year, death year is set missing for that person throughout; no invented correction."}
    panel.loc[invalid_death, "death_year"] = np.nan
    assert panel[["person_id", "wave"]].duplicated().sum() == 0
    for col in [*ADL, "female", "married", "smoking", "hypertension", "diabetes", "stroke", "heart", "lung", "cancer", "proxy"]:
        assert panel[col].dropna().isin([0, 1]).all(), col
    for w, z in panel.groupby("wave"):
        ad = z[list(ADL)]
        summary = {"wave": int(w), "rows": len(z), "persons": z.person_id.nunique(), "year_min": z.year.min(), "year_max": z.year.max(),
                   "age_min": z.age.min(), "age_max": z.age.max(), "female_fraction": z.female.mean(), "dates_missing": int(z.date_year.isna().sum()),
                   "adl_all5_known": int(ad.notna().all(axis=1).sum()), "adl_any_known_positive": int(ad.eq(1).any(axis=1).sum()),
                   "adl_all5_zero": int(ad.eq(0).all(axis=1).sum()), "adl_all5_missing": int(ad.isna().all(axis=1).sum()),
                   "adl_item_missing_fraction": ad.isna().mean().to_dict(), "proxy_n": int(z.proxy.eq(1).sum())}
        audit["wave_summaries"].append(summary)
    audit["rows"], audit["persons"] = len(panel), panel.person_id.nunique()
    audit["duplicate_person_wave"] = int(panel[["person_id", "wave"]].duplicated().sum())
    audit["temporal_checks"] = {"death_year_before_alive_interview_rows": int((panel.death_year.notna() & panel.death_year.lt(panel.year)).sum()),
                                "date_year_decreasing_with_wave_rows": int(panel.sort_values(["person_id", "wave"]).groupby("person_id").date_year.diff().lt(0).sum()),
                                "month_valid_fraction_of_nonmissing": float(panel.month.dropna().between(1, 12).mean())}
    audit["data_loss"] = {"wide_persons_without_any_alive_interview": int(len(d)-panel.person_id.nunique()), "item_missing_rows_dropped": 0,
                          "nonresponse_grid_cells_not_written": int(len(d)*maxwave-len(panel))}
    audit["missing_fraction"] = panel.isna().mean().to_dict()
    audit["interview_proxy_death_weight"] = {"proxy_observed": int(panel.proxy.notna().sum()), "death_year_persons": int(panel.loc[panel.death_year.notna(), "person_id"].nunique()),
                                              "survey_weights_observed": int(panel.survey_weight.notna().sum()), "interview_alive_rule": "inw=1 and iwstat response indicator, not inferred from ADL"}
    audit["output_fields"] = {"married": "Marriage/cohabiting partnership status: 1=married, spouse absent, civil partner, or partnered (source mstat 1/2/3); never/separated/divorced/widowed=0 (source mstat 4/5/7/8). This accommodates ELSA wave8-9 merged marriage/partnership category.",
                               "memory_score": "complete observed immediate+delayed recall sum; ELSA 0-20, MHAS 0-16; local units retained",
                               "depression_score": "complete-item raw total only: ELSA CESD8, MHAS modified CESD9; local units retained",
                               "extra_mobility_score": "complete-item local total: ELSA4/MHAS5; not a common-scale feature",
                               "extra_education_category": "MHAS first-interview ISCED3 education; ELSA wave-specific IFS categories (local scale, see join audit)",
                               "extra_heart_attack": "ever reported heart attack 0/1, distinct from common heart variable",
                               "extra_respiratory_disease": "MHAS ever respiratory disease including asthma 0/1, distinct from common lung",
                               "extra_interview_status": "status descriptor only, excluded from predictors"}
    audit["runtime_seconds"] = time.perf_counter() - started
    PANELS.mkdir(parents=True, exist_ok=True)
    (BASE / "audit").mkdir(exist_ok=True)
    panel.to_csv(PANELS / f"{cohort}.csv.gz", index=False, encoding="utf-8", compression="gzip")
    (BASE / f"audit/{cohort}_data_audit.json").write_text(json.dumps(jsonable(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cohort": cohort, "rows": len(panel), "persons": panel.person_id.nunique(), "waves": maxwave,
                      "adl_all5_known": int(panel[list(ADL)].notna().all(axis=1).sum()), "runtime_s": round(audit["runtime_seconds"], 2)}, ensure_ascii=False), flush=True)


def add_elsa_education(panel):
    # Do not use the static harmonized RAEDUCL: its first nonmissing report can postdate a landmark.
    audit = {"rule": "wave-specific IFS edqual/edend labels explicitly use current and previous waves only; exact ID+wave join; forward carry permitted only from an earlier observed interview, never backward fill",
             "education_category_scale": "ELSA-specific edqual 1 degree/NVQ4-5;2 higher ed below degree;3 NVQ3/A-level;4 NVQ2/O-level;5 NVQ1/CSE;6 foreign/other;7 no qualification. Do not treat as completed years or cross-cohort ISCED scale.",
             "waves": []}
    frames = []
    for w in [1, 2, 3, 4, 5, 6, 7, 9]:
        p = ifs_file(f"wave_{w}_ifs_derived_variables{'_v2' if w == 9 else ''}.dta")
        a, mm = pyreadstat.read_dta(p, usecols=["idauniq", "edqual", "edend"], apply_value_formats=False)
        assert not a.idauniq.duplicated().any()
        a["person_id"] = ident(a.idauniq, "ELSA", width=6)
        a["wave"] = w
        a["extra_education_category"] = clean(a.edqual, 1, 7)
        a["extra_education_leaving_age_category"] = clean(a.edend, 2, 8)
        matched = panel.loc[panel.wave.eq(w), "person_id"].isin(a.person_id)
        audit["waves"].append({"wave": w, "file": str(p), "rows": len(a), "duplicate_ids": 0,
                               "panel_rows_matched": int(matched.sum()), "panel_rows_unmatched": int((~matched).sum()),
                               "variable_labels": mm.column_names_to_labels,
                               "value_labels": mm.variable_value_labels})
        frames.append(a[["person_id", "wave", "extra_education_category", "extra_education_leaving_age_category"]])
    e = pd.concat(frames, ignore_index=True)
    panel = panel.drop(columns="extra_education_category").merge(e, on=["person_id", "wave"], how="left", validate="one_to_one")
    panel = panel.sort_values(["person_id", "wave"], kind="stable")
    before = panel.extra_education_category.notna().sum()
    for c in ["extra_education_category", "extra_education_leaving_age_category"]:
        panel[c] = panel.groupby("person_id", sort=False)[c].ffill()
    audit["forward_carry_education_cells"] = int(panel.extra_education_category.notna().sum()-before)
    audit["wave_8_rule"] = "No wave-8 IFS derived file in supplied archive; only previous observed education carried forward when available. No future wave-9 borrowing."
    return panel, audit


if __name__ == "__main__":
    if "--inspect-ifs" in sys.argv:
        inspect_ifs()
    else:
        for cohort in ["ELSA", "MHAS"]:
            run(cohort)
