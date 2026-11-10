"""Build HRS person-wave panel from supplied RAND 1992-2022 v1 and raw HRS.

Only local files are read. No modelling or train/test assignment is performed.
RAND individual ADL items, rather than an ADL summary, are used. Education is
rebuilt from dated raw interviews to avoid RAND's future-informed backfilling.
"""
from __future__ import annotations

import io
import json
import re
import time
import zipfile
from pathlib import Path

from bootstrap_env import PANELS, RAW_DATA, ROOT as BASE
import numpy as np
import pandas as pd
import pyreadstat

SOURCE = RAW_DATA / "HRS"
CACHE = BASE / "private" / "HRS_cache"
OUT = PANELS / "HRS.csv.gz"
AUDIT = BASE / "audit" / "HRS_data_audit.json"
RAND_ARCHIVE = SOURCE / "randhrs1992_2022v1_SAS.zip"
RAND_NAME = "randhrs1992_2022v1.sas7bdat"
ADL = {"adl_dress": "DRESSA", "adl_bath": "BATHA", "adl_eat": "EATA",
       "adl_bed": "BEDA", "adl_toilet": "TOILTA"}
DISEASE = {"hypertension": "HIBP", "diabetes": "DIAB", "cancer": "CANCR",
           "lung": "LUNG", "heart": "HEART", "stroke": "STROK"}
ALERT_DUPLICATES = {"HRS_086903_022", "HRS_913258_011", "HRS_557889_010", "HRS_554793_010"}
SOURCES: list[dict] = []
CHECKS: list[dict] = []
RAW_DICTIONARY: dict = {}


def numeric(x):
    return pd.to_numeric(x, errors="coerce").astype(float)


def valid(x, low, high):
    x = numeric(x)
    return x.where(x.between(low, high))


def ids(d):
    h = d["HHID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    p = d["PN"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(3)
    if not h.str.fullmatch(r"\d{6}").all() or not p.str.fullmatch(r"\d{3}").all():
        raise ValueError("Invalid HRS household/person identifier")
    return "HRS_" + h + "_" + p


def raw_stata(zp: Path, desired: list[str], cache_name: str | None = None):
    """Read selected columns from one fat file, keeping memory below 4 GB."""
    with zipfile.ZipFile(zp) as z:
        member = next(n for n in z.namelist() if n.lower().endswith(".dta"))
        data = io.BytesIO(z.read(member))
    with pd.io.stata.StataReader(data, convert_categoricals=False) as r:
        lab = r.variable_labels()
    lookup = {k.upper(): k for k in lab}
    cols = [lookup[k] for k in desired if k in lookup]
    data.seek(0)
    d = pd.read_stata(data, columns=cols, convert_categoricals=False, convert_missing=False)
    d.columns = d.columns.str.upper()
    RAW_DICTIONARY.update({k.upper(): {"label": lab[k], "archive": str(zp), "member": member}
                           for k in cols})
    SOURCES.append({"path": str(zp), "member": member, "selected_columns": list(d.columns)})
    if cache_name:
        d.to_pickle(CACHE / cache_name)
    return d


def core2022(section: str, desired: list[str]):
    zp = SOURCE / "h22core.zip"
    with zipfile.ZipFile(zp) as outer:
        zn = next(n for n in outer.namelist() if n.lower().endswith("sta.zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(zn))) as z:
            member = next(n for n in z.namelist() if n.lower() == f"h22{section}_r.dta")
            data = io.BytesIO(z.read(member))
    with pd.io.stata.StataReader(data, convert_categoricals=False) as r:
        lab = r.variable_labels()
    lookup = {k.upper(): k for k in lab}
    selected = [lookup[c] for c in desired if c in lookup]
    data.seek(0)
    d = pd.read_stata(data, columns=selected, convert_categoricals=False)
    d.columns = d.columns.str.upper()
    RAW_DICTIONARY.update({k.upper(): {"label": lab[k], "archive": str(zp), "member": member}
                           for k in selected})
    SOURCES.append({"path": str(zp), "member": member, "selected_columns": list(d.columns)})
    return d


def extract_source_documents():
    CACHE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAND_ARCHIVE) as z:
        for n in [RAND_NAME, "randhrs1992_2022v1.pdf", "sasfmts.sas7bdat"]:
            p = CACHE / n
            if not p.exists() or p.stat().st_size != z.getinfo(n).file_size:
                print("Extracting", n, flush=True)
                z.extract(n, CACHE)
    # Keep exact original codebook text for the selected contemporary variables.
    with zipfile.ZipFile(SOURCE / "h22core.zip") as z:
        n = next(n for n in z.namelist() if n.lower().endswith("cb.zip"))
        with zipfile.ZipFile(io.BytesIO(z.read(n))) as zz:
            for section in ["B", "D", "G", "A", "C"]:
                member = next(n for n in zz.namelist() if n.lower() == f"h22{section.lower()}_r.txt")
                (CACHE / f"H22{section}_R_codebook.txt").write_bytes(zz.read(member))
    with zipfile.ZipFile(SOURCE / "h20core.zip") as z:
        n = next(n for n in z.namelist() if n.lower().endswith("cb.zip"))
        with zipfile.ZipFile(io.BytesIO(z.read(n))) as zz:
            member = next(n for n in zz.namelist() if n.lower() == "h2020cb.txt")
            text = zz.read(member).decode("utf-8", errors="replace")
            flag = re.search(r"^RADLIMP\s+", text, re.M)
            if flag is None:
                raise ValueError("Missing raw 2020 ADL imputation documentation")
            (CACHE / "H20_ADL_imputation_source.txt").write_text(text[flag.start():flag.start()+5000], encoding="utf-8")


def raw_education():
    """Dated observations only; fill past observations forward, never backward."""
    cache = CACHE / "raw_education_and_2020_adl.pkl"
    provenance_cache = CACHE / "raw_education_sources.json"
    if cache.exists() and provenance_cache.exists():
        saved = json.loads(provenance_cache.read_text(encoding="utf-8"))
        RAW_DICTIONARY.update(saved["dictionary"])
        SOURCES.extend(saved["sources"])
        CHECKS.extend(saved["checks"])
        return pd.read_pickle(cache)
    rows = []
    starts = len(SOURCES), len(CHECKS)
    for year, variable in [(1992, "V207"), (1993, "B125"), (1995, "D649"),
                           (1996, "E649")] + [(y, {1998: "F987", 2000: "G1074"}.get(y, f"{'HJKLMNOPQR'[(y-2002)//2]}B014"))
                                                      for y in range(1998, 2022, 2)]:
        matches = list((SOURCE / "fat" / str(year)).glob("*.zip"))
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError(f"Expected one RAND fat-file archive for {year}, found {len(matches)}.")
        desired = ["HHID", "PN", variable]
        if year == 2020:
            desired += ["RADLIMP", "RG014", "RG021", "RG023", "RG025", "RG030",
                        "RG013", "RG012", "RG015", "RG022", "RG024", "RG029", "RG031"]
        print("Raw education/ADL", year, flush=True)
        d = raw_stata(matches[0], desired)
        d["person_id"] = ids(d)
        dup = int(d["person_id"].duplicated().sum())
        CHECKS.append({"kind": "raw_education", "year": year, "rows": len(d), "duplicate_person_ids": dup,
                       "variable_present": variable in d})
        if dup:
            raise ValueError(f"duplicate IDs in raw {year}")
        if variable in d:
            e = pd.DataFrame({"person_id": d["person_id"], "education_years": valid(d[variable], 0, 17),
                              "education_observed_year": year})
            rows.append(e)
        if year == 2020:
            d.to_pickle(CACHE / "raw2020_adl.pkl")
    d = core2022("b", ["HHID", "PN", "SB014"])
    rows.append(pd.DataFrame({"person_id": ids(d), "education_years": valid(d["SB014"], 0, 17),
                              "education_observed_year": 2022}))
    out = pd.concat(rows, ignore_index=True)
    out.to_pickle(cache)
    provenance_cache.write_text(json.dumps({"dictionary": RAW_DICTIONARY, "sources": SOURCES[starts[0]:],
                                            "checks": CHECKS[starts[1]:]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def run():
    t0 = time.perf_counter()
    extract_source_documents()
    _, meta = pyreadstat.read_sas7bdat(str(CACHE / RAND_NAME), metadataonly=True)
    labs = meta.column_names_to_labels
    # Include early waves solely to initialize past-only health-history states.
    suffixes = ["IWSTAT", "IWENDY", "IWENDM", "AGEY_E", "MSTAT", "BMI", "SMOKEN", "SHLT",
                "CESD", "CESDM", "PROXY", "WTRESP", "WTCRNH", "WTR_NH", "IMRC", "DLRC",
                "FIMRC", "FDLRC", "IMRCP", "DLRCP", "FIMRCP", "FDLRCP", "IMRCW", "DLRCW",
                "FIMRCW", "FDLRCW", *ADL.values(), *DISEASE.values()]
    wanted = ["HHID", "PN", "HHIDPN", "RAGENDER", "RADYEAR", "RAEDYRS"]
    wanted += [f"R{w}{s}" for w in range(1, 17) for s in suffixes]
    cols = list(dict.fromkeys(c for c in wanted if c in labs))
    selected_cache = CACHE / "rand_selected.pkl"
    if selected_cache.exists():
        wide = pd.read_pickle(selected_cache)
    else:
        print("Reading RAND selected", len(cols), "columns", flush=True)
        wide, _ = pyreadstat.read_sas7bdat(str(CACHE / RAND_NAME), usecols=cols, user_missing=False,
                                         disable_datetime_conversion=True)
        wide.to_pickle(selected_cache)
    wide["person_id"] = ids(wide)
    if wide["person_id"].duplicated().any():
        raise ValueError("RAND person ID is not one-to-one")
    rand_id_check = (numeric(wide["HHIDPN"]) == numeric(wide["HHID"].astype(str).str.strip()) * 1000
                     + numeric(wide["PN"].astype(str).str.strip()))
    if not rand_id_check.all():
        raise ValueError("HHIDPN inconsistent with HHID/PN")
    SOURCES.append({"path": str(RAND_ARCHIVE), "member": RAND_NAME, "version": "RAND HRS Longitudinal File 1992-2022 V1",
                    "rows": len(wide), "columns_selected": len(cols), "uncompressed_source_bytes": (CACHE / RAND_NAME).stat().st_size})
    education = raw_education()
    # 1992 ID correction is already present in the supplied RAND fat release.
    old_education_id = education["person_id"].eq("HRS_042857_040") & education["education_observed_year"].eq(1992)
    CHECKS.append({"kind": "official1992_ID_correction", "old_ids_corrected": int(old_education_id.sum())})
    education.loc[old_education_id, "person_id"] = "HRS_204070_010"

    def col(name):
        return wide[name] if name in wide else pd.Series(np.nan, index=wide.index)

    # Future disputes can revise the RAND *E disease variables backward. We use
    # current raw-recode variables plus past history only instead.
    state = {out: pd.Series(np.nan, index=wide.index) for out in DISEASE}
    waves = []
    wave_summaries = []
    raw2020 = pd.read_pickle(CACHE / "raw2020_adl.pkl").set_index("person_id")
    raw2018 = raw_stata(next((SOURCE / "fat" / "2018").glob("*.zip")),
                        ["HHID", "PN", "QD182WM1", "QD183WM1"])
    raw2018["person_id"] = ids(raw2018)
    missing_web_im = set(raw2018.loc[numeric(raw2018["QD182WM1"]).eq(96), "person_id"])
    missing_web_de = set(raw2018.loc[numeric(raw2018["QD183WM1"]).eq(96), "person_id"])
    cognitive2022 = core2022("d", ["HHID", "PN", "SD174", "SD184", "SD174W", "SD184W"])
    cognitive2022["person_id"] = ids(cognitive2022)
    cognitive2022 = cognitive2022.set_index("person_id")
    for w in range(1, 17):
        p = f"R{w}"
        disease_vals = {}
        for name, suffix in DISEASE.items():
            raw = numeric(col(p + suffix))
            value = pd.Series(np.nan, index=wide.index)
            # 6=had before but not now, and remains an ever-diagnosis.
            value.loc[raw.isin([1, 3, 6])] = 1.0
            value.loc[raw.eq(0)] = state[name].where(state[name].eq(1), 0.0)
            value.loc[raw.eq(4)] = 0.0  # disputes that prior diagnosis ever existed
            if name == "stroke":
                value.loc[raw.eq(2)] = np.nan  # possible stroke/TIA is not a confirmed stroke
            # No current unknown response is turned into a zero.
            state[name] = value.combine_first(state[name])
            disease_vals[name] = value
        if w < 4:
            continue
        responded = numeric(col(p + "IWSTAT")).eq(1)
        year = valid(col(p + "IWENDY"), 1997, 2025)
        month = valid(col(p + "IWENDM"), 1, 12)
        out = pd.DataFrame(index=wide.index)
        out["cohort"] = "HRS"
        out["person_id"] = wide["person_id"]
        out["household_id"] = "HRS_" + wide["HHID"].astype(str).str.strip().str.zfill(6)
        out["wave"] = w
        out["year"] = year
        out["month"] = month
        out["date_year"] = year + (month - 0.5) / 12
        out.loc[month.isna(), "date_year"] = year + 0.5
        out["date_quality"] = np.where(month.notna(), "month", "year")
        out["age"] = valid(col(p + "AGEY_E"), 0, 120)
        out["female"] = numeric(col("RAGENDER")).map({1: 0.0, 2: 1.0})
        # Merge most recent valid raw education recorded by this nominal wave.
        e = education.loc[education["education_observed_year"].le(1990 + 2 * w)].dropna(subset=["education_years"])
        e = e.sort_values("education_observed_year").drop_duplicates("person_id", keep="last").set_index("person_id")
        out["education_years"] = out["person_id"].map(e["education_years"])
        out["extra_education_observed_year"] = out["person_id"].map(e["education_observed_year"])
        marriage = numeric(col(p + "MSTAT"))
        out["married"] = marriage.map({1: 1., 2: 1., 3: 1., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0.})
        out["bmi"] = valid(col(p + "BMI"), 10, 80)
        out["smoking"] = valid(col(p + "SMOKEN"), 0, 1)
        for k, v in disease_vals.items():
            out[k] = v
        out["self_rated_health"] = (5 - valid(col(p + "SHLT"), 1, 5)) / 4
        out["depression_score"] = valid(col(p + "CESD"), 0, 8).where(numeric(col(p + "CESDM")).eq(0))
        if w <= 13:
            immediate = valid(col(p + "IMRC"), 0, 10).where(numeric(col(p + "FIMRC")).eq(0))
            delayed = valid(col(p + "DLRC"), 0, 10).where(numeric(col(p + "FDLRC")).eq(0))
        elif w <= 15:
            immediate = valid(col(p + "IMRCP"), 0, 10).where(numeric(col(p + "FIMRCP")).eq(0))
            delayed = valid(col(p + "DLRCP"), 0, 10).where(numeric(col(p + "FDLRCP")).eq(0))
            immediate = immediate.combine_first(valid(col(p + "IMRCW"), 0, 10).where(numeric(col(p + "FIMRCW")).eq(0)))
            delayed = delayed.combine_first(valid(col(p + "DLRCW"), 0, 10).where(numeric(col(p + "FDLRCW")).eq(0)))
            if w == 14:
                invalid_im = out["person_id"].isin(missing_web_im) & numeric(col(p+"IMRCP")).isna()
                invalid_de = out["person_id"].isin(missing_web_de) & numeric(col(p+"DLRCP")).isna()
                CHECKS.append({"kind": "official2018_web_recall_correction",
                               "immediate_code96_records": len(missing_web_im),
                               "delayed_code96_records": len(missing_web_de),
                               "immediate_additional_values_masked": int((invalid_im & immediate.notna()).sum()),
                               "delayed_additional_values_masked": int((invalid_de & delayed.notna()).sum())})
                immediate.loc[invalid_im] = np.nan
                delayed.loc[invalid_de] = np.nan
        else:
            im = valid(cognitive2022["SD174"], 0, 10)
            de = valid(cognitive2022["SD184"], 0, 10)
            if "SD174W" in cognitive2022:
                im = im.combine_first(valid(cognitive2022["SD174W"], 0, 10))
            if "SD184W" in cognitive2022:
                de = de.combine_first(valid(cognitive2022["SD184W"], 0, 10))
            immediate = out["person_id"].map(im)
            delayed = out["person_id"].map(de)
        out["extra_memory_immediate"] = immediate
        out["extra_memory_delayed"] = delayed
        out["memory_score"] = immediate + delayed
        for output, suffix in ADL.items():
            out[output] = valid(col(p + suffix), 0, 1)
        out["extra_adl_imputed"] = 0.0
        if w == 15:
            # Raw codebook: 3=RG021,RG023,RG025,RG030 imputed (N=1308).
            # Dressing RG014 was observed and is retained.
            imputed = out["person_id"].map(numeric(raw2020["RADLIMP"])).eq(3)
            out.loc[imputed, ["adl_bath", "adl_eat", "adl_bed", "adl_toilet"]] = np.nan
            out.loc[imputed, "extra_adl_imputed"] = 1.0
        out["proxy"] = valid(col(p + "PROXY"), 0, 1)
        weight = valid(col(p + "WTCRNH"), 0, 1e9)
        out["survey_weight"] = weight.combine_first(valid(col(p + "WTRESP"), 0, 1e9))
        out["death_year"] = valid(col("RADYEAR"), 1992, 2030)
        out["interview_alive"] = 1.0
        raw_stroke = numeric(col(p + "STROK"))
        out["extra_possible_stroke_tia"] = raw_stroke.map(lambda v: float(v == 2) if pd.notna(v) else np.nan)
        out = out.loc[responded].copy()
        duplicate_alert_n = int(out["person_id"].isin(ALERT_DUPLICATES).sum())
        out = out.loc[~out["person_id"].isin(ALERT_DUPLICATES)].copy()
        if out["date_year"].isna().any():
            raise ValueError(f"respondent without actual interview date in wave {w}")
        if out["person_id"].duplicated().any():
            raise ValueError(f"duplicate person-wave in {w}")
        full = out[list(ADL)].notna().all(axis=1)
        summary = {"wave": w, "nominal_year": 1990 + 2*w, "rows": len(out), "people": out["person_id"].nunique(),
                   "actual_year_min": float(out["year"].min()), "actual_year_max": float(out["year"].max()),
                   "month_missing_n": int(out["month"].isna().sum()), "five_adl_complete": int(full.sum()),
                   "any_five_adl_positive": int(out[list(ADL)].eq(1).any(axis=1).sum()),
                   "adl_imputed_removed": int(out["extra_adl_imputed"].sum()),
                   "education_available": int(out["education_years"].notna().sum()),
                   "official_duplicate_alert_rows_removed": duplicate_alert_n,
                   "proxy_n": int(out["proxy"].eq(1).sum()), "positive_survey_weight_n": int(out["survey_weight"].gt(0).sum())}
        wave_summaries.append(summary)
        waves.append(out)
        print("wave", w, summary, flush=True)
    panel = pd.concat(waves, ignore_index=True).sort_values(["person_id", "date_year"])
    if panel.duplicated(["person_id", "wave"]).any():
        raise ValueError("duplicate person-wave in final panel")
    nonpositive_deltas = int(panel.sort_values(["person_id", "wave"]).groupby("person_id")["date_year"].diff().le(0).sum())
    if nonpositive_deltas:
        raise ValueError("Nonpositive actual visit-date difference across nominal waves")
    # Directly observed raw 2020 codes must agree with the individual RAND items.
    item_checks = []
    final2020 = panel.loc[panel["wave"].eq(15)].set_index("person_id")
    for output, source in {"adl_dress": "RG014", "adl_bath": "RG021", "adl_eat": "RG023",
                            "adl_bed": "RG025", "adl_toilet": "RG030"}.items():
        raw = numeric(raw2020[source]).reindex(final2020.index)
        expected = raw.map({1: 1., 6: 1., 5: 0.})
        eligible = expected.notna() & final2020["extra_adl_imputed"].eq(0)
        mismatches = int(expected.loc[eligible].ne(final2020.loc[eligible, output]).sum())
        item_checks.append({"wave": 15, "output": output, "raw_variable": source,
                            "direct_observed_comparisons": int(eligible.sum()), "mismatches": mismatches})
        if mismatches:
            raise ValueError(f"Direct raw/RAND ADL disagreement: {output} {mismatches}")
    CHECKS.extend(item_checks)
    CHECKS.append({"kind": "raw2022_cognitive_join", "raw_rows": len(cognitive2022),
                   "RAND_wave16_rows": int(panel["wave"].eq(16).sum()),
                   "RAND_wave16_persons_absent_from_raw_D": int((~panel.loc[panel["wave"].eq(16), "person_id"].isin(cognitive2022.index)).sum())})
    # Original sources are retained in audit; no person IDs are written there.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT, index=False, compression="gzip")
    selected_dictionary = {c: labs[c] for c in cols}
    audit = {
        "cohort": "HRS", "status": "panel_ready", "created_local_date": "2026-10-22",
        "source_versions_and_paths": SOURCES,
        "selected_variable_dictionary": selected_dictionary,
        "raw_variable_dictionary": RAW_DICTIONARY,
        "source_documentation": {"RAND_codebook": str(CACHE / "randhrs1992_2022v1.pdf"),
                                 "raw_codebooks": [str(CACHE / f"H22{s}_R_codebook.txt") for s in ["A", "B", "C", "D", "G"]] + [str(CACHE / "H20_ADL_imputation_source.txt")]},
        "public_data_alerts_verified_2026_10_22": {
            "index": "https://hrsdata.isr.umich.edu/data-products/data-alerts",
            "2020_duplicates": "https://hrsdata.isr.umich.edu/data-products/data-alerts/77988",
            "2022_duplicates": "https://hrsdata.isr.umich.edu/data-products/data-alerts/77987",
            "2018_web_cognition": "https://hrsdata.isr.umich.edu/data-products/data-alerts/50630",
            "1992_id_correction": "https://hrsdata.isr.umich.edu/data-products/data-alerts/49695",
            "2000_education_combined": "https://hrsdata.isr.umich.edu/data-products/data-alerts/39276",
            "action": "Delete records of four duplicate IDs identified by current 2020/2022 alerts (five observed person-waves); do not manually transfer values. Mask 2018 web-recall no-response code96; supplied RAND nonimputation flags already remove these. Correct 1992 raw education ID if needed (already corrected in source). Avoid 2000 combined education variable G1074A, using original G1074 and past raw reports instead."},
        "id_rules": "person_id=HRS_+6-character HHID+_+3-character PN; household_id=HRS_+HHID. Preserve leading zeros. Assert HHIDPN=1000*HHID+PN. No cross-cohort person linking.",
        "date_rules": "Actual RAND RwIWENDY/RwIWENDM, date_year=year+(month-0.5)/12. Month unavailable -> year+0.5 and date_quality=year. No nominal wave date used in elapsed-time calculation.",
        "five_item_rules": {
            "variables": ADL,
            "RAND_rule": "RAND individual RwDRESSA/RwBATHA/RwEATA/RwBEDA/RwTOILTA: 0 no difficulty, 1 any difficulty. Yes/cannot do->1; no->0; don't do/DK/refusal->special missing unless current help yes supplies evidence for difficulty. No summary adopted.",
            "structural_skip": "RAND codebook pp.927-930: from wave3 no difficulty on initial physical tasks -> ADLs=0; from wave4 exactly one prior task difficulty plus no dressing difficulty -> remaining ADLs=0. These zeros follow documented interview routing and are not blanket missing-value recoding.",
            "missing": "All SAS special missing values are NaN; never become 0. Wave15 RADLIMP=3 participants (1308; vendor imputation caused by routing error) have only bathing/eating/bed/toilet reset to NaN and extra_adl_imputed=1; observed dressing is retained. These rows remain for attrition.",
            "wave_choice": "Waves4-16 (1998-2022) share post-wave4 ADL routing. Earlier waves excluded for changed measurement and missing toilet question.",
            "ADL_source_pages": [916, 927, 928, 929, 930, 931, 932]},
        "feature_rules": {
            "education": "Rebuild from original dated raw V207/B125/D649/E649/F987/G1074/xB014; valid0..17 (17=17+). Use most recent report available at/before nominal wave only. Never use RAND RAEDYRS, whose first-nonmissing-all-waves/Tracker fallback can backfill from future.",
            "gender": "RAND RAGENDER 1 male->0 female,2 female->1; stable trait.",
            "married": "RAND RwMSTAT1 married/2 spouse absent/3 partnered ->1;4..8->0. Thus includes cohabiting partner as in harmonized cohort feature.",
            "diseases": "Current RAND raw-recode RwHIBP/DIAB/CANCR/LUNG/HEART/STROK, not retrospective-cleaned *E variables: codes1/3/6->ever diagnosis1;4 explicit disputes prior diagnosis->0;0 no-new-diagnosis retains past known positive, otherwise0. Unresolved/DK/RF/missing->NaN. Earlier waves initialize past-only positive history. Stroke code2 possible/TIA->NaN, flagged separately.",
            "bmi": "RAND self-reported RwBMI, preserve values10..80; outside becomesNaN.",
            "smoking": "RAND RwSMOKEN0/1 current smoking.",
            "self_rated_health": "RwSHLT1 excellent..5 poor mapped(5-x)/4=0 poor to1 excellent.",
            "depression_score": "Raw local CESD8 score0..8 only if RwCESDM=0; no incomplete partial sum treated as complete.",
            "memory_score": "Immediate+delayed raw, nonimputed recall each0..10. W4-13 RAND RwIMRC/DLRC only imputation flags0; W14-15 FTF/TEL P or web W only flags0; W16 raw SD174/SD184 (or W counterparts), because RAND cognitive imputations stop atW15. Proxy missing preserved.",
            "survey_weight": "RAND RwWTCRNH combined respondent/nursing-home weight; fall back to RwWTRESP if absent, retain 0; >0 required for weighted analysis.",
            "death_year": "RAND RADYEAR, 1992..2030; this is an outcome/attrition variable and must never enter predictor matrix.",
            "interview_alive": "Only actual core respondents (RwIWSTAT=1) are panel visits, all alive1. Death_year supports post-baseline attrition. Nonresponse/dead records are not fabricated as visits."},
        "additional_validated_predictors": {
            "extra_memory_immediate": "Observed immediate recall0..10, nonimputed.",
            "extra_memory_delayed": "Observed delayed recall0..10, nonimputed.",
            "extra_adl_imputed": "2020 vendor ADL imputation/routing-error flag; label-quality flag, exclude from model predictors.",
            "extra_education_observed_year": "Dated provenance of education report, not intended as clinical predictor.",
            "extra_possible_stroke_tia": "Current raw stroke code2; distinguish possible stroke/TIA from confirmed stroke."},
        "interview_proxy_death_weight_availability": {k: int(panel[k].notna().sum()) for k in ["proxy", "death_year", "survey_weight", "interview_alive"]},
        "row_person_wave_summary": {"rows": len(panel), "persons": panel["person_id"].nunique(), "waves": sorted(panel["wave"].unique().tolist()),
                                    "actual_year_range": [float(panel["year"].min()), float(panel["year"].max())], "by_wave": wave_summaries},
        "aggregate_missingness": {k: {"missing_n": int(panel[k].isna().sum()), "missing_fraction": float(panel[k].isna().mean())} for k in panel.columns},
        "join_checks": {"RAND_person_id_duplicates": int(wide["person_id"].duplicated().sum()), "HHIDPN_mismatches": int((~rand_id_check).sum()),
                        "final_person_wave_duplicates": int(panel.duplicated(["person_id", "wave"]).sum()),
                        "nonpositive_visit_date_differences": nonpositive_deltas, "raw_checks": CHECKS},
        "data_loss": {"excluded_pre1998_interviews": "Waves1-3; measurement/routing differences.", "rows_dropped_for_missing_ADL": 0,
                      "imputed2020_ADL_rows_retained_but_label_missing": int(panel["extra_adl_imputed"].sum()),
                      "official_duplicate_alert_rows_removed": sum(x["official_duplicate_alert_rows_removed"] for x in wave_summaries),
                      "age_filter_applied": False},
        "unresolved_issues": [
            "Single-machine four-client simulation is concept validation, not a deployed federation or jurisdictional privacy implementation.",
            "RAND any-difficulty ADLs count help-yes when the difficulty answer is missing; explicitly note this harmonization choice and test strict complete-case sensitivity if needed.",
            "2020 routing-error ADL imputations are excluded from observed labels; this increases wave15 missingness and can change attrition.",
            "HRS 2022 early-GenX refreshment cohort has no official survey weights in current release. Retain rows for unweighted models; exclude missing weights from weighted sensitivity.",
            "RAND death year and response status can incorporate retrospective mortality ascertainment; use only for outcome/attrition analyses.",
            "Supplied RAND source version is fixed; official data alerts checked 2026-10-22 and relevant duplicate/cognition/ID corrections applied."],
        "runtime_seconds": time.perf_counter() - t0,
        "panel_path": str(OUT),
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"panel": str(OUT), "rows": len(panel), "persons": panel["person_id"].nunique(),
                      "five_adl_complete": int(panel[list(ADL)].notna().all(axis=1).sum()),
                      "seconds": audit["runtime_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
