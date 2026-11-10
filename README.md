# WaveCert experiment code

This package contains the WaveCert experiments for _Certifying Temporal Patterns in Sparse Survey Event Logs_. The simulations generate their own inputs. The survey preparation scripts build interview panels and visit-level event files from the study releases. Run the commands from the directory containing this README.

## Install and check

Use a Python environment with the packages in requirements.txt. The archived run recorded Python 3.12.14, NumPy 2.5.4, pandas 2.3.3, SciPy 1.18.1 and pyreadstat 1.3.6.

~~~text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py"
python src/simulation_runner.py --stage smoke --scenarios null chain --repetitions 2 --n 80 --workers 1
~~~

The short run creates its own data. Inspect simulation/smoke/manifest.json for the failure count and simulation/smoke/scenario_macro_summary_mcse.csv for the results.

## Run the simulated experiments

The structural experiment tests certain and possible matches under eight observation scenarios. The predictive experiment uses two matched representations and a rotating held-out domain. Run the two-scenario comparison before the eight-scenario extension. It supplies the reference ledger that the extension verifies and reuses.

~~~text
python src/simulation_runner.py --stage main --chain-length 2 --repetitions 200 --n 1000 --workers 3
python src/simulation_prediction_fair.py --reps 200 --n 400 --workers 3
python src/simulation_direction8.py --reps 200 --n 400 --workers 3
python tests/independent_statistics_audit.py
~~~

The extension writes results/simulation_direction8_2026-10-27/paired_comparisons.csv. The independent check uses the full 200-repetition design. Several runners resume from their local ledgers. Use a fresh extraction when changing sample sizes or repetition counts.

Core matching and the predictive state model are in src/interval_core.py and src/interval_model.py. Additional computational workloads have their own entry points.

~~~text
python tests/scale_interval_core.py
python src/evaluation_calibration_checks.py --mode sampling
~~~

## Run the survey experiments

The primary study uses the RAND HRS Longitudinal File 1992–2022 V1, Harmonized ELSA G.3 (2002–2019, June 2023 release), and the 2011, 2013, 2015, 2018 and 2020 CHARLS waves. The preparation code also screens MHAS. The study portals provide the source releases. [HRS](https://hrsdata.isr.umich.edu/data-products/rand-hrs-longitudinal-file-2022) · [ELSA](https://www.elsa-project.ac.uk/accessing-elsa-data) · [CHARLS](https://charls.charlsdata.com/pages/data/111/en.html)

### Build the interview panels and visit-level event files

Place the study downloads in the layout below, relative to the raw-data directory. Its default is `private/raw` under this package. Set `WAVECERT_RAW_DIR` to use another directory. Keep the original variable names, Stata value labels and archive members.

| Subdirectory | Input files |
| --- | --- |
| `HRS/` | `randhrs1992_2022v1_SAS.zip`, `h20core.zip` and `h22core.zip`, including their supplied codebooks. |
| `HRS/fat/<year>/` | One RAND HRS fat-file Stata ZIP per available vintage for 1992, 1993, 1995, 1996 and the even years 1998–2020. These supply dated education reports. The 2018 and 2020 files also supply the web-recall and ADL checks. |
| `ELSA/` | Harmonized ELSA G.3 Stata file as `h_elsa_g3.dta`. |
| `ELSA/ifs/` | Unpacked IFS files `wave_<w>_ifs_derived_variables.dta` for waves 1–7 and `wave_9_ifs_derived_variables_v2.dta`. |
| `CHARLS/<year>/` | Unpacked Stata modules for 2011, 2013, 2015, 2018 and 2020. Keep the release filenames for health status, demographics and the available sample, weight and exit modules. |
| `MHAS/` | Harmonized MHAS C.2 as `H_MHAS_c2.dta` and its End-of-Life B supplement as `H_MHAS_EOL_b.dta`. The original preparation screened this fourth cohort. |

From the package directory run the four commands in order. Use a fresh working copy when changing releases because the HRS extractor caches selected raw columns locally.

~~~text
python src/etl_hrs.py
python src/etl_elsa_mhas.py
python src/etl_charls.py
python src/extend_event_status.py
~~~

The first three commands write one person-by-wave file per study to `private/panels/<cohort>.csv.gz`. The last command reads those panels and the source releases. It restores the ELSA diagnosis at the original interview when the retrospective dispute flag is 6, constructs visit-level widowhood and applies the year-specific CHARLS questionnaire routing. It writes `private/event_status/<cohort>_event_status.csv.gz`. The join key is `person_id` and `wave`. Variable mappings and join counts are written to `audit/`.

If the corresponding panels and event files are already prepared, start with the primary analysis below. `WAVECERT_PANEL_DIR` overrides the panel directory for both preparation and analysis. Event-status files and the HRS extraction cache use the package's `private/` directory.

### Run the analyses

With these inputs in place, run the primary analysis, household bootstrap and source-support counts in this order.

~~~text
python src/real_prepare.py
python src/finalize_real_inputs.py
python src/real_benchmark.py --run
python src/real_statistics.py conditional
python src/run_component_sensitivity.py
python src/survey_summaries.py calibration
python src/real_bootstrap_fast.py --mode main --reps 200 --workers 3
python src/real_interpretation.py --mode support
~~~

The primary metrics are in `results/real/metrics.csv`. With the paper's releases and filters, Table I reports 40,090 eligible landmarks, 29,252 assessable follow-ups and 7,460 held-out observations across HRS, ELSA and CHARLS. The historical household check writes `audit/household_component_split_audit.csv`. The paper reports 417 ELSA components spanning discovery and confirmation, involving 839 participants. Section IV-B uses `results/interpretation/chain_support_source_and_confirmation.csv`. The calibration command generates the observed frequencies and household-bootstrap intervals for Figure 4(c) in `results/real/observed_window_calibration.csv`. If the bootstrap records fits that reach the iteration cap, run `python src/real_bootstrap_convergence.py` to repeat the affected draws at the higher cap as in Section IV-F.

For the expanded-history comparison in Section IV-C and the two perturbations in Section IV-E, run the following commands.

~~~text
python src/extended_history8.py
python src/real_supplement.py --run --phase all
python src/survey_summaries.py ablations
~~~

The expanded-history metrics are in `extended_history8/results/real/metrics.csv`. The combined supplement run writes both matched contrasts to `results/real_supplement/experiment_long.csv`. The last command writes the three cohort-specific Brier changes, their equal-cohort mean and their between-cohort sample SD to `results/real_supplement/observation_ablation_summary.csv` for Table VI.
