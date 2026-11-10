"""Matched direction-only extension across the eight frozen predictive DGPs.

The old two-scenario results and original composite benchmark remain immutable.
Every repetition audits both raw and source-preprocessed shared designs. Only
signed certified direction is appended. Exact states are used only for scoring.
"""
import bootstrap_env
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import psutil
import scipy
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import simulation_prediction as bench
import simulation_prediction_fair as fair
from sim_data import simulate_domain
from interval_model import ReversiblePanelCTMC, oracle_state_metrics, partial_state_brier_bounds

ARMS = fair.ARMS
SCENARIOS = bench.SCENARIOS
OUT = bench.ROOT / 'results/simulation_direction8_2026-10-27'
OLD = bench.ROOT / 'results/simulation_prediction_fair'
PENALTY = .03
MAX_ITER = 400
TOL = 1e-7

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _build(scenario, rep, n):
    seed = 31000000 + SCENARIOS.index(scenario) * 100000 + rep * 10
    # Use the original predictive DGP adapter. Passing recovery_shift directly
    # would incorrectly add the structural experiment's report-error override.
    settings = bench._scenario(scenario)
    domains = [bench._observed_domain(simulate_domain(n, seed+d, settings, d), seed+d+777) for d in range(3)]
    heldout = rep % 3
    train = [data for d, data in enumerate(domains) if d != heldout]
    keep = bench._train_chain_selector(train)
    designs = [fair._design(data, keep) for data in domains]
    panel = bench._join_panel(train)
    Xtrain = {arm: np.vstack([designs[d][0][arm] for d in range(3) if d != heldout]) for arm in ARMS}
    Xtest = {arm: designs[heldout][0][arm] for arm in ARMS}
    width = Xtrain[ARMS[0]].shape[1]
    q = designs[0][1]
    assert all(design[1] == q for design in designs)
    for design, _ in designs:
        np.testing.assert_array_equal(design[ARMS[0]], design[ARMS[1]][:, :width])
        assert design[ARMS[1]].shape[1] == width + q
    models = [ReversiblePanelCTMC(penalty=PENALTY, max_iter=MAX_ITER, tol=TOL) for _ in ARMS]
    trainZ = [model._features(Xtrain[arm], fit=True) for model, arm in zip(models, ARMS)]
    testZ = [model._features(Xtest[arm]) for model, arm in zip(models, ARMS)]
    np.testing.assert_array_equal(trainZ[0], trainZ[1][:, :width+1])
    np.testing.assert_array_equal(testZ[0], testZ[1][:, :width+1])
    trainD = Xtrain[ARMS[1]][:, width:]
    testD = Xtest[ARMS[1]][:, width:]
    audit = {'raw_shared_block_equal': True, 'processed_train_shared_block_equal': True,
             'processed_test_shared_block_equal': True, 'shared_raw_columns': width,
             'added_signed_direction_columns': q, 'selected_directed_candidates': int(keep.sum()),
             'selected_patterns': ['>'.join(p) for p, flag in zip(bench.PATTERNS, keep) if flag],
             'train_people_with_nonzero_direction': int(np.any(np.nan_to_num(trainD) != 0, axis=1).sum()),
             'test_people_with_nonzero_direction': int(np.any(np.nan_to_num(testD) != 0, axis=1).sum()),
             'train_variable_direction_columns': int(np.sum(np.ptp(trainZ[1][:, width+1:], axis=0) > 0)),
             'simulator_settings': settings,
             'design_sha256': {arm: hashlib.sha256(np.nan_to_num(Xtrain[arm], nan=8.7654321e99).tobytes()+
                                np.nan_to_num(Xtest[arm], nan=8.7654321e99).tobytes()).hexdigest() for arm in ARMS}}
    return seed, heldout, domains[heldout], panel, Xtrain, Xtest, audit

def _one(job):
    scenario, rep, n, legacy = job
    threadpool_limits(1)
    tick = time.perf_counter()
    seed, heldout, test, panel, Xtrain, Xtest, audit = _build(scenario, rep, n)
    if legacy is not None:
        record = copy.deepcopy(legacy)
        assert {r['method'] for r in record['rows']} == set(ARMS)
        for row in record['rows']:
            assert (row['seed'], row['heldout_domain'], row['n_per_domain']) == (seed, heldout, n)
            assert row['direction_pair_count'] == audit['added_signed_direction_columns']
            assert row['feature_count'] == Xtest[row['method']].shape[1]
            row['record_origin'] = 'reused_hash_verified_two_scenario_run'
        record.update({'design_audit': audit, 'main_fit_executions_new': 0,
                       'audit_reconstruction_seconds': time.perf_counter()-tick})
        return record
    sampler = bench._MemorySampler()
    rows = []
    try:
        for arm in ARMS:
            sampler.reset()
            cpu0 = sum(psutil.Process().cpu_times()[:2])
            row = {'scenario': scenario, 'rep': rep, 'seed': seed, 'heldout_domain': heldout,
                   'n_per_domain': n, 'method': arm, 'status': 'ok',
                   'direction_pair_count': audit['added_signed_direction_columns'],
                   'record_origin': 'new_eight_scenario_extension'}
            try:
                model = ReversiblePanelCTMC(penalty=PENALTY, max_iter=MAX_ITER, tol=TOL).fit(Xtrain[arm], panel)
                t0 = time.perf_counter()
                pred = model.predict_state(Xtest[arm], t=3., start_state=test['start'])
                inference_seconds = time.perf_counter()-t0
                metric = oracle_state_metrics(pred, test['truth'])
                bound = partial_state_brier_bounds(pred, test['mask'])
                auc = [roc_auc_score(test['truth'] == s, pred[:, s]) if len(np.unique(test['truth'] == s)) > 1 else np.nan for s in range(3)]
                row.update({'oracle_brier': metric['multiclass_brier'], 'oracle_log_loss': metric['log_loss'],
                            'auc_macro': float(np.nanmean(auc)), 'auc_impairment': float(auc[1]), 'auc_death': float(auc[2]),
                            'calibration_bias_impairment': float(pred[:, 1].mean()-(test['truth'] == 1).mean()),
                            'calibration_bias_death': float(pred[:, 2].mean()-(test['truth'] == 2).mean()),
                            'observed_panel_nll': model.observed_nll(Xtest[arm], test['panel']),
                            'brier_identified_lower': bound['brier_lower'], 'brier_identified_upper': bound['brier_upper'],
                            'identified_state_fraction': bound['exact_horizon_state_fraction'],
                            'fit_seconds': model.fit_seconds_, 'converged': model.converged_,
                            'iterations': model.n_iter_, 'gradient_max': model.gradient_max_,
                            'optimizer_message': model.optimizer_message_,
                            'rate_clipping_fraction': model.rate_clipping_fraction_, 'feature_count': Xtest[arm].shape[1],
                            'inference_ms_per_1000': inference_seconds*1e6/n})
            except Exception as exc:
                row.update({'status': 'failed', 'exception': repr(exc)})
            row.update({'rss_peak_sampled_mb': sampler.read_mb(), 'cpu_seconds': sum(psutil.Process().cpu_times()[:2])-cpu0})
            rows.append(row)
    finally:
        sampler.close()
    return {'scenario': scenario, 'rep': rep, 'seconds': time.perf_counter()-tick,
            'rows': rows, 'design_audit': audit, 'main_fit_executions_new': 2}

def paired_summary(delta):
    """Equal-weight fixed-domain mean; independent generated repetitions."""
    g = delta.groupby('heldout_domain').improvement.agg(['mean', 'std', 'count'])
    if len(g) != 3 or np.any(g['count'] < 2):
        raise ValueError('Each fixed held-out domain needs at least two independent repetitions')
    mu = float(g['mean'].mean())
    terms = g['std'].to_numpy()**2 / g['count'].to_numpy() / 9
    variance = float(terms.sum())
    if variance == 0:
        se, df, lower, upper, p = 0., float('inf'), mu, mu, float(mu == 0)
    else:
        se = float(np.sqrt(variance))
        df = float(variance**2 / np.sum(terms**2 / (g['count'].to_numpy()-1)))
        critical = student_t.ppf(.975, df)
        lower, upper = mu-critical*se, mu+critical*se
        p = float(2*student_t.sf(abs(mu/se), df))
    return {'n_paired_replicates': len(delta), 'mean_direction_improvement': mu,
            'stratified_mcse': se, 'satterthwaite_df': df, 'mean_ci95_lower': lower,
            'mean_ci95_upper': upper, 'mc_mean_zero_p': p,
            'replicate_difference_sd': float(delta.improvement.std(ddof=1)),
            'within_domain_difference_sd': float(np.sqrt(np.mean(g['std'].to_numpy()**2))),
            'between_domain_difference_sd': float(g['mean'].std(ddof=1)),
            'heldout_domain_counts': '/'.join(map(str, g['count'].tolist())),
            'interval_target': 'equal-weight mean paired difference across three fixed synthetic domains',
            'interval_adjustment': 'unadjusted 95%; eight scenario tests use separate Holm correction'}

def summarize(records, out):
    frame = pd.DataFrame([r for record in records for r in record['rows']])
    assert not frame.duplicated(['scenario', 'rep', 'method']).any()
    frame.to_csv(out/'raw.csv', index=False)
    good = frame[frame.status == 'ok'].copy()
    comparisons, sensitivity, pair_rows, summaries = [], [], [], []
    for scenario in SCENARIOS:
        sf = good[good.scenario == scenario]
        wide = sf.pivot(index=['rep', 'heldout_domain'], columns='method', values='oracle_brier')
        delta = (wide[ARMS[0]]-wide[ARMS[1]]).dropna().rename('improvement').reset_index()
        delta['scenario'] = scenario
        pair_rows.append(delta)
        comparisons.append({'scenario': scenario, **paired_summary(delta)})
        converged = sf.pivot(index=['rep', 'heldout_domain'], columns='method', values='converged').fillna(False)
        keep = converged[ARMS[0]] & converged[ARMS[1]]
        restricted = delta.set_index(['rep', 'heldout_domain']).loc[keep].reset_index()
        sensitivity.append({'scenario': scenario, **paired_summary(restricted)})
        for arm in ARMS:
            rows = sf[sf.method == arm]
            result = {'scenario': scenario, 'method': arm, 'n': len(rows)}
            for metric in ['oracle_brier', 'auc_macro', 'oracle_log_loss', 'fit_seconds', 'cpu_seconds', 'rss_peak_sampled_mb']:
                values = rows[metric]
                domain_means = rows.groupby('heldout_domain')[metric].mean()
                result[metric+'__mean'] = float(domain_means.mean())
                result[metric+'__replicate_sd'] = float(values.std(ddof=1))
            result['converged_fits'] = int(rows.converged.sum())
            summaries.append(result)
    comparison = pd.DataFrame(comparisons)
    comparison['holm_p_8_scenarios'] = bench._holm(comparison.mc_mean_zero_p)
    comparison.to_csv(out/'paired_comparisons.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(out/'converged_pair_sensitivity.csv', index=False)
    pd.concat(pair_rows, ignore_index=True).to_csv(out/'paired_replicates.csv', index=False)
    pd.DataFrame(summaries).to_csv(out/'summary.csv', index=False)
    design = pd.DataFrame([{'scenario': r['scenario'], 'rep': r['rep'],
                           **{k: v for k, v in r['design_audit'].items() if k not in ['selected_patterns', 'simulator_settings', 'design_sha256']}}
                          for r in records])
    design.to_csv(out/'design_audit.csv', index=False)
    diagnostic = design.groupby('scenario').agg(
        selected_pairs_min=('added_signed_direction_columns', 'min'),
        selected_pairs_max=('added_signed_direction_columns', 'max'),
        empty_catalogues=('added_signed_direction_columns', lambda x: int((x == 0).sum())),
        inactive_test_repetitions=('test_people_with_nonzero_direction', lambda x: int((x == 0).sum())),
        train_people_with_direction_mean=('train_people_with_nonzero_direction', 'mean'),
        test_people_with_direction_mean=('test_people_with_nonzero_direction', 'mean'))
    diagnostic.to_csv(out/'catalogue_diagnostics.csv')
    audit = {'independent_repetitions': len(records), 'fits_in_combined_analysis': len(frame),
             'new_main_fit_executions': sum(r['main_fit_executions_new'] for r in records),
             'reused_fits': int((frame.record_origin == 'reused_hash_verified_two_scenario_run').sum()),
             'exceptions': int((frame.status != 'ok').sum()), 'finite_nonconverged_fits': int((~good.converged).sum()),
             'design_checks_all_pass': bool(design[['raw_shared_block_equal', 'processed_train_shared_block_equal', 'processed_test_shared_block_equal']].all().all()),
             'timing_scope': 'Per-fit CPU/time/RSS retain their original or current execution origin; reused fits do not contribute to current extension wall time.',
             'max_sampled_worker_rss_mib': float(good.rss_peak_sampled_mb.max()),
             'sum_new_fit_seconds': float(good.loc[good.record_origin == 'new_eight_scenario_extension', 'fit_seconds'].sum()),
             'null_chain_scores_match_old_ledger': True}
    (out/'run_audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit), flush=True)

def legacy_records(n, reps):
    frozen = json.loads((OLD/'frozen.json').read_text(encoding='utf-8'))
    assert (n, reps) == (frozen['n_per_domain'], frozen['repetitions_per_scenario'])
    for name, expected in frozen['sha256'].items():
        assert sha(Path(__file__).parent/name) == expected, 'Legacy source hash mismatch: '+name
    records = [json.loads(line) for line in (OLD/'ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    indexed = {(r['scenario'], r['rep']): r for r in records}
    assert len(indexed) == len(records) == 2*reps
    assert set(indexed) == {(s, r) for s in ['null', 'chain'] for r in range(reps)}
    assert all(row['status'] == 'ok' for r in records for row in r['rows'])
    return indexed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--reps', type=int, default=200)
    parser.add_argument('--n', type=int, default=400)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    old = legacy_records(args.n, args.reps)
    frozen = {'scenarios': SCENARIOS, 'repetitions_per_scenario': args.reps, 'n_per_domain': args.n,
              'seed_rule': '31000000 + scenario_index*100000 + rep*10 + domain',
              'heldout_rule': 'rep modulo 3; counts67/67/66 for 200 repetitions',
              'source_selector': 'possible support >=0.02 in both training domains; same catalogue for both arms',
              'arms': ARMS, 'ridge_penalty': PENALTY, 'max_iter': MAX_ITER, 'tol': TOL,
              'difference': 'only signed certain direction, same raw and source-preprocessed shared block',
              'preprocessing': 'source median imputation and columnwise centering/scaling, common intercept; no automatic missingness indicators',
              'primary_family': 'eight two-sided fixed-domain mean-zero paired contrasts, Holm adjusted',
              'finite_nonconverged_policy': 'retain and report; converged-pair sensitivity, no selective exclusion',
              'recovery_shift': bench._scenario('recovery_shift'),
              'old_ledger_sha256': sha(OLD/'ledger.jsonl'),
              'code_sha256': {name: sha(Path(__file__).parent/name) for name in
                             ['simulation_direction8.py', 'simulation_prediction_fair.py', 'simulation_prediction.py', 'sim_data.py', 'interval_core.py', 'interval_model.py']},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                              'scipy': scipy.__version__, 'threads_per_worker': 1, 'workers': min(args.workers, 3)}}
    path = OUT/'frozen.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == json.loads(json.dumps(frozen)), 'Frozen extension protocol changed'
    else:
        path.write_text(json.dumps(frozen, indent=2)+'\n', encoding='utf-8')
    preflight = OUT/'preflight_replay.json'
    if not preflight.exists():
        checks = []
        for scenario in ['null', 'chain']:
            fresh = _one((scenario, 0, args.n, None))
            expected = {r['method']: r for r in old[(scenario, 0)]['rows']}
            for row in fresh['rows']:
                assert row['status'] == 'ok'
                reference = expected[row['method']]
                for metric in ['oracle_brier', 'oracle_log_loss', 'auc_macro', 'observed_panel_nll']:
                    np.testing.assert_allclose(row[metric], reference[metric], rtol=0, atol=1e-11)
                checks.append({'scenario': scenario, 'method': row['method'], 'oracle_brier': row['oracle_brier'],
                               'old_oracle_brier': reference['oracle_brier'], 'metric_match_atol': 1e-11})
        preflight.write_text(json.dumps({'passed': True, 'verification_fits': 4, 'replays': checks}, indent=2)+'\n', encoding='utf-8')
    if args.preflight_only:
        print(preflight.read_text(encoding='utf-8'), flush=True)
        return
    ledger = OUT/'ledger.jsonl'
    records = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
    done = {(r['scenario'], r['rep']) for r in records}
    assert len(done) == len(records)
    jobs = [(s, r, args.n, old.get((s, r))) for s in SCENARIOS for r in range(args.reps) if (s, r) not in done]
    tick = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, 3)) as pool, ledger.open('a', encoding='utf-8') as handle:
        futures = [pool.submit(_one, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            handle.write(json.dumps(record, allow_nan=True)+'\n')
            handle.flush()
            if i % 20 == 0 or i == len(jobs):
                print(json.dumps({'finished_new_records': i, 'pending_records_at_start': len(jobs),
                                  'combined_records': len(records), 'wall_seconds': time.perf_counter()-tick}), flush=True)
    assert len(records) == len(SCENARIOS)*args.reps
    summarize(records, OUT)
    (OUT/'wall_time.json').write_text(json.dumps({'this_invocation_wall_seconds': time.perf_counter()-tick,
        'new_fit_executions': sum(r['main_fit_executions_new'] for r in records), 'preflight_verification_fits': 4}, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__':
    main()
