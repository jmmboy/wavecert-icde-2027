"""Independent reaggregation of the completed eight-scenario direction test.

This auditor uses csv/statistics arithmetic, not the analysis summarizer.
SciPy supplies only Student t tail probabilities and quantiles. It performs
no simulation, fitting or modification of the frozen experiment outputs.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import bootstrap_env  # runtime dependency paths only, no analysis functions
from scipy.stats import t

EXP = Path(__file__).resolve().parents[1]
HERE = EXP / 'results' / 'simulation_direction8_2026-10-27'
SCENARIOS = ('null', 'chain', 'overlap', 'irregular', 'informative_dropout',
             'informative_visits', 'hidden_impairment_death', 'recovery_shift')
ARMS = ('fair_unordered_overlap_width', 'fair_plus_certain_direction')
checks = Counter()
maximum_scaled_error = defaultdict(float)

def read(name):
    with (HERE / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def close(actual, expected, context, tail=False):
    actual = float(actual)
    expected = float(expected)
    assert math.isfinite(actual) and math.isfinite(expected), context
    error = abs(actual - expected)
    tolerance = max(1e-300, abs(expected) * 5e-9) if tail else 1e-12 + abs(expected) * 2e-11
    assert error <= tolerance, (context, actual, expected, error, tolerance)
    checks['numeric_fields'] += 1
    maximum_scaled_error['tail_relative' if tail else 'ordinary_absolute'] = max(
        maximum_scaled_error['tail_relative' if tail else 'ordinary_absolute'],
        error / max(abs(expected), 1e-300) if tail else error)

def b(value):
    assert value in ('True', 'False'), value
    return value == 'True'

def contrast(rows):
    domains = defaultdict(list)
    for heldout, value in rows:
        domains[heldout].append(value)
    assert set(domains) == {0, 1, 2}
    means = [st.mean(domains[h]) for h in range(3)]
    variances = [st.variance(domains[h]) for h in range(3)]
    ns = [len(domains[h]) for h in range(3)]
    mu = st.mean(means)
    variance_terms = [variances[h] / (9 * ns[h]) for h in range(3)]
    variance = sum(variance_terms)
    assert variance > 0, 'No degenerate contrast is expected in this completed run'
    se = math.sqrt(variance)
    df = variance**2 / sum(variance_terms[h]**2 / (ns[h]-1) for h in range(3))
    critical = float(t.ppf(.975, df))
    return {'n_paired_replicates': sum(ns), 'mean_direction_improvement': mu,
            'stratified_mcse': se, 'satterthwaite_df': df,
            'mean_ci95_lower': mu-critical*se, 'mean_ci95_upper': mu+critical*se,
            'mc_mean_zero_p': float(2*t.sf(abs(mu/se), df)),
            'replicate_difference_sd': st.stdev([v for h, v in rows]),
            'within_domain_difference_sd': math.sqrt(st.mean(variances)),
            'between_domain_difference_sd': st.stdev(means),
            'heldout_domain_counts': '/'.join(map(str, ns))}

raw = read('raw.csv')
assert len(raw) == 3200
by_key = {}
for row in raw:
    key = (row['scenario'], int(row['rep']), row['method'])
    assert key not in by_key
    by_key[key] = row
    assert row['scenario'] in SCENARIOS and row['method'] in ARMS
    assert int(row['heldout_domain']) == int(row['rep']) % 3
    assert int(row['n_per_domain']) == 400
    assert row['status'] == 'ok' and b(row['converged'])
    assert 0 <= float(row['oracle_brier']) <= 2
assert set(by_key) == {(s, r, a) for s in SCENARIOS for r in range(200) for a in ARMS}
checks['raw_fits'] = len(raw)

paired = {}
sens = {}
pair_records = read('paired_replicates.csv')
assert len(pair_records) == 1600
pair_lookup = {(r['scenario'], int(r['rep'])): r for r in pair_records}
assert len(pair_lookup) == len(pair_records)
for scenario in SCENARIOS:
    differences = []
    converged_differences = []
    for rep in range(200):
        left, right = [by_key[(scenario, rep, arm)] for arm in ARMS]
        value = float(left['oracle_brier']) - float(right['oracle_brier'])
        heldout = rep % 3
        differences.append((heldout, value))
        if b(left['converged']) and b(right['converged']):
            converged_differences.append((heldout, value))
        record = pair_lookup[(scenario, rep)]
        assert int(record['heldout_domain']) == heldout
        close(record['improvement'], value, 'replicate delta')
    paired[scenario] = contrast(differences)
    sens[scenario] = contrast(converged_differences)
    assert paired[scenario]['heldout_domain_counts'] == '67/67/66'

# Holm step-down is calculated without the production helper.
previous = 0.
for rank, scenario in enumerate(sorted(SCENARIOS, key=lambda s: paired[s]['mc_mean_zero_p'])):
    adjusted = min(1., (8-rank) * paired[scenario]['mc_mean_zero_p'])
    previous = max(previous, adjusted)
    paired[scenario]['holm_p_8_scenarios'] = previous

for name, expected in [('paired_comparisons.csv', paired), ('converged_pair_sensitivity.csv', sens)]:
    rows = read(name)
    assert len(rows) == 8 and {r['scenario'] for r in rows} == set(SCENARIOS)
    for row in rows:
        for field, value in expected[row['scenario']].items():
            if field == 'heldout_domain_counts':
                assert row[field] == value
            else:
                close(row[field], value, name + '/' + row['scenario'] + '/' + field,
                      tail=field in ('mc_mean_zero_p', 'holm_p_8_scenarios'))

# Arm means use equal domain weights; displayed SD is across all repetitions.
summary = read('summary.csv')
assert len(summary) == 16
summary_lookup = {(r['scenario'], r['method']): r for r in summary}
for key, record in summary_lookup.items():
    rows = [r for r in raw if (r['scenario'], r['method']) == key]
    assert len(rows) == int(record['n']) == int(record['converged_fits']) == 200
    for metric in ['oracle_brier', 'auc_macro', 'oracle_log_loss', 'fit_seconds',
                   'cpu_seconds', 'rss_peak_sampled_mb']:
        values = [float(r[metric]) for r in rows]
        by_domain = [[float(r[metric]) for r in rows if int(r['heldout_domain']) == h]
                     for h in range(3)]
        close(record[metric+'__mean'], st.mean(st.mean(v) for v in by_domain), key + (metric, 'mean'))
        close(record[metric+'__replicate_sd'], st.stdev(values), key + (metric, 'SD'))
for s in SCENARIOS:
    close(float(summary_lookup[s, ARMS[0]]['oracle_brier__mean']) -
          float(summary_lookup[s, ARMS[1]]['oracle_brier__mean']),
          paired[s]['mean_direction_improvement'], s + '/arm difference matches paired mean')

# Verify all available reused fields against the immutable old JSON ledger.
legacy_path = EXP / 'results/simulation_prediction_fair/ledger.jsonl'
legacy = [json.loads(line) for line in legacy_path.read_text(encoding='utf-8').splitlines()]
assert len(legacy) == 400
for record in legacy:
    for old in record['rows']:
        current = by_key[old['scenario'], int(old['rep']), old['method']]
        assert current['record_origin'] == 'reused_hash_verified_two_scenario_run'
        for key, value in old.items():
            if isinstance(value, bool):
                assert b(current[key]) == value
            elif isinstance(value, (float, int)):
                close(current[key], value, 'legacy/' + key)
            else:
                assert str(current[key]) == str(value), key
        checks['reused_fields_rows'] += 1

design = read('design_audit.csv')
assert len(design) == 1600
design_index = {(r['scenario'], int(r['rep'])): r for r in design}
assert len(design_index) == 1600
assert set(design_index) == {(s, r) for s in SCENARIOS for r in range(200)}
execution_ledger = [json.loads(line) for line in (HERE / 'ledger.jsonl').read_text(encoding='utf-8').splitlines()]
ledger_index = {(r['scenario'], int(r['rep'])): r for r in execution_ledger}
assert set(ledger_index) == set(design_index) and len(execution_ledger) == 1600
for (s, rep), row in design_index.items():
    assert all(b(row[k]) for k in ['raw_shared_block_equal', 'processed_train_shared_block_equal',
                                  'processed_test_shared_block_equal'])
    assert int(row['added_signed_direction_columns']) == 6
    assert 6 <= int(row['selected_directed_candidates']) <= 12
    original_design = ledger_index[s, rep]['design_audit']
    assert len(original_design['selected_patterns']) == int(row['selected_directed_candidates'])
    assert len({tuple(sorted(p.split('>'))) for p in original_design['selected_patterns']}) == 6
    for field, value in row.items():
        if field in ('scenario', 'rep'):
            continue
        expected = original_design[field]
        if isinstance(expected, bool):
            assert b(value) == expected
        else:
            close(value, expected, 'design ledger/' + field)
    assert int(row['train_variable_direction_columns']) == 6
    assert int(row['train_people_with_nonzero_direction']) > 0
    assert int(row['test_people_with_nonzero_direction']) > 0
    for arm in ARMS:
        expected_columns = int(row['shared_raw_columns']) + (6 if arm == ARMS[1] else 0)
        assert int(by_key[s, rep, arm]['feature_count']) == expected_columns
        assert int(by_key[s, rep, arm]['direction_pair_count']) == 6

catalogue = read('catalogue_diagnostics.csv')
assert len(catalogue) == 8
for row in catalogue:
    rows = [r for r in design if r['scenario'] == row['scenario']]
    expected = {'selected_pairs_min': min(int(r['added_signed_direction_columns']) for r in rows),
                'selected_pairs_max': max(int(r['added_signed_direction_columns']) for r in rows),
                'empty_catalogues': sum(int(r['added_signed_direction_columns']) == 0 for r in rows),
                'inactive_test_repetitions': sum(int(r['test_people_with_nonzero_direction']) == 0 for r in rows),
                'train_people_with_direction_mean': st.mean(int(r['train_people_with_nonzero_direction']) for r in rows),
                'test_people_with_direction_mean': st.mean(int(r['test_people_with_nonzero_direction']) for r in rows)}
    for field, value in expected.items():
        close(row[field], value, 'catalogue/' + field)

frozen = json.loads((HERE / 'frozen.json').read_text(encoding='utf-8'))
assert frozen['old_ledger_sha256'] == sha(legacy_path)
for name, expected in frozen['code_sha256'].items():
    assert sha(EXP / 'src' / name) == expected, name
audit = json.loads((HERE / 'run_audit.json').read_text(encoding='utf-8'))
assert audit['independent_repetitions'] == 1600 and audit['fits_in_combined_analysis'] == 3200
assert audit['new_main_fit_executions'] == 2400 and audit['reused_fits'] == 800
assert audit['exceptions'] == audit['finite_nonconverged_fits'] == 0
assert audit['design_checks_all_pass'] and audit['null_chain_scores_match_old_ledger']
assert sum(r['main_fit_executions_new'] for r in execution_ledger) == 2400
assert sum(r['main_fit_executions_new'] == 0 for r in execution_ledger) == 400
close(audit['sum_new_fit_seconds'], sum(float(r['fit_seconds']) for r in raw
      if r['record_origin'] == 'new_eight_scenario_extension'), 'sum new fit seconds')
close(audit['max_sampled_worker_rss_mib'], max(float(r['rss_peak_sampled_mb']) for r in raw), 'max RSS')

result = {'passed': True, 'scope': 'Read-only independent csv/statistics reaggregation, no model or simulation executed.',
          'checks': dict(checks), 'maximum_errors': dict(maximum_scaled_error),
          'headline': {'analysis_fits': 3200, 'independent_repetitions': 1600,
                       'new_fits': 2400, 'reused_fits': 800, 'exceptions': 0, 'nonconverged': 0,
                       'negative_direction_increments': sum(paired[s]['mean_direction_improvement'] < 0 for s in SCENARIOS),
                       'holm_rejections_at_005': sum(paired[s]['holm_p_8_scenarios'] < .05 for s in SCENARIOS),
                       'empty_catalogues': 0, 'inactive_test_repetitions': 0,
                       'signed_coordinates_every_repetition': 6,
                       'selected_directed_candidates_range': [min(int(r['selected_directed_candidates']) for r in design),
                                                              max(int(r['selected_directed_candidates']) for r in design)],
                       'recovery_shift_repetitions_with_11_selected_directions': sum(r['scenario'] == 'recovery_shift' and
                                                                                   int(r['selected_directed_candidates']) == 11 for r in design)},
          'paired_recomputed': paired,
          'source_hashes': {name: sha(Path(__file__) if name == 'independent_statistics_audit.py' else HERE/name) for name in ['raw.csv', 'summary.csv', 'paired_comparisons.csv',
               'converged_pair_sensitivity.csv', 'paired_replicates.csv', 'design_audit.csv', 'catalogue_diagnostics.csv',
               'frozen.json', 'run_audit.json', 'ledger.jsonl', 'independent_statistics_audit.py']},
          'interpretation': 'The negative paired differences indicate higher Brier for added signed direction under this generator, selector, finite sample and fixed penalty; they are not a universal no-direction result. Pointwise Monte Carlo intervals and Holm tests address different reporting functions.'}
(HERE / 'independent_statistics_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'passed': True, 'headline': result['headline'], 'maximum_errors': result['maximum_errors']}, indent=2))
