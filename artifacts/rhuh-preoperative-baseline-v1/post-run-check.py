"""Saved-record RHUH audit. No fitting, producer imports, or patient-image reads.

Tolerance was fixed at 1e-12 before inspecting results. NumPy is used only to
reproduce the declared PCG64 label permutations, not predictions or metrics.
"""
from __future__ import annotations
import collections
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/rhuh-preoperative-baseline-v1/run-01'
REPORT = Path(__file__).with_suffix('.json')
TOL = 1e-12
MODELS = ('FOLD_PREVALENCE', 'KPS', 'KPS_CE_VOLUME')
FEATURES = ['preoperative_kps', 'preoperative_ce_volume_cm3']
DELTAS = collections.defaultdict(float)
COUNTS = collections.Counter()
WARNINGS = collections.Counter()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_text())


def compare(actual, expected, name, category='metric'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), name + ': keys')
        for key in expected:
            compare(actual[key], expected[key], name + '.' + key, category)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), name + ': length')
        for i, value in enumerate(expected):
            compare(actual[i], value, f'{name}[{i}]', category)
    elif isinstance(expected, float):
        require(type(actual) in (float, int) and math.isfinite(actual), name + ': finite numeric')
        delta = abs(actual - expected)
        DELTAS[category] = max(DELTAS[category], delta)
        require(delta <= TOL, f'{name}: {actual!r} vs {expected!r}, delta={delta!r}')
    else:
        require(type(actual) is type(expected) and actual == expected, name + ': exact value/type')


def inventory(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def ratio(a, b):
    return {'value': a / b if b else None, 'null_reason': None if b else 'zero_denominator'}


def metrics(y, predictions):
    result = {}
    for name, values in predictions.items():
        c = {k: 0 for k in ('TN', 'FP', 'FN', 'TP')}
        for label, p in zip(y, values, strict=True):
            c[('TN', 'FP', 'FN', 'TP')[2 * label + int(p >= .5)]] += 1
        positives = [p for p, label in zip(values, y) if label]
        negatives = [p for p, label in zip(values, y) if not label]
        # Pairwise AUC uses the already validated saved probabilities, preserving
        # their exact tie pattern instead of introducing transcendental roundoff.
        auc = (math.fsum(float(a > b) + .5 * float(a == b)
                         for a in positives for b in negatives) / (len(positives) * len(negatives)))
        result[name] = {
            'n': len(y), 'brier': math.fsum((p-label)**2 for p, label in zip(values, y)) / len(y),
            'confusion': c, 'accuracy': (c['TN'] + c['TP']) / len(y),
            'balanced_accuracy': (c['TP']/len(positives) + c['TN']/len(negatives))/2,
            'sensitivity': ratio(c['TP'], len(positives)), 'specificity': ratio(c['TN'], len(negatives)),
            'precision': ratio(c['TP'], c['TP']+c['FP']),
            'negative_predictive_value': ratio(c['TN'], c['TN']+c['FN']),
            'roc_auc': None if name == MODELS[0] else auc,
            'auc_null_reason': 'LOO_prevalence_inverse_label_ordering_artifact' if name == MODELS[0] else None}
    return {'models': result, 'confidence_intervals': None, 'clinical_calibration_validated': False,
            'paired_brier': {a+'_minus_'+b: result[a]['brier']-result[b]['brier']
                             for a, b in ((MODELS[2], MODELS[1]), (MODELS[1], MODELS[0]), (MODELS[2], MODELS[0]))}}


def folds(rows, ids, X, y, job):
    require(len(rows) == len(ids), job + ': full denominator')
    predictions = {name: [] for name in MODELS}
    for holdout, row in enumerate(rows):
        prefix = f'{job}/{holdout}'
        require(row['job'] == job and row['holdout_index'] == holdout and row['status'] == 'completed', prefix)
        require(type(row['target']) is int and row['target'] == y[holdout], prefix + ': label')
        train = [i for i in range(len(ids)) if i != holdout]
        if not job.startswith('permutation:'):
            require(row['holdout_patient_id'] == ids[holdout], prefix + ': holdout ID')
            require(row['training_patient_ids'] == [ids[i] for i in train], prefix + ': training IDs')
        else:
            require('holdout_patient_id' not in row and 'training_patient_ids' not in row, prefix + ': compact scope')
        require(set(row['models']) == set(MODELS), prefix + ': comparators')
        for name, model in row['models'].items():
            require(model['shadow_prediction_identical'] is (job == 'observed'), prefix + ': shadow status')
            require(math.isfinite(model['elapsed_seconds']) and model['elapsed_seconds'] >= 0, prefix + ': timing')
            if name == MODELS[0]:
                positive = sum(y[i] for i in train)
                require(model['fit'] is False and model['training_count'] == len(train)
                        and model['training_positive_count'] == positive, prefix + ': prevalence fit')
                p = positive / len(train)
            else:
                width = 1 if name == MODELS[1] else 2
                require(model['features'] == FEATURES[:width] and model['converged'] is True, prefix + ': features/convergence')
                require(len(model['iterations']) == 1 and type(model['iterations'][0]) is int
                        and 0 <= model['iterations'][0] < 1000, prefix + ': iterations')
                require(len(model['coefficients']) == 1 and len(model['coefficients'][0]) == width
                        and len(model['intercept']) == 1, prefix + ': model shape')
                coefficients = model['coefficients'][0]
                require(all(math.isfinite(v) for v in coefficients + model['intercept']), prefix + ': finite weights')
                means = [math.fsum(X[i][j] for i in train)/len(train) for j in range(width)]
                variances = [math.fsum((X[i][j]-means[j])**2 for i in train)/len(train) for j in range(width)]
                scales = [math.sqrt(v) if v else 1. for v in variances]
                for key, expected in [('scaler_mean', means), ('scaler_variance', variances), ('scaler_scale', scales)]:
                    compare(model[key], expected, prefix+'/'+name+'/'+key, key)
                z = model['intercept'][0] + math.fsum(coefficients[j] *
                    ((X[holdout][j]-model['scaler_mean'][j])/model['scaler_scale'][j]) for j in range(width))
                p = 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))
                for warning in model['warnings']:
                    WARNINGS[warning['category']] += 1
                    require(warning['category'] == 'FutureWarning' and "'penalty' was deprecated" in warning['message'], prefix + ': unexpected warning')
                COUNTS['logistic_predictions_reconstructed'] += 1
            compare(model['probability'], p, prefix+'/'+name+'/probability', 'probability')
            require(0 <= model['probability'] <= 1, prefix + ': probability bounds')
            compare(model['predicted'], int(p >= .5), prefix+'/'+name+'/predicted')
            compare(model['squared_error'], (p-y[holdout])**2, prefix+'/'+name+'/squared_error', 'squared_error')
            predictions[name].append(model['probability'])
            COUNTS['all_predictions_checked'] += 1
        COUNTS['folds_checked'] += 1
    return metrics(y, predictions)


def ledger(name):
    return [json.loads(line) for line in (OUT/name).read_text().splitlines()]


def audit():
    before = inventory(OUT)
    baseline, result, state, cohort, runtime = [read(OUT/name) for name in
        ('baseline.json','result.json','state.json','cohort.json','runtime.json')]
    release_path = ROOT/'manifests/experiments/rhuh-preoperative-baseline-v1.release.json'
    require(sha(release_path) == 'a87dd3457e8aa33fd043e9df71cf58fe3aee3cbc26b22f2a9766159f404682df', 'released bytes')
    release = read(release_path)
    require(release['authorized'] is True and release['schema'] == 'rhuh-baseline-release-v1', 'release authority')
    require(release['source_commit'] == 'e5272a8fe76b257ce7b3c8c64f8f0cdd5b76c069', 'source commit')
    require(baseline['release_binding'] == {'path':str(release_path.relative_to(ROOT)), 'sha256':sha(release_path)}, 'baseline release')
    require(baseline['source_bindings'] == release['source_bindings'] and baseline['study_binding'] == release['study'], 'baseline sources')
    require(baseline['directory'] == str(OUT) and release['output_directory'] == str(OUT.relative_to(ROOT)), 'output authority')
    study = read(ROOT/release['study']['path'])
    require(study == baseline['study'], 'saved study equality')
    unhashed = {k:v for k,v in study.items() if k != 'declaration_content_hash'}
    require(study['declaration_content_hash'] == 'sha256:'+digest(canonical(unhashed)), 'declaration canonical hash')
    expected_inputs = {}
    bindings = [release['study'], baseline['release_binding'], *release['source_bindings'].values(),
                release['source_archive'], release['interpreter'], release['dependency_manifest'],
                {'path':study['source']['local_path'],'sha256':study['source']['sha256']}]
    for binding in bindings:
        p = (ROOT/binding['path']).resolve()
        require(sha(p) == binding['sha256'], 'bound bytes '+str(p))
        expected_inputs[str(p)] = binding['sha256']
    dep = read(ROOT/release['dependency_manifest']['path'])
    require(dep == baseline['dependency'], 'dependency receipt equality')
    inventories = {'isolated':{'directory':dep['directory'], 'files':dep['files']}, **dep['shared_packages']}
    for name, record in inventories.items():
        folder = Path(record['directory'])
        require(inventory(folder) == record['files'], 'runtime exact inventory '+name)
        for filename, h in record['files'].items():
            expected_inputs[str((folder/filename).resolve())] = h
    require(expected_inputs == baseline['inputs'] and len(expected_inputs) == 3660, 'complete exact input closure')
    wanted = {b['path']: b['sha256'] for b in release['source_bindings'].values()}
    wanted[release['study']['path']] = release['study']['sha256']
    with tarfile.open(ROOT/release['source_archive']['path']) as archive:
        require(archive.pax_headers.get('comment') == release['source_commit'], 'archive commit')
        seen = set()
        for member in archive:
            if member.isdir(): continue
            require(member.isfile() and member.name in wanted and member.name not in seen, 'archive closed inventory')
            content = archive.extractfile(member).read()
            require(digest(content) == wanted[member.name], 'archive bytes '+member.name)
            git = subprocess.run(['git','show',release['source_commit']+':'+member.name], cwd=ROOT, check=True, capture_output=True)
            require(git.stdout == content, 'committed source '+member.name)
            seen.add(member.name)
        require(seen == set(wanted) and len(seen) == 5, 'five archived committed files')
    require(runtime['packages'] == dep['packages'] and all(v == '1' for v in runtime['thread_environment'].values()), 'runtime versions/threads')
    for name, binding in runtime['loaded_module_files'].items():
        parent = Path(dep['shared_packages'][name]['directory']) if name in ('numpy','scipy') else Path(dep['directory'])/'sklearn'
        require(Path(binding['path']) == parent/'__init__.py' and expected_inputs[binding['path']] == binding['sha256'], 'loaded module origin '+name)
    require(all(p['num_threads'] == 1 and p['filepath'] in expected_inputs for p in runtime['threadpools']), 'recorded thread pools')
    raw = (ROOT/study['source']['local_path']).read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    source = list(reader); headers = reader.fieldnames
    ids = [r['Patient ID'] for r in source]
    require(ids == study['cohort']['ordered_patient_ids'] == cohort['patient_ids'], 'all forty ordered IDs')
    categories = [r['Postoperative Neurological Deficit'] for r in source]
    require(dict(collections.Counter(categories)) == study['endpoint']['expected_raw_counts'] and categories == cohort['raw_categories'], 'raw category preservation')
    y = [study['endpoint']['mapping'][c] for c in categories]
    X = [[float(r['Preoperative KPS']), float(r['Preoperative  contrast enhancing tumor volume (cm3)'])] for r in source]
    require(cohort['labels'] == y and cohort['features'] == X and cohort['feature_order'] == FEATURES, 'only permitted features and source labels')
    for i, row in enumerate(source):
        projected = {'schema':'rhuh-preoperative-projection-v1', 'collection_id':'RHUH-GBM',
            'source_version':'TCIA-v1-2023-06-09', 'patient_id':ids[i], 'preoperative_kps':int(X[i][0]),
            'preoperative_ce_volume_cm3':X[i][1], 'units':{'preoperative_kps':'KPS_score_points','preoperative_ce_volume_cm3':'cm3'},
            'phase':'preoperative', 'assessment_time':None, 'available_at':None,
            'scope':'preoperative_observational_research', 'primary_planner_authorized':False}
        require(projected == cohort['projection_records'][i] and 'sha256:'+digest(canonical(projected)) == cohort['projection_hashes'][i], 'independent allowed projection '+ids[i])
    protected = study['prohibited_field_invariance']['protected_headers']
    shadow = [{k:row[k] if k in protected else source[(i+1)%40][k] for k in headers} for i,row in enumerate(source)]
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=headers)
    writer.writeheader(); writer.writerows(shadow)
    require(digest(stream.getvalue().encode()) == cohort['shadow_sha256'], 'exact forbidden-field shadow bytes')
    require(cohort['invariance_features_hashes_labels'] is True and cohort['source_sha256_provenance_only'] == digest(raw), 'shadow/provenance scope')
    require(all(all(a[k] == b[k] for k in protected) for a,b in zip(source,shadow)), 'shadow protected fields unchanged')
    observed = folds(ledger('observed-folds.jsonl'), ids, X, y, 'observed')
    compare(state['observed'], observed, 'observed')
    import numpy as np  # Only declared label-generator replay, never fitting.
    require(str(Path(np.__file__).resolve()) == runtime['loaded_module_files']['numpy']['path'], 'audit PCG64 import origin')
    rng = np.random.Generator(np.random.PCG64(20261005))
    permutation_rows, permutation_summaries = ledger('permutation-folds.jsonl'), ledger('permutations.jsonl')
    require(len(permutation_rows) == 3960 and len(permutation_summaries) == 99, 'permutation denominators')
    statistics = []
    for i, summary in enumerate(permutation_summaries):
        labels = rng.permutation(y).tolist()
        require(summary['index'] == i and summary['label_vector'] == labels and summary['status'] == 'completed'
                and summary['fold_record_count'] == 40 and summary['folds_ledger'] == 'permutation-folds.jsonl', 'permutation map '+str(i))
        m = folds(permutation_rows[i*40:(i+1)*40], ids, X, labels, 'permutation:'+str(i))
        compare(summary['metrics'], m, 'permutation metrics '+str(i))
        statistic = m['paired_brier']['KPS_CE_VOLUME_minus_FOLD_PREVALENCE']
        compare(summary['statistic'], statistic, 'permutation statistic')
        statistics.append(summary['statistic'])
    statistic = state['observed']['paired_brier']['KPS_CE_VOLUME_minus_FOLD_PREVALENCE']
    tail_count = sum(v <= statistic for v in statistics)
    require(state['permutation']['planned'] == state['permutation']['completed'] == 99, 'completed permutation authority')
    compare(state['permutation']['statistic'], statistic, 'observed null statistic')
    compare(state['permutation']['p_value'], (1+tail_count)/100, 'fixed p denominator')
    influence_rows, influence_summaries = ledger('influence-folds.jsonl'), ledger('influence.jsonl')
    positives = [ids[i] for i, label in enumerate(y) if label]
    require(len(influence_rows) == 546 and len(influence_summaries) == 14, 'influence denominators')
    differences = []
    for j, (pid, summary) in enumerate(zip(positives,influence_summaries,strict=True)):
        require(summary['deleted_patient_id'] == pid and summary['status'] == 'completed'
                and summary['fold_record_count'] == 39 and summary['folds_ledger'] == 'influence-folds.jsonl', 'deletion identity')
        keep = [i for i in range(40) if ids[i] != pid]
        m = folds(influence_rows[j*39:(j+1)*39], [ids[i] for i in keep], [X[i] for i in keep], [y[i] for i in keep], 'delete:'+pid)
        compare(summary['metrics'], m, 'influence metrics '+pid)
        d = m['paired_brier']['KPS_CE_VOLUME_minus_KPS']
        compare(summary['paired_difference'], d, 'influence difference')
        compare(summary['change_from_full_cohort'], d-observed['paired_brier']['KPS_CE_VOLUME_minus_KPS'], 'influence change')
        differences.append(d)
    summary = {'minimum':min(differences),'maximum':max(differences),'negative':sum(v<0 for v in differences),
               'positive':sum(v>0 for v in differences),'zero':sum(v==0 for v in differences)}
    compare(state['influence']['summary'], summary, 'influence summary')
    require(state['influence']['completed'] == state['influence']['planned'] == 14, 'completed influence authority')
    require(COUNTS == {'logistic_predictions_reconstructed':9092,'all_predictions_checked':13638,'folds_checked':4546}, 'all fold accounting')
    require(state['fits'] == {'attempted':9092,'completed':9092} and state['failure'] is None
            and state['status'] == result['status'] == 'completed_observational_baseline', 'terminal completion')
    worker = read(OUT/'worker-result.json')
    require({k:v for k,v in worker.items() if k != 'worker_elapsed_seconds'} == state, 'state/worker equality')
    for key in ('baseline','worker'):
        require(sha(ROOT/result[key]['path']) == result[key]['sha256'], 'terminal hash '+key)
    supervision = read(OUT/'supervision/supervision.json')
    require(result['supervision'] == supervision, 'saved supervision equality')
    require(supervision['status'] == 'completed' and type(supervision['exit_code']) is int and supervision['exit_code'] == 0
            and all(supervision[k] is None for k in ('kill_reason','error','cleanup_error'))
            and supervision['no_retry'] is True and result['no_retry'] is True, 'terminal resource status')
    require(supervision['wall_cap_seconds'] == 195 and supervision['rss_cap_bytes'] == 1024**3
            and 0 <= supervision['sampled_peak_process_group_rss_bytes'] < 1024**3
            and 0 < worker['worker_elapsed_seconds'] < 180 and worker['worker_elapsed_seconds'] < supervision['elapsed_seconds'] < 195, 'fixed caps')
    require(sha(Path(supervision['command'][0]).resolve()) == release['interpreter']['sha256']
            and supervision['command'][-1] == result['baseline']['sha256'], 'actual interpreter/worker authority')
    phases = state['observed_seconds']+state['permutation']['elapsed_seconds']+state['influence']['elapsed_seconds']
    require(0 < phases <= state['evaluation_elapsed_seconds'] < worker['worker_elapsed_seconds'], 'nested timing scopes')
    require(before == inventory(OUT), 'raw output bytes unchanged during audit')
    require(all(sha(path) == h for path,h in expected_inputs.items()), 'bound input bytes unchanged after audit')
    return {'status':'passed', 'absolute_numeric_tolerance':TOL, 'relative_tolerance':0,
        'counts':dict(COUNTS), 'maximum_absolute_differences':dict(DELTAS), 'warnings':dict(WARNINGS),
        'source_commit':release['source_commit'], 'source_archive':release['source_archive'],
        'release':baseline['release_binding'], 'runtime_receipt':release['dependency_manifest'],
        'bound_inputs_verified_before_and_after':len(expected_inputs), 'runtime_inventory_counts':{k:len(v['files']) for k,v in inventories.items()},
        'raw_output_before_after_sha256':before, 'observed_recomputed':observed,
        'permutation':{'completed':99,'tail_count':tail_count,'denominator':100,'p_value':(1+tail_count)/100,
            'scope':'Unconditional full-model versus prevalence chance diagnostic; not incremental volume significance.'},
        'influence':{'completed':14,'summary':summary},
        'timing':{'evaluation_seconds':state['evaluation_elapsed_seconds'],'worker_seconds':worker['worker_elapsed_seconds'],
            'parent_seconds':supervision['elapsed_seconds'],'sampled_peak_process_group_rss_bytes':supervision['sampled_peak_process_group_rss_bytes'],
            'scope':'Nested intervals, not additive; sampled RSS may miss brief peaks.'},
        'fits_executed_by_this_checker':0, 'clinical_calibration_validated':False, 'planner_reward_authorized':False,
        'limits':['Checks saved evidence and declared byte inventories; no claim of hostile-host tamper resistance or per-instant monitoring.',
            'Frozen executed source records the fixed estimator configuration; saved coefficients do not independently prove the optimizer optimum.',
            'Forty previously QA-inspected patients, fourteen recorded category-positive labels; no untouched external test.',
            'Added volume worsens primary Brier versus KPS; secondary AUC/accuracy does not supersede this.',
            'Outcome domain, exact assessment time, and whether new or worsened remain unknown; no route or causal harm conclusion.']}


if __name__ == '__main__':
    started = time.monotonic()
    prior = read(REPORT).get('attempts',[]) if REPORT.exists() else []
    try:
        result = audit()
    except Exception as error:
        result = {'status':'failed', 'error_type':type(error).__name__, 'error':str(error),
                  'absolute_numeric_tolerance':TOL,'counts':dict(COUNTS),'maximum_absolute_differences':dict(DELTAS)}
    result['checker_sha256'] = sha(__file__)
    result['checked_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['audit_elapsed_seconds'] = time.monotonic()-started
    attempt = {k:v for k,v in result.items() if k in ('status','error_type','error','counts','maximum_absolute_differences','checker_sha256','checked_at_utc')}
    result['attempts'] = prior+[attempt]
    REPORT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','counts','maximum_absolute_differences') if k in result},sort_keys=True))
    if result['status'] != 'passed':
        print(result['error']); raise SystemExit(1)
