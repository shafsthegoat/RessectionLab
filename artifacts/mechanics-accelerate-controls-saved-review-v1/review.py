"""Independent saved-output replay. Never launches a solver or reads patients."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ATTEMPT = ROOT / 'outputs/mechanics/accelerate-controls-v1/attempt-01'
COMMIT = '2cb40187be6364c71072c46d8dd02ba50ec61d44'
RUNTIME_SHA = '13c4f60cbae8ad89232996e4f7d773bafbd2489f23a669c355f5ac09e417cc57'
CHECKERS = {'mechanics_febio_verification.py': 'b3f373743d6d107de929bf3283c167c6a3e0a68dfe74396ea0f1f8ad7cce2be5',
 'mechanics_patient_constraints.py': '04119660cf0a230428fe862d7897ea17953c51cf8756f87fa3152b426e303c66'}
CASES = ('zero', 'translation', 'finite_stretch', 'shear', 'shear_double_stiffness',
 'tet10_affine', 'mpc_translation', 'mpc_nonrigid')
CAPS = {'aggregate_seconds': 60, 'process_group_rss_bytes': 3221225472,
 'numerical_threads': 1, 'maximum_cases': 8, 'time_steps': 4, 'retries': 0}
THREADS = {'OMP_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE', 'VECLIB_MAXIMUM_THREADS': '1',
 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
OLD = b'<linear_solver type="skyline" />'
NEW = (b'<linear_solver type="accelerate"><iterative>0</iterative>'
 b'<factorization>4</factorization><order_method>0</order_method>'
 b'<print_condition_number>0</print_condition_number></linear_solver>')

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for raw in iter(lambda: stream.read(1024*1024), b''): digest.update(raw)
    return digest.hexdigest()

def read(path):
    assert Path(path).stat().st_size < 4*1024**2
    return json.loads(Path(path).read_text())

def bind(record):
    path = (ROOT / record['path']).resolve()
    assert path.is_relative_to(ROOT) and sha(path) == record['sha256']
    return path

def inventory(directory):
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}

start = time.monotonic()
raw_before = inventory(ATTEMPT)
execution = read(ATTEMPT / 'execution.json')
results = read(ATTEMPT / 'results.json')
base = read(ATTEMPT / 'execution-baseline.json')
assert sha(ATTEMPT / 'execution.json') == '7b3a15d237b034ec1988c5088f09b81adb935974b50dde59d2e87b028e61c0a5'
assert sha(ATTEMPT / 'results.json') == '765dc5ae422ac66d399b095a6f4df95e2e108e2276583441c13210d7440e24cf'
assert sha(ATTEMPT / 'execution-baseline.json') == '2a5c447b3d5bc407e71c189bce81fa374b2daeb88d9d95069e1ff6d924ae6a94'
assert execution['status'] == results['status'] == 'completed'
assert execution['worker_started'] is True and execution['no_retry'] is results['no_retry'] is True
assert not any(k in execution for k in ('error', 'summary_error'))
assert execution['thread_environment'] == THREADS
supervision = read(ATTEMPT / 'supervision/supervision.json')
assert execution['supervision'] == supervision
assert supervision['status'] == 'completed' and supervision['exit_code'] == 0
assert all(supervision.get(k) is None for k in ('kill_reason', 'error', 'cleanup_error'))
assert supervision['wall_cap_seconds'] == 60 and supervision['rss_cap_bytes'] == 3221225472
assert 0 < supervision['elapsed_seconds'] < 60
assert 0 < supervision['sampled_peak_process_group_rss_bytes'] < 3221225472
assert execution['command'] == supervision['command']
archive = Path(base['source_directory'])
assert archive == ROOT / 'build/validation/mechanics-accelerate-controls-execution-2cb40187/source'
assert Path(supervision['cwd']) == archive
assert execution['command'][1:] == [str(archive/'scripts/mechanics_accelerate_controls.py'), 'worker', '--output', str(ATTEMPT)]
assert base['source_commit'] == COMMIT and base['caps'] == CAPS
assert base['case_order'] == list(CASES) and base['attempt_directory'] == str(ATTEMPT)
release = read(base['release_path'])
assert release == base['release'] and release['authorized'] is True and release['root_release']
assert release['source_commit'] == COMMIT and release['source_directory'] == str(archive)
assert release['attempt_directory'] == str(ATTEMPT) and release['repository_directory'] == str(ROOT)
assert release['runtime_identity'] == {'path': 'artifacts/febio-accelerate-csc-runtime-v3/runtime-identity.json', 'sha256': RUNTIME_SHA}
identity = read(bind(release['runtime_identity']))
assert base['runtime_identity_sha256'] == results['runtime_identity_sha256'] == RUNTIME_SHA
assert base['executable'] == identity['executable']
assert base['solver_backend'] == results['solver_backend'] == 'accelerate'
assert results['solver_invocations'] == 8 and [r['case'] for r in results['cases']] == list(CASES)
input_before = {}
for filename, expected in base['input_hashes'].items():
    path = Path(filename).resolve()
    assert path.is_relative_to(archive) or path.is_relative_to(ROOT/'artifacts') or path.is_relative_to(ROOT/'data/optional-runtimes')
    assert sha(path) == expected
    input_before[str(path)] = expected
assert len(input_before) == 58
for record in (execution['inputs_after'], execution['final_inputs_after'], results['inputs_after']):
    assert set(record) == set(input_before) and all(r == {'unchanged': True} for r in record.values())
archive_before = inventory(archive)
assert len(archive_before) == 28
for filename, expected in archive_before.items():
    path = ROOT / filename
    relative = path.relative_to(archive)
    blob = subprocess.run(['git', '-C', str(ROOT), 'show', COMMIT+':'+str(relative)],
                          capture_output=True, check=True, timeout=5).stdout
    assert hashlib.sha256(blob).hexdigest() == expected == input_before[str(path)]
declaration = read(archive/'artifacts/mechanics-accelerate-controls-v1/declaration.json')
assert base['declaration'] == declaration
assert sha(archive/'artifacts/mechanics-accelerate-controls-v1/declaration.json') == '46a538f54471b76f3b75f50c31be2de6530b94450f88d2b6f9f8698ac27a54fd'
modules = []
for filename, expected in CHECKERS.items():
    path = archive / 'scripts' / filename
    assert sha(path) == expected
    spec = importlib.util.spec_from_file_location('saved_independent_'+filename[:-3], path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path
    modules.append(module)
hx, tet = modules
assert Path(tet.patch.__file__) == archive/'scripts/mechanics_febio_verification.py'
assert results['checker_import_paths'] == [str(archive/'scripts'/name) for name in CHECKERS]
assert results['numpy_version'] == hx.np.__version__
case_reviews = []; text_records = {}
for index, row in enumerate(results['cases']):
    name = row['case']; directory = ATTEMPT/name; dec = declaration['cases'][name]
    assert row['status'] == 'passed' and row['solver_exit_code'] == 0 and row['checker_passed'] is True
    assert 0 <= row['solver_seconds'] < 60
    assert row['command'] == [base['executable'], '-noconfig', '-no_title', '-i', name+'.feb', '-o', name+'.log']
    original = (archive/dec['original_path']).read_bytes()
    adapted = (directory/(name+'.feb')).read_bytes()
    assert adapted == (archive/dec['adapted_path']).read_bytes()
    assert original.count(OLD) == adapted.count(NEW) == 1 and original.replace(OLD, NEW, 1) == adapted
    assert sha(directory/(name+'.feb')) == dec['adapted_sha256'] == row['deck_sha256']
    console = hx.read_bounded_text(directory/'console.txt')
    selections = [line.strip() for line in console.splitlines() if 'selecting linear solver' in line.lower()]
    assert selections and all(re.search(r'\bselecting linear solver accelerate\b', line, re.I) for line in selections)
    assert row['backend_evidence'] == {'solver_backend':'accelerate','actual_selection_lines':selections,
      'solver_xml':NEW.decode(),'solver_only_change_verified':True,'runtime_identity_sha256':RUNTIME_SHA,
      'executed_deck_sha256':row['deck_sha256']}
    raw = [hx.read_bounded_text(directory/(name+'.'+suffix)) for suffix in ('nodes.log','elements.log','log')]
    text_records[name] = raw
    checked = (hx if index < 5 else tet).check_outputs(name, *raw)
    assert checked['passed'] is True and checked == read(directory/'checked.json')
    assert [s['time'] for s in checked['states']] == [0., .25, .5, .75, 1.]
    actual_outputs = {str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}
    assert actual_outputs == row['output_hashes']
    case_reviews.append({'case':name,'checker_passed':True,'saved_checked_result_exactly_reproduced':True,
      'solver_selection_count':len(selections),'states':len(checked['states']),
      'minimum_actual_sampled_J':min(s['minimum_actual_sampled_J'] for s in checked['states']),
      'maximum_constraint_error_m':max((s.get('maximum_constraint_error_m',0.) for s in checked['states']), default=0.),
      'solver_seconds':row['solver_seconds'],'raw_output_hashes':row['output_hashes']})
scaling = hx.check_stiffness_scaling(*text_records['shear'][:2], *text_records['shear_double_stiffness'][:2])
assert scaling['passed'] is True and scaling == results['stiffness_scaling']
summary_records = {}
for filename, names, field, status in [('hex8-summary.json',CASES[:5],'rows','passed_all_five_fixed_patch_controls'),
                                     ('tet10_mpc-summary.json',CASES[5:],'case_rows','three_actual_fixed_software_controls_passed')]:
    summary = read(ATTEMPT/filename)
    assert summary['status'] == status and summary['execution_accepted'] is True
    assert summary['source_commit'] == COMMIT and summary['runtime_identity_sha256'] == RUNTIME_SHA
    assert summary['solver_backend'] == 'accelerate' and summary['numerical_threads'] == 1
    assert summary['solver_invocations'] == len(names) and summary['case_order'] == list(names)
    assert summary['original_numerical_checker_unchanged'] is True
    assert set(summary['checker_inputs_after']) == set(results['checker_import_paths'])
    assert all(r == {'unchanged':True} for r in summary['checker_inputs_after'].values())
    for key, target in [('execution','execution.json'),('results','results.json'),('baseline','execution-baseline.json')]:
        assert bind(summary['evidence'][key]) == ATTEMPT/target
    assert [r['case'] for r in summary[field]] == list(names)
    for item in summary[field]:
        row = next(r for r in results['cases'] if r['case'] == item['case'])
        assert item['passed'] is True and {k:v for k,v in item.items() if k not in ('passed','evidence')} == row
        name = row['case']; directory = ATTEMPT/name
        expected_paths = {'original_deck':archive/declaration['cases'][name]['original_path'],
          'executed_deck':directory/(name+'.feb'),'console':directory/'console.txt',
          'nodes':directory/(name+'.nodes.log'),'elements':directory/(name+'.elements.log'),
          'solver_log':directory/(name+'.log'),'checked':directory/'checked.json',
          'checker_source':archive/'scripts'/('mechanics_febio_verification.py' if name in CASES[:5] else 'mechanics_patient_constraints.py')}
        assert set(item['evidence']) == set(expected_paths)
        for key, path in expected_paths.items(): assert bind(item['evidence'][key]) == path
    if field == 'rows': assert summary['stiffness_scaling'] == scaling
    summary_records[filename] = {'path':str((ATTEMPT/filename).relative_to(ROOT)), 'sha256':sha(ATTEMPT/filename)}
assert raw_before == inventory(ATTEMPT)
assert archive_before == inventory(archive)
assert all(sha(path) == expected for path,expected in input_before.items())
record = {'schema':'mechanics-accelerate-controls-saved-review-v1','status':'eight_saved_numerical_controls_independently_passed',
 'audit_source_sha256':sha(__file__),'elapsed_seconds':time.monotonic()-start,'source_commit':COMMIT,
 'runtime_identity_sha256':RUNTIME_SHA,'checker_pins':CHECKERS,'checker_import_paths':results['checker_import_paths'],
 'numpy_version':hx.np.__version__,'case_reviews':case_reviews,'stiffness_scaling_exactly_reproduced':True,
 'saved_summary_bindings':summary_records,'recorded_original_solver_calls':8,'new_solver_calls_by_auditor':0,
 'observed_run':{'launcher_seconds':execution['launcher_seconds'],'worker_seconds':results['worker_seconds'],
 'supervised_seconds':supervision['elapsed_seconds'],'sampled_peak_process_group_rss_bytes':supervision['sampled_peak_process_group_rss_bytes'],
 'caps':CAPS,'thread_environment':THREADS},
 'raw_attempt_inventory':raw_before,'source_archive_inventory':archive_before,'all_58_run_input_hashes_reverified':True,
 'all_audited_files_unchanged':True,'patient_or_measured_specimen_curve_access':False,
 'scope':'Actual eight analytical software-control outputs re-evaluated using the original unchanged checkers; no solver re-execution.',
 'limits':['These tiny controls do not establish patient mesh convergence, tissue properties, specimen fit, patient observation support, cutting/contact mechanics or clinical validity.',
 'Process-group RSS was sampled; between-sample peaks may be missed. The first solver call includes unseparated startup overhead; no latency-distribution claim.',
 'Original runtime identity is retained as historical build evidence pending numerical controls; this separate review records the subsequent limited numerical acceptance.']}
(OUT/'review.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'elapsed_seconds':record['elapsed_seconds'],
 'receipt_sha256':sha(OUT/'review.json'),'case_minimum_J':{r['case']:r['minimum_actual_sampled_J'] for r in case_reviews},
 'maximum_constraint_error_m':max(r['maximum_constraint_error_m'] for r in case_reviews)}))
