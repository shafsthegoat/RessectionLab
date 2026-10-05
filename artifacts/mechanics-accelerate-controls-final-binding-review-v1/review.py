"""Narrow frozen declaration review. Stdlib only; never imports/executes a solver."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PINS = {
 'artifacts/mechanics-accelerate-controls-generation-v1/generation.json': '6c82de6c63d575074698b22d3c812e292956958f3f6018c94f0d04752bdc0806',
 'artifacts/mechanics-accelerate-controls-v1/declaration.json': '46a538f54471b76f3b75f50c31be2de6530b94450f88d2b6f9f8698ac27a54fd',
 'artifacts/febio-accelerate-csc-runtime-v3/runtime-identity.json': '13c4f60cbae8ad89232996e4f7d773bafbd2489f23a669c355f5ac09e417cc57',
 'artifacts/febio-accelerate-runtime-saved-review-v3/review.json': '96982e2b8b1e368e320ad63179295542c7b77ac66692b8a6f8e4f9a8878296a1',
 'artifacts/mechanics-accelerate-controls-independent-review-v1/verification.json': '5ebfae49d6e53075da9e3698c83c974242a3c959ed2c721ea8159216b88638f1',
}
CASES = ('zero', 'translation', 'finite_stretch', 'shear', 'shear_double_stiffness',
         'tet10_affine', 'mpc_translation', 'mpc_nonrigid')
SOURCES = ('scripts/mechanics_accelerate_controls.py', 'scripts/mechanics_hbe_backend.py',
 'scripts/mechanics_hbe_access.py', 'scripts/mechanics_hbe_evaluation.py',
 'scripts/mechanics_patient_constraints_run.py', 'scripts/mechanics_patient_constraints.py',
 'scripts/mechanics_febio_verification.py', 'scripts/febio_runtime.py')
CHECKERS = {'scripts/mechanics_febio_verification.py': 'b3f373743d6d107de929bf3283c167c6a3e0a68dfe74396ea0f1f8ad7cce2be5',
 'scripts/mechanics_patient_constraints.py': '04119660cf0a230428fe862d7897ea17953c51cf8756f87fa3152b426e303c66'}
CAPS = {'aggregate_seconds': 60, 'process_group_rss_bytes': 3221225472,
 'numerical_threads': 1, 'maximum_cases': 8, 'time_steps': 4, 'retries': 0}
OLD = b'<linear_solver type="skyline" />'
NEW = (b'<linear_solver type="accelerate"><iterative>0</iterative>'
 b'<factorization>4</factorization><order_method>0</order_method>'
 b'<print_condition_number>0</print_condition_number></linear_solver>')

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def read(relative):
    return json.loads((ROOT / relative).read_text())

start = time.monotonic()
assert all(sha(ROOT / path) == expected for path, expected in PINS.items())
generation, declaration, identity, review, prior = map(read, PINS)
assert generation['generation_source_commit'] == '5ee77bc2d2d015deed9e5edb28dd7e3fb90cce09'
assert generation['status'] == 'verified_declaration_prepared_no_solver'
assert generation['execution_authorized'] is False and generation['solver_invocations'] == 0
assert declaration['schema'] == 'mechanics-accelerate-controls-v1-declaration'
assert declaration['status'] == 'prepared_not_executed'
assert generation['caps'] == declaration['caps'] == CAPS
assert generation['case_order'] == declaration['case_order'] == list(CASES)
assert set(declaration['cases']) == set(CASES) and set(declaration['sources']) == set(SOURCES)
assert generation['source_hashes'] == declaration['sources']
assert generation['archive_inventory_before'] == generation['archive_inventory_after']
archive = (ROOT / generation['generation_archive']).resolve()
assert archive.is_relative_to(ROOT / 'build/validation')
assert len(generation['archive_inventory_before']) == 27
assert {str(p.relative_to(archive)) for p in archive.rglob('*') if p.is_file()} == set(generation['archive_inventory_before'])
source_rows = {}
for relative, expected in generation['archive_inventory_before'].items():
    raw = subprocess.run(['git', '-C', str(ROOT), 'show',
        generation['generation_source_commit'] + ':' + relative],
        capture_output=True, check=True, timeout=5).stdout
    assert hashlib.sha256(raw).hexdigest() == expected
    assert sha(archive / relative) == sha(ROOT / relative) == expected
    source_rows[relative] = expected
assert generation['generator_import_path'] == str(archive / SOURCES[0])
assert generation['backend_import_path'] == str(archive / SOURCES[1])
assert generation['helper_import_paths'] == {
 'scripts.mechanics_hbe_access': str(archive / SOURCES[2]),
 'scripts.mechanics_hbe_evaluation': str(archive / SOURCES[3])}
for path, expected in CHECKERS.items():
    assert declaration['sources'][path] == sha(ROOT / path) == expected
assert prior['source_before'][SOURCES[0]] == declaration['sources'][SOURCES[0]]
assert prior['result'] == '31 passed in 0.38s'
deck_rows = {}
for index, case in enumerate(CASES):
    row = declaration['cases'][case]
    folder = 'mechanics-febio-verification-v1' if index < 5 else 'mechanics-patient-constraints-runtime-v1'
    assert row['original_path'] == f'artifacts/{folder}/decks/{case}.feb'
    assert row['adapted_path'] == f'artifacts/mechanics-accelerate-controls-v1/decks/{case}.feb'
    assert row['checker'] == ('hex8' if index < 5 else 'tet10_mpc')
    original = (ROOT / row['original_path']).read_bytes()
    adapted = (ROOT / row['adapted_path']).read_bytes()
    staged = ROOT / generation['staging_directory'] / 'decks' / (case + '.feb')
    assert original.count(OLD) == adapted.count(NEW) == 1
    assert original.replace(OLD, NEW, 1) == adapted == staged.read_bytes()
    assert sha(ROOT / row['original_path']) == row['original_sha256']
    assert sha(ROOT / row['adapted_path']) == row['adapted_sha256']
    xml = ET.fromstring(adapted)
    assert len(xml.findall('.//linear_solver')) == 1
    assert xml.findtext('Control/time_steps') == '4'
    assert float(xml.findtext('Control/step_size')) == .25
    deck_rows[case] = {'original_sha256': row['original_sha256'],
      'adapted_sha256': row['adapted_sha256'], 'exact_solver_only_replacement': True}
constants = {}
for node in ast.parse((archive / SOURCES[1]).read_text()).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        try: constants[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError): pass
assert declaration['profile_id'] == constants['PROFILE_ID'] == 'accelerate_csc_v1'
assert constants['PATCH_SHA'] == identity['source_patch_sha256'] == '67a1858f2d55046796d7ccb46758ca06e5bfb673e75a539d158b1a17652f3340'
assert constants['PATCHED_SOURCE_SHA'] == identity['patched_source_sha256'] == '476ac8471ea681a99c298352a50aba2a7a5baa71635e1678155a7f58b72ba04a'
assert generation['runtime_identity']['sha256'] == review['runtime_identity_sha256'] == PINS['artifacts/febio-accelerate-csc-runtime-v3/runtime-identity.json']
assert generation['runtime_identity']['path'] == 'artifacts/febio-accelerate-csc-runtime-v3/runtime-identity.json'
assert generation['saved_runtime_review']['sha256'] == PINS['artifacts/febio-accelerate-runtime-saved-review-v3/review.json']
assert review['status'] == 'saved_build_and_static_runtime_closure_review_passed'
assert identity['status'] == 'isolated_patched_runtime_built_pending_numerical_controls'
assert identity['solver_executed'] is False and identity['models_executed'] == 0
assert len(identity['libraries']) == 13 and len(review['installed_macho']) == 13
assert {r['path']: r['sha256'] for r in review['installed_macho']} == identity['libraries']
assert generation['runtime_input_hashes_before'] == generation['runtime_input_hashes_after']
assert len(generation['runtime_input_hashes_before']) == 31
for filename, expected in generation['runtime_input_hashes_before'].items():
    path = Path(filename).resolve()
    assert path.is_relative_to(ROOT / 'artifacts') or path.is_relative_to(ROOT / 'data/optional-runtimes')
    assert sha(path) == expected
assert read('artifacts/mechanics-accelerate-controls-v1/release-template.json')['authorized'] is False
assert all(sha(ROOT / path) == expected for path, expected in PINS.items())
assert all(sha(ROOT / path) == sha(archive / path) == expected for path, expected in source_rows.items())
result = {'schema': 'mechanics-accelerate-controls-final-binding-review-v1',
 'status': 'final_preparation_bindings_review_passed_execution_not_authorized',
 'audit_source_sha256': sha(__file__), 'elapsed_seconds': time.monotonic() - start,
 'input_bindings': PINS, 'generation_source_commit': generation['generation_source_commit'],
 'generation_archive': generation['generation_archive'], 'archive_current_git_equal_files': 27,
 'declared_source_files': 8, 'unchanged_original_checker_pins': CHECKERS,
 'runtime_identity_sha256': generation['runtime_identity']['sha256'],
 'runtime_generation_inputs_rehashed': 31, 'all_read_inputs_unchanged': True,
 'deck_checks': deck_rows, 'caps': CAPS,
 'prior_generic_controls_reused': {'passed': 31, 'repeated': False},
 'solver_invocations': 0, 'patient_or_measured_curve_access': False,
 'next_gate': 'Commit declaration and archive exact 28-file execution closure; root supplies separate one-attempt release with this exact runtime identity.',
 'limits': ['Accepted build/static-closure evidence is preserved; numerical runtime behavior remains untested here.',
            'No synthetic-control success, patient mechanics, calibration, clinical or surgical accuracy is claimed.']}
(OUT / 'review.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                  'receipt_sha256': sha(OUT / 'review.json')}))
