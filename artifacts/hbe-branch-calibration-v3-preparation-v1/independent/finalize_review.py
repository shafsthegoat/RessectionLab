"""Finalize independent review, hashing saved evidence only; no execution release."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = json.loads((OUT/'baseline.json').read_text())
checked = json.loads((OUT/'guarded-checks.json').read_text())
v2 = json.loads((ROOT/'manifests/experiments/hbe-01-03-branch-calibration-v2.json').read_text())
v3 = json.loads((ROOT/'manifests/experiments/hbe-01-03-branch-calibration-v3.json').read_text())
changes = []


def walk(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                changes.append({'path': path+'/'+key, 'kind': 'added' if key not in a else 'removed'})
            else:
                walk(a[key], b[key], path+'/'+key)
    elif a != b:
        changes.append({'path': path, 'before': a, 'after': b})


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


walk(v2, v3)
fields = sorted(set(v2)-{'schema', 'study_id', 'output_root', 'csv_schemas', 'runtime_migration'})
assert len(fields) == 23 and all(v2[key] == v3[key] for key in fields)
assert v2['runtime_migration'] == v3['runtime_migration']
assert {row['path'] for row in changes} == {
    '/csv_schemas/compression/header', '/csv_schemas/tension/header',
    '/csv_header_migration', '/schema', '/study_id', '/output_root'}
current = {p: sha(ROOT/p) for p in before['existing_files_sha256']}
outputs = {str(p.relative_to(ROOT)): sha(p) for version in ('v1', 'v2')
           for p in (ROOT/f'outputs/mechanics/hbe-01-03-branch-calibration-{version}').rglob('*') if p.is_file()}
candidates = {p: sha(ROOT/p) for p in before['candidate_sources']}
diff_sha = hashlib.sha256(git('diff', '--binary')).hexdigest()
staged_sha = hashlib.sha256(git('diff', '--cached', '--binary')).hexdigest()
assert current == before['existing_files_sha256']
assert outputs == before['prior_output_sha256'] and len(outputs) == 14
assert candidates == before['candidate_sources']
assert diff_sha == before['tracked_diff_sha256'] and staged_sha == before['staged_diff_sha256']
assert not (ROOT/v3['output_root']).exists()
assert checked['pytest_exit_code'] == 0 and not any(checked['guard_audit'].values())
for path in candidates:
    if path.endswith('.py'):
        ast.parse((ROOT/path).read_text())
    assert subprocess.run(['git', 'ls-files', '--error-unmatch', path], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 1
    assert sha(OUT/'source-snapshot'/path) == candidates[path]
assert subprocess.run(['git', 'merge-base', '--is-ancestor', 'eabe64b', 'HEAD'], cwd=ROOT).returncode == 0
lineage_commit = {}
for kind, mirror in [
    ('diagnosis', 'artifacts/hbe-branch-calibration-v2-failure-audit-v1/axial-diagnostic/diagnosis.json'),
    ('failure_review', 'artifacts/hbe-branch-calibration-v2-failure-audit-v1/failure-review/verification.json'),
]:
    binding = v3['csv_header_migration'][kind]
    digest = hashlib.sha256(git('show', 'eabe64b:'+mirror)).hexdigest()
    assert digest == binding['sha256'] == sha(ROOT/mirror) == sha(ROOT/binding['path'])
    lineage_commit[kind] = {'bound_build_path': binding['path'], 'committed_mirror': mirror,
                            'sha256': digest, 'committed_in': 'eabe64b'}
interpreter = Path(v3['interpreter']['path'])
current_interpreter = {'resolved_path': str(interpreter.resolve()), 'sha256': sha(interpreter)}
assert current_interpreter['sha256'] == v3['interpreter']['sha256']
relevant_paths = [p for p in current if p.startswith(('scripts/', 'tests/', 'manifests/'))
                  and ('branch_calibration' in p or 'branch-calibration' in p)]
assert git('diff', '--name-only', 'HEAD', '--', *relevant_paths) == b''
report = {
    'schema': 'hbe-branch-calibration-v3-independent-review-v1',
    'status': 'go_for_candidate_source_commit_and_separate_fresh_execution_release',
    'go_no_go': 'GO for exact reviewed v3 header-only preparation; no execution release is issued by this review.',
    'actual_execution_authorized': False, 'source_commit_created': False,
    'review_head': before['head'], 'final_head': git('rev-parse', 'HEAD').decode().strip(),
    'candidate_sources': candidates, 'findings': [], 'exact_manifest_delta': changes,
    'preserved': {
        'scientific_historical_fields': fields, 'scientific_historical_field_count': len(fields),
        'v1_runtime_migration_unchanged_and_full_chain_reverified': True,
        'interpreter': current_interpreter,
        'all_csv_nonheader_fields_unchanged': True,
        'torsion_schema_unchanged_and_uninspected': True,
        'numerical_access_freeze_runtime_functions': 'AST-equal to v2 after explicit branch namespace/schema normalization; only preflight lineage hook changes and new migration verifier.',
        'runner_readout': 'Entire source text equal after version namespace/schema normalization.',
        'source_binding_closure': '26 inherited source bindings plus 3 v3 sources; 8 inherited declarations and v3 study remain exact-hash checked against committed source archive before data access.',
        'split': 'Both axial branches remain calibration; both torsion branches remain conditional held-out validation after durable parameter/prediction freeze and numerical confirmation.',
        'bounds': v3['budgets'], 'fitted_parameters': v3['fitted_parameters'],
        'physical_binary_threshold': v3['physical_binary_threshold'],
    },
    'failure_lineage': {
        'v1': 'Interpreter drift before measured-data access, zero fit/native calls; exact saved failed state retained.',
        'v2': 'Compression member 1,222 bytes decoded, parse failed on displacement header before a numeric curve or fit; tension payload unread; torsion unread; calibration_responses_accessed remains null.',
        'evidence_committed_in_ancestor': lineage_commit,
        'metadata_bindings_verified_and_rehashed_count': len(checked['lineage_bound_and_rehashed']),
    },
    'verification': {
        'command': '.venv/bin/python -B build/hbe-branch-calibration-v3-independent-v1/run_guarded_checks.py',
        'focused_test_result': '227 passed, 1 deselected in 0.96s',
        'deselected': 'The v2 synthetic analytical-fit test, to honor a literal no-fit restriction.',
        'independent_strict_header_fixture_checks': 28,
        'real_header_evidence': v3['csv_header_migration']['diagnosis'],
        'fixture_scope': 'Only authenticated literal displacement,force header reused; all numeric rows independently invented; no archive reopened.',
        'guarded_check_result': {'path': str((OUT/'guarded-checks.json').relative_to(ROOT)), 'sha256': sha(OUT/'guarded-checks.json')},
        'test_log': {'path': str((OUT/'pytest-independent.log').relative_to(ROOT)), 'sha256': sha(OUT/'pytest-independent.log')},
        'guard_audit': checked['guard_audit'],
    },
    'integrity': {
        'candidate_sources_unchanged': True, 'candidate_sources_still_untracked': True,
        'existing_files_unchanged_during_review': len(current),
        'owner_baseline_mismatches': before['owner_baseline_mismatches'],
        'owner_baseline_mismatch_interpretation': 'Only PROJECT_STATUS.md changed in unrelated committed research/docs activity before this review; the other 216 owner-baseline files match.',
        'v1_v2_tracked_source_changes': [], 'desktop_preexisting_diff_unchanged': True,
        'tracked_diff_paths': before['tracked_diff_paths'], 'tracked_diff_sha256': diff_sha,
        'staged_diff_sha256': staged_sha, 'prior_output_files_unchanged': len(outputs),
        'new_v3_output_root_exists': False, 'measured_archive_opens': 0,
        'torsion_payload_opens': 0, 'fit_calls': 0, 'native_calls': 0, 'output_reruns': 0,
    },
    'torsion_header_judgment': {
        'current_v3_recommendation': 'Preserve inherited header:null and strict uninspected holdout for this exact scope; accept that an unexpected literal torsion header can cause a fail-closed parse after the durable freeze and native confirmation.',
        'current_v3_risk': 'A syntax mismatch could spend the two bounded native runs before failing; no completed torque-validation claim is warranted until conditional reveal succeeds.',
        'separately_declared_check_scientifically_justifiable': True,
        'rationale': 'Header syntax alone can resolve a format contract without exposing response values or fitting to hold-out outcomes; independence of numerical torque-response evaluation is preserved if the scientific model, objective, roles, thresholds and prediction-freeze rules are fixed beforehand.',
        'provenance_tradeoff': 'A CSV header is member payload, not ZIP central-directory metadata. Calling this metadata-only must not imply no torsion payload access. A separate authorized diagnostic must record scope, exact member/hash identity, actual plaintext exposure and decompressor buffering limits; any earlier complete-sealing claim needs an explicit amendment.',
        'required_separate_scope': 'Bounded first-record inspection with no numeric-row parsing or display, no fitting, no native run and no schema autodetection; save literal headers and provenance only, review any declared schema amendment, then issue a fresh explicit execution release. This is outside the present exact two-axial-header migration.',
        'reviewer_torsion_inspection_performed': False,
    },
    'limitations': [
        'Preparation verification is not an executed calibration or numerical/physical validation result.',
        'The inherited access ledger remains batch-attempt/completion accounting; v2 per-member exposure is source/traceback-derived, not a durable per-member read receipt.',
        'The corrected axial header schema was tested only on synthetic numeric rows; measured numeric curves remain unexamined in this review.',
    ],
}
(OUT/'verification.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
index = {'schema': 'hbe-branch-calibration-v3-independent-artifact-index-v1',
         'files': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*'))
                   if p.is_file() and p.name != 'artifact-index.json'}}
(OUT/'artifact-index.json').write_text(json.dumps(index, indent=2, sort_keys=True)+'\n')
print(json.dumps({'status': report['status'], 'verification_sha256': sha(OUT/'verification.json'),
                  'artifact_index_sha256': sha(OUT/'artifact-index.json'),
                  'test_result': report['verification']['focused_test_result'],
                  'independent_header_cases_passed': 28}, indent=2))
