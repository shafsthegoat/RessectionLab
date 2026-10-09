"""Build an unauthorized frozen-source candidate without running study code."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
DEST = ROOT / 'build/hbe-branch-calibration-v3'
FROZEN = DEST / 'source'
STUDY_PATH = 'manifests/experiments/hbe-01-03-branch-calibration-v3.json'
STUDY_SHA = 'e644ab3a46d52c6530b9264e9315ce7eb139e03cdb93e197ca1d60a81329d01c'
RESUME = sys.argv[1:] == ['--resume-preparation']
assert not sys.argv[1:] or RESUME

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save(path, obj):
    if RESUME and path.exists():
        assert json.loads(path.read_text()) == obj, 'Refuse changing existing preparation: ' + str(path)
        return
    with path.open('x') as stream:
        stream.write(json.dumps(obj, indent=2, sort_keys=True) + '\n')

def binding(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest(path.read_bytes())}

def verify(binding):
    path = ROOT / binding['path']
    data = path.read_bytes()
    assert digest(data) == binding['sha256'], binding['path']
    return data

def git_state():
    return {'head': git('rev-parse', 'HEAD').decode().strip(),
            'tracked_diff_sha256': digest(git('diff', '--binary')),
            'staged_diff_sha256': digest(git('diff', '--cached', '--binary')),
            'tracked_status': git('status', '--porcelain', '--untracked-files=no').decode().splitlines()}

started = datetime.now(timezone.utc).isoformat()
initial = git_state()
if RESUME:
    baseline = json.loads((AUDIT / 'baseline.json').read_text())
    assert initial == baseline['git']
    started = baseline['started_utc']
commit = git('rev-parse', 'e96f706^{commit}').decode().strip()
preparation_commit = git('rev-parse', 'e8b7c52^{commit}').decode().strip()
receipt_commit = git('rev-parse', 'e596c8f^{commit}').decode().strip()
study_bytes = git('show', commit + ':' + STUDY_PATH)
assert digest(study_bytes) == STUDY_SHA
study = json.loads(study_bytes)
assert DEST.exists() is RESUME, 'Preparation path existence must match explicit resume mode'
assert not (ROOT / study['output_root']).exists(), 'Refuse existing v3 execution output'
for ancestor, descendant in [(commit, preparation_commit), (preparation_commit, receipt_commit),
                             (receipt_commit, initial['head'])]:
    subprocess.run(['git', 'merge-base', '--is-ancestor', ancestor, descendant], cwd=ROOT, check=True)

core = ast.parse(git('show', commit + ':scripts/mechanics_hbe_branch_calibration_v3.py'))
new_sources = next(ast.literal_eval(n.value) for n in core.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'NEW_SOURCES' for t in n.targets))
source_paths = {k: v['path'] for k, v in study['inherited_sources'].items()}
source_paths.update({k: 'scripts/' + v for k, v in new_sources.items()})
assert len(source_paths) == 29 and len(set(source_paths.values())) == 29
declarations = study['inherited_declarations'] + [{'path': STUDY_PATH, 'sha256': STUDY_SHA}]
expected = {b['path']: b['sha256'] for b in declarations}
for key, path in source_paths.items():
    data = git('show', commit + ':' + path)
    expected[path] = digest(data)
    if key in study['inherited_sources']:
        assert expected[path] == study['inherited_sources'][key]['sha256']
assert len(declarations) == 9 and len(expected) == 38
for path, sha in expected.items():
    assert digest(git('show', commit + ':' + path)) == sha
    assert digest((ROOT / path).read_bytes()) == sha, 'Live declared file differs: ' + path

roles = json.loads(verify(study['roles']))
assert roles['selection']['specimen_id'] == 'HBE_01_03'
permitted = [m['path'] for m in roles['calibration']['members']]
held_out = [m['path'] for m in roles['held_out_validation']['members']]
assert permitted == ['HBE_01/HBE_01_03/compression_c3.csv', 'HBE_01/HBE_01_03/tension_c3.csv']
assert held_out == ['HBE_01/HBE_01_03/torsion_l1_c3_neg.csv', 'HBE_01/HBE_01_03/torsion_l1_c3_pos.csv']
receipts = {}
for key, b in study['csv_header_migration'].items():
    if isinstance(b, dict) and set(b) == {'path', 'sha256'}:
        data = verify(b)
        assert git('show', receipt_commit + ':' + b['path']) == data
        receipts[key] = dict(b, bytes=len(data), committed_at=receipt_commit)
changed_receipts = git('diff-tree', '--no-commit-id', '--name-only', '-r', receipt_commit).decode().splitlines()
assert len(changed_receipts) == 9
assert set(changed_receipts) == {v['path'] for k, v in receipts.items() if k != 'previous_study'}
prior_state = json.loads(verify(study['csv_header_migration']['previous_state']))
assert prior_state['calibration_access_attempted'] is True
assert prior_state['calibration_responses_accessed'] is None
assert prior_state['held_out_access_attempted'] is False and prior_state['held_out_responses_accessed'] is False
assert prior_state['native_calls'] == prior_state['mesher_calls'] == 0
assert prior_state['automatic_retry'] is False and 'fit' not in prior_state

review_path = ROOT / 'artifacts/hbe-branch-calibration-v3-preparation-v1/independent/verification.json'
review_bytes = review_path.read_bytes()
assert git('show', preparation_commit + ':' + str(review_path.relative_to(ROOT))) == review_bytes
review = json.loads(review_bytes)
assert review['findings'] == [] and review['actual_execution_authorized'] is False
assert review['status'] == 'go_for_candidate_source_commit_and_separate_fresh_execution_release'
for path, sha in review['candidate_sources'].items():
    assert digest(git('show', commit + ':' + path)) == sha

interpreter = Path(study['interpreter']['path']).resolve()
assert str(interpreter) == study['interpreter']['path']
assert digest(interpreter.read_bytes()) == study['interpreter']['sha256']
prior_supervision = json.loads(verify(study['csv_header_migration']['previous_supervision']))
invocation = Path(prior_supervision['command'][0])
assert invocation == ROOT / '.venv/bin/python' and invocation.resolve() == interpreter
backend_bytes = verify(study['backend_profile'])
backend = json.loads(backend_bytes)
assert backend['runtime_identity'] == study['runtime_identity']
verify(study['runtime_identity'])

protected_bindings = [{'path': p, 'sha256': s} for p, s in expected.items()]
protected_bindings += [study['roles'], study['backend_profile'], study['runtime_identity']]
protected_bindings += [{k: v[k] for k in ('path', 'sha256')} for v in receipts.values()]
protected_bindings += [binding(review_path)]
save(AUDIT / 'baseline.json', {'schema': 'hbe-v3-release-preparation-baseline-v1',
     'started_utc': started, 'git': initial, 'protected_bindings': protected_bindings,
     'v3_output_root_exists': False, 'measured_archive_opened': False})

if not RESUME:
    DEST.mkdir(exist_ok=False)
archive = DEST / 'source.tar'
archive_command = ['git', 'archive', '--format=tar', '--output=' + str(archive), commit, '--', *sorted(expected)]
if not RESUME:
    subprocess.run(archive_command, cwd=ROOT, check=True)
    FROZEN.mkdir(exist_ok=False)
members = []
with tarfile.open(archive, 'r:') as bundle:
    assert bundle.pax_headers.get('comment') == commit
    all_members = bundle.getmembers()
    files = [member for member in all_members if member.isfile()]
    assert len(files) == 38 and {member.name for member in files} == set(expected)
    assert len({m.name for m in all_members}) == len(all_members)
    assert all(member.isfile() or member.isdir() for member in all_members)
    for member in all_members:
        target = FROZEN / member.name
        assert target.resolve().is_relative_to(FROZEN)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            data = bundle.extractfile(member).read()
            assert digest(data) == expected[member.name]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not RESUME:
                with target.open('xb') as stream:
                    stream.write(data)
            assert digest(target.read_bytes()) == expected[member.name]
            blob = git('rev-parse', commit + ':' + member.name).decode().strip()
            members.append({'path': member.name, 'sha256': expected[member.name],
                            'bytes': len(data), 'git_blob': blob,
                            'extracted_path': str(target.relative_to(ROOT))})
source_bindings = {key: binding(FROZEN / path) for key, path in source_paths.items()}
candidate = {'schema': 'hbe-branch-calibration-release-v3', 'authorized': False,
             'phase': 'calibrate_and_validate', 'authorized_native_calls': 2,
             'study': {'path': STUDY_PATH, 'sha256': STUDY_SHA},
             'interpreter': study['interpreter'], 'backend_profile': study['backend_profile'],
             'predecessor_failure': study['csv_header_migration']['previous_state'],
             'permitted_members': permitted, 'conditional_held_out_members': held_out,
             'source_commit': commit, 'source_archive': binding(archive),
             'source_bindings': source_bindings, 'output_root': study['output_root'],
             'execution': {'access_ledger_path': study['output_root'] + '/experiment/access.jsonl',
                           'csv_schemas': study['csv_schemas']}}
candidate_path = DEST / 'release.candidate.json'
save(candidate_path, candidate)
save(AUDIT / 'archive-verification.json', {'schema': 'hbe-v3-source-archive-verification-v1',
     'source_commit': commit, 'global_commit_comment': commit, 'source_archive': binding(archive),
     'archive_bytes': archive.stat().st_size, 'file_member_count': len(members),
     'python_source_count': 29, 'declaration_count': 9, 'members': sorted(members, key=lambda m: m['path']),
     'archive_command_argv': archive_command, 'archive_command_shell': shlex.join(archive_command),
     'all_archive_and_extracted_hashes_match_commit_and_declaration': True})

environment = {key: value for key, value in os.environ.items()
               if not key.startswith('PYTHON') and key not in {'__PYVENV_LAUNCHER__', 'LD_PRELOAD', 'LD_LIBRARY_PATH'}}
environment.update({k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')})
import_command = [str(invocation), '-I', '-B', str(AUDIT / 'guarded_imports.py'), str(ROOT),
                  str(FROZEN), json.dumps(source_bindings, sort_keys=True)]
child = subprocess.run(import_command, cwd=FROZEN, env=environment, capture_output=True, text=True, timeout=60)
attempt = 1 + len(list(AUDIT.glob('import-audit*.stdout.json')))
import_stdout = AUDIT / (f'import-audit-attempt-{attempt:02d}.stdout.json' if RESUME else 'import-audit.stdout.json')
import_stderr = AUDIT / (f'import-audit-attempt-{attempt:02d}.stderr.txt' if RESUME else 'import-audit.stderr.txt')
with import_stdout.open('x') as stream:
    stream.write(child.stdout)
with import_stderr.open('x') as stream:
    stream.write(child.stderr)
assert child.returncode == 0, 'Guarded imports failed; inspect saved audit'
imports = json.loads(child.stdout)
assert imports['status'] == 'passed' and imports['module_count'] == 29 and imports['violations'] == []

for b in protected_bindings:
    verify(b)
assert initial == git_state(), 'Tracked worktree, staging or HEAD changed during preparation'
assert not (ROOT / study['output_root']).exists()
assert not (DEST / 'release.json').exists()
assert digest(interpreter.read_bytes()) == study['interpreter']['sha256']
assert {p.relative_to(FROZEN).as_posix() for p in FROZEN.rglob('*') if p.is_file()} == set(expected)
for path, sha in expected.items():
    assert digest((FROZEN / path).read_bytes()) == sha
execution_command = [str(invocation), '-I', '-B', str(FROZEN / source_paths['branch_calibration_runner']),
                     '--root', str(ROOT), '--release', 'build/hbe-branch-calibration-v3/release.json',
                     '--release-sha256', '<ROOT_AUTHORIZED_RELEASE_SHA256>', '--execute']
commands = {'schema': 'hbe-v3-release-preparation-commands-v1', 'archive_argv': archive_command,
            'import_audit_argv': import_command, 'import_audit_cwd': str(FROZEN),
            'future_execution_argv_template': execution_command, 'future_execution_cwd': str(FROZEN),
            'future_execution_command_shell_template': shlex.join(execution_command),
            'executed': ['git archive', 'guarded import-origin audit (initial absent-cache probe blocked)',
                         'guarded import-origin audit (direct base interpreter lacked SciPy)',
                         'guarded import-origin audit (exact predecessor virtualenv invocation)'] if RESUME else
                        ['git archive', 'guarded import-origin audit'],
            'preparation_resume_mode': RESUME,
            'future_execution_executed': False,
            'authorization_gate': 'Root must independently review these bytes and create a distinct release.json with authorized=true, then hash that exact file. Candidate hash cannot be used for authorized release.'}
save(AUDIT / 'commands.json', commands)
verification = {'schema': 'hbe-v3-release-preparation-audit-v1',
    'status': 'prepared_unauthorized_candidate_for_root_review',
    'go_no_go': 'GO for root independent review and possible separate one-attempt release; NO-GO for execution with this unauthorized candidate.',
    'actual_execution_authorized': False, 'candidate': binding(candidate_path),
    'source_commit': commit, 'reviewed_preparation_commit': preparation_commit,
    'predecessor_receipts_commit': receipt_commit,
    'reviewed_preparation': binding(review_path), 'source_archive': binding(archive),
    'archive_verification': binding(AUDIT / 'archive-verification.json'),
    'import_origin_verification': binding(import_stdout),
    'initial_import_guard_stop': binding(AUDIT / 'import-audit.stdout.json') if RESUME else None,
    'prior_import_audits': [binding(p) for p in sorted(AUDIT.glob('import-audit*.stdout.json')) if p != import_stdout],
    'import_guard_adjustment': 'Allow absent .pyc probes only within frozen scripts/__pycache__; no cache bytes exist. Permit finite_array for import-time reference-element constants. Use exact v2 virtualenv invocation, allow its third-party site-packages, and exclude its editable live src path.' if RESUME else None,
    'interpreter_invocation': {'path': str(invocation), 'resolved_path': str(interpreter),
                               'sha256': study['interpreter']['sha256'],
                               'origin': study['csv_header_migration']['previous_supervision']},
    'commands': binding(AUDIT / 'commands.json'), 'protected_input_count': len(protected_bindings),
    'predecessor_metadata': receipts, 'predecessor_state_retained': prior_state,
    'preserved': {'study': candidate['study'], 'interpreter': study['interpreter'],
                 'backend_profile': study['backend_profile'], 'runtime_identity': study['runtime_identity'],
                 'roles': study['roles'], 'calibration_members': permitted,
                 'conditional_held_out_members': held_out, 'budgets': study['budgets'],
                 'torsion_schema_uninspected_and_unchanged': True},
    'checks': {'archive_global_comment_matches_exact_source_commit': True,
               'archive_exact_38_members': True, 'source_imports_all_frozen': True,
               'tracked_worktree_staging_and_head_unchanged': True,
               'existing_inputs_rehashed_unchanged': True, 'v3_output_root_absent': True,
               'authorized_release_absent': True, 'preflight_run': False,
               'measured_archive_opened': False, 'measured_curve_or_torsion_opened': False,
               'fit_calls': 0, 'native_calls': 0, 'mesher_calls': 0,
               'tracked_edits_or_commits': 0, 'synthrad_or_desktop_mutations': 0},
    'limitations': ['This preparation audit does not call v3 preflight or verify all historical native input files.',
                    'No measured calibration, fitting, native confirmation or physical validation occurred.',
                    'The unchanged torsion framing remains uninspected and may later fail closed.',
                    'Read-only sealing uses filesystem modes plus recorded hashes; it is not a tamper-proof filesystem guarantee.'],
    'finished_utc': datetime.now(timezone.utc).isoformat()}
save(AUDIT / 'verification.json', verification)
with (AUDIT / 'RESULT.txt').open('x') as stream:
    stream.write('Prepared exact HBE_01_03 v3 source archive and unauthorized release candidate.\n'
                 'GO for root independent review; NO-GO for execution until a separate root-authorized release exists.\n'
                 '38 files (29 Python sources, 9 declarations), commit ' + commit + '.\n'
                 'Archive SHA256: ' + candidate['source_archive']['sha256'] + '\n'
                 'Candidate SHA256: ' + binding(candidate_path)['sha256'] + '\n'
                 'Guarded imports verified all 29 modules came from frozen source; no live checkout imports.\n'
                 'No preflight, measured archive/member read, fit, native call, output root, or tracked edit.\n')
index = {p.name: {'sha256': digest(p.read_bytes()), 'bytes': p.stat().st_size}
         for p in AUDIT.iterdir() if p.is_file()}
save(AUDIT / 'artifact-index.json', {'schema': 'hbe-v3-release-preparation-artifact-index-v1', 'files': index})
for path in list(FROZEN.rglob('*')) + [FROZEN, archive, candidate_path] + list(AUDIT.iterdir()) + [AUDIT]:
    path.chmod(0o555 if path.is_dir() else 0o444)
print(json.dumps({'candidate': binding(candidate_path), 'source_archive': binding(archive),
                  'audit': binding(AUDIT / 'verification.json'),
                  'artifact_index': binding(AUDIT / 'artifact-index.json'),
                  'status': verification['status']}, indent=2, sort_keys=True))
