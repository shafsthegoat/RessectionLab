"""Independent release audit: source/metadata reads and guarded imports only."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREP = ROOT / 'build/hbe-branch-calibration-v3-prep-review-v1'
FROZEN = ROOT / 'build/hbe-branch-calibration-v3/source'
CANDIDATE = ROOT / 'build/hbe-branch-calibration-v3/release.candidate.json'
COMMIT = 'e96f706ff1857a5791ed69b5ac0d05aff5756e55'
checks, files, blockers = {}, {}, []

def sha(data):
    return hashlib.sha256(data).hexdigest()

def data(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if path.suffix.lower() in {'.csv', '.zip', '.feb', '.log'}:
        raise ValueError('Not a source/metadata audit input: ' + str(path))
    content = path.read_bytes()
    name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    files[name] = {'sha256': sha(content), 'bytes': len(content)}
    return content

def read(path):
    return json.loads(data(path))

def check(name, predicate):
    checks[name] = bool(predicate)
    if not predicate:
        blockers.append(name)

def binding(item):
    content = data(item['path'])
    check('binding:' + item['path'], sha(content) == item['sha256'])
    return json.loads(content)

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def tracked_state():
    return {'head': git('rev-parse', 'HEAD').decode().strip(),
            'unstaged_diff_sha256': sha(git('diff', '--binary')),
            'staged_diff_sha256': sha(git('diff', '--cached', '--binary')),
            'tracked_status': git('status', '--short', '-uno').decode()}

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

start = tracked_state()
candidate = read(CANDIDATE)
study = binding(candidate['study'])
frozen_study = read(FROZEN / candidate['study']['path'])
check('candidate_unauthorized', candidate.get('authorized') is False)
check('authorized_release_absent', not (CANDIDATE.parent / 'release.json').exists())
check('output_root_absent', not (ROOT / study['output_root']).exists())
check('exact_v3_study', candidate['schema'] == 'hbe-branch-calibration-release-v3'
      and study['schema'] == 'hbe-branch-calibration-v3'
      and study['study_id'] == 'hbe-01-03-branch-calibration-v3'
      and frozen_study == study)
check('exact_v3_output_root', candidate['output_root'] == study['output_root']
      == 'outputs/mechanics/hbe-01-03-branch-calibration-v3')
check('candidate_scope', candidate['phase'] == 'calibrate_and_validate'
      and type(candidate['authorized_native_calls']) is int
      and candidate['authorized_native_calls'] == 2)
check('source_commit', candidate['source_commit'] == COMMIT
      == git('rev-parse', COMMIT).decode().strip())

sources = candidate['source_bindings']
expected_names = {k: Path(v['path']).name for k, v in study['inherited_sources'].items()}
expected_names.update(branch_calibration='mechanics_hbe_branch_calibration_v3.py',
                      branch_calibration_readout='mechanics_hbe_branch_calibration_v3_readout.py',
                      branch_calibration_runner='mechanics_hbe_branch_calibration_v3_experiment.py')
check('exact_29_source_keys', sources.keys() == expected_names.keys() and len(sources) == 29)
check('inherited_source_hashes_preserved', all(sources[k]['sha256'] == v['sha256']
      for k, v in study['inherited_sources'].items()))
wanted = {'scripts/' + expected_names[k]: v['sha256'] for k, v in sources.items()}
wanted.update({v['path']: v['sha256'] for v in study['inherited_declarations'] + [candidate['study']]})
archive_bytes = data(candidate['source_archive']['path'])
check('archive_hash', sha(archive_bytes) == candidate['source_archive']['sha256'])
check('exact_38_member_plan', len(wanted) == 38)
member_results = []
with tarfile.open(ROOT / candidate['source_archive']['path']) as bundle:
    members = bundle.getmembers()
    regular = [m for m in members if m.isfile()]
    check('archive_comment_exact_commit', bundle.pax_headers.get('comment') == COMMIT)
    check('archive_exact_regular_members', len(regular) == 38 and {m.name for m in regular} == wanted.keys())
    check('archive_unique_safe_members', len({m.name for m in members}) == len(members)
          and all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                  and (m.isfile() or m.isdir()) for m in members))
    for member in regular:
        archived = bundle.extractfile(member).read()
        extracted = data(FROZEN / member.name)
        committed = git('show', COMMIT + ':' + member.name)
        hashes = {k: sha(v) for k, v in [('archived', archived), ('extracted', extracted), ('committed', committed)]}
        check('member:' + member.name, set(hashes.values()) == {wanted[member.name]})
        check('sealed:' + member.name, (FROZEN / member.name).stat().st_mode & 0o222 == 0
              and not (FROZEN / member.name).is_symlink())
        member_results.append({'path': member.name, 'bytes': len(archived), 'expected_sha256': wanted[member.name], **hashes})
check('exact_extracted_files', {str(p.relative_to(FROZEN)) for p in FROZEN.rglob('*') if p.is_file()} == wanted.keys())
regenerated = git('archive', '--format=tar', COMMIT, '--', *sorted(wanted))
check('byte_identical_git_archive', regenerated == archive_bytes)
for k, v in sources.items():
    check('source_origin:' + k, (ROOT / v['path']).resolve() == FROZEN / 'scripts' / expected_names[k])
core_text = (FROZEN / 'scripts/mechanics_hbe_branch_calibration_v3.py').read_text()
tree = ast.parse(core_text)
constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in {'DECLARATION_PATH', 'DECLARATION_SHA256'}}
check('source_pins_exact_study', candidate['study'] == {'path': constants['DECLARATION_PATH'], 'sha256': constants['DECLARATION_SHA256']})

roles = binding(study['roles'])
check('exact_role_members', candidate['permitted_members'] == [x['path'] for x in roles['calibration']['members']]
      and candidate['conditional_held_out_members'] == [x['path'] for x in roles['held_out_validation']['members']])
check('exact_execution_schema_and_ledger', candidate['execution'] == {
      'csv_schemas': study['csv_schemas'], 'access_ledger_path': study['output_root'] + '/experiment/access.jsonl'})
check('interpreter_binding_exact', candidate['interpreter'] == study['interpreter'])
interpreter = Path(study['interpreter']['path'])
check('interpreter_hash', sha(data(interpreter)) == study['interpreter']['sha256'])
backend = binding(candidate['backend_profile'])
runtime_identity = binding(study['runtime_identity'])
check('backend_binding_exact', candidate['backend_profile'] == study['backend_profile']
      and backend['runtime_identity'] == study['runtime_identity'] and backend['profile_id'] == 'accelerate_csc_v1')

migration = study['csv_header_migration']
records = {k: binding(v) for k, v in migration.items() if isinstance(v, dict) and set(v) == {'path', 'sha256'}}
previous = records['previous_study']
prior_migration = previous['runtime_migration']
v1_records = {k: binding(v) for k, v in prior_migration.items() if isinstance(v, dict) and set(v) == {'path', 'sha256'}}
previous_fixed = {k: v for k, v in previous.items() if k not in {'schema', 'study_id', 'output_root', 'csv_schemas'}}
study_fixed = {k: v for k, v in study.items() if k not in {'schema', 'study_id', 'output_root', 'csv_schemas', 'csv_header_migration'}}
schemas = deepcopy(previous['csv_schemas'])
for branch in ('compression', 'tension'):
    check('predecessor_headerless:' + branch, schemas[branch]['header'] is None)
    schemas[branch]['header'] = ['displacement', 'force']
check('only_declared_v3_changes', previous_fixed == study_fixed and study['csv_schemas'] == schemas)
check('v1_v2_lineage_preserved', study['runtime_migration'] == previous['runtime_migration']
      and previous['runtime_migration']['automatic_retry'] is False)
check('candidate_v2_partial_failure_binding', candidate['predecessor_failure'] == migration['previous_state'])
state = records['previous_state']
check('v2_failure_honest_partial_access', state['status'] == 'failed_or_incomplete'
      and state['calibration_access_attempted'] is True and state['calibration_responses_accessed'] is None
      and state['error'] == {'type': 'ValueError', 'message': "could not convert string to float: 'displacement'"}
      and state['held_out_access_attempted'] is False and state['held_out_responses_accessed'] is False
      and state['native_calls'] == state['mesher_calls'] == 0 and state['runs'] == {}
      and state['automatic_retry'] is False and state['physical_validation_pass'] is None
      and not {'fit', 'freeze', 'predictions', 'held_out_metrics'} & state.keys())
check('v2_terminal_bindings', records['previous_result']['state'] == migration['previous_state']
      and records['previous_result']['release'] == migration['previous_release']
      and records['previous_result']['supervision'] == records['previous_supervision']
      and records['previous_publication']['result'] == migration['previous_result']
      and records['previous_publication']['accepted'] is False
      and records['previous_started'] == {'release': migration['previous_release'], 'no_retry': True})
diagnosis = records['diagnosis']
check('axial_observed_headers_only', diagnosis['frozen_v2_state'] == state
      and diagnosis['torsion_member_content_opened'] is False and migration['torsion_header_inspected'] is False
      and diagnosis['fitted'] is False and diagnosis['native_calls'] == 0
      and diagnosis['opened_member_names'] == candidate['permitted_members']
      and diagnosis['access_ledger_records'] == [records['previous_access_ledger']]
      and all(e['branch'] == b and e['member'] == m['path'] and e['bytes'] == m['bytes']
              and e['crc32'] == m['crc32'] and e['raw_first_line'] == 'displacement,force\n'
              and e['header_fields'] == ['displacement', 'force'] and e['header_field_count'] == 2
              and e['utf8_bom_present'] is False and e['diagnostic_used_numeric_values'] is False
              and e['v2_declared_schema'] == previous['csv_schemas'][b]
              for b, m, e in zip(('compression', 'tension'), roles['calibration']['members'], diagnosis['members'], strict=True)))
exposure = records['failure_review']['exposure']
check('honest_partial_exposure_lineage', exposure['compression']['full_payload_read_and_csv_rows_decoded'] is True
      and exposure['compression']['numeric_curve_parse_completed'] is False
      and exposure['tension']['uncompressed_member_payload_read'] is False
      and exposure['held_out_torsion']['remained_sealed'] is True
      and exposure['held_out_torsion']['uncompressed_response_member_reads'] == 0)
check('heldout_schema_unchanged', all(study['csv_schemas'][b] == previous['csv_schemas'][b]
      and study['csv_schemas'][b]['header'] is None for b in ('torsion_neg', 'torsion_pos')))
caps = study['budgets']
check('resource_caps_exact_and_unchanged', caps == previous['budgets']
      and caps['aggregate_seconds'] == 3600 and caps['sampled_family_RSS_bytes'] == 3 * 1024**3
      and caps['new_total_output_bytes'] == 2 * 1024**3 and caps['maximum_native_calls'] == 2
      and caps['maximum_mesher_calls'] == 0 and caps['runtime_threads'] == 1
      and caps['native_seconds_by_branch'] == {'compression': 2100, 'tension': 420}
      and caps['combined_deck_preparation_seconds'] == 60 and caps['paired_raw_streaming_seconds'] == 600
      and caps['automatic_retry'] is False and caps['same_disk_new_source_archive_in_total'] is True)

prep_index = read(PREP / 'artifact-index.json')
for name, entry in prep_index['files'].items():
    content = data(PREP / name)
    check('prep_receipt:' + name, len(content) == entry['bytes'] and sha(content) == entry['sha256'])
commands = read(PREP / 'commands.json')
invocation = records['previous_supervision']['command'][0]
check('exact_predecessor_venv_invocation', invocation == str(ROOT / '.venv/bin/python')
      and Path(invocation).resolve() == interpreter.resolve())
check('future_command_frozen_path', commands['future_execution_argv_template'] == [invocation, '-I', '-B',
      str(FROZEN / 'scripts/mechanics_hbe_branch_calibration_v3_experiment.py'), '--root', str(ROOT),
      '--release', 'build/hbe-branch-calibration-v3/release.json', '--release-sha256',
      '<ROOT_AUTHORIZED_RELEASE_SHA256>', '--execute']
      and commands['future_execution_cwd'] == str(FROZEN) and commands['future_execution_executed'] is False)
guard = data(PREP / 'guarded_imports.py')
(OUT / 'guarded_imports.py').write_bytes(guard)
argv = [invocation, '-I', '-B', str(OUT / 'guarded_imports.py'), str(ROOT), str(FROZEN), json.dumps(sources)]
proc = subprocess.run(argv, cwd=FROZEN, capture_output=True, timeout=45)
(OUT / 'import-audit.stdout.json').write_bytes(proc.stdout)
(OUT / 'import-audit.stderr.txt').write_bytes(proc.stderr)
imports = json.loads(proc.stdout)
check('independent_guarded_import_exit_zero', proc.returncode == 0 and proc.stderr == b'')
check('independent_import_origins_exact', imports['status'] == 'passed' and imports['violations'] == []
      and imports['module_count'] == 29 and imports['imported_modules'] == {
          'scripts.' + Path(v['path']).stem: v for v in sources.values()}
      and imports['scripts_namespace_paths'] == [str(FROZEN / 'scripts')]
      and imports['interpreter'] == str(interpreter.resolve()) and imports['invocation_path'] == invocation
      and not any(imports[k] for k in ['preflight_called', 'measured_archive_opened', 'measured_or_torsion_member_opened', 'fit_called', 'native_called']))
check('no_live_source_import_path', str(ROOT) not in imports['sys_path']
      and str(ROOT / 'src') not in imports['sys_path']
      and all(p == str(FROZEN) or not Path(p).is_relative_to(ROOT)
              or p == str(ROOT / '.venv/lib/python3.12/site-packages') for p in imports['sys_path']))
startup_probes = {}
for mode in ('launcher', 'worker'):
    # Preserve site-added editable paths to check actual startup resolution rather
    # than relying only on the preparation helper's filtered sys.path.
    actual_guard = guard.decode().replace(
        "sys.path[:] = [str(frozen)] + [p for p in sys.path if p and (\n    not Path(p).resolve().is_relative_to(root) or Path(p).resolve() == site_packages)]",
        "sys.path[:] = [str(frozen)] + " + ("[]" if mode == 'launcher' else "[str(frozen / 'scripts')]")
        + " + [p for p in sys.path if p and Path(p).resolve() != Path(__file__).resolve().parent]")
    actual_guard = actual_guard.replace("    for item in expected.values():\n        importlib.import_module",
        "    importlib.import_module('scripts.mechanics_hbe_branch_calibration_v3_experiment')\n    for item in expected.values():\n        importlib.import_module")
    probe = OUT / ('guarded_' + mode + '_imports.py')
    probe.write_text(actual_guard)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('PYTHON') or key in ('__PYVENV_LAUNCHER__', 'LD_PRELOAD', 'LD_LIBRARY_PATH'):
            env.pop(key)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    startup_argv = [invocation] + (['-I'] if mode == 'launcher' else []) + ['-B', str(probe), str(ROOT), str(FROZEN), json.dumps(sources)]
    probe_result = subprocess.run(startup_argv, cwd=FROZEN if mode == 'launcher' else ROOT,
                                  env=env, capture_output=True, timeout=45)
    (OUT / (mode + '-import-audit.stdout.json')).write_bytes(probe_result.stdout)
    (OUT / (mode + '-import-audit.stderr.txt')).write_bytes(probe_result.stderr)
    probe_imports = json.loads(probe_result.stdout)
    check(mode + '_startup_import_origins_exact', probe_result.returncode == 0
          and probe_imports['status'] == 'passed' and probe_imports['violations'] == []
          and probe_imports['imported_modules'] == imports['imported_modules']
          and probe_imports['scripts_namespace_paths'] == [str(FROZEN / 'scripts')])
    startup_probes[mode] = {'argv': startup_argv, 'sys_path': probe_imports.get('sys_path'),
                            'status': probe_imports['status'], 'returncode': probe_result.returncode}
end = tracked_state()
check('tracked_state_unchanged', start == end)
check('still_unauthorized_and_unstarted', read(CANDIDATE)['authorized'] is False
      and not (CANDIDATE.parent / 'release.json').exists() and not (ROOT / study['output_root']).exists())
check('all_source_metadata_inputs_unchanged', all(sha(Path(p).read_bytes() if Path(p).is_absolute()
      else (ROOT / p).read_bytes()) == v['sha256'] for p, v in list(files.items())))
save('archive-members.json', member_results)
save('input-inventory.json', files)
save('commands.json', {'import_audit_argv': argv, 'cwd': str(FROZEN), 'returncode': proc.returncode,
     'startup_import_probes': startup_probes,
     'audit_interpreter_note': 'Initial host python3 was too old for zip(strict=True); it stopped before import probes or report creation. The complete audit ran using the pinned .venv interpreter with -I -B.',
     'future_execution_argv_template_reviewed_but_not_executed': commands['future_execution_argv_template']})
report = {
    'schema': 'hbe-branch-calibration-v3-independent-release-audit-v1',
    'finished_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'go_for_root_authorization_only' if not blockers else 'no_go',
    'execution_authorized': False,
    'go_no_go': 'GO for root to consider a distinct, hash-bound, one-attempt authorized release; NO-GO for execution using the unauthorized candidate.' if not blockers else 'NO-GO: resolve listed blockers before authorization.',
    'candidate': {'path': str(CANDIDATE.relative_to(ROOT)), **files[str(CANDIDATE.relative_to(ROOT))]},
    'source_archive': candidate['source_archive'], 'source_commit': COMMIT,
    'study': candidate['study'], 'checks': checks, 'blocking_findings': blockers,
    'archive': {'regular_members': 38, 'python_sources': 29, 'declarations': 9,
                'bytes': len(archive_bytes), 'git_archive_byte_identical': checks['byte_identical_git_archive']},
    'interpreter_invocation': invocation, 'resource_caps': caps,
    'retained_v2_state': state, 'retained_v2_exposure': exposure,
    'observed_axial_headers_from_pinned_diagnostic': {e['branch']: e['raw_first_line'] for e in diagnosis['members']},
    'static_source_review': {
        'authorization': 'launch rejects authorized!=true before output creation; preflight rejects candidate and authenticates exact v3 bindings.',
        'import_origin': 'Runner inserts its frozen source parent; preflight compares every loaded module origin to release path. Guarded imports plus launcher/worker startup probes that retained editable project paths resolved all 29 sources to the frozen tree. Use the exact reviewed venv/frozen-runner command.',
        'heldout': 'Axial fit, predictions, two successful fitted confirmations, independent paired evidence, exact input rehash and durable freeze precede the only held-out read.',
        'caps': 'One 3600-second deadline; 3 GiB sampled process-group RSS; 2 GiB new output including archive; two native calls at most (2100/420 seconds), 60-second preparation, 600-second paired readout, one thread, no mesher, no retry.',
        'failure': 'Failure remains failed_or_incomplete with partial calibration access null, false held-out flags, zero native/mesher calls and no physical validation claim.',
    },
    'actions_not_taken': ['no measured archive/member read', 'no measured CSV parsing', 'no fit', 'no native execution',
                          'no mesher', 'no preflight', 'no authorization', 'no output-root creation', 'no tracked edit'],
    'limitations': ['Historical native primitives and backend executable/libraries were not rehashed; these remain responsibilities of the future authorized preflight.',
                    'Torsion header remains uninspected/headerless and may later fail closed.',
                    'Startup probes import the actual runner under a non-main module name and perform no entrypoint function, preflight, or scientific work.',
                    'Filesystem read-only modes and hashes are integrity evidence, not tamper-proof storage.',
                    'Resource controls are sampled, with no hard-real-time guarantee.'],
    'tracked_state_before': start, 'tracked_state_after': end,
}
save('verification.json', report)
(OUT / 'RESULT.txt').write_text(report['go_no_go'] + '\n'
    + 'Source commit: ' + COMMIT + '\nArchive SHA256: ' + candidate['source_archive']['sha256']
    + '\nCandidate SHA256: ' + files[str(CANDIDATE.relative_to(ROOT))]['sha256']
    + '\nBlocking findings: ' + str(blockers) + '\n')
save('artifact-index.json', {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
     for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'artifact-index.json'})
print(json.dumps({'status': report['status'], 'blockers': blockers, 'checks': len(checks),
                 'verification_sha256': sha((OUT / 'verification.json').read_bytes()),
                 'candidate_sha256': report['candidate']['sha256'], 'archive_sha256': candidate['source_archive']['sha256']}, indent=2))
