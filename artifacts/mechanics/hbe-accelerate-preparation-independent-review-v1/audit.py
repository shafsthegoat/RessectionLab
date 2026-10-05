"""Saved preparation audit only: stdlib, no solver imports or measurement ZIP reads."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
BASE = 'artifacts/mechanics/hbe-accelerate-experiment-preparation-v1/'
COMMIT = '6486dbcfb0395a4c5b9df3f140b08141d0d96fd7'
seen = {}


def checked(path, expected=None):
    path = (ROOT / path).resolve()
    assert path.is_relative_to(ROOT)
    assert path.suffix not in ('.csv', '.zip', '.nii', '.gz')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    if str(path) in seen:
        assert seen[str(path)] == digest, str(path)
    seen[str(path)] = digest
    return path


def read(path):
    return json.loads(checked(path).read_bytes())


def bound(record, *, parse=False):
    path = checked(record['path'], record['sha256'])
    if 'bytes' in record:
        assert path.stat().st_size == record['bytes']
    return json.loads(path.read_bytes()) if parse else path


start = time.monotonic()
archive = read(BASE + 'source-archive.json')
draft = read(BASE + 'release-draft.json')
refusal = read(BASE + 'draft-refusal.json')
preparation = read(BASE + 'preparation-result.json')
profile_check = read(BASE + 'profile-verification.json')
profile = bound(preparation['profile'], parse=True)
originals = bound(preparation['original_cases'], parse=True)
manifest = bound(preparation['backend_decks'], parse=True)
raw_index = read(BASE + 'raw-input-index.json')
assert archive['source_commit'] == draft['source_commit'] == COMMIT
assert archive['file_count'] == len(archive['members']) == 14
source_tar = bound(archive['source_archive'])
assert source_tar.stat().st_size == 317440
assert draft['source_archive_sha256'] == archive['source_archive']['sha256']
assert draft['execution']['source_archive']['sha256'] == archive['source_archive']['sha256']
with tarfile.open(source_tar, 'r:') as bundle:
    files = [member for member in bundle if member.isfile()]
    assert len(files) == len({member.name for member in files}) == 14
    assert {member.name for member in files} == set(archive['members'])
    for member in files:
        data = bundle.extractfile(member).read()
        expected = archive['members'][member.name]
        assert len(data) == expected['bytes']
        assert hashlib.sha256(data).hexdigest() == expected['sha256']
        committed = subprocess.run(['git', '-C', str(ROOT), 'show', COMMIT + ':' + member.name],
                                   capture_output=True, check=True, timeout=10).stdout
        assert data == committed
        frozen = checked(archive['frozen_source_root'] + '/' + member.name, expected['sha256'])
        assert frozen.stat().st_mode & 0o222 == 0
source = checked(archive['frozen_source_root'] + '/scripts/mechanics_hbe_backend.py')
constants = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(source.read_bytes()).body
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in ('SKYLINE_XML', 'ACCELERATE_XML')}
assert constants['ACCELERATE_XML'] == ('<linear_solver type="accelerate"><iterative>0</iterative>'
    '<factorization>4</factorization><order_method>0</order_method>'
    '<print_condition_number>0</print_condition_number></linear_solver>')
expected_cases = {f'{mode}:N{n}:S60:reference' for mode in ('compression', 'tension', 'torsion_neg', 'torsion_pos') for n in (4, 8, 12)}
expected_cases |= {f'{mode}:N12:S120:reference' for mode in ('compression', 'tension', 'torsion_neg', 'torsion_pos')}
expected_cases |= {f'{mode}:N12:S120:double_mu' for mode in ('compression', 'torsion_pos')}
assert set(originals) == set(manifest['cases']) == expected_cases
assert preparation['prepared_cases'] == 18
assert manifest['transform_source_sha256'] == seen[str(source)]
for name, original in originals.items():
    row = manifest['cases'][name]
    assert row['original'] == original
    adapted = row['adapted']
    assert adapted['mesh'] == original['mesh']
    bound(original['mesh'])
    old = bound(original['deck']).read_bytes()
    new = bound(adapted['deck']).read_bytes()
    assert old.count(constants['SKYLINE_XML'].encode()) == 1
    assert new == old.replace(constants['SKYLINE_XML'].encode(), constants['ACCELERATE_XML'].encode(), 1)
    assert len(ET.fromstring(old).findall('.//linear_solver')) == 1
    assert len(ET.fromstring(new).findall('.//linear_solver')) == 1
    old_loading = bound(original['loading'], parse=True)
    new_loading = bound(adapted['loading'], parse=True)
    assert old_loading['deck_sha256'] == original['deck']['sha256']
    old_loading['deck_sha256'] = adapted['deck']['sha256']
    assert old_loading == new_loading
    assert preparation['preserved_original_deck_bindings'][name] == original['deck']
for record in raw_index['files'].values():
    # Index paths are keys; values contain byte counts and hashes.
    assert set(record) >= {'bytes', 'sha256'}
for name, record in raw_index['files'].items():
    path = bound({'path': str(Path(raw_index['root']) / name), **record})
    assert path.stat().st_mode & 0o222 == 0
assert raw_index['file_count'] == len(raw_index['files']) == 37
for inventory in (profile_check['inputs'], preparation['prepared_inputs']):
    for name, digest in inventory.items():
        checked(name, digest)
runtime = bound(profile['runtime_identity'], parse=True)
assert runtime['status'] == 'isolated_patched_runtime_built_pending_numerical_controls'
assert runtime['patched_source_sha256'] == '476ac8471ea681a99c298352a50aba2a7a5baa71635e1678155a7f58b72ba04a'
assert profile_check['profile'] == preparation['profile'] == draft['execution']['backend_profile']
assert draft['execution']['backend_decks'] == preparation['backend_decks']
assert draft['prerequisite_evidence']['runtime'] == profile['runtime_identity']
assert draft['prerequisite_evidence']['analytic_verification'] == profile['hex8_controls']
hx = bound(profile['hex8_controls'], parse=True)
tet = bound(profile['tet10_mpc_controls'], parse=True)
assert hx['evidence'] == tet['evidence']
assert hx['solver_invocations'] == 5 and tet['solver_invocations'] == 3
assert hx['runtime_identity_sha256'] == tet['runtime_identity_sha256'] == profile['runtime_identity']['sha256']
control_review = bound(profile_check['independent_control_review'], parse=True)
assert control_review['status'] == 'eight_saved_numerical_controls_independently_passed'
assert control_review['runtime_identity_sha256'] == profile['runtime_identity']['sha256']
protocol = read(archive['frozen_source_root'] + '/manifests/experiments/hbe-01-03-mechanics-poc-v1.json')
roles = read(archive['frozen_source_root'] + '/manifests/experiments/hbe-01-03-specimen-roles-v1.json')
assert draft['execution']['caps'] == protocol['budgets']
assert sum(draft['execution']['caps']['call_allocation'].values()) == draft['execution']['authorized_specimen_solver_calls'] == 20
assert draft['permitted_members'] == [row['path'] for row in roles['calibration']['members']]
assert draft['archive_sha256'] == roles['source']['archive_sha256']
for mode, schema in draft['execution']['csv_schemas'].items():
    assert schema == {'coordinate_column': 0, 'response_column': 1, 'delimiter': ',', 'header': None,
        'coordinate_unit': 'rad' if mode.startswith('torsion') else 'm',
        'response_unit': 'Nm' if mode.startswith('torsion') else 'N'}
assert draft['execution']['csv_schema_provenance']['status'] == 'prospectively_fixed_unverified_format_assumption'
assert 'continuation' not in draft['execution']
assert draft['authorized'] is False and draft['schema'] == 'hbe-calibration-release-draft-v1'
assert refusal['status'] == 'draft_rejected_before_execution'
assert refusal['exception'] == 'Explicit committed calibration release required'
assert refusal['experiment_directory_created'] is False
bound(refusal['draft'])
bound(refusal['executing_orchestrator'])
assert refusal['executing_orchestrator'] == draft['execution']['source_bindings']['orchestrator']
assert len(draft['execution']['source_bindings']) == 9
for record in draft['execution']['source_bindings'].values():
    bound(record)
    relative = str(Path(record['path']).relative_to(archive['frozen_source_root']))
    assert record['sha256'] == archive['members'][relative]['sha256']
for record in draft['execution']['independent_evidence'].values():
    bound(record)
experiment = ROOT / 'outputs/mechanics/hbe-01-03-poc-v1/experiment-accelerate-csc-v1'
marker = ROOT / 'outputs/mechanics/hbe-01-03-poc-v1/.specimen-accelerate-csc-v1-started.json'
assert not experiment.exists() and not marker.exists()
assert preparation['solver_invocations'] == profile_check['solver_invocations'] == refusal['solver_invocations'] == 0
for name, digest in tuple(seen.items()):
    checked(name, digest)
record = {
    'status': 'saved_preparation_consistency_review_passed',
    'scope': 'Read-only hashes, source archive and XML/JSON comparisons. No production code imports, solver, physics replay, CSV/ZIP/patient reads or broad tests.',
    'source_commit': COMMIT, 'source_archive': archive['source_archive'],
    'archive_files_exact_to_commit': 14, 'solver_only_deck_copies_verified': 18,
    'raw_input_files_verified': 37, 'raw_input_bytes': sum(row['bytes'] for row in raw_index['files'].values()),
    'unique_input_files_rehashed_before_and_after': len(seen), 'all_inputs_unchanged': True,
    'runtime_identity': profile['runtime_identity'], 'profile': preparation['profile'],
    'backend_decks': preparation['backend_decks'], 'draft': refusal['draft'],
    'control_review': profile_check['independent_control_review'],
    'caps_identical_to_frozen_protocol': True, 'original_csv_assumptions_unchanged': True,
    'calibration_members_only': True, 'continuation_absent': True,
    'draft_expected_refusal_receipt_and_source_bound': True,
    'draft_refusal_not_reexecuted_by_this_audit': True,
    'experiment_directory_and_marker_absent': True, 'new_solver_calls': 0, 'measured_members_read': 0,
    'execution_authorized_by_this_review': False,
    'limits': ['Saved profile replay and eight-control interpretation are bound to the independent upstream review, not reexecuted here.',
               'Unsupported draft refusal is an early gate, not evidence that every later supported-release preflight condition has been executed.',
               'No specimen accuracy or constitutive-law validation is established by preparation.'],
    'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'elapsed_seconds': time.monotonic() - start,
    'verified_inputs_sha256': hashlib.sha256(json.dumps(seen, sort_keys=True).encode()).hexdigest(),
}
Path(__file__).with_name('review.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
