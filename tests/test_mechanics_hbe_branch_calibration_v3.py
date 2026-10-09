"""Pinned real axial headers with synthetic values; no measured or native execution."""
from contextlib import nullcontext
from copy import deepcopy
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import zipfile

import pytest

from scripts import mechanics_hbe_branch_calibration_v2 as previous_core
from scripts import mechanics_hbe_branch_calibration_v3 as core
from scripts import mechanics_hbe_branch_calibration_v3_readout as readout
from scripts import mechanics_hbe_branch_calibration_v3_experiment as runner

ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT/core.DECLARATION_PATH).read_text())
BINDINGS = ('previous_study', 'previous_release', 'previous_state', 'previous_result',
            'previous_supervision', 'previous_publication', 'previous_started',
            'previous_access_ledger', 'diagnosis', 'failure_review')
FIXED_FIELDS = sorted(set(STUDY) - {'schema', 'study_id', 'output_root', 'csv_schemas',
                                   'csv_header_migration'})


@pytest.fixture(autouse=True)
def no_measured_or_native_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Measured fit, solver, qualification and deck execution forbidden')
    for module, name in [(subprocess, 'Popen'), (core.runtime, 'supervise'), (core.old, 'solve'),
                         (core, 'qualify'), (core.evaluation, 'fit_scale'), (runner, 'prepare'),
                         (core.common, 'pure_child'), (readout, 'stream_frames')]:
        monkeypatch.setattr(module, name, forbidden)
    original_open = io.open
    native_paths = {ROOT/binding['path'] for q in STUDY['axial_references'].values()
                    for binding in q['native_primitives'].values()}
    native_paths.update(ROOT/binding['path'] for q in STUDY['torsion_references'].values()
                        for binding in q['primitives'].values())

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            path = Path(file).resolve()
            if path.is_relative_to(ROOT/'data/mechanics') or path in native_paths:
                raise AssertionError('Actual measured archive/native primitive access forbidden')
        return original_open(file, *args, **kwargs)
    monkeypatch.setattr(io, 'open', guarded_open)


@pytest.fixture
def study():
    return deepcopy(STUDY)


@pytest.fixture
def records(study):
    return {key: json.loads((ROOT/study['csv_header_migration'][key]['path']).read_text())
            for key in BINDINGS}


class MetadataRegistry:
    """Saved metadata only, including the exact preserved v1 runtime failure chain."""
    def __init__(self, study, records):
        self.records = {study['csv_header_migration'][key]['path']: value for key, value in records.items()}
        for binding in study['runtime_migration'].values():
            if isinstance(binding, dict) and set(binding) == {'path', 'sha256'}:
                self.records[binding['path']] = json.loads((ROOT/binding['path']).read_text())
        self.records[study['roles']['path']] = json.loads((ROOT/study['roles']['path']).read_text())
        self.calls = []

    def bound(self, binding, *, json_value=True):
        self.calls.append(binding['path'])
        return deepcopy(self.records[binding['path']])

    def add(self, binding):
        pass


def test_exact_manifest_two_header_changes_and_all_scientific_fields_preserved(study, records):
    assert hashlib.sha256((ROOT/core.DECLARATION_PATH).read_bytes()).hexdigest() == core.DECLARATION_SHA256
    assert core.DECLARATION_PATH != previous_core.DECLARATION_PATH
    assert len(FIXED_FIELDS) == 24  # 23 scientific/historical fields plus unchanged v1 runtime lineage.
    for key in FIXED_FIELDS:
        assert study[key] == records['previous_study'][key]
    expected = deepcopy(records['previous_study']['csv_schemas'])
    for branch in core.AXIAL:
        assert expected[branch]['header'] is None
        expected[branch]['header'] = ['displacement', 'force']
    assert study['csv_schemas'] == expected
    assert len(study['inherited_sources']) == 26
    assert len(study['inherited_declarations']) == 8
    for binding in list(study['inherited_sources'].values()) + study['inherited_declarations']:
        assert hashlib.sha256((ROOT/binding['path']).read_bytes()).hexdigest() == binding['sha256']


def test_both_actual_failure_chains_authenticated_without_member_reads(study):
    registry = core.Registry(ROOT)
    checked = core.verify_csv_header_migration(study, registry)
    assert set(checked) == set(BINDINGS)
    expected = {study['csv_header_migration'][key]['path'] for key in BINDINGS}
    expected.update(binding['path'] for binding in study['runtime_migration'].values()
                    if isinstance(binding, dict))
    expected.add(study['roles']['path'])
    assert set(registry.entries) == expected
    assert checked['previous_state']['calibration_responses_accessed'] is None
    assert checked['failure_review']['exposure']['compression']['uncompressed_bytes_read'] == 1222
    assert checked['failure_review']['exposure']['tension']['uncompressed_member_payload_read'] is False
    assert checked['previous_state']['native_calls'] == 0


@pytest.mark.parametrize('field', FIXED_FIELDS)
def test_every_fixed_field_refuses_header_excuse(study, records, field):
    study[field] = {'unexpected_header_excuse': True}
    # Build the metadata fixture from the real declaration, not the poisoned field.
    with pytest.raises(ValueError, match='scientific and historical'):
        core.verify_csv_header_migration(study, MetadataRegistry(STUDY, records))


@pytest.mark.parametrize('branch,field,value', [
    ('compression', 'header', None), ('tension', 'header', ['force', 'displacement']),
    ('compression', 'coordinate_column', 1), ('tension', 'response_column', 0),
    ('compression', 'coordinate_unit', 'mm'), ('tension', 'response_unit', 'kN'),
    ('compression', 'delimiter', ';'), ('torsion_neg', 'header', ['guessed', 'header']),
    ('torsion_pos', 'header', ['guessed', 'header']),
])
def test_only_two_exact_header_edits_are_allowed(study, records, branch, field, value):
    study['csv_schemas'][branch][field] = value
    with pytest.raises(ValueError, match='except two axial headers'):
        core.verify_csv_header_migration(study, MetadataRegistry(STUDY, records))


@pytest.mark.parametrize('field,value', [
    ('calibration_access_attempted', False), ('calibration_responses_accessed', False),
    ('calibration_responses_accessed', True), ('held_out_access_attempted', True),
    ('held_out_responses_accessed', 0), ('automatic_retry', True), ('native_calls', 1),
    ('native_calls', False), ('mesher_calls', 1), ('runs', {'compression': {}}),
    ('physical_validation_pass', True), ('fit', {}), ('predictions', {}), ('freeze', {}),
    ('held_out_metrics', {}), ('calibration_metrics', {}), ('preparation_seconds', 1.),
    ('status', 'completed'), ('error', {'type': 'ValueError', 'message': 'different failure'}),
])
def test_prior_partial_access_cannot_be_relabelled(study, records, field, value):
    records['previous_state'][field] = value
    with pytest.raises(ValueError, match='partial axial access'):
        core.verify_csv_header_migration(study, MetadataRegistry(study, records))


@pytest.mark.parametrize('record,field,value', [
    ('previous_release', 'authorized', False), ('previous_release', 'authorized_native_calls', True),
    ('previous_release', 'study', {}), ('previous_release', 'interpreter', {}),
    ('previous_release', 'predecessor_failure', {}), ('previous_release', 'execution', {}),
    ('previous_release', 'permitted_members', []), ('previous_release', 'conditional_held_out_members', []),
    ('previous_result', 'release', {}), ('previous_result', 'state', {}),
    ('previous_result', 'supervision', {}), ('previous_result', 'physical_validation_pass', True),
    ('previous_supervision', 'no_retry', False), ('previous_supervision', 'exit_code', True),
    ('previous_supervision', 'cleanup_error', 'orphan'), ('previous_supervision', 'kill_reason', 'timeout'),
    ('previous_publication', 'accepted', True), ('previous_publication', 'result', {}),
    ('previous_started', 'no_retry', False), ('previous_started', 'no_retry', 1),
    ('previous_access_ledger', 'phase', 'calibration_completed'),
    ('previous_access_ledger', 'sequence', False), ('previous_access_ledger', 'members', []),
    ('diagnosis', 'fitted', True), ('diagnosis', 'torsion_member_content_opened', True),
    ('diagnosis', 'native_calls', True), ('diagnosis', 'frozen_v2_state', {}),
    ('diagnosis', 'access_ledger_records', []), ('diagnosis', 'opened_member_names', []),
    ('diagnosis', 'members', []), ('diagnosis', 'archive', {}),
    ('failure_review', 'failure', {}), ('failure_review', 'exposure', {}),
    ('failure_review', 'output_bindings', {}), ('failure_review', 'status', 'completed'),
])
def test_predecessor_receipt_joins_fail_closed(study, records, record, field, value):
    records[record][field] = value
    with pytest.raises(ValueError):
        core.verify_csv_header_migration(study, MetadataRegistry(study, records))


@pytest.mark.parametrize('branch_index', [0, 1])
@pytest.mark.parametrize('field,value', [
    ('raw_first_line', 'guessed,header\n'), ('header_fields', ['force', 'displacement']),
    ('member', 'unrelated.csv'), ('bytes', 1), ('crc32', '00000000'), ('sha256', 'not-a-digest'),
    ('diagnostic_used_numeric_values', True), ('utf8_bom_present', True),
    ('v2_declared_schema', {}), ('record_widths', [3]),
])
def test_literal_header_evidence_cannot_drift(study, records, branch_index, field, value):
    records['diagnosis']['members'][branch_index][field] = value
    with pytest.raises(ValueError, match='literal axial header'):
        core.verify_csv_header_migration(study, MetadataRegistry(study, records))


@pytest.mark.parametrize('key', BINDINGS)
def test_missing_predecessor_record_refused_before_io(study, key):
    del study['csv_header_migration'][key]
    with pytest.raises(ValueError, match='migration declaration'):
        core.verify_csv_header_migration(study, SimpleNamespace(bound=lambda b: pytest.fail('unexpected read')))


@pytest.mark.parametrize('field,value', [
    ('automatic_retry', True), ('scientific_fields_except_axial_headers_unchanged', False),
    ('reason', 'retry_for_better_result'), ('torsion_header_inspected', True),
    ('payload_accounting', 'both axial members fully parsed'),
])
def test_migration_authority_and_accounting_remain_explicit(study, records, field, value):
    study['csv_header_migration'][field] = value
    with pytest.raises(ValueError, match='migration declaration'):
        core.verify_csv_header_migration(study, MetadataRegistry(study, records))


def test_predecessor_metadata_is_rechecked_after_preflight(tmp_path, study):
    registry = core.Registry(ROOT)
    core.verify_csv_header_migration(study, registry)
    for binding in registry.entries.values():
        target = tmp_path/binding['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/binding['path']).read_bytes())
    copied = core.Registry(tmp_path)
    core.verify_csv_header_migration(study, copied)
    (tmp_path/study['csv_header_migration']['previous_state']['path']).write_text('{}\n')
    with pytest.raises(ValueError, match='hash'):
        copied.verify_all(time.monotonic()+5)


def release_for(study):
    return {'schema': 'hbe-branch-calibration-release-v3', 'authorized': True,
            'phase': 'calibrate_and_validate', 'study': {'path': core.DECLARATION_PATH,
            'sha256': core.DECLARATION_SHA256}, 'authorized_native_calls': 2,
            'backend_profile': study['backend_profile'], 'interpreter': study['interpreter'],
            'predecessor_failure': study['csv_header_migration']['previous_state']}


def preflight_fixture(monkeypatch, study, records, release):
    registry = MetadataRegistry(study, records)
    registry.records['new-release.json'] = release
    monkeypatch.setattr(core, 'Registry', lambda root: registry)
    monkeypatch.setattr(core, 'declaration', lambda root: study)
    monkeypatch.setattr(core.backend, 'verify_profile', lambda *a: pytest.fail('backend reached before migration'))
    return {'path': 'new-release.json', 'sha256': 'a'*64}


def test_bad_predecessor_stops_preflight_before_backend_or_data(monkeypatch, study, records):
    records['previous_state']['native_calls'] = 1
    binding = preflight_fixture(monkeypatch, study, records, release_for(study))
    with pytest.raises(ValueError, match='partial axial access'):
        core.preflight(ROOT, binding)


def test_new_release_requires_exact_predecessor_identity(monkeypatch, study, records):
    release = release_for(study)
    release['predecessor_failure'] = study['runtime_migration']['previous_state']
    binding = preflight_fixture(monkeypatch, study, records, release)
    with pytest.raises(ValueError, match='Exact committed'):
        core.preflight(ROOT, binding)


def test_v2_release_cannot_authorize_v3(monkeypatch, study, records):
    binding = preflight_fixture(monkeypatch, study, records, records['previous_release'])
    with pytest.raises(ValueError, match='Exact committed'):
        core.preflight(ROOT, binding)


def test_replaced_interpreter_refused_before_backend_or_data(monkeypatch, study, records):
    release = release_for(study)
    sources = deepcopy(study['inherited_sources'])
    sources.update({key: {'path': 'scripts/'+name, 'sha256': 'c'*64}
                    for key, name in core.NEW_SOURCES.items()})
    release.update(source_bindings=sources, source_archive={'path': 'new-source.tar', 'sha256': 'd'*64},
                   source_commit='e'*40)
    registry = MetadataRegistry(study, records)
    registry.records['new-release.json'] = release
    for binding in list(sources.values()) + study['inherited_declarations'] + [release['study'], release['source_archive']]:
        registry.records[binding['path']] = 'source-only-placeholder'
    monkeypatch.setattr(core, 'Registry', lambda root: registry)
    monkeypatch.setattr(core, 'declaration', lambda root: study)
    monkeypatch.setattr(core.receipts, 'verify_archive', lambda *a: None)
    monkeypatch.setattr(core.runtime, 'sha', lambda path: '0'*64)
    monkeypatch.setattr(core.backend, 'verify_profile', lambda *a: pytest.fail('backend reached wrong interpreter'))
    with pytest.raises(ValueError, match='Interpreter identity differs'):
        core.preflight(ROOT, {'path': 'new-release.json', 'sha256': 'a'*64})


@pytest.mark.parametrize('branch,sign,index', [('compression', -1, 0), ('tension', 1, 1)])
def test_pinned_real_axial_header_with_synthetic_rows(study, records, branch, sign, index):
    # Only the literal first line comes from the authenticated real-header diagnostic.
    # Every numeric value below is synthetic. No archive is reopened and no fit occurs.
    binding = study['csv_header_migration']['diagnosis']
    assert hashlib.sha256((ROOT/binding['path']).read_bytes()).hexdigest() == binding['sha256']
    evidence = records['diagnosis']['members'][index]
    assert evidence['branch'] == branch
    header = evidence['raw_first_line'].encode('utf-8')
    assert header == b'displacement,force\n'
    data = header + f'0,0\n{sign*.001},{sign*.002}\n'.encode()
    parsed = core.access.parse_member_csv(data, branch, study['csv_schemas'][branch])
    assert tuple(parsed.coordinate) == (0., sign*.001)
    assert tuple(parsed.response) == (0., sign*.002)
    assert parsed.source_sha256 == hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError, match='displacement'):
        core.access.parse_member_csv(data, branch, records['previous_study']['csv_schemas'][branch])


@pytest.mark.parametrize('header', [
    b'force,displacement\n', b'Displacement,force\n', b'displacement, force\n',
    b'displacement,force,extra\n', b'displacement,displacement\n', b'\xef\xbb\xbfdisplacement,force\n',
    b'#comment\ndisplacement,force\n', b'\ndisplacement,force\n', b'unknown,header\n', b'',
])
def test_unknown_missing_duplicate_or_prefixed_headers_never_autoskip(study, header):
    with pytest.raises(ValueError, match='header differs; no autodetection'):
        core.access.parse_member_csv(header+b'0,0\n-.001,-.002\n', 'compression', study['csv_schemas']['compression'])


def test_repeated_header_and_nonnumeric_rows_are_not_skipped(study):
    for extra in (b'displacement,force\n', b'bad,row\n'):
        data = b'displacement,force\n0,0\n'+extra+b'-.001,-.002\n'
        with pytest.raises(ValueError):
            core.access.parse_member_csv(data, 'compression', study['csv_schemas']['compression'])


def test_explicit_headerless_contract_still_works_on_synthetic_axial_rows(study):
    schema = dict(study['csv_schemas']['compression'], header=None)
    parsed = core.access.parse_member_csv(b'0,0\n-.001,-.002\n', 'compression', schema)
    assert tuple(parsed.response) == (0., -.002)


def synthetic_context(tmp_path, study):
    """Small invented archive; held-out entries contain sentinel text, never read."""
    synthetic = deepcopy(study)
    synthetic['output_root'] = 'synthetic-output'
    payloads = {'compression.csv': b'wrong,force\n0,0\n-.001,-.002\n',
                'tension.csv': b'displacement,force\n0,0\n.001,.002\n',
                'sealed-neg': b'SEALED SYNTHETIC SENTINEL', 'sealed-pos': b'SEALED SYNTHETIC SENTINEL'}
    archive = tmp_path/'synthetic.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for name, data in payloads.items():
            bundle.writestr(name, data)
    with zipfile.ZipFile(archive) as bundle:
        members = [{'path': name, 'bytes': bundle.getinfo(name).file_size,
                    'crc32': f'{bundle.getinfo(name).CRC:08x}'} for name in payloads]
    roles = {'source': {'archive_path': 'synthetic.zip', 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                        'archive_bytes': archive.stat().st_size},
             'calibration': {'members': members[:2]}, 'held_out_validation': {'members': members[2:]}}
    def save(path, value):
        data = core.access.canonical_json(value)
        (tmp_path/path).write_bytes(data)
        return {'path': path, 'sha256': hashlib.sha256(data).hexdigest()}
    synthetic['roles'] = save('roles.json', roles)
    synthetic['protocol'] = save('protocol.json', {'roles': synthetic['roles']})
    release = {'execution': {'csv_schemas': synthetic['csv_schemas'],
                            'access_ledger_path': 'synthetic-output/experiment/access.jsonl'},
               'source_archive': save('source-placeholder.json', {'synthetic': True})}
    release_binding = save('release.json', release)
    registry = SimpleNamespace(verify_all=lambda *args: None, read=lambda binding: release)
    directory = tmp_path/synthetic['output_root']/'experiment'
    directory.mkdir(parents=True)
    return {'root': tmp_path, 'study': synthetic, 'release': release, 'release_binding': release_binding,
            'registry': registry, 'directory': directory, 'references': {}}


def test_first_member_header_failure_stops_worker_before_other_payloads_fit_decks_solver(monkeypatch, tmp_path, study):
    context = synthetic_context(tmp_path, study)
    opened = []
    original = zipfile.ZipFile.open
    def tracked_open(bundle, name, *args, **kwargs):
        member = name.filename if isinstance(name, zipfile.ZipInfo) else name
        opened.append(member)
        assert member == 'compression.csv', 'Second axial or sealed payload was opened after first failure'
        return original(bundle, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, 'open', tracked_open)
    monkeypatch.setattr(core, 'declaration', lambda root: context['study'])
    monkeypatch.setattr(core, 'preflight', lambda *args, **kwargs: context)
    monkeypatch.setattr(core.old, 'OutputWatch', lambda *args: nullcontext(SimpleNamespace(active=None)))
    with pytest.raises(ValueError, match='header differs; no autodetection'):
        runner.worker(tmp_path, context['release_binding'], time.monotonic()+30)
    assert opened == ['compression.csv']
    state = json.loads((context['directory']/'state.json').read_text())
    assert state['status'] == 'failed_or_incomplete'
    assert state['calibration_access_attempted'] is True
    assert state['calibration_responses_accessed'] is None
    assert state['held_out_access_attempted'] is False
    assert state['held_out_responses_accessed'] is False
    assert state['native_calls'] == state['mesher_calls'] == 0
    assert state['runs'] == {}
    assert not any(key in state for key in ['fit', 'predictions', 'preparation', 'freeze'])
    assert set(p.name for p in context['directory'].iterdir()) == {'state.json', 'access.jsonl'}
    ledger = json.loads((context['directory']/'access.jsonl').read_text())
    assert ledger['phase'] == 'calibration_attempt'
    # This is intentionally a batch intent record, not a claim both payloads were read.
    assert ledger['members'] == ['compression.csv', 'tension.csv']


def test_existing_started_marker_refuses_reuse_before_any_child(monkeypatch, tmp_path, study):
    context = synthetic_context(tmp_path, study)
    release = {'authorized': True, 'schema': 'hbe-branch-calibration-release-v3',
               'study': {'path': core.DECLARATION_PATH, 'sha256': core.DECLARATION_SHA256},
               'source_archive': context['release']['source_archive']}
    path = tmp_path/'launch-release.json'
    path.write_bytes(core.access.canonical_json(release))
    binding = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    marker = context['directory'].parent/'.started.json'
    marker.write_text('preserved prior attempt\n')
    monkeypatch.setattr(core, 'declaration', lambda root: context['study'])
    with pytest.raises(FileExistsError):
        runner.launch(tmp_path, binding)
    assert marker.read_text() == 'preserved prior attempt\n'


def functions(path):
    tree = ast.parse(path.read_text())
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def normalize_v3(text):
    text = text.replace('mechanics_hbe_branch_calibration_v3', 'mechanics_hbe_branch_calibration_v2')
    for kind in ('calibration-release', 'calibration-state', 'calibration-result', 'calibration-publication',
                 'fitted-preparation', 'fitted-execution', 'parameter-prediction-freeze', 'fitted-paired-evidence'):
        text = text.replace(f'hbe-branch-{kind}-v3', f'hbe-branch-{kind}-v2')
    return text


def test_all_prior_numerical_access_runtime_and_runner_methods_unchanged():
    previous = functions(ROOT/'scripts/mechanics_hbe_branch_calibration_v2.py')
    candidate = functions(ROOT/'scripts/mechanics_hbe_branch_calibration_v3.py')
    assert set(candidate) == set(previous) | {'verify_csv_header_migration'}
    for name in set(previous)-{'preflight'}:
        assert normalize_v3(candidate[name]) == previous[name], name
    expected_preflight = previous['preflight'].replace('verify_runtime_migration', 'verify_csv_header_migration')
    expected_preflight = expected_preflight.replace("value='runtime_migration'", "value='csv_header_migration'")
    assert normalize_v3(candidate['preflight']) == expected_preflight
    for suffix in ['_readout', '_experiment']:
        before = (ROOT/f'scripts/mechanics_hbe_branch_calibration_v2{suffix}.py').read_text()
        after = (ROOT/f'scripts/mechanics_hbe_branch_calibration_v3{suffix}.py').read_text()
        assert normalize_v3(after) == before
