"""Analytical backend-binding contracts; no executable, solver or HBE access.

Profile verification is stubbed only in explicit propagation tests. The actual
profile verifier has separate prerequisite controls owned by the backend module.
Deck adaptation and access/execution binding checks below remain real.
"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from scripts import mechanics_hbe_access as a
from scripts import mechanics_hbe_backend as backend
from scripts import mechanics_hbe_readout as readout


def save(root, relative, value, *, raw=False):
    data = value.encode() if raw else a.canonical_json(value)
    path = root/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {'path': relative, 'sha256': a.digest(data)}


def prepared(root):
    prefix = root/'constructed-runtime'
    executable = save(root, 'constructed-runtime/install/bin/febio4', 'not executable', raw=True)
    identity = {'executable_sha256': executable['sha256'], 'runtime_version': '4.13.0', 'architecture': 'arm64'}
    runtime = save(root, 'constructed-runtime/runtime.json', identity)
    profile = save(root, 'constructed-profile.json', {'test_only': 'not verification evidence'})
    controls = save(root, 'constructed-controls.json', {'test_only': 'not actual controls'})
    context = {'profile_id': 'accelerate_csc_v1', 'binding': profile, 'runtime_identity': runtime,
               'runtime': identity, 'hex8_controls': controls, 'tet10_mpc_controls': controls,
               'prefix': str(prefix), 'inputs': {str(root/controls['path']): controls['sha256']}}
    original = '<febio_spec><Control><solver><linear_solver type="skyline" /></solver></Control><Material>fixed-law</Material></febio_spec>'
    original_binding = save(root, 'original.feb', original, raw=True)
    adapted = backend.transform_deck(original)
    primitives = {'deck': save(root, 'run/specimen.feb', adapted, raw=True)}
    for key, name in [('nodes', 'nodes.log'), ('elements', 'elements.log'), ('solver', 'solver.log'),
                      ('mesh', 'mesh.json'), ('loading', 'loading.json')]:
        primitives[key] = save(root, 'run/'+name, {'test_only': key})
    run_id = 'compression:N4:S60:reference'
    record = {'schema': 'hbe-run-execution-v1', 'run_id': run_id, 'protocol_sha256': '1'*64,
              'primitive_bindings': primitives, 'execution': {'exit_code': 0, 'timed_out': False, 'elapsed_seconds': 1},
              'runtime_identity': runtime, 'executable': executable, 'backend_profile': profile,
              'backend_source_deck': original_binding, 'cwd': 'run',
              'command': [str(root/executable['path']), '-noconfig', '-no_title', '-i', 'specimen.feb', '-o', 'solver.log']}
    row = {'primitive_bindings': primitives, 'execution_binding': save(root, 'execution.json', record),
           'mesh_N': 4, 'steps': 60, 'branch': 'compression', 'mu_Pa': 1000}
    return profile, context, record, row


def isolated_profile(monkeypatch, profile, context, calls=None):
    """Only isolate already-reviewed profile verification, never infer from row."""
    def verify(root, binding):
        if calls is not None:
            calls.append(copy.deepcopy(binding))
        if binding is None:
            return None
        assert binding == profile
        return context
    monkeypatch.setattr(a, '_verified_backend', verify)


def test_explicit_profile_requires_ninth_source_without_changing_legacy_inventory():
    assert a.source_files_for_release({}) == a.SOURCE_FILES
    release = {'execution': {'backend_profile': {'path': 'future.json', 'sha256': '9'*64}}}
    assert a.source_files_for_release(release) == {**a.SOURCE_FILES, 'backend_profile': 'mechanics_hbe_backend.py'}
    with pytest.raises(ValueError, match='exact artifact binding'):
        a.source_files_for_release({'execution': {'backend_profile': None}})
    release['execution']['source_bindings'] = dict.fromkeys(a.SOURCE_FILES)
    with pytest.raises(ValueError, match='source inventory'):
        a.verify_release_provenance('.', release, '1'*64)


def test_invalid_actual_profile_cannot_authorize_a_row(tmp_path):
    profile, _, record, row = prepared(tmp_path)
    with pytest.raises(ValueError, match='complete repaired-Accelerate profile'):
        a.verify_run_execution(tmp_path, record['run_id'], row, '1'*64,
                               expected_backend_profile=profile)


def test_verified_profile_and_exact_solver_only_deck_are_required(tmp_path, monkeypatch):
    profile, context, record, row = prepared(tmp_path)
    isolated_profile(monkeypatch, profile, context)
    checked = a.verify_run_execution(tmp_path, record['run_id'], row, '1'*64,
                                      expected_backend_profile=profile)
    assert checked['backend_profile'] == profile
    changed = (tmp_path/row['primitive_bindings']['deck']['path']).read_text().replace('fixed-law', 'changed-law')
    row['primitive_bindings']['deck'] = save(tmp_path, 'run/specimen.feb', changed, raw=True)
    record['primitive_bindings'] = row['primitive_bindings']
    row['execution_binding'] = save(tmp_path, 'execution.json', record)
    with pytest.raises(ValueError, match='outside the exact solver subtree'):
        a.verify_run_execution(tmp_path, record['run_id'], row, '1'*64,
                               expected_backend_profile=profile)


@pytest.mark.parametrize('change', ['missing_profile', 'different_profile', 'runtime_path', 'prefix', 'missing_original'])
def test_mixed_backend_or_runtime_is_rejected_even_with_refreshed_receipt(tmp_path, monkeypatch, change):
    profile, context, record, row = prepared(tmp_path)
    isolated_profile(monkeypatch, profile, context)
    if change == 'missing_profile':
        del record['backend_profile']
    elif change == 'different_profile':
        record['backend_profile'] = save(tmp_path, 'other-profile.json', {'different': True})
    elif change == 'runtime_path':
        record['runtime_identity'] = save(tmp_path, 'same-runtime-different-path.json', context['runtime'])
    elif change == 'prefix':
        record['executable'] = save(tmp_path, 'other-bin/febio4', 'not executable', raw=True)
        record['command'][0] = str(tmp_path/record['executable']['path'])
    else:
        del record['backend_source_deck']
    row['execution_binding'] = save(tmp_path, 'changed-execution.json', record)
    with pytest.raises(ValueError, match='backend|scientific deck'):
        a.verify_run_execution(tmp_path, record['run_id'], row, '1'*64,
                               expected_backend_profile=profile)


def test_default_legacy_release_cannot_infer_authorization_from_row(tmp_path, monkeypatch):
    profile, context, record, row = prepared(tmp_path)
    isolated_profile(monkeypatch, profile, context)
    monkeypatch.setattr(a, 'RUNTIME_IDENTITY_SHA256', context['runtime_identity']['sha256'])
    with pytest.raises(ValueError, match='not authorized'):
        a.verify_run_execution(tmp_path, record['run_id'], row, '1'*64)


def test_full_replay_verifies_profile_once_and_rejects_second_mixed_row(tmp_path, monkeypatch):
    """Only readout physics is replaced here; this test verifies profile propagation."""
    profile, context, first, first_row = prepared(tmp_path)
    calls = []
    isolated_profile(monkeypatch, profile, context, calls)
    second = copy.deepcopy(first)
    second['run_id'] = 'tension:N4:S60:reference'
    second['backend_profile'] = save(tmp_path, 'other.json', {'unreleased': True})
    second_row = copy.deepcopy(first_row)
    second_row['branch'] = 'tension'
    second_row['execution_binding'] = save(tmp_path, 'second.json', second)
    read_calls = []
    def analytic_read(*args, **kwargs):
        read_calls.append(kwargs['expected_branch'])
        return {'test_only': 'no numerical proof'}, None
    monkeypatch.setattr(readout, 'read_run', analytic_read)
    monkeypatch.setattr(readout, 'build_numerical_evidence', lambda *args, **kwargs: {'test_only': True})
    report = {'runs': {first['run_id']: first_row, second['run_id']: second_row}, 'source_bindings': {}}
    with pytest.raises(ValueError, match='explicitly released profile'):
        a._replay_evidence(tmp_path, report, protocol_sha256='1'*64, fitted_mu_Pa=1000,
                           expected_backend_profile=profile)
    assert calls == [profile]
    assert read_calls == ['compression']


def test_numerical_validator_forwards_only_explicit_profile(tmp_path, monkeypatch):
    profile = save(tmp_path, 'profile.json', {'test_only': True})
    report = {'test_only': 'this isolates propagation, not numerical verification'}
    binding = save(tmp_path, 'report.json', report)
    seen = []
    monkeypatch.setattr(a, '_check_report', lambda *args, **kwargs: report)
    monkeypatch.setattr(a, 'check_numeric_report', lambda *args, **kwargs: report)
    def replay(*args, **kwargs):
        seen.append(kwargs['expected_backend_profile'])
        return report
    monkeypatch.setattr(a, '_replay_evidence', replay)
    assert a.validate_numerical_evidence(tmp_path, binding, protocol_sha256='1'*64,
        fitted_mu_Pa=1000, expected_backend_profile=profile) == report
    assert seen == [profile]


def test_saved_freeze_propagates_profile_and_captures_all_verified_profile_inputs(tmp_path, monkeypatch):
    """Analytical lifecycle isolation, not a substitute for20 actual FEM receipts."""
    spec = importlib.util.spec_from_file_location('backend_access_owner_fixtures',
        Path(__file__).with_name('test_mechanics_hbe_access.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.analytical_prerequisite_stubs(monkeypatch)
    study = fixture.study_fixture(tmp_path)
    profile = save(tmp_path, 'profile.json', {'test_only': 'explicit future profile'})
    extra = save(tmp_path, 'profile-control.json', {'test_only': 'hash-bound input, not numerical evidence'})
    release = json.loads((tmp_path/study.release_binding['path']).read_text())
    release['execution']['backend_profile'] = profile
    release_binding = save(tmp_path, 'release.json', release)
    study = a.ReleasedStudy(tmp_path, study.protocol_binding, release_binding, ledger_path='access.jsonl')
    context = {'profile_id': 'accelerate_csc_v1', 'binding': profile,
               'inputs': {str(tmp_path/extra['path']): extra['sha256']}}
    isolated_profile(monkeypatch, profile, context)
    seen = []
    def analytical_replay(root, binding, *, protocol_sha256, fitted_mu_Pa, expected_backend_profile=None):
        seen.append(expected_backend_profile)
        report = a.verify_binding(root, binding, read_json=True)
        return a._check_report(report, protocol_sha256=protocol_sha256,
                               fitted_mu_Pa=fitted_mu_Pa, require_fitted=True)
    monkeypatch.setattr(a, 'validate_numerical_evidence', analytical_replay)
    frozen, *_ = fixture.frozen_fixture(tmp_path, study)
    record = a.verify_binding(tmp_path, frozen, read_json=True)
    assert seen == [profile]
    assert extra in record['verified_file_bindings']
    (tmp_path/extra['path']).write_text('{}')
    with pytest.raises(ValueError, match='hash changed'):
        study.evaluate_held_out(freeze_binding=frozen,
            schemas=fixture.schemas(('torsion_neg', 'torsion_pos')))
