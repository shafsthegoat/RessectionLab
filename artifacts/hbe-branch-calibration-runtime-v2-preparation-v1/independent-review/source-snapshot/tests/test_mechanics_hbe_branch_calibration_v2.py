"""Runtime-only migration controls: saved metadata and analytical values, no responses."""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mechanics_hbe_branch_calibration as old
from scripts import mechanics_hbe_branch_calibration_readout as old_readout
from scripts import mechanics_hbe_branch_calibration_v2 as core
from scripts import mechanics_hbe_branch_calibration_v2_readout as readout
from scripts import mechanics_hbe_branch_calibration_v2_experiment as runner

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ('previous_study', 'previous_release', 'previous_state', 'previous_result',
            'previous_supervision', 'previous_publication', 'previous_started', 'diagnosis')
STUDY = json.loads((ROOT/core.DECLARATION_PATH).read_text())
FIXED_FIELDS = sorted(set(STUDY) - {'schema', 'study_id', 'output_root', 'interpreter', 'runtime_migration'})


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Measured/native/qualification execution forbidden in migration controls')
    for module, name in [(subprocess, 'Popen'), (core.runtime, 'supervise'), (core.old, 'solve'),
                         (core.access.ReleasedStudy, '_read_selected'), (core, 'qualify'),
                         (runner, 'worker'), (runner, 'prepare')]:
        monkeypatch.setattr(module, name, forbidden)


@pytest.fixture
def study():
    return deepcopy(STUDY)


@pytest.fixture
def records(study):
    return {key: json.loads((ROOT/study['runtime_migration'][key]['path']).read_text()) for key in BINDINGS}


class MetadataRegistry:
    """No filesystem/member access; bindings map only to copied saved metadata."""
    def __init__(self, study, records):
        self.records = {study['runtime_migration'][key]['path']: value for key, value in records.items()}
        self.calls = []

    def bound(self, binding, *, json_value=True):
        self.calls.append(binding['path'])
        return deepcopy(self.records[binding['path']])

    def add(self, binding):
        pass


def test_exact_manifest_pin_and_preserved_scientific_design(study, records):
    assert hashlib.sha256((ROOT/core.DECLARATION_PATH).read_bytes()).hexdigest() == core.DECLARATION_SHA256
    assert core.DECLARATION_PATH != old.DECLARATION_PATH
    assert study['interpreter']['sha256'] == '2498a31965647f1507a53e391842b23c29d96f0ef4550475f34a1d2b30c23ddb'
    assert records['previous_study']['interpreter']['sha256'] == 'b33be71b340c3829b6610a70cf5a89123a11d0afd23bc8ec5de385b410efb43a'
    for key in FIXED_FIELDS:
        assert study[key] == records['previous_study'][key]
    assert study['output_root'] != records['previous_study']['output_root']
    assert len(study['inherited_sources']) == 26
    assert len(study['inherited_declarations']) == 8
    for binding in list(study['inherited_sources'].values()) + study['inherited_declarations']:
        assert hashlib.sha256((ROOT/binding['path']).read_bytes()).hexdigest() == binding['sha256']


def test_actual_failed_metadata_authenticated_and_registered(study):
    registry = core.Registry(ROOT)
    checked = core.verify_runtime_migration(study, registry)
    assert set(checked) == set(BINDINGS)
    assert set(registry.entries) == {study['runtime_migration'][key]['path'] for key in BINDINGS}
    assert checked['previous_state']['calibration_responses_accessed'] is False
    assert checked['previous_state']['native_calls'] == 0


@pytest.mark.parametrize('field', FIXED_FIELDS)
def test_every_unchanged_field_refuses_runtime_excuse(study, records, field):
    study[field] = {'unexpected_runtime_excuse': True}
    with pytest.raises(ValueError, match='scientific and historical'):
        core.verify_runtime_migration(study, MetadataRegistry(study, records))


@pytest.mark.parametrize('field,value', [
    ('calibration_access_attempted', True), ('calibration_responses_accessed', None),
    ('held_out_access_attempted', True), ('held_out_responses_accessed', 0),
    ('automatic_retry', True), ('native_calls', 1), ('native_calls', False),
    ('mesher_calls', 1), ('runs', {'compression': {}}), ('physical_validation_pass', True),
    ('fit', {}), ('predictions', {}), ('freeze', {}), ('held_out_metrics', {}),
    ('calibration_metrics', {}), ('preparation_seconds', 1.),
    ('status', 'completed'), ('error', {'type': 'ValueError', 'message': 'some other failure'}),
])
def test_predecessor_cannot_contain_access_or_work(study, records, field, value):
    records['previous_state'][field] = value
    with pytest.raises(ValueError, match='zero-call interpreter failure'):
        core.verify_runtime_migration(study, MetadataRegistry(study, records))


@pytest.mark.parametrize('record,field,value', [
    ('previous_release', 'authorized', False), ('previous_release', 'authorized_native_calls', True),
    ('previous_release', 'study', {'path': 'other', 'sha256': '0'*64}),
    ('previous_release', 'interpreter', {'path': '/unknown', 'sha256': '0'*64}),
    ('previous_result', 'release', {'path': 'other', 'sha256': '0'*64}),
    ('previous_result', 'state', {'path': 'other', 'sha256': '0'*64}),
    ('previous_result', 'supervision', {}), ('previous_result', 'physical_validation_pass', True),
    ('previous_supervision', 'no_retry', False), ('previous_supervision', 'exit_code', True),
    ('previous_supervision', 'cleanup_error', 'orphan'), ('previous_supervision', 'kill_reason', 'timeout'),
    ('previous_publication', 'accepted', True), ('previous_publication', 'result', {}),
    ('previous_started', 'no_retry', False), ('previous_started', 'no_retry', 1), ('diagnosis', 'pinned', {}),
    ('diagnosis', 'actual', {}), ('diagnosis', 'sanitized_child', {}),
    ('diagnosis', 'first_execution', {'native_calls': 1}),
])
def test_predecessor_receipt_joins_fail_closed(study, records, record, field, value):
    records[record][field] = value
    with pytest.raises(ValueError):
        core.verify_runtime_migration(study, MetadataRegistry(study, records))


@pytest.mark.parametrize('key', BINDINGS)
def test_missing_predecessor_record_refused(study, key):
    del study['runtime_migration'][key]
    with pytest.raises(ValueError, match='migration declaration'):
        core.verify_runtime_migration(study, SimpleNamespace(bound=lambda b: pytest.fail('unexpected read')))


@pytest.mark.parametrize('field,value', [('automatic_retry', True), ('scientific_fields_unchanged', False),
                                         ('reason', 'retry_for_better_result')])
def test_migration_authority_is_explicit(study, records, field, value):
    study['runtime_migration'][field] = value
    with pytest.raises(ValueError, match='migration declaration'):
        core.verify_runtime_migration(study, MetadataRegistry(study, records))


def test_predecessor_content_remains_rechecked_after_preflight(tmp_path, study):
    for key in BINDINGS:
        rel = study['runtime_migration'][key]['path']
        target = tmp_path/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/rel).read_bytes())
    registry = core.Registry(tmp_path)
    core.verify_runtime_migration(study, registry)
    path = tmp_path/study['runtime_migration']['previous_state']['path']
    path.write_text('{}\n')
    with pytest.raises(ValueError, match='hash|Hash'):
        registry.verify_all(time.monotonic()+5)


def release_for(study):
    return {'schema': 'hbe-branch-calibration-release-v2', 'authorized': True,
            'phase': 'calibrate_and_validate', 'study': {'path': core.DECLARATION_PATH,
            'sha256': core.DECLARATION_SHA256}, 'authorized_native_calls': 2,
            'backend_profile': study['backend_profile'], 'interpreter': study['interpreter'],
            'predecessor_failure': study['runtime_migration']['previous_state']}


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
    with pytest.raises(ValueError, match='zero-call interpreter failure'):
        core.preflight(ROOT, binding)


def test_new_release_requires_exact_predecessor_identity(monkeypatch, study, records):
    release = release_for(study)
    release['predecessor_failure'] = {'path': 'other-state', 'sha256': 'b'*64}
    binding = preflight_fixture(monkeypatch, study, records, release)
    with pytest.raises(ValueError, match='Exact committed'):
        core.preflight(ROOT, binding)


def test_old_release_cannot_authorize_v2(monkeypatch, study, records):
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


def functions(path):
    tree = ast.parse(path.read_text())
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def normalize_v2(text):
    # Only branch-owned namespace/schema changes; arithmetic must match exactly.
    return (text.replace('mechanics_hbe_branch_calibration_v2', 'mechanics_hbe_branch_calibration')
            .replace('hbe-branch-calibration-release-v2', 'hbe-branch-calibration-release-v1')
            .replace('hbe-branch-calibration-state-v2', 'hbe-branch-calibration-state-v1')
            .replace('hbe-branch-calibration-result-v2', 'hbe-branch-calibration-result-v1')
            .replace('hbe-branch-calibration-publication-v2', 'hbe-branch-calibration-publication-v1')
            .replace('hbe-branch-fitted-preparation-v2', 'hbe-branch-fitted-preparation-v1')
            .replace('hbe-branch-fitted-execution-v2', 'hbe-branch-fitted-execution-v1')
            .replace('hbe-branch-parameter-prediction-freeze-v2', 'hbe-branch-parameter-prediction-freeze-v1')
            .replace('hbe-branch-fitted-paired-evidence-v2', 'hbe-branch-fitted-paired-evidence-v1'))


def test_all_original_numerical_and_access_methods_unchanged():
    previous = functions(ROOT/'scripts/mechanics_hbe_branch_calibration.py')
    candidate = functions(ROOT/'scripts/mechanics_hbe_branch_calibration_v2.py')
    assert set(candidate) == set(previous) | {'verify_runtime_migration'}
    for name in set(previous)-{'preflight'}:
        assert normalize_v2(candidate[name]) == previous[name], name
    for suffix in ['_readout', '_experiment']:
        previous = (ROOT/f'scripts/mechanics_hbe_branch_calibration{suffix}.py').read_text()
        candidate = (ROOT/f'scripts/mechanics_hbe_branch_calibration_v2{suffix}.py').read_text()
        assert normalize_v2(candidate) == previous


def test_identical_streamed_scaling_with_unfavorable_component_retained():
    reducers = [old_readout.ScaleReduction(1750.), readout.ScaleReduction(1750.)]
    rest = np.array([[0., 0., 0.], [.001, .002, .003]])
    for i in range(121):
        ref = {'index': i, 'current': rest+i*1e-7, 'raw': np.full((2,3), i*1e-8), 'torque': i*1e-8}
        fitted = {'index': i, 'current': ref['current'].copy(), 'raw': ref['raw']*1.75,
                  'torque': ref['torque']*1.75}
        if i == 60:
            fitted['raw'][1,2] += 1e-5
        for reducer in reducers:
            reducer.update(ref, fitted, rest)
    assert reducers[0].finish() == reducers[1].finish()
    assert reducers[1].finish()['reaction']['actual'] > reducers[1].finish()['reaction']['limit']


def test_v2_fit_is_same_objective_on_analytical_observations():
    refs = {b: core.evaluation.curve(b, [0., sign*.001], [0., sign*.002])
            for b, sign in [('compression', -1), ('tension', 1)]}
    observed = {b: core.evaluation.scale_prediction(row, 1.5) for b, row in refs.items()}
    assert core.evaluation is old.evaluation
    assert core.evaluation.fit_scale(observed, refs) == old.evaluation.fit_scale(observed, refs)
    assert core.evaluation.fit_scale(observed, refs)['scale'] == pytest.approx(1.5)
