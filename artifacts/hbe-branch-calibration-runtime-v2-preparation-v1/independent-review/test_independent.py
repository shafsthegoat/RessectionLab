"""Read saved metadata only; never read curves, primitives, or execute solvers."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import mechanics_hbe_branch_calibration_v2 as core
from scripts import mechanics_hbe_branch_calibration_v2_readout as readout
from scripts import mechanics_hbe_branch_calibration_v2_experiment as runner

STUDY = json.loads((ROOT/core.DECLARATION_PATH).read_text())
KEYS = ('previous_study', 'previous_release', 'previous_state', 'previous_result',
        'previous_supervision', 'previous_publication', 'previous_started', 'diagnosis')
PUBLISHED = {
    'previous_study': 'manifests/experiments/hbe-01-03-branch-calibration-v1.json',
    'previous_release': 'artifacts/hbe-branch-calibration-preparation-v1/attempt-01-launcher/release.json',
    'previous_state': 'artifacts/hbe-branch-calibration-preparation-v1/attempt-01/experiment/state.json',
    'previous_result': 'artifacts/hbe-branch-calibration-preparation-v1/attempt-01/experiment/result.json',
    'previous_supervision': 'artifacts/hbe-branch-calibration-preparation-v1/attempt-01/experiment/supervision/supervision.json',
    'previous_publication': 'artifacts/hbe-branch-calibration-preparation-v1/attempt-01/experiment/publication-check.json',
    'previous_started': 'artifacts/hbe-branch-calibration-preparation-v1/attempt-01/.started.json',
}


def forbidden(*args, **kwargs):
    raise AssertionError('Review prohibits measured/primitive/backend/solver/worker execution')


@pytest.fixture(autouse=True)
def no_scientific_execution(monkeypatch):
    for module, name in ((subprocess, 'Popen'), (core.runtime, 'supervise'), (core.old, 'solve'),
                         (core.access.ReleasedStudy, '_read_selected'), (core, 'qualify'),
                         (runner, 'worker'), (runner, 'prepare'), (core.backend, 'verify_profile'),
                         (readout, 'paired_evidence')):
        monkeypatch.setattr(module, name, forbidden)


@pytest.fixture
def records():
    return {key: json.loads((ROOT/STUDY['runtime_migration'][key]['path']).read_text()) for key in KEYS}


class MemoryRegistry:
    def __init__(self, records):
        self.values = {STUDY['runtime_migration'][key]['path']: deepcopy(value) for key,value in records.items()}
        self.calls = []
    def bound(self, binding, *, json_value=True):
        self.calls.append(binding['path'])
        return deepcopy(self.values[binding['path']])
    def add(self, *args, **kwargs):
        pass


def release():
    return {'schema':'hbe-branch-calibration-release-v2', 'authorized':True,
            'phase':'calibrate_and_validate', 'study':{'path':core.DECLARATION_PATH,'sha256':core.DECLARATION_SHA256},
            'authorized_native_calls':2, 'backend_profile':STUDY['backend_profile'],
            'interpreter':STUDY['interpreter'], 'predecessor_failure':STUDY['runtime_migration']['previous_state']}


def mock_preflight(monkeypatch, records, authority):
    registry = MemoryRegistry(records)
    registry.values['mock-release.json'] = authority
    monkeypatch.setattr(core, 'Registry', lambda root: registry)
    monkeypatch.setattr(core, 'declaration', lambda root: deepcopy(STUDY))
    return registry, {'path':'mock-release.json','sha256':'a'*64}


def test_exact_scientific_projection_and_current_interpreter(records):
    old = records['previous_study']
    excluded = {'schema','study_id','output_root','interpreter'}
    assert set(STUDY) == set(old) | {'runtime_migration'}
    fixed = {key:value for key,value in old.items() if key not in excluded}
    assert fixed == {key:value for key,value in STUDY.items() if key not in excluded | {'runtime_migration'}}
    assert len(fixed) == 23
    assert STUDY['interpreter']['path'] == old['interpreter']['path']
    assert STUDY['interpreter']['sha256'] == '2498a31965647f1507a53e391842b23c29d96f0ef4550475f34a1d2b30c23ddb'
    assert old['interpreter']['sha256'] == 'b33be71b340c3829b6610a70cf5a89123a11d0afd23bc8ec5de385b410efb43a'
    binary = Path(sys.executable).resolve()
    assert binary == Path(STUDY['interpreter']['path']).resolve()
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == STUDY['interpreter']['sha256']
    assert runner.core is readout.core is core
    assert core.NEW_SOURCES == {'branch_calibration':'mechanics_hbe_branch_calibration_v2.py',
        'branch_calibration_readout':'mechanics_hbe_branch_calibration_v2_readout.py',
        'branch_calibration_runner':'mechanics_hbe_branch_calibration_v2_experiment.py'}


def test_actual_predecessor_metadata_registers_exact_eight_files():
    registry = core.Registry(ROOT)
    result = core.verify_runtime_migration(STUDY, registry)
    assert set(result) == set(KEYS)
    assert set(registry.entries) == {STUDY['runtime_migration'][key]['path'] for key in KEYS}
    registry.verify_all(time.monotonic()+5)
    assert result['previous_state']['native_calls'] == 0
    assert result['previous_state']['calibration_responses_accessed'] is False
    assert result['previous_state']['held_out_responses_accessed'] is False


def test_published_copies_alone_reproduce_original_missing_path_failure(tmp_path):
    for key, published in PUBLISHED.items():
        payload = (ROOT/published).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == STUDY['runtime_migration'][key]['sha256']
        destination = tmp_path/published
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(payload)
    with pytest.raises(FileNotFoundError):
        core.verify_runtime_migration(STUDY, core.Registry(tmp_path))


def test_exact_original_path_metadata_bundle_closes_portability_without_rewriting(tmp_path):
    for key in KEYS:
        binding = STUDY['runtime_migration'][key]
        payload = (ROOT/binding['path']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == binding['sha256']
        destination = tmp_path/binding['path']
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(payload)
    registry = core.Registry(tmp_path)
    checked = core.verify_runtime_migration(STUDY, registry)
    assert len(checked) == len(registry.entries) == 8
    registry.verify_all(time.monotonic()+5)
    failed_state = tmp_path/STUDY['runtime_migration']['previous_state']['path']
    failed_state.write_bytes(b'{}\n')
    with pytest.raises(ValueError, match='[Hh]ash'):
        registry.verify_all(time.monotonic()+5)


@pytest.mark.parametrize('key,field,value', [
    ('previous_state','held_out_access_attempted',True),
    ('previous_state','native_calls',False),
    ('previous_result','state',{'path':'another-state.json','sha256':'0'*64}),
    ('diagnosis','actual',{'path':STUDY['interpreter']['path'],'sha256':'0'*64}),
])
def test_changed_predecessor_refused_before_backend_or_measured_access(monkeypatch,records,key,field,value):
    records[key][field] = value
    registry, binding = mock_preflight(monkeypatch,records,release())
    with pytest.raises(ValueError):
        core.preflight(ROOT,binding)
    assert registry.calls[0] == 'mock-release.json'


def test_old_release_cannot_authorize_runtime_migration(monkeypatch,records):
    registry,binding = mock_preflight(monkeypatch,records,records['previous_release'])
    with pytest.raises(ValueError,match='Exact committed'):
        core.preflight(ROOT,binding)
    assert registry.calls == ['mock-release.json']


@pytest.mark.parametrize('fault',['wrong_interpreter','wrong_source_origin'])
def test_runtime_and_source_origin_refused_before_backend_or_data(monkeypatch,records,fault):
    authority = release()
    sources = deepcopy(STUDY['inherited_sources'])
    sources.update({key:{'path':'scripts/'+name,'sha256':hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()}
                    for key,name in core.NEW_SOURCES.items()})
    authority.update(source_bindings=sources,source_archive={'path':'mock-source.tar','sha256':'d'*64},source_commit='e'*40)
    if fault == 'wrong_source_origin':
        sources['branch_calibration']['path'] = 'scripts/mechanics_hbe_branch_calibration.py'
    registry,binding = mock_preflight(monkeypatch,records,authority)
    for item in list(sources.values()) + STUDY['inherited_declarations'] + [authority['study'],authority['source_archive']]:
        registry.values[item['path']] = 'source-metadata-placeholder'
    monkeypatch.setattr(core.receipts,'verify_archive',lambda *args:None)
    if fault == 'wrong_interpreter':
        monkeypatch.setattr(core.runtime,'sha',lambda path:'0'*64)
    with pytest.raises(ValueError,match='Interpreter identity differs' if fault=='wrong_interpreter' else 'source origin'):
        core.preflight(ROOT,binding)


def stub_reader(monkeypatch,events):
    obj = object.__new__(core.BranchReleasedStudy)
    obj.context = {'release':{'execution':{'csv_schemas':STUDY['csv_schemas']}}}
    monkeypatch.setattr(obj,'_release',lambda:None)
    monkeypatch.setattr(obj,'_events',lambda:deepcopy(events))
    monkeypatch.setattr(obj,'_read_selected',forbidden)
    return obj


def test_held_out_refused_without_durable_freeze(monkeypatch):
    obj = stub_reader(monkeypatch,[])
    with pytest.raises(ValueError,match='Audited durable freeze'):
        obj.evaluate_held_out(freeze_binding={'path':'not-created.json','sha256':'0'*64},
                              schemas={b:STUDY['csv_schemas'][b] for b in core.TORSION})


def test_calibration_refused_after_held_out_exposure(monkeypatch):
    obj = stub_reader(monkeypatch,[{'phase':'held_out_attempt'}])
    with pytest.raises(ValueError,match='Held-out exposure seals'):
        obj.read_calibration({b:STUDY['csv_schemas'][b] for b in core.AXIAL})


def test_failed_numerical_confirmation_cannot_freeze_or_reveal(monkeypatch,tmp_path):
    obj = stub_reader(monkeypatch,[])
    obj._calibration = {'pure-test-marker':True}
    fit,pred = {'mu_Pa':1500.,'scale':1.5},{'pure-test-prediction':True}
    obj.root=tmp_path;obj.deadline=time.monotonic()+5
    registry=SimpleNamespace(bound=lambda binding:fit if binding['path']=='fit.json' else pred,
                             add=lambda binding:None, verify_all=forbidden)
    obj.context.update(registry=registry,directory=tmp_path,references={b:{} for b in core.AXIAL})
    monkeypatch.setattr(core,'curve_from_row',lambda *args:'pure-test-reference')
    monkeypatch.setattr(core.evaluation,'fit_scale',lambda *args:fit)
    monkeypatch.setattr(core,'predictions',lambda *args:pred)
    monkeypatch.setattr(readout,'paired_evidence',lambda *args:{'passed':False})
    monkeypatch.setattr(core.receipts,'durable_json',lambda *args:{'path':'failed-numerical.json','sha256':'f'*64})
    monkeypatch.setattr(core.access,'exclusive_json',forbidden)
    with pytest.raises(ValueError,match='confirmation failed; held-out remains sealed'):
        obj.freeze_predictions(fit_binding={'path':'fit.json'},prediction_binding={'path':'pred.json'},
                               fitted_runs={},output_path='freeze.json')


def test_numerical_and_access_methods_are_unchanged_modulo_version_names():
    replacements = {'mechanics_hbe_branch_calibration_v2':'mechanics_hbe_branch_calibration'}
    for stem in ('hbe-branch-calibration-release','hbe-branch-calibration-state','hbe-branch-calibration-result',
                 'hbe-branch-calibration-publication','hbe-branch-fitted-preparation','hbe-branch-fitted-execution',
                 'hbe-branch-parameter-prediction-freeze','hbe-branch-fitted-paired-evidence'):
        replacements[stem+'-v2']=stem+'-v1'
    def normalize(text):
        for old,new in replacements.items():text=text.replace(old,new)
        return text
    for suffix in ('_experiment','_readout'):
        assert normalize((ROOT/f'scripts/mechanics_hbe_branch_calibration_v2{suffix}.py').read_text()) == (ROOT/f'scripts/mechanics_hbe_branch_calibration{suffix}.py').read_text()
    def functions(path):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    old = functions(ROOT/'scripts/mechanics_hbe_branch_calibration.py')
    new = functions(ROOT/'scripts/mechanics_hbe_branch_calibration_v2.py')
    assert set(new) == set(old)|{'verify_runtime_migration'}
    for name in set(old)-{'preflight'}:assert normalize(new[name]) == old[name],name
