"""Complete native execution and its bounded transaction boundary."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from resectionlab import native_backend as backend
from resectionlab import surgical_engine as module
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene


def empty_engine():
    return SurgicalEngine(SurgicalScene(np.zeros((8, 8, 8), np.uint16), np.eye(4), [], {}, {}))


@pytest.fixture
def compiled():
    if backend._module is None:
        pytest.skip('compiled executor required')
    return backend._module


def test_public_step_runs_native_loop_and_physics_with_exact_reference_result(compiled, monkeypatch):
    action = dict(slot='left', operation='inspect', duration=.3)
    source = empty_engine()
    with monkeypatch.context() as reference:
        reference.setattr(backend, '_module', None)
        expected = source.fresh()
        receipt = expected.step(action)
        snapshot = expected.snapshot()
    actual = source.fresh()
    backend.reset_counters()
    monkeypatch.setattr(SurgicalEngine, '_physical_substep_reference',
                        lambda *args: pytest.fail('native executor called Python physical loop'))
    assert actual.step(action) == receipt
    assert actual.snapshot() == snapshot
    identity = backend.backend_identity()
    counts = identity['counters']
    resident = identity.get('residentExecution', {})
    assert counts['trajectoryCalls'] + resident.get('trajectories', 0) == 1
    assert counts['physicalSubstepCalls'] + resident.get('physicalSubsteps', 0) == 3


@pytest.mark.parametrize('failure', [None, ValueError('stop'), RuntimeError('cancel')])
def test_disk_drift_is_rejected_before_publication_and_scope_resets(compiled, monkeypatch, tmp_path, failure):
    watched = tmp_path / 'owned-native-input'
    watched.write_bytes(b'original')
    files = dict(backend._files)
    files[watched] = (backend._stat(watched), hashlib.sha256(watched.read_bytes()).hexdigest())
    monkeypatch.setattr(backend, '_files', files)
    engine = empty_engine()
    before = copy.deepcopy(engine.state)

    def effect(*args):
        watched.write_bytes(b'changed')
        # Disk checks belong to the bounded executor boundary.
        backend._check_live()
        if failure is not None:
            raise failure
        return []

    monkeypatch.setattr(SurgicalEngine, '_effect', effect)
    with pytest.raises(backend.NativeIntegrityError, match='file changed'):
        engine.step(dict(slot='left', operation='apply', duration=.1))
    assert getattr(backend._execution, 'depth', 0) == 0
    assert engine.state == before and not engine.actions and not engine.history


def test_inner_binding_still_detects_environment_change(compiled, monkeypatch):
    engine = empty_engine()
    before = copy.deepcopy(engine.state)

    def effect(*args):
        monkeypatch.setenv('RESSECTIONLAB_NUMERICAL_BACKEND', 'changed')
        backend._check_live()

    monkeypatch.setattr(SurgicalEngine, '_effect', effect)
    with pytest.raises(backend.NativeIntegrityError, match='environment changed'):
        engine.step(dict(slot='left', operation='apply', duration=.1))
    assert getattr(backend._execution, 'depth', 0) == 0
    assert engine.state == before and not engine.history


def test_owned_digest_preserves_bits_and_custom_graph_fallback(compiled):
    shared = [1, -0., 'é😀', None, True]
    value = {'last': shared, 'first': (shared, 10**100)}
    assert module._owned_digest(value) == module._owned_digest_reference(value)
    marker = object()
    for unusual in ({'x': np.int64(2)}, {'x': object()}, {1: 'key'}, 3):
        assert compiled.owned_digest(unusual, json.dumps, hashlib.sha256, marker) is marker
    for invalid in ({'x': float('nan')}, {'x': float('inf')}):
        assert compiled.owned_digest(invalid, json.dumps, hashlib.sha256, marker) is marker


def test_native_occupancy_matches_original_indexing_and_falls_back_on_strides(compiled):
    from resectionlab.surgical_occupancy import MaterialOccupancy
    labels = np.ones(17, np.uint16)
    labels[4] = 0
    owner = MaterialOccupancy(labels, [0, 3, -1], {3: 12, -1: 2})
    ids = np.array([[0, 3, -1], [4, 8, -17]], np.int64)
    expected = np.array([[owner[int(i)] for i in row] for row in ids], np.int64)
    marker = object()
    actual = compiled.occupancy_values(labels, ids, owner.cleared, owner.moved, marker)
    assert actual.dtype == expected.dtype and actual.shape == ids.shape
    assert actual.tobytes() == expected.tobytes() and actual.flags.owndata
    assert compiled.occupancy_values(labels, ids[:, ::-1], owner.cleared, owner.moved, marker) is marker
    with pytest.raises(IndexError, match='out of bounds'):
        compiled.occupancy_values(labels, np.array([17], np.int64), owner.cleared, owner.moved, marker)


def test_function_identity_batch_retains_metadata_and_mutable_default_markers(compiled):
    from resectionlab.surgical_geometry_cache import _function_identity, _DEFAULT_ATOMS
    def plain(): pass
    def scalar(x=2, *, y='same'): pass
    def mutable(x=[]): pass
    actual = compiled.function_identities((plain, scalar), _DEFAULT_ATOMS)
    assert actual == tuple(_function_identity(f) for f in (plain, scalar))
    first = compiled.function_identities((mutable, len), _DEFAULT_ATOMS)
    second = compiled.function_identities((mutable, len), _DEFAULT_ATOMS)
    assert first[0][:-1] == second[0][:-1] and first[0][-1] is not second[0][-1]
    assert first[1][:2] == second[1][:2] and first[1][-1] is not second[1][-1]


def test_custom_material_rows_keep_both_reference_lookup_passes(compiled):
    labels = np.zeros((8, 8, 8), np.uint16)
    labels[3, 3, 3] = 1
    scene = SurgicalScene(labels, np.eye(4),
                          [dict(id=1, name='Generated bone', material='bone')], {}, {})

    class Region(dict):
        def __getitem__(self, key):
            if key == 'material':
                calls.append(key)
            return super().__getitem__(key)

    outcomes = []
    for native in (False, True):
        calls = []
        engine = SurgicalEngine(scene)
        engine.by_label[1] = Region(engine.by_label[1])
        engine.state['slots']['left']['position'] = [3., 3., 3.]
        execute = backend.physical_substep if native else SurgicalEngine._physical_substep_reference
        events = execute(engine, 'left', 'inspect', np.array([3., 3., 3.]),
                         np.array([0., 0., 0., 1.]), False, .1)
        outcomes.append((engine.state, events, calls))
    assert outcomes[0] == outcomes[1]
    assert len(outcomes[0][2]) >= 3


def test_function_identity_preserves_attribute_audit_events(compiled):
    code = '''
import sys
from resectionlab import native_backend
from resectionlab.surgical_geometry_cache import _function_identity, _DEFAULT_ATOMS
def f(value=2): pass
events=[]
def observe(event,args):
    if event=='object.__getattr__' and args[0] is f:
        events.append(args[1])
sys.addaudithook(observe)
_function_identity(f)
expected=events[:]
events.clear()
native_backend._module.function_identities((f,),_DEFAULT_ATOMS)
assert events==expected, (events,expected)
assert expected==['__code__','__defaults__','__kwdefaults__']
'''
    env = dict(os.environ, PYTHONPATH=str(Path(module.__file__).resolve().parents[1]))
    subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True, text=True)

