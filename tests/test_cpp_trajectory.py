"""Exact bounded trajectory work beneath unchanged physical microsteps."""
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from resectionlab import surgical_motion as motion
from resectionlab import surgical_engine as engine_module
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_tools import TOOLS


@pytest.fixture
def compiled(monkeypatch):
    monkeypatch.setattr(motion.native_backend, 'cpp_enabled', lambda: True)


def interpolation():
    return Slerp([0, 1], Rotation.from_quat([[0., 0., 0., 1.], [.1, -.2, .3, .9]]))


def next_quaternion(slerp, index, count, chunk, rotation_type=Rotation, slerp_type=Slerp):
    value, chunk = motion._trajectory_quaternion(slerp, index, count, chunk, rotation_type, slerp_type)
    return (slerp((index + 1) / count).as_quat() if value is None else value), chunk


@pytest.mark.parametrize('count', [1, 31, 32, 33, 65, 300])
def test_chunk_rows_match_scalar_bits_and_own_their_arrays(compiled, count):
    slerp = interpolation()
    chunk = None
    rows = []
    starts = []
    for index in range(count):
        value, chunk = next_quaternion(slerp, index, count, chunk)
        expected = slerp((index + 1) / count).as_quat()
        assert value.dtype == expected.dtype and value.shape == expected.shape
        assert value.tobytes() == expected.tobytes()
        assert value.flags.owndata and value.flags.writeable
        assert len(chunk[1]) <= 32
        assert not np.shares_memory(value, chunk[1])
        starts.append(chunk[0])
        rows.append(value)
    assert sorted(set(starts)) == list(range(0, count, 32))
    if len(rows) > 1:
        before = rows[1].tobytes()
        rows[0][:] = 900.
        assert rows[1].tobytes() == before


def test_chunk_times_use_the_original_python_divisions(compiled):
    calls = []
    target = Slerp.__call__.__code__

    def observe(frame, event, arg):
        if event == 'call' and frame.f_code is target:
            calls.append(frame.f_locals['times'])

    slerp = interpolation()
    sys.setprofile(observe)
    try:
        chunk = None
        for index in range(65):
            _, chunk = next_quaternion(slerp, index, 65, chunk)
    finally:
        sys.setprofile(None)
    assert calls == [[(j + 1) / 65 for j in range(start, min(start + 32, 65))]
                     for start in (0, 32, 64)]


@pytest.mark.parametrize('kind', ['rotation_alias', 'slerp_alias', 'custom_interpolator', 'reference'])
def test_custom_dispatch_and_reference_retain_scalar_calls(compiled, monkeypatch, kind):
    slerp = interpolation()
    rotation_type, slerp_type = Rotation, Slerp
    if kind == 'rotation_alias':
        rotation_type = lambda *args: None
    elif kind == 'slerp_alias':
        slerp_type = lambda *args: None
    elif kind == 'custom_interpolator':
        class CustomSlerp(Slerp):
            pass
        slerp = CustomSlerp([0, 1], Rotation.from_quat([[0., 0., 0., 1.], [.1, -.2, .3, .9]]))
    else:
        monkeypatch.setattr(motion.native_backend, 'cpp_enabled', lambda: False)
    for index in range(2):
        value, chunk = motion._trajectory_quaternion(slerp, index, 2, None, rotation_type, slerp_type)
        assert value is None and chunk is None


def test_midchunk_hook_mutation_discards_rows_before_restoration(compiled, monkeypatch):
    slerp = interpolation()
    _, chunk = next_quaternion(slerp, 0, 33, None)
    original = Slerp.__call__
    calls = []

    def customized(self, times):
        calls.append(times)
        self.rotvecs[:] *= .5
        return original(self, times)

    monkeypatch.setattr(Slerp, '__call__', customized)
    actual, chunk = next_quaternion(slerp, 1, 33, chunk)
    assert calls == [2 / 33]
    assert chunk is None
    assert actual.tobytes() == original(slerp, 2 / 33).as_quat().tobytes()
    monkeypatch.setattr(Slerp, '__call__', original)
    actual, chunk = next_quaternion(slerp, 2, 33, chunk)
    assert chunk[0] == 2
    assert actual.tobytes() == original(slerp, 3 / 33).as_quat().tobytes()


def test_equal_but_replaced_default_identity_invalidates_chunk(compiled, monkeypatch):
    slerp = interpolation()
    _, chunk = next_quaternion(slerp, 0, 33, None)
    function = Rotation.as_quat
    defaults = tuple(list(function.__defaults__))
    assert defaults == function.__defaults__ and defaults is not function.__defaults__
    monkeypatch.setattr(function, '__defaults__', defaults)
    value, chunk = motion._trajectory_quaternion(slerp, 1, 33, chunk, Rotation, Slerp)
    assert value is None and chunk is None


def test_changed_scipy_code_is_not_hidden_by_a_prefetched_chunk(compiled, monkeypatch):
    slerp = interpolation()
    _, chunk = next_quaternion(slerp, 0, 33, None)

    def changed(self, times):
        raise RuntimeError('changed trajectory hook')

    monkeypatch.setattr(Slerp.__call__, '__code__', changed.__code__)
    with pytest.raises(RuntimeError, match='changed trajectory hook'):
        next_quaternion(slerp, 1, 33, chunk)


@pytest.mark.parametrize('mode', ['warn', 'raise', 'call'])
def test_underflow_policy_changes_use_scalar_dispatch_and_clear_chunk(compiled, mode):
    with np.errstate(under='ignore'):
        slerp = Slerp([0, 1], Rotation.from_quat([[0., 0., 0., 1.], [1e-310, 0., 0., 1.]]))
        _, chunk = next_quaternion(slerp, 0, 3, None)
    callbacks = []
    previous = np.seterrcall(lambda *args: callbacks.append(args))
    try:
        with np.errstate(under=mode):
            value, cleared = motion._trajectory_quaternion(slerp, 1, 3, chunk, Rotation, Slerp)
            assert value is None and cleared is None and callbacks == []
            if mode == 'raise':
                with pytest.raises(FloatingPointError, match='underflow'):
                    next_quaternion(slerp, 1, 3, chunk)
            else:
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter('always')
                    actual, cleared = next_quaternion(slerp, 1, 3, chunk)
                actual_callbacks = list(callbacks)
                callbacks.clear()
                with warnings.catch_warnings(record=True) as expected_warnings:
                    warnings.simplefilter('always')
                    expected = slerp(2 / 3).as_quat()
                assert actual.tobytes() == expected.tobytes()
                assert actual_callbacks == callbacks
                assert [str(row.message) for row in observed] == [str(row.message) for row in expected_warnings]
                assert cleared is None
        with np.errstate(under='ignore'):
            actual, recovered = next_quaternion(slerp, 2, 3, None)
            assert recovered[0] == 2
            assert actual.tobytes() == slerp(1.).as_quat().tobytes()
    finally:
        np.seterrcall(previous)


def test_warmed_angular_cache_preserves_underflow_errors(compiled):
    pose = {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
    end = {'position': [0., 0., 0.], 'quaternion': [.3, 1e-310, 0., float(np.sqrt(.91))]}
    parameters = {'radiusMm': 2.}
    with np.errstate(under='ignore'):
        motion.swept_envelopes(TOOLS['cranial_drill'], parameters, pose, end)
    with np.errstate(under='raise'):
        with pytest.raises(FloatingPointError, match='underflow'):
            motion._swept_envelopes_reference(TOOLS['cranial_drill'], parameters, pose, end)
        with pytest.raises(FloatingPointError, match='underflow'):
            motion.swept_envelopes(TOOLS['cranial_drill'], parameters, pose, end)


@pytest.mark.parametrize('kind', ['geterr', 'geterr_code', 'geterr_default', 'c_getter'])
def test_changed_error_policy_accessor_is_not_called(compiled, monkeypatch, kind):
    slerp = interpolation()
    _, chunk = next_quaternion(slerp, 0, 33, None)
    calls = []
    original = np.geterr
    namespace = original.__globals__

    def changed():
        calls.append(True)
        return original()

    if kind == 'geterr':
        monkeypatch.setattr(np, 'geterr', changed)
    elif kind == 'geterr_code':
        def broken():
            raise AssertionError('custom accessor was invoked')
        monkeypatch.setattr(original, '__code__', broken.__code__)
    elif kind == 'geterr_default':
        monkeypatch.setattr(original, '__defaults__', ())
    else:
        monkeypatch.setitem(namespace, '_get_extobj_dict', changed)
    value, chunk = motion._trajectory_quaternion(slerp, 1, 33, chunk, Rotation, Slerp)
    assert value is None and chunk is None and calls == []
    slerp(2 / 33).as_quat()
    assert calls == []
    pose = {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
    motion.swept_envelopes(TOOLS['cranial_drill'], {'radiusMm': 2.}, pose, pose)
    assert calls == []


@pytest.mark.parametrize('operation', ['trajectory', 'sweep'])
@pytest.mark.parametrize('owner', [Rotation, Slerp])
def test_instance_attribute_hooks_keep_the_scalar_callback_trace(compiled, monkeypatch, operation, owner):
    slerp = interpolation()
    _, chunk = next_quaternion(slerp, 0, 33, None)
    pose = {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
    parameters = {'radiusMm': 2.}
    motion.swept_envelopes(TOOLS['cranial_drill'], parameters, pose, pose)
    calls = []

    def intercepted(self, name):
        calls.append(name)
        return object.__getattribute__(self, name)

    monkeypatch.setattr(owner, '__getattribute__', intercepted, raising=False)
    if operation == 'trajectory':
        expected = slerp(2 / 33).as_quat()
        expected_calls = list(calls)
        calls.clear()
        actual, cleared = next_quaternion(slerp, 1, 33, chunk)
        assert cleared is None
        assert actual.tobytes() == expected.tobytes()
    else:
        motion._swept_envelopes_reference(TOOLS['cranial_drill'], parameters, pose, pose)
        expected_calls = list(calls)
        calls.clear()
        motion.swept_envelopes(TOOLS['cranial_drill'], parameters, pose, pose)
    assert expected_calls and calls == expected_calls


@pytest.mark.parametrize('owner', [Rotation, Slerp])
@pytest.mark.parametrize('hook', ['__getattribute__', '__getattr__', '__setattr__', '__delattr__', '__del__'])
def test_added_object_protocol_hooks_are_declined_without_invocation(compiled, monkeypatch, owner, hook):
    slerp = interpolation()
    _, chunk = next_quaternion(slerp, 0, 33, None)

    def customized(*args):
        raise AssertionError('qualification invoked an instance hook')

    with monkeypatch.context() as patch:
        patch.setattr(owner, hook, customized, raising=False)
        value, cleared = motion._trajectory_quaternion(slerp, 1, 33, chunk, Rotation, Slerp)
        assert value is None and cleared is None
    assert motion._rotation_helpers_unchanged()


@pytest.mark.parametrize('operation', ['trajectory', 'sweep'])
def test_replaced_descriptor_does_not_receive_extra_class_lookups(compiled, monkeypatch, operation):
    original = Slerp.__dict__['__call__']
    calls = []

    class Descriptor:
        def __get__(self, instance, owner):
            calls.append('class' if instance is None else 'instance')
            return original if instance is None else original.__get__(instance, owner)

    slerp = interpolation()
    _, chunk = next_quaternion(slerp, 0, 33, None)
    pose = {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
    parameters = {'radiusMm': 2.}
    motion.swept_envelopes(TOOLS['cranial_drill'], parameters, pose, pose)
    monkeypatch.setattr(Slerp, '__call__', Descriptor())
    if operation == 'trajectory':
        next_quaternion(slerp, 1, 33, chunk)
    else:
        motion.swept_envelopes(TOOLS['cranial_drill'], parameters, pose, pose)
    assert calls == ['instance']


@pytest.mark.parametrize('hook', ['descriptor', 'geterr', 'c_getter', 'binding_kind', 'attribute', 'lifecycle', 'new', 'new_late'])
def test_preimport_custom_accessors_are_not_used_by_qualification(hook):
    program = '''
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from resectionlab.surgical_tools import TOOLS
calls = []
hook = __import__('sys').argv[1]
if hook == 'descriptor':
    original = Slerp.__dict__['__call__']
    class Descriptor:
        def __get__(self, instance, owner):
            calls.append('class' if instance is None else 'instance')
            return original if instance is None else original.__get__(instance, owner)
    Slerp.__call__ = Descriptor()
elif hook == 'binding_kind':
    original = Rotation.__dict__['from_quat'].__func__
    Rotation.from_quat = classmethod(original)
elif hook == 'attribute':
    def intercepted(self, name):
        calls.append(name)
        return object.__getattribute__(self, name)
    Slerp.__getattribute__ = intercepted
elif hook == 'lifecycle':
    def intercepted(self):
        calls.append('deleted')
    Slerp.__del__ = intercepted
elif hook in ('new', 'new_late'):
    def intercepted(cls, *args):
        calls.append('created')
        return object.__new__(cls)
    if hook == 'new':
        Slerp.__new__ = intercepted
else:
    original_geterr = np.geterr
    original_c_getter = original_geterr.__globals__['_get_extobj_dict']
    def changed():
        calls.append('accessor')
        return original_geterr() if hook == 'geterr' else original_c_getter()
    if hook == 'geterr':
        np.geterr = changed
    else:
        original_geterr.__globals__['_get_extobj_dict'] = changed
from resectionlab import surgical_motion as motion
assert calls == []
motion.native_backend.cpp_enabled = lambda: True
if hook == 'binding_kind':
    assert not motion._rotation_captures_genuine
else:
    slerp = Slerp([0, 1], Rotation.from_quat([[0., 0., 0., 1.], [.1, -.2, .3, .9]]))
    if hook == 'new_late':
        Slerp.__new__ = intercepted
    calls.clear()
    value, chunk = motion._trajectory_quaternion(slerp, 0, 33, None, Rotation, Slerp)
    assert value is None and chunk is None
    assert calls == []
    slerp(1 / 33).as_quat()
    expected = list(calls)
    if hook == 'attribute':
        assert expected
    else:
        assert calls == (['instance'] if hook == 'descriptor' else [])
    calls.clear()
    value, chunk = motion._trajectory_quaternion(slerp, 1, 33, None, Rotation, Slerp)
    assert value is None and chunk is None and calls == []
    slerp(1 / 33).as_quat()
    assert calls == expected
    if hook == 'descriptor':
        pose = {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
        calls.clear()
        motion.swept_envelopes(TOOLS['cranial_drill'], {'radiusMm': 2.}, pose, pose)
        assert calls == ['instance']
print(json.dumps({'rotation': motion._rotation_captures_genuine,
                  'getter': motion._numpy_error_captures_genuine}))
'''
    root = Path(__file__).resolve().parents[1]
    child = subprocess.run([sys.executable, '-c', program, hook], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / 'src')), capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert result['rotation'] is (hook in ('geterr', 'c_getter', 'new_late'))
    assert result['getter'] is (hook not in ('geterr', 'c_getter'))


def test_preimport_scipy_hook_is_not_admitted():
    program = '''
import json
from scipy.spatial.transform import Rotation, Slerp
original = Slerp.__call__
calls = []
def customized(self, times):
    calls.append(times)
    return original(self, times)
Slerp.__call__ = customized
from resectionlab import surgical_motion as motion
motion.native_backend.cpp_enabled = lambda: True
slerp = Slerp([0, 1], Rotation.from_quat([[0., 0., 0., 1.], [.1, -.2, .3, .9]]))
chunk = None
for index in range(2):
    value, chunk = motion._trajectory_quaternion(slerp, index, 2, chunk, Rotation, Slerp)
    assert value is None and chunk is None
    slerp((index + 1) / 2).as_quat()
print(json.dumps(calls))
'''
    root = Path(__file__).resolve().parents[1]
    child = subprocess.run([sys.executable, '-c', program], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / 'src')), capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == [.5, 1.]


def test_real_engine_keeps_each_check_before_each_physical_substep():
    events = []

    class RecordingEngine(SurgicalEngine):
        def _substep(self, key, operation, position, quaternion, active, duration):
            events.append(('substep', quaternion.tobytes(), duration))
            return super()._substep(key, operation, position, quaternion, active, duration)

    scene = SurgicalScene(np.zeros((8, 8, 8), np.uint16), np.eye(4), [], {}, {})
    engine = RecordingEngine(scene)
    result = engine._execute({'slot': 'left', 'operation': 'inspect'}, 3.3, [],
                             check=lambda: events.append(('check',)))
    assert result['status'] == 'accepted'
    assert [event[0] for event in events] == ['check', 'substep'] * 33
    expected = Slerp([0, 1], Rotation.from_quat([[0., 0., 0., 1.]] * 2))
    for index, event in enumerate(events[1::2]):
        assert event[1] == expected((index + 1) / 33).as_quat().tobytes()
        assert event[2] == 3.3 / 33
    elapsed = 0.
    for _ in range(33):
        elapsed += 3.3 / 33
    assert engine.state['elapsedSeconds'] == elapsed


def test_real_engine_callback_can_replace_and_restore_the_interpolator(monkeypatch):
    events, scalar_hooks, batch_sizes = [], [], []
    original = Slerp.__call__
    selected = motion.native_backend.cpp_enabled()

    def customized(self, times):
        if sys._getframe(1).f_code is SurgicalEngine._execute.__code__:
            scalar_hooks.append(times)
        return original(self, times)

    def observe(frame, event, arg):
        if (event == 'call' and frame.f_code is original.__code__
                and frame.f_back.f_code is motion._trajectory_quaternion.__code__):
            batch_sizes.append(len(frame.f_locals['times']))

    class RecordingEngine(SurgicalEngine):
        def _substep(self, key, operation, position, quaternion, active, duration):
            events.append('substep')
            return super()._substep(key, operation, position, quaternion, active, duration)

    checks = 0

    def check():
        nonlocal checks
        checks += 1
        events.append('check')
        if checks == 2:
            monkeypatch.setattr(Slerp, '__call__', customized)
        elif checks == 3:
            monkeypatch.setattr(Slerp, '__call__', original)

    engine = RecordingEngine(SurgicalScene(np.zeros((8, 8, 8), np.uint16), np.eye(4), [], {}, {}))
    sys.setprofile(observe)
    try:
        result = engine._execute({'slot': 'left', 'operation': 'inspect'}, 3.3, [], check=check)
    finally:
        sys.setprofile(None)
    assert result['status'] == 'accepted'
    assert events == ['check', 'substep'] * 33
    assert scalar_hooks == [2 / 33]
    assert batch_sizes == ([32, 31] if selected else [])


def test_live_engine_constructor_alias_is_rechecked_after_the_callback(compiled, monkeypatch):
    slerp = interpolation()
    _, chunk = next_quaternion(slerp, 0, 33, None)
    monkeypatch.setattr(engine_module, 'Slerp', lambda *args: None)
    value, chunk = motion._trajectory_quaternion(
        slerp, 1, 33, chunk, engine_module.Rotation, engine_module.Slerp)
    assert value is None and chunk is None


def test_real_proof_epoch_does_not_depend_on_trajectory_buffer():
    from resectionlab.surgical_proof_cache import implementation_epoch
    from resectionlab.surgical_verified_cache import implementation_fingerprint
    before = implementation_epoch()
    assert before is not None
    slerp = interpolation()
    chunk = None
    for index in range(33):
        _, chunk = next_quaternion(slerp, index, 33, chunk)
    assert implementation_epoch() == before
    assert implementation_fingerprint() is not None
