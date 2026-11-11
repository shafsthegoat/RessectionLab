"""Ordered contact batching beneath unchanged whole-cell guards."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

from resectionlab import geometry, native_backend, surgical_motion as motion
from resectionlab import surgical_bone_transport as bone


def capsules(distance=10., count=2):
    return [(np.array([distance + index, 0., 0.]), np.array([distance + index, 0., 0.]),
             .5, 'shaft' if index == 0 else 'working') for index in range(count)]


def scalar(first, second, predicate=None):
    predicate = motion.segments_within_distance if predicate is None else predicate
    return any(predicate(a, b, c, d, ar + br + 1e-9)
               for a, b, ar, _ in first for c, d, br, _ in second)


def outcome(operation, *args):
    try:
        return ('value', operation(*args))
    except Exception as error:
        return (type(error).__name__, str(error))


def test_real_batch_preserves_order_typed_thresholds_and_owned_packing():
    first, second = capsules(0., 2), capsules(10., 3)
    first = [(a, b, np.float64(width), part) for a, b, width, part in first]
    captured = []
    code = native_backend.try_segment_batch.__code__

    def observe(frame, event, arg):
        if event == 'call' and frame.f_code is code:
            captured.append((frame.f_locals['endpoints'], frame.f_locals['thresholds']))

    before = dict(native_backend._counters)
    sys.setprofile(observe)
    try:
        assert motion.pair_conflict(first, second) is scalar(first, second) is False
    finally:
        sys.setprofile(None)
    if native_backend._module is None:
        assert captured == []
    else:
        assert len(captured) == 1
        points, thresholds = captured[0]
        assert points.flags.owndata and thresholds.flags.owndata
        expected = [(a, b, c, d) for a, b, _, _ in first for c, d, _, _ in second]
        assert points.tobytes() == np.array(expected, dtype=np.float64).tobytes()
        expected = [ar + br + 1e-9 for _, _, ar, _ in first for _, _, br, _ in second]
        assert thresholds.tobytes() == np.array(expected, dtype=np.float64).tobytes()
        assert native_backend._counters['segmentBatchCalls'] - before['segmentBatchCalls'] == 1
        assert native_backend._counters['segmentBatchExamined'] - before['segmentBatchExamined'] == 6


@pytest.mark.parametrize('kind', ['malformed', 'subclass', 'float32', 'byteorder', 'shape',
                                 'width', 'overflow-width', 'nan', 'overflow-point'])
def test_earlier_hit_keeps_later_invalid_input_unobserved(kind):
    first, second = capsules(0., 1), capsules(.2, 2)
    calls = []

    class Width:
        def __add__(self, other):
            calls.append('width')
            raise RuntimeError('later width')

    class Array(np.ndarray):
        def __array__(self, *args, **kwargs):
            calls.append('array')
            return super().__array__(*args, **kwargs)

    a, b, width, part = second[1]
    if kind == 'malformed': second[1] = (a, b, width)
    elif kind == 'subclass': second[1] = (a.view(Array), b, width, part)
    elif kind == 'float32': second[1] = (a.astype(np.float32), b, width, part)
    elif kind == 'byteorder': second[1] = (a.astype('>f8'), b, width, part)
    elif kind == 'shape': second[1] = (a[:2], b, width, part)
    elif kind == 'width': second[1] = (a, b, Width(), part)
    elif kind == 'overflow-width': second[1] = (a, b, np.float64(1e308), part)
    elif kind == 'nan': second[1] = (np.array([np.nan, 0., 0.]), b, width, part)
    else: second[1] = (np.array([1e308, 0., 0.]), b, width, part)
    with np.errstate(over='raise', invalid='raise'):
        assert scalar(first, second) is motion.pair_conflict(first, second) is True
    assert calls == []


def test_earlier_hit_skips_a_later_numpy_threshold_overflow():
    first, second = capsules(0., 2), capsules(.2, 2)
    first[1] = (*first[1][:2], np.float64(1e308), 'working')
    second[0] = (*second[0][:2], np.float64(1e308), 'working')
    with np.errstate(over='raise'):
        assert scalar(first, second) is motion.pair_conflict(first, second) is True


def test_consumed_nonfinite_errors_and_invalid_epsilon_keep_scalar_priority(monkeypatch):
    first, second = capsules(0., 1), capsules(10., 2)
    second[1] = (np.array([np.nan, 0., 0.]), second[1][1], .5, 'working')
    assert outcome(motion.pair_conflict, first, second) == outcome(scalar, first, second)
    for eps in (0., float('nan'), np.float64(1e-9), object()):
        with monkeypatch.context() as patch:
            patch.setattr(geometry, '_EPS', eps)
            assert motion._try_capsule_contact(first, second, motion.segments_within_distance) is None
            assert outcome(motion.pair_conflict, first, second) == outcome(scalar, first, second)


def test_ambiguity_replays_original_loop_without_extra_reference_hook_calls(monkeypatch):
    first, second = capsules(0., 1), capsules(10., 3)
    second[1] = (np.array([1.000000001, 0., 0.]), np.array([1.000000001, 0., 0.]), .5, 'working')
    second[2] = capsules(.2, 1)[0]
    original = np.asarray
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(type(args[0]).__name__)
        return original(*args, **kwargs)

    monkeypatch.setattr(np, 'asarray', wrapped)
    expected = scalar(first, second)
    expected_calls = list(calls)
    calls.clear()
    assert motion.pair_conflict(first, second) == expected
    assert calls == expected_calls and calls


@pytest.mark.parametrize('owner,name', [(geometry, 'segments_within_distance'),
    (geometry, 'segment_segment_distance'), (geometry, 'point_segment_distances'),
    (native_backend, 'try_segment_predicate'), (native_backend, 'try_segment_batch'),
    (native_backend, '_check_live'), (native_backend.math, 'isfinite')])
def test_named_helper_replacements_keep_original_callback_trace(monkeypatch, owner, name):
    original = getattr(owner, name)
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapped)
    first, second = capsules(0., 1), capsules(10., 2)
    expected = scalar(first, second)
    expected_calls = list(calls)
    calls.clear()
    assert motion.pair_conflict(first, second) == expected
    assert calls == expected_calls


@pytest.mark.parametrize('attribute,replacement', [('__defaults__', ()), ('__kwdefaults__', {})])
@pytest.mark.parametrize('owner,name', [(geometry, 'segments_within_distance'),
    (geometry, 'segment_segment_distance'), (geometry, 'point_segment_distances'),
    (native_backend, 'try_segment_predicate'), (native_backend, 'try_segment_batch'),
    (native_backend, '_check_live')])
def test_in_place_helper_default_mutations_decline_batch(monkeypatch, attribute, replacement, owner, name):
    monkeypatch.setattr(getattr(owner, name), attribute, replacement)
    first, second = capsules(0., 1), capsules(10., 2)
    assert motion._try_capsule_contact(first, second, motion.segments_within_distance) is None
    assert outcome(motion.pair_conflict, first, second) == outcome(scalar, first, second)


def test_in_place_predicate_code_mutation_is_called(monkeypatch):
    def changed(*args):
        return True

    monkeypatch.setattr(geometry.segments_within_distance, '__code__', changed.__code__)
    first, second = capsules(0., 1), capsules(10., 2)
    assert motion.pair_conflict(first, second) is True


@pytest.mark.parametrize('binding', ['array', 'float64', 'dtype'])
def test_numpy_replacements_keep_scalar_result_error_and_hook_trace(monkeypatch, binding):
    first, second = capsules(0., 1), capsules(10., 2)
    original = getattr(np, binding)
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(binding)
        return original(*args, **kwargs)

    monkeypatch.setattr(np, binding, wrapped)
    expected = outcome(scalar, first, second)
    expected_calls = list(calls)
    calls.clear()
    assert outcome(motion.pair_conflict, first, second) == expected
    assert calls == expected_calls


@pytest.mark.parametrize('copied_builtins', [False, True])
def test_pair_any_hook_still_receives_its_original_generator(monkeypatch, copied_builtins):
    original = motion._CONTACT_ANY
    calls = []

    def wrapped(values):
        calls.append(type(values).__name__)
        return original(values)

    operation = motion.pair_conflict
    if copied_builtins:
        namespace = dict(operation.__globals__)
        namespace['__builtins__'] = dict(operation.__builtins__, any=wrapped)
        operation = FunctionType(operation.__code__, namespace)
        namespace['pair_conflict'] = operation
    else:
        monkeypatch.setattr(motion, 'any', wrapped, raising=False)
    assert operation(capsules(0., 1), capsules(10., 2)) is False
    assert calls == ['generator']


def test_saved_pair_function_remains_callable_through_a_public_wrapper(monkeypatch):
    original = motion.pair_conflict

    class Wrapper:
        def __call__(self, first, second):
            return original(first, second)

    monkeypatch.setattr(motion, 'pair_conflict', Wrapper())
    assert motion.pair_conflict([], []) is False
    assert motion.pair_conflict(capsules(0., 1), capsules(10., 2)) is False


def test_replaced_geometry_backend_keeps_its_predicate_callbacks(monkeypatch):
    calls = []

    def changed(*args):
        calls.append('predicate')
        return 0

    monkeypatch.setattr(geometry, 'native_backend', SimpleNamespace(try_segment_predicate=changed))
    first, second = capsules(0., 1), capsules(10., 2)
    assert scalar(first, second) is False
    expected = list(calls)
    calls.clear()
    assert motion.pair_conflict(first, second) is False
    assert calls == expected


def test_replaced_math_owner_does_not_receive_new_accessor_callbacks(monkeypatch):
    original = native_backend.math.isfinite
    calls = []

    class Math:
        @property
        def isfinite(self):
            calls.append('accessor')
            return original

    monkeypatch.setattr(native_backend, 'math', Math())
    assert motion.pair_conflict(capsules(0., 1), capsules(10., 2)) is False
    assert calls == []


@pytest.mark.parametrize('name', ['type', 'float', 'len'])
def test_new_adapter_builtin_names_do_not_add_callback_dispatch(monkeypatch, name):
    original = dict(motion._CONTACT_ADAPTER_BUILTINS)[name]
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(native_backend, name, wrapped, raising=False)
    first, second = capsules(0., 1), capsules(10., 2)
    expected = scalar(first, second)
    expected_calls = list(calls)
    calls.clear()
    assert motion.pair_conflict(first, second) == expected
    assert calls == expected_calls == []


@pytest.mark.parametrize('hook', ['predicate', 'array', 'float64', 'dtype', 'isfinite', 'math_owner',
                                 'adapter_type', 'adapter_float', 'adapter_len'])
def test_preimport_hooks_cannot_be_captured_as_stock(hook):
    program = r'''
import builtins,json,sys
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from resectionlab import geometry,native_backend
a,b=np.array([0.,0.,0.]),np.array([10.,0.,0.])
first,second=[(a,a,.5,'shaft')],[(b,b,.5,'shaft')]*2
hook=sys.argv[1]
calls=[]
if hook=='predicate':
    def changed(*args):
        calls.append('predicate')
        return False
    geometry.segments_within_distance=changed
elif hook.startswith('adapter_'):
    name=hook.removeprefix('adapter_')
    original=getattr(builtins,name)
    def changed(*args,**kwargs):
        calls.append(hook)
        return original(*args,**kwargs)
    setattr(native_backend,name,changed)
elif hook=='math_owner':
    original=native_backend.math.isfinite
    class Math:
        @property
        def isfinite(self):
            calls.append('accessor')
            return original
    native_backend.math=Math()
else:
    owner=native_backend.math if hook=='isfinite' else np
    original=getattr(owner,hook)
    def changed(*args,**kwargs):
        calls.append(hook)
        return original(*args,**kwargs)
    setattr(owner,hook,changed)
from resectionlab import surgical_motion as motion
assert motion._CONTACT_DTYPE is None
calls.clear()
def scalar():
    return any(motion.segments_within_distance(a,b,c,d,ar+br+1e-9)
        for a,b,ar,_ in first for c,d,br,_ in second)
def outcome(operation):
    try:return ['value',operation()]
    except Exception as error:return [type(error).__name__,str(error)]
expected=outcome(scalar);expected_calls=list(calls);calls.clear()
actual=outcome(lambda:motion.pair_conflict(first,second))
assert actual==expected and calls==expected_calls
print(json.dumps({'outcome':actual,'trace':calls}))
'''
    root = Path(__file__).resolve().parents[1]
    child = subprocess.run([sys.executable, '-c', program, hook], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / 'src')), capture_output=True, text=True, timeout=20)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)


def test_held_contacts_keep_hand_order_and_grip_exempts_only_working_parts():
    point = np.array([0., 0., 0.])
    far, hit = capsules(10., 1)[0], capsules(.2, 1)[0]
    working = (*hit[:3], 'working')
    source = 17
    choices = {'left': [far, working], 'right': [far, far]}
    expected = False if native_backend._module is not None else None
    assert motion._held_cell_contact(point, point, .5, choices, 'left', source, {source}, motion.segments_within_distance) is expected
    choices['left'][0] = hit
    assert motion._held_cell_contact(point, point, .5, choices, 'left', source, {source}, motion.segments_within_distance) == ('left' if expected is False else None)
    choices = {'right': [hit, far], 'left': [hit]}
    assert motion._held_cell_contact(point, point, .5, choices, 'left', source, {source}, motion.segments_within_distance) == ('right' if expected is False else None)


@pytest.fixture(scope='module')
def held_bone():
    from test_surgical_bone_transport import shell, cut_shell, grasp_shell
    engine, contour, _, _ = shell()
    cut_shell(engine, contour)
    grasp_shell(engine)
    return engine


def test_real_held_cell_sequence_keeps_guard_order_and_32_cell_cancellation(held_bone, monkeypatch):
    engine = held_bone.clone()
    events, cells = [], []
    original_cells = engine.cells

    def queried(*args, **kwargs):
        events.append('occupancy')
        return original_cells(*args, **kwargs)

    def protected(*args, **kwargs):
        events.append('protection')
        return []

    def observe(frame, event, arg):
        if event == 'call' and frame.f_code is motion._held_cell_contact.__code__:
            events.append('tools'); cells.append(frame.f_locals['source_id'])

    boundaries = []
    failure = ValueError('cancel at cell 32')

    def check():
        if sys._getframe(1).f_code is bone.check_sweep.__code__:
            boundaries.append(len(cells))
            if len(boundaries) == 2:
                raise failure

    monkeypatch.setattr(engine, 'cells', queried)
    monkeypatch.setattr(engine, 'protection', SimpleNamespace(capsule_hits=protected))
    monkeypatch.setattr(engine, '_transition_check', check, raising=False)
    before = engine.snapshot()
    sys.setprofile(observe)
    try:
        with pytest.raises(ValueError) as observed:
            bone.check_sweep(engine, 'left', engine.state['holds']['left']['offset'], engine.state['slots']['left'])
    finally:
        sys.setprofile(None)
    assert observed.value is failure
    assert boundaries == [0, 32]
    assert events == ['protection', 'occupancy', 'tools'] * 32
    assert engine.snapshot() == before


@pytest.mark.parametrize('failure', ['protection', 'occupancy'])
def test_real_held_guards_refuse_before_tool_queries(held_bone, monkeypatch, failure):
    engine = held_bone.clone()
    calls = []
    monkeypatch.setattr(bone, '_held_cell_contact', lambda *args: calls.append('tools'))
    if failure == 'protection':
        monkeypatch.setattr(engine, 'protection', SimpleNamespace(capsule_hits=lambda *args, **kwargs: [1]))
        reason = 'PROTECTED_STRUCTURE'
    else:
        anchor = engine._cranial_access.bone_anchor_cells[0]
        monkeypatch.setattr(engine, 'cells', lambda *args, **kwargs: [anchor])
        reason = 'BONE_FLAP_SWEPT_ANATOMY_COLLISION'
    before = engine.snapshot()
    with pytest.raises(ValueError, match=reason):
        bone.check_sweep(engine, 'left', engine.state['holds']['left']['offset'], engine.state['slots']['left'])
    assert calls == [] and engine.snapshot() == before


def test_contact_work_preserves_qualified_proof_epoch():
    from resectionlab.surgical_proof_cache import implementation_epoch
    from resectionlab.surgical_verified_cache import implementation_fingerprint
    before = implementation_epoch()
    assert before is not None and implementation_fingerprint() is not None
    assert motion.pair_conflict(capsules(0., 1), capsules(10., 2)) is False
    assert implementation_epoch() == before
