"""Fused native capsule construction retains the scalar geometry byte for byte."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab import surgical_motion as motion
from resectionlab.surgical_tools import TOOLS


@pytest.fixture(autouse=True)
def native_selection():
    if not motion.native_backend.cpp_enabled():
        pytest.skip('compiled backend required')
    assert motion._envelope_helpers_unchanged()


def assert_exact(actual, expected):
    assert len(actual) == len(expected)
    for first, second in zip(actual, expected):
        for left, right in zip(first[:2], second[:2]):
            assert type(left) is type(right)
            assert left.dtype == right.dtype and left.shape == right.shape
            assert left.tobytes() == right.tobytes()
        assert type(first[2]) is type(second[2])
        assert np.float64(first[2]).tobytes() == np.float64(second[2]).tobytes()
        assert first[3] == second[3]


@pytest.mark.parametrize('name', list(TOOLS))
def test_all_instruments_match_scalar_capsules(name):
    rng = np.random.default_rng(361)
    tool = TOOLS[name]
    for revision in (1, 2):
        parameters = {'radiusMm': tool['radiusMm'], 'depthMm': 2.75,
                      'jawOpeningMm': 1.3, 'contactGeometryRevision': revision}
        for inflation in (0., .125, np.float64(.125)):
            for _ in range(8):
                position, quaternion = rng.normal(size=3), rng.normal(size=4)
                expected = motion._envelopes_reference(tool, parameters, position, quaternion, inflation)
                assert_exact(motion.envelopes(tool, parameters, position, quaternion, inflation), expected)


@pytest.mark.parametrize('quaternion', [[0., 0., 0., 1.], [-0., 0., -0., -1.],
                                       [1., 0., 0., 0.], [0., 0., 1., 1.]])
@pytest.mark.parametrize('depth', [0., -0., -2., 1., 5., 6.])
def test_signed_zero_and_clipped_depth_are_exact(quaternion, depth):
    for name in ('cranial_drill', 'cusa', 'bipolar_forceps', 'scalpel'):
        tool = TOOLS[name]
        parameters = {'radiusMm': tool['radiusMm'], 'depthMm': depth, 'jawOpeningMm': -0.}
        position = [-0., 0., -0.]
        assert_exact(motion.envelopes(tool, parameters, position, quaternion),
                     motion._envelopes_reference(tool, parameters, position, quaternion))


def test_native_construction_runs_without_scalar_apply_calls():
    tool = TOOLS['cusa']
    parameters = {'radiusMm': 1.5, 'contactGeometryRevision': 2}
    position, quaternion = [1., 2., 3.], [0., .2, 0., .9]
    expected = motion._envelopes_reference(tool, parameters, position, quaternion)
    before = motion.native_backend._counters['motionEnvelopeCalls']
    actual = motion.envelopes(tool, parameters, position, quaternion)
    assert motion.native_backend._counters['motionEnvelopeCalls'] == before + 1
    assert_exact(actual, expected)
    for first, second in zip(actual, expected):
        assert first[0].flags.owndata and first[1].flags.owndata
        first[0][:] = 123.
        first[1][:] = -123.
    assert_exact(motion.envelopes(tool, parameters, position, quaternion), expected)


@pytest.mark.parametrize('kind', ['apply', 'matrix', 'selector', 'numpy_array', 'device'])
def test_changed_helpers_retain_original_callback_trace(monkeypatch, kind):
    import scipy.spatial.transform._rotation as owner
    calls = []
    target, attribute = {
        'apply': (Rotation, 'apply'), 'matrix': (Rotation, 'as_matrix'),
        'selector': (owner, 'select_backend'), 'numpy_array': (np, 'array'),
        'device': (owner, 'xp_device'),
    }[kind]
    original = getattr(target, attribute)

    def changed(*args, **kwargs):
        calls.append(attribute)
        return original(*args, **kwargs)

    monkeypatch.setattr(target, attribute, changed)
    tool, parameters = TOOLS['cusa'], {'radiusMm': 1.5}
    position, quaternion = [0., 0., 0.], [0., 0., 0., 1.]
    expected = motion._envelopes_reference(tool, parameters, position, quaternion)
    expected_calls = list(calls)
    calls.clear()
    actual = motion.envelopes(tool, parameters, position, quaternion)
    assert calls == expected_calls
    assert_exact(actual, expected)


def test_custom_metadata_and_finite_pad_keep_reference_path():
    calls = []

    class Parameters(dict):
        def __getitem__(self, key):
            calls.append(key)
            return super().__getitem__(key)

    for name, parameters in [('cusa', Parameters(radiusMm=1.5)),
                             ('scalpel', {'radiusMm': 1., 'retractorFaceContactRevision': 1})]:
        calls.clear()
        tool = TOOLS[name]
        position, quaternion = [0., 0., 0.], [0., 0., 0., 1.]
        expected = motion._envelopes_reference(tool, parameters, position, quaternion)
        expected_calls = list(calls)
        calls.clear()
        actual = motion.envelopes(tool, parameters, position, quaternion)
        assert calls == expected_calls
        assert_exact(actual, expected)


@pytest.mark.parametrize('mode', ['warn', 'raise', 'call'])
def test_nondefault_underflow_policy_uses_original_helper(mode):
    tool, parameters = TOOLS['cusa'], {'radiusMm': 1.5}
    before = motion.native_backend._counters['motionEnvelopeCalls']
    with np.errstate(under=mode):
        actual = motion.envelopes(tool, parameters, [0., 0., 0.], [0., 0., 0., 1.])
        expected = motion._envelopes_reference(tool, parameters, [0., 0., 0.], [0., 0., 0., 1.])
    assert motion.native_backend._counters['motionEnvelopeCalls'] == before
    assert_exact(actual, expected)


@pytest.mark.parametrize('case', ['scalar_subclass', 'finite_pad', 'nonfinite_position', 'nonfinite_quaternion'])
def test_unsupported_inputs_normalize_exactly_once(case):
    tool, parameters = TOOLS['cusa'], {'radiusMm': 1.5}
    position, quaternion = [0., 0., 0.], [0., 0., 0., 1.]
    if case == 'scalar_subclass':
        parameters['radiusMm'] = np.float64(1.5)
    elif case == 'finite_pad':
        tool = TOOLS['scalpel']
        parameters['retractorFaceContactRevision'] = 1
    elif case == 'nonfinite_position':
        position[0] = float('inf')
    else:
        quaternion[0] = float('inf')
    target = Rotation.__dict__['from_quat'].__func__.__code__
    calls = []

    def observed(frame, event, argument):
        if event == 'call' and frame.f_code is target:
            calls.append(True)

    sys.setprofile(observed)
    try:
        actual = motion.envelopes(tool, parameters, position, quaternion)
    finally:
        sys.setprofile(None)
    assert len(calls) == 1
    assert_exact(actual, motion._envelopes_reference(tool, parameters, position, quaternion))


def test_preimport_apply_descriptor_is_not_called_by_admission():
    program = '''
from scipy.spatial.transform import Rotation
original = Rotation.__dict__['apply']
calls = []
class Descriptor:
    def __get__(self, instance, owner):
        calls.append('class' if instance is None else 'instance')
        return original if instance is None else original.__get__(instance, owner)
Rotation.apply = Descriptor()
from resectionlab import surgical_motion as motion
assert calls == []
assert motion._ENVELOPE_HELPERS is None
from resectionlab.surgical_tools import TOOLS
arguments = (TOOLS['cusa'], {'radiusMm': 1.5}, [0., 0., 0.], [0., 0., 0., 1.])
motion._envelopes_reference(*arguments)
expected = list(calls)
calls.clear()
motion.envelopes(*arguments)
assert calls == expected and calls
'''
    root = Path(__file__).resolve().parents[1]
    child = subprocess.run([sys.executable, '-c', program], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / 'src'), RESSECTIONLAB_NUMERICAL_BACKEND='reference'),
        text=True, capture_output=True, timeout=20)
    assert child.returncode == 0, child.stderr


@pytest.mark.parametrize('kind', ['device', 'numpy_predicate', 'atleast', 'asarray', 'numpy_atleast'])
def test_preimport_callable_metadata_hooks_are_not_used(kind):
    program = '''
import sys
import numpy as np
import scipy.spatial.transform._rotation as owner
calls = []
kind = sys.argv[1]
target, name = {'device': (owner, 'xp_device'), 'numpy_predicate': (owner, 'is_numpy'),
    'atleast': (owner.xpx, 'atleast_nd'), 'asarray': (next(iter(owner.backend_registry)), 'asarray'),
    'numpy_atleast': (np, 'atleast_2d')}[kind]
original = vars(target)[name]
class Callable:
    def __getattribute__(self, name):
        calls.append(name)
        return object.__getattribute__(self, name)
    def __call__(self, *args, **kwargs):
        calls.append('called')
        return original(*args, **kwargs)
setattr(target, name, Callable())
from resectionlab import surgical_motion as motion
assert calls == []
assert motion._ENVELOPE_HELPERS is None
from resectionlab.surgical_tools import TOOLS
arguments = (TOOLS['cusa'], {'radiusMm': 1.5}, [0., 0., 0.], [0., 0., 0., 1.])
motion._envelopes_reference(*arguments)
expected = list(calls)
calls.clear()
motion.envelopes(*arguments)
assert calls == expected
'''
    root = Path(__file__).resolve().parents[1]
    child = subprocess.run([sys.executable, '-c', program, kind], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / 'src'), RESSECTIONLAB_NUMERICAL_BACKEND='reference'),
        text=True, capture_output=True, timeout=20)
    assert child.returncode == 0, child.stderr
