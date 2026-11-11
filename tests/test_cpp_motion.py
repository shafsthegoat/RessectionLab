"""Exact swept geometry with bounded reuse of SciPy angular calculations."""
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from resectionlab import surgical_motion as motion
from resectionlab.surgical_tools import TOOLS


@pytest.fixture(autouse=True)
def compiled_selection(monkeypatch):
    monkeypatch.setattr(motion.native_backend, 'cpp_enabled', lambda: True)
    motion._sweep_rotation_cache.clear()
    yield
    motion._sweep_rotation_cache.clear()


def assert_exact(first, second):
    assert len(first) == len(second)
    for left, right in zip(first, second):
        for a, b in zip(left[:2], right[:2]):
            assert a.dtype == b.dtype
            assert a.shape == b.shape
            assert a.tobytes() == b.tobytes()
        assert np.float64(left[2]).tobytes() == np.float64(right[2]).tobytes()
        assert left[3] == right[3]


@pytest.mark.parametrize('name', list(TOOLS))
def test_sweeps_match_original_for_every_instrument(name):
    rng = np.random.default_rng(179)
    tool = TOOLS[name]
    for revision in (1, 2):
        parameters = {'radiusMm': tool['radiusMm'], 'depthMm': 2.75,
                      'jawOpeningMm': 1.3, 'contactGeometryRevision': revision,
                      'retractorFaceContactRevision': 1}
        for _ in range(5):
            start = {'position': rng.normal(size=3), 'quaternion': rng.normal(size=4).tolist()}
            end = {'position': rng.normal(size=3), 'quaternion': rng.normal(size=4)}
            expected = motion._swept_envelopes_reference(tool, parameters, start, end)
            assert_exact(motion.swept_envelopes(tool, parameters, start, end), expected)
            assert_exact(motion.swept_envelopes(tool, parameters, start, end), expected)
    assert len(motion._sweep_rotation_cache) == 10


def test_cache_shares_only_angular_work_and_detaches_outputs():
    tool = TOOLS['cusa']
    parameters = {'radiusMm': 1.5, 'depthMm': 2.}
    start = {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
    end = {'position': np.array([1., 2., 3.]), 'quaternion': np.array([0., 0., 0., 1.])}
    result = motion.swept_envelopes(tool, parameters, start, end)
    for a, b, _, _ in result:
        a[:] = 900.
        b[:] = -900.
    result.clear()
    end['position'][0] = 4.
    parameters['radiusMm'] = 2.
    tool = dict(tool, shaftRadiusMm=.75)
    expected = motion._swept_envelopes_reference(tool, parameters, start, end)
    assert_exact(motion.swept_envelopes(tool, parameters, start, end), expected)
    assert len(motion._sweep_rotation_cache) == 1


def test_cache_key_preserves_quaternion_bits_and_bounds_entries():
    start = {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
    end = copy.deepcopy(start)
    parameters = {'radiusMm': 2.}
    for first in (0., -0.):
        end['quaternion'][0] = first
        motion.swept_envelopes(TOOLS['cranial_drill'], parameters, start, end)
    assert len(motion._sweep_rotation_cache) == 2
    for angle in np.linspace(.01, 1., 100):
        end['quaternion'] = Rotation.from_rotvec([angle, 0., 0.]).as_quat()
        motion.swept_envelopes(TOOLS['cranial_drill'], parameters, start, end)
    assert len(motion._sweep_rotation_cache) == 64


@pytest.mark.parametrize('kind', ['subclass', 'float32', 'mapping', 'position_subclass'])
def test_unusual_inputs_retain_original_path(kind):
    class Quaternion(list):
        pass

    class Pose(dict):
        pass

    start = {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
    end = copy.deepcopy(start)
    if kind == 'subclass':
        end['quaternion'] = Quaternion(end['quaternion'])
    elif kind == 'float32':
        end['quaternion'] = np.asarray(end['quaternion'], np.float32)
    elif kind == 'position_subclass':
        end['position'] = Quaternion(end['position'])
    else:
        end = Pose(end)
    parameters = {'radiusMm': 2.}
    assert_exact(motion.swept_envelopes(TOOLS['cranial_drill'], parameters, start, end),
                 motion._swept_envelopes_reference(TOOLS['cranial_drill'], parameters, start, end))
    assert len(motion._sweep_rotation_cache) == 0


@pytest.mark.parametrize('mutation', ['replace', 'code', 'keyword_default'])
def test_scipy_hook_changes_bypass_previously_cached_results(monkeypatch, mutation):
    start = {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
    end = copy.deepcopy(start)
    parameters = {'radiusMm': 2.}
    motion.swept_envelopes(TOOLS['cranial_drill'], parameters, start, end)
    if mutation == 'keyword_default':
        monkeypatch.setitem(Rotation.from_quat.__kwdefaults__, 'scalar_first', True)
        expected = motion._swept_envelopes_reference(TOOLS['cranial_drill'], parameters, start, end)
        assert_exact(motion.swept_envelopes(TOOLS['cranial_drill'], parameters, start, end), expected)
    else:
        def changed(self, times):
            raise RuntimeError('changed SciPy hook was called')
        if mutation == 'replace':
            monkeypatch.setattr(Slerp, '__call__', changed)
        else:
            monkeypatch.setattr(Slerp.__call__, '__code__', changed.__code__)
        with pytest.raises(RuntimeError, match='changed SciPy hook was called'):
            motion.swept_envelopes(TOOLS['cranial_drill'], parameters, start, end)
    assert len(motion._sweep_rotation_cache) == 1


def test_reference_selection_uses_original_without_cache(monkeypatch):
    monkeypatch.setattr(motion.native_backend, 'cpp_enabled', lambda: False)
    pose = {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
    parameters = {'radiusMm': 2.}
    expected = motion._swept_envelopes_reference(TOOLS['cranial_drill'], parameters, pose, pose)
    for _ in range(2):
        assert_exact(motion.swept_envelopes(TOOLS['cranial_drill'], parameters, pose, pose), expected)
    assert len(motion._sweep_rotation_cache) == 0


def test_cache_preserves_real_implementation_qualification(monkeypatch):
    monkeypatch.undo()
    if not motion.native_backend.cpp_enabled():
        pytest.skip('real compiled backend required')
    from resectionlab.surgical_proof_cache import implementation_epoch
    from resectionlab.surgical_verified_cache import implementation_fingerprint
    before = implementation_epoch()
    assert before is not None
    pose = {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
    parameters = {'radiusMm': 2.}
    motion.swept_envelopes(TOOLS['cranial_drill'], parameters, pose, pose)
    assert len(motion._sweep_rotation_cache) == 1
    assert implementation_epoch() == before
    assert implementation_fingerprint() is not None
