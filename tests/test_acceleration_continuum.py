"""Opt-in continuum fast paths: exact paths must be bit-identical to reference."""
import copy
import json

import numpy as np
import pytest

from resectionlab import acceleration
from resectionlab import accel_continuum
from resectionlab import surgical_continuum as continuum
from resectionlab import surgical_deformation_v2 as brain
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from test_surgical_continuum import block, material


def mixed(n=5, affine=None):
    """Two soft materials with different branches and an oblique affine."""
    source = block(n, affine, branches=(500., 200.), times=(.5, 2.))
    labels = source.labels.copy(); labels[: n // 2] = 2
    properties = copy.deepcopy(source.properties)
    properties[continuum.PROPERTY]['materials']['2'] = material(mu=700., nu=.45, branches=(100., 50.))
    regions = [*source.regions, {'id': 2, 'name': 'Generated second block', 'material': 'tumor'}]
    return SurgicalScene(labels, source.affine, regions, source.source, properties)


AFFINE = np.array([[1., .2, 0., 4.], [0., 1.1, .3, -2.], [.1, 0., .8, 7.], [0., 0., 0., 1.]])


@pytest.mark.parametrize('affine', [None, AFFINE])
def test_vectorized_assembly_is_bit_identical(affine):
    scene = mixed(affine=affine)
    mesh = continuum.voxel_mesh(scene, range(scene.labels.size))
    materials = scene.properties[continuum.PROPERTY]['materials']
    expected, expected_gradients = continuum.assemble(scene, mesh, materials)
    with acceleration.override('continuum'):
        actual, gradients = continuum.assemble(scene, mesh, materials)
    assert np.array_equal(gradients, expected_gradients)
    for a, b in zip(actual, expected):
        assert np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
        assert a.data.tobytes() == b.data.tobytes()


def test_cached_operators_are_identical_and_read_only():
    expected = continuum.element_operators(AFFINE)
    first = accel_continuum.element_operators(AFFINE, continuum.element_operators)
    second = accel_continuum.element_operators(AFFINE.copy(), continuum.element_operators)
    assert first is second
    for a, b in zip(first, expected):
        assert a.tobytes() == b.tobytes()
        with pytest.raises(ValueError):
            a.flat[0] = 1.


def test_vectorized_exposure_is_bit_identical():
    scene = mixed(affine=AFFINE)
    mesh = continuum.voxel_mesh(scene, range(scene.labels.size))
    p = scene.properties[continuum.PROPERTY]
    _, gradients = continuum.assemble(scene, mesh, p['materials'])
    rng = np.random.default_rng(3)
    u = rng.normal(size=len(mesh.nodes) * 3) * 1e-3
    memories = [u * .3, u * -.2]
    for peaks in (False, True):
        expected = continuum.exposure_increment(scene, mesh, gradients, u, memories, p['materials'], .01, include_peaks=peaks)
        with acceleration.override('continuum'):
            actual = continuum.exposure_increment(scene, mesh, gradients, u, memories, p['materials'], .01, include_peaks=peaks)
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def held(n=9):
    source = block(n, branches=(500.,), times=(.5,))
    e = SurgicalEngine(source)
    m = n // 2
    c = int(np.ravel_multi_index((m, m, m), source.shape))
    brain.begin_hold(e, 'left', [c], [float(m)] * 3)
    brain.set_target(e, 'left', [m + .1, m, m])
    return e, c


def run_episode(features):
    with acceleration.override(*features):
        e, c = held()
        brain.tick(e, .1)
        neighbour = c + 1
        e.state['removed'] = sorted(set(e.state['removed']) | {neighbour})
        brain.tick(e, .1)
        brain.release(e, 'left')
        brain.tick(e, .2)
        return e.state


def test_continuum_episode_state_is_bit_identical():
    reference = run_episode(())
    accelerated = run_episode(('continuum',))
    assert json.dumps(accelerated, sort_keys=True) == json.dumps(reference, sort_keys=True)


def _numbers(value, out):
    if isinstance(value, dict):
        for key in sorted(value):
            _numbers(value[key], out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _numbers(item, out)
    elif isinstance(value, float):
        out.append(value)
    return out


def test_reordered_factorization_matches_reference_within_roundoff():
    reference = run_episode(())
    reordered = run_episode(('continuum', 'sparse-ordering'))
    assert json.dumps(_strip(reordered), sort_keys=True) == json.dumps(_strip(reference), sort_keys=True)
    a, b = np.asarray(_numbers(reordered, [])), np.asarray(_numbers(reference, []))
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=1e-9, atol=1e-13)


def _strip(value):
    """Structure only: same keys, list lengths and non-float values."""
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strip(item) for item in value]
    return None if isinstance(value, float) else value


def test_nested_dissection_is_a_block_permutation_and_solves():
    scene = mixed(n=6)
    mesh = continuum.voxel_mesh(scene, range(scene.labels.size))
    matrices, _ = continuum.assemble(scene, mesh, scene.properties[continuum.PROPERTY]['materials'])
    k = (matrices[0] + .3 * matrices[1]).tocsc()
    free = np.ones(k.shape[0], bool); free[:3 * 7 * 7] = False
    a = k[free][:, free].tocsc()
    perm = accel_continuum.nested_dissection(a)
    assert np.array_equal(np.sort(perm), np.arange(a.shape[0]))
    assert np.array_equal(perm.reshape(-1, 3) % 3, np.tile(np.arange(3), (a.shape[0] // 3, 1)))
    factor = accel_continuum.OrderedFactor(a)
    rhs = np.random.default_rng(0).random((a.shape[0], 2))
    assert np.abs(a @ factor.solve(rhs) - rhs).max() < 1e-10


def test_unknown_feature_names_are_refused():
    before = acceleration.active()
    with pytest.raises(ValueError, match='unknown acceleration'):
        with acceleration.override('contnuum'):
            pass
    assert acceleration.active() == before
