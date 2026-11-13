"""Independent whole-cell geometry controls; no anatomical qualification."""
import itertools

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.surgical_protection_transfer import (
    transfer_inferior_slab, verify_inferior_slab)


def source(shape=(5, 6, 7), last=2):
    data = np.zeros(shape, np.uint8); data[:, :, :last+1] = 1
    return data, verify_inferior_slab(data)


def exact_intersection(cell, low, high):
    """Independent SAT of a parallelepiped and source axis-aligned box."""
    corners = np.asarray(list(itertools.product((0., 1.), repeat=3)))
    box = low+corners*(high-low)
    # The caller supplies corners in itertools ordering.
    edges = [cell[4]-cell[0], cell[2]-cell[0], cell[1]-cell[0]]
    axes = [*np.eye(3), *(np.cross(edges[j], edges[k]) for j, k in ((0, 1), (1, 2), (2, 0))),
            *(np.cross(edge, axis) for edge in edges for axis in np.eye(3))]
    for axis in axes:
        if np.linalg.norm(axis) <= 1e-12: continue
        a, b = cell@axis, box@axis
        if a.max() < b.min()-1e-9 or b.max() < a.min()-1e-9: return False
    return True


def test_exact_source_recipe_and_identity_full_cells_are_not_center_sampled():
    data, slab = source()
    before = data.copy()
    result = transfer_inferior_slab(slab, np.eye(4), data.shape, np.eye(4), np.eye(4))
    expected = np.zeros(data.shape, bool); expected[:, :, :4] = True
    np.testing.assert_array_equal(result['slabOverlapMask'], expected)
    assert not result['outsideSourceGridMask'][1:-1, 1:-1, 1:-1].any()
    assert result['outsideSourceGridMask'][0].all()
    assert result['outsideSourceGridMask'][-1].all()
    assert result['report']['registrationOrAnatomyQualified'] is False
    np.testing.assert_array_equal(data, before)
    with pytest.raises(ValueError): result['protectedMask'].flags.writeable = True


@pytest.mark.parametrize('damage', ['hole', 'detached-plane', 'other-label', 'empty'])
def test_claimed_slab_with_any_missing_or_extra_cell_refuses(damage):
    data, _ = source()
    if damage == 'hole': data[2, 3, 1] = 0
    elif damage == 'detached-plane': data[:, :, 5] = 1
    elif damage == 'other-label': data[0, 0, 5] = 2
    else: data[:] = 0
    with pytest.raises(ValueError): verify_inferior_slab(data)


def test_source_escape_stays_unknown_and_protected_even_when_slab_not_hit():
    data, slab = source(); target = np.eye(4); target[0, 3] = 20.
    result = transfer_inferior_slab(slab, np.eye(4), data.shape, target, np.eye(4))
    assert not result['slabOverlapMask'].any()
    assert result['outsideSourceGridMask'].all() and result['protectedMask'].all()


def test_tiny_source_grid_escape_cannot_be_rounded_into_clear_space():
    _, slab = source(); target = np.eye(4); target[:3, 3] = [-5e-11, 2., 5.]
    result = transfer_inferior_slab(slab, np.eye(4), (1, 1, 1), target, np.eye(4))
    assert not result['slabOverlapMask'][0, 0, 0]
    assert result['outsideSourceGridMask'][0, 0, 0]
    assert result['protectedMask'][0, 0, 0]


@pytest.mark.parametrize('seed', range(5))
def test_oblique_sheared_cells_enclose_independent_exact_intersections(seed):
    rng = np.random.default_rng(seed); _, slab = source()
    a = np.eye(4); a[:3, :3] = [[.7, .2, 0.], [0., 1.1, .15], [.1, 0., .9]]
    target = np.eye(4); target[:3, :3] = Rotation.from_rotvec(rng.normal(size=3)*.2).as_matrix()@np.diag([.9, 1.3, 1.1])
    target[:3, 3] = rng.uniform(-1., 1., 3)
    registration = np.eye(4); registration[:3, :3] = Rotation.from_euler('z', 17., degrees=True).as_matrix(); registration[:3, 3] = [.2, -.4, .3]
    shape = (5, 5, 5)
    result = transfer_inferior_slab(slab, a, shape, target, registration, chunk_cells=7)
    mapping = np.linalg.inv(a)@np.linalg.inv(registration)@target
    offsets = np.asarray(list(itertools.product((-.5, .5), repeat=3)))
    actual_hits = 0
    for index in np.ndindex(shape):
        cell = (offsets+index)@mapping[:3, :3].T+mapping[:3, 3]
        if exact_intersection(cell, np.full(3, -.5), np.array([4.5, 5.5, 2.5])):
            assert result['slabOverlapMask'][index], index
            actual_hits += 1
    assert actual_hits > 0


def test_continuous_uncertainty_encloses_sampled_actual_rigid_transforms():
    _, slab = source(); shape = (5, 6, 7); pivot = np.array([2., 2.5, 3.])
    target = np.eye(4); target[:3, 3] = [.7, .2, .5]
    result = transfer_inferior_slab(slab, np.eye(4), shape, target, np.eye(4),
        assumed_translation_mm=.6, assumed_rotation_degrees=13., rotation_pivot_target_ras_mm=pivot)
    offsets = np.asarray(list(itertools.product((-.5, .5), repeat=3)))
    rng = np.random.default_rng(17)
    hits = 0
    for _ in range(14):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        q = Rotation.from_rotvec(axis*np.deg2rad(rng.uniform(-13., 13.))).as_matrix()
        delta = rng.normal(size=3); delta *= .6/np.linalg.norm(delta)
        actual = np.eye(4); actual[:3, :3] = q; actual[:3, 3] = pivot-q@pivot+delta
        inverse = np.linalg.inv(actual)
        for index in np.ndindex(shape):
            world = offsets+index+target[:3, 3]
            cell = world@inverse[:3, :3].T+inverse[:3, 3]
            if exact_intersection(cell, np.full(3, -.5), np.array([4.5, 5.5, 2.5])):
                assert result['slabOverlapMask'][index]
                hits += 1
    assert hits > 100


def test_larger_stress_grows_exclusions_and_chunks_are_identical():
    _, slab = source(); args = (slab, np.eye(4), (5, 6, 7), np.eye(4), np.eye(4))
    smaller = transfer_inferior_slab(*args)
    a = transfer_inferior_slab(*args, assumed_translation_mm=.3, chunk_cells=1)
    b = transfer_inferior_slab(*args, assumed_translation_mm=.3, chunk_cells=111)
    assert np.all(a['protectedMask'][smaller['protectedMask']])
    np.testing.assert_array_equal(a['protectedMask'], b['protectedMask'])
    assert a['report'] == b['report']


def test_budget_and_cancellation_refuse_without_a_partial_return_or_source_change():
    data, slab = source(); args = (slab, np.eye(4), data.shape, np.eye(4), np.eye(4))
    with pytest.raises(ValueError, match='allocation budget'): transfer_inferior_slab(*args, max_target_cells=1)
    calls = []
    def cancel():
        calls.append(1)
        if len(calls) == 3: raise TimeoutError('cancel')
    with pytest.raises(TimeoutError): transfer_inferior_slab(*args, chunk_cells=2, check=cancel)
    assert len(calls) == 3 and data.sum() == 90


@pytest.mark.parametrize('kwargs', [{'assumed_translation_mm': -1.}, {'assumed_rotation_degrees': 181.},
    {'assumed_rotation_degrees': 1.}, {'chunk_cells': True}, {'max_target_cells': 0}])
def test_invalid_stresses_and_bounds_do_not_default_to_clear(kwargs):
    _, slab = source()
    with pytest.raises(ValueError): transfer_inferior_slab(slab, np.eye(4), (5, 6, 7), np.eye(4), np.eye(4), **kwargs)
