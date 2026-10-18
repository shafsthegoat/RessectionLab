"""Mechanics invariants, independent of instrument reachability."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
from resectionlab.surgical_deformation_v2 import begin_hold, set_target, release, tick, profile


def engine():
    shape = (13, 13, 13)
    scene = SimpleNamespace(shape=shape, affine=np.eye(4), inverse=np.eye(4),
        labels=np.ones(shape, dtype=np.uint8), properties={'tissuePhysicsV2': {'brain': profile()}})
    return SimpleNamespace(scene=scene, by_label={1: {'material': 'soft_tissue'}, 2: {'material': 'tumor'},
        3: {'material': 'bone'}, 4: {'material': 'vessel'}},
        state={'removed': [], 'holds': {}, 'cutEdges': [], 'transectedEdges': []})


def cell(e, p): return int(np.ravel_multi_index(p, e.scene.shape))
def displacement(e, c): return np.array(e.state['deformation'].get(str(c), [0., 0., 0.]))


def held():
    e = engine(); c = cell(e, (6, 6, 6))
    begin_hold(e, 'left', [c], [6, 6, 6]); set_target(e, 'left', [6.4, 6, 6])
    return e, c


def test_neighbors_deform_smoothly_and_recover_over_time():
    e, c = held(); tick(e, .25)
    near = cell(e, (6, 7, 6)); far = cell(e, (6, 8, 6))
    assert 0 < np.linalg.norm(displacement(e, far)) < np.linalg.norm(displacement(e, near)) < np.linalg.norm(displacement(e, c))
    before = displacement(e, c).copy(); release(e, 'left')
    assert np.array_equal(displacement(e, c), before)
    tick(e, 1 / 120)
    assert 0 < np.linalg.norm(displacement(e, c)) < np.linalg.norm(before)
    tick(e, 2.)
    assert np.linalg.norm(displacement(e, c)) < .1 * np.linalg.norm(before)


def test_nonuniform_partition_and_replay_are_identical():
    a, c = held(); b = copy.deepcopy(a); replay = copy.deepcopy(a)
    tick(a, .3)
    for dt in [.013, .077, .041, .169]: tick(b, dt)
    tick(replay, .3)
    assert a.state == b.state == replay.state


def test_two_holds_and_removal_never_regenerate():
    e, c = held(); d = cell(e, (8, 6, 6))
    begin_hold(e, 'right', [d], [8, 6, 6]); set_target(e, 'right', [7.7, 6, 6]); tick(e, .2)
    assert displacement(e, c)[0] > 0 and displacement(e, d)[0] < 0
    e.state['removed'] = [c]; tick(e, .1); release(e, 'left'); tick(e, .1)
    assert str(c) not in e.state['deformation'] and c not in e.state['deformationModel']['active']
    assert e.state['removed'] == [c]


def test_cut_bonds_do_not_transmit_to_isolated_node():
    e, c = held(); adjacent = cell(e, (6, 7, 6))
    from resectionlab.surgical_deformation_v2 import _neighbors
    e.state['cutEdges'] = [sorted((adjacent, n)) for n, _ in _neighbors(e, adjacent)]
    tick(e, .5)
    assert np.linalg.norm(displacement(e, c)) > 0
    assert np.linalg.norm(displacement(e, adjacent)) == 0


def test_stiffness_caps_and_fixed_material():
    brain, c = held(); tumor, _ = held(); tumor.scene.labels[:] = 2
    tick(brain, .2); tick(tumor, .2)
    assert np.linalg.norm(displacement(tumor, c)) < np.linalg.norm(displacement(brain, c))
    bone = cell(brain, (7, 6, 6)); brain.scene.labels.flat[bone] = 3
    # Fresh patch must never acquire a fixed-material node.
    fresh = engine(); fresh.scene.labels.flat[bone] = 3
    begin_hold(fresh, 'left', [c, bone], [6, 6, 6]); set_target(fresh, 'left', [100, 6, 6]); tick(fresh, .5)
    assert bone not in fresh.state['deformationModel']['active']
    assert np.linalg.norm(displacement(fresh, c)) <= .5
    field = np.array(list(fresh.state['deformation'].values()))
    assert np.isfinite(field).all() and np.max(np.linalg.norm(field, axis=1)) <= 3
    assert np.isfinite(fresh.state['deformationModel']['elasticEnergyNmm'])


def test_invalid_time_and_target_rejected():
    e, _ = held()
    with pytest.raises(ValueError): tick(e, float('nan'))
    with pytest.raises(ValueError): set_target(e, 'left', [float('inf'), 0, 0])


def test_rigid_hold_is_not_a_viscoelastic_constraint():
    a, c = held(); b = copy.deepcopy(a)
    other = cell(a, (6, 7, 6))
    b.state['holds']['right'] = {'cells': [other], 'origin': [6, 7, 6], 'offset': [0, 1, 0]}
    tick(a, .2); tick(b, .2)
    assert a.state['deformation'] == b.state['deformation']


@pytest.mark.parametrize('key,value', [('fixedStepSeconds', 1e-15), ('fixedStepSeconds', 1.), ('radiusMm', 1e20)])
def test_profile_numerical_domain_rejects_unbounded_work(key, value):
    e, _ = held(); e.scene.properties['tissuePhysicsV2']['brain'][key] = value
    with pytest.raises(ValueError): tick(e, .1)


def test_obstacle_stop_preserves_neighbor_strain_bound():
    e = engine(); c = cell(e, (6, 6, 6)); bone = cell(e, (7, 6, 6))
    e.scene.labels.flat[bone] = 3
    begin_hold(e, 'left', [c], [6, 6, 6]); set_target(e, 'left', [9, 6, 6]); tick(e, .5)
    assert e.state['deformationModel']['maxNeighborDisplacementRatio'] <= profile()['maxEdgeStrain'] + 1e-12
