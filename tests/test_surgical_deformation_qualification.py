"""Solver qualification against independent analytical controls; no tissue fit."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab import surgical_deformation_v2 as brain


def coupon(*, step=1 / 120, target=.1):
    p = brain.profile(); p['fixedStepSeconds'] = step
    shape = (7, 7, 7)
    # Isolated node with a declared weak foundation and elastic hold spring.
    labels = np.zeros(shape, dtype=np.uint16); labels[3, 3, 3] = 1
    e = SimpleNamespace(scene=SimpleNamespace(shape=shape, affine=np.eye(4), inverse=np.eye(4),
        labels=labels, properties={'tissuePhysicsV2': {'brain': p}}),
        by_label={1: {'material': 'soft_tissue'}},
        state={'removed': [], 'holds': {}, 'cutEdges': [], 'transectedEdges': []})
    cell = int(np.ravel_multi_index((3, 3, 3), shape))
    brain.begin_hold(e, 'left', [cell], [3., 3., 3.])
    brain.set_target(e, 'left', [3. + target, 3., 3.])
    return e, cell


def test_optional_diagnostics_preserve_identical_state_and_have_no_global_leak():
    a, _ = coupon(); b = copy.deepcopy(a)
    with brain.diagnostic_scope() as report:
        brain.tick(a, .1)
    frozen = copy.deepcopy(report)
    brain.tick(b, .1)
    assert a.state == b.state
    assert report == frozen
    assert report['substeps'] == 12
    assert report['maxFreeResidualN'] < 1e-14
    assert report['maxAbsDiscreteEnergyBalanceNmm'] < 1e-14
    assert not any(report[k] for k in ('targetCapNodeTicks', 'forceCapNodeTicks',
        'displacementCapNodeSteps', 'strainProjectionSteps', 'collisionRejectedSteps'))


def test_independent_implicit_euler_single_node_solution_and_time_convergence():
    errors = []
    for h in (1 / 60, 1 / 120, 1 / 240):
        e, c = coupon(step=h)
        p = e.scene.properties['tissuePhysicsV2']['brain']
        k = p['brainStiffnessNPerMm'] * .05 + p['holdStiffnessNPerMm']
        force = p['holdStiffnessNPerMm'] * .1
        equilibrium = force / k
        duration = .1
        brain.tick(e, duration)
        u = e.state['deformation'][str(c)][0]
        implicit_euler = equilibrium * (1 - (1 + h * k / p['dragNsPerMm']) ** -round(duration / h))
        continuous = equilibrium * (1 - np.exp(-duration * k / p['dragNsPerMm']))
        assert u == pytest.approx(implicit_euler, abs=1e-14)
        errors.append(abs(u - continuous))
    assert errors[2] < errors[1] < errors[0]


def test_target_and_force_caps_are_reported_with_attempted_solve_scope():
    e, _ = coupon(target=20.)
    e.scene.properties['tissuePhysicsV2']['brain']['maxForceN'] = .001
    with brain.diagnostic_scope() as report:
        brain.tick(e, .1)
    assert report['targetCapNodeTicks'] == report['forceCapNodeTicks'] == 1
    assert 'attempted' in report['scope']


def test_nested_scopes_are_isolated_and_reset_after_exception():
    e, _ = coupon()
    with brain.diagnostic_scope() as outer:
        brain.tick(e, .01)
        with pytest.raises(RuntimeError):
            with brain.diagnostic_scope() as inner:
                brain.tick(e, .01)
                raise RuntimeError('cancelled attempted work')
        assert outer['ticks'] == inner['ticks'] == 1
        brain.tick(e, .01)
        assert outer['ticks'] == 2 and inner['ticks'] == 1


def test_strain_projection_residual_is_visible():
    e, c = coupon(target=2.)
    # An anchored neighboring cell makes the requested strain exceed the
    # declared numerical cap. No biological strain threshold is asserted.
    e.scene.labels[4, 3, 3] = 1
    neighbor = int(np.ravel_multi_index((4, 3, 3), e.scene.shape))
    e.state['deformationModel']['active'].append(neighbor)
    e.state['deformationModel']['anchors'] = [neighbor]
    e.scene.properties['tissuePhysicsV2']['brain']['maxEdgeStrain'] = .01
    with brain.diagnostic_scope() as report:
        brain.tick(e, .1)
    assert report['strainProjectionSteps'] > 0
    assert report['maxFreeResidualN'] > .001
    assert report['maxAbsDiscreteEnergyBalanceNmm'] > 1e-8
    assert e.state['deformationModel']['maxNeighborDisplacementRatio'] <= .01 + 1e-12


def test_collision_rejection_is_visible_even_when_energy_defect_is_small():
    e, _ = coupon(target=2.)
    e.scene.labels[4, 3, 3] = 2; e.by_label[2] = {'material': 'bone'}
    with brain.diagnostic_scope() as report:
        brain.tick(e, .1)
    assert report['collisionRejectedSteps'] > 0
    assert report['maxFreeResidualN'] > .001


def test_gravity_driven_displacement_cap_is_reported():
    from resectionlab.surgical_position import validate_position
    e, c = coupon()
    brain.release(e, 'left')
    e.state['patientPosition'] = validate_position({'preset': 'supine'})
    e.scene.properties['tissuePhysicsV2']['brain']['maxDisplacementMm'] = .01
    with brain.diagnostic_scope() as report:
        brain.tick(e, 1.)
    assert report['displacementCapNodeSteps'] > 0
    assert np.linalg.norm(e.state['deformation'][str(c)]) == pytest.approx(.01)
    assert report['maxFreeResidualN'] > 0
