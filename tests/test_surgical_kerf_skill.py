"""Native checked stock-kerf prefixes; generated geometry, no patient claims."""
import copy

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_staged_skills import SkillLimits, checked_kerf_motion
from resectionlab.surgical_transition_meter import TransitionMeter


def pose(z): return {'position': [12., 12., z], 'quaternion': [0., 0., 0., 1.]}


def source(*, protected=False):
    labels = np.zeros((32, 32, 32), np.uint16)
    labels[12, 12, 6] = 1
    labels[12, 13, 8] = 1
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'bone', 'material': 'bone', **({'protected': True} if protected else {})}],
        {'provenance': 'GENERATED software kerf regression'}, {'tissuePhysicsV2': default_profile()})
    engine = SurgicalEngine(scene)
    for action in (
        {'slot': 'right', 'operation': 'configure', 'mechanicsRevision': 2, 'duration': .1},
        {'slot': 'right', 'operation': 'exchange', 'instrument': 'craniotome', 'duration': .1},
        {'slot': 'right', 'operation': 'place', 'pose': pose(-4.), 'duration': .1}):
        assert engine.step(action)['status'] == 'accepted'
    return engine


def test_shaft_blocked_kerf_proposes_actual_prefix_and_rechecks_before_acceptance():
    engine = source(); before = engine.snapshot(); meter = TransitionMeter(100)
    action = {'slot': 'right', 'operation': 'apply', 'pose': pose(12.), 'duration': 2.}
    rejected = engine.preview(action)
    assert (rejected['status'], rejected['reason']) == ('partial', 'SHAFT_COLLISION')
    result, receipt = checked_kerf_motion(engine, action, limits=SkillLimits(40, 100), meter=meter)
    assert receipt['status'] == 'accepted' and receipt['executionMode'] == 'checked_collision_prefix'
    assert not receipt['requestedPoseReached']
    assert receipt['observedEndPose'] == {'position': [12., 12., 9.], 'quaternion': [0., 0., 0., 1.]}
    assert engine.snapshot() == before
    assert result.metrics()['nonTargetRemovedMm3'] == 1
    assert result.metrics()['targetRemovedMm3'] == 0
    assert result.actions[-1] != action
    assert len(result.actions) == len(engine.actions)+1 and meter.calls == 3
    assert all(row['status'] == 'accepted' for row in result.history)
    assert SurgicalEngine.restore(engine.scene, result.snapshot()).snapshot() == result.snapshot()


@pytest.mark.parametrize('controls', [{'max_total_non_target_mm3': 0.}, {'allowed_removed_label_ids': (2,)}])
def test_shortening_never_widens_tissue_or_label_permission(controls):
    engine = source(); before = engine.snapshot()
    action = {'slot': 'right', 'operation': 'apply', 'pose': pose(12.), 'duration': 2.}
    if 'allowed_removed_label_ids' in controls:
        with pytest.raises(ValueError, match='no permitted bone'):
            checked_kerf_motion(engine, action, limits=SkillLimits(40, 100, **controls))
    else:
        result, receipt = checked_kerf_motion(engine, action, limits=SkillLimits(40, 100, **controls))
        assert receipt['status'] == 'refused' and receipt['reason'] == 'TOTAL_NON_TARGET_CAP'
        assert result.snapshot() == before
    assert engine.snapshot() == before


def test_protected_proposal_is_atomic_and_never_clipped():
    engine = source(protected=True); before = engine.snapshot()
    result, receipt = checked_kerf_motion(engine,
        {'slot': 'right', 'operation': 'apply', 'pose': pose(12.), 'duration': 2.},
        limits=SkillLimits(40, 100))
    assert receipt['status'] == 'refused' and receipt['reason'].startswith('PROTECTED_STRUCTURE')
    assert receipt['clippedProposal'] is None
    assert result.snapshot() == engine.snapshot() == before


@pytest.mark.parametrize('parameters', [{'radiusMm': .9}, {'depthMm': 1.}])
def test_stock_dimensions_cannot_be_changed_to_make_a_kerf_pass(parameters):
    engine = source()
    with pytest.raises(ValueError, match='STOCK_KERF_GEOMETRY_REQUIRED'):
        checked_kerf_motion(engine, {'slot': 'right', 'operation': 'apply', 'pose': pose(12.),
                                   'parameters': parameters, 'duration': 2.}, limits=SkillLimits(40, 100))


def test_interrupt_discards_probe_credit_and_preserves_authoritative_source():
    engine = source(); before = copy.deepcopy(engine.snapshot())
    meter = TransitionMeter(1)
    result, receipt = checked_kerf_motion(engine,
        {'slot': 'right', 'operation': 'apply', 'pose': pose(12.), 'duration': 2.},
        limits=SkillLimits(40, 1), meter=meter)
    assert receipt['status'] == 'interrupted' and meter.calls == 1
    assert result.snapshot() == engine.snapshot() == before
