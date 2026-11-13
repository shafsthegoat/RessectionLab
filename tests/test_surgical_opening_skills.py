"""Native stock preparation macros; no opening/clinical inference from labels."""
import copy

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_opening_skills import stock_kerf_options, checked_stock_opening_macro
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_staged_skills import SkillLimits
from resectionlab.surgical_transition_meter import TransitionMeter


def source():
    labels = np.zeros((32, 32, 32), np.uint16); labels[12, 12, 6] = 1
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'bone', 'material': 'bone'}],
        {'provenance': 'GENERATED macro software regression'}, {'tissuePhysicsV2': default_profile()})
    engine = SurgicalEngine(scene)
    for action in (
        {'slot': 'right', 'operation': 'configure', 'mechanicsRevision': 2, 'duration': .1},
        {'slot': 'right', 'operation': 'exchange', 'instrument': 'craniotome', 'duration': .1}):
        assert engine.step(action)['status'] == 'accepted'
    options = stock_kerf_options(scene.identity, {'A': [12., 12., -4.], 'quaternion': [0., 0., 0., 1.]}, max_depth_mm=16.)
    return engine, options


def test_complete_native_macro_charges_active_return_and_replays_without_source_mutation():
    engine, options = source(); before = engine.snapshot(); meter = TransitionMeter(100)
    assert len(options) == 20
    candidate, receipt = checked_stock_opening_macro(engine, options[0], limits=SkillLimits(40, 100), meter=meter)
    assert receipt['status'] == 'accepted', receipt['reason']
    assert receipt['completeMacroAdmitted'] and receipt['activeReturnIsTreatment']
    assert receipt['nonTargetIncrementMm3'] == 1. and receipt['targetIncrementMm3'] == 0
    assert all(part['status'] == 'accepted' for part in receipt['parts'])
    assert all(not slot['active'] and not slot['inserted'] and not slot['grip'] for slot in candidate.state['slots'].values())
    assert candidate.state['elapsedSeconds'] > engine.state['elapsedSeconds']+9.
    assert meter.calls == receipt['physicalCalls'] == sum(part.get('physicalCallCost', {}).get('publicCalls', 0) for part in receipt['parts'])
    assert engine.snapshot() == before
    assert SurgicalEngine.restore(engine.scene, candidate.snapshot()).snapshot() == candidate.snapshot()


def test_refused_macro_cannot_acquire_treatment_credit_or_widen_caller_tissue_cap():
    engine, options = source(); before = engine.snapshot()
    candidate, receipt = checked_stock_opening_macro(engine, options[0],
        limits=SkillLimits(40, 100, max_total_non_target_mm3=0), meter=TransitionMeter(100))
    assert not receipt['completeMacroAdmitted'] and receipt['retentionRole'] == 'stopped_native_prefix_not_admitted'
    assert not candidate.state['removed'] and engine.snapshot() == before


def test_macro_requires_complete_cleanup_and_cannot_mutate_stock_option_scales():
    engine, options = source()
    bad = copy.deepcopy(options[0]); bad['rollMicroIntervalDeg'] = 1.
    with pytest.raises(ValueError, match='declared finite family'):
        checked_stock_opening_macro(engine, bad, limits=SkillLimits(40, 100), meter=TransitionMeter(100))
    engine.state['slots']['left']['active'] = True
    with pytest.raises(ValueError, match='COMPLETE_CLEANUP_REQUIRED'):
        checked_stock_opening_macro(engine, options[0], limits=SkillLimits(40, 100), meter=TransitionMeter(100))


@pytest.mark.parametrize('mutation', ['depth', 'lateral', 'rotation'])
def test_finite_family_checks_actual_waypoints_before_any_native_probe(mutation):
    engine, options = source(); before = engine.snapshot()
    bad = copy.deepcopy(options[0]); meter = TransitionMeter(100)
    if mutation == 'depth':
        bad['path'] = [{'position': [12., 12., 28.], 'quaternion': [0., 0., 0., 1.]}]
    elif mutation == 'lateral':
        bad['path'][0]['position'][0] += .1
    else:
        bad['path'][0]['quaternion'] = [0., 0., 1., 0.]
    with pytest.raises(ValueError, match='declared straight .12 mm family'):
        checked_stock_opening_macro(engine, bad, limits=SkillLimits(40, 100), meter=meter)
    assert meter.calls == 0 and engine.snapshot() == before
