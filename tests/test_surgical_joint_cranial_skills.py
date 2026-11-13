"""Native joint-policy accounting; these generated controls are not patients."""
import copy
from dataclasses import replace
import gc

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_joint_cranial_skills import (
    checked_joint_actions, checked_joint_cusa_motion, freeze_joint_cranial_limits,
    joint_cranial_ledger)
from resectionlab.surgical_transition_meter import TransitionMeter
from test_surgical_bone_transport import pose, request
from test_surgical_membrane_transport import (
    bone_actions, joint_source, membrane_cut_actions, membrane_grasp_actions)


def prepared(*, healthy=False, **limits):
    engine, contract, contour, patch, rim = joint_source()
    if healthy:
        from resectionlab.surgical_craniotomy import PROPERTY, binding, contract_hash
        from resectionlab.surgical_cranial_stages import freeze_cranial_stage_contract
        from resectionlab.surgical_engine import SurgicalEngine
        from resectionlab.surgical_scene import SurgicalScene
        labels = engine.scene.labels.copy()
        labels[32, 32, 66] = 4
        regions = engine.scene.regions+[{'id': 4, 'name': 'Generated retained brain contact',
                                         'material': 'soft_tissue'}]
        props = copy.deepcopy(engine.scene.properties)
        raw = props.pop(PROPERTY)
        base = SurgicalScene(labels, engine.scene.affine, regions, engine.scene.source, props)
        raw.update(binding(base)); raw['contentHash'] = contract_hash(raw); props[PROPERTY] = raw
        source = SurgicalEngine(SurgicalScene(labels, engine.scene.affine, regions, engine.scene.source, props))
        contract = freeze_cranial_stage_contract(source, membrane_cells=patch,
            retained_membrane_cells=rim, membrane_anchor_cells=contract.membrane_anchor_cells,
            work_cells=tuple(sorted((*contract.work_cells,
                int(np.ravel_multi_index((32, 32, 66), labels.shape))))),
            evidence=contract.as_dict()['evidence'])
        engine = SurgicalEngine(source.scene, cranial_stages=contract)
    meter = TransitionMeter(800)
    policy = freeze_joint_cranial_limits(engine, meter=meter, **{
        'max_bone_mm3': 285., 'max_preparation_nonbone_mm3': 49.,
        'max_work_nonbone_mm3': 20., 'max_native_actions': 300,
        'max_public_calls': 800, 'max_wall_seconds': 180., **limits})
    return engine, contract, contour, patch, rim, meter, policy


def preparation_program(engine, contract, contour):
    return (bone_actions(contour)+membrane_cut_actions(engine, contract)+
            membrane_grasp_actions()+[request('move', pose=pose(z=54.5)),
                request('extract'), request('withdraw', pose=pose(z=54.5))])


@pytest.fixture(scope='module')
def opened():
    engine, contract, contour, patch, rim, meter, policy = prepared()
    original = engine.snapshot()
    result, receipt = checked_joint_actions(engine,
        preparation_program(engine, contract, contour), limits=policy, meter=meter,
        phase='generated.joint.preparation')
    assert receipt['status'] == 'accepted', receipt
    assert engine.snapshot() == original
    return result, receipt, meter, policy, engine


def test_whole_bone_and_actual_membrane_preparation_stay_in_unique_native_total(opened):
    result, receipt, meter, policy, source = opened
    ledger = receipt['ledger']
    assert ledger['boneRemovedMm3'] == 285.
    assert ledger['preparationNonboneRemovedMm3'] == 49.
    assert ledger['workNonTargetRemovedMm3'] == ledger['targetRemovedMm3'] == 0.
    assert ledger['wholeNonTargetRemovedMm3'] == result.metrics()['nonTargetRemovedMm3'] == 334.
    assert ledger['withinLimits'] and ledger['unauthorizedRemovedSourceCells'] == 0
    assert receipt['stageReadiness']['duraCleared'] and receipt['stageReadiness']['wholeExitComplete']
    assert receipt['admittedPatientTask'] is False and receipt['caseQualificationRequired'] is True
    assert receipt['sequenceOutcome']['dimensions']['nonTargetRemovedMm3']['cumulative'] == 334.
    assert receipt['sequenceOutcome']['dimensions']['motorEncounterMm3']['cumulative'] is None
    assert len(source.actions) == 0 and meter.calls == 2*len(result.actions)


def test_continuation_preserves_full_preparation_and_original_shared_call_cost(opened):
    engine, _, meter, policy, _ = opened
    before = engine.snapshot()
    old_calls = meter.calls
    branch, receipt = checked_joint_actions(engine, [request('inspect')], limits=policy,
        meter=meter, phase='generated.joint.after-preparation')
    assert receipt['status'] == 'accepted'
    assert receipt['ledger']['wholeNonTargetRemovedMm3'] == 334.
    assert receipt['ledger']['spentPublicCalls'] == old_calls+2
    assert engine.snapshot() == before and len(branch.actions) == len(engine.actions)+1
    with pytest.raises(ValueError, match='FREEZE_BEFORE_ACTIONS'):
        freeze_joint_cranial_limits(branch, meter=meter, max_bone_mm3=600.,
            max_preparation_nonbone_mm3=600., max_work_nonbone_mm3=200.,
            max_native_actions=300, max_public_calls=800, max_wall_seconds=180.)


def test_preparation_cap_refuses_whole_extraction_without_erasing_accepted_bone():
    engine, contract, contour, patch, rim, meter, policy = prepared(max_preparation_nonbone_mm3=48.)
    branch, receipt = checked_joint_actions(engine,
        preparation_program(engine, contract, contour), limits=policy, meter=meter)
    assert receipt['status'] == 'refused' and receipt['reason'] == 'TOTAL_PREPARATION_NONBONE_CAP'
    assert receipt['ledger']['boneRemovedMm3'] == 285.
    assert receipt['ledger']['preparationNonboneRemovedMm3'] == 0.
    assert receipt['ledger']['wholeNonTargetRemovedMm3'] == 285.
    assert not set(patch) & set(branch.state['removed'])
    assert set(branch.state['holds']['left']['cells']) == set(patch)
    assert receipt['attempts'][-1]['previewLedger']['preparationNonboneRemovedMm3'] == 49.
    assert not engine.actions and not engine.state['removed']


def test_resealed_limit_substitution_and_same_identity_edit_are_rejected():
    engine, _, _, _, _, meter, policy = prepared()
    changed = replace(policy, max_work_nonbone_mm3=200.)
    changed = replace(changed, content_hash=semantic_digest(changed.declaration()))
    with pytest.raises(ValueError, match='REGISTERED_LIMIT_IDENTITY'):
        checked_joint_actions(engine, [], limits=changed, meter=meter)
    old_cap, old_hash = policy.max_work_nonbone_mm3, policy.content_hash
    object.__setattr__(policy, 'max_work_nonbone_mm3', 200.)
    object.__setattr__(policy, 'content_hash', semantic_digest(policy.declaration()))
    try:
        with pytest.raises(ValueError, match='LIMITS_CHANGED'):
            checked_joint_actions(engine, [], limits=policy, meter=meter)
    finally:
        object.__setattr__(policy, 'max_work_nonbone_mm3', old_cap)
        object.__setattr__(policy, 'content_hash', old_hash)


def test_dropping_limits_cannot_reopen_the_same_live_meter_budget():
    engine, _, _, _, _, meter, policy = prepared()
    del policy
    gc.collect()
    with pytest.raises(ValueError, match='BUDGET_ALREADY_FROZEN'):
        freeze_joint_cranial_limits(engine, meter=meter, max_bone_mm3=999.,
            max_preparation_nonbone_mm3=999., max_work_nonbone_mm3=200.,
            max_native_actions=300, max_public_calls=800, max_wall_seconds=180.)


def test_exact_last_call_is_available_but_next_proposal_is_not_executed():
    engine, _, _, _, _, meter, policy = prepared(max_public_calls=2)
    branch, receipt = checked_joint_actions(engine, [request('configure', mechanicsRevision=2,
        clockRevision=1)], limits=policy, meter=meter)
    assert receipt['status'] == 'accepted' and meter.calls == 2 and len(branch.actions) == 1
    before = branch.snapshot()
    stopped, receipt = checked_joint_actions(branch, [request('inspect')], limits=policy, meter=meter)
    assert receipt['status'] == 'interrupted' and meter.calls == 2
    assert stopped.snapshot() == before


def test_outcome_readout_time_cannot_publish_a_program_after_the_wall_bound(monkeypatch):
    import resectionlab.surgical_joint_cranial_skills as module
    engine, _, _, _, _, meter, policy = prepared()
    real_clock, outcome = module.time.monotonic, module._outcome
    offset = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: real_clock()+offset[0])

    def measured(*args):
        result = outcome(*args)
        offset[0] = policy.max_wall_seconds+1.
        return result

    monkeypatch.setattr(module, '_outcome', measured)
    branch, receipt = checked_joint_actions(engine,
        [request('configure', mechanicsRevision=2, clockRevision=1)], limits=policy, meter=meter)
    assert len(branch.actions) == 1 and len(engine.actions) == 0
    assert receipt['status'] == 'interrupted'
    assert receipt['ledger']['withinLimits'] is False
    assert 'WALL_BUDGET' in receipt['ledger']['refusalReasons']


def test_unmetered_or_unlogged_prefix_cannot_hide_prior_actions():
    engine, _, _, _, _, meter, policy = prepared()
    changed = engine.clone()
    assert changed.step(request('configure', mechanicsRevision=2, clockRevision=1))['status'] == 'accepted'
    with pytest.raises(ValueError, match='CHARGED_PREFIX'):
        checked_joint_actions(changed, [], limits=policy, meter=meter)
    branch, _ = checked_joint_actions(engine, [request('configure', mechanicsRevision=2,
        clockRevision=1)], limits=policy, meter=meter)
    branch.state['elapsedSeconds'] += 1.
    with pytest.raises(ValueError, match='UNLOGGED_NATIVE_STATE'):
        joint_cranial_ledger(branch, policy)


def test_long_cusa_motion_stops_at_its_frozen_continuation_budget_and_keeps_costs(opened):
    engine, _, meter, policy, _ = opened
    branch, setup = checked_joint_actions(engine,
        [request('exchange', instrument='cusa'), request('place', pose=pose())],
        limits=policy, meter=meter, phase='generated.joint.cusa.setup')
    assert setup['status'] == 'accepted', setup
    branch, result = checked_joint_cusa_motion(branch,
        request('apply', pose=pose(z=68.), duration=.05), limits=policy, meter=meter,
        phase='generated.joint.cusa.work')
    assert result['status'] == 'interrupted' and result['reason'] == 'TISSUE_CONTACT_CONTINUATION_BUDGET'
    assert result['ledger']['targetRemovedMm3'] > 0.
    assert result['ledger']['wholeNonTargetRemovedMm3'] == 334.
    assert result['ledger']['workNonTargetRemovedMm3'] == 0.
    assert result['contactContinuations']
    branch, exit_receipt = checked_joint_actions(branch,
        [request('deactivate'), request('withdraw', pose=pose(z=54.5))],
        limits=policy, meter=meter, phase='generated.joint.cusa.exit')
    assert exit_receipt['status'] == 'accepted'
    assert exit_receipt['stageReadiness']['wholeExitComplete']
    assert joint_cranial_ledger(branch, policy)['wholeNonTargetRemovedMm3'] == 334.


def test_checked_cusa_reaches_a_finite_contact_endpoint_then_exits(opened):
    engine, _, meter, policy, _ = opened
    branch, setup = checked_joint_actions(engine,
        [request('exchange', instrument='cusa'), request('place', pose=pose())],
        limits=policy, meter=meter, phase='generated.joint.short-cusa.setup')
    assert setup['status'] == 'accepted'
    branch, receipt = checked_joint_cusa_motion(branch,
        request('apply', pose=pose(z=66.5), duration=30.), limits=policy, meter=meter,
        phase='generated.joint.short-cusa.work')
    assert receipt['status'] == 'accepted', receipt
    assert branch.state['slots']['left']['position'][2] == pytest.approx(66.5, abs=1e-6)
    assert receipt['ledger']['targetRemovedMm3'] > 0.
    assert receipt['ledger']['wholeNonTargetRemovedMm3'] == 334.
    branch, exit_receipt = checked_joint_actions(branch,
        [request('deactivate'), request('withdraw', pose=pose(z=54.5))], limits=policy, meter=meter)
    assert exit_receipt['status'] == 'accepted' and exit_receipt['stageReadiness']['wholeExitComplete']


def test_work_non_target_budget_is_not_refunded_by_bone_or_membrane_preparation():
    engine, contract, contour, _, _, meter, policy = prepared(healthy=True, max_work_nonbone_mm3=0.)
    branch, preparation = checked_joint_actions(engine,
        preparation_program(engine, contract, contour), limits=policy, meter=meter)
    assert preparation['status'] == 'accepted', preparation
    branch, setup = checked_joint_actions(branch,
        [request('exchange', instrument='cusa'), request('place', pose=pose())], limits=policy, meter=meter)
    assert setup['status'] == 'accepted'
    branch, receipt = checked_joint_cusa_motion(branch,
        request('apply', pose=pose(z=68.), duration=.05), limits=policy, meter=meter)
    assert receipt['status'] == 'refused' and receipt['reason'] == 'TOTAL_WORK_NON_TARGET_CAP', receipt
    assert receipt['ledger']['boneRemovedMm3'] == 285.
    assert receipt['ledger']['preparationNonboneRemovedMm3'] == 49.
    assert receipt['ledger']['wholeNonTargetRemovedMm3'] == 334.
    assert receipt['ledger']['workNonTargetRemovedMm3'] == 0.
    assert receipt['ledger']['targetRemovedMm3'] == 0.
    assert receipt['parts'][-1]['attempts'][-1]['previewLedger']['workNonTargetRemovedMm3'] == 1.
