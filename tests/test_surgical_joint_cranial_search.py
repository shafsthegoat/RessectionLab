"""Generated joint-search controls, never evidence of a completed patient task."""
from dataclasses import replace

import pytest

from resectionlab.surgical_joint_cranial_search import (
    checked_joint_sequence_search, freeze_joint_resection_goal,
    joint_goal_assessment)
from resectionlab.surgical_joint_cranial_skills import joint_cranial_ledger
from test_surgical_bone_transport import pose, request
from test_surgical_joint_cranial_skills import prepared, preparation_program
from test_surgical_membrane_transport import bone_actions


def inventory(engine, contract, contour):
    removed = set(engine.state['removed'])
    if not set(contract.bone_cells) <= removed:
        kind, commands = 'opening', bone_actions(contour)
    elif not set(contract.membrane_cells) <= removed:
        kind = 'preparation'
        commands = preparation_program(engine, contract, contour)[len(bone_actions(contour)):]
        commands += [request('exchange', instrument='cusa'), request('place', pose=pose())]
    else:
        return [{'id': 'unreached-long-work', 'sourceStateId': engine.state_id, 'kind': 'cusa',
            'motion': request('apply', pose=pose(z=68.), duration=.05)},
            {'id': 'target-work', 'sourceStateId': engine.state_id, 'kind': 'cusa',
            'motion': request('apply', pose=pose(z=66.5), duration=30.)}]
    return [{'id': kind, 'sourceStateId': engine.state_id, 'kind': kind, 'actions': commands}]


def run_search(horizon):
    source, contract, contour, _, _, meter, limits = prepared(max_public_calls=1200)
    goal = freeze_joint_resection_goal(source, limits=limits)
    before = source.snapshot()
    result = checked_joint_sequence_search(source, propose=lambda e: inventory(e, contract, contour),
        limits=limits, goal=goal, meter=meter, horizon=horizon, max_expansions=6)
    assert source.snapshot() == before
    return source, contract, meter, limits, goal, result


@pytest.fixture(scope='module')
def searched():
    return run_search(3)


def test_zero_target_opening_then_membrane_preparation_enables_frozen_task(searched):
    source, contract, meter, limits, goal, result = searched
    assert result.selected is not None, result.evidence
    selected = result.selected
    assert [r['kind'] for r in selected.path] == ['opening', 'preparation', 'cusa']
    assert [r['preparationOnly'] for r in selected.path] == [True, True, False]
    assert not selected.path[0]['exit']['stageReadiness']['duraCleared']
    assert selected.path[0]['exit']['exitComplete']
    assert selected.goal_assessment['completedFrozenTask']
    assert selected.goal_assessment['originalTargetFractionRemoved'] >= .5
    assert selected.goal_assessment['zonesReachingRequiredOriginalFraction'] >= 3
    assert selected.ledger['boneRemovedMm3'] == 285.
    assert selected.ledger['preparationNonboneRemovedMm3'] == 49.
    assert selected.ledger['wholeNonTargetRemovedMm3'] == 334.
    assert selected.ledger['workNonTargetRemovedMm3'] == 0.
    assert not selected.goal_assessment['admittedPatientTask']
    assert result.evidence['physicalCallsDuringSearch'] == meter.calls
    assert result.evidence['decision'] == 'COMPLETED_FROZEN_TASK'
    assert result.stopped_prefixes and not result.evidence['budgetExhausted']
    failed = next(r for r in result.attempts if r['proposalId'] == 'unreached-long-work')
    assert failed['work']['reason'] == 'TISSUE_CONTACT_CONTINUATION_BUDGET'
    assert failed['failureCategory'] == 'CANDIDATE_CONTROLLER_LIMIT'
    assert failed['work']['ledger']['targetRemovedMm3'] > 0.
    assert failed['work']['publicCalls'] > 0
    assert not source.actions and not source.state['removed']


def test_matched_immediate_selection_cannot_use_zero_target_opening():
    source, _, meter, limits, _, result = run_search(1)
    assert result.selected is None and not result.alternatives
    assert result.evidence['decision'] == 'STOP'
    assert result.attempts[0]['status'] == 'no_immediate_target'
    assert result.attempts[0]['work']['ledger']['boneRemovedMm3'] == 285.
    assert meter.calls > 0
    with pytest.raises(ValueError, match='FREEZE_BEFORE_ACTIONS'):
        freeze_joint_resection_goal(source, limits=limits)


def test_selected_full_native_branch_replays_with_original_preparation_cost(searched):
    source, _, meter, limits, _, result = searched
    final = result.selected.exited
    old_calls = meter.calls
    replay = meter.wrap(source).clone()
    # Replay each recorded partial once. A proposal executor intentionally
    # interprets partial contact as a request for a new remainder; applying that
    # controller to an already expanded action log would duplicate continuation.
    with meter.phase('generated.joint.search.cold-replay'):
        for action, expected in zip(final.actions, final.history):
            preview = replay.preview(action, include_view=False, include_history=False)
            actual = replay.step(action)
            assert actual['status'] == expected['status'] == preview['status']
            assert actual['stateId'] == expected['stateAfter'] == preview['stateId']
    assert replay.snapshot() == final.snapshot()
    assert joint_cranial_ledger(replay, limits)['wholeNonTargetRemovedMm3'] == 334.
    assert meter.calls-old_calls == 2*len(final.actions)


def test_goal_cannot_be_refrozen_after_preparation_or_resealed_with_new_threshold(searched):
    source, _, _, limits, goal, result = searched
    with pytest.raises(ValueError, match='FREEZE_BEFORE_ACTIONS'):
        freeze_joint_resection_goal(result.selected.work, limits=limits)
    changed = replace(goal, minimum_target_fraction=.01)
    with pytest.raises(ValueError, match='REGISTERED_GOAL'):
        joint_goal_assessment(result.selected.exited, goal=changed, limits=limits,
            exit_receipt=result.selected.path[-1]['exit'])
    old = goal.minimum_target_fraction
    object.__setattr__(goal, 'minimum_target_fraction', .01)
    try:
        with pytest.raises(ValueError, match='FROZEN_GOAL_CHANGED'):
            joint_goal_assessment(source, goal=goal, limits=limits, exit_receipt={})
    finally:
        object.__setattr__(goal, 'minimum_target_fraction', old)


def test_original_target_and_spatial_thresholds_are_never_reduced_to_contact():
    source, _, _, _, _, _, limits = prepared()
    with pytest.raises(ValueError, match='target fraction'):
        freeze_joint_resection_goal(source, limits=limits, minimum_target_fraction=.01)
    with pytest.raises(ValueError, match='target fraction'):
        freeze_joint_resection_goal(source, limits=limits, minimum_zones=1)
    with pytest.raises(ValueError, match='target fraction'):
        freeze_joint_resection_goal(source, limits=limits, minimum_zone_fraction_of_original=.001)
    goal = freeze_joint_resection_goal(source, limits=limits)
    assessment = joint_goal_assessment(source, goal=goal, limits=limits,
        exit_receipt={'status': 'accepted', 'finalStateId': source.state_id,
            'stageReadiness': {'wholeExitComplete': True, 'workEntrySupported': True}})
    assert not assessment['completedFrozenTask']
    assert 'NATIVE_WORK_AND_WHOLE_EXIT_REQUIRED' in assessment['reasons']


def test_empty_inventory_is_a_finite_generator_limit_not_anatomical_impossibility():
    source, _, _, _, _, meter, limits = prepared()
    goal = freeze_joint_resection_goal(source, limits=limits)
    result = checked_joint_sequence_search(source, propose=lambda e: [], limits=limits, goal=goal)
    assert result.selected is None and result.evidence['decision'] == 'STOP'
    assert result.evidence['expansions'] == 0 and meter.calls == 0
    assert 'does not prove anatomical impossibility' in result.evidence['scope']


def test_proposals_cannot_be_transplanted_from_another_native_state():
    source, _, _, _, _, meter, limits = prepared()
    goal = freeze_joint_resection_goal(source, limits=limits)
    with pytest.raises(ValueError, match='current native source state'):
        checked_joint_sequence_search(source, propose=lambda e: [{'id': 'stale', 'kind': 'opening',
            'sourceStateId': 'other', 'actions': [request('inspect')]}], limits=limits, goal=goal)
    assert meter.calls == 0
