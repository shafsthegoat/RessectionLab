"""Complete generated bone-opening controls, never patient/treatment evidence."""
import copy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_craniotomy import PROPERTY, contract_hash
from resectionlab.surgical_craniotomy_skills import (
    checked_craniotomy_actions, freeze_bone_opening_limits)
from resectionlab.surgical_staged_skills import SkillLimits
from resectionlab.surgical_transition_meter import TransitionMeter
from test_surgical_bone_transport import (
    accepted, cut_shell, grasp_shell, pose, request, shell)


@pytest.fixture(scope='module')
def sequence():
    engine, contour, flap, kerf = shell()
    cut_shell(engine, contour)
    grasp_shell(engine)
    accepted(engine, 'move', pose=pose(z=54.5))
    accepted(engine, 'extract')
    accepted(engine, 'withdraw', pose=pose(z=54.5))
    assert len(kerf) == 236 and len(flap) == 49
    assert engine.observe()['craniotomy']['openingComplete']
    return copy.deepcopy(engine.actions)


def budget(engine, cap, *, calls=100, actions=30, wall=60., meter=None):
    meter = TransitionMeter(calls) if meter is None else meter
    limits = freeze_bone_opening_limits(engine, max_total_bone_mm3=cap,
        max_native_actions=actions, max_public_calls=calls,
        max_wall_seconds=wall, meter=meter)
    return limits, meter


def test_complete_prospective_285_bone_only_policy_source_unchanged(sequence):
    source, _, flap, kerf = shell()
    before = source.snapshot()
    labels = source.scene.labels.copy()
    limits, meter = budget(source, 285.)
    result, report = checked_craniotomy_actions(source, sequence, limits=limits)
    assert report['status'] == 'accepted' and report['admittedOpening']
    assert report['evidenceScope'] == 'idealized-engineering'
    assert report['clinicalValidation'] == 'none'
    assert set(result.state['removed']) == flap | kerf
    assert report['ledger']['boneRemovedMm3'] == 285.
    assert report['ledger']['nonTargetRemovedMm3'] == 285.
    assert report['ledger']['additionalNonboneRemovedMm3'] == 0.
    assert report['ledger']['targetRemovedMm3'] == 0.
    assert report['physicalCallCost'] == {'publicCalls': 2*len(sequence),
        'previewCalls': len(sequence), 'directStepCalls': len(sequence)}
    assert report['sharedTransitionMeter'] == meter.report()
    assert all(row['endpointReached'] for row in report['attempts'])
    assert report['nativeOpening']['openingComplete']
    assert not result.state['holds'] and not result.state.get('detachedFragments')
    assert all(not hand['inserted'] and not hand['active'] and not hand['grip']
               for hand in result.state['slots'].values())
    assert source.snapshot() == before
    np.testing.assert_array_equal(source.scene.labels, labels)


def test_old_200_policy_refuses_before_unsafe_commit_and_keeps_actual_prefix(sequence):
    source, _, _, kerf = shell()
    limits, meter = budget(source, 200.)
    result, report = checked_craniotomy_actions(source, sequence, limits=limits)
    assert report['status'] == 'refused' and report['reason'] == 'TOTAL_BONE_CAP'
    assert not report['admittedOpening']
    assert 0 < report['ledger']['boneRemovedMm3'] <= 200.
    assert set(result.state['removed']) < kerf
    assert report['attempts'][-1]['previewBoneRemovedMm3'] > 200.
    assert meter.preview_calls == meter.direct_step_calls+1
    assert result.actions == sequence[:len(result.actions)]
    assert report['committedPrefix']['actionPrefixSeal'] == semantic_digest(result.actions)
    assert source.actions == [] and source.state['removed'] == []
    assert SkillLimits(30, 100).max_total_non_target_mm3 == 200.
    with pytest.raises(ValueError, match='at most 200'):
        SkillLimits(30, 100, max_total_non_target_mm3=285.)


def test_flap_extraction_charges_whole_49_not_only_exterior_tool_contact(sequence):
    source, _, flap, kerf = shell()
    limits, meter = budget(source, 260.)
    result, report = checked_craniotomy_actions(source, sequence, limits=limits)
    assert report['reason'] == 'TOTAL_BONE_CAP' and not report['admittedOpening']
    assert report['attempts'][-1]['action']['operation'] == 'extract'
    assert report['attempts'][-1]['previewBoneRemovedMm3'] == 285.
    assert set(report['attempts'][-1]['previewRemovedCellIds']) == flap
    assert set(result.state['removed']) == kerf
    assert set(result.state['holds']['left']['cells']) == flap
    assert not result.observe()['craniotomy']['apertureEvacuated']
    assert meter.preview_calls == meter.direct_step_calls+1


def test_cancellation_keeps_accepted_bone_prefix_and_all_spent_calls(sequence):
    source, _, _, _ = shell()
    limits, meter = budget(source, 285.)
    def cancel():
        if any(row.get('actionsAfter', 0) >= 4 for row in meter.trace):
            raise InterruptedError('cancel after actual accepted bone prefix')
    result, report = checked_craniotomy_actions(source, sequence, limits=limits, check=cancel)
    assert report['status'] == 'interrupted' and 'InterruptedError' in report['reason']
    assert not report['admittedOpening']
    assert result.actions == sequence[:4] and result.state['removed']
    assert report['ledger']['boneRemovedMm3'] > 0.
    assert report['ledger']['spentPublicCalls'] == 8
    assert source.actions == []


@pytest.mark.parametrize('cap_kind', ['calls', 'actions', 'wall'])
def test_hard_stop_returns_real_prefix_without_admission(sequence, monkeypatch, cap_kind):
    source, _, _, _ = shell()
    options = {'calls': 8} if cap_kind == 'calls' else {'actions': 4} if cap_kind == 'actions' else {}
    limits, meter = budget(source, 285., **options)
    if cap_kind == 'wall':
        from resectionlab import surgical_craniotomy_skills as skills
        real_clock = skills.time.monotonic
        def clock():
            return (limits.frozen_at+61. if any(row.get('actionsAfter', 0) >= 4
                    for row in meter.trace) else real_clock())
        monkeypatch.setattr(skills.time, 'monotonic', clock)
    result, report = checked_craniotomy_actions(source, sequence, limits=limits)
    assert report['status'] == 'interrupted' and not report['admittedOpening']
    assert result.actions == sequence[:4]
    assert report['ledger']['spentPublicCalls'] == meter.calls
    assert report['ledger']['boneRemovedMm3'] > 0.


def test_shared_meter_charges_speculation_replay_and_repeated_attempts_without_reset(sequence):
    source, _, _, _ = shell()
    meter = TransitionMeter(10)
    speculative = meter.wrap(source).clone()
    speculative.preview(sequence[0], include_view=False)
    replay = speculative.fresh()
    replay.step(sequence[0])  # Actual public replay work, not unmetered engine.replay.
    limits, _ = budget(source, 285., calls=10, meter=meter)
    first, report = checked_craniotomy_actions(source, sequence, limits=limits)
    assert len(first.actions) == 4 and report['ledger']['spentPublicCalls'] == 10
    assert report['physicalCallCost']['publicCalls'] == 8
    assert report['sharedTransitionMeter']['physicalCalls'] == 10
    second, stopped = checked_craniotomy_actions(source, sequence, limits=limits)
    assert second.actions == [] and stopped['ledger']['spentPublicCalls'] == 10
    assert stopped['physicalCallCost']['publicCalls'] == 0
    with pytest.raises(ValueError, match='replace shared'):
        checked_craniotomy_actions(source, sequence, limits=limits, meter=TransitionMeter(10))


def test_existing_bone_prefix_is_not_refunded(sequence):
    source, contour, _, _ = shell()
    meter = TransitionMeter(100)
    source = meter.wrap(source)
    cut_shell(source, contour)
    assert source.metrics()['nonTargetRemovedMm3'] == 236.
    with pytest.raises(ValueError, match='already exceeds'):
        budget(source, 200., meter=meter)
    limits, meter = budget(source, 260., meter=meter)
    result, report = checked_craniotomy_actions(source, sequence[len(source.actions):], limits=limits)
    assert report['reason'] == 'TOTAL_BONE_CAP'
    assert report['ledger']['boneRemovedMm3'] == 236.
    assert report['ledger']['nativeActions'] == len(result.actions)
    assert report['sourceNativeActions'] == len(source.actions)


@pytest.mark.parametrize('tamper', ['state', 'actions', 'source', 'contract', 'limits'])
def test_frozen_exact_bindings_refuse_tampering_before_native_calls(sequence, tamper):
    source, _, _, _ = shell()
    limits, meter = budget(source, 285.)
    if tamper == 'state':
        source.state['slots']['left']['position'][0] += 1.
    elif tamper == 'actions':
        source.actions.append(request('inspect'))
    elif tamper == 'source':
        source.scene.source['kind'] = 'different source'
    elif tamper == 'contract':
        raw = source.scene.properties[PROPERTY]
        raw['entry']['maximumInwardMm'] = 29.
        raw['contentHash'] = contract_hash(raw)
    else:
        limits = replace(limits, max_total_bone_mm3=999.)
    with pytest.raises(ValueError):
        checked_craniotomy_actions(source, sequence, limits=limits)
    assert meter.calls == 0


def test_incomplete_aperture_or_withdrawals_never_admitted(sequence):
    source, _, _, _ = shell()
    limits, _ = budget(source, 285.)
    result, report = checked_craniotomy_actions(source, sequence[:-1], limits=limits)
    assert report['status'] == 'refused' and report['reason'] == 'COMPLETE_NATIVE_OPENING_REQUIRED'
    assert report['nativeOpening']['apertureEvacuated']
    assert not report['nativeOpening']['openingComplete'] and not report['admittedOpening']
    assert result.state['slots']['left']['inserted']


def test_budget_has_no_default_bone_allowance_and_cannot_be_mutated():
    source, _, _, _ = shell()
    with pytest.raises(TypeError, match='max_total_bone_mm3'):
        freeze_bone_opening_limits(source, max_native_actions=30, max_public_calls=100,
                                  max_wall_seconds=60., meter=TransitionMeter(100))
    limits, _ = budget(source, 285.)
    with pytest.raises(FrozenInstanceError):
        limits.max_total_bone_mm3 = 999.
    assert limits.max_additional_nonbone_mm3 == limits.max_additional_target_mm3 == 0.


def test_native_wrong_entry_refusal_and_inactive_treatment_scope_are_retained(sequence):
    source, _, _, _ = shell()
    limits, meter = budget(source, 285.)
    unsafe = copy.deepcopy(sequence)
    unsafe[2]['pose'] = pose(50, 32, 62)
    result, report = checked_craniotomy_actions(source, unsafe, limits=limits)
    assert report['status'] == 'refused' and not report['admittedOpening']
    assert 'CRANIAL_ACCESS_' in report['reason']
    assert len(result.actions) == 2 and result.state['removed'] == []
    assert meter.calls == 5

    source, _, _, _ = shell()
    limits, meter = budget(source, 285.)
    treatment = [request('exchange', instrument='cusa')]
    result, report = checked_craniotomy_actions(source, treatment, limits=limits)
    assert report['reason'] == 'BONE_ONLY_ACTION_REQUIRED'
    assert result.actions == [] and meter.calls == 0


def test_unknown_attachment_evidence_never_authorizes_opening(sequence):
    source, _, _, _ = shell(proof_status='unknown')
    meter = TransitionMeter(100)
    with pytest.raises(ValueError, match='EVIDENCE_REQUIRED'):
        budget(source, 285., meter=meter)
    assert meter.calls == 0 and source.actions == []


def test_deadline_is_not_restarted_on_repeated_invocation(sequence, monkeypatch):
    from resectionlab import surgical_craniotomy_skills as skills
    source, _, _, _ = shell()
    limits, meter = budget(source, 285.)
    monkeypatch.setattr(skills.time, 'monotonic', lambda: limits.frozen_at+61.)
    for _ in range(2):
        result, report = checked_craniotomy_actions(source, sequence, limits=limits)
        assert report['status'] == 'interrupted' and 'wall limit' in report['reason']
        assert result.actions == [] and not report['admittedOpening']
        assert report['ledger']['wallSecondsSinceFreeze'] == 61.
    assert meter.calls == 0


@pytest.mark.parametrize('mode', ['bare', 'wrapped_after_prefix', 'missing_commit', 'unrelated_branch'])
def test_unknown_or_unrelated_prefix_call_costs_never_admitted(mode):
    source, contour, _, _ = shell()
    meter = TransitionMeter(100)
    if mode in ('missing_commit', 'unrelated_branch'):
        source = meter.wrap(source)
    cut_shell(source, contour)
    if mode == 'wrapped_after_prefix':
        source = meter.wrap(source)
    elif mode == 'missing_commit':
        meter.trace[:] = [row for row in meter.trace if not (
            row['kind'] == 'step' and row['actionsBefore'] == 2)]
    elif mode == 'unrelated_branch':
        source = meter.wrap(source.unwrap())  # Matching states, no charged ancestry.
    calls_before = meter.calls
    with pytest.raises(ValueError, match='CRANIOTOMY_PREFIX_CALL_EVIDENCE_REQUIRED'):
        budget(source, 285., meter=meter)
    assert meter.calls == calls_before


def test_same_meter_clone_ancestry_proves_prefix_costs(sequence):
    source, contour, _, _ = shell()
    meter = TransitionMeter(100)
    source = meter.wrap(source)
    cut_shell(source, contour)
    source = source.clone()
    grasp_shell(source)
    source = source.clone()
    limits, _ = budget(source, 285., meter=meter)
    before = meter.calls
    result, report = checked_craniotomy_actions(source, sequence[len(source.actions):], limits=limits)
    assert report['admittedOpening'] and report['ledger']['boneRemovedMm3'] == 285.
    assert report['ledger']['spentPublicCalls'] == 2*len(sequence)
    assert report['physicalCallCost']['publicCalls'] == meter.calls-before
    assert source.state['holds'] and not result.state['holds']


def test_refreezing_identical_policy_cannot_reset_deadline(sequence, monkeypatch):
    from resectionlab import surgical_craniotomy_skills as skills
    source, _, _, _ = shell()
    limits, meter = budget(source, 285.)
    monkeypatch.setattr(skills.time, 'monotonic', lambda: limits.frozen_at+61.)
    same, _ = budget(source, 285., meter=meter)
    assert same is limits
    result, report = checked_craniotomy_actions(source, sequence, limits=same)
    assert report['status'] == 'interrupted' and 'wall limit' in report['reason']
    assert result.actions == [] and meter.calls == 0


@pytest.mark.parametrize('change', ['bone', 'actions', 'calls', 'wall'])
def test_failed_attempt_cannot_raise_frozen_budget_after_observing_cost(sequence, change):
    source, _, _, _ = shell()
    limits, meter = budget(source, 200.)
    _, report = checked_craniotomy_actions(source, sequence, limits=limits)
    assert report['reason'] == 'TOTAL_BONE_CAP'
    kwargs = {'calls': 101} if change == 'calls' else {'actions': 31} if change == 'actions' else {'wall': 61.} if change == 'wall' else {}
    with pytest.raises(ValueError, match='CRANIOTOMY_BUDGET_ALREADY_FROZEN'):
        budget(source, 285. if change == 'bone' else 200., meter=meter, **kwargs)


def attachment(cell=(32, 32, 60), *, name='fixture-periosteal', kind='periosteal'):
    return {'id': name, 'kind': kind,
            'cells': [int(np.ravel_multi_index(cell, (64, 64, 96)))]}


def separation_sequence(sequence, *, point=None):
    return [*copy.deepcopy(sequence[:8]),
        request('exchange', instrument='periosteal_elevator'),
        request('place', pose=pose()),
        request('apply', pose=point or pose(z=59.5)),
        request('withdraw', pose=pose()),
        *copy.deepcopy(sequence[8:])]


def test_actual_periosteal_release_is_source_bound_and_full_native_opening_passes(sequence):
    source, _, flap, kerf = shell(attachments=[attachment()])
    before = source.snapshot()
    limits, meter = budget(source, 285.)
    assert limits.flap_attachment_ids == ('fixture-periosteal',)
    actions = separation_sequence(sequence)
    result, report = checked_craniotomy_actions(source, actions, limits=limits)
    assert report['admittedOpening'], report['reason']
    assert report['ledger']['separatedFlapAttachmentIds'] == ['fixture-periosteal']
    assert result.state['separatedAttachments'] == ['fixture-periosteal']
    assert set(result.state['removed']) == flap | kerf
    assert report['ledger']['targetRemovedMm3'] == 0.
    assert report['ledger']['additionalNonboneRemovedMm3'] == 0.
    assert report['ledger']['spentPublicCalls'] == 2*len(actions)
    assert meter.direct_step_calls == len(actions)
    assert source.snapshot() == before


def test_intact_declared_flap_attachment_still_refuses_native_grasp(sequence):
    source, _, _, kerf = shell(attachments=[attachment()])
    limits, _ = budget(source, 285.)
    result, report = checked_craniotomy_actions(source, sequence, limits=limits)
    assert report['status'] == 'refused' and not report['admittedOpening']
    assert 'ATTACHMENT' in report['reason']
    assert report['attempts'][-1]['action']['operation'] == 'grasp'
    assert set(result.state['removed']) == kerf
    assert result.state['separatedAttachments'] == [] and not result.state['holds']


def test_nonflap_attachment_release_is_not_authorized_by_bone_workflow(sequence):
    # Before cutting, an elevator can reach this declared kerf cell outside F.
    source, _, _, _ = shell(attachments=[attachment((26, 32, 60), name='kerf-periosteal')])
    limits, meter = budget(source, 285.)
    assert limits.flap_attachment_ids == ()
    actions = [copy.deepcopy(sequence[0]),
        request('exchange', instrument='periosteal_elevator'),
        request('place', pose=pose(26, 32, 55)),
        request('apply', pose=pose(26, 32, 59.5))]
    result, report = checked_craniotomy_actions(source, actions, limits=limits)
    assert report['status'] == 'refused' and not report['admittedOpening']
    assert report['reason'] == 'NONBONE_EFFECT_NOT_ALLOWED'
    assert result.state['separatedAttachments'] == []
    assert result.state['removed'] == []
    assert meter.preview_calls == meter.direct_step_calls+1


@pytest.mark.parametrize('change', ['bone', 'deadline', 'calls'])
def test_resealed_replacement_cannot_replace_the_prospective_policy(sequence, change):
    source, _, _, _ = shell()
    limits, meter = budget(source, 200.)
    changed = (replace(limits, max_total_bone_mm3=285.) if change == 'bone' else
               replace(limits, frozen_at=limits.frozen_at+1000.) if change == 'deadline' else
               replace(limits, max_public_calls=1000))
    changed = replace(changed, content_hash=semantic_digest(changed.declaration()))
    with pytest.raises(ValueError, match='CRANIOTOMY_FROZEN_BUDGET_IDENTITY_REQUIRED'):
        checked_craniotomy_actions(source, sequence, limits=changed)
    assert meter.calls == 0 and source.actions == []


@pytest.mark.parametrize('tamper', ['completed_without_actions', 'intact_with_orphan_history'])
def test_actionless_runtime_state_cannot_be_frozen_as_intact_source(sequence, tamper):
    source, _, _, _ = shell()
    if tamper == 'completed_without_actions':
        # A real public-native complete state with its action/cost provenance
        # erased must not be admitted as an unexecuted zero-cost source.
        for action in sequence:
            assert source.step(action)['status'] == 'accepted'
        assert source.observe()['craniotomy']['openingComplete']
        source.actions.clear()
        source.history.clear()
    else:
        source.history.append({'status': 'accepted', 'action': request('inspect')})
    meter = TransitionMeter(100)
    with pytest.raises(ValueError, match='CRANIOTOMY_CANONICAL_INTACT_SOURCE_REQUIRED'):
        budget(source, 285., meter=meter)
    assert meter.calls == 0


def test_runtime_state_and_history_must_still_be_canonical_at_invocation(sequence):
    source, _, _, _ = shell()
    limits, meter = budget(source, 285.)
    # Even a recomputed state binding cannot turn a direct mutation into work.
    source.state['removed'].append(next(iter(source._cranial_access.kerf_cells)))
    changed = replace(limits, source_state_id=source.state_id,
                      baseline_removed_cells=tuple(source.state['removed']))
    changed = replace(changed, content_hash=semantic_digest(changed.declaration()))
    with pytest.raises(ValueError, match='CRANIOTOMY_FROZEN_BUDGET_IDENTITY_REQUIRED'):
        checked_craniotomy_actions(source, sequence, limits=changed)
    assert meter.calls == 0
