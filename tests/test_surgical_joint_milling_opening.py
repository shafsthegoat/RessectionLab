"""Generated, prospectively bounded joint milling controls; never patients."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.core import semantic_digest
from resectionlab.surgical_cranial_stages import freeze_cranial_stage_contract, PROOFS
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_joint_cranial_skills import (
    checked_joint_actions, freeze_joint_cranial_limits, joint_cranial_ledger)
from resectionlab.surgical_joint_milling_opening import (
    _checked_staging_program, _content_hash, checked_joint_milling_opening)
from resectionlab.surgical_milling import PROPERTY, binding, contract_hash, milling_readiness
from resectionlab.surgical_milling_contact_proposals import bind_milling_approach
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_transition_meter import TransitionMeter
from test_surgical_milling_native import milling_shell


def fixture(*, point_scheme='center', max_public_calls=4000, park_other=True):
    """Fixed81A +9-cell membrane, explicit native other-hand exterior setup."""
    old, aperture = milling_shell()
    labels = old.scene.labels.copy()
    labels[25:40, 25:40, 64] = 3
    patch = tuple(int(np.ravel_multi_index((x, y, 64), labels.shape))
                  for x in range(31, 34) for y in range(31, 34))
    rim = tuple(sorted(set(map(int, np.flatnonzero(labels == 3)))-set(patch)))
    regions = old.scene.regions+[{'id': 3, 'name': 'Generated milling membrane', 'material': 'membrane'}]
    props = copy.deepcopy(old.scene.properties)
    raw = props.pop(PROPERTY)
    base = SurgicalScene(labels, old.scene.affine, regions, old.scene.source, props)
    raw.update(binding(base)); raw['contentHash'] = contract_hash(raw); props[PROPERTY] = raw
    scene = SurgicalScene(labels, base.affine, regions, base.source, props)
    source = SurgicalEngine(scene)
    proof = {'status': 'reviewed', 'provenance': 'Generated fixed81A joint milling control; no patient evidence',
             'evidenceSha256': '4'*64}
    stage = freeze_cranial_stage_contract(source, membrane_cells=patch,
        retained_membrane_cells=rim, membrane_anchor_cells=(rim[0],),
        work_cells=tuple(map(int, np.flatnonzero(labels == 2))),
        evidence={name: dict(proof) for name in PROOFS})
    source = SurgicalEngine(scene, cranial_stages=stage)
    # All bounds are frozen before geometry/staging/native actions. Do not
    # refreeze after an accepted prefix or create a continuation meter.
    meter = TransitionMeter(max_public_calls)
    limits = freeze_joint_cranial_limits(source, meter=meter, max_bone_mm3=81.,
        max_preparation_nonbone_mm3=0., max_work_nonbone_mm3=0.,
        max_native_actions=2000, max_public_calls=max_public_calls,
        max_wall_seconds=120., max_bone_contact_replans=6)
    approach = bind_milling_approach(source, quaternion=[0., 0., 0., 1.],
        outside_plane_point_ras_mm=[32., 32., 55.], maximum_approach_mm=10.,
        point_scheme=point_scheme)
    if park_other:
        # This is explicit generated setup, not a pose invented by the macro.
        # The right hand is initially unposed; its first setup has no previous
        # physical route. Later left staging is always a native checked sweep.
        current, receipt = checked_joint_actions(source, [
            {'slot': 'right', 'operation': 'configure', 'duration': .1,
             'mechanicsRevision': 2, 'clockRevision': 1},
            {'slot': 'right', 'operation': 'place', 'duration': .1,
             'pose': {'position': [48., 48., 55.], 'quaternion': [0., 0., 0., 1.]}}],
            limits=limits, meter=meter, phase='generated.milling.explicit-other-hand-setup')
        assert receipt['status'] == 'accepted', receipt
    else:
        current = source
    return source, current, aperture, approach, meter, limits


def record_case(name, result, receipt):
    destination = os.environ.get('RESECTIONLAB_MILLING_TEST_EVIDENCE_DIR')
    if destination:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        with (path/(name+'.json')).open('x') as stream:
            json.dump({'receipt': receipt, 'finalNativeStateId': result.state_id,
                'actualRemovedCells': result.state['removed'],
                'actualDebris': result.state['debris']}, stream, allow_nan=False)


def run(current, approach, limits, **kwargs):
    return checked_joint_milling_opening(current, approach=approach, limits=limits,
        drill_seconds=30., suction_seconds=2., staging_seconds=.1, **kwargs)


@pytest.mark.parametrize('kwargs', [
    {'max_batches': 9}, {'max_batches': True}, {'max_contacts': 33},
    {'max_examined_cells': 2049}, {'max_contact_probes': 2049},
    {'max_contact_probes': 0}, {'point_scheme': 'unfrozen_random_probe'},
    {'staging_seconds': False}, {'slot': 'third'}])
def test_macro_bounds_are_explicit_and_never_relax_joint_limits(kwargs):
    _, current, _, approach, meter, limits = fixture(park_other=False)
    before, calls = current.snapshot(), meter.calls
    values = dict(drill_seconds=30., suction_seconds=2., staging_seconds=.1)
    values.update(kwargs)
    with pytest.raises(ValueError, match='EXPLICIT_BOUNDED_SPEC'):
        checked_joint_milling_opening(current, approach=approach, limits=limits, **values)
    assert current.snapshot() == before and meter.calls == calls


def test_initial_unposed_setup_once_then_only_swept_staging_commands():
    _, current, _, _, meter, _ = fixture(park_other=False)
    pose = {'position': [32., 32., 55.], 'quaternion': [0., 0., 0., 1.]}
    actions = [{'slot': 'left', 'operation': 'place', 'duration': .1, 'pose': pose},
               {'slot': 'left', 'operation': 'place', 'duration': .1, 'pose': pose}]
    transformed, modes = _checked_staging_program(current, actions, 'left', .3)
    assert [action['operation'] for action in transformed] == ['place', 'withdraw']
    assert transformed[1]['duration'] == .3
    assert modes == ['initial_unposed_exterior_setup', 'checked_fixed_attitude_exterior_translation']
    assert actions[1]['operation'] == 'place' and meter.calls == 0


def test_already_posed_attitude_change_is_refused_before_contact_generation():
    _, current, _, approach, meter, limits = fixture()
    quaternion = Rotation.from_euler('y', 20., degrees=True).as_quat().tolist()
    current, part = checked_joint_actions(current, [
        {'slot': 'left', 'operation': 'place', 'duration': .1,
         'pose': {'position': [32., 32., 55.], 'quaternion': quaternion}}],
        limits=limits, meter=meter)
    assert part['status'] == 'accepted'
    calls, before = meter.calls, current.snapshot()
    with pytest.raises(ValueError, match='POSED_ATTITUDE_STAGING_UNQUALIFIED'):
        run(current, approach, limits, max_batches=1, max_contacts=1)
    assert meter.calls == calls and current.snapshot() == before


def test_no_progress_clock_changes_stop_without_nominal_treatment_credit(monkeypatch):
    import resectionlab.surgical_joint_milling_opening as module
    _, current, _, approach, meter, limits = fixture()
    calls = []
    def empty_effect(*args, **kwargs):
        calls.append(kwargs)
        return {'contentHash': '0'*64,
                'actions': [{'slot': 'left', 'operation': 'inspect', 'duration': .1}]}
    monkeypatch.setattr(module, 'generate_milling_contact_proposals', empty_effect)
    before = current.snapshot()
    result, receipt = run(current, approach, limits)
    assert len(calls) == receipt['batchCount'] == 1
    assert receipt['reason'] == 'MILLING_NO_ACTUAL_BONE_OR_DEBRIS_PROGRESS'
    assert receipt['ledger']['boneRemovedMm3'] == receipt['ledger']['targetRemovedMm3'] == 0.
    assert result.state['elapsedSeconds'] > current.state['elapsedSeconds']
    assert receipt['publicCalls'] == 2 and not receipt['admittedBoneOpening']
    assert current.snapshot() == before


def test_native_refusal_stops_one_batch_and_keeps_preview_cost(monkeypatch):
    import resectionlab.surgical_joint_milling_opening as module
    _, current, _, approach, meter, limits = fixture()
    calls = []
    def bad_native(*args, **kwargs):
        calls.append(1)
        return {'contentHash': '0'*64, 'actions': [
            {'slot': 'left', 'operation': 'place', 'duration': .1,
             'pose': {'position': [32., 32., 60.], 'quaternion': [0., 0., 0., 1.]}}]}
    monkeypatch.setattr(module, 'generate_milling_contact_proposals', bad_native)
    before, spent = current.snapshot(), meter.calls
    result, receipt = run(current, approach, limits)
    assert receipt['status'] == 'refused' and 'PLACE_OUTSIDE_MATERIAL' in receipt['reason']
    assert len(calls) == 1 and receipt['publicCalls'] == 1
    assert meter.calls == spent+1 and result.state_id == current.state_id
    assert not receipt['appliedActions'] and receipt['ledger']['boneRemovedMm3'] == 0.
    assert current.snapshot() == before


def test_cancellation_uses_original_policy_and_no_new_calls():
    _, current, _, approach, meter, limits = fixture()
    before, spent = current.snapshot(), meter.calls
    def cancel():
        raise InterruptedError('generated cancellation')
    result, receipt = run(current, approach, limits, check=cancel)
    assert receipt['status'] == 'interrupted' and 'InterruptedError' in receipt['reason']
    assert result.state_id == current.state_id and meter.calls == spent
    assert current.snapshot() == before


def test_no_new_meter_or_resealed_limits_can_refund_a_prefix():
    _, current, _, approach, meter, limits = fixture()
    before, spent = current.snapshot(), meter.calls
    with pytest.raises(ValueError, match='SHARED_METER'):
        run(current, approach, limits, meter=TransitionMeter(4000))
    with pytest.raises(ValueError, match='REGISTERED_LIMIT_IDENTITY'):
        run(current, approach, replace(limits, max_bone_mm3=100.))
    assert current.snapshot() == before and meter.calls == spent


@pytest.fixture(scope='module')
def center_prefix():
    source, current, aperture, approach, meter, limits = fixture()
    before = current.snapshot()
    result, receipt = run(current, approach, limits, max_batches=1,
        max_contacts=32, max_examined_cells=81, max_contact_probes=81)
    record_case('center-default-native-prefix', result, receipt)
    return source, current, aperture, approach, meter, limits, before, result, receipt


def test_center_default_retains_truthful_full_A_controller_deficit(center_prefix):
    _, current, aperture, _, meter, limits, before, result, receipt = center_prefix
    assert receipt['status'] == 'refused' and receipt['reason'] == 'MILLING_BATCH_BUDGET'
    assert 0 < len(result.state['removed']) < len(aperture)
    assert set(receipt['remainingOriginalACells']) == aperture-set(result.state['removed'])
    assert not receipt['admittedBoneOpening'] and not receipt['admittedPatientTask']
    assert receipt['spec']['pointScheme'] == 'center'
    assert receipt['ledger']['boneRemovedMm3'] == len(result.state['removed'])
    assert receipt['ledger']['targetRemovedMm3'] == 0.
    assert milling_readiness(result)['historyVerified'] and milling_readiness(result)['debrisConserved']
    assert current.snapshot() == before and meter.calls > 0
    assert joint_cranial_ledger(result, limits)['boneRemovedMm3'] == receipt['ledger']['boneRemovedMm3']


def test_recorded_hash_excludes_wall_but_binds_actual_native_actions(center_prefix):
    *_, receipt = center_prefix
    assert receipt['specHash'] == semantic_digest(receipt['spec'])
    assert receipt['contentHash'] == _content_hash(receipt)
    altered = copy.deepcopy(receipt)
    altered['wallSeconds'] += 100.
    altered['ledger']['wallSecondsSinceMeterCreation'] += 100.
    assert _content_hash(altered) == receipt['contentHash']
    altered['appliedActions'][0]['duration'] += .1
    assert _content_hash(altered) != receipt['contentHash']


@pytest.fixture(scope='module')
def completed_quarter_opening():
    scheme = 'center_and_quarter_cell_corners'
    source, current, aperture, approach, meter, limits = fixture(point_scheme=scheme)
    before = current.snapshot()
    result, receipt = run(current, approach, limits, point_scheme=scheme,
        max_batches=8, max_contacts=32, max_examined_cells=81, max_contact_probes=729)
    record_case('quarter-whole81-native-opening', result, receipt)
    return source, current, aperture, approach, meter, limits, before, result, receipt


def test_declared_quarter_probes_complete_whole_native_opening_without_dural_or_target_credit(completed_quarter_opening):
    _, current, aperture, _, meter, _, before, result, receipt = completed_quarter_opening
    assert receipt['status'] == 'accepted', (receipt['reason'], receipt['nativeOpening'], receipt['remainingOriginalACells'])
    assert receipt['admittedBoneOpening'] and receipt['openingComplete']
    assert set(result.state['removed']) == aperture and not receipt['remainingOriginalACells']
    assert not result.state['debris'] and receipt['nativeOpening']['generatedBoneDebrisMm3'] == 81.
    assert receipt['nativeOpening']['debrisRemovedMm3'] == 81.
    assert receipt['nativeOpening']['debrisPresentMm3'] == receipt['nativeOpening']['debrisEscapedMm3'] == 0.
    assert receipt['nativeOpening']['historyVerified'] and receipt['nativeOpening']['debrisConserved']
    assert receipt['stageReadiness']['bonePrefixVerified'] and not receipt['stageReadiness']['duraCleared']
    assert receipt['wholeInstrumentExitComplete'] and not receipt['admittedPatientTask']
    assert receipt['ledger']['boneRemovedMm3'] == receipt['ledger']['wholeNonTargetRemovedMm3'] == 81.
    assert receipt['ledger']['preparationNonboneRemovedMm3'] == receipt['ledger']['workNonTargetRemovedMm3'] == receipt['ledger']['targetRemovedMm3'] == 0.
    assert receipt['batchCount'] <= 8 and any(batch['kind'] == 'current_tagged_debris' for batch in receipt['batches'])
    assert receipt['publicCalls'] == 2*receipt['nativeActionsAppended']
    assert sum(mode == 'initial_unposed_exterior_setup' for batch in receipt['batches'] for mode in batch.get('stagingModes', [])) == 1
    assert current.snapshot() == before


def test_already_complete_native_opening_adds_no_actions_or_calls(completed_quarter_opening):
    _, _, _, approach, meter, limits, _, result, _ = completed_quarter_opening
    calls, before = meter.calls, result.snapshot()
    again, receipt = run(result, approach, limits, point_scheme='center_and_quarter_cell_corners')
    assert receipt['status'] == 'accepted' and receipt['reason'] == 'ALREADY_COMPLETE_NATIVE_BONE_OPENING'
    assert receipt['admittedBoneOpening'] and receipt['publicCalls'] == 0
    assert not receipt['appliedActions'] and meter.calls == calls and again.snapshot() == before


@pytest.mark.parametrize('bound_scheme, argument', [
    ('center', 'center_and_quarter_cell_corners'),
    ('center_and_quarter_cell_corners', 'center')])
def test_explicit_point_scheme_declarations_cannot_silently_differ(bound_scheme, argument):
    _, current, _, approach, meter, limits = fixture(point_scheme=bound_scheme, park_other=False)
    before, calls = current.snapshot(), meter.calls
    with pytest.raises(ValueError, match='POINT_SCHEME_BINDING_DIFFERS'):
        run(current, approach, limits, point_scheme=argument)
    assert meter.calls == calls and current.snapshot() == before


def test_final_hash_readout_wall_crossing_invalidates_completed_opening(completed_quarter_opening, monkeypatch):
    import resectionlab.surgical_joint_milling_opening as module
    _, _, _, approach, meter, limits, _, result, _ = completed_quarter_opening
    late = False
    original_hash = module._content_hash
    def slow_hash(value):
        nonlocal late
        digest = original_hash(value)
        late = True
        return digest
    monkeypatch.setattr(module, '_content_hash', slow_hash)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda:
        limits.frozen_at+limits.max_wall_seconds+1 if late else time.monotonic()))
    before, calls = result.snapshot(), meter.calls
    branch, receipt = run(result, approach, limits, point_scheme='center_and_quarter_cell_corners')
    assert receipt['status'] == 'interrupted' and receipt['reason'] == 'WALL_BUDGET_AT_FINAL_READOUT'
    assert receipt['openingComplete'] and not receipt['admittedBoneOpening']
    assert not receipt['ledger']['withinLimits'] and 'WALL_BUDGET' in receipt['ledger']['refusalReasons']
    assert receipt['ledger']['wallSecondsSinceMeterCreation'] == limits.max_wall_seconds+1
    assert receipt['finalReadout']['wallOverrun']
    assert branch.snapshot() == before and meter.calls == calls
    assert receipt['contentHash'] == original_hash(receipt)


def test_cancellation_after_final_hash_retains_complete_physics_without_admission(completed_quarter_opening, monkeypatch):
    import resectionlab.surgical_joint_milling_opening as module
    _, _, _, approach, meter, limits, _, result, _ = completed_quarter_opening
    late = False
    original_hash = module._content_hash
    def hash_then_cancel(value):
        nonlocal late
        digest = original_hash(value)
        late = True
        return digest
    def check():
        if late:
            raise InterruptedError('generated cancellation after final receipt hash')
    monkeypatch.setattr(module, '_content_hash', hash_then_cancel)
    before, calls = result.snapshot(), meter.calls
    branch, receipt = run(result, approach, limits,
        point_scheme='center_and_quarter_cell_corners', check=check)
    assert receipt['status'] == 'interrupted' and 'after final receipt hash' in receipt['reason']
    assert receipt['openingComplete'] and not receipt['admittedBoneOpening']
    assert receipt['retentionRole'] == 'stopped_native_prefix_not_admitted'
    assert receipt['finalReadout']['cancelledOrInterrupted']
    assert branch.snapshot() == before and meter.calls == calls
    assert receipt['contentHash'] == original_hash(receipt)


def test_multileg_program_places_only_initial_global_and_exits_with_final_withdraw():
    from resectionlab.surgical_joint_milling_opening import _multileg_entry_actions, _multileg_exit_actions
    from test_surgical_milling_contact_proposals import staging_plate
    engine, _, _, plan = staging_plate()
    entry = _multileg_entry_actions(engine, plan, 'cranial_drill', 'left', .3)
    assert [a['operation'] for a in entry] == ['configure', 'place', 'move']
    assert entry[1]['pose'] == plan['globalPark'] and entry[-1]['pose'] == plan['inbound'][-1]
    # This pure program-shape control asserts no native ancestry or clearance.
    engine.state['slots']['left'].update(copy.deepcopy(plan['inbound'][-1]))
    engine.actions.append({'slot': 'left', 'operation': 'move'})
    exit_actions = _multileg_exit_actions(engine, plan, 'left', .3)
    assert [a['operation'] for a in exit_actions] == ['withdraw']
    assert exit_actions[-1]['pose'] == plan['globalPark']
    engine.state['slots']['left'].update(copy.deepcopy(plan['globalPark']))
    reentry = _multileg_entry_actions(engine, plan, 'frazier_suction', 'left', .3)
    assert [a['operation'] for a in reentry] == ['configure', 'exchange', 'move']
    assert reentry[1]['instrument'] == 'frazier_suction' and all(a['operation'] != 'place' for a in reentry)


def test_multileg_posed_start_matches_declared_parking_attitude_not_working_attitude():
    from resectionlab.surgical_joint_milling_opening import _initial_multileg_staging
    from resectionlab.surgical_milling_contact_proposals import bind_milling_staging_plan
    from test_surgical_milling_contact_proposals import staging_plate
    engine, _, approach, old = staging_plate()
    park = copy.deepcopy(old['globalPark'])
    park['quaternion'] = Rotation.from_euler('x', -90., degrees=True).as_quat().tolist()
    plan = bind_milling_staging_plan(engine, approach=approach, global_park=park,
        inbound=[old['globalPark'], *old['inbound']], outbound=[*old['outbound'], park])
    engine.state['slots']['left'].update(copy.deepcopy(park))
    engine.actions.append({'slot': 'left', 'operation': 'place'})
    # Pure pose validation only: the actual macro additionally authenticates
    # the frozen initial state and every native staging preview/commit.
    _initial_multileg_staging(engine, plan, 'left')
    engine.state['slots']['left']['position'][1] += 1
    with pytest.raises(ValueError, match='GLOBAL_START_POSE_DIFFERS'):
        _initial_multileg_staging(engine, plan, 'left')
    engine.state['slots']['left'].update(copy.deepcopy(park), active=True)
    with pytest.raises(ValueError, match='INITIAL_STAGING_UNQUALIFIED'):
        _initial_multileg_staging(engine, plan, 'left')


@pytest.mark.skipif(os.environ.get('RESECTIONLAB_STAGED_MILLING_NATIVE') != '1',
                    reason='Prospective full generated staging protocol requires explicit launch')
def test_generated_noncollinear_staged_controller_completes_actual81A_debris_and_global_exit():
    """Frozen generated native witness; not part of the pure implementation run."""
    from resectionlab.surgical_joint_cranial_exit import joint_exit_complete
    from resectionlab.surgical_milling_contact_proposals import bind_milling_staging_plan
    from test_surgical_milling_contact_proposals import staging_plate
    bare, aperture, _, plan = staging_plate()
    patch = tuple(int(np.ravel_multi_index((x, y, 64), bare.scene.shape))
                  for x in range(47, 50) for y in range(47, 50))
    rim = tuple(sorted(set(map(int, np.flatnonzero(bare.scene.labels == 3)))-set(patch)))
    proof = {'status': 'reviewed', 'provenance': 'Generated noncollinear plate staging; no patient evidence',
             'evidenceSha256': '6'*64}
    stage = freeze_cranial_stage_contract(bare, membrane_cells=patch,
        retained_membrane_cells=rim, membrane_anchor_cells=(rim[0],),
        work_cells=tuple(map(int, np.flatnonzero(bare.scene.labels == 2))),
        evidence={name: dict(proof) for name in PROOFS})
    source = SurgicalEngine(bare.scene, cranial_stages=stage)
    meter = TransitionMeter(4000)
    limits = freeze_joint_cranial_limits(source, meter=meter, max_bone_mm3=81.,
        max_preparation_nonbone_mm3=0., max_work_nonbone_mm3=0.,
        max_native_actions=2000, max_public_calls=4000, max_wall_seconds=300.,
        max_bone_contact_replans=6)
    approach = bind_milling_approach(source, quaternion=[0., 0., 0., 1.],
        outside_plane_point_ras_mm=[48., 48., 55.], maximum_approach_mm=10.,
        point_scheme='center_and_quarter_cell_corners')
    plan = bind_milling_staging_plan(source, approach=approach, global_park=plan['globalPark'],
        inbound=plan['inbound'], outbound=plan['outbound'])
    before = source.snapshot()
    result, receipt = run(source, approach, limits, staging_plan=plan,
        point_scheme='center_and_quarter_cell_corners', max_batches=8,
        max_contacts=32, max_examined_cells=81, max_contact_probes=729)
    record_case('noncollinear-staged-full-opening', result, receipt)
    assert receipt['status'] == 'accepted', receipt['reason']
    assert receipt['admittedBoneOpening'] and not receipt['admittedPatientTask']
    assert set(result.state['removed']) == aperture and not receipt['remainingOriginalACells']
    assert receipt['batchCount'] <= 8 and receipt['publicCalls'] <= 4000
    assert receipt['nativeActionsAppended'] <= 2000 and receipt['ledger']['withinLimits']
    assert receipt['ledger']['boneRemovedMm3'] == 81.
    assert receipt['ledger']['preparationNonboneRemovedMm3'] == receipt['ledger']['workNonTargetRemovedMm3'] == 0.
    opening = milling_readiness(result)
    assert opening['historyVerified'] and opening['debrisConserved'] and opening['openingComplete']
    assert opening['generatedBoneDebrisMm3'] == opening['debrisRemovedMm3'] == 81.
    assert opening['debrisPresentMm3'] == opening['debrisEscapedMm3'] == 0.
    assert joint_exit_complete(result) and receipt['stageReadiness']['bonePrefixVerified']
    operations = [a['operation'] for a in receipt['appliedActions']]
    assert operations.count('place') == 1 and operations.count('exchange') == 1
    assert all(t['native']['status'] == 'accepted' for t in receipt['stagingTransitions'])
    assert [t['kind'] for t in receipt['stagingTransitions']] == ['inbound', 'outbound', 'inbound', 'outbound']
    assert source.snapshot() == before


@pytest.mark.parametrize('bound,argument', [('surface_first', 'axial_diverse_current_a'),
                                         ('axial_diverse_current_a', 'surface_first')])
def test_cell_schedule_mode_mismatch_is_invalid_source_before_any_native_work(bound, argument):
    _, current, _, original, meter, limits = fixture(park_other=False)
    approach = bind_milling_approach(current, quaternion=original['quaternion'],
        outside_plane_point_ras_mm=original['outsidePlanePointRasMm'], maximum_approach_mm=10.,
        cell_schedule=bound)
    before, spent = current.snapshot(), meter.calls
    with pytest.raises(ValueError, match='CELL_SCHEDULE_BINDING_DIFFERS'):
        run(current, approach, limits, cell_schedule=argument)
    assert current.snapshot() == before and meter.calls == spent == 0


def test_axial_macro_binding_uses_original_guard_and_retains_cancelled_zero_work_prefix(monkeypatch):
    import resectionlab.surgical_joint_milling_opening as module
    _, current, _, old, meter, limits = fixture(park_other=False)
    approach = bind_milling_approach(current, quaternion=old['quaternion'],
        outside_plane_point_ras_mm=old['outsidePlanePointRasMm'], maximum_approach_mm=10.,
        cell_schedule='axial_diverse_current_a')
    calls = []
    def cancelled_binding(*args, **kwargs):
        assert kwargs['cell_schedule'] == 'axial_diverse_current_a' and callable(kwargs['check'])
        kwargs['check']()
        calls.append('guarded source binding')
        raise InterruptedError('generated axial binding cancellation')
    monkeypatch.setattr(module, 'bind_milling_approach', cancelled_binding)
    before = current.snapshot()
    result, receipt = run(current, approach, limits, cell_schedule='axial_diverse_current_a')
    assert calls == ['guarded source binding']
    assert receipt['status'] == 'interrupted' and 'axial binding cancellation' in receipt['reason']
    assert receipt['spec']['cellSchedule'] == 'axial_diverse_current_a'
    assert not receipt['admittedBoneOpening'] and not receipt['appliedActions']
    assert receipt['publicCalls'] == meter.calls == 0 and result.snapshot() == before


@pytest.mark.parametrize('schedule', [None, True, 'deepest_column', {'mode': 'axial_diverse_current_a'}])
def test_cell_schedule_rejects_unsupported_specs_without_relaxing_limits(schedule):
    _, current, _, approach, meter, limits = fixture(park_other=False)
    with pytest.raises(ValueError, match='EXPLICIT_BOUNDED_SPEC'):
        run(current, approach, limits, cell_schedule=schedule)
    assert meter.calls == 0 and not current.actions
