"""Generated native tests of source-bound proposals; no patient qualification."""
import copy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.surgical_craniotomy_skills import checked_craniotomy_actions, freeze_bone_opening_limits
from resectionlab.surgical_milling import PROPERTY, milling_readiness
from resectionlab.surgical_milling_contact_proposals import (
    bind_milling_approach, generate_milling_contact_proposals, generate_milling_debris_proposals, proposal_hash)
from resectionlab.surgical_transition_meter import TransitionMeter
from test_surgical_milling import rebuild
from test_surgical_milling_native import milling_shell


def generated(*, world_rotation=None):
    engine, _ = milling_shell()
    raw = copy.deepcopy(engine.scene.properties[PROPERTY])
    affine = np.diag([.5, .5, 1., 1.])
    raw['entry']['centerRasMm'] = [16., 16., 59.5]
    tilt = Rotation.from_rotvec(np.deg2rad(20.)*np.array([-1., 1., 0.])/np.sqrt(2))
    outside = np.array([16., 16., 45.])
    if world_rotation is not None:
        transform = np.eye(4)
        transform[:3, :3] = world_rotation.as_matrix()
        transform[:3, 3] = [50., -20., 13.]
        affine = transform@affine
        raw['entry']['centerRasMm'] = (world_rotation.apply(raw['entry']['centerRasMm'])+transform[:3, 3]).tolist()
        for key in ('normalRasUnit', 'uRasUnit'):
            raw['entry'][key] = world_rotation.apply(raw['entry'][key]).tolist()
        outside = world_rotation.apply(outside)+transform[:3, 3]
        tilt = world_rotation*tilt
    engine, _ = rebuild(engine, raw, affine=affine)
    approach = bind_milling_approach(engine, quaternion=tilt.as_quat().tolist(),
        outside_plane_point_ras_mm=outside.tolist(), maximum_approach_mm=25.)
    return engine, approach


def propose(engine, approach, **kwargs):
    return generate_milling_contact_proposals(engine, approach=approach,
        drill_seconds=30., suction_seconds=2., **kwargs)


def test_explicit_axis_aligned_circumsphere_formula_is_not_a_native_contact_guarantee():
    # Retain the failed circumsphere-only baseline, then invert the native
    # closed-cell kernel with the same stock tool and fixed .10mm advance.
    engine, _ = milling_shell()
    approach = bind_milling_approach(engine, quaternion=[0., 0., 0., 1.],
        outside_plane_point_ras_mm=[32., 32., 55.], maximum_approach_mm=10.)
    before = engine.snapshot()
    result = propose(engine, approach, max_contacts=1)
    assert result['contacts'] and result['actions']
    assert all(not row['circumsphereOnlyNativeCellHit'] for row in result['inverseContactDiagnostics'])
    contact = result['contacts'][0]
    assert contact['drillPose']['position'][2] == pytest.approx(57.6)
    assert contact['suctionPose']['position'][2] == pytest.approx(58.1)
    assert contact['cell'] in engine.cells(contact['drillPose']['position'], contact['drillPose']['position'], 2.)
    assert result['nominalUncoveredCurrentACells']
    assert result['controllerDeficit'] and result['nativeCalls'] == 0
    assert result['contentHash'] == proposal_hash(result)
    second_approach = bind_milling_approach(engine, quaternion=[0., 0., 0., 1.],
        outside_plane_point_ras_mm=[32., 32., 55.], maximum_approach_mm=10.)
    assert second_approach['sourceOrderingHash'] == approach['sourceOrderingHash']
    assert propose(engine, second_approach, max_contacts=1)['contentHash'] == result['contentHash']
    assert not result['openingComplete'] and not result['admittedOpening']
    assert engine.snapshot() == before


def test_oblique_affine_proposals_are_deterministic_json_and_bound_to_actual_source():
    rotation = Rotation.from_euler('xyz', [17., -11., 28.], degrees=True)
    engine, approach = generated(world_rotation=rotation)
    before = engine.snapshot()
    result = propose(engine, approach, max_contacts=2)
    again = propose(engine, approach, max_contacts=2)
    assert {k: v for k, v in result.items() if k != 'sourceGeometryCost'} == {
        k: v for k, v in again.items() if k != 'sourceGeometryCost'}
    assert json.loads(json.dumps(result)) == result
    assert result['contacts'] and result['sceneId'] == engine.scene.identity
    direction = Rotation.from_quat(approach['quaternion']).apply([0., 0., 1.])
    for contact in result['contacts']:
        p = np.asarray(contact['drillPose']['position'])
        center = engine.scene.points([contact['cell']])[0]
        assert np.allclose(center-p, direction*(contact['nativeLeadingDrillContactDistanceMm']-.1))
        assert contact['cell'] in engine.cells(p, p, 2.)
        suction = np.asarray(contact['suctionPose']['position'])
        assert np.allclose(center-suction, direction*(contact['nativeLeadingSuctionContactDistanceMm']-.1))
        assert contact['cell'] in engine.cells(suction, suction, 1.5)
        assert not np.allclose(suction-p, .5*direction)
        assert contact['nativeRemovalRequiredBeforeSuction']
        assert all('parameters' not in action for action in contact['actions'])
    assert result['nativeVerificationRequired'] and not result['admittedOpening']
    assert engine.snapshot() == before
    plain, _ = generated()
    with pytest.raises(ValueError, match='APPROACH_SOURCE_BINDING'):
        propose(plain, approach)
    wrong = copy.deepcopy(approach)
    wrong['quaternion'] = (rotation*Rotation.from_euler('x', 45., degrees=True)).as_quat().tolist()
    with pytest.raises(ValueError, match='ATTITUDE_OUTSIDE_ENTRY_CONE'):
        propose(engine, wrong)


@pytest.fixture(scope='module')
def executed_generated_contact():
    """The actual generated actions go through existing preview/step/meter policy."""
    engine, _ = milling_shell()
    approach = bind_milling_approach(engine, quaternion=[0., 0., 0., 1.],
        outside_plane_point_ras_mm=[32., 32., 55.], maximum_approach_mm=10.)
    source = engine.snapshot()
    proposal = propose(engine, approach, max_contacts=1)
    assert len(proposal['contacts']) == 1
    meter = TransitionMeter(100)
    limits = freeze_bone_opening_limits(engine, max_total_bone_mm3=81.,
        max_native_actions=50, max_public_calls=100, max_wall_seconds=90., meter=meter)
    candidate, receipt = checked_craniotomy_actions(engine, proposal['actions'], limits=limits)
    assert engine.snapshot() == source
    return engine, approach, proposal, candidate, receipt, meter


def test_generated_program_performs_finite_native_work_and_never_admits_partial_A(executed_generated_contact):
    engine, _, proposal, candidate, receipt, meter = executed_generated_contact
    removed = set(candidate.state['removed'])
    assert removed and removed == set(proposal['contacts'][0]['nominalCurrentAContactCells'])
    assert removed < set(engine._cranial_access.aperture_cells)
    assert any(row.get('capacityPartial') for row in receipt['attempts'])
    assert receipt['nativeActionsAppended'] > len(proposal['actions'])
    assert receipt['status'] == 'refused' and receipt['reason'] == 'COMPLETE_NATIVE_OPENING_REQUIRED'
    assert not receipt['admittedOpening'] and not receipt['admittedBoneOpening']
    status = milling_readiness(candidate)
    assert status['historyVerified'] and status['debrisConserved']
    assert status['generatedBoneDebrisMm3'] > status['debrisRemovedMm3'] > 0
    assert status['debrisPresentMm3'] > 0
    assert not status['apertureRemoved'] and not status['apertureCleared'] and not status['openingComplete']
    assert all(not slot['inserted'] and not slot['active'] for slot in candidate.state['slots'].values())
    assert candidate.metrics()['targetRemovedMm3'] == 0
    assert meter.calls == 2*receipt['nativeActionsAppended']
    # The source remains intact, while the stopped native prefix remains charged.
    assert not engine.state['removed'] and meter.calls > 0


def test_current_native_cavity_changes_shortlist_without_ledger_or_debris_shortcuts(executed_generated_contact):
    _, approach, old, candidate, _, meter = executed_generated_contact
    before_calls, before = meter.calls, candidate.snapshot()
    result = propose(candidate, approach, max_contacts=1)
    assert result['sourceStateId'] != old['sourceStateId']
    assert result['sourceActionsSeal'] != old['sourceActionsSeal']
    assert set(result['remainingOriginalACells']) == set(old['remainingOriginalACells'])-set(candidate.state['removed'])
    assert result['contacts'][0]['cell'] != old['contacts'][0]['cell']
    assert not set(result['examinedCells']) & set(candidate.state['removed'])
    assert result['controllerDeficit'] and result['nativeOpening']['debrisPresentMm3'] > 0
    assert result['currentDebrisReplanRequired'] and result['currentDebrisCells']
    assert not result['openingComplete'] and not result['admittedOpening']
    assert candidate.snapshot() == before and meter.calls == before_calls


def test_explicit_bounds_retain_entire_A_and_report_unexamined_deficit():
    engine, approach = generated()
    result = propose(engine, approach, max_examined_cells=5, max_contacts=1)
    assert len(result['examinedCells']) == 5
    assert len(result['unexaminedCurrentACells']) == 76
    assert len(result['remainingOriginalACells']) == 81
    assert result['matchingSuctionChecks'] <= 1 and len(result['contacts']) <= 1
    assert result['nominalUncoveredCurrentACells'] and result['controllerDeficit']
    for kwargs in ({'max_contacts': 33}, {'max_examined_cells': 2049}, {'max_contacts': True}):
        with pytest.raises(ValueError, match='BOUNDS_REQUIRED'):
            propose(engine, approach, **kwargs)


@pytest.mark.parametrize('membership', ['protected', 'original_target', 'unknown_original_target'])
def test_no_shortlist_can_trim_global_protected_or_original_target_A(membership):
    from resectionlab.core import array_digest, immutable_array
    from resectionlab.surgical_scene import _bind_original_target_memberships
    engine, approach = generated()
    cell = engine._cranial_access.aperture_cells[0]
    if membership == 'protected':
        engine._protected_ids = frozenset([cell])
        reason = 'PROTECTED_STRUCTURE'
    else:
        mask = engine.scene.labels == 2
        mask.ravel()[cell] = True
        raw = copy.deepcopy(engine.scene.properties[PROPERTY])
        engine, _ = rebuild(engine, raw, source={**engine.scene.source,
            'annotationHashes': {'Generated target': array_digest(mask)}})
        if membership == 'original_target':
            _bind_original_target_memberships(engine.scene,
                {'Generated target': immutable_array(np.flatnonzero(mask), np.int64)})
        reason = 'TARGET_OVERLAP' if membership == 'original_target' else 'ORIGINAL_TARGET_EVIDENCE_REQUIRED'
    before = engine.snapshot()
    with pytest.raises(ValueError, match=reason):
        propose(engine, approach, max_contacts=1)
    assert engine.snapshot() == before


def test_exterior_plane_and_current_shafts_are_checked_without_assuming_A_removed():
    engine, approach = generated()
    bad = copy.deepcopy(approach)
    bad['outsidePlanePointRasMm'] = [16., 16., 58.]
    # Different cell lateral offsets can project either side of this plane;
    # none creates an eligible whole-stock exterior staging pose.
    result = propose(engine, bad)
    assert not result['contacts']
    assert any(row['reason'] == 'OUTSIDE_POSE_NOT_AT_ORIGINAL_EXIT' for row in result['rejections'])
    # Native flags alone cannot create an accepted cavity for shaft entry.
    engine.state['removed'] = list(engine._cranial_access.aperture_cells)
    with pytest.raises(ValueError, match='NATIVE_CAVITY_HISTORY_REQUIRED'):
        propose(engine, approach)


def test_deeper_A_cell_does_not_let_a_shaft_pass_through_current_unremoved_A():
    engine, approach = generated()
    labels = engine.scene.labels.copy()
    labels[32, 32, 65] = 1
    deep = int(np.ravel_multi_index((32, 32, 65), labels.shape))
    raw = copy.deepcopy(engine.scene.properties[PROPERTY])
    raw['apertureCells'] = sorted([*raw['apertureCells'], deep])
    engine, _ = rebuild(engine, raw, labels=labels)
    approach = bind_milling_approach(engine, quaternion=approach['quaternion'],
        outside_plane_point_ras_mm=approach['outsidePlanePointRasMm'], maximum_approach_mm=25.)
    before = engine.snapshot()
    result = propose(engine, approach)
    rejection = next(row for row in result['rejections'] if row['cell'] == deep)
    assert rejection['reason'] == 'CURRENT_CAVITY_SHAFT_OCCUPIED'
    assert rejection['witnessCell'] in engine._cranial_access.aperture_cells
    assert deep in result['nominalUncoveredCurrentACells'] and result['controllerDeficit']
    assert engine.snapshot() == before


@pytest.mark.parametrize('material', ['unknown', 'soft_tissue'])
def test_actual_nonbone_and_unknown_working_footprint_are_refused(material):
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_milling import binding, contract_hash
    from resectionlab.surgical_scene import SurgicalScene
    engine, old_approach = generated()
    labels = engine.scene.labels.copy()
    labels[31, 31, 59] = 3
    obstruction = int(np.ravel_multi_index((31, 31, 59), labels.shape))
    regions = [*engine.scene.regions, {'id': 3, 'name': 'Explicit brain or unknown footprint', 'material': material}]
    props = copy.deepcopy(engine.scene.properties)
    raw = props.pop(PROPERTY)
    raw['exit']['planeVoxelBoundary'] = 58.  # Bound the new original obstruction too.
    base = SurgicalScene(labels, engine.scene.affine, regions, engine.scene.source, props)
    raw.update(binding(base))
    raw['contentHash'] = contract_hash(raw)
    engine = SurgicalEngine(SurgicalScene(labels, base.affine, regions, base.source, {**props, PROPERTY: raw}))
    approach = bind_milling_approach(engine, quaternion=old_approach['quaternion'],
        outside_plane_point_ras_mm=old_approach['outsidePlanePointRasMm'], maximum_approach_mm=25.)
    before = engine.snapshot()
    result = propose(engine, approach)
    assert any(row.get('witnessCell') == obstruction and
               row['reason'] == 'WORKING_CONTACT_OUTSIDE_ALLOWED_CURRENT_A' for row in result['rejections'])
    assert all(obstruction not in row['nominalCurrentAContactCells'] for row in result['contacts'])
    assert engine.snapshot() == before


def test_sheared_native_distance_bound_is_not_claimed_as_exact_contact():
    engine, old_approach = generated()
    affine = engine.scene.affine.copy()
    affine[0, 1] = .1
    raw = copy.deepcopy(engine.scene.properties[PROPERTY])
    raw['entry']['centerRasMm'] = [19.2, 16., 59.5]
    engine, _ = rebuild(engine, raw, affine=affine)
    approach = bind_milling_approach(engine, quaternion=old_approach['quaternion'],
        outside_plane_point_ras_mm=[19.2, 16., 45.], maximum_approach_mm=25.)
    before = engine.snapshot()
    with pytest.raises(ValueError, match='SHEARED_AFFINE_CONTACT_UNSUPPORTED'):
        propose(engine, approach)
    assert engine.snapshot() == before


def test_all_A_outward_order_is_explicit_cached_and_not_a_permanent_first_ID_trap(monkeypatch):
    engine, old_approach = generated()
    affine = engine.scene.affine.copy()
    affine[0, 0], affine[0, 3] = -.5, 32.
    raw = copy.deepcopy(engine.scene.properties[PROPERTY])
    engine, _ = rebuild(engine, raw, affine=affine)
    approach = bind_milling_approach(engine, quaternion=old_approach['quaternion'],
        outside_plane_point_ras_mm=[16., 16., 45.], maximum_approach_mm=25.)
    order = approach['sourceOrdering']
    assert order['orderingCells'] == 81 and order['orderingWallSeconds'] > 0
    axis = Rotation.from_quat(approach['quaternion']).apply([0., 0., 1.])
    expected = min(engine._cranial_access.aperture_cells,
                   key=lambda cell: (round(float(engine.scene.points([cell])[0]@axis), 9), cell))
    assert order['orderedCells'][0] == expected
    assert expected > min(engine._cranial_access.aperture_cells)
    calls, original = [], engine.scene.points
    def points(ids):
        calls.append(len(ids))
        return original(ids)
    monkeypatch.setattr(engine.scene, 'points', points)
    result = propose(engine, approach, max_examined_cells=1, max_contacts=1)
    assert result['examinedCells'] == [expected]
    assert result['orderingCells'] == 81 and result['examinedContactCells'] == 1
    assert result['orderingCoordinatesComputedThisCall'] == 0
    assert result['sourceGeometryCost']['orderingReused']
    assert result['sourceGeometryCost']['generationWallSeconds'] > 0
    assert calls and max(calls) == 1
    bad = copy.deepcopy(approach)
    bad['sourceOrdering']['orderedCells'].reverse()
    with pytest.raises(ValueError, match='SOURCE_ORDERING_BINDING_REQUIRED'):
        propose(engine, bad)


def test_generated_actual_debris_followup_drains_leftover_and_keeps_residual_A_refused():
    engine, _ = milling_shell()
    approach = bind_milling_approach(engine, quaternion=[0., 0., 0., 1.],
        outside_plane_point_ras_mm=[32., 32., 55.], maximum_approach_mm=10.)
    proposal = propose(engine, approach, max_contacts=1)
    meter = TransitionMeter(100)
    limits = freeze_bone_opening_limits(engine, max_total_bone_mm3=81.,
        max_native_actions=50, max_public_calls=100, max_wall_seconds=90., meter=meter)
    prefix, first = checked_craniotomy_actions(engine, proposal['actions'], limits=limits)
    assert prefix.state['debrisMm3'] == pytest.approx(3.)
    spent = meter.calls
    before = prefix.snapshot()
    cleanup = generate_milling_debris_proposals(prefix, approach=approach,
        suction_seconds=2., max_contacts=1)
    assert cleanup['contacts'] and cleanup['nativeCalls'] == 0
    assert cleanup['currentDebrisMm3'] == pytest.approx(3.)
    assert cleanup['contentHash'] == proposal_hash(cleanup)
    assert cleanup['sourceStateId'] == prefix.state_id and cleanup['currentDebrisSeal']
    assert cleanup['orderingCells'] == len(prefix.state['debris'])
    assert cleanup['examinedContactCells'] <= 2048
    assert cleanup['nominalWholeTaggedDebrisContactCoverage']
    assert prefix.snapshot() == before and meter.calls == spent
    # Existing bone-only policy binds one complete source program. Replay the
    # generated prefix + computed continuation on its same frozen meter; the
    # first discarded probe remains charged, and no new budget is created.
    final, receipt = checked_craniotomy_actions(engine,
        proposal['actions']+cleanup['actions'], limits=limits)
    status = milling_readiness(final)
    assert final.state['removed'] == prefix.state['removed']
    assert not final.state['debris'] and status['debrisPresentMm3'] == 0
    assert status['generatedBoneDebrisMm3'] == status['debrisRemovedMm3'] == 5.
    assert status['debrisEscapedMm3'] == 0 and status['debrisConserved'] and status['historyVerified']
    assert all(not slot['inserted'] and not slot['active'] for slot in final.state['slots'].values())
    assert not status['apertureRemoved'] and not status['openingComplete']
    assert receipt['reason'] == 'COMPLETE_NATIVE_OPENING_REQUIRED' and not receipt['admittedOpening']
    assert meter.calls == spent+2*receipt['nativeActionsAppended']
    cleared = generate_milling_debris_proposals(final, approach=approach, suction_seconds=2.)
    assert not cleared['actions'] and not cleared['currentTaggedDebris']
    assert cleared['controllerDeficit'] and not cleared['openingComplete']


def test_native_A_gone_with_real_debris_still_generates_suction_candidates():
    from test_surgical_milling_native import mill_aperture
    engine, _ = milling_shell()
    approach = bind_milling_approach(engine, quaternion=[0., 0., 0., 1.],
        outside_plane_point_ras_mm=[32., 32., 55.], maximum_approach_mm=10.)
    meter = TransitionMeter(160)
    current = meter.wrap(engine).clone()
    mill_aperture(current)
    status = milling_readiness(current)
    assert status['apertureRemoved'] and status['debrisPresentMm3'] == 81.
    assert not status['openingComplete']
    calls, before = meter.calls, current.snapshot()
    result = generate_milling_debris_proposals(current, approach=approach,
        suction_seconds=2., max_examined_cells=81, max_contacts=1)
    assert not result['remainingOriginalACells']
    assert result['contacts'] and result['actions']
    assert result['currentDebrisMm3'] == 81. and result['controllerDeficit']
    assert not result['openingComplete'] and not result['admittedOpening']
    assert result['orderingCells'] == 81 and result['examinedContactCells'] == 81
    assert not result['unexaminedTaggedDebrisCells']
    assert result['nominalUncoveredTaggedDebrisCells']
    assert current.snapshot() == before and meter.calls == calls
