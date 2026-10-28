"""Generated geometry contracts for bounded three-direction public-engine access."""
import copy

import numpy as np
import pytest

from resectionlab.core import array_digest, semantic_digest
from resectionlab.surgical_access_planning import (
    _execute_brain_corridor, access_availability, compare_access_approaches)
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_planning import validate_saved_comparison
from resectionlab.surgical_scene import SurgicalScene


@pytest.mark.parametrize('oblique',[False,True])
def test_contact_limited_corridor_replans_from_observed_tip_instead_of_requested_depth(oblique):
    from resectionlab.surgical_cusa_v3 import default_profile
    from scipy.spatial.transform import Rotation
    labels=np.zeros((24,24,24),np.uint16)
    if oblique:
        labels[9,9,9]=1;labels[12,9,12]=2
        origin=np.asarray([9.,9.,9.]);inward=np.asarray([1.,0.,1.])/np.sqrt(2.)
        q=Rotation.align_vectors([inward],[[0.,0.,1.]])[0].as_quat().tolist()
        length=np.sqrt(18.);entry=[12,9,12];surface=9
    else:
        labels[9,9,12]=1;labels[9,9,9]=2
        origin=np.asarray([9.,9.,12.]);inward=np.asarray([0.,0.,-1.])
        q=[1.,0.,0.,0.];length=3.;entry=[9,9,9];surface=12
    profile=default_profile();profile['capacityMm3PerSecond']=.01
    scene=SurgicalScene(labels,np.eye(4),[
        {'id':1,'name':'brain','material':'soft_tissue'},
        {'id':2,'name':'target','material':'tumor','target':True}],
        {'kind':'generated contact-limited corridor'}, {'cusaV3':profile})
    engine=SurgicalEngine(standardize(scene))
    pose=lambda point:{'position':list(map(float,point)),'quaternion':q}
    route={'nativeAxis':2,'targetEntryIndex':entry,
        'pathType':'oblique_mapped_brain' if oblique else 'axis_mapped_brain',
        'surfacePose':pose(origin),'entryPose':pose(origin+inward*length),
        'outsidePose':pose(origin-inward*8.),'sourceBrainSurfaceIndex':surface,
        'inwardRasUnit':inward.tolist(),'pathLengthMm':length,'corridorRadiusMm':1.5}
    before=engine.snapshot()
    candidate,director=_execute_brain_corridor(engine,route,max_actions=20,
        healthy_cap=200.,check=lambda:None)
    assert any(row['reason']=='TISSUE_ASPIRATION_IN_PROGRESS' for row in candidate.history)
    assert np.dot(np.asarray(candidate.state['slots']['left']['position'])-origin,inward)<0.
    # Pending native contact consumes budget at the real boundary. Requested
    # endpoints cannot be counted as completed spatial progress.
    assert len(candidate.actions)==20
    assert director['status']=='stopped' and director['stopReason']=='brain_corridor_action_budget'
    assert not candidate.state['removed'] and candidate.state['cusaV3']['progressExact']
    assert engine.snapshot()==before
    replay=SurgicalEngine.restore(engine.scene,candidate.snapshot())
    assert replay.snapshot()==candidate.snapshot()


def _engine(*, skull=True, axis0_barrier=False, affine=None):
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[7:13, 7:13, 7:13] = 2
    regions = [{'id': 2, 'name': 'generated target', 'material': 'tumor', 'target': True}]
    if skull:
        for axis in range(3):
            plane = [slice(None)] * 3
            plane[axis] = 16
            labels[tuple(plane)] = 3
        regions.append({'id': 3, 'name': 'generated skull', 'material': 'bone'})
    if axis0_barrier:
        labels[14, :, :] = 1
        regions.append({'id': 1, 'name': 'generated obstruction', 'material': 'soft_tissue'})
    scene = standardize(SurgicalScene(labels, np.eye(4) if affine is None else affine,
        regions, {'provenance': 'generated software test; no patient observation'}, {}))
    return SurgicalEngine(scene)


def test_contact_limited_corridor_accumulates_capacity_then_resects_and_withdraws():
    from resectionlab.surgical_cusa_v3 import default_profile
    labels=np.zeros((24,24,24),np.uint16)
    labels[9,9,12]=1;labels[8:11,8:11,8:10]=2
    scene=standardize(SurgicalScene(labels,np.eye(4),[
        {'id':1,'name':'brain','material':'soft_tissue'},
        {'id':2,'name':'target','material':'tumor','target':True}],
        {'kind':'generated native CUSA contact continuation'}, {'cusaV3':default_profile()}))
    engine=SurgicalEngine(scene)
    pose=lambda z:{'position':[9.,9.,z],'quaternion':[1.,0.,0.,0.]}
    route={'nativeAxis':2,'targetEntryIndex':[9,9,9],
        'surfacePose':pose(12.),'entryPose':pose(9.),'outsidePose':pose(20.),
        'inwardRasUnit':[0.,0.,-1.],'sourceBrainSurfaceIndex':12,'corridorRadiusMm':1.5}
    candidate,director=_execute_brain_corridor(engine,route,max_actions=48,
        healthy_cap=200.,check=lambda:None)
    assert sum(row['status']=='partial' for row in candidate.history)>1
    assert candidate.metrics()['targetRemovedMm3']>=5.
    assert candidate.metrics()['nonTargetRemovedMm3']==1.
    assert director['stopReason']=='bounded_target_progress'
    assert not candidate.state['holds'] and all(not slot['inserted'] and not slot['active']
        for slot in candidate.state['slots'].values())
    assert candidate.state['cusaV3']['aspiratedSolidMm3']==len(candidate.state['removed'])*scene.voxel_mm3
    assert SurgicalEngine.restore(scene,candidate.snapshot()).snapshot()==candidate.snapshot()


def _compare(engine):
    return compare_access_approaches(engine, acknowledge_unknowns=True,
                                     max_actions_per_approach=16, max_wall_seconds=90.)


def test_three_real_skull_opening_and_target_routes_replay_from_one_intact_state():
    engine = _engine()
    before, labels_hash = engine.snapshot(), array_digest(engine.scene.labels)
    branches, report = _compare(engine)
    assert report['status'] == 'three_executable_source_approaches'
    assert report['directionFailures'] == [] and len(branches) == 3
    assert [plan['route']['nativeAxis'] for _, plan in branches] == [2, 0, 1]
    assert engine.snapshot() == before and array_digest(engine.scene.labels) == labels_hash
    for candidate, plan in branches:
        assert plan['startStateId'] == engine.state_id and plan['prefixActions'] == 0
        assert plan['route']['sourceSceneId'] == engine.scene.identity
        assert plan['route']['accessCompleteness'] == 'source_skull_and_target'
        assert plan['summary']['boneRemovedMm3'] > 0
        assert plan['summary']['targetRemovedMm3'] > 0
        assert plan['summary']['residualTargetMm3'] < 216.
        assert plan['residualTarget']['cells'] * engine.scene.voxel_mm3 == plan['residualTarget']['volumeMm3']
        assert plan['verification']['exactStateMatch'] and plan['verification']['endpointVectorMatch']
        assert len(candidate.actions) == len(plan['decisions']) <= 16
        assert candidate.history[0]['stateBefore'] == engine.state_id
    assert len({tuple(plan['route']['entryPose']['quaternion']) for _, plan in branches}) == 3
    record = {'branches': {}}
    for index, (candidate, plan) in enumerate(branches):
        plan['branchId'] = 'access-' + str(index)
        record['branches'][plan['branchId']] = candidate.snapshot()
    report['comparisonSeal'] = semantic_digest(report)
    validate_saved_comparison(engine.scene, record, report)
    bad = copy.deepcopy(report)
    bad['alternatives'][1]['route']['nativeAxis'] = 2
    bad.pop('comparisonSeal')
    bad['comparisonSeal'] = semantic_digest(bad)
    with pytest.raises(ValueError, match='route binding'):
        validate_saved_comparison(engine.scene, record, bad)


def test_missing_skull_uses_cusa_source_tissue_only_and_reports_unknown_access():
    engine = _engine(skull=False)
    availability = access_availability(engine)
    assert availability['accessAvailable'] and not availability['skullAvailable']
    before = engine.snapshot()
    branches, report = _compare(engine)
    assert report['status'] == 'three_source_tissue_routes_skull_unsupported'
    assert len(branches) == 3 and engine.snapshot() == before
    for candidate, plan in branches:
        assert plan['route']['accessCompleteness'] == 'unsupported_skull_access_source_tissue_only'
        assert plan['summary']['boneRemovedMm3'] is None
        assert plan['summary']['targetRemovedMm3'] > 0
        assert plan['truncated'] and plan['stopReason'] == 'unsupported_skull_access_bounded_source_tissue_progress'
        tool = 'cranial_drill'
        for action in candidate.actions:
            if action['slot'] == 'left' and action['operation'] == 'exchange':
                tool = action['instrument']
            assert action['operation'] != 'apply' or tool != 'cranial_drill'


def test_quantitative_outcome_binding_cannot_be_reassigned_to_another_action_history():
    engine = _engine(skull=False)
    branches, report = _compare(engine)
    record = {'branches': {}}
    for index, (candidate, plan) in enumerate(branches):
        plan['branchId'] = 'outcome-' + str(index)
        record['branches'][plan['branchId']] = candidate.snapshot()
    report['comparisonSeal'] = semantic_digest(report)
    validate_saved_comparison(engine.scene, record, report)
    for key, value in (('stateId', 'other'), ('actionCount', 0),
                       ('actionHistoryHash', 'other'), ('sourceLabelsHash', 'other')):
        changed = copy.deepcopy(report)
        changed['alternatives'][0]['modeledOutcomes'][key] = value
        changed.pop('comparisonSeal')
        changed['comparisonSeal'] = semantic_digest(changed)
        with pytest.raises(ValueError, match='modeled outcomes differ'):
            validate_saved_comparison(engine.scene, record, changed)


def test_infeasible_direction_is_reported_without_discarding_executable_baseline():
    engine = _engine(axis0_barrier=True)
    branches, report = _compare(engine)
    assert report['status'] == 'insufficient_distinct_routes'
    assert {plan['route']['nativeAxis'] for _, plan in branches} == {1, 2}
    assert report['directionFailures'] == [{'nativeAxis': 0, 'method': 'axis0-access',
        'reason': 'No source-skull and target-bearing opening exists'}]
    assert all(plan['summary']['targetRemovedMm3'] > 0 for _, plan in branches)
    record = {'branches': {}}
    for index, (candidate, plan) in enumerate(branches):
        plan['branchId'] = 'retained-' + str(index)
        record['branches'][plan['branchId']] = candidate.snapshot()
    report['comparisonSeal'] = semantic_digest(report)
    validate_saved_comparison(engine.scene, record, report)


def test_missing_target_and_oblique_grid_refuse_honestly_and_cancellation_is_atomic():
    affine = np.eye(4)
    affine[0, 1] = .1
    angled = _engine(affine=affine)
    assert not access_availability(angled)['accessAvailable']
    with pytest.raises(ValueError, match='diagonal'):
        _compare(angled)
    engine = _engine()
    before = engine.snapshot()
    def cancel():
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        compare_access_approaches(engine, acknowledge_unknowns=True, check=cancel)
    assert engine.snapshot() == before


def test_skull_stripped_target_behind_mapped_brain_keeps_three_failures_and_no_branches():
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[7:13, 7:13, 7:13] = 2
    for axis in range(3):
        plane = [slice(None)] * 3
        plane[axis] = 14
        labels[tuple(plane)] = 1
    scene = standardize(SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'mapped brain barrier', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'source target', 'material': 'tumor', 'target': True}],
        {'provenance': 'generated software test; no patient observation'}, {}))
    engine = SurgicalEngine(scene)
    before = engine.snapshot()
    branches, report = _compare(engine)
    assert branches == [] and report['status'] == 'insufficient_distinct_routes'
    assert {failure['nativeAxis'] for failure in report['directionFailures']} == {0, 1, 2}
    assert not report['availability']['skullAvailable']
    assert engine.snapshot() == before
    record = {'branches': {}}
    report['comparisonSeal'] = semantic_digest(report)
    validate_saved_comparison(scene, record, report)


def test_protected_target_cannot_be_promoted_from_geometric_capacity():
    from resectionlab.surgical_rl_protection import GuardConfig, ProtectionAnnotation, ProtectionSet
    plain = _engine(skull=False)
    scene = plain.scene
    protected = ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem',
        scene.labels == 2, provenance='generated exclusion test')])
    engine = SurgicalEngine(scene, protection=protected,
                            guard_config=GuardConfig(require_critical_coverage=False))
    before = engine.snapshot()
    branches, report = _compare(engine)
    assert branches == [] and report['status'] == 'insufficient_distinct_routes'
    assert {failure['nativeAxis'] for failure in report['directionFailures']} == {0, 1, 2}
    assert engine.snapshot() == before


def _brain_shell_engine():
    labels = np.zeros((30, 30, 30), np.uint16)
    labels[7:22, 7:22, 7:22] = 2
    for axis in range(3):
        plane = [slice(None)] * 3
        plane[axis] = 22
        labels[tuple(plane)] = 1
    scene = standardize(SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'mapped brain', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'source target', 'material': 'tumor', 'target': True}],
        {'provenance': 'generated software test; no patient observation'}, {}))
    return SurgicalEngine(scene)


def test_explicit_brain_corridor_opt_in_executes_harmful_bounded_real_actions():
    engine = _brain_shell_engine()
    before = engine.snapshot()
    branches, report = compare_access_approaches(engine, acknowledge_unknowns=True,
        allow_modeled_brain_corridor=True, max_actions_per_approach=16, max_wall_seconds=90.)
    assert report['status'] == 'three_source_tissue_routes_skull_unsupported'
    assert report['allowModeledBrainCorridor'] and report['limits']['maxAdditionalHealthyRemovedMm3'] == 200.
    assert len(branches) == 3 and engine.snapshot() == before
    assert {plan['route']['nativeAxis'] for _, plan in branches} == {0, 1, 2}
    for candidate, plan in branches:
        assert plan['route']['accessCompleteness'] == 'unsupported_skull_access_brain_corridor'
        assert all(row['entrySeparationMm'] >= 12. and row['inwardAngleDeg'] >= 30.
                   for row in plan['route']['noveltyComparisons'])
        assert plan['route']['sourceBrainThicknessMm'] > 0
        assert 0 < plan['summary']['healthyRemovedMm3'] <= 200.
        assert plan['summary']['targetRemovedMm3'] > 0
        assert plan['summary']['boneRemovedMm3'] is None
        assert plan['verification']['exactStateMatch']
        assert not any(a['instrument'] == 'cranial_drill' for a in candidate.actions
                       if a['operation'] == 'exchange')
    record = {'branches': {}}
    for index, (candidate, plan) in enumerate(branches):
        plan['branchId'] = 'corridor-' + str(index)
        record['branches'][plan['branchId']] = candidate.snapshot()
    report['comparisonSeal'] = semantic_digest(report)
    validate_saved_comparison(engine.scene, record, report)


def test_near_duplicate_brain_entries_do_not_fill_three_route_gate():
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[7:13, 7:13, 7:13] = 2
    for axis in range(3):
        plane = [slice(None)]*3
        plane[axis] = 13
        labels[tuple(plane)] = 1
    scene = standardize(SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'mapped brain', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'source target', 'material': 'tumor', 'target': True}],
        {'provenance': 'generated near-duplicate route test'}, {}))
    engine = SurgicalEngine(scene)
    branches, report = compare_access_approaches(engine, acknowledge_unknowns=True,
        allow_modeled_brain_corridor=True, max_actions_per_approach=16)
    assert report['status'] == 'insufficient_distinct_routes'
    assert len(branches) < 3
    assert report['limits']['minimumBrainEntrySeparationMm'] == 12.
    assert report['limits']['minimumBrainCorridorAngleDeg'] == 30.
    assert any('bounded candidate search' in row['reason'] for row in report['directionFailures'])


def test_brain_corridor_cap_and_protection_refuse_without_mutating_source():
    engine = _brain_shell_engine()
    before = engine.snapshot()
    branches, report = compare_access_approaches(engine, acknowledge_unknowns=True,
        allow_modeled_brain_corridor=True, max_actions_per_approach=16,
        max_healthy_removed_mm3=1., max_wall_seconds=90.)
    assert branches == [] and report['status'] == 'insufficient_distinct_routes'
    assert all('HEALTHY_REMOVAL_CAP' in row['reason'] for row in report['directionFailures'])
    assert engine.snapshot() == before
    from resectionlab.surgical_rl_protection import GuardConfig, ProtectionAnnotation, ProtectionSet
    scene = engine.scene
    protection = ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem',
        scene.labels == 1, provenance='generated brain exclusion')])
    protected = SurgicalEngine(scene, protection=protection,
                               guard_config=GuardConfig(require_critical_coverage=False))
    protected_before = protected.snapshot()
    branches, report = compare_access_approaches(protected, acknowledge_unknowns=True,
        allow_modeled_brain_corridor=True, max_actions_per_approach=16, max_wall_seconds=90.)
    assert branches == [] and report['status'] == 'insufficient_distinct_routes'
    assert all('PROTECTED_STRUCTURE' in row['reason'] for row in report['directionFailures'])
    assert protected.snapshot() == protected_before


def test_arbitrary_soft_tissue_cannot_be_recast_as_brain_corridor():
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[7:13, 7:13, 7:13] = 2
    for axis in range(3):
        plane = [slice(None)]*3
        plane[axis] = 13
        labels[tuple(plane)] = 1
    scene = standardize(SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'mapped fascia', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'source target', 'material': 'tumor', 'target': True}],
        {'provenance': 'generated software test; no patient observation'}, {}))
    engine = SurgicalEngine(scene)
    assert not access_availability(engine)['modeledBrainLayerAvailable']
    branches, report = compare_access_approaches(engine, acknowledge_unknowns=True,
        allow_modeled_brain_corridor=True, max_actions_per_approach=16)
    assert branches == []
    assert len(report['directionFailures']) == 3


def test_checked_slow_withdrawal_recovers_a_tight_oblique_corridor():
    from scipy.spatial.transform import Rotation
    labels = np.zeros((40, 40, 40), np.uint16)
    labels[4:27, 4:27, 4:27] = 1
    labels[14:17, 17:20, 13:16] = 2
    scene = standardize(SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'brain', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'tumor', 'material': 'tumor', 'target': True}],
        {'provenance': 'generated tight-corridor software test'}, {}))
    engine = SurgicalEngine(scene)
    surface = np.array([20., 20., 26.])
    goal = np.array([15., 18., 14.])
    length = float(np.linalg.norm(goal-surface))
    inward = (goal-surface)/length
    quaternion = Rotation.align_vectors([inward], [[0., 0., 1.]])[0].as_quat().tolist()
    outside = surface-inward*8.
    route = {'nativeAxis': 2, 'pathType': 'oblique_mapped_brain',
        'targetEntryIndex': goal.astype(int).tolist(), 'sourceBrainSurfaceIndex': 26,
        'entryPose': {'position': goal.tolist(), 'quaternion': quaternion},
        'outsidePose': {'position': outside.tolist(), 'quaternion': quaternion},
        'surfacePose': {'position': surface.tolist(), 'quaternion': quaternion},
        'inwardRasUnit': inward.tolist(), 'pathLengthMm': length,
        'corridorRadiusMm': .6,
        'accessCompleteness': 'unsupported_skull_access_brain_corridor'}
    before = engine.snapshot()
    candidate, director = _execute_brain_corridor(engine, route, max_actions=48,
        healthy_cap=200., check=lambda: None)
    recovery = director['withdrawalRecovery']
    assert recovery['fastPreviewRejection'] == 'SHAFT_COLLISION'
    assert recovery['slowStatus'] == 'accepted'
    assert candidate.actions[-1]['operation'] == 'withdraw'
    assert candidate.actions[-1]['pose'] == route['outsidePose']
    assert not candidate.state['slots']['left']['inserted']
    assert engine.snapshot() == before
    replay = engine.fresh()
    for action in candidate.actions[:-1]:
        assert replay.step(action)['status'] == 'accepted'
    previous = replay.metrics()
    fast = dict(candidate.actions[-1], duration=recovery['fastDurationSeconds'])
    assert replay.preview(fast, include_view=False)['reason'] == 'SHAFT_COLLISION'
    slow = replay.preview(candidate.actions[-1], include_view=False)
    assert slow['status'] == 'accepted'
    assert replay.step(candidate.actions[-1])['stateId'] == candidate.state_id == slow['stateId']
    assert replay.metrics()['nonTargetRemovedMm3'] == previous['nonTargetRemovedMm3']
    assert replay.metrics()['targetRemovedMm3'] == previous['targetRemovedMm3']


def test_desktop_bridge_installs_selectable_source_tissue_routes_without_skull(tmp_path):
    from resectionlab.core import CaseData, SourceRef
    from resectionlab.desktop_bridge import BridgeSession, _Request
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[7:13, 7:13, 7:13] = 1
    case = CaseData('generated-access-bridge', labels.astype('float32'),
        {'target': labels == 1}, np.eye(4),
        (SourceRef('generated', 'simulated:access-bridge', provenance='simulated'),),
        metadata={'anatomy_layers': {'target': {'kind': 'tumor'}}})
    session = BridgeSession(tmp_path / 'transfers')
    try:
        session._install_case(case, {}, _Request('fixture-install'))
        def command(name, prior=None, **extra):
            args = {'caseHash': case.semantic_hash, 'command': name, **extra}
            if prior is not None:
                args['expectedStateId'] = prior['view']['stateId']
            return session.execute('surgicalRehearsal', args, _Request(name), lambda *_: None)
        created = command('create')
        assert created['planningAvailability']['accessAvailable']
        assert not created['planningAvailability']['skullAvailable']
        planned = command('plan', created, opponent='anatomy-access', acknowledgeResearchUnknowns=True)
        assert planned['planning']['status'] == 'three_source_tissue_routes_skull_unsupported'
        assert planned['branchId'] == created['branchId']
        assert planned['view']['stateId'] == created['view']['stateId']
        assert len(planned['planning']['alternatives']) == 3
        alternative = planned['planning']['alternatives'][1]
        selected = command('selectBranch', planned,
                           branchId=alternative['branchId'], position=len(alternative['decisions']))
        assert selected['view']['metrics']['targetRemovedMm3'] > 0
        assert selected['view']['stateId'] == planned['planning']['alternatives'][1]['finalStateId']
        saved_path = tmp_path / 'generated-access.ressectionlab'
        session.execute('saveCase', {'caseHash': case.semantic_hash, 'path': str(saved_path),
            'workspace': {}, 'overwrite': False}, _Request('save'), lambda *_: None)
        reopened = BridgeSession(tmp_path / 'reopen-transfers')
        try:
            loaded = reopened.execute('loadCase', {'path': str(saved_path)},
                                      _Request('load'), lambda *_: None)
            observed = reopened.execute('surgicalRehearsal', {'caseHash': loaded['caseHash'],
                'command': 'inspect'}, _Request('inspect'), lambda *_: None)
            assert observed['planning']['status'] == 'three_source_tissue_routes_skull_unsupported'
            assert observed['view']['stateId'] == selected['view']['stateId']
        finally:
            reopened.close()
    finally:
        session.close()
