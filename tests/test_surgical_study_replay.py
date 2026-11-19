"""Replay joins an exact native target and remains read-only across a saved workspace."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.imaging import save_case
from resectionlab.surgical_study_replay import verify_recording
from test_surgical_task_admission import prepared
from test_anatomy_operative_field import study


def test_replay_join_refuses_foreign_patient_target_pixels_or_noncausal_endpoint(prepared):
    case, _, record, native = prepared[0]
    with pytest.raises(ValueError, match='patient or native study revision'):
        verify_recording(prepared[1][3], 'tumor', case, record)
    with pytest.raises(ValueError, match='Select the recorded target'):
        verify_recording(native, 'brain', case, record)
    substitute = replace(case, mri=np.zeros_like(case.mri))
    with pytest.raises(ValueError, match='reference image differs'):
        verify_recording(native, 'tumor', substitute, record)
    fabricated = copy.deepcopy(record)
    fabricated['branches'][record['branchId']]['stateId'] = 'sha256:' + '0'*64
    with pytest.raises(ValueError, match='Replay state identity differs'):
        verify_recording(native, 'tumor', case, fabricated)


def test_recorded_branch_keeps_source_and_ledger_through_seek_rejection_and_save(prepared, tmp_path):
    case, engine, record, native = prepared[0]
    archive = save_case(case, tmp_path/'recorded.ressectionlab', artifacts={'surgical_toolbox': record})
    session = BridgeSession(tmp_path/'transfers')
    def call(operation, args):
        return session.execute(operation, args, _Request(operation), lambda *_: None)
    def command(kind, **kwargs):
        return call('surgicalRehearsal', {'caseHash': case.semantic_hash, 'command': kind, **kwargs})
    try:
        opened = call('anatomyOpen', {'path': str(native.root/'patient-anatomy.json')})
        source = {'collectionId': opened['collectionId'], 'snapshotId': native.identity(),
                  'targetLayerId': 'tumor', 'path': str(archive)}
        with pytest.raises(ValueError, match='Native anatomy changed'):
            call('openBoundSurgicalReplay', {**source, 'snapshotId': 'stale'})
        installed = call('openBoundSurgicalReplay', source)
        receipt = installed['verifiedStudyReplay']
        assert receipt['endpointStateId'] == engine.state_id and receipt['actions'] == len(engine.actions)
        assert receipt['sourceSnapshotId'] == native.identity()
        sampling = installed['verifiedTargetSampling']
        assert sampling['engineAnnotationMm3'] == sampling['nativeAnnotationMm3'] == case.compartments['tumor'].sum()
        saved = tmp_path/'saved.ressectionlab'
        endpoint = command('inspect', includeSequenceOutcome=True)
        assert endpoint['view']['originalTargetResidualMm3'] == endpoint['sequenceOutcome']['originalTarget']['residualMm3']
        with pytest.raises(ValueError, match='STUDY_REPLAY_ONLY'):
            command('step', expectedStateId=engine.state_id, action={'slot': 'right', 'operation': 'activate'})
        with pytest.raises(ValueError, match='STUDY_REPLAY_ONLY'):
            command('inspect', branchId='main')
        from resectionlab.desktop_bridge import BridgeError
        with pytest.raises(BridgeError, match='recorded engine branch'):
            call('createRehearsal', {'caseHash': case.semantic_hash})
        with pytest.raises(BridgeError, match='cannot start a frozen policy'):
            call('saveCase', {'caseHash': case.semantic_hash, 'path': str(saved), 'forFrozenExecution': True})
        assert not saved.exists()
        initial = command('seek', position=0, expectedStateId=engine.state_id)
        assert initial['readOnly'] and not initial['view']['metrics']['nonTargetRemovedMm3']
        frame = command('seekFrame', position=0, frameIndex=0, branchId=record['branchId'],
            expectedStateId=initial['view']['stateId'])
        assert frame['view']['originalTargetResidualMm3'] == initial['view']['originalTargetResidualMm3']
        command('seek', position=len(engine.actions), expectedStateId=initial['view']['stateId'])
        assert command('inspect', includeSequenceOutcome=True)['sequenceOutcome'] == endpoint['sequenceOutcome']
        call('saveCase', {'caseHash': case.semantic_hash, 'path': str(saved), 'workspace': {}, 'overwrite': False})
        reopened = call('loadCase', {'path': str(saved)})
        assert reopened['artifacts']['nativeStudyReplay'] == receipt and 'verifiedStudyReplay' not in reopened
        replay = command('inspect', includeSequenceOutcome=True)
        assert replay['readOnly'] and replay['view']['slots'] == endpoint['view']['slots']
        assert replay['sequenceOutcome'] == endpoint['sequenceOutcome']
        from resectionlab.surgical_rl_env import SurgicalRLEnv
        with pytest.raises(ValueError, match='STUDY_REPLAY_ONLY'):
            SurgicalRLEnv.from_patient(saved)
        # The imaging hash also identifies other archives of this same case.
        # A matching patient/image alone must never authenticate a branch.
        for key in ('sourceSnapshotId', 'branchId', 'sceneId', 'targetBinding', 'snapshotHash', 'endpointStateId'):
            forged = save_case(case, tmp_path/(key+'.ressectionlab'), artifacts={
                'surgical_toolbox': record, 'nativeStudyReplay': {**receipt, key: 'stale'}})
            with pytest.raises(ValueError, match='STUDY_REPLAY_BINDING_CHANGED'):
                call('loadCase', {'path': str(forged)})
        assert command('inspect')['view']['stateId'] == endpoint['view']['stateId']
    finally:
        session.close()


def test_selected_plan_export_preserves_target_and_current_revision_without_action_authority(study, tmp_path):
    from resectionlab.imaging import load_case, read_case_artifacts
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_patient import prepare_surgical_environment
    from resectionlab.surgical_planning_admission import planning_admission

    native, args = study
    approach = {key: args[key] for key in ('approachId', 'snapshotId', 'parameterHash', 'targetLayerId')}
    with pytest.raises(ValueError, match='selected approach target'):
        prepare_surgical_environment(native.root, tmp_path/'wrong-target',
            layer_ids={'brain': 'brain', 'tumor': 'other-lesion'}, approach=approach)
    with pytest.raises(ValueError, match='approach parameters changed'):
        prepare_surgical_environment(native.root, tmp_path/'stale',
            layer_ids={'brain': 'brain', 'tumor': 'lesion'}, approach={**approach, 'parameterHash': 'stale'})
    with pytest.raises(ValueError, match='route measurements changed'):
        prepare_surgical_environment(native.root, tmp_path/'stale-measurements',
            layer_ids={'brain': 'brain', 'tumor': 'lesion'}, approach={**approach, 'measurementHash': 'stale'})
    report = prepare_surgical_environment(native.root, tmp_path/'bound',
        layer_ids={'brain': 'brain', 'tumor': 'lesion'}, approach=approach)
    case = load_case(report['casePath'])
    # A freshly extracted portable native study has different absolute paths.
    # The same route, source bytes and grid must still name the same decision.
    import shutil
    restored = shutil.copytree(native.root, tmp_path/'other-extraction')
    exported_again = prepare_surgical_environment(restored, tmp_path/'bound-again',
        layer_ids={'brain': 'brain', 'tumor': 'lesion'}, approach=approach)
    assert exported_again['caseHash'] == report['caseHash']
    assert exported_again['environment']['sceneId'] == report['environment']['sceneId']
    proposed = case.metadata['surgical_adapter']['selectedApproach']
    assert proposed['targetLayerId'] == 'lesion' and proposed['sourceSnapshotId'] == native.identity()
    assert proposed['id'] == approach['approachId'] and proposed['parameterHash'] == approach['parameterHash']
    marker = case.metadata['surgical_properties']['cranialPlanningCandidateV1']
    assert marker['schema'] == 'resectionlab.planned-derivative-inspection.v1'
    assert marker['sourceSnapshotId'] == proposed['sourceSnapshotId']
    assert marker['targetBinding'] == proposed['targetBinding'] and marker['planRevision'] == proposed['planRevision']
    from resectionlab.core import thaw_json
    from resectionlab.surgical_scene import from_case
    altered = thaw_json(case.metadata)
    del altered['surgical_properties']['cranialPlanningCandidateV1']
    with pytest.raises(ValueError, match='Missing planned derivative inspection binding'):
        from_case(replace(case, metadata=altered))
    scene = load_environment(report['casePath'])
    record = read_case_artifacts(report['casePath'])['surgical_toolbox']
    engine = SurgicalEngine.restore(scene, record['branches']['main'])
    assert not planning_admission(scene)['available'] and not engine.actions
    original = engine.state_id
    outcome = engine.step({'slot': 'right', 'operation': 'inspect', 'duration': 1.})
    assert outcome['status'] == 'rejected' and engine.state_id == original

    # The intact decision survives a portable save and is measured independently
    # again on fresh-process hydration. A geometric corridor is never tissue loss.
    from resectionlab.surgical_selected_decision import selected_plan_decision
    from resectionlab.surgical_patient import surgical_environment_options
    with pytest.raises(ValueError, match='admission or source binding'):
        selected_plan_decision(engine, {'available': True}, branch_id='main')
    session = BridgeSession(tmp_path/'selected-transfers')
    def call(operation, args):
        return session.execute(operation, args, _Request(operation), lambda *_: None)
    try:
        call('loadCase', {'path': report['casePath']})
        response = call('surgicalRehearsal', {'caseHash': case.semantic_hash, 'command': 'inspect'})
        decision = response['selectedPlanDecision']
        assert response['readOnly'] and decision['stateId'] == response['view']['stateId'] == original
        assert decision['caseHash'] == case.semantic_hash and decision['sceneId'] == scene.identity
        assert decision['forecast'] == surgical_environment_options(native)['selectedRouteForecast']
        assert decision['forecast']['physicalCandidatesExamined'] == 0
        assert decision['geometry']['lengthMm'] > 0
        assert decision['measured'] == {'actions': 0, 'targetRemovedMm3': 0., 'nonTargetRemovedMm3': 0.,
            'originalTargetResidualMm3': response['view']['originalTargetResidualMm3'], 'elapsedSeconds': 0.,
            'activeTools': 0, 'insertedTools': 0}
        assert decision['measured']['originalTargetResidualMm3'] == pytest.approx(case.compartments['tumor'].sum()*scene.voxel_mm3)
        assert decision['preparation']['layers']['tumor']['sampledMaskHash'] == case.metadata['surgical_adapter']['sourceLayers']['tumor']['sampledMaskHash']
        saved = tmp_path/'selected-stop.ressectionlab'
        call('saveCase', {'caseHash': case.semantic_hash, 'path': str(saved), 'workspace': {}, 'overwrite': False})
    finally:
        session.close()
    session = BridgeSession(tmp_path/'cold-transfers')
    try:
        call('loadCase', {'path': str(saved)})
        assert call('surgicalRehearsal', {'caseHash': case.semantic_hash,
                    'command': 'inspect'})['selectedPlanDecision'] == decision
    finally:
        session.close()


def test_exact_grid_fixed_recording_rejects_substitution_and_preserves_registration_scope():
    """Optional real-data regression on the reproducible, ignored paired study."""
    from pathlib import Path
    from resectionlab.patient_anatomy import PatientImagingCollection
    from resectionlab.anatomy_approaches import read_approaches
    from resectionlab.anatomy_display import display_affine
    from resectionlab.surgical_exact_grid_replay import fixed_route_identity, verify_selected_route
    from resectionlab.surgical_study_replay import _recorded_branch
    from resectionlab.workspace_bundle import load_workspace

    root = Path(__file__).resolve().parents[1]/'testcases/outputs/engine-operative-integration/episode'
    if not (root/'replay.ressectionlab').is_file():
        pytest.skip('Build the paired IXI077 companion using scripts/prepare_ixi077_study_replay.py')
    study = PatientImagingCollection(root/'native')
    case, artifacts, _ = load_workspace(root/'replay.ressectionlab')
    record = artifacts['surgical_toolbox']
    exact = record['exactGridReplay']
    scene, snapshot, target = _recorded_branch(case, record)
    route = exact['recordedApproach']
    assert fixed_route_identity(case, scene, snapshot, exact)['recordedApproach'] == route
    verify_selected_route(study, exact, route['id'])
    for key in ('scriptSha256', 'executionReportSha256'):
        changed = {**exact, 'actionGenerator': {**exact['actionGenerator'], key: 'a'*64}}
        with pytest.raises(ValueError, match='generator differs'):
            fixed_route_identity(case, scene, snapshot, changed)
    with pytest.raises(ValueError, match='Unknown fixed CUSA execution report'):
        fixed_route_identity(case, scene, snapshot, {**exact, 'executionReport': exact['executionReport']+' '})
    with pytest.raises(ValueError, match='Recorded engine branch differs'):
        fixed_route_identity(case, scene, {**snapshot, 'actions': snapshot['actions'][1:]}, exact)
    with pytest.raises(ValueError, match='Select the recorded approach'):
        verify_selected_route(study, exact, 'different-approach')
    with pytest.raises(ValueError, match='route differs'):
        verify_selected_route(study, {**exact, 'recordedApproach': {**route, 'measurementHash': 'a'*64}}, route['id'])
    for sid in (exact['movingSeriesId'], exact['registeredSeriesId']):
        assert display_affine(study, study.series(sid), preview=True)[1]
        with pytest.raises(ValueError, match='needs review'):
            study.registration_matrix(sid, require_review=True)
    assessed = next(item for item in read_approaches(study) if item['id'] == route['id'])
    vessel = next(item for item in assessed['measurements']['structures'] if item['kind'] == 'vessels')
    assert not vessel['available'] and 'clearanceMm' not in vessel
    receipt, sampling = verify_recording(study, target, case, record, approach_id=route['id'])
    assert receipt['endpointStateId'] == snapshot['stateId'] and receipt['actions'] == 10
    assert sampling['nativeAnnotationMm3'] == sampling['engineAnnotationMm3']
