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
    report = prepare_surgical_environment(native.root, tmp_path/'bound',
        layer_ids={'brain': 'brain', 'tumor': 'lesion'}, approach=approach)
    case = load_case(report['casePath'])
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
