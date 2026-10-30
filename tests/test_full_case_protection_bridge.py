"""Generated numerical controls for case-bound desktop protection transactions.

These small source-grid fixtures verify software identity and persistence only.
They provide no anatomical or physiological validation.
"""
import copy

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, freeze_json, thaw_json
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import (
    GuardConfig, ProtectionAnnotation, ProtectionSet, save_protection,
)
from resectionlab.surgical_scene import SurgicalScene


def call(session, operation, **args):
    return session.execute(operation, args, _Request(operation), lambda *_: None)


@pytest.fixture
def opened(tmp_path):
    mask = np.zeros((16, 16, 16), bool)
    mask[5:10, 5:10, 7:10] = True
    case = CaseData(case_id='GENERATED-case-binding-control',
        mri=np.zeros(mask.shape, np.float32), compartments={'target': mask}, affine=np.eye(4),
        source_refs=(SourceRef('generated-binding-control', 'synthetic://case-binding-control',
                               native_frame='RAS+', provenance='simulated'),),
        metadata={'is_synthetic': True, 'anatomy_layers': {'target': {'kind': 'tumor'}}})
    session = BridgeSession(tmp_path / 'transfers')
    payload = session._install_case(case, {}, _Request('install'))
    key = payload['caseHash']

    def command(name, previous=None, **extra):
        return call(session, 'surgicalRehearsal', caseHash=key, command=name,
            **({'expectedStateId': previous['view']['stateId']} if previous else {}), **extra)

    initial = command('create')
    scene = session.surgical_sessions[key][1].scene
    protection = ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem',
        np.indices(scene.shape).sum(axis=0) == 0,
        provenance='GENERATED numerical identity control; not acquired anatomy')], strict=True)
    manifest = tmp_path / 'explicit-protection.json'
    save_protection(protection, manifest)
    yield session, key, command, initial, manifest, scene
    session.close()


def assert_unchanged(session, key, before, state_id):
    assert thaw_json(session.cases[key].artifacts) == before
    assert session.surgical_sessions[key][1].state_id == state_id


def test_missing_protection_blocks_execution_but_preserves_inspection(opened):
    session, key, command, initial, _, _ = opened
    readiness = initial['fullCaseReadiness']
    assert readiness['status'] == 'blocked'
    assert {'inspect', 'replay', 'attachProtection'} <= set(readiness['availableActions'])
    assert 'primary' not in readiness['availableActions']
    assert readiness['caseHash'] == key and readiness['stateId'] == initial['view']['stateId']
    before = thaw_json(session.cases[key].artifacts)
    with pytest.raises(ValueError, match='protection|Protection|evidence|annotations'):
        command('fullCase', initial, scenario='primary', maxActions=1)
    assert_unchanged(session, key, before, initial['view']['stateId'])
    assert command('inspect')['view']['stateId'] == initial['view']['stateId']


@pytest.mark.parametrize('change', ['source', 'labels', 'affine'])
def test_matching_filename_cannot_attach_other_source_or_grid(opened, tmp_path, change):
    session, key, command, initial, _, scene = opened
    labels, affine, source = scene.labels.copy(), scene.affine.copy(), copy.deepcopy(scene.source)
    if change == 'source':
        source['otherSource'] = 'same patient name does not establish identity'
    elif change == 'labels':
        labels[0, 0, 1] = next(r['id'] for r in scene.regions)
    else:
        affine[0, 3] += .01
    other = SurgicalScene(labels, affine, scene.regions, source, scene.properties)
    protection = ProtectionSet(other, [ProtectionAnnotation.from_mask(other, 'brainstem',
        np.indices(other.shape).sum(axis=0) == 0, provenance='GENERATED adversarial control')])
    path = tmp_path / change / 'explicit-protection.json'
    save_protection(protection, path)
    before = thaw_json(session.cases[key].artifacts)
    with pytest.raises(ValueError, match='source|grid|match'):
        command('attachProtection', initial, protectionPath=str(path))
    assert_unchanged(session, key, before, initial['view']['stateId'])


def test_attachment_save_cold_reopen_and_existing_action_keep_exact_contract(opened, tmp_path):
    session, key, command, initial, manifest, scene = opened
    original = thaw_json(session.cases[key].artifacts)['surgical_toolbox']['branches']['main']
    attached = command('attachProtection', initial, protectionPath=str(manifest))
    assert attached['fullCaseReadiness']['status'] == 'ready'
    assert {'primary', 'vascular'} <= set(attached['fullCaseReadiness']['availableActions'])
    assert attached['branchId'] != 'main'
    record = thaw_json(session.cases[key].artifacts)['surgical_toolbox']
    assert record['branches']['main'] == original
    contract = record['branches'][attached['branchId']]['protectionContract']
    assert contract['sourceBinding'] and contract['sceneId'] == scene.identity
    assert attached['fullCaseReadiness']['protectionContractId'] == contract['contentHash']
    action = command('step', attached, action={'slot': 'right', 'operation': 'inspect', 'duration': .1})
    assert action['position'] == 1
    assert action['fullCaseReadiness']['status'] == 'ready'
    before = thaw_json(session.cases[key].artifacts)
    with pytest.raises(ValueError, match='initial|intact'):
        command('attachProtection', action, protectionPath=str(manifest))
    assert_unchanged(session, key, before, action['view']['stateId'])
    destination = tmp_path / 'GENERATED-bound-checkpoint.ressectionlab'
    call(session, 'saveCase', caseHash=key, path=str(destination), workspace={}, overwrite=False)
    # External manifest files are not needed to reopen or resume the save.
    manifest.unlink()
    manifest.with_suffix('.npz').unlink()
    reopened = BridgeSession(tmp_path / 'cold-transfers')
    try:
        payload = call(reopened, 'loadCase', path=str(destination))
        inspected = call(reopened, 'surgicalRehearsal', caseHash=payload['caseHash'], command='inspect')
        assert inspected['view']['stateId'] == action['view']['stateId']
        assert inspected['fullCaseReadiness']['status'] == 'ready'
        assert inspected['fullCaseReadiness']['protectionContractId'] == contract['contentHash']
        current = thaw_json(reopened.cases[payload['caseHash']].artifacts)['surgical_toolbox']
        assert current['branches'][current['branchId']]['protectionContract'] == contract
        progressed = call(reopened, 'surgicalRehearsal', caseHash=payload['caseHash'], command='fullCase',
            expectedStateId=inspected['view']['stateId'], scenario='primary', maxActions=1)
        assert progressed['fullCase']['startPosition'] == 1
        assert progressed['fullCaseReadiness']['protectionContractId'] == contract['contentHash']
    finally:
        reopened.close()


def test_relaxed_checkpoint_cannot_bypass_binding_via_saved_director(opened):
    session, key, command, initial, _, scene = opened
    engine = SurgicalEngine(scene, guard_config=GuardConfig(require_critical_coverage=False))
    assert engine.step({'slot': 'right', 'operation': 'inspect', 'duration': .1})['status'] == 'accepted'
    artifacts = thaw_json(session.cases[key].artifacts)
    record = artifacts['surgical_toolbox']
    record['branches']['main'] = engine.snapshot()
    record['position'] = 1
    record['fullCase'] = {'main': {'schema': 'resectionlab.full-case-rehearsal.v1',
        'scenario': 'primary', 'status': 'running', 'lastStateId': engine.state_id}}
    session.cases[key].artifacts = freeze_json(artifacts)
    inspected = command('inspect')
    assert inspected['fullCaseReadiness']['status'] == 'blocked'
    assert inspected['fullCaseReadiness']['recovery']['action'] == 'seekInitial'
    before = thaw_json(session.cases[key].artifacts)
    with pytest.raises(ValueError, match='protection|Protection|evidence|annotations|initial'):
        command('fullCase', inspected, scenario='primary', maxActions=1)
    assert_unchanged(session, key, before, inspected['view']['stateId'])
    intact = command('seek', inspected, position=0)
    assert intact['fullCaseReadiness']['recovery']['action'] == 'attachProtection'


def test_recovery_attachment_preserves_original_action_branch(opened):
    session, key, command, initial, manifest, _ = opened
    action = command('step', initial, action={'slot': 'right', 'operation': 'inspect', 'duration': .1})
    before = thaw_json(session.cases[key].artifacts)['surgical_toolbox']['branches']['main']
    intact = command('seek', action, position=0)
    attached = command('attachProtection', intact, protectionPath=str(manifest))
    assert attached['position'] == 0 and attached['fullCaseReadiness']['status'] == 'ready'
    record = thaw_json(session.cases[key].artifacts)['surgical_toolbox']
    assert record['branches']['main'] == before
    assert record['parents'][attached['branchId']] == {'branchId': 'main', 'position': 0}
