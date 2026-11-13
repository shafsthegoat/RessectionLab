"""Public native craniotomy controls; generated geometry is not patient evidence."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.geometry import GeometryScene, capsule_voxel_indices
from resectionlab.surgical_craniotomy import PROPERTY, SCHEMA, binding, contract_hash
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene


def pose(x=32., y=32., z=55.):
    return {'position': [x, y, z], 'quaternion': [0., 0., 0., 1.]}


def request(operation, *, slot='left', **kwargs):
    return {'slot': slot, 'operation': operation, 'duration': .1, **kwargs}


def accepted(engine, operation, **kwargs):
    action = request(operation, **kwargs)
    before = engine.state_id
    preview = engine.preview(action, include_view=False)
    assert engine.state_id == before
    result = engine.step(action)
    assert result['status'] == 'accepted', result
    assert result['stateId'] == preview['stateId']
    return result


def shell(*, proof_status='reviewed', obstacle=None, source_gap=False, protected=False, attachments=()):
    """Freeze the source-cell contour before executing any native command."""
    labels = np.zeros((64, 64, 96), np.uint16)
    labels[20:45, 20:45, 60] = 1
    labels[20:45, 20:45, 75] = 1
    labels[20, 20:45, 60:76] = labels[44, 20:45, 60:76] = 1
    labels[20:45, 20, 60:76] = labels[20:45, 44, 60:76] = 1
    labels[31:34, 31:34, 67:70] = 2
    if source_gap:
        labels[38:43, 30:35, 60] = 0
    if obstacle is not None:
        labels[obstacle] = 3
    regions = [{'id': 1, 'name': 'Idealized shell', 'material': 'bone'},
               {'id': 2, 'name': 'Generated target', 'material': 'tumor', 'target': True}]
    if obstacle is not None:
        regions.append({'id': 3, 'name': 'Transport obstacle', 'material': 'soft_tissue', 'protected': protected})
    source = {'kind': 'generated closed-shell craniotomy control; not a patient'}
    anchor = int(np.ravel_multi_index((20, 20, 75), labels.shape))
    properties = {'attachments': copy.deepcopy(list(attachments)), 'boneAnchors': [anchor]}
    base = SurgicalScene(labels, np.eye(4), regions, source, properties)
    contour = [pose(x, y, 58.3) for x, y in ((26, 26), (38, 26), (38, 38), (26, 38), (26, 26))]
    grid = GeometryScene(np.zeros(labels.shape, bool), np.eye(4))
    kerf = set()
    for start, end in zip(contour, contour[1:]):
        indices = capsule_voxel_indices(grid, start['position'], end['position'], 2.)
        ids = np.ravel_multi_index(indices.T, labels.shape)
        kerf.update(map(int, ids[labels.ravel()[ids] == 1]))
    flap = set(map(int, np.flatnonzero(labels.ravel()))) & set(
        int(np.ravel_multi_index((x, y, 60), labels.shape)) for x in range(29, 36) for y in range(29, 36))
    bone = set(map(int, np.flatnonzero(labels.ravel() == 1)))
    proof = {'status': proof_status, 'provenance': 'Explicit idealized generated shell, no patient qualification',
             'evidenceSha256': '1'*64 if proof_status == 'reviewed' else None}
    contract = {'schema': SCHEMA, **binding(base), 'windowCells': sorted(flap | kerf),
        'flapCells': sorted(flap), 'kerfCells': sorted(kerf),
        'retainedSkullCells': sorted(bone-flap-kerf), 'boneAnchorCells': [anchor],
        'entry': {'centerRasMm': [32., 32., 59.5], 'normalRasUnit': [0., 0., 1.],
                  'uRasUnit': [1., 0., 0.], 'halfExtentsMm': [10., 10.],
                  'maximumInwardMm': 30., 'maximumAngleDegrees': 30.},
        'exit': {'axis': 2, 'side': -1, 'planeVoxelBoundary': float(min(np.nonzero(labels)[2])-1),
                 'minimumVoxelBoundary': [-.5, -.5, -.5],
                 'maximumVoxelBoundary': [63.5, 63.5, 95.5]},
        'evidence': {'scope': 'idealized-engineering', **{name: copy.deepcopy(proof)
                      for name in ('window', 'support', 'attachments', 'exit')}}}
    contract['contentHash'] = contract_hash(contract)
    properties[PROPERTY] = contract
    scene = SurgicalScene(labels, np.eye(4), regions, source, properties)
    if protected:
        from resectionlab.surgical_rl_protection import GuardConfig, ProtectionSet
        # Isolate actual whole-flap contact with zero additional clearance,
        # declared before any source action; no patient settings are changed.
        engine = SurgicalEngine(scene, protection=ProtectionSet(scene, clearance_mm=0., required=()),
                                guard_config=GuardConfig(require_critical_coverage=False))
    else:
        engine = SurgicalEngine(scene)
    return engine, contour, flap, kerf


def cut_shell(engine, contour, hand='left'):
    accepted(engine, 'configure', mechanicsRevision=2, slot=hand)
    if engine.state['slots'][hand]['instrument'] != 'cranial_drill':
        accepted(engine, 'exchange', instrument='cranial_drill', slot=hand)
    accepted(engine, 'place', pose=pose(26, 26, 55), slot=hand)
    accepted(engine, 'apply', pose=contour[0], slot=hand)
    for point in contour[1:]:
        accepted(engine, 'apply', pose=point, slot=hand)
    outside_z = min(55., engine._cranial_access.exit.plane_voxel_boundary-2.5)
    accepted(engine, 'withdraw', pose=pose(26, 26, outside_z), slot=hand)


def grasp_shell(engine, hand='left'):
    accepted(engine, 'exchange', instrument='bone_flap_holder', slot=hand)
    accepted(engine, 'place', pose=pose(), parameters={'jawOpeningMm': 0.}, slot=hand)
    accepted(engine, 'grasp', pose=pose(z=59.5), slot=hand)


@pytest.mark.parametrize('hand', ['left', 'right'])
def test_full_native_cut_grasp_lift_exterior_evacuation_withdrawal_and_replay(hand):
    engine, contour, flap, kerf = shell()
    original = engine.scene.labels.copy()
    cut_shell(engine, contour, hand)
    assert set(engine.state['removed']) == kerf
    assert engine.observe()['craniotomy']['openingComplete'] is False
    grasp_shell(engine, hand)
    assert set(engine.state['holds'][hand]['cells']) == flap
    assert set(engine.state['removed']) == kerf
    before = engine.state_id
    result = engine.step(request('extract', slot=hand))
    assert result['status'] == 'rejected' and 'MOVE_THROUGH_EXIT' in result['reason']
    assert engine.state_id == before
    accepted(engine, 'move', pose=pose(z=54.5), slot=hand)
    assert not set(engine.state['removed']) & flap
    lifted = engine.snapshot()
    assert SurgicalEngine.restore(engine.scene, lifted).state_id == engine.state_id
    accepted(engine, 'extract', slot=hand)
    assert set(engine.state['removed']) == flap | kerf
    assert engine.metrics()['targetRemovedMm3'] == 0.
    assert engine.metrics()['nonTargetRemovedMm3'] == len(flap | kerf)
    assert engine.observe()['craniotomy']['apertureEvacuated'] is True
    assert engine.observe()['craniotomy']['openingComplete'] is False
    accepted(engine, 'move', pose=pose(z=64.), slot=hand)
    before = engine.snapshot()
    result = engine.step(request('withdraw', pose=pose(z=64.), slot=hand))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'CRANIAL_ACCESS_WITHDRAWAL_NOT_AT_EXTERIOR_EXIT'
    assert engine.snapshot() == before and not engine.observe()['craniotomy']['openingComplete']
    accepted(engine, 'withdraw', pose=pose(z=54.5), slot=hand)
    assert engine.observe()['craniotomy']['openingComplete'] is True
    before = engine.state_id
    assert engine.step(request('extract', slot=hand))['status'] == 'rejected'
    assert engine.state_id == before
    assert engine.replay(engine.actions).state_id == engine.state_id
    assert SurgicalEngine.restore(engine.scene, engine.snapshot()).state_id == engine.state_id
    np.testing.assert_array_equal(engine.scene.labels, original)


@pytest.mark.parametrize('point', [pose(50, 32, 62), pose(32, 32, 80)])
def test_wrong_or_inferior_entry_rejected_atomically_before_removal(point):
    engine, _, _, _ = shell()
    accepted(engine, 'configure', mechanicsRevision=2)
    accepted(engine, 'place', pose=pose(50, 32, 55))
    before = engine.snapshot()
    result = engine.step(request('apply', pose=point))
    assert result['status'] == 'rejected' and result['reasonCode'].startswith('CRANIAL_ACCESS_')
    assert engine.snapshot() == before


def test_held_flap_requires_closed_jaws_and_rotation_refuses_whole_multi_leg_action():
    from scipy.spatial.transform import Rotation
    engine, contour, _, _ = shell()
    cut_shell(engine, contour)
    grasp_shell(engine)
    before = engine.snapshot()
    result = engine.step(request('grasp', parameters={'jawOpeningMm': 2.}))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'BONE_FLAP_RELEASE_BEFORE_OPENING_JAWS'
    assert engine.snapshot() == before
    turn = pose(z=58.5)
    turn['quaternion'] = Rotation.from_euler('z', 90, degrees=True).as_quat().tolist()
    result = engine.step(request('move', path=[pose(z=58.5), turn]))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'BONE_FLAP_ROTATION_UNSUPPORTED'
    assert engine.snapshot() == before


def test_holder_shaft_has_no_whole_flap_exemption():
    from resectionlab.surgical_bone_transport import check_sweep
    engine, contour, _, _ = shell()
    cut_shell(engine, contour)
    grasp_shell(engine)
    before = engine.snapshot()
    with pytest.raises(ValueError, match='BONE_FLAP_SHAFT_COLLISION'):
        check_sweep(engine, 'left', [0, 0, 0], pose(z=62.5))
    assert engine.snapshot() == before


@pytest.mark.parametrize('operation', ['move', 'apply', 'grasp'])
def test_every_held_pose_operation_transports_flap_and_refuses_rotation(operation):
    from scipy.spatial.transform import Rotation
    engine, contour, _, _ = shell()
    cut_shell(engine, contour)
    grasp_shell(engine)
    origin = engine.state['holds']['left']['origin'][:]
    accepted(engine, operation, pose=pose(z=57.5))
    hold = engine.state['holds']['left']
    assert hold['offset'] == [0, 0, -2] and hold['origin'] == origin
    before = engine.snapshot()
    turn = pose(z=56.5)
    turn['quaternion'] = Rotation.from_euler('z', 90, degrees=True).as_quat().tolist()
    result = engine.step(request(operation, path=[pose(z=56.5), turn]))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'BONE_FLAP_ROTATION_UNSUPPORTED'
    assert engine.snapshot() == before


@pytest.mark.parametrize('protected', [False, True])
def test_entire_flap_checks_obstacle_outside_holder_trajectory(protected):
    engine, contour, _, _ = shell(obstacle=(35, 35, 54), protected=protected)
    cut_shell(engine, contour)
    grasp_shell(engine)
    before = engine.snapshot()
    result = engine.step(request('move', pose=pose(z=54.5)))
    assert result['status'] == 'rejected'
    assert result['reasonCode'] == ('PROTECTED_STRUCTURE' if protected else 'BONE_FLAP_SWEPT_ANATOMY_COLLISION')
    assert engine.snapshot() == before


def test_flap_lift_checks_parked_other_hand_outside_holder_path():
    engine, contour, _, _ = shell()
    cut_shell(engine, contour)
    grasp_shell(engine)
    accepted(engine, 'place', slot='right', pose=pose(35, 32, 57))
    before = engine.snapshot()
    result = engine.step(request('move', pose=pose(z=54.5)))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'BONE_FLAP_OTHER_TOOL_COLLISION'
    assert engine.snapshot() == before


def test_moving_other_hand_checks_parked_full_flap_not_only_holder():
    engine, contour, _, _ = shell()
    cut_shell(engine, contour)
    grasp_shell(engine)
    accepted(engine, 'move', pose=pose(z=57.5))
    accepted(engine, 'place', slot='right', pose=pose(25, 35, 56.5))
    before = engine.snapshot()
    result = engine.step(request('move', slot='right', pose=pose(40, 35, 56.5)))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'BONE_FLAP_MOVING_TOOL_COLLISION'
    assert engine.snapshot() == before


def test_release_keeps_flap_in_field_and_regrasp_cannot_refund_or_drop_it():
    engine, contour, flap, kerf = shell()
    cut_shell(engine, contour)
    grasp_shell(engine)
    accepted(engine, 'move', pose=pose(z=57.5))
    accepted(engine, 'release')
    assert not engine.state['holds'] and len(engine.state['detachedFragments']) == 1
    assert set(engine.state['removed']) == kerf
    accepted(engine, 'grasp')
    assert not engine.state['detachedFragments']
    assert engine.state['holds']['left']['offset'] == [0, 0, -2]
    assert set(engine.state['holds']['left']['cells']) == flap
    accepted(engine, 'move', pose=pose(z=54.5))
    accepted(engine, 'extract')
    accepted(engine, 'withdraw', pose=pose(z=54.5))
    assert set(engine.state['removed']) == flap | kerf
    assert engine.replay(engine.actions).state_id == engine.state_id


def test_full_tumor_entry_cannot_use_cropped_or_existing_void_before_flap_evacuation():
    engine, _, _, _ = shell(source_gap=True)
    accepted(engine, 'configure', mechanicsRevision=2)
    accepted(engine, 'exchange', instrument='cusa')
    before = engine.snapshot()
    result = engine.step(request('place', pose=pose(40, 32, 64)))
    assert result['status'] == 'rejected'
    assert result['reasonCode'] == 'CRANIAL_ACCESS_REQUIRES_EVACUATED_FLAP'
    assert engine.snapshot() == before


def test_unknown_attachment_review_stays_inspectable_and_blocks_bone_work():
    engine, contour, _, _ = shell(proof_status='unknown')
    accepted(engine, 'configure', mechanicsRevision=2)
    accepted(engine, 'place', pose=pose(26, 26, 55))
    before = engine.snapshot()
    result = engine.step(request('apply', pose=contour[0]))
    assert result['status'] == 'rejected' and 'EVIDENCE_REQUIRED' in result['reason']
    assert engine.snapshot() == before


def test_native_case_contract_and_full_flap_history_survive_save_reopen(tmp_path):
    from resectionlab.core import CaseData, SourceRef, thaw_json
    from resectionlab.imaging import load_case, read_case_artifacts, save_case
    from resectionlab.surgical_scene import from_case
    generated, contour, flap, kerf = shell()
    props = copy.deepcopy(generated.scene.properties)
    raw = props.pop(PROPERTY)
    regions = generated.scene.regions
    metadata = {'surgical_properties': props,
                'anatomy_layers': {row['name']: {'kind': 'skull' if row['material'] == 'bone' else 'tumor'}
                                   for row in regions}}
    base = CaseData(case_id='generated-complete-craniotomy-control',
                    mri=np.zeros(generated.scene.shape, np.float32), affine=np.eye(4),
                    compartments={row['name']: generated.scene.labels == row['id'] for row in regions},
                    metadata=metadata, source_refs=(SourceRef('generated', 'generated:closed-shell', provenance='simulated'),))
    raw.update(binding(from_case(base, regions)))
    raw['contentHash'] = contract_hash(raw)
    metadata = thaw_json(base.metadata)
    metadata['surgical_properties'][PROPERTY] = raw
    case = replace(base, metadata=metadata)
    engine = SurgicalEngine(from_case(case, regions))
    assert engine.scene.source['cranialAccessBaseCaseHash'] == base.semantic_hash
    cut_shell(engine, contour)
    grasp_shell(engine)
    accepted(engine, 'move', pose=pose(z=54.5))
    # Exercise both a retained transported flap and final evacuation archives.
    for name in ('lifted', 'complete'):
        if name == 'complete':
            accepted(engine, 'extract')
            accepted(engine, 'withdraw', pose=pose(z=54.5))
        snapshot = engine.snapshot()
        path = save_case(case, tmp_path/(name+'.ressectionlab'), artifacts={'craniotomySnapshot': snapshot})
        reopened = load_case(path)
        assert reopened.semantic_hash == case.semantic_hash
        restored = SurgicalEngine.restore(from_case(reopened, regions), read_case_artifacts(path)['craniotomySnapshot'])
        assert restored.state_id == engine.state_id
        assert restored.observe()['craniotomy'] == engine.observe()['craniotomy']
    assert set(engine.state['removed']) == flap | kerf
    for changed in (replace(case, mri=np.ones(case.mri.shape, np.float32)),
                    replace(case, affine=np.diag([1.1, 1., 1., 1.])),
                    replace(case, metadata={**metadata, 'unrelated_constraint': 'changed'})):
        with pytest.raises(ValueError, match='CRANIOTOMY_CONTRACT_INVALID'):
            from_case(changed, regions)
