"""Generated declaration/topology controls, not native flap execution evidence."""
import copy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from resectionlab.core import array_digest, immutable_array, semantic_digest
from resectionlab.surgical_craniotomy import (
    PROPERTY, SCHEMA, binding, contract_hash, validate_contract,
    require_evidence, eligible_bone_component, validate_bone_scope)
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene, _bind_original_target_memberships


def shell_fixture(*, width=3, thickness=1, bridge=None, attachments=True,
                  proof_status='reviewed', affine=None):
    """Source shell + predeclared window; removed state is a topology input only."""
    low, edge, top = 2, width+5, thickness+3
    shape = (edge+4, edge+4, top+thickness+10)
    labels = np.zeros(shape, np.uint16)
    labels[low:edge+1, low:edge+1, top:top+thickness+5] = 1
    labels[low+1:edge, low+1:edge, top+thickness:top+thickness+4] = 0
    labels[5, 5, top+thickness+1] = 2
    flap_mask = np.zeros(shape, bool); flap_mask[4:4+width, 4:4+width, top:top+thickness] = True
    window_mask = np.zeros(shape, bool); window_mask[3:5+width, 3:5+width, top:top+thickness] = True
    if bridge == 'face':
        labels[3:5, 5, top+thickness] = 1  # under the planned kerf, attached by faces
    if bridge == 'diagonal':
        labels[3, 3, top+thickness] = 1  # touches F only at an edge/corner
    regions = [{'id': 1, 'name': 'closed shell', 'material': 'bone'},
               {'id': 2, 'name': 'declared target', 'material': 'tumor', 'target': True}]
    props = {'attachments': []} if attachments else {}
    a = np.eye(4) if affine is None else np.asarray(affine)
    source = {'kind': 'generated', 'description': 'Idealized closed-shell topology control; no patient evidence'}
    initial = SurgicalScene(labels, a, regions, source, props)
    flap = np.flatnonzero(flap_mask).tolist(); window = np.flatnonzero(window_mask).tolist()
    kerf = sorted(set(window)-set(flap))
    retained = sorted(set(np.flatnonzero(labels == 1).tolist())-set(window))
    center_index = np.array([(width+7)/2, (width+7)/2, top-.5])
    normal = a[:3, 2]/np.linalg.norm(a[:3, 2]); u = a[:3, 0]/np.linalg.norm(a[:3, 0])
    assert abs(normal @ u) < 1e-6  # Fixture affine uses orthogonal physical axes.
    raw = {'schema': SCHEMA, **binding(initial), 'windowCells': window, 'flapCells': flap,
        'kerfCells': kerf, 'retainedSkullCells': retained, 'boneAnchorCells': [retained[0]],
        'entry': {'centerRasMm': (a[:3, :3] @ center_index+a[:3, 3]).tolist(),
            'normalRasUnit': normal.tolist(), 'uRasUnit': u.tolist(),
            'halfExtentsMm': [float((width+2)/2*np.linalg.norm(a[:3, 0])), float((width+2)/2*np.linalg.norm(a[:3, 1]))],
            'maximumInwardMm': float((thickness+5)*np.linalg.norm(a[:3, 2])), 'maximumAngleDegrees': 30},
        'exit': {'axis': 2, 'side': -1, 'planeVoxelBoundary': float(top-1),
            'minimumVoxelBoundary': [-.5]*3, 'maximumVoxelBoundary': (np.asarray(shape)-.5).tolist()},
        'evidence': {'scope': 'idealized-engineering', **{name: {'status': proof_status,
            'provenance': 'Idealized generated complete coverage; not clinical or patient evidence',
            'evidenceSha256': '1'*64 if proof_status == 'reviewed' else None}
            for name in ('window', 'support', 'attachments', 'exit')}}}
    raw['contentHash'] = contract_hash(raw)
    scene = SurgicalScene(labels, a, regions, source, {**props, PROPERTY: raw})
    engine = SurgicalEngine(scene)
    engine.state['removed'] = kerf[:]  # No cutting/execution claim; test pure topology query.
    return engine, raw


def rebuild(engine, raw, *, properties=None, labels=None, affine=None, source=None, regions=None):
    old = engine.scene
    props = {k: copy.deepcopy(v) for k, v in old.properties.items() if k != PROPERTY}
    props.update(properties or {})
    scene = SurgicalScene(old.labels if labels is None else labels,
        old.affine if affine is None else affine, old.regions if regions is None else regions,
        old.source if source is None else source, props)
    raw = copy.deepcopy(raw); raw.update(binding(scene)); raw['contentHash'] = contract_hash(raw)
    scene = SurgicalScene(scene.labels, scene.affine, scene.regions, scene.source, {**props, PROPERTY: raw})
    result = SurgicalEngine(scene); result.state['removed'] = list(engine.state['removed'])
    return result, raw


def test_exact_complete_shell_flap_is_immutable_and_read_only():
    engine, raw = shell_fixture(); before = engine.snapshot()
    contract = validate_contract(engine.scene)
    assert binding(engine.scene) == {k: raw[k] for k in binding(engine.scene)}
    assert validate_contract(engine.scene, contract) == contract
    assert contract.as_dict() == raw
    component = eligible_bone_component(engine)
    assert component.cells == tuple(raw['flapCells'])
    assert component.volume_mm3 == len(component.cells)*engine.scene.voxel_mm3
    assert component.contract_hash == raw['contentHash']
    assert engine.snapshot() == before
    with pytest.raises(FrozenInstanceError): component.volume_mm3 = 0.
    with pytest.raises(TypeError): contract.entry.center_ras_mm[0] = 0.
    detached = contract.as_dict(); detached['flapCells'].clear()
    assert contract.flap_cells


@pytest.mark.parametrize('bridge', ['face', 'diagonal'])
def test_planned_kerf_cannot_hide_uncut_face_or_diagonal_bridge(bridge):
    engine, _ = shell_fixture(bridge=bridge)
    before = engine.snapshot()
    with pytest.raises(ValueError, match='BONE_BRIDGE'):
        eligible_bone_component(engine)
    assert engine.snapshot() == before


def test_incomplete_kerf_is_not_separated_flap():
    engine, _ = shell_fixture(); engine.state['removed'].pop()
    with pytest.raises(ValueError, match='KERF_INCOMPLETE'): eligible_bone_component(engine)


@pytest.mark.parametrize('name', ['window', 'support', 'attachments', 'exit'])
@pytest.mark.parametrize('status', ['unknown', 'unreviewed'])
def test_pending_proposals_remain_inspectable_but_every_evidence_gate_blocks(name, status):
    engine, raw = shell_fixture(); raw['evidence'][name]['status'] = status
    raw['evidence'][name]['evidenceSha256'] = None
    engine, _ = rebuild(engine, raw)
    pending = validate_contract(engine.scene)
    assert pending.proofs
    with pytest.raises(ValueError, match=name.upper()+'_EVIDENCE_REQUIRED'):
        eligible_bone_component(engine)


def test_missing_contract_attachments_and_empty_anchors_do_not_mean_free():
    scene = SurgicalScene(np.ones((3, 3, 3), np.uint16), np.eye(4),
        [{'id': 1, 'name': 'bone', 'material': 'bone'}], {}, {})
    assert validate_contract(scene) is None
    with pytest.raises(ValueError, match='CONTRACT_REQUIRED'):
        eligible_bone_component(SurgicalEngine(scene))
    engine, _ = shell_fixture(attachments=False)
    with pytest.raises(ValueError, match='ATTACHMENT_COVERAGE_REQUIRED'): eligible_bone_component(engine)
    engine, raw = shell_fixture(); raw['boneAnchorCells'] = []
    engine, _ = rebuild(engine, raw)
    with pytest.raises(ValueError, match='SUPPORT_ANCHORS_REQUIRED'): eligible_bone_component(engine)


def test_intact_attachments_block_even_when_breakable_until_explicitly_separated():
    engine, raw = shell_fixture()
    attachment = {'id': 'periosteal', 'kind': 'periosteal', 'cells': [raw['flapCells'][0]], 'breakable': True}
    engine, _ = rebuild(engine, raw, properties={'attachments': [attachment]})
    with pytest.raises(ValueError, match='SEPARATE_ATTACHMENTS_BEFORE_GRASP'): eligible_bone_component(engine)
    engine.state['separatedAttachments'] = ['periosteal']
    assert eligible_bone_component(engine).cells == tuple(raw['flapCells'])


@pytest.mark.parametrize('change', ['source', 'affine', 'target', 'physics', 'attachment'])
def test_predeclared_source_frame_target_and_other_constraints_are_exactly_bound(change):
    engine, raw = shell_fixture(); scene = engine.scene
    kwargs = {'labels': scene.labels, 'affine': scene.affine, 'regions': scene.regions,
              'source': scene.source, 'properties': scene.properties}
    if change == 'source': kwargs['source'] = {**scene.source, 'newVersion': 1}
    if change == 'affine': kwargs['affine'] = scene.affine.copy(); kwargs['affine'][0, 3] += .01
    if change == 'target': kwargs['regions'] = copy.deepcopy(scene.regions); kwargs['regions'][1]['target'] = False
    if change == 'physics': kwargs['properties'] = {**scene.properties, 'extraConstraint': {'enabled': True}}
    if change == 'attachment': kwargs['properties'] = {**scene.properties, 'attachments': [{'id': 'new', 'kind': 'soft', 'cells': [raw['flapCells'][0]]}]}
    with pytest.raises(ValueError, match='binding differs'):
        candidate = SurgicalScene(**kwargs)
        validate_contract(candidate)


@pytest.mark.parametrize('bad', [True, -1, 1.5, 10**9])
def test_voxel_identity_types_and_bounds_are_strict(bad):
    engine, raw = shell_fixture(); raw['flapCells'][0] = bad; raw['contentHash'] = contract_hash(raw)
    with pytest.raises(ValueError, match='source integer cells'): validate_contract(engine.scene, raw)


def test_partition_and_anchor_tampering_cannot_omit_other_source_skull():
    engine, raw = shell_fixture(); raw['retainedSkullCells'].pop(); raw['contentHash'] = contract_hash(raw)
    with pytest.raises(ValueError, match='partition'): validate_contract(engine.scene, raw)
    engine, raw = shell_fixture(); raw['boneAnchorCells'] = [raw['flapCells'][0]]; raw['contentHash'] = contract_hash(raw)
    with pytest.raises(ValueError, match='anchors'): validate_contract(engine.scene, raw)


def test_protected_and_original_target_overlay_remain_exclusions():
    engine, raw = shell_fixture(); engine._protected_ids = frozenset([raw['flapCells'][0]])
    with pytest.raises(ValueError, match='PROTECTED_STRUCTURE'): eligible_bone_component(engine)
    engine, raw = shell_fixture(); mask = engine.scene.labels == 2; mask.ravel()[raw['flapCells'][0]] = True
    engine, raw = rebuild(engine, raw, source={**engine.scene.source, 'annotationHashes': {'declared target': array_digest(mask)}})
    with pytest.raises(ValueError, match='ORIGINAL_TARGET_EVIDENCE_REQUIRED'): eligible_bone_component(engine)
    before = binding(engine.scene)
    _bind_original_target_memberships(engine.scene, {'declared target': immutable_array(np.flatnonzero(mask), np.int64)})
    assert binding(engine.scene) == before
    with pytest.raises(ValueError, match='TARGET_OVERLAP'): eligible_bone_component(engine)


def test_exact_large_component_does_not_use_omitted_display_summaries():
    engine, raw = shell_fixture(width=23, thickness=8)
    assert len(raw['flapCells']) > 4096
    engine.state['flapComponents'] = [{'cells': [], 'cellsOmitted': True, 'cellCount': len(raw['flapCells']), 'separable': None}]
    assert eligible_bone_component(engine).cells == tuple(raw['flapCells'])


@pytest.mark.parametrize('bad', [-1., 80.01, True, float('nan')])
def test_entry_angle_is_prospectively_bounded(bad):
    engine, raw = shell_fixture(); raw['entry']['maximumAngleDegrees'] = bad
    if not np.isfinite(bad):
        raw['contentHash'] = 'invalid'
    else: raw['contentHash'] = contract_hash(raw)
    with pytest.raises(ValueError): validate_contract(engine.scene, raw)


def test_full_affine_window_cells_and_exit_extent_are_checked():
    from scipy.spatial.transform import Rotation
    affine = np.eye(4); affine[:3, :3] = Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix() @ np.diag([.7, 1.1, 1.8]); affine[:3, 3] = [10., -5., 3.]
    engine, raw = shell_fixture(affine=affine)
    assert eligible_bone_component(engine).volume_mm3 == pytest.approx(len(raw['flapCells'])*.7*1.1*1.8)
    raw['entry']['halfExtentsMm'][0] -= .01; raw['contentHash'] = contract_hash(raw)
    with pytest.raises(ValueError, match='full affine cells'): validate_contract(engine.scene, raw)
    engine, raw = shell_fixture(); raw['exit']['maximumVoxelBoundary'][0] += 1.; raw['contentHash'] = contract_hash(raw)
    with pytest.raises(ValueError, match='voxel-face'): validate_contract(engine.scene, raw)


def test_forged_frozen_objects_are_fully_revalidated():
    engine, _ = shell_fixture(); contract = validate_contract(engine.scene)
    with pytest.raises(ValueError, match='content or source'):
        eligible_bone_component(engine, replace(contract, flap_cells=contract.flap_cells[:-1]))
    with pytest.raises(ValueError, match='COVERAGE_PROOFS_REQUIRED'):
        require_evidence(replace(contract, proofs=contract.proofs[:-1]))


def test_regrasp_requires_the_exact_typed_detached_object_and_valid_transform():
    engine, raw = shell_fixture(); contract = validate_contract(engine.scene)
    detached = {'kind': 'bone-flap-v1', 'cells': raw['flapCells'][:], 'offset': [0, 0, -1], 'baseOffset': [0, 0, 0],
        'contractHash': contract.content_hash, 'componentHash': semantic_digest(raw['flapCells'])}
    engine.state['detachedFragments'] = [detached]
    with pytest.raises(ValueError, match='MATERIAL_ALREADY_HELD'): eligible_bone_component(engine)
    assert eligible_bone_component(engine, allow_detached=detached).cells == contract.flap_cells
    with pytest.raises(ValueError, match='DETACHED_BINDING_INVALID'):
        eligible_bone_component(engine, allow_detached=copy.deepcopy(detached))
    detached['offset'][0] = True
    with pytest.raises(ValueError, match='DETACHED_TRANSFORM_INVALID'):
        eligible_bone_component(engine, allow_detached=detached)


def test_induced_native_case_hash_projection_is_stable_but_other_source_stays_bound():
    engine, raw = shell_fixture()
    engine, raw = rebuild(engine, raw, source={**engine.scene.source, 'caseHash': 'sha256:'+'a'*64})
    source = {**engine.scene.source, 'caseHash': 'sha256:'+'b'*64,
              'cranialAccessBaseCaseHash': 'sha256:'+'a'*64}
    scene = SurgicalScene(engine.scene.labels, engine.scene.affine, engine.scene.regions,
                          source, engine.scene.properties)
    assert validate_contract(scene).content_hash == raw['contentHash']
    assert binding(scene) == binding(engine.scene)
    with pytest.raises(ValueError, match='binding differs'):
        SurgicalScene(scene.labels, scene.affine, scene.regions,
                      {**source, 'changedOriginalMetadata': True}, scene.properties)


def test_present_null_is_not_an_absent_legacy_declaration():
    engine, _ = shell_fixture()
    with pytest.raises(ValueError, match='cannot be null'):
        SurgicalScene(engine.scene.labels, engine.scene.affine, engine.scene.regions,
                      engine.scene.source, {**engine.scene.properties, PROPERTY: None})


def test_retained_skull_islands_need_their_own_source_support():
    engine, raw = shell_fixture(); labels = engine.scene.labels.copy(); labels[0, 0, 0] = 1
    raw['retainedSkullCells'] = [0, *raw['retainedSkullCells']]
    engine, _ = rebuild(engine, raw, labels=labels)
    with pytest.raises(ValueError, match='RETAINED_SKULL_SUPPORT_INCOMPLETE'):
        eligible_bone_component(engine)


def test_corner_touching_flap_islands_are_not_one_rigid_face_connected_piece():
    engine, raw = shell_fixture()
    cells = [int(np.ravel_multi_index(point, engine.scene.shape)) for point in [(4, 4, 4), (5, 5, 4)]]
    raw['flapCells'] = cells; raw['kerfCells'] = sorted(set(raw['windowCells'])-set(cells))
    engine, raw = rebuild(engine, raw); engine.state['removed'] = raw['kerfCells'][:]
    with pytest.raises(ValueError, match='FLAP_NOT_FACE_CONNECTED'):
        eligible_bone_component(engine)


def test_declared_original_anchors_cannot_be_omitted_from_contract():
    engine, raw = shell_fixture()
    with pytest.raises(ValueError, match='source bone anchors'):
        rebuild(engine, raw, properties={'boneAnchors': [raw['flapCells'][0]]})


def test_exit_inside_source_anatomy_and_inadequate_room_do_not_authorize_grasp():
    engine, raw = shell_fixture(); raw['exit']['planeVoxelBoundary'] = 4.
    engine, _ = rebuild(engine, raw)
    with pytest.raises(ValueError, match='EXIT_NOT_SOURCE_EXTERIOR'): eligible_bone_component(engine)
    engine, raw = shell_fixture(); raw['exit']['minimumVoxelBoundary'][2] = 2.5
    engine, _ = rebuild(engine, raw)
    with pytest.raises(ValueError, match='EXIT_WORKSPACE_INSUFFICIENT'): eligible_bone_component(engine)


@pytest.mark.parametrize('membership', ['original-target', 'protected', 'unknown-target'])
def test_precut_scope_cannot_erase_target_or_protection_under_bone_ownership(membership):
    engine, raw = shell_fixture(); cell = raw['kerfCells'][0]
    if membership == 'protected':
        engine._protected_ids = frozenset([cell])
        reason = 'PROTECTED_STRUCTURE'
    else:
        mask = engine.scene.labels == 2; mask.ravel()[cell] = True
        engine, raw = rebuild(engine, raw, source={**engine.scene.source,
            'annotationHashes': {'declared target': array_digest(mask)}})
        if membership == 'original-target':
            _bind_original_target_memberships(engine.scene,
                {'declared target': immutable_array(np.flatnonzero(mask), np.int64)})
        reason = 'TARGET_OVERLAP' if membership == 'original-target' else 'ORIGINAL_TARGET_EVIDENCE_REQUIRED'
    engine.state['removed'] = []
    before = engine.snapshot()
    with pytest.raises(ValueError, match=reason): validate_bone_scope(engine)
    assert engine.snapshot() == before
