"""Small generated export/authority controls; never patient execution."""
import copy
import hashlib
import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from resectionlab.core import thaw_json
from resectionlab.imaging import load_case
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import from_case, original_target_memberships
from resectionlab.surgical_source_inspection import (
    PROPERTY, REASON, inspection_case, export_inspection_case)


@pytest.fixture
def source(tmp_path):
    shape = (10, 9, 8)
    affine = np.diag([-.5, .5, .5, 1.]); affine[:3, 3] = [10., -3., 100.]
    collection = PatientImagingCollection.create(tmp_path/'collection', 'GENERATED:inspection', timepoint='preoperative')
    for i, name in enumerate(('MRI', 'CT-BRAIN', 'CT-BONE')):
        path = tmp_path/(name+'.nii.gz')
        image = nib.Nifti1Image((np.arange(np.prod(shape)).reshape(shape)+i).astype(np.float32), affine)
        image.header.set_xyzt_units('mm'); nib.save(image, path)
        collection.add_series(path, series_id=name, modality='T1CE' if i == 0 else 'CT')
    target = np.zeros(shape, np.uint8); target[3:6, 3:6, 3:5] = 1
    critical = np.zeros(shape, np.uint8); critical[:5] = 1
    unknown = np.zeros(shape, np.uint8); unknown[:, :, 0] = 1
    for name, mask, kind, role in [('model-target', target, 'tumor', 'target'),
                                  ('inferior-container', critical, 'brainstem', 'critical'),
                                  ('padding-unknown', unknown, 'unknown', 'critical')]:
        row = collection.add_layer(mask, affine, layer_id=name, source_series='MRI', kind=kind,
            role=role, color='#abcdef', provenance={'kind': 'model_inference' if role == 'target'
                else 'conservative_source_container', 'meaning': 'Generated metadata fixture only'})
        row['coverage'] = 'Generated support only; no clinical claim.'
    collection.save()
    options = dict(source_group='GENERATED:inspection', target_layer_ids=['model-target'],
        expected_originals={r['original']['name']: r['original']['sha256'] for r in collection.document['series']},
        expected_reference_grid={'seriesId': 'MRI', 'originalSha256': collection.series('MRI')['original']['sha256'],
            'shape': list(shape), 'affineRasMm': affine.tolist()},
        expected_target_cells=int(target.sum()), blockers=[{'code': 'LOCAL_MATERIAL_AND_STAGE_REVIEW_REQUIRED',
            'message': 'Local A/R, membrane, source space and complete native task remain unqualified.'}],
        expected_target_ids_sha256=hashlib.sha256(np.flatnonzero(target).astype('<u4').tobytes()).hexdigest())
    return SimpleNamespace(collection=collection, options=options, target=target.astype(bool),
                           critical=critical.astype(bool), unknown=unknown.astype(bool), tmp=tmp_path)


def built(source):
    return inspection_case(source.collection.root, **source.options)


def test_exact_grid_all_originals_arbitrary_masks_full_target_and_provenance_retained(source):
    case = built(source)
    assert np.array_equal(case.affine, source.collection.patient_affine(source.collection.series('MRI')))
    assert np.array_equal(case.mri, source.collection.array(source.collection.series('MRI')['data']))
    for name, mask in [('model-target', source.target), ('inferior-container', source.critical), ('padding-unknown', source.unknown)]:
        assert np.array_equal(case.compartments[name], mask)
        assert np.array_equal(case.source_compartments[name], mask)
    assert not {'brain', 'vessels', 'air', 'bone'} & set(case.compartments)
    assert len([r for r in case.source_refs if r.source_id.startswith('image:')]) == 3
    assert case.metadata['surgical_properties'][PROPERTY]['fullOriginalTargetCells'] == int(source.target.sum())
    assert case.metadata['source_inspection_annotations']['model-target']['sourceLayer']['provenance']['kind'] == 'model_inference'
    scene = from_case(case)
    assert len(original_target_memberships(scene)['model-target']) == int(source.target.sum())
    assert not scene.source.get('protectedAnatomyV1')  # No 67M-ID/KD allocation.
    assert set(r['material'] for r in scene.regions) == {'tumor', 'unknown'}
    assert int(np.count_nonzero(scene.labels == next(r['id'] for r in scene.regions if r['target']))) < int(source.target.sum())
    engine = SurgicalEngine(scene)
    assert engine.protection is None and engine.actions == []


@pytest.mark.parametrize('operation', ['configure', 'exchange', 'place', 'move', 'apply', 'withdraw', 'inspect', 'deactivate', 'activate'])
def test_every_direct_native_action_and_preview_is_rejected_before_transition(source, monkeypatch, operation):
    engine = SurgicalEngine(from_case(built(source)))
    before = copy.deepcopy(engine.snapshot()); metrics = engine.metrics()
    monkeypatch.setattr(engine, '_transition_clone', lambda: pytest.fail('Inspection cannot start a native transition'))
    action = {'slot': 'left', 'operation': operation}
    result = engine.step(action)
    assert result['status'] == 'rejected' and result['reason'] == REASON and result['completedFraction'] == 0
    preview = engine.preview(action)
    assert preview['status'] == 'rejected' and preview['reason'] == REASON and preview['preview']
    assert engine.snapshot() == before and engine.metrics() == metrics
    assert engine.actions == engine.history == []


def test_save_cold_open_marker_identity_and_empty_snapshot_restore_preserve_no_permission(source):
    destination = source.tmp/'inspection.ressectionlab'
    export_inspection_case(source.collection.root, destination, **source.options)
    first = built(source); cold = load_case(destination)
    assert cold.semantic_hash == first.semantic_hash
    assert thaw_json(cold.metadata) == thaw_json(first.metadata)
    scene = from_case(cold); engine = SurgicalEngine(scene)
    restored = SurgicalEngine.restore(scene, json.loads(json.dumps(engine.snapshot())), use_cache=False)
    assert restored.snapshot() == engine.snapshot()
    assert restored.step({'slot': 'right', 'operation': 'configure'})['reason'] == REASON
    assert np.array_equal(cold.source_compartments['inferior-container'], source.critical)


def test_marker_edit_cannot_unlock_the_same_scene_and_malformed_presence_is_blocked(source):
    from resectionlab.surgical_scene import SurgicalScene
    scene = from_case(built(source)); engine = SurgicalEngine(scene)
    del scene.properties[PROPERTY]
    with pytest.raises(ValueError, match='Source scene changed'):
        engine.step({'slot': 'left', 'operation': 'inspect'})
    with pytest.raises(ValueError, match='Source scene changed'):
        SurgicalEngine(scene)
    base = from_case(built(source))
    malformed = SurgicalScene(base.labels, base.affine, base.regions, base.source, {PROPERTY: False})
    assert SurgicalEngine(malformed).step({'slot': 'left', 'operation': 'inspect'})['reason'] == REASON


@pytest.mark.parametrize('field', ['protectionContract', 'cranialStagesContract', 'actions'])
def test_cold_inspection_rejects_executable_records_before_guard_or_replay(source, monkeypatch, field):
    from resectionlab import surgical_rl_protection as protection
    scene = from_case(built(source)); engine = SurgicalEngine(scene)
    snapshot = engine.snapshot()
    snapshot[field] = [{'slot': 'left', 'operation': 'configure'}] if field == 'actions' else {}
    monkeypatch.setattr(protection, 'restore_protection_contract', lambda *a: pytest.fail('No guard allocation'))
    monkeypatch.setattr(SurgicalEngine, 'step', lambda *a, **k: pytest.fail('No action replay'))
    with pytest.raises(ValueError, match='SOURCE_INSPECTION_ONLY'):
        SurgicalEngine.restore(scene, snapshot, use_cache=True)


@pytest.mark.parametrize('change', ['original', 'target-count', 'target-hash', 'grid', 'reference-grid', 'patient'])
def test_exact_export_contract_cannot_drop_original_target_or_frame(source, change):
    options = copy.deepcopy(source.options)
    if change == 'original': options['expected_originals'].pop('CT-BRAIN.nii.gz')
    elif change == 'target-count': options['expected_target_cells'] -= 1
    elif change == 'target-hash': options['expected_target_ids_sha256'] = '0'*64
    elif change == 'reference-grid': options['expected_reference_grid']['affineRasMm'][0][3] += .5
    elif change == 'patient': options['source_group'] = 'OTHER:person'
    else:
        source.collection.document['layers'][0]['affine'][0][3] += .5
        source.collection.save()
    with pytest.raises(ValueError, match='SOURCE_INSPECTION'):
        inspection_case(source.collection.root, **options)


def test_candidate_presence_blocks_no_bone_no_registry_and_direct_adapter_search(source, monkeypatch):
    from resectionlab import surgical_cranial_planning as planning
    scene = from_case(built(source)); engine = SurgicalEngine(scene)
    record = {'branchId': 'main', 'position': 0, 'branches': {'main': engine.snapshot()}}
    monkeypatch.setattr(planning, 'search_opening_forest', lambda *a, **k: pytest.fail('No planning on inspection source'))
    dto = planning.cranial_planning_availability(scene, record)
    assert dto['present'] and not dto['available'] and dto['roots'] == []
    assert dto['blockers'][0]['code'] == REASON
    assert dto['blockers'][1]['code'] == 'LOCAL_MATERIAL_AND_STAGE_REVIEW_REQUIRED'
    with pytest.raises(ValueError, match='QUALIFIED_COMPLETE_SEQUENCE_ROOT_REQUIRED'):
        planning.compare_cranial_strategies(scene, record)
    assert not engine.actions and not engine.history


@pytest.mark.parametrize('command', ['step', 'preview', 'plan', 'generateStrategies', 'generateApproaches', 'generatePreparations', 'attachProtection', 'fullCase'])
def test_real_bridge_refuses_manual_and_every_planner_before_session_state(source, tmp_path, command):
    from resectionlab.desktop_bridge import BridgeSession, BridgeError
    from resectionlab.surgical_bridge import dispatch
    case = built(source)
    owner = BridgeSession(tmp_path/'bridge-transfers')
    owner._get_case = lambda _: SimpleNamespace(case=case, artifacts={})
    args = {'caseHash': case.semantic_hash, 'command': command}
    if command == 'step': args['action'] = {'slot': 'left', 'operation': 'configure'}
    if command == 'plan': args.update(opponent='anatomy-strategy', acknowledgeResearchUnknowns=True)
    with pytest.raises(BridgeError, match='inspection only'):
        dispatch(owner, 'surgicalRehearsal', args, SimpleNamespace(check=lambda: None), lambda *a: None)
    assert not getattr(owner, 'surgical_sessions', {})


def test_existing_scene_without_marker_keeps_native_inspection_action():
    from resectionlab.surgical_scene import engineering_scene
    engine = SurgicalEngine(engineering_scene())
    assert engine.step({'slot': 'left', 'operation': 'inspect'})['status'] == 'accepted'


def test_manifest_is_pinned_before_payload_reads_and_rechecked_after_export(source):
    calls = 0
    def mutate_after_initial_snapshot():
        nonlocal calls
        calls += 1
        if calls == 2:
            path = source.collection.root/'patient-anatomy.json'
            path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='SOURCE_INSPECTION_MANIFEST_CHANGED'):
        inspection_case(source.collection.root, **source.options, check=mutate_after_initial_snapshot)


def test_existing_typed_critical_evidence_is_preserved_without_review_promotion(source):
    from resectionlab.core import SourceRef, array_digest
    from resectionlab.critical_evidence import CriticalStructureEvidence
    image = source.collection.series('MRI')
    annotation = SourceRef('generated-annotation', 'file:///generated-fixture/annotation.npy',
        'a'*64, license='Generated software fixture, not patient evidence', provenance='observed')
    identity = {'dataset': 'GENERATED', 'participant': 'inspection', 'timepoint': 'preoperative'}
    evidence = CriticalStructureEvidence(evidence_id='retained-unreviewed', structure='vessels',
        mask=source.critical, annotation_coverage=np.ones_like(source.critical),
        affine_ras_mm=source.collection.patient_affine(image),
        case_id='GENERATED:inspection-source-inspection',
        reference_image_hash=array_digest(source.collection.array(image['data'])),
        reference_source_id='image:MRI', reference_source_sha256=image['original']['sha256'],
        source=annotation, source_binding={**identity, 'linkage_source': annotation.to_dict(),
            'coverage_source': annotation.to_dict(), 'coverage_meaning': 'source_documented_annotation_domain'},
        derivation={'method': 'identity_grid', 'source_image_sha256': image['original']['sha256'],
                    'annotation_sha256': annotation.sha256},
        lineage={'kind': 'unknown', 'source': annotation.to_dict(), 'model_sha256': []})
    case = inspection_case(source.collection.root, **source.options,
        critical_evidence={evidence.evidence_id: evidence}, evidence_identity=identity)
    assert case.critical_evidence[evidence.evidence_id] is evidence
    assert evidence.review is None and evidence.lineage['kind'] == 'unknown'
    assert np.array_equal(case.critical_evidence[evidence.evidence_id].mask, source.critical)
    engine = SurgicalEngine(from_case(case))
    assert engine.protection is None and engine.step({'slot': 'left', 'operation': 'inspect'})['reason'] == REASON


def test_explicit_mapping_protection_retained_without_ids_or_index_allocation(source, monkeypatch):
    import resectionlab.surgical_scene as scene_module
    from resectionlab import surgical_rl_protection as protection
    case = built(source)
    mapping = [{'id': i+1, 'name': name, 'material': 'tumor' if name == 'model-target' else 'unknown',
                'target': name == 'model-target', 'protected': True} for i, name in enumerate(case.compartments)]
    original_flatnonzero = scene_module.np.flatnonzero
    def no_exclusion_expansion(mask):
        assert np.array_equal(mask, source.target), 'Only the small complete target can expand to source IDs'
        return original_flatnonzero(mask)
    monkeypatch.setattr(scene_module.np, 'flatnonzero', no_exclusion_expansion)
    monkeypatch.setattr(protection, 'declared_scene_protection', lambda *a: pytest.fail('No physical protection index during inspection'))
    scene = from_case(case, mapping)
    assert len(scene.source['inspectionProtectionDeclarations']) == len(mapping)
    assert 'protectedAnatomyV1' not in scene.source
    engine = SurgicalEngine(scene)
    assert engine.protection is None and not engine._protected_ids
    assert engine.step({'slot': 'left', 'operation': 'configure'})['reason'] == REASON
    assert 'physical protection index not constructed' in engine.guard_report()['unknownStructures'][0]


def test_real_bridge_case_create_inspect_save_and_cold_open_stay_unacted(source):
    from resectionlab.desktop_bridge import BridgeSession, _Request, BridgeError
    from resectionlab.imaging import save_case, read_case_artifacts
    case = built(source)
    owner = BridgeSession(source.tmp/'warm-transfers')
    cold_owner = BridgeSession(source.tmp/'cold-transfers')
    try:
        owner._install_case(case, {}, _Request('install-inspection'))
        def call(session, args):
            return session.execute('surgicalRehearsal', args, _Request('inspection'), lambda *_: None)
        created = call(owner, {'caseHash': case.semantic_hash, 'command': 'create'})
        available = created['cranialPlanningAvailability']
        assert available['present'] and not available['available'] and available['roots'] == []
        assert available['blockers'][0]['code'] == REASON
        assert created['position'] == 0 and created['view']['history'] == []
        raw = thaw_json(owner.cases[case.semantic_hash].artifacts)
        source_snapshot = raw['surgical_toolbox']['branches']['main']
        assert source_snapshot['actions'] == [] and 'protectionContract' not in source_snapshot
        with pytest.raises(BridgeError, match='inspection only'):
            call(owner, {'caseHash': case.semantic_hash, 'command': 'plan',
                'opponent': 'anatomy-cranial-strategy', 'acknowledgeResearchUnknowns': True})
        assert thaw_json(owner.cases[case.semantic_hash].artifacts) == raw
        path = save_case(case, source.tmp/'warm-saved.ressectionlab', artifacts=raw)
        cold = load_case(path)
        cold_owner._install_case(cold, read_case_artifacts(path), _Request('cold-install-inspection'))
        inspected = call(cold_owner, {'caseHash': cold.semantic_hash, 'command': 'inspect'})
        assert inspected['view']['stateId'] == created['view']['stateId']
        assert inspected['position'] == 0 and inspected['cranialPlanningAvailability'] == available
        assert thaw_json(cold_owner.cases[cold.semantic_hash].artifacts)['surgical_toolbox']['branches']['main'] == source_snapshot
    finally:
        owner.close(); cold_owner.close()
