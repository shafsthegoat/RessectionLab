"""Small generated export controls: the marker is a provenance label, never a refusal (owner decision 2026-11-27)."""
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
    PROPERTY, inspection_case, export_inspection_case)

RETIRED = 'SOURCE_INSPECTION_ONLY'
INSPECT = {'slot': 'left', 'operation': 'inspect', 'duration': .5}


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


def test_marked_source_is_operable_on_the_reference_path(source):
    from resectionlab import surgical_native_engine
    engine = SurgicalEngine(from_case(built(source)))
    before = copy.deepcopy(engine.snapshot())
    preview = engine.preview(INSPECT)
    assert preview['preview'] and preview['status'] == 'accepted'
    assert engine.snapshot() == before
    result = engine.step(INSPECT)
    assert result['status'] == 'accepted' and result.get('reason') != RETIRED and len(engine.actions) == 1
    # The compiled admission still binds the marker to its retired contract.
    assert surgical_native_engine.create(engine) in ((None, 'legacy-source-marker'), (None, 'reference-selected'))


def test_save_cold_open_marker_identity_and_acted_snapshot_restore(source):
    destination = source.tmp/'inspection.ressectionlab'
    export_inspection_case(source.collection.root, destination, **source.options)
    first = built(source); cold = load_case(destination)
    assert cold.semantic_hash == first.semantic_hash
    assert thaw_json(cold.metadata) == thaw_json(first.metadata)
    scene = from_case(cold); engine = SurgicalEngine(scene)
    restored = SurgicalEngine.restore(scene, json.loads(json.dumps(engine.snapshot())), use_cache=False)
    assert restored.snapshot() == engine.snapshot()
    assert restored.step(dict(INSPECT, slot='right'))['status'] == 'accepted'
    acted = SurgicalEngine.restore(scene, json.loads(json.dumps(restored.snapshot())), use_cache=False)
    assert acted.snapshot() == restored.snapshot() and len(acted.actions) == 1
    assert np.array_equal(cold.source_compartments['inferior-container'], source.critical)


def test_marker_edit_is_detected_and_malformed_marker_is_only_a_label(source):
    from resectionlab.surgical_scene import SurgicalScene
    scene = from_case(built(source)); engine = SurgicalEngine(scene)
    del scene.properties[PROPERTY]
    with pytest.raises(ValueError, match='Source scene changed'):
        SurgicalEngine(scene)
    base = from_case(built(source))
    malformed = SurgicalScene(base.labels, base.affine, base.regions, base.source, {PROPERTY: False})
    assert SurgicalEngine(malformed).step(INSPECT)['status'] == 'accepted'


def test_marked_source_accepts_declared_protection_contract(source):
    from resectionlab.surgical_rl_protection import GuardConfig, ProtectionAnnotation, ProtectionSet, grid_binding
    scene = from_case(built(source))
    cells = tuple(int(i) for i in np.flatnonzero(scene.labels.reshape(-1) == 0)[:4])
    protection = ProtectionSet(scene, [ProtectionAnnotation('declared', cells, grid_binding(scene),
        'Generated test declaration', 'unreviewed', 'declared')], required=())
    engine = SurgicalEngine(scene, protection=protection, guard_config=GuardConfig(require_critical_coverage=False))
    assert engine.protection is protection and engine.guard_report()['configured']


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


def test_marker_adds_no_cranial_blocker_and_no_bone_offers_no_cranial_planner(source):
    from resectionlab import surgical_cranial_planning as planning
    scene = from_case(built(source)); engine = SurgicalEngine(scene)
    record = {'branchId': 'main', 'position': 0, 'branches': {'main': engine.snapshot()}}
    # Without bone or a root registry there is simply no cranial planner; the marker refuses nothing.
    assert planning.cranial_planning_availability(scene, record) is None
    with pytest.raises(ValueError, match='QUALIFIED_COMPLETE_SEQUENCE_ROOT_REQUIRED'):
        planning.compare_cranial_strategies(scene, record)


@pytest.mark.parametrize('command', ['step', 'preview'])
def test_real_bridge_admits_manual_actions_on_a_marked_source(source, command):
    from resectionlab.desktop_bridge import BridgeSession, _Request
    case = built(source)
    owner = BridgeSession(source.tmp/'bridge-transfers')
    try:
        owner._install_case(case, {}, _Request('install-inspection'))
        def call(args):
            return owner.execute('surgicalRehearsal', {'caseHash': case.semantic_hash, **args}, _Request(command), lambda *_: None)
        created = call({'command': 'create'})
        assert created['planningAdmission']['available'] and created['planningAdmission']['sourceClass'] == 'patient-or-unverified'
        result = call({'command': command, 'action': INSPECT, 'expectedStateId': created['view']['stateId']})
        outcome = result['outcome'] if command == 'step' else result
        assert outcome['status'] == 'accepted' and RETIRED not in str(outcome.get('reason'))
        assert command == 'preview' or result['position'] == 1
    finally:
        owner.close()


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
    assert engine.protection is None and engine.step(INSPECT)['status'] == 'accepted'


def test_explicit_mapping_protection_becomes_real_protection(source):
    case = built(source)
    mapping = [{'id': i+1, 'name': name, 'material': 'tumor' if name == 'model-target' else 'unknown',
                'target': name == 'model-target', 'protected': name != 'model-target'} for i, name in enumerate(case.compartments)]
    scene = from_case(case, mapping)
    rows = scene.source['protectedAnatomyV1']
    assert {row['structureId'] for row in rows} == {'inferior-container', 'padding-unknown'}
    assert 'inspectionProtectionDeclarations' not in scene.source
    by_name = {row['structureId']: row for row in rows}
    assert len(by_name['inferior-container']['cells']) == int(source.critical.sum())
    engine = SurgicalEngine(scene)
    assert engine.protection is not None and engine._protected_ids
    assert engine.guard_report()['configured']


def test_real_bridge_case_create_inspect_save_and_cold_open(source):
    from resectionlab.desktop_bridge import BridgeSession, _Request
    from resectionlab.imaging import save_case, read_case_artifacts
    case = built(source)
    owner = BridgeSession(source.tmp/'warm-transfers')
    cold_owner = BridgeSession(source.tmp/'cold-transfers')
    try:
        owner._install_case(case, {}, _Request('install-inspection'))
        def call(session, args):
            return session.execute('surgicalRehearsal', args, _Request('inspection'), lambda *_: None)
        created = call(owner, {'caseHash': case.semantic_hash, 'command': 'create'})
        assert created.get('cranialPlanningAvailability') is None  # No bone: no cranial planner, no marker blocker.
        assert created['position'] == 0 and created['view']['history'] == []
        assert created['planningAdmission']['available']
        raw = thaw_json(owner.cases[case.semantic_hash].artifacts)
        source_snapshot = raw['surgical_toolbox']['branches']['main']
        assert source_snapshot['actions'] == []
        with pytest.raises(Exception) as refused:
            call(owner, {'caseHash': case.semantic_hash, 'command': 'plan',
                'opponent': 'anatomy-cranial-strategy', 'acknowledgeResearchUnknowns': True})
        assert RETIRED not in str(refused.value) and 'inspection only' not in str(refused.value)
        path = save_case(case, source.tmp/'warm-saved.ressectionlab', artifacts=raw)
        cold = load_case(path)
        cold_owner._install_case(cold, read_case_artifacts(path), _Request('cold-install-inspection'))
        inspected = call(cold_owner, {'caseHash': cold.semantic_hash, 'command': 'inspect'})
        assert inspected['view']['stateId'] == created['view']['stateId'] and inspected['position'] == 0
        assert thaw_json(cold_owner.cases[cold.semantic_hash].artifacts)['surgical_toolbox']['branches']['main'] == source_snapshot
    finally:
        owner.close(); cold_owner.close()