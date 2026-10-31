"""Generated native-source controls for admission; no independent patient claim."""
import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from resectionlab import surgical_task_admission as admission
from resectionlab.core import semantic_digest
from resectionlab.imaging import load_case, read_case_artifacts, save_case
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_patient import prepare_surgical_environment
from resectionlab.surgical_interactive_planning import (generate_interactive_approaches,
    interactive_planning_availability, materialize_interactive_approach, validate_saved_interactive_report)


def step(engine, operation, *, slot='right', **values):
    outcome = engine.step({'slot': slot, 'operation': operation, 'duration': .1, **values})
    assert outcome['status'] == 'accepted', outcome


def pose(z): return {'position': [18., 18., z], 'quaternion': [0., 0., 0., 1.]}


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp('generated-task-admission')
    declaration_root = root/'declarations'; (declaration_root/'generated').mkdir(parents=True)
    declaration = {'role': 'DEVELOPMENT', 'cases': ['different-patient-A', 'different-patient-B'], 'mirror_revision': 'generated-native-revision'}
    (declaration_root/'generated/development-declaration.json').write_text(json.dumps(declaration))
    patch = pytest.MonkeyPatch(); patch.setattr(admission, '_DECLARATIONS_ROOT', declaration_root)
    results = []
    try:
        for patient in declaration['cases']:
            folder = root/patient; folder.mkdir()
            shape = (36, 36, 36); image = nib.Nifti1Image(np.indices(shape).sum(axis=0).astype('float32'), np.eye(4))
            image.header.set_xyzt_units('mm'); nib.save(image, folder/'scan.nii.gz')
            collection = PatientImagingCollection.create(folder/'native', patient)
            collection.document['study_timepoint'] = 'preoperative'
            collection.add_series(folder/'scan.nii.gz', series_id='T1', modality='T1', timepoint='preoperative')
            collection.series('T1')['source_mirror_revision'] = declaration['mirror_revision']
            brain = np.zeros(shape, np.uint8); brain[7:29, 7:29, 10:34] = 1
            tumor = np.zeros(shape, np.uint8); tumor[14:23, 14:23, 20:32] = 1
            label_image = nib.Nifti1Image(tumor, np.eye(4)); label_image.header.set_xyzt_units('mm')
            nib.save(label_image, folder/'annotation.nii.gz')
            collection.add_layer(brain, np.eye(4), layer_id='brain', source_series='T1', kind='brain', role='context', color='#aaaaaa')
            collection.add_layer(tumor, np.eye(4), layer_id='tumor', source_series='T1', kind='tumor', role='target', color='#aa00aa',
                provenance={'kind': 'observed_source_annotation', 'source': collection.original(folder/'annotation.nii.gz')})
            collection.save()
            exported = prepare_surgical_environment(collection.root, folder/'scene', layer_ids={'brain': 'brain', 'tumor': 'tumor'})
            path = Path(exported['casePath']); case = load_case(path); record = read_case_artifacts(path)['surgical_toolbox']
            e = SurgicalEngine.restore(load_environment(path), record['branches']['main'])
            step(e, 'place', slot='left', pose={'position': [-20., -20., -20.], 'quaternion': [0., 0., 0., 1.]})
            step(e, 'place', pose=pose(0.), parameters={'radiusMm': 5.})
            step(e, 'apply', pose=pose(20.))
            step(e, 'withdraw')
            step(e, 'inspect', parameters={'radiusMm': 1.5, 'power': 1., 'duty': 1.})
            step(e, 'place', pose={'position': [45., 45., -10.], 'quaternion': [0., 0., 0., 1.]})
            step(e, 'configure', mechanicsRevision=2, clockRevision=1)
            task = admission.task_spec(entry={'centerMm': [18., 18., 12.], 'radiusMm': 2.},
                axis=[0., 0., 1.], target={'centerMm': [18., 18., 28.], 'radiusMm': 8.}, allow_unreviewed=True)
            contract = admission.prepare_admission(e, native_collection=collection.root, task=task)
            admission.bind_admission(e, contract)
            record.update(branchId='admitted', position=len(e.actions), taskAdmission=contract,
                branches={'main': record['branches']['main'], 'admitted': e.snapshot()},
                parents={'main': None, 'admitted': {'branchId': 'main', 'position': 0}}, timeline={'admitted': e.history})
            results.append((case, e, record, collection))
        yield results
    finally:
        patch.undo()


@pytest.mark.parametrize('index', [0, 1])
def test_same_declarative_interface_admits_two_distinct_sources(prepared, index):
    case, e, record, _ = prepared[index]
    before = e.snapshot(); contract = record['taskAdmission']
    assert admission.validate_admission(e, contract)
    assert contract['sourceEvidence']['patientId'] == f'different-patient-{"AB"[index]}'
    assert contract['sourceEvidence']['frame'] == 'RAS+'
    assert set(contract['sourceEvidence']['unknownEvidence']) == set(admission.UNKNOWN_ROLES)
    assert contract['sourceEvidence']['anatomy']['brain']['reviewStatus'] == 'unreviewed'
    assert contract['task']['inputContract'] == admission.INPUT_CONTRACT
    available = interactive_planning_availability(e)
    assert available['available'] and available['admission'] == contract
    assert available['evidence']['sourcePatient'] == contract['sourceEvidence']['patientId']
    assert available['defaults']['entry'] == contract['task']['entry']
    assert e.snapshot() == before and case.semantic_hash == e.scene.source['caseHash']


@pytest.mark.parametrize('field', ['sourceEvidence', 'models', 'baseline', 'verification'])
def test_resealed_admission_cannot_exchange_bound_facts(prepared, field):
    _, e, record, _ = prepared[0]
    changed = copy.deepcopy(record['taskAdmission'])
    if field == 'sourceEvidence': changed[field]['anatomy']['brain']['reviewStatus'] = 'reviewed'
    elif field == 'models': changed[field]['physicalClockRevision'] = 0
    elif field == 'baseline': changed[field]['position'] += 1
    else: changed[field]['verifiedOriginalSha256'] = []
    changed.pop('admissionSeal'); changed['admissionSeal'] = semantic_digest(changed)
    with pytest.raises(ValueError, match='TASK_ADMISSION'):
        admission.validate_admission(e, changed)


def test_observation_contract_and_allowed_actions_cannot_be_extended_by_resealing(prepared):
    _, e, record, _ = prepared[0]
    for key, value in [('allowedOperations', ['apply']), ('evaluationOnlyInputs', ['private-target']), ('inputContract', 'MRI-only')]:
        changed = copy.deepcopy(record['taskAdmission']); changed['task'][key] = value
        changed.pop('admissionSeal'); changed['admissionSeal'] = semantic_digest(changed)
        with pytest.raises(ValueError, match='task scope'):
            admission.validate_admission(e, changed)


def test_changed_state_and_missing_review_acceptance_refuse(prepared):
    _, e, record, _ = prepared[0]
    different = e.clone(); step(different, 'inspect')
    assert not interactive_planning_availability(different)['available']
    changed = copy.deepcopy(record['taskAdmission']); changed['task']['allowUnreviewedEvidence'] = False
    changed.pop('admissionSeal'); changed['admissionSeal'] = semantic_digest(changed)
    with pytest.raises(ValueError, match='unreviewed'):
        admission.validate_admission(e, changed)


def test_acquired_byte_verification_is_not_replaced_by_caller_hashes(prepared, tmp_path):
    _, e, record, collection = prepared[0]
    path = collection.path(collection.series('T1')['original']['path']); before = path.read_bytes()
    try:
        path.write_bytes(before+b'changed')
        with pytest.raises(ValueError, match='original acquired image or annotation bytes differ'):
            admission.prepare_admission(e, native_collection=collection.root, task=record['taskAdmission']['task'])
    finally:
        path.write_bytes(before)


def test_cold_binding_rechecks_original_annotation_bytes(prepared):
    _, e, record, collection = prepared[0]
    path = collection.path(collection.layer('tumor')['provenance']['source']['path']); before = path.read_bytes()
    try:
        path.write_bytes(before+b'changed annotation')
        admission._SOURCE_CHECKS.clear()
        with pytest.raises(ValueError, match='original acquired image or annotation bytes differ'):
            admission.bind_admission(SurgicalEngine.restore(e.scene, e.snapshot()), record['taskAdmission'])
    finally:
        path.write_bytes(before)


def test_resealed_source_label_exchange_fails_native_ownership_reconstruction(prepared):
    from resectionlab.surgical_scene import SurgicalScene
    _, e, record, _ = prepared[0]
    labels = e.scene.labels.copy(); labels[0, 0, 0] = 1
    scene = SurgicalScene(labels, e.scene.affine, e.scene.regions, e.scene.source, e.scene.properties)
    changed = copy.deepcopy(record['taskAdmission']); changed['sourceSceneId'] = scene.identity
    changed['sourceEvidence'] = admission.source_evidence(scene)
    changed.pop('admissionSeal'); changed['admissionSeal'] = semantic_digest(changed)
    with pytest.raises(ValueError, match='material ownership does not match'):
        admission.bind_admission(SurgicalEngine(scene), changed)


@pytest.mark.parametrize('name', ['entry', 'target'])
def test_edited_regions_cannot_escape_admitted_bounds(prepared, name):
    _, e, _, _ = prepared[0]
    defaults = interactive_planning_availability(e)['defaults']; defaults[name]['centerMm'][0] += .1
    from resectionlab.surgical_interactive_planning import normalize_constraints
    with pytest.raises(ValueError, match='exceeds the admitted task region'):
        normalize_constraints(e, defaults)


def test_missing_target_has_explicit_refusal_in_common_interface():
    from resectionlab.surgical_scene import SurgicalScene
    labels = np.ones((4, 4, 4), np.uint16)
    e = SurgicalEngine(SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'brain', 'material': 'soft_tissue'}], {}, {}))
    available = interactive_planning_availability(e)
    assert not available['available'] and 'tumor target is required' in available['reason']
    with pytest.raises(ValueError, match='source-declared tumor target'):
        admission.source_evidence(e.scene)


def test_later_acquired_observation_cannot_enter_preoperative_task(prepared):
    from resectionlab.patient_anatomy import digest
    from resectionlab.surgical_scene import SurgicalScene
    _, e, _, _ = prepared[0]
    source = copy.deepcopy(e.scene.source); adapter = source['adapter']; manifest = adapter['sourceManifest']
    manifest['series'][0]['timepoint'] = 'postoperative'
    manifest['snapshot_id'] = digest({k: v for k, v in manifest.items() if k != 'snapshot_id'})
    adapter['sourceCollectionHash'] = manifest['snapshot_id']
    scene = SurgicalScene(e.scene.labels, e.scene.affine, e.scene.regions, source, e.scene.properties)
    with pytest.raises(ValueError, match='preoperative source stage'):
        admission.source_evidence(scene)


def test_free_air_cannot_be_declared_as_locally_executed_exposure(prepared):
    _, e, record, _ = prepared[0]
    changed = copy.deepcopy(record['taskAdmission']); changed['task']['entry']['centerMm'] = [-10., -10., -10.]
    changed.pop('admissionSeal'); changed['admissionSeal'] = semantic_digest(changed)
    with pytest.raises(ValueError, match='locally removed source tissue'):
        admission.validate_admission(e, changed)


def test_interior_goal_uses_first_contact_and_preserves_historical_geometry_rejection():
    from resectionlab.surgical_scene import SurgicalScene
    from resectionlab.surgical_joint_geometry import post_exposure_proposal
    labels = np.zeros((40, 40, 40), np.uint16); labels[16:21, 16:21, 20:31] = 1
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'target', 'material': 'tumor', 'target': True}],
        {'kind': 'generated first-contact geometry'}, {})
    e = SurgicalEngine(scene)
    step(e, 'configure', mechanicsRevision=2, clockRevision=1)
    entry, goal = np.array([18., 18., 12.]), np.array([18., 18., 27.])
    old, rejection = post_exposure_proposal(e, entry, goal, 1.5, lambda: None)
    assert old is None and rejection['reason'] == 'SHAFT_COLLISION'
    route, rejection = post_exposure_proposal(e, entry, goal, 1.5, lambda: None, stop_at_first_contact=True)
    assert rejection is None and route['targetPose']['position'][2] < 20.
    for action in route['actionProgram']+route['withdrawalProgram']:
        assert e.step(action)['status'] == 'accepted'
    assert not e.state['removed']


@pytest.mark.parametrize('obstacle,reason', [('protected', 'PROTECTED_STRUCTURE'),
    ('non-target', 'NON_TARGET_TRAVERSAL_REQUIRED'), ('margin', 'NON_TARGET_CLEARANCE'), ('blocked-start', 'SHAFT_COLLISION')])
def test_first_contact_refuses_earlier_constraints_with_correct_reason(obstacle, reason):
    from resectionlab.surgical_scene import SurgicalScene
    from resectionlab.surgical_joint_geometry import post_exposure_proposal
    from resectionlab.surgical_rl_protection import ProtectionAnnotation, ProtectionSet, GuardConfig
    labels = np.zeros((40, 40, 90), np.uint16)
    labels[16:21, 16:21, 70:81] = 1
    if obstacle == 'non-target': labels[18, 18, 66] = 2
    if obstacle == 'margin': labels[21, 18, 66] = 2
    if obstacle == 'blocked-start': labels[18, 18, 30] = 2
    affine = np.eye(4); affine[2, 3] = -50.
    scene = SurgicalScene(labels, affine, [{'id': 1, 'name': 'target', 'material': 'tumor', 'target': True},
        {'id': 2, 'name': 'retained brain', 'material': 'soft_tissue'}], {'kind': 'generated first-contact refusal'}, {})
    options = {}
    if obstacle == 'protected':
        mask = np.zeros(labels.shape, bool); mask[18, 18, 70] = True
        protection = ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem', mask,
            provenance='Generated exclusion control')], clearance_mm=0.)
        options = {'protection': protection, 'guard_config': GuardConfig(require_critical_coverage=False)}
    e = SurgicalEngine(scene, **options); step(e, 'configure', mechanicsRevision=2, clockRevision=1)
    route, rejection = post_exposure_proposal(e, np.array([18., 18., 12.]), np.array([18., 18., 27.]), 1.5,
        lambda: None, stop_at_first_contact=True, non_target_clearance_mm=1. if obstacle == 'margin' else 0.)
    assert route is None and rejection['reason'] == reason


def test_missing_required_evidence_is_a_sealed_refusal(prepared):
    _, e, _, _ = prepared[0]
    defaults = interactive_planning_availability(e)['defaults']; defaults['requiredEvidence'].append('functional')
    report = generate_interactive_approaches(e, defaults)
    assert report['schema'].endswith('.v2') and report['status'] == 'required_evidence_unavailable'
    assert report['missingRequiredEvidence'] == ['functional'] and not report['routes']
    assert validate_saved_interactive_report(e.scene, e.snapshot(), report)


def test_generation_exact_native_materialization_and_workspace_reopen(prepared, tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request
    case, e, record, _ = prepared[0]
    before = e.snapshot(); report = generate_interactive_approaches(e, interactive_planning_availability(e)['defaults'])
    assert report['schema'].endswith('.v2') and report['routes'], report['rejections'][:5]
    candidate, receipt = materialize_interactive_approach(e, report, report['routes'][0]['id'])
    assert receipt['publicAccessAccepted'] and receipt['publicWithdrawalAccepted']
    assert candidate.state['removed'] == e.state['removed'] and e.snapshot() == before
    assert validate_saved_interactive_report(e.scene, before, report)
    path = tmp_path/'admitted.ressectionlab'; save_case(case, path, artifacts={'surgical_toolbox': record})
    session = BridgeSession(tmp_path/'transfers')
    def call(op, **values): return session.execute(op, values, _Request('admission'), lambda *_: None)
    try:
        loaded = call('loadCase', path=str(path))
        opened = call('surgicalRehearsal', caseHash=loaded['caseHash'], command='inspect')
        assert opened['interactivePlanningAvailability']['available']
        assert opened['interactivePlanningAvailability']['admission'] == record['taskAdmission']
        generated = call('surgicalRehearsal', caseHash=loaded['caseHash'], command='generateApproaches',
            expectedStateId=opened['view']['stateId'], constraints=report['constraints'], acknowledgeResearchUnknowns=True)
        assert generated['interactivePlanning']['routes']
        save = tmp_path/'saved.ressectionlab'; call('saveCase', caseHash=loaded['caseHash'], path=str(save))
        assert read_case_artifacts(save)['surgical_toolbox']['taskAdmission'] == record['taskAdmission']
    finally:
        session.close()
