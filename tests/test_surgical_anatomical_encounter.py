"""Generated source-bound endpoint controls; no biological validation claim."""
import copy
import shutil
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_patient import prepare_surgical_environment
from resectionlab.surgical_task_admission import admission, prepare_admission, bind_admission
from resectionlab.surgical_anatomical_encounter import (AnatomicalObservation, EncounterTracker,
    axial_envelope, context_options, project_context)
from resectionlab.surgical_two_tool import (TwoToolObservation, task_spec, defaults, generate_comparison,
    plan_observed_strategies, validate_report, materialize)
from resectionlab.surgical_planning_observation import project_observation
from test_surgical_two_tool import qualified, prepared


@pytest.fixture(scope='module')
def context_case(qualified, prepared, tmp_path_factory):
    _, original, _, _ = qualified; native = prepared[0][3]
    folder = tmp_path_factory.mktemp('generated-source-context')
    collection = PatientImagingCollection.create(folder/'native', native.document['patient_id'])
    collection.document['study_timepoint'] = 'preoperative'
    collection.add_series(native.path(native.series('T1')['original']['path']), series_id='T1', modality='T1', timepoint='preoperative')
    collection.series('T1')['source_mirror_revision'] = native.series('T1')['source_mirror_revision']
    brain = native.array(native.layer('brain')['data']) == 1
    tumor = native.array(native.layer('tumor')['data']) == 1
    categorical = np.where(brain, 2, 0).astype(np.uint16); categorical[tumor] = 4
    image = nib.Nifti1Image(categorical, np.eye(4)); image.header.set_xyzt_units('mm')
    nib.save(image, folder/'generated-annotation.nii.gz')
    provenance = {'kind': 'observed_source_annotation', 'source': collection.original(folder/'generated-annotation.nii.gz')}
    collection.add_layer(brain.astype(np.uint8), np.eye(4), layer_id='brain', source_series='T1', kind='brain', role='context', color='#aaaaaa')
    collection.add_layer(categorical, np.eye(4), layer_id='target', source_series='T1', kind='tumor', role='target',
        label_value=4, color='#aa00aa', provenance=provenance)
    collection.add_layer(categorical, np.eye(4), layer_id='context', source_series='T1', kind='context', role='context',
        label_value=2, color='#eeee00', provenance=provenance)
    collection.save()
    exported = prepare_surgical_environment(collection.root, folder/'scene', layer_ids={'brain': 'brain', 'tumor': 'target'})
    from resectionlab.imaging import load_case, read_case_artifacts
    path = Path(exported['casePath']); case = load_case(path); record = read_case_artifacts(path)['surgical_toolbox']
    e = SurgicalEngine(load_environment(path))
    for action in original.actions: assert e.step(copy.deepcopy(action))['status'] != 'rejected'
    value = prepare_admission(e, native_collection=collection.root, task=admission(original)['task']); bind_admission(e, value)
    record.update(taskAdmission=value, branchId='admitted', position=len(e.actions),
        branches={'main': record['branches']['main'], 'admitted': e.snapshot()},
        parents={'main': None, 'admitted': {'branchId': 'main', 'position': 0}}, timeline={'admitted': copy.deepcopy(e.history)})
    controls = defaults(e)['defaults']; controls['geometry'].update(candidateCount=1, maxCandidates=8)
    controls['encounter'] = {'layerId': 'context', 'marginMm': 0., 'maxNearMm3': 1000.}
    return case, e, record, controls, collection


@pytest.fixture(scope='module')
def context_report(context_case):
    report = generate_comparison(context_case[1], context_case[3])
    assert report['publicPlan']['decision']['operation'] == 'COMPARE'
    return report


def test_exact_source_context_binding_and_native_prefixes(context_case, context_report):
    _, e, _, _, _ = context_case
    options = context_options(e); assert len(options) == 1 and options[0]['layerId'] == 'context'
    obs, binding = project_context(e, 'context')
    assert context_report['contextBinding'] == binding
    assert binding['labelValue'] == 2 and binding['reviewStatus'] != 'reviewed'
    assert obs.manifest()['sampling'] == 'exact-identity-source-cell-membership'
    assert not obs.mask.flags.writeable and not obs.affine.flags.writeable
    assert validate_report(e, context_report)
    for row in context_report['publicPlan']['alternatives']:
        native, _, receipt = materialize(e, context_report, row['id'])
        assert receipt['anatomicalEncounter'] == row['anatomicalEncounter']
        assert row['anatomicalEncounter']['newRemovedContextMm3'] == 0.
        assert row['anatomicalEncounter']['baselineRemovedContextMm3'] > 0.
        assert SurgicalEngine.restore(native.scene, native.snapshot()).snapshot() == native.snapshot()


def test_declared_margin_filters_candidates_and_reports_stop(context_case):
    _, e, _, controls, _ = context_case
    raw = copy.deepcopy(controls); raw['encounter'].update(marginMm=5., maxNearMm3=0.)
    report = generate_comparison(e, raw)
    assert report['publicPlan']['decision']['operation'] == 'STOP'
    assert report['publicPlan']['decision']['reason'] == 'ANATOMICAL_CONTEXT_LIMIT'
    rejected = [r for r in report['publicPlan']['rejections'] if r['reason'].startswith('ANATOMICAL_CONTEXT_LIMIT')]
    assert rejected and all(r['anatomicalEncounter']['nearMm3'] > 0. for r in rejected)
    assert validate_report(e, report)
    same, _, receipt = materialize(e, report, 'stop')
    assert same.snapshot() == e.snapshot() and receipt['anatomicalEncounter']['nearMm3'] == 0.


def test_context_constraints_and_private_references_do_not_cross_the_public_boundary(context_case):
    _, e, _, controls, _ = context_case
    context, _ = project_context(e, 'context'); value = admission(e)
    base = project_observation(e, value['task'], ['brain', 'target'])
    observed = TwoToolObservation(base, task_spec(with_context=True), context)
    raw = copy.deepcopy(controls); raw['encounter'].pop('layerId')
    first = plan_observed_strategies(observed, raw)
    changed = e.clone(); changed._private_evaluation_truth = np.ones(e.scene.shape, bool)
    changed._private_evaluator = lambda: pytest.fail('Private reference used')
    other = TwoToolObservation(project_observation(changed, value['task'], ['brain', 'target']), task_spec(with_context=True), context)
    assert first == plan_observed_strategies(other, raw)
    assert 'layerId' not in str(observed.manifest()) and 'sourceCollectionHash' not in str(observed.manifest())


@pytest.mark.parametrize('field', ['binding', 'mask-hash', 'metric', 'prefix', 'limit', 'feasible'])
def test_self_resealed_context_forgery_is_rejected(context_case, context_report, field):
    _, e, _, _, _ = context_case; report = copy.deepcopy(context_report); public = report['publicPlan']
    if field == 'binding': report['contextBinding']['reviewStatus'] = 'reviewed'
    elif field == 'mask-hash': public['publicObservation']['anatomicalContext']['maskHash'] = 'private-canary'
    elif field == 'metric': public['alternatives'][0]['anatomicalEncounter']['nearMm3'] = 123.
    elif field == 'prefix': public['alternatives'][0]['contextPrefixHashes'][0] = 'private-canary'
    elif field == 'limit': public['constraints']['encounter']['maxNearMm3'] = -1.
    else: public['alternatives'][0]['feasible'] = False
    public['publicPlanSeal'] = semantic_digest({k:v for k,v in public.items() if k != 'publicPlanSeal'})
    report['reportSeal'] = semantic_digest({k:v for k,v in report.items() if k != 'reportSeal'})
    with pytest.raises(ValueError): validate_report(e, report)


def test_context_budget_and_exact_envelope_are_independent_of_action_partition(context_case):
    _, e, _, _, _ = context_case; context, _ = project_context(e, 'context')
    tracker = EncounterTracker(context, e, {'marginMm': 3., 'maxNearMm3': 1000.})
    start = {'position': [18., 18., 0.], 'quaternion': [0., 0., 0., 1.]}
    mid = {**start, 'position': [18., 18., 14.]}; end = {**start, 'position': [18., 18., 28.]}
    slot = copy.deepcopy(e.state['slots']['left']); slot['instrument'] = 'cusa'; params = e.parameters(slot)
    tracker.add_cover(e, axial_envelope(start, end, 'cusa', params))
    other = EncounterTracker(context, e, tracker.limit)
    for a,b in ((start,mid),(mid,end),(end,start)): other.add_cover(e, axial_envelope(a,b,'cusa',params))
    assert tracker.receipt(e) == other.receipt(e)
    assert tracker.receipt(e)['nearMm3'] >= tracker.receipt(e)['shaftNearMm3']
    with pytest.raises(ValueError, match='qualified CUSA/Frazier'):
        axial_envelope(start, end, 'tumor_forceps', params)  # Roll-sensitive jaws need a different sweep proof.


def test_anatomical_stop_requires_complete_ordered_candidate_coverage(context_case):
    _, e, _, controls, _ = context_case
    raw = copy.deepcopy(controls); raw['geometry'].update(candidateCount=2, maxCandidates=20)
    raw['encounter'].update(marginMm=5., maxNearMm3=0.)
    report = generate_comparison(e, raw)
    assert len(report['publicPlan']['approaches']['routes']) == 2
    assert report['publicPlan']['decision']['reason'] == 'ANATOMICAL_CONTEXT_LIMIT'
    assert len(report['publicPlan']['rejections']) == 2 and validate_report(e, report)
    for kind in ('omit', 'duplicate', 'reverse', 'invent', 'truncate-parent'):
        forged = copy.deepcopy(report); public = forged['publicPlan']
        if kind == 'omit': public['rejections'].pop()
        elif kind == 'duplicate': public['rejections'].append(copy.deepcopy(public['rejections'][0]))
        elif kind == 'reverse': public['rejections'].reverse()
        elif kind == 'invent': public['rejections'][0]['anatomicalEncounter']['nearMm3'] = 12345.
        else:
            public['rejections'].pop(); parent = public['approaches']; parent['routes'].pop()
            parent['status'] = 'insufficient_distinct_routes'
            parent['decision'].update(candidateIds=[row['id'] for row in parent['routes']], reason=parent['status'])
            parent['publicPlanSeal'] = semantic_digest({k:v for k,v in parent.items() if k != 'publicPlanSeal'})
        public['publicPlanSeal'] = semantic_digest({k:v for k,v in public.items() if k != 'publicPlanSeal'})
        forged['reportSeal'] = semantic_digest({k:v for k,v in forged.items() if k != 'reportSeal'})
        with pytest.raises(ValueError, match='coverage'): validate_report(e, forged)


def test_changing_public_context_changes_feasibility_while_private_truth_stays_fixed(context_case):
    _, e, _, controls, _ = context_case
    context, _ = project_context(e, 'context')
    base = project_observation(e, admission(e)['task'], ['brain', 'target'])
    raw = copy.deepcopy(controls); raw['encounter'].pop('layerId'); raw['encounter'].update(marginMm=5., maxNearMm3=0.)
    first = plan_observed_strategies(TwoToolObservation(base, task_spec(with_context=True), context), raw)
    distant = np.zeros(e.scene.shape, bool); distant[0, 35, 35] = True
    other = AnatomicalObservation(distant, e.scene.affine, 'unreviewed')
    second = plan_observed_strategies(TwoToolObservation(base, task_spec(with_context=True), other), raw)
    assert first['decision']['reason'] == 'ANATOMICAL_CONTEXT_LIMIT'
    assert second['decision']['operation'] == 'COMPARE'
    assert first['publicObservation'] != second['publicObservation']


def test_context_selection_and_source_bytes_are_rechecked_on_warm_validation(context_case, context_report):
    _, e, _, _, collection = context_case
    for layer in ('target', 'brain', 'missing'):
        with pytest.raises(ValueError, match='unavailable'): project_context(e, layer)
    assert validate_report(e, context_report)
    path = collection.path(collection.layer('context')['provenance']['source']['path']); original = path.read_bytes()
    try:
        path.write_bytes(original+b'changed context source')
        with pytest.raises(ValueError, match='original acquired image or annotation bytes differ'): validate_report(e, context_report)
    finally: path.write_bytes(original)


def test_empty_or_misregistered_context_is_not_safe_empty_evidence(context_case):
    _, e, _, _, _ = context_case
    with pytest.raises(ValueError, match='nonempty'):
        AnatomicalObservation(np.zeros(e.scene.shape, bool), e.scene.affine, 'unreviewed')
    shear = e.scene.affine.copy(); shear[0, 1] = .1
    with pytest.raises(ValueError, match='orthogonal'):
        AnatomicalObservation(np.ones(e.scene.shape, bool), shear, 'unreviewed')
    context, _ = project_context(e, 'context'); shifted = context.affine.copy(); shifted[0, 3] += 1.
    other = AnatomicalObservation(context.mask, shifted, 'unreviewed')
    base = project_observation(e, admission(e)['task'], ['brain', 'target'])
    with pytest.raises(ValueError, match='frame differs'): TwoToolObservation(base, task_spec(with_context=True), other)
    damaged = AnatomicalObservation(context.mask, context.affine, 'unreviewed')
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always'); damaged.mask.dtype = np.uint8
    assert all(issubclass(w.category, DeprecationWarning) and 'Setting the dtype' in str(w.message) for w in emitted)
    with pytest.raises(ValueError, match='layout changed'): damaged.manifest()
    with pytest.raises(ValueError, match='layout changed'): EncounterTracker(damaged, e, {'marginMm': 0., 'maxNearMm3': 0.})


def test_explicit_unknown_context_units_refuse_without_breaking_the_qualified_parent_task(context_case, tmp_path):
    _, original, _, _, native = context_case
    root = tmp_path/'unknown-context-units'; shutil.copytree(native.root, root)
    other = PatientImagingCollection(root); other.layer('context')['units_status'] = 'unknown'; other.save()
    exported = prepare_surgical_environment(root, tmp_path/'scene', layer_ids={'brain': 'brain', 'tumor': 'target'})
    engine = SurgicalEngine(load_environment(exported['casePath']))
    for action in original.actions: assert engine.step(copy.deepcopy(action))['status'] != 'rejected'
    value = prepare_admission(engine, native_collection=root, task=admission(original)['task']); bind_admission(engine, value)
    assert defaults(engine)['available'] and context_options(engine) == []
    with pytest.raises(ValueError, match='unavailable'): project_context(engine, 'context')
