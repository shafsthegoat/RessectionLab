"""Explicit native-kind compatibility; generated sources, no patient claims."""
import copy
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from resectionlab import surgical_anatomical_encounter as encounter
from resectionlab.core import array_digest, semantic_digest
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_patient import prepare_surgical_environment
from resectionlab.surgical_task_admission import admission, prepare_admission, bind_admission
from resectionlab.surgical_two_tool import generate_comparison, validate_report
from test_surgical_anatomical_encounter import context_case, qualified, prepared

PATH = Path(__file__).resolve().parents[1]/'scripts/verify_anatomical_encounter.py'
spec = importlib.util.spec_from_file_location('context_compatibility_independent_audit', PATH)
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


@pytest.fixture(scope='module')
def rebuild(context_case, tmp_path_factory):
    root = tmp_path_factory.mktemp('generated-context-compatibility'); count = 0
    original = context_case[1]; original_native = context_case[4]
    def build(mutate=lambda collection, row: None, *, allow_unreviewed=True):
        nonlocal count
        count += 1; folder = root/str(count)
        shutil.copytree(original_native.root, folder/'native')
        collection = PatientImagingCollection(folder/'native'); row = collection.layer('context')
        row['kind'] = 'lesion-context'; row['provenance']['annotation_value'] = row['label_value']
        mutate(collection, row); collection.save()
        exported = prepare_surgical_environment(collection.root, folder/'scene', layer_ids={'brain':'brain', 'tumor':'target'})
        engine = SurgicalEngine(load_environment(exported['casePath']))
        for action in original.actions:
            assert engine.step(copy.deepcopy(action))['status'] != 'rejected'
        task = copy.deepcopy(admission(original)['task']); task['allowUnreviewedEvidence'] = allow_unreviewed
        value = prepare_admission(engine, native_collection=collection.root, task=task)
        bind_admission(engine, value)
        return engine, collection
    return build


@pytest.fixture(scope='module')
def lesion_case(rebuild):
    return rebuild()


def report_for(engine, observation, binding):
    return {'taskAdmission':admission(engine), 'contextBinding':binding,
            'publicPlan':{'publicObservation':{'anatomicalContext':observation.manifest()}}}


def expected_legacy_binding(engine, native, observation):
    row = native.layer('context'); contract = admission(engine)
    public = {'schema':'resectionlab.source-context-encounter.v1',
        'method':'unique-retained-source-cells-in-complete-nominal-axial-tool-envelope-v1',
        'maskHash':array_digest(observation.mask), 'affineHash':array_digest(observation.affine),
        'shape':list(observation.mask.shape), 'role':'supplied-categorical-context', 'reviewStatus':'unreviewed',
        'sampling':'exact-identity-source-cell-membership', 'coverage':'source-grid-only; outside-unassessed',
        'scope':('Whole-cell geometric near-encounter within a declared research margin; '
                 'not tissue removal, mechanical injury, function or clinical deficit. '
                 'Only the supplied source image grid is assessed; outside anatomy remains unknown.'),
        'evaluationReferencesExposed':False}
    value = {'schema':'resectionlab.source-context-encounter.v1', 'sourceCollectionHash':native.identity(),
        'nativeManifestSha256':contract['verification']['nativeManifestSha256'], 'layerId':'context',
        'labelValue':row['label_value'], 'revision':row.get('revision',0), 'sourceLabel':row.get('label'),
        'sourceArray':row['data'], 'original':row['provenance']['source'],
        'registeredAffine':observation.affine.tolist(), 'registration':native.document['transforms'],
        'reviewStatus':row['review_status'], 'coverage':row.get('coverage','unknown'), 'provenance':row['provenance'],
        'verifiedSiblingLayerId':'target', 'publicObservation':public}
    value['bindingSeal'] = semantic_digest(value)
    return value


def test_legacy_private_binding_and_public_manifest_are_exactly_preserved(context_case):
    engine, native = context_case[1], context_case[4]
    observation, binding = encounter.project_context(engine, 'context')
    expected = expected_legacy_binding(engine, native, observation)
    assert binding == expected and 'eligibility' not in binding
    assert json.dumps(binding,sort_keys=True,separators=(',',':')) == json.dumps(expected,sort_keys=True,separators=(',',':'))
    mask, affine = audit.load_source_context(engine.clone(), report_for(engine, observation, binding))
    assert np.array_equal(mask, observation.mask) and np.array_equal(affine, observation.affine)


def test_observed_lesion_context_has_explicit_private_proof_and_unchanged_public_geometry(context_case, lesion_case):
    legacy = context_case[1]; engine, _ = lesion_case
    old, old_binding = encounter.project_context(legacy, 'context')
    observation, binding = encounter.project_context(engine, 'context')
    assert binding['schema'] == 'resectionlab.source-context-binding.v2'
    assert binding['eligibility'] == {'schema':'resectionlab.native-context-eligibility.v1',
        'rule':'observed-lesion-context-sharing-verified-categorical-source-v1',
        'nativeKind':'lesion-context', 'nativeRole':'context',
        'provenanceKind':'observed_source_annotation', 'sourceSeries':'T1'}
    assert binding['bindingSeal'] == semantic_digest({k:v for k,v in binding.items() if k!='bindingSeal'})
    assert observation.manifest() == old.manifest() == old_binding['publicObservation']
    assert np.array_equal(observation.mask, old.mask)
    assert encounter.EncounterTracker(observation, engine, {'marginMm':3.,'maxNearMm3':1000.}).receipt(engine) == \
           encounter.EncounterTracker(old, legacy, {'marginMm':3.,'maxNearMm3':1000.}).receipt(legacy)
    public = json.dumps(observation.manifest())
    assert all(key not in public for key in ('lesion-context','eligibility','layerId','sourceSeries','sourceCollectionHash'))
    assert [r['layerId'] for r in encounter.context_options(engine)] == ['context']
    mask, affine = audit.load_source_context(engine.clone(), report_for(engine, observation, binding))
    assert np.array_equal(mask, observation.mask) and np.array_equal(affine, observation.affine)


@pytest.mark.parametrize('change', ['unknown-kind','critical-role','model-provenance','annotation-label',
    'source-label','bool-label-declaration','edits','original-data','empty-original-data','original-label',
    'source-array','source-original','source-affine','source-series','unknown-units','explicit-bad-units'])
def test_unqualified_native_context_metadata_does_not_get_compatibility(rebuild, change):
    def mutate(collection, row):
        if change=='unknown-kind': row['kind']='functional-context'
        elif change=='critical-role': row['role']='critical'
        elif change=='model-provenance': row['provenance']['kind']='model_estimate'
        elif change=='annotation-label': row['provenance']['annotation_value']=1
        elif change=='source-label': row['provenance']['source_label_value']=1
        elif change=='bool-label-declaration': row['provenance']['annotation_value']=True
        elif change=='edits': row['edits']=[{'author':'generated','operations':[]}]
        elif change=='original-data': row['original_data']=copy.deepcopy(row['data'])
        elif change=='empty-original-data': row['original_data']=None
        elif change=='original-label': row['original_label_value']=2
        elif change=='source-array': row['data']=copy.deepcopy(collection.layer('brain')['data'])
        elif change=='source-original': row['provenance']['source']=copy.deepcopy(collection.series('T1')['original'])
        elif change=='source-affine': row['affine'][0][3]+=.25
        elif change=='source-series':
            series=collection.series('T1'); added=collection.add_series(collection.path(series['original']['path']),
                series_id='T2', modality='T2', timepoint='preoperative')
            added['source_mirror_revision']=series['source_mirror_revision']; row['source_series']='T2'
        elif change=='unknown-units': row['units_status']='unknown'
        else: row['units_status']='meter'
    engine, _ = rebuild(mutate)
    assert encounter.context_options(engine) == []
    with pytest.raises(ValueError,match='unavailable or not bound'):
        encounter.project_context(engine,'context')


@pytest.mark.parametrize('label', [0, True, 3, 4])
def test_compatibility_cannot_create_a_missing_or_target_overlapping_class(rebuild, label):
    def mutate(_, row):
        row['label_value']=label; row['provenance']['annotation_value']=label
    engine, _ = rebuild(mutate)
    with pytest.raises(ValueError): encounter.project_context(engine,'context')


def test_unreviewed_context_still_requires_explicit_research_acceptance(rebuild):
    def selected_reviewed(collection, _):
        collection.layer('brain')['review_status']='reviewed'; collection.layer('target')['review_status']='reviewed'
    engine, _ = rebuild(selected_reviewed,allow_unreviewed=False)
    with pytest.raises(ValueError,match='unreviewed context'):
        encounter.project_context(engine,'context')


@pytest.mark.parametrize('change', ['downgrade','unknown-version','missing-proof','proof-schema','rule',
                                  'native-kind','native-role','source-series','extra-field'])
def test_independent_auditor_rejects_resealed_eligibility_forgeries(lesion_case, change):
    engine, _ = lesion_case; observation,binding=encounter.project_context(engine,'context')
    if change=='downgrade': binding['schema']='resectionlab.source-context-encounter.v1'; binding.pop('eligibility')
    elif change=='unknown-version': binding['schema']='resectionlab.source-context-binding.v99'
    elif change=='missing-proof': binding.pop('eligibility')
    elif change=='proof-schema': binding['eligibility']['schema']='resectionlab.native-context-eligibility.v99'
    elif change=='rule': binding['eligibility']['rule']='trust-me'
    elif change=='native-kind': binding['eligibility']['nativeKind']='context'
    elif change=='native-role': binding['eligibility']['nativeRole']='critical'
    elif change=='source-series': binding['eligibility']['sourceSeries']='different-series'
    else: binding['eligibility']['approved']=True
    binding['bindingSeal']=semantic_digest({k:v for k,v in binding.items() if k!='bindingSeal'})
    with pytest.raises(ValueError): audit.load_source_context(engine.clone(),report_for(engine,observation,binding))


def test_legacy_native_kind_cannot_claim_new_binding_even_when_resealed(context_case):
    engine=context_case[1]; observation,binding=encounter.project_context(engine,'context')
    binding.update(schema='resectionlab.source-context-binding.v2',eligibility={})
    binding['bindingSeal']=semantic_digest({k:v for k,v in binding.items() if k!='bindingSeal'})
    with pytest.raises(ValueError,match='unsupported source context binding or native kind'):
        audit.load_source_context(engine.clone(),report_for(engine,observation,binding))


def test_admitted_comparison_binds_compatibility_without_public_schema_changes(lesion_case, context_case):
    engine,_=lesion_case; report=generate_comparison(engine,context_case[3])
    assert report['schema']=='resectionlab.two-tool-comparison.v2'
    assert report['contextBinding']['schema']=='resectionlab.source-context-binding.v2'
    assert validate_report(engine,report)
    forged=copy.deepcopy(report); forged['contextBinding']['eligibility']['rule']='invented'
    forged['contextBinding']['bindingSeal']=semantic_digest({k:v for k,v in forged['contextBinding'].items() if k!='bindingSeal'})
    forged['reportSeal']=semantic_digest({k:v for k,v in forged.items() if k!='reportSeal'})
    with pytest.raises(ValueError,match='source context binding differs'): validate_report(engine,forged)
