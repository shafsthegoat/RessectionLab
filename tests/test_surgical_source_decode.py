"""Small generated originals and coordinated, internally consistent substitutions."""
import copy
import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from resectionlab import surgical_source_decode as decode
from resectionlab import surgical_task_admission as admission
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_patient import prepare_surgical_environment


class Cancelled(RuntimeError):
    pass


@pytest.mark.parametrize('shape,reason', [((2**32, 2**32, 4), 'bounded three-dimensional'), ((3, 4, 5), 'imaging shape differs')])
def test_original_header_bounds_and_native_shape_are_checked_before_pixel_allocation(source, monkeypatch, shape, reason):
    header = nib.Nifti2Header(); header.set_data_shape(shape); header.set_data_dtype(np.float32)
    header.set_xyzt_units('mm'); header.set_sform(source.affine, code=1); header['vox_offset'] = 544
    path = source.root/'header-only.nii'
    with path.open('wb') as stream:
        header.write_to(stream); stream.write(b'\0\0\0\0')
    source.collection.series('T1')['original'] = source.collection.original(path)
    def forbidden(*a, **k): pytest.fail('Invalid original header reached pixel allocation')
    monkeypatch.setattr(nib.Nifti2Image, 'get_fdata', forbidden)
    with pytest.raises(ValueError, match=reason): verify(source)


@pytest.fixture(autouse=True)
def isolated_source_cache():
    old = admission._SOURCE_CHECKS.copy()
    admission._SOURCE_CHECKS.clear()
    yield
    admission._SOURCE_CHECKS.clear()
    admission._SOURCE_CHECKS.update(old)


@pytest.fixture
def source(tmp_path, monkeypatch):
    shape = (12, 12, 12)
    affine = np.diag([-1., 1., 1., 1.]); affine[0, 3] = 11.
    pixels = np.indices(shape).sum(axis=0).astype(np.float32)
    labels = np.zeros(shape, np.uint8)
    labels[5:9, 5:9, 5:9] = 4
    labels[2:4, 2:4, 2:4] = 2
    labels[4, 2:4, 2:4] = 1
    def nifti(path, data, *, matrix=affine, units='mm'):
        image = nib.Nifti1Image(data, matrix)
        image.header.set_xyzt_units(units)
        nib.save(image, path)
        return path
    image_path = nifti(tmp_path/'scan.nii.gz', pixels)
    annotation_path = nifti(tmp_path/'labels.nii.gz', labels)
    collection = PatientImagingCollection.create(tmp_path/'native', 'generated-decoding-patient')
    collection.document['study_timepoint'] = 'preoperative'
    collection.add_series(image_path, series_id='T1', modality='T1', timepoint='preoperative')
    collection.series('T1')['source_mirror_revision'] = 'generated-decode-revision'
    brain = np.zeros(shape, np.uint8); brain[1:11, 1:11, 1:11] = 1
    collection.add_layer(brain, affine, layer_id='brain', source_series='T1', kind='brain', role='context', color='#aaaaaa',
                         provenance={'model': 'generated-unreviewed-estimate'})
    collection.add_layer(labels, affine, layer_id='tumor', source_series='T1', kind='tumor', role='target', color='#aa00aa',
        label_value=4, provenance={'kind': 'observed_source_annotation', 'annotation_value': 4,
                                  'source': collection.original(annotation_path)})
    declaration_root = tmp_path/'declarations'; (declaration_root/'generated').mkdir(parents=True)
    (declaration_root/'generated/development-declaration.json').write_text(json.dumps({
        'role':'DEVELOPMENT', 'cases':['generated-decoding-patient'], 'mirror_revision':'generated-decode-revision'}))
    monkeypatch.setattr(admission, '_DECLARATIONS_ROOT', declaration_root)
    counter = 0
    def export():
        nonlocal counter
        counter += 1; collection.save()
        report = prepare_surgical_environment(collection.root, tmp_path/f'export-{counter}',
                                              layer_ids={'brain':'brain', 'tumor':'tumor'})
        return load_environment(report['casePath'])
    def replace_annotation(data, *, units='mm', matrix=affine):
        path = nifti(tmp_path/'replacement-labels.nii.gz', data, matrix=matrix, units=units)
        collection.layer('tumor')['provenance']['source'] = collection.original(path)
    return SimpleNamespace(collection=collection, export=export, nifti=nifti, pixels=pixels,
        labels=labels, affine=affine, root=tmp_path, replace_annotation=replace_annotation)


def verify(source, scene=None, *, check=lambda: None):
    return admission.verify_source(source.export() if scene is None else scene,
                                   native_collection=source.collection.root, check=check)


def old_integrity_accepts_but_decode_refuses(source, monkeypatch, match):
    """Demonstrate the exploit passes the independent hashes/ownership fence."""
    scene = source.export()
    with monkeypatch.context() as patch:
        patch.setattr(decode, 'verify_original_decodes', lambda *args, **kwargs: None)
        receipt = verify(source, scene)
        assert receipt['verifiedOriginalSha256'] and receipt['verifiedArraySha256']
    admission._SOURCE_CHECKS.clear()
    with pytest.raises(ValueError, match=match):
        verify(source, scene)
    assert not admission._SOURCE_CHECKS


def test_authentic_decoded_originals_pass_without_mutating_source(source):
    scene = source.export(); before = copy.deepcopy(source.collection.document)
    receipt = verify(source, scene)
    assert source.collection.document == before
    assert len(receipt['verifiedOriginalSha256']) == 2
    assert len(receipt['verifiedArraySha256']) == 3
    assert len(admission._SOURCE_CHECKS) == 1


def test_registry_qualified_older_source_also_requires_decoding_identity(source, monkeypatch):
    from resectionlab import surgical_source_roles as roles
    c = source.collection; c.series('T1').pop('source_mirror_revision')
    patient = c.document['patient_id']; declared = source.root/'manifests'; declared.mkdir()
    originals = [(c.series('T1')['original'], 'structural_mri', 'observed_preprocessed_imaging'),
                 (c.layer('tumor')['provenance']['source'], 'supplied_tumor_annotation', 'observed_source_annotation')]
    acquired = {'schema_version':'1.0', 'case_id':patient, 'patient_cluster_id':patient,
        'visit':'baseline_preoperative', 'source_collection':{'name':'generated-collection'},
        'split':{'role':'development_demo', 'outer_test_eligible':False},
        'mirror':{'repository':'generated/source', 'revision':'generated-older-revision'},
        'files':[dict(sha256=o['sha256'], source_remote_path=patient+'/'+o['name'], role=role,
                      provenance=provenance, size_bytes=c.path(o['path']).stat().st_size,
                      **({'modality':'T1'} if role=='structural_mri' else {}))
                 for o, role, provenance in originals]}
    registry = {'schema_version':1, 'identity_links':[], 'records':[{
        'record_id':'older-generated', 'collection_id':'generated-collection',
        'identity':'generated-collection:'+patient, 'identity_status':'mirror_attributed',
        'visit_id':'baseline_preoperative', 'derivative_of':None, 'outer_split':'development',
        'uses':['global_development','case_optimization'], 'source_manifest':'manifests/acquired.json'}]}
    (declared/'cohort_registry.json').write_text(json.dumps(registry))
    (declared/'acquired.json').write_text(json.dumps(acquired))
    monkeypatch.setattr(roles, 'PROJECT_ROOT', source.root)
    monkeypatch.setattr(roles, 'REGISTRY_PATH', declared/'cohort_registry.json')
    scene = source.export()
    assert admission.source_evidence(scene)['roleQualification']['recordId'] == 'older-generated'
    assert verify(source, scene)
    data = source.pixels.copy(); data[6, 6, 6] += 1.
    c.series('T1')['data'] = c.put_array(data)
    old_integrity_accepts_but_decode_refuses(source, monkeypatch, 'imaging voxels differ')


@pytest.mark.parametrize('change', ['mri-pixels', 'mri-affine', 'mri-spacing', 'categorical-pixels',
    'categorical-other-class', 'categorical-affine', 'categorical-native-dtype', 'label-semantics'])
def test_coordinated_native_reseal_cannot_substitute_acquired_original(source, monkeypatch, change):
    c = source.collection; image = c.series('T1'); tumor = c.layer('tumor')
    if change == 'mri-pixels':
        data = source.pixels.copy(); data[6, 6, 6] += 1.; image['data'] = c.put_array(data)
    elif change == 'mri-affine':
        matrix = source.affine.copy(); matrix[0, 3] += .25; image['affine'] = matrix.tolist()
    elif change == 'mri-spacing': image['grid_spacing_mm'][0] = 1.01
    elif change.startswith('categorical-'):
        if change == 'categorical-affine':
            matrix = source.affine.copy(); matrix[0, 3] += .25; tumor['affine'] = matrix.tolist()
        else:
            data = source.labels.copy()
            if change == 'categorical-pixels': data[5, 5, 5] = 0
            elif change == 'categorical-other-class': data[2, 2, 2] = 1
            elif change == 'categorical-native-dtype': data = data.astype(np.int32)
            tumor['data'] = c.put_array(data)
    else: tumor['label_value'] = 2
    old_integrity_accepts_but_decode_refuses(source, monkeypatch, 'decode binding')


@pytest.mark.parametrize('marker', ['edits', 'original_data', 'original_label_value'])
def test_edited_supplied_annotation_needs_a_qualified_derivation_even_when_pixels_match(source, monkeypatch, marker):
    row = source.collection.layer('tumor')
    row[marker] = {'edits':[{'author':'generated', 'operations':[]}],
                   'original_data':copy.deepcopy(row['data']), 'original_label_value':4}[marker]
    old_integrity_accepts_but_decode_refuses(source, monkeypatch, 'edited supplied')


@pytest.mark.parametrize('invalid', [4.+2.**-28, -1., 65536., np.nan, np.inf])
def test_original_categorical_values_are_checked_before_integer_or_float32_coercion(source, monkeypatch, invalid):
    data = source.labels.astype(np.float64); data[0, 0, 0] = invalid
    source.replace_annotation(data)
    old_integrity_accepts_but_decode_refuses(source, monkeypatch, 'not exact categorical')


def test_full_uint16_source_classes_are_preserved_not_only_selected_label(source):
    data = source.labels.astype(np.uint16); data[0, 0, 0] = 300
    source.replace_annotation(data)
    source.collection.layer('tumor')['data'] = source.collection.put_array(data)
    assert verify(source)


def test_lossless_uint16_native_storage_of_low_valued_source_classes_passes(source):
    data = source.labels.astype(np.uint16)
    assert data.max() < 256
    source.collection.layer('tumor')['data'] = source.collection.put_array(data)
    assert verify(source)


def test_explicit_source_label_value_cannot_contradict_selected_class(source, monkeypatch):
    row = source.collection.layer('tumor')
    row['provenance']['source_label_value'] = 2
    old_integrity_accepts_but_decode_refuses(source, monkeypatch, 'source label contradicts')


def test_scaled_original_image_uses_decoded_float32_import_not_stored_integer_values(source):
    raw = source.pixels.astype(np.int16)
    path = source.root/'scaled-image.nii.gz'; image = nib.Nifti1Image(raw, source.affine)
    image.header.set_xyzt_units('mm'); image.header.set_slope_inter(2., 10.); nib.save(image, path)
    loaded = nib.load(path); row = source.collection.series('T1')
    row['original'] = source.collection.original(path)
    row['data'] = source.collection.put_array(loaded.get_fdata(dtype=np.float32))
    assert not np.array_equal(raw, source.collection.array(row['data']))
    assert verify(source)


def test_paired_nifti_cannot_read_an_unfingerprinted_image_sidecar(source):
    scene = source.export(); collection = source.collection
    directory = collection.root/'paired'; directory.mkdir()
    path = directory/'scan.hdr'
    image = nib.Nifti1Pair(source.pixels, source.affine)
    image.header.set_xyzt_units('mm'); nib.save(image, path)
    assert path.is_file() and path.with_suffix('.img').is_file()
    assert type(nib.load(path)) is nib.Nifti1Pair
    # The original reference names just the header. Loading it could otherwise
    # consume .img bytes outside the original-member fingerprint/cache key.
    collection.series('T1')['original'] = {'path':'paired/scan.hdr', 'name':'scan.hdr'}
    reads = []
    def forbid_array_read(_):
        reads.append(True)
        pytest.fail('Paired source reached array validation before format refusal')
    collection.array = forbid_array_read
    with pytest.raises(ValueError, match='single-file|NIfTI'):
        decode.verify_original_decodes(collection, scene.source['adapter'], check=lambda: None)
    assert not reads


@pytest.mark.parametrize('units,shift,shape_change,passes', [
    ('unknown', 0., False, True), ('unknown', .25, False, False), ('unknown', 0., True, False),
    ('meter', 0., False, False), ('micron', 0., False, False)])
def test_unknown_annotation_units_inherit_only_exact_verified_parent_grid(source, monkeypatch, units, shift, shape_change, passes):
    data = source.labels[:-1] if shape_change else source.labels
    matrix = source.affine.copy(); matrix[0, 3] += shift
    source.replace_annotation(data, units=units, matrix=matrix)
    row = source.collection.layer('tumor'); row['data'] = source.collection.put_array(data); row['affine'] = matrix.tolist()
    if passes: assert verify(source)
    else: old_integrity_accepts_but_decode_refuses(source, monkeypatch, 'units are not bound')


@pytest.mark.parametrize('units', ['unknown', 'meter', 'micron'])
def test_native_mm_claim_cannot_override_original_image_units(source, monkeypatch, units):
    path = source.nifti(source.root/'wrong-unit-image.nii.gz', source.pixels, units=units)
    source.collection.series('T1')['original'] = source.collection.original(path)
    old_integrity_accepts_but_decode_refuses(source, monkeypatch, 'imaging units differ')


def test_model_estimates_are_not_recast_as_observed_categorical_originals(source):
    row = source.collection.layer('brain')
    row['provenance'].update(source={'path':'originals/not-acquired/model.nii.gz','sha256':'f'*64},
                             inference_receipt='private/no-clinical-truth.json')
    row['edits'] = [{'author':'generated-model-review', 'operations':[]}]
    scene = source.export()
    assert verify(source, scene)
    facts = admission.source_evidence(scene)
    assert facts['anatomy']['brain']['kind'] == 'model-estimate'
    assert facts['anatomy']['brain']['reviewStatus'] == 'unreviewed'


@pytest.mark.parametrize('member', ['image-original', 'annotation-original', 'image-array', 'annotation-array'])
def test_warm_source_proof_does_not_hide_member_corruption(source, member):
    scene = source.export(); verify(source, scene)
    c = source.collection
    reference = {'image-original':c.series('T1')['original'],
        'annotation-original':c.layer('tumor')['provenance']['source'],
        'image-array':c.series('T1')['data'], 'annotation-array':c.layer('tumor')['data']}[member]
    path = c.path(reference['path']); payload = bytearray(path.read_bytes()); payload[-1] ^= 1; path.write_bytes(payload)
    with pytest.raises(ValueError, match='bytes differ|checksum differs'):
        verify(source, scene)


def test_warm_proof_reuses_only_unchanged_members_and_changed_metadata_redecodes(source, monkeypatch):
    scene = source.export(); calls = []
    original = decode.verify_original_decodes
    def capture(*args, **kwargs):
        calls.append(True); return original(*args, **kwargs)
    monkeypatch.setattr(decode, 'verify_original_decodes', capture)
    verify(source, scene); verify(source, scene)
    assert len(calls) == 1
    path = source.collection.path(source.collection.series('T1')['data']['path'])
    path.touch()
    verify(source, scene)
    assert len(calls) == 2


def test_cancellation_during_decode_never_publishes_a_completed_proof(source, monkeypatch):
    scene = source.export(); original = decode.verify_original_decodes; callbacks = 0
    def interrupted(collection, adapter, *, check):
        def cancel_in_decode():
            nonlocal callbacks
            callbacks += 1
            if callbacks == 2: raise Cancelled('cancelled during original decode')
            check()
        return original(collection, adapter, check=cancel_in_decode)
    with monkeypatch.context() as patch:
        patch.setattr(decode, 'verify_original_decodes', interrupted)
        with pytest.raises(Cancelled, match='during original decode'):
            verify(source, scene)
    assert callbacks == 2 and not admission._SOURCE_CHECKS
    assert verify(source, scene)


def test_warm_proof_still_honors_cancellation(source):
    scene = source.export(); receipt = verify(source, scene)
    def cancel(): raise Cancelled('cancelled warm source check')
    with pytest.raises(Cancelled, match='warm source check'):
        admission._verify_source_bytes(scene, receipt, check=cancel)
    assert verify(source, scene)


def test_source_changed_after_decoding_cannot_publish_a_mixed_proof(source, monkeypatch):
    scene = source.export(); original = decode.verify_original_decodes
    def changed(collection, adapter, *, check):
        original(collection, adapter, check=check)
        path = collection.path(collection.series('T1')['original']['path'])
        path.touch()
    monkeypatch.setattr(decode, 'verify_original_decodes', changed)
    with pytest.raises(ValueError, match='changed during byte verification'):
        verify(source, scene)
    assert not admission._SOURCE_CHECKS
