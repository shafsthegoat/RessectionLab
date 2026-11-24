"""Coverage and branch checks for the native CTA benchmark."""

import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest
from resectionlab.anatomy_display import add_cta_arterial_coverage
from resectionlab.anatomy_evidence import vascular_evidence_supported
from resectionlab.patient_anatomy import PatientImagingCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_annotated_cta_vessels import BASELINE, comparison, native_image, positive_agreement, prediction, publisher_roi
from prepare_annotated_cta_vessel_comparison import check_layer_geometry, within


def test_publisher_region_preserves_native_origin_and_rejects_lattice_changes(tmp_path):
    shape = (16, 15, 14)
    affine = np.diag([-.55, -.6, .75, 1.])
    affine[:3, 3] = (70., 55., -42.)
    original = nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine)
    text = 'Size (Voxels):  7 8 6\nLocation (Voxels):  4 2 3\n'
    region = publisher_roi(text, shape)
    assert original.slicer[region].shape == (7, 8, 6)
    assert np.allclose(original.slicer[region].affine[:3, 3], nib.affines.apply_affine(affine, [4, 2, 3]))
    assert within(np.ones(shape, dtype=bool), region).sum() == 7 * 8 * 6
    with pytest.raises(ValueError, match='exceeds'):
        publisher_roi(text.replace('4 2 3', '12 2 3'), shape)
    shifted = original.affine.copy()
    shifted[0, 3] += .01
    path = tmp_path / 'shifted.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), shifted), path)
    with pytest.raises(ValueError, match='native CTA lattice'):
        native_image(path, original)


def test_outside_roi_path_agreement_counts_only_known_positives_in_mm():
    labels = np.zeros((15, 11, 9), dtype=np.uint8)
    labels[2:12, 5, 4] = 17
    prediction_mask = labels > 0
    prediction_mask[7:12, 5, 4] = False
    prediction_mask[7:12, 6, 4] = True  # One 2 mm voxel away, outside the matching tolerance.
    prediction_mask[10:, 0, 0] = True  # Unlabelled predictions outside coverage are unscored.
    region = (slice(2, 7), slice(3, 8), slice(2, 7))
    agreement = positive_agreement(prediction_mask, labels, np.diag([1., 2., 1., 1.]), region, 1.)
    assert agreement == {'17': {'label_voxels': 5, 'label_voxels_covered': 0,
                                'skeleton_points': 5, 'skeleton_points_supported': 1}}


def test_connected_score_keeps_only_seeded_native_components():
    scores = np.zeros((7, 7, 7), dtype=np.float32)
    scores[0, 0, 0] = .9
    scores[1, 1, 1] = .4
    scores[5:7, 5:7, 5:7] = .6
    expected = np.zeros(scores.shape, dtype=bool)
    expected[0, 0, 0] = expected[1, 1, 1] = True
    assert np.array_equal(prediction(scores, np.zeros_like(scores), {'low': .35, 'seed': .65}), expected)


def test_undeclared_annotation_units_require_exact_acquired_source_lattice(tmp_path):
    frame = np.diag([-.5, -.5, .75, 1.])
    voxels = np.zeros((9, 8, 7), dtype=np.uint8)
    voxels[3:6, 3, 3] = 17
    acquired = nib.Nifti1Image(voxels, frame)
    acquired.header.set_xyzt_units('mm')
    scan, annotation = (tmp_path / 'scan.nii.gz', tmp_path / 'labels.nii.gz')
    nib.save(acquired, scan)
    nib.save(nib.Nifti1Image(voxels, frame), annotation)
    study = PatientImagingCollection.create(tmp_path / 'study', 'fixture')
    study.add_series(scan, series_id='cta', modality='CTA')
    layer = study.add_layer(voxels, frame, layer_id='right-m2', source_series='cta', kind='artery',
                            role='context', color='#65e9ba', label_value=17,
                            provenance={'kind': 'observed_source_annotation', 'source': study.original(annotation)})
    layer['units_status'] = 'unknown'
    check_layer_geometry(study, layer)
    assert np.array_equal(study.mask(layer), voxels == 17)
    layer['affine'][0][3] += .2
    with pytest.raises(ValueError, match='exact acquired CTA lattice'):
        check_layer_geometry(study, layer)


def test_subject_regression_prevents_promotion_despite_mean_gain():
    def result(f1):
        return {'centerline_f1': f1, 'dice': .6, 'complete_branch_segment_recall': .4,
                'small_vessel_recall': .4, 'boundary_hd95_mm': 3.}

    policy = {'max_subject_f1_loss': .01, 'max_subject_dice_loss': .01, 'max_subject_small_vessel_recall_loss': .05,
              'max_subject_hd95_increase_mm': .75, 'min_mean_f1_gain': .02, 'min_mean_branch_recall_gain': .10}
    cases = {'A': {BASELINE: result(.5), 'candidate': result(.7)},
             'B': {BASELINE: result(.6), 'candidate': result(.55)}}
    decision = comparison(cases, 'candidate', policy)
    assert decision['mean_f1_gain'] > .02
    assert decision['eligible'] is False
    assert json.loads(json.dumps(decision)) == decision


def test_cta_coverage_retains_both_sources_and_refuses_unsupplied_positives(tmp_path):
    frame = np.diag([-.5, -.5, .75, 1.])
    shape = (16, 14, 12)
    focused = np.zeros(shape, dtype=np.uint8)
    focused[3:8, 4, 4] = 5
    wider = np.zeros_like(focused)
    wider[6:11, 4, 4] = 17
    wider[6:11, 7, 4] = 19
    wider[6:11, 8, 4] = 35
    scan_file = tmp_path / 'cta.nii.gz'
    image = nib.Nifti1Image(np.zeros(shape, dtype=np.int16), frame)
    image.header.set_xyzt_units('mm')
    nib.save(image, scan_file)
    study = PatientImagingCollection.create(tmp_path / 'study', 'fixture')
    study.add_series(scan_file, series_id='cta', modality='CTA')

    def observed(name, source, stored, label, **provenance):
        path = tmp_path / f'{name}.nii.gz'
        nib.save(nib.Nifti1Image(source, frame), path)
        row = study.add_layer(stored, frame, layer_id=name, source_series='cta', kind='artery', role='context',
                              color='#65e9ba', label_value=label,
                              provenance={'kind': 'supplied_annotation', 'source': study.original(path), **provenance})
        row['units_status'] = 'unknown'
        return row

    core = observed('core', focused, focused, 5)
    wide_mask = np.isin(wider, [17, 19]).astype(np.uint8)
    wide = observed('wider', wider, wide_mask, 1, source_label_values=[17, 19])
    study.document['display'] = {'initial_layers': ['core']}
    original = wide['data']
    wrong = wide_mask.copy()
    wrong[10, 9, 4] = 1
    wide['data'] = study.put_array(wrong)
    with pytest.raises(ValueError, match='differs from its supplied arterial positive labels'):
        add_cta_arterial_coverage(study, 'cta', ['core'], 'wider')
    wide['data'] = original
    union, counts = add_cta_arterial_coverage(study, 'cta', ['core'], 'wider')
    assert np.array_equal(study.mask(union), (focused == 5) | (wide_mask == 1))
    assert np.array_equal(study.mask(core), focused == 5)
    assert np.array_equal(study.mask(wide), wide_mask == 1)
    assert counts['focus_only_voxels'] == 3 and counts['wider_only_voxels'] == 8
    assert study.document['display']['initial_layers'] == [union['id']]
    assert not vascular_evidence_supported(study, union)


def test_workbench_coverage_commit_repeat_correction_and_cold_reopen(tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request

    frame = np.diag([.5, .5, .75, 1.])
    shape = (16, 14, 12)

    def image(name, data):
        path = tmp_path / f'{name}.nii.gz'
        nifti = nib.Nifti1Image(data, frame)
        nifti.header.set_xyzt_units('mm')
        nib.save(nifti, path)
        return path

    study = PatientImagingCollection.create(tmp_path / 'study', 'fixture')
    study.add_series(image('scan', np.zeros(shape, dtype=np.int16)), series_id='cta', modality='CTA')
    masks = {}
    for key, start, stop in [('focused', 3, 8), ('outer', 6, 11), ('distal', 10, 14)]:
        mask = np.zeros(shape, dtype=np.uint8)
        mask[start:stop, 4, 4] = 1
        study.add_layer(mask, frame, layer_id=key, source_series='cta', kind='artery', role='context', color='#af352b',
                        provenance={'kind': 'observed_source_annotation', 'source': study.original(image(key, mask))})
        masks[key] = mask
    original_identity = study.save('ordinary intake')
    manifest = study.root / 'patient-anatomy.json'
    session = BridgeSession(tmp_path / 'worker')

    def call(operation, args):
        return session.execute(operation, args, _Request(operation), lambda *_: None)

    try:
        bound = call('anatomyOpen', {'path': str(manifest)})
        identity = {'collectionId': bound['collectionId'], 'snapshotId': original_identity}
        options = {'seriesId': 'cta', 'focusLayerIds': ['focused'], 'widerLayerIds': ['outer', 'distal']}
        result = call('anatomyCtaArterialCoverage', {**identity, **options})
        assert result['counts']['wider_only_voxels'] == 6
        assert result['document']['snapshot_id'] != original_identity
        assert 'initial_layers' not in result['document']['display']
        combined = result['layerId']
        cold = PatientImagingCollection(study.root)
        np.testing.assert_array_equal(cold.mask(cold.layer(combined)), np.logical_or.reduce(list(masks.values())))
        with pytest.raises(ValueError, match='reload'):
            call('anatomyCtaArterialCoverage', {**identity, **options})
        current = {**identity, 'snapshotId': cold.identity()}
        repeated = call('anatomyCtaArterialCoverage', {**current, **options})
        assert repeated['document'] == result['document']
        with pytest.raises(ValueError, match='input annotation'):
            call('anatomyEdit', {**current, 'layerId': combined, 'action': 'undo'})
        with pytest.raises(ValueError, match='input annotation'):
            call('anatomyGraph', {**current, 'layerId': combined})
        point = [6, 2, 3]
        edit = {**current, 'layerId': 'distal', 'points': [point], 'radiusMm': .3}
        call('anatomyEdit', {**edit, 'action': 'erase'})
        with pytest.raises(ValueError, match='active correction'):
            call('anatomyCtaArterialCoverage', {**current, **options})
        call('anatomyEdit', {**edit, 'action': 'reject'})
        assert PatientImagingCollection(study.root).identity() == current['snapshotId']
        call('anatomyEdit', {**edit, 'action': 'erase'})
        call('anatomyEdit', {**edit, 'action': 'commit', 'author': 'fixture'})
        reopened = call('anatomyOpen', {'path': str(manifest)})
        assert all(row['id'] != combined for row in reopened['document']['layers'])
        option, = reopened['document']['display']['cta_arterial_coverage']
        assert option['retired'] and option['focus_layers'] == ['focused']
        historical = json.loads((study.root / 'revisions' / f'{cold.identity()}.json').read_text())
        assert historical == cold.document
        latest = PatientImagingCollection(study.root)
        assert not latest.mask(latest.layer('distal'))[12, 4, 4]
        np.testing.assert_array_equal(cold.mask(cold.layer('distal')), masks['distal'])
    finally:
        session.close()
