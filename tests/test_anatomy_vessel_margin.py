import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.measure import euler_number
from resectionlab.anatomy_vessels import refine_margin, refine_layer
from resectionlab.anatomy_editor import AnatomyDraft
from resectionlab.patient_anatomy import PatientImagingCollection, PatientAnatomySnapshot
from resectionlab.anatomy_planning import planning_case
from test_patient_anatomy import patient


def test_supported_margin_preserves_small_components_and_withholds_bridges():
    mask = np.zeros((15, 15, 15), bool)
    mask[3:12, 5, 7] = True
    mask[3:12, 8, 7] = True
    mask[1, 1, 1] = True
    p = mask.astype(float)
    p[3:12, 6:8, 7] = .4  # lower-score gap must remain unconnected
    p[5, 4, 7] = .45
    p[13, 13, 13] = .49  # unsupported remote high-score fragment
    result, report = refine_margin(mask, p, np.eye(4), radius_mm=1.1)
    assert report['added_voxels'] > 0
    assert np.all(result[mask]) and result[1, 1, 1]
    assert not result[13, 13, 13]
    assert report['before']['components_26'] == report['after']['components_26'] == 3
    assert euler_number(result, connectivity=3) == euler_number(mask, connectivity=3)
    labels, _ = ndi.label(result, structure=np.ones((3, 3, 3)))
    assert labels[5, 5, 7] != labels[5, 8, 7]
    before_labels, count = ndi.label(mask, structure=np.ones((3, 3, 3)))
    ownership = [np.unique(labels[before_labels == i]).tolist() for i in range(1, count+1)]
    assert all(len(ids) == 1 and ids[0] > 0 for ids in ownership)
    assert len({ids[0] for ids in ownership}) == count


def test_preserves_loop_and_cavity_and_obeys_physical_margin():
    mask = np.zeros((13, 13, 13), bool)
    mask[3:10, 3:10, 3:10] = True
    mask[4:9, 4:9, 4:9] = False
    p = ndi.binary_dilation(mask).astype(float)*.4
    p[mask] = .9
    a = np.diag([.5, .5, 2., 1.])
    result, report = refine_margin(mask, p, a, radius_mm=.8)
    assert report['added_voxels'] > 0
    assert euler_number(mask, connectivity=3) == euler_number(result, connectivity=3)
    assert not np.any(result[:, :, 2]) and not np.any(result[:, :, 10])
    distance = ndi.distance_transform_edt(~mask, sampling=[.5, .5, 2.])
    assert distance[result & ~mask].max() <= .8


def test_corner_background_path_cannot_close_a_tunnel():
    # A full-cube background connectivity test wrongly accepts this center and
    # changes Euler 0->1. N18/face connectivity must withhold the addition.
    occupied = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 2, 0],
                [0, 2, 1], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0],
                [1, 1, 2], [1, 2, 0], [1, 2, 1], [2, 0, 0], [2, 0, 1],
                [2, 0, 2], [2, 2, 0], [2, 2, 1]]
    mask = np.zeros((5, 5, 5), bool)
    mask[tuple((np.array(occupied)+1).T)] = True
    p = mask.astype(float); p[2, 2, 2] = .4
    result, report = refine_margin(mask, p, np.eye(4), radius_mm=1.8)
    assert report['added_voxels'] == 0
    assert np.array_equal(result, mask)


def test_irregular_component_ownership_and_euler_are_preserved():
    rng = np.random.default_rng(23)
    for _ in range(12):
        mask = rng.random((7, 7, 7)) < .2
        p = np.full(mask.shape, .4)
        result, _ = refine_margin(mask, p, np.eye(4), radius_mm=1.8)
        before, count = ndi.label(mask, structure=np.ones((3, 3, 3)))
        after, final_count = ndi.label(result, structure=np.ones((3, 3, 3)))
        pairs = np.unique(np.stack([before[mask], after[mask]], axis=1), axis=0)
        assert len(pairs) == count == final_count == len(np.unique(pairs[:, 1]))
        assert euler_number(mask, connectivity=3) == euler_number(result, connectivity=3)


def test_refinement_reaches_existing_planner_and_roundtrip(patient):
    c = patient
    row = c.layer('vessels')
    mask = c.mask(row)
    p = mask.astype(np.float32)*.9
    p[6:19, 9, 10] = .4
    row['probability'] = c.put_array(p)
    c.save('retain fixture scores')
    source = dict(c.series('mri')['original'])
    original = dict(row['data'])
    before = PatientAnatomySnapshot.capture(c)
    old_case = planning_case(before)
    draft = AnatomyDraft(c, 'vessels')
    draft.apply(c, 'refine-vessels', threshold=.35, radius_mm=.8)
    assert np.count_nonzero(draft.mask) > np.count_nonzero(mask)
    receipt = draft.operations[-1]['anatomy_estimate']
    assert receipt['probability_sha256'] == row['probability']['sha256']
    assert receipt['source_image_sha256'] == source['sha256']
    revised = draft.mask.copy()
    draft.step('undo')
    assert np.array_equal(draft.mask, mask)
    draft.step('redo')
    assert np.array_equal(draft.mask, revised)
    c.commit_edit('vessels', draft.mask, author='test', expected=before.snapshot_id,
                  operations=draft.operations)
    reopened = PatientImagingCollection(c.root)
    after = PatientAnatomySnapshot.capture(reopened)
    new_case = planning_case(after)
    assert old_case.planning_hash != new_case.planning_hash
    assert new_case.critical_evidence['vessels'].mask.sum() > old_case.critical_evidence['vessels'].mask.sum()
    assert reopened.series('mri')['original'] == source
    assert reopened.layer('vessels')['original_data'] == original
    assert reopened.layer('vessels')['review_status'] == 'edited; anatomical review pending'
    assert reopened.layer('vessels')['edits'][-1]['operations'][0]['anatomy_estimate'] == receipt


def test_repeated_preview_and_commit_remain_bound_to_original_support(patient):
    c = patient
    row = c.layer('vessels')
    source = c.mask(row)
    # Many supported consecutive shells: repeated edits must not walk outward.
    row['probability'] = c.put_array(np.full(source.shape, .4, np.float32))
    c.save('fixture source scores')
    first = AnatomyDraft(c, 'vessels')
    first.apply(c, 'refine-vessels', threshold=.35, radius_mm=.51)
    first.apply(c, 'refine-vessels', threshold=.35, radius_mm=.51)
    c.commit_edit('vessels', first.mask, author='test', expected=first.snapshot_id,
                  operations=first.operations)
    second = AnatomyDraft(c, 'vessels')
    second.apply(c, 'refine-vessels', threshold=.35, radius_mm=.51)
    a = c.patient_affine(row)
    distance = ndi.distance_transform_edt(~source, sampling=np.linalg.norm(a[:3, :3], axis=0))
    assert np.any(second.mask & ~source)
    assert distance[second.mask & ~source].max() <= .51
    assert second.operations[-1]['anatomy_estimate']['margin_support_data'] == row['original_data']


def test_cancellation_rejects_preview_without_modifying_source():
    mask = np.zeros((5, 5, 5), bool); mask[2, 2, 2] = True
    with pytest.raises(InterruptedError, match='cancelled'):
        refine_margin(mask, np.full(mask.shape, .4), np.eye(4), radius_mm=1.1,
                      cancelled=lambda: True)
    assert mask.sum() == 1


def test_requires_source_scores_and_finite_exact_geometry(patient):
    with pytest.raises(ValueError, match='retained source scores'):
        refine_layer(patient, patient.layer('vessels'))
    mask = np.zeros((3, 3, 3), bool)
    for p in [np.zeros((3, 3, 2)), np.full(mask.shape, np.nan)]:
        with pytest.raises(ValueError, match='exact mask grid'):
            refine_margin(mask, p, np.eye(4))
    with pytest.raises(ValueError, match='orthogonal'):
        a = np.eye(4); a[0, 1] = .2
        refine_margin(mask, mask.astype(float), a)


def test_changed_acquired_source_rejects_refinement(patient):
    row = patient.layer('vessels')
    row['probability'] = patient.put_array(patient.mask(row).astype(np.float32))
    source = patient.path(patient.series('mri')['original']['path'])
    source.write_bytes(source.read_bytes()+b'changed-source')
    with pytest.raises(ValueError, match='source image changed'):
        refine_layer(patient, row)


def test_bridge_preview_commit_and_portable_reopen_preserve_estimate(patient, tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request
    from resectionlab.anatomy_package import save_package, open_package
    c = patient
    row = c.layer('vessels')
    p = c.mask(row).astype(np.float32)*.9; p[6:19, 9, 10] = .4
    row['probability'] = c.put_array(p)
    c.save('source scores')
    before = c.identity()
    bridge = BridgeSession(tmp_path/'transfer')
    def call(op, args):
        return bridge.execute(op, args, _Request('margin-test'), lambda *x: None)
    opened = call('anatomyOpen', {'path': str(c.root/'patient-anatomy.json')})
    args = {'collectionId': opened['collectionId'], 'snapshotId': before,
            'layerId': 'vessels'}
    preview = call('anatomyEdit', {**args, 'action': 'refine-vessels',
                                  'threshold': .35, 'radiusMm': .8})
    assert preview['draft']['changedVoxels'] > 0
    assert PatientImagingCollection(c.root).identity() == before
    committed = call('anatomyEdit', {**args, 'action': 'commit', 'author': 'test reviewer'})
    assert committed['draft'] is None
    assert committed['document']['snapshot_id'] != before
    with pytest.raises(ValueError, match='changed'):
        call('anatomyGraph', args)
    revised = PatientImagingCollection(c.root)
    package = tmp_path/'refined.rsworkspace'
    save_package(revised, package)
    reopened = PatientImagingCollection(open_package(package, tmp_path/'reopened').parent)
    assert reopened.document == revised.document
    assert np.array_equal(reopened.mask(reopened.layer('vessels')), revised.mask(revised.layer('vessels')))
    assert reopened.layer('vessels')['edits'][0]['operations'][0]['anatomy_estimate']['version'] == 'source-score-simple-margin-v1'
