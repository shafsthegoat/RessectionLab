from types import SimpleNamespace

import pytest

from resectionlab import surgical_strategy_comparison as comparison


@pytest.mark.parametrize('difference', ['initial', 'caseHash', 'sceneId', 'instrumentCatalogHash',
                                        'physicsHash', 'targetBinding', 'identical actions'])
def test_incompatible_recordings_are_not_ranked(monkeypatch, difference):
    shared = {name: name for name in ('patientId', 'sourceSnapshotId', 'targetLayerId', 'targetBinding',
                                      'caseHash', 'sceneId', 'instrumentCatalogHash', 'physicsHash')}

    def read_arm(collection, target_id, path, request):
        receipt = dict(shared, actionsHash=path)
        arm = {'baselineStateId': 'intact'}
        if path == 'second':
            if difference == 'initial':
                arm['baselineStateId'] = 'already operated'
            elif difference == 'identical actions':
                receipt['actionsHash'] = 'first'
            else:
                receipt[difference] = 'different'
        return arm, receipt, {}

    monkeypatch.setattr(comparison, '_read_arm', read_arm)
    with pytest.raises(ValueError, match='same study|same recorded action'):
        comparison.compare_recordings(None, 'target', ['first', 'second'],
                                      SimpleNamespace(check=lambda: None), lambda *args: None)


def test_endpoint_sections_use_same_source_slice_and_distinct_verified_removals():
    import base64
    from io import BytesIO

    import numpy as np
    from PIL import Image

    from resectionlab.surgical_comparison_sections import comparison_sections

    shape = (41, 51, 35)
    affine = np.array([[0., -1., 0., 75.], [1., 0., 0., -30.],
                       [0., 0., 1.5, 15.], [0., 0., 0., 1.]])
    reference = {'id': 'reference', 'modality': 'T1', 'original': {'sha256': 'acquired'}, 'data': {'shape': shape}}
    scan = np.indices(shape).sum(axis=0).astype(np.float32)
    document = {'reference_series': reference['id'], 'transforms': {'reference': {'status': 'reference'}}}
    collection = SimpleNamespace(document=document, series=lambda _: reference,
                                 patient_affine=lambda _: affine, array=lambda _: scan, identity=lambda: 'snapshot')

    def ras(ijk):
        return (affine[:3, :3] @ ijk + affine[:3, 3]).tolist()

    entries = [(10, 10, 10), (20, 30, 10)]
    targets = [(12, 20, 10), (22, 25, 15)]
    arms = [dict(entry=ras(e), target=ras(t)) for e, t in zip(entries, targets)]
    grid = dict(shape=shape, affine=affine, original=np.flatnonzero(scan.ravel() == 60))
    grids = [dict(grid, removed=[np.ravel_multi_index(e, shape)]) for e in entries]
    result = comparison_sections(collection, arms, grids, lambda: None)
    assert result['sourceSha256'] == 'acquired'
    for selected, plane in enumerate(result['planes']):
        offset = np.array([arms[0]['entry'], arms[1]['entry'], arms[selected]['target']]) - plane['center']
        np.testing.assert_allclose(offset @ np.cross(plane['u'], plane['v']), 0., atol=1e-10)
        images = []
        for arm, view in zip(arms, plane['arms']):
            rgb = np.asarray(Image.open(BytesIO(base64.b64decode(view['png']))))
            amber = np.all(rgb == [249, 173, 80], axis=-1)
            assert amber.sum() == view['removedPixels'] > 0
            delta = np.array(arm['entry']) - plane['center']
            xy = (np.array([plane['u'], plane['v']]) @ delta / plane['spacingMm'] + (plane['size'] - 1) / 2)
            x, y = np.rint(xy).astype(int)
            assert amber[y, x]
            images.append((rgb, amber))
        (a, a_removed), (b, b_removed) = images
        assert not np.any(a_removed & b_removed)
        assert np.array_equal(a[~(a_removed | b_removed)], b[~(a_removed | b_removed)])
        y, x = np.where((a[..., 0] > 0) & (a[..., 0] < 255) & (a[..., 0] == a[..., 2]))
        xy = np.stack((x, y), axis=1) - (plane['size'] - 1) / 2
        ras_points = plane['center'] + plane['spacingMm'] * xy @ np.array([plane['u'], plane['v']])
        voxel = (ras_points - affine[:3, 3]) @ np.linalg.inv(affine[:3, :3]).T
        sample = scan.ravel()[::max(1, scan.size // 20000)]
        low, high = np.percentile(sample[sample != 0], [.5, 99.5])
        expected = np.rint(np.clip((voxel.sum(axis=1) - low) / (high - low), 0, 1) * 255)
        np.testing.assert_allclose(a[y, x, 0], expected, atol=1)
    first, second = result['planes']
    for axis in ('u', 'v'):
        assert np.dot(first[axis], second[axis]) >= 0


def test_shared_entry_and_collinear_routes_have_a_valid_section_plane():
    import numpy as np
    from resectionlab.surgical_comparison_sections import _section_normal

    entries = np.zeros((2, 3))
    for targets in (np.array([[3., 1., 1.], [3., 1., 1.]]), np.array([[2., 1., 0.], [-1., 0., 1.]])):
        normal = _section_normal(entries, targets, 0)
        np.testing.assert_allclose(targets @ normal, 0., atol=1e-10)
        assert np.linalg.norm(normal) == pytest.approx(1.)


def test_unsupported_reference_preserves_metrics_but_validation_and_cancellation_propagate():
    from resectionlab.surgical_comparison_sections import comparison_sections

    row = {'id': 'ref', 'data': {'shape': [16, 16, 16]}, 'original': {'sha256': 'bound'}, 'synthetic': True}
    document = {'reference_series': 'ref', 'transforms': {'ref': {'status': 'reference'}}}
    def invalid_geometry(_):
        raise ValueError('Original geometry changed')
    def cancel():
        raise InterruptedError('Cancelled')
    collection = SimpleNamespace(document=document, series=lambda _: row, patient_affine=invalid_geometry)
    assert 'acquired reference' in comparison_sections(collection, [], [], lambda: None)['unavailable']
    with pytest.raises(InterruptedError):
        comparison_sections(collection, [], [], cancel)
    row['synthetic'] = False
    document['transforms']['ref']['status'] = 'unreviewed'
    assert 'placement' in comparison_sections(collection, [], [], lambda: None)['unavailable']
    document['transforms']['ref']['status'] = 'reference'
    with pytest.raises(ValueError, match='Original geometry changed'):
        comparison_sections(collection, [], [], lambda: None)
