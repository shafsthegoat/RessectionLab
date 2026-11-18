import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from skimage.measure import marching_cubes
from resectionlab.surface_fairing import fair_surface, intersection_vertices, POLICY
from resectionlab.anatomy_qa import mask_quality


@pytest.mark.parametrize('kind', ['tumor', 'vessel', 'context'])
def test_native_worker_policy_parity(kind):
    node = shutil.which('node')
    if not node: pytest.skip('Node required for cross-stack display parity')
    grid = np.indices((10, 10, 10))
    mask = ((grid-4.5)**2).sum(0) < 12
    points, faces, _, _ = marching_cubes(mask.astype(float), .5)
    affine = np.array([[-.7, .2, 0, 100], [0, 1.3, .2, -20], [0, 0, 2, 40], [0, 0, 0, 1.]])
    expected, diagnostics = fair_surface(points, faces, affine, kind)
    data = {'points': points.tolist(), 'faces': faces.ravel().tolist(), 'affine': affine.tolist(), 'kind': kind}
    code = "import {fairIndexedSurface} from './desktop/src/viewer/surfaceSmoothing.ts';let data='';for await(const c of process.stdin)data+=c;const d=JSON.parse(data);const s=fairIndexedSurface(Float32Array.from(d.points.flat()),Uint32Array.from(d.faces),d.affine,d.kind);process.stdout.write(JSON.stringify({points:[...s.positions],diagnostics:s.diagnostics}));"
    process = subprocess.run([node, '--experimental-strip-types', '--input-type=module', '-e', code], input=json.dumps(data), text=True, capture_output=True, cwd=Path(__file__).resolve().parents[1], check=True)
    actual = json.loads(process.stdout)
    np.testing.assert_allclose(np.array(actual['points']).reshape(-1, 3), expected, atol=2e-6, rtol=0)
    assert actual['diagnostics']['policy'] == POLICY['version']
    assert actual['diagnostics']['restoredComponents'] == diagnostics['restoredComponents']


def test_source_quality_retains_empty_fragmented_and_truncated_evidence():
    mask = np.zeros((8, 8, 8), bool); mask[0, 1, 1] = True; mask[5, 5, 5] = True
    before = mask.copy(); q = mask_quality(mask, 'bound-source', probability=np.zeros_like(mask, float), alignment_status='unreviewed')
    assert q['componentCount'] == 2 and q['boundaryTouchingVoxels'] == 1
    assert q['probabilityMismatchVoxels'] == 2 and q['accuracy'] == 'unmeasured'
    np.testing.assert_array_equal(before, mask)
    empty = mask_quality(np.zeros_like(mask), 'empty-source')
    assert empty['voxelCount'] == empty['componentCount'] == 0 and empty['issues']


def test_physical_intersection_gate_detects_only_new_contacts():
    points = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [.5, .5, -1], [.5, .5, 1], [1.5, .5, 0.]])
    faces = np.arange(6).reshape(2, 3)
    assert not len(intersection_vertices(points, points, faces))
    source = points.copy(); source[3:, 0] += 10
    assert len(intersection_vertices(source, points, faces)) == 6


def test_packaged_policy_copy_matches_canonical_resource():
    desktop = Path(__file__).resolve().parents[1]/'desktop/src/viewer/surface_policy.json'
    assert json.loads(desktop.read_text()) == POLICY


def test_empty_native_mesh_has_explicit_descriptor_without_zero_length_transfer(tmp_path):
    import nibabel as nib
    from resectionlab.patient_anatomy import PatientImagingCollection
    from resectionlab.desktop_bridge import BridgeSession, _Request
    source = tmp_path / 'source.nii.gz'
    empty = tmp_path / 'empty.nii.gz'
    for path, values in ((source, np.ones((6, 6, 6), np.float32)), (empty, np.zeros((6, 6, 6), np.uint8))):
        image = nib.Nifti1Image(values, np.eye(4))
        image.header.set_xyzt_units('mm')
        nib.save(image, path)
    c=PatientImagingCollection.create(tmp_path/'native','missing-target');c.add_series(source,series_id='t1',modality='T1');c.add_layer(np.zeros((6,6,6),np.uint8),np.eye(4),layer_id='empty',source_series='t1',kind='tumor',role='target',color='#aabbcc');c.save()
    bridge=BridgeSession(tmp_path/'transfers');request=_Request('empty');progress=lambda *_:None
    opened=bridge.execute('anatomyOpen',{'path':str(c.root/'patient-anatomy.json')},request,progress)
    view=bridge.execute('anatomyView',{'collectionId':opened['collectionId'],'snapshotId':c.identity(),'seriesId':'t1','layers':['empty']},request,progress)
    assert view['compartments'][0]['surface']=={'empty':True}
    assert view['compartments'][0]['qa']['voxelCount']==0


def test_native_anisotropic_context_moves_across_coarse_slices_with_bounded_topology(monkeypatch):
    from resectionlab.anatomy_geometry import _arrays, _poly
    from resectionlab.surface_fairing import NATIVE_POLICY

    grid = np.indices((33, 35, 15))
    mask = (((grid - np.array([16, 17, 7])[:, None, None, None])
             / np.array([11, 13, 5])[:, None, None, None]) ** 2).sum(0) < 1
    # The surface's opening and disconnected thin fragment exercise the
    # unchanged categorical connectivity and the independent contact audit.
    mask[15:17, 17, 9:13] = False
    mask[2, 4:8, 4] = True
    points, faces = _arrays(_poly(mask.astype(np.uint8), False))
    before = faces.copy()
    affine = np.diag([1.2, 1.2, 4., 1.])
    positions, diagnostics = fair_surface(points, faces, affine, 'brain', native=True)
    displacement = (positions - points) @ affine[:3, :3].T
    assert np.abs(displacement[:, 2]).max() > .65
    assert .65 < diagnostics['maxDisplacementMm'] <= 1.
    assert np.linalg.norm(positions-points, axis=1).max() <= .7
    assert not len(intersection_vertices(points @ affine[:3, :3].T, positions @ affine[:3, :3].T, faces))
    np.testing.assert_array_equal(faces, before)
    original = points[faces] @ affine[:3, :3].T
    display = positions[faces] @ affine[:3, :3].T
    original_normals, display_normals = (np.cross(f[:, 1]-f[:, 0], f[:, 2]-f[:, 0]) for f in (original, display))
    assert np.all(np.einsum('ij,ij->i', original_normals, display_normals) > 0)

    monkeypatch.setitem(NATIVE_POLICY, 'maxContextTriangles', len(faces)-1)
    unchanged, fallback = fair_surface(points, faces, affine, 'brain', native=True)
    np.testing.assert_array_equal(unchanged, points)
    assert fallback['maxDisplacementMm'] == 0
    assert 'audit budget' in fallback['fallbackReason']


def test_native_display_lesion_volume_is_bounded_per_closed_component():
    from resectionlab.anatomy_geometry import _arrays, _poly
    from resectionlab.surface_fairing import NATIVE_POLICY

    grid = np.indices((31, 31, 31))
    solid = ((grid - 15) ** 2).sum(0) < 81
    solid[2, 3:5, 3] = True
    points, faces = _arrays(_poly(solid.astype(np.uint8), False))
    before = faces.copy()
    positions, diagnostics = fair_surface(points, faces, np.eye(4), 'tumor', native=True)
    assert 0 < diagnostics['maxDisplacementMm'] <= 1
    assert diagnostics['maxComponentVolumeChangeFraction'] <= NATIVE_POLICY['lesionComponentVolumeTolerance']
    np.testing.assert_array_equal(faces, before)
    for mask in ((points[:, 0] < 5)[faces[:, 0]], (points[:, 0] > 5)[faces[:, 0]]):
        def six_times_volume(buffer):
            a, b, c = buffer[faces[mask]].transpose(1, 0, 2).astype(float)
            return np.einsum('ij,ij->', a, np.cross(b, c))
        change = abs(six_times_volume(positions) / six_times_volume(points) - 1)
        assert change <= NATIVE_POLICY['lesionComponentVolumeTolerance']
