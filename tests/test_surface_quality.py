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
    source=tmp_path/'source.nii.gz'; nib.save(nib.Nifti1Image(np.ones((6,6,6),np.float32),np.eye(4)),source)
    empty=tmp_path/'empty.nii.gz';nib.save(nib.Nifti1Image(np.zeros((6,6,6),np.uint8),np.eye(4)),empty)
    c=PatientImagingCollection.create(tmp_path/'native','missing-target');c.add_series(source,series_id='t1',modality='T1');c.add_layer(np.zeros((6,6,6),np.uint8),np.eye(4),layer_id='empty',source_series='t1',kind='tumor',role='target',color='#aabbcc');c.save()
    bridge=BridgeSession(tmp_path/'transfers');request=_Request('empty');progress=lambda *_:None
    opened=bridge.execute('anatomyOpen',{'path':str(c.root/'patient-anatomy.json')},request,progress)
    view=bridge.execute('anatomyView',{'collectionId':opened['collectionId'],'snapshotId':c.identity(),'seriesId':'t1','layers':['empty']},request,progress)
    assert view['compartments'][0]['surface']=={'empty':True}
    assert view['compartments'][0]['qa']['voxelCount']==0
