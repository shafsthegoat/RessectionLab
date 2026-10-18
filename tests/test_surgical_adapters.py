import hashlib
import nibabel as nib
import numpy as np
import pytest
from resectionlab.surgical_adapters import from_nifti, from_mesh, from_native_collection


def test_arbitrary_labels_rotated_anisotropic_and_immutable(tmp_path):
    raw = np.zeros((3, 4, 5), np.int32)
    raw[1, 1, 2] = 123456
    raw[2, 2, 3] = 7
    affine = np.array([[0, -2, 0, 12], [3, 0, 0, -5], [0, 0, 4, 8], [0, 0, 0, 1.]])
    im = nib.Nifti1Image(raw, affine)
    im.header.set_xyzt_units('mm')
    path = tmp_path / 'arbitrary.nii.gz'
    nib.save(im, path)
    before = path.read_bytes()
    scene = from_nifti(path, {123456: 'bone', 7: 'tumor'}, frame='RAS+')
    np.testing.assert_array_equal(scene.affine, affine)
    np.testing.assert_array_equal(scene.source_arrays['labels'], raw)
    assert scene.labels[1, 1, 2] == 1
    assert scene.labels[2, 2, 3] == 2
    assert scene.voxel_mm3 == pytest.approx(24)
    assert path.read_bytes() == before
    assert scene.source['sha256'] == hashlib.sha256(before).hexdigest()
    assert not scene.source_arrays['labels'].flags.writeable
    with pytest.raises(ValueError, match='Mapping'):
        from_nifti(path, {7: 'bone'}, frame='RAS+')


def test_units_and_frame_required(tmp_path):
    path = tmp_path / 'unknown.nii'
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.int16), np.eye(4)), path)
    with pytest.raises(ValueError, match='units'):
        from_nifti(path, {1: 'bone'}, frame='RAS+')
    with pytest.raises(ValueError, match='frame'):
        from_nifti(path, {1: 'bone'}, frame=None, units='mm')
    scene = from_nifti(path, {1: 'bone'}, frame='LPS+', units='mm')
    np.testing.assert_array_equal(scene.affine, np.diag([-1, -1, 1, 1]))


CUBE = '''v 0 0 0
v 2 0 0
v 2 2 0
v 0 2 0
v 0 0 2
v 2 0 2
v 2 2 2
v 0 2 2
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
'''


def test_cube_voxelization(tmp_path):
    path = tmp_path / 'cube.obj'
    path.write_text(CUBE)
    scene = from_mesh(path, 'bone', resolution=.5, units='mm', frame='RAS+')
    assert scene.shape == (4, 4, 4)
    assert np.count_nonzero(scene.labels) == 64
    assert scene.voxel_mm3 * np.count_nonzero(scene.labels) == pytest.approx(8)
    assert path.read_text() == CUBE
    assert scene.source['topology'] == 'solid'
    assert len(scene.source['faces']) == 12


def test_open_sheet_and_strand_are_explicit(tmp_path):
    path = tmp_path / 'sheet.obj'
    path.write_text('v 0 0 0\nv 2 0 0\nv 0 2 0\nf 1 2 3\n')
    with pytest.raises(ValueError, match='closed manifold'):
        from_mesh(path, 'membrane', resolution=.5, units='mm', frame='RAS+')
    scene = from_mesh(path, 'membrane', resolution=.5, units='mm', frame='RAS+', topology='sheet', thickness_mm=.5)
    assert np.count_nonzero(scene.labels) > 0
    assert len(scene.source['faces']) == 1
    path.write_text('v 0 0 0\nv 2 0 0\nl 1 2\n')
    scene = from_mesh(path, 'vessel', resolution=.5, units='mm', frame='RAS+', topology='strand', thickness_mm=.5)
    assert np.count_nonzero(scene.labels) > 0


def test_native_registration_and_preservation(tmp_path):
    from resectionlab.patient_anatomy import PatientImagingCollection
    p = tmp_path / 'source.nii'
    im = nib.Nifti1Image(np.zeros((4, 4, 4), np.int16), np.diag([2., 3., 4., 1.]))
    im.header.set_xyzt_units('mm')
    nib.save(im, p)
    c = PatientImagingCollection.create(tmp_path / 'collection', 'arbitrary')
    c.add_series(p, series_id='scan', modality='T1', timepoint='preoperative')
    mask = np.zeros((4, 4, 4), np.uint8)
    mask[1, 1, 1] = 9
    c.add_layer(mask, im.affine, layer_id='annotation', source_series='scan', kind='bone', role='context', color='#aaaaaa', label_value=9)
    identity = c.identity()
    scene = from_native_collection(c, frame='RAS+')
    assert scene.labels[1, 1, 1] == 1
    np.testing.assert_array_equal(scene.source_arrays['annotation'], mask)
    assert c.identity() == identity
    c.document['transforms']['scan']['status'] = 'unreviewed'
    with pytest.raises(ValueError, match='Registration'):
        from_native_collection(c, frame='RAS+')


def test_binary_stl_and_nonmanifold_rejection(tmp_path):
    import struct
    from resectionlab.surgical_adapters import _mesh
    obj = tmp_path / 'cube.obj'
    obj.write_text(CUBE)
    vertices, faces, _ = _mesh(obj)
    path = tmp_path / 'cube.stl'
    data = b' ' * 80 + struct.pack('<I', len(faces))
    for face in faces:
        data += struct.pack('<12fH', 0., 0., 0., *vertices[face].ravel(), 0)
    path.write_bytes(data)
    scene = from_mesh(path, 'bone', resolution=1., units='mm', frame='RAS+')
    assert np.count_nonzero(scene.labels) == 8
    assert path.read_bytes() == data
    obj.write_text(CUBE + 'f 1 3 2\n')
    with pytest.raises(ValueError, match='closed manifold'):
        from_mesh(obj, 'bone', resolution=1., units='mm', frame='RAS+')


def test_surface_buffers_match_file_adapter_and_preserve_caller_buffers(tmp_path):
    from resectionlab.surgical_adapters import from_surface_buffers, _mesh
    path = tmp_path / 'cube.obj'
    path.write_text(CUBE)
    vertices, faces, _ = _mesh(path)
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)
    before_v, before_f = vertices.copy(), faces.copy()
    mapped = {'id': 4, 'name': 'arbitrary', 'material': 'bone', 'target': False}
    scene = from_surface_buffers(vertices, faces, mapped, resolution=.5, frame='LPS+', units='mm')
    file_scene = from_mesh(path, mapped, resolution=.5, frame='LPS+', units='mm')
    np.testing.assert_array_equal(scene.labels, file_scene.labels)
    np.testing.assert_array_equal(scene.affine, file_scene.affine)
    np.testing.assert_array_equal(vertices, before_v)
    np.testing.assert_array_equal(faces, before_f)
    assert scene.source_arrays['vertices'].dtype == np.float32
    assert scene.source_arrays['faces'].dtype == np.int32
    assert not scene.source_arrays['vertices'].flags.writeable
    assert not scene.source_arrays['faces'].flags.writeable
    vertices[:] = 99
    faces[:] = 0
    np.testing.assert_array_equal(scene.source_arrays['vertices'], before_v)
    np.testing.assert_array_equal(scene.source_arrays['faces'], before_f)


def test_surface_buffer_validation_and_declared_sheet():
    from resectionlab.surgical_adapters import from_surface_buffers
    vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    faces = np.array([[0, 1, 2]])
    options = dict(resolution=.5, frame='RAS+', units='mm')
    with pytest.raises(ValueError, match='closed manifold'):
        from_surface_buffers(vertices, faces, 'membrane', **options)
    sheet = from_surface_buffers(vertices, faces, 'membrane', topology='sheet', thickness_mm=.5, **options)
    assert np.count_nonzero(sheet.labels)
    with pytest.raises(ValueError, match='integer index'):
        from_surface_buffers(vertices, faces.astype(float), 'bone', **options)
    with pytest.raises(ValueError, match='outside vertices'):
        from_surface_buffers(vertices, np.array([[0, 1, 3]]), 'bone', **options)
    with pytest.raises(ValueError, match='units'):
        from_surface_buffers(vertices, faces, 'bone', **{**options, 'units': None})
