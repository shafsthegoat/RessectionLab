"""Acquired operative texture samples physical orientation and preserves uncovered space."""
import nibabel as nib
import numpy as np
import pytest

from resectionlab.anatomy_operative_intensity import _texture_grid, operative_intensity
from resectionlab.patient_anatomy import PatientImagingCollection, world


def _scan(directory, study, sid, modality, values, affine):
    path = directory / (sid + ".nii.gz")
    image = nib.Nifti1Image(values.astype(np.float32), affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return study.add_series(path, series_id=sid, modality=modality)


def test_texture_uses_target_source_trilinear_native_values_and_zero_outside(tmp_path):
    study = PatientImagingCollection.create(tmp_path / "study", "case")
    native = np.array([[0., -1., 0., 24.],[2., 0., 0., -5.],[0.,0.,1.5,3.],[0.,0.,0.,1.]])
    x,y,z = np.indices((15,18,13))
    _scan(tmp_path,study,"reference","T1",x+10*y+2*z+100,native)
    source = _scan(tmp_path,study,"lesion-source","T2",12*x+y+4*z+1,native)
    study.set_registration("lesion-source", np.eye(4), status="source-provided", method="fixture")
    # Texture axes are permuted relative to acquisition axes, including a reversal.
    basis = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    frame = {"shape":[33,25,22], "spacingMm":.75, "origin":[-12.,-9.,-3.]}
    grid = np.eye(4)
    grid[:3,:3] = basis.T * frame["spacingMm"]
    target = world(native, [3.7,8.1,2.5])
    grid[:3,3] = target + basis.T @ frame["origin"]
    texture = operative_intensity(study,source["id"],grid,frame,check=lambda:None)
    assert texture["sourceSeries"] == "lesion-source"
    assert texture["sourceSha256"] == source["original"]["sha256"]
    brightness = texture["signal"]
    local = np.moveaxis(np.indices(brightness.shape),0,-1)*texture["spacingMm"] + texture["origin"]
    native_points = world(np.linalg.inv(native),target + local @ basis)
    covered = np.all((native_points>=0)&(native_points<=np.array(x.shape)-1),axis=-1)
    assert covered.any() and not covered.all()
    np.testing.assert_array_equal(brightness>0,covered)
    assert brightness.flags.c_contiguous and brightness.dtype==np.uint8
    assert covered.sum() == texture["coveredVoxels"]
    v = native_points @ [12.,1.,4.] + 1
    lo,hi = texture["window"]
    expected = 1 + np.rint(254*np.clip((v[covered]-lo)/(hi-lo),0,1))
    assert np.max(np.abs(expected.astype(int)-brightness[covered].astype(int))) <= 1
    assert np.max(brightness)==255 and np.min(brightness[covered])==1
    assert np.asarray(texture["spacingMm"])*(np.asarray(texture["shape"])-1)==pytest.approx(
        frame["spacingMm"]*(np.asarray(frame["shape"])-1))
    assert texture["sourceVoxelToPatientRasMm"] == native.tolist()


def test_unreviewed_projection_synthetic_and_uncalibrated_sources_cannot_texture_field(tmp_path):
    study = PatientImagingCollection.create(tmp_path / "study", "case")
    x,y,z = np.indices((12,12,12))
    _scan(tmp_path,study,"flat","T1",np.ones_like(x),np.eye(4))
    for sid,mod in (("moving","T1CE"),("MIP","MRI"),("synthetic","T2"),("raw","CT")):
        row = _scan(tmp_path,study,sid,mod,x+y+z,np.eye(4))
        row["synthetic"] = sid=="synthetic"
        study.set_registration(sid,np.eye(4),status="unreviewed" if sid=="moving" else "source-provided",method="fixture")
    frame={"shape":[12,12,12],"spacingMm":1.,"origin":[0.,0.,0.]}
    assert operative_intensity(study,"moving",np.eye(4),frame,check=lambda:None) is None
    study.series("raw")["intensity_units"]="HU"
    ct = operative_intensity(study,"moving",np.eye(4),frame,check=lambda:None)
    assert ct["sourceSeries"]=="raw" and ct["windowUnits"]=="HU"
    def stop():
        raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        operative_intensity(study,"flat",np.eye(4),frame,check=stop)


def test_anisotropic_acquisition_controls_route_texture_pitch_and_endpoints():
    # A thick sagittal acquisition, with a finer in-plane pitch than the 1 mm label grid.
    native = np.diag([2., .5, .5, 1.])
    frame = {"shape":[21,21,41], "spacingMm":1.}
    grid = np.eye(4)
    grid[:3,3] = [11., -3., 9.]
    shape, pitch, texture = _texture_grid(grid, frame, native)
    np.testing.assert_array_equal(shape, [16,61,121])
    np.testing.assert_allclose(pitch, np.array([2., .5, .5])/1.5)

    # Local X and Z now cross both the thick and thin acquisition axes.
    cosine = np.sqrt(.5)
    grid[:3,:3] = [[cosine,0,-cosine],[0,1,0],[cosine,0,cosine]]
    shape, pitch, texture = _texture_grid(grid, frame, native)
    np.testing.assert_array_equal(shape, [55,61,108])
    assert pitch[0] <= 1 / (cosine * (1 / 2 + 1 / .5))
    np.testing.assert_allclose(texture[:3,3], grid[:3,3])
    np.testing.assert_allclose(world(texture, shape-1), world(grid, np.array(frame["shape"])-1))


def test_fine_acquisition_spans_the_entire_field_with_bounded_memory():
    grid = np.diag([.5,.5,.5,1.])
    native = np.diag([.04,.04,.04,1.])
    frame = {"shape":[161,161,399], "spacingMm":.5}
    shape, pitch, texture = _texture_grid(grid, frame, native)
    assert 1_900_000 <= np.prod(shape) <= 2_000_000
    assert shape.max() <= 256
    assert pitch.max() / pitch.min() < 1.02  # isotropic acquisition remains isotropic at the cap
    np.testing.assert_allclose(world(texture, shape-1), world(grid, np.array(frame["shape"])-1))
    np.testing.assert_allclose(pitch*(shape-1), (np.array(frame["shape"])-1)*.5)

    # A narrow, very long field hits the edge limit independently of its total size.
    shape, _, _ = _texture_grid(grid, {"shape":[3,3,399], "spacingMm":.5}, native)
    assert shape.tolist()==[39,39,256]
