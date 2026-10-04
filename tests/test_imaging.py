"""Adversarial import and durable case-revision tests with analytic landmarks."""
from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
import json
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pytest

from resectionlab.core import ContextField, PatientContext, Plan, SourceRef, StalePlanError
from resectionlab.imaging import (
    ImagingError, audit_diffusion, create_synthetic_case, inspect_nifti,
    file_sha256, load_case, load_fractional_annotation_case, load_nifti_case,
    read_case_artifacts, revise_compartment, save_case,
)


def write_image(path, data, affine=None, units="mm", qcode=1, scode=1):
    affine = np.eye(4) if affine is None else affine
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    image.header.set_xyzt_units(units)
    image.set_qform(affine, qcode)
    image.set_sform(affine, scode)
    nib.save(image, path)
    return path


@pytest.fixture
def inputs(tmp_path):
    mri = np.arange(8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10)
    labels = np.zeros(mri.shape, dtype=np.int16)
    labels[2:4, 3:5, 4:6] = 1
    labels[4:6, 3:5, 4:6] = 4
    return write_image(tmp_path / "mri.nii.gz", mri), write_image(tmp_path / "mask.nii.gz", labels)


def test_import_preserves_label_meaning_and_does_not_invent_function(inputs):
    case = load_nifti_case(*inputs)
    assert set(case.compartments) == {"label_1", "label_4"}
    assert case.metadata["benchmark_track"] == "annotation_assisted"
    assert case.metadata["clinical_deficit_probability"] is None
    assert case.metadata["functional_evidence"]["language"] == "unassessed"
    assert case.brain_mask is None
    assert "motor_function_unassessed" in case.unknowns
    assert all(reference.sha256 for reference in case.source_refs)
    with pytest.raises(ImagingError, match="LABEL_CONVENTION_MISMATCH"):
        load_nifti_case(*inputs, label_map={1: "core"})
    with pytest.raises(ImagingError, match="INVALID_COMPARTMENT_NAMES"):
        load_nifti_case(*inputs, label_map={1: "target", 4: "target"})
    release_map = load_nifti_case(*inputs, label_map={1: "core", 2: "flair_abnormality", 4: "enhancing"})
    assert set(release_map.compartments) == {"core", "enhancing"}


def test_valid_qform_only_accepts_zero_inactive_sform(tmp_path):
    affine = np.diag([-1.2, 1.8, 2.4, 1])
    affine[:3, 3] = [22, -14, 9]
    path = write_image(tmp_path / "qform_only.nii", np.indices((8, 9, 10))[0], affine, scode=0)
    image = nib.load(path)
    image.header["srow_x"] = 0
    image.header["srow_y"] = 0
    image.header["srow_z"] = 0
    nib.save(image, path)
    qc = inspect_nifti(path)
    assert qc["qform_code"] == 1 and qc["sform_code"] == 0
    assert qc["orientation"] == ["L", "A", "S"]
    np.testing.assert_allclose(qc["affine_ras_mm"], affine, atol=1e-6)


def test_sform_only_accepted_and_unresolved_frame_rejected(tmp_path):
    path = write_image(tmp_path / "sform.nii", np.ones((5, 6, 7)), qcode=0)
    assert inspect_nifti(path)["sform_code"] == 1
    path = write_image(tmp_path / "neither.nii", np.ones((5, 6, 7)), qcode=0, scode=0)
    with pytest.raises(ImagingError, match="TRANSFORM_UNRESOLVED"):
        inspect_nifti(path)


def test_qform_sform_disagreement_is_never_silently_fixed(tmp_path):
    path = write_image(tmp_path / "conflict.nii", np.ones((6, 6, 6)))
    image = nib.load(path)
    transform = image.affine.copy()
    transform[0, 3] = 10
    image.set_sform(transform, 1)
    nib.save(image, path)
    with pytest.raises(ImagingError, match="QFORM_SFORM_DISAGREEMENT"):
        inspect_nifti(path)


def test_unknown_units_rejected_and_meter_affine_converted(tmp_path):
    unknown = write_image(tmp_path / "unknown.nii", np.ones((5, 6, 7)), units="unknown")
    with pytest.raises(ImagingError, match="UNKNOWN_SPATIAL_UNITS"):
        inspect_nifti(unknown)
    affine = np.diag([0.001, 0.002, 0.003, 1.0])
    affine[:3, 3] = [0.01, -0.02, 0.03]
    path = write_image(tmp_path / "meters.nii", np.indices((5, 6, 7))[0], affine, units="meter")
    case = load_nifti_case(path)
    np.testing.assert_allclose(case.voxel_to_world([2, 3, 4]), [12, -14, 42], atol=1e-5)
    assert case.voxel_volume_mm3 == pytest.approx(6, rel=1e-6)


def test_oblique_anisotropic_landmarks_retain_frame(tmp_path):
    angle = np.deg2rad(25)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([1.2, 1.8, 2.5])
    affine[:3, 3] = [13, -17, 8]
    path = write_image(tmp_path / "oblique.nii", np.indices((8, 9, 10))[0], affine)
    case = load_nifti_case(path)
    point = np.array([3, 5, 7])
    expected = affine[:3, :3] @ point + affine[:3, 3]
    np.testing.assert_allclose(case.voxel_to_world(point), expected, atol=1e-5)
    np.testing.assert_allclose(case.world_to_voxel(expected), point, atol=1e-5)
    assert case.metadata["imaging_qc"]["structural"]["anisotropic"]


@pytest.mark.parametrize("failure", ["shape", "affine", "fractional", "nonfinite", "empty"])
def test_rejects_unsafe_annotations(inputs, tmp_path, failure):
    mri, _ = inputs
    values = np.zeros((8, 9, 10), dtype=np.float32)
    values[2, 3, 4] = 1
    affine = np.eye(4)
    expected = {"shape": "MASK_SHAPE_MISMATCH", "affine": "MASK_AFFINE_MISMATCH",
                "fractional": "INVALID_LABELS", "nonfinite": "NONFINITE_DATA", "empty": "EMPTY_TARGET_ANNOTATION"}
    if failure == "shape":
        values = values[:-1]
    elif failure == "affine":
        affine[0, 0] = -1  # a left-right flip with equal array shape
    elif failure == "fractional":
        values[2, 3, 4] = 0.2
    elif failure == "nonfinite":
        values[2, 3, 4] = np.nan
    else:
        values[:] = 0
    mask = write_image(tmp_path / "badmask.nii", values, affine)
    with pytest.raises(ImagingError, match=expected[failure]):
        load_nifti_case(mri, mask)


def test_brain_mask_is_separate_from_target_and_must_be_binary(inputs, tmp_path):
    brain = np.ones((8, 9, 10))
    path = write_image(tmp_path / "brain.nii", brain)
    case = load_nifti_case(*inputs, brain_mask_path=path)
    assert case.brain_mask.all()
    assert set(case.compartments) == {"label_1", "label_4"}
    brain[0, 0, 0] = 2
    write_image(path, brain)
    with pytest.raises(ImagingError, match="BRAIN_MASK_NOT_BINARY"):
        load_nifti_case(*inputs, brain_mask_path=path)


def test_correction_preserves_originals_and_invalidates_plan():
    original = create_synthetic_case((32, 32, 32))
    corrected_mask = original.compartments["enhancing"].copy()
    corrected_mask[0, 0, 0] = True
    revised = revise_compartment(original, "enhancing", corrected_mask, reason="Reviewer includes one edge voxel.")
    assert revised.revision == original.revision + 1
    assert revised.semantic_hash != original.semantic_hash
    assert not original.compartments["enhancing"][0, 0, 0]
    assert not revised.source_compartments["enhancing"][0, 0, 0]
    assert revised.compartments["enhancing"][0, 0, 0]
    corrected_mask[:] = False
    assert revised.compartments["enhancing"][0, 0, 0]
    with pytest.raises(ValueError):
        revised.source_compartments["enhancing"].setflags(write=True)
    plan = Plan("old", original.semantic_hash, np.array([[0, 0, 0], [1, 1, 1]]), "generic")
    with pytest.raises(StalePlanError):
        plan.assert_current(revised)


def test_portable_bundle_roundtrip_originals_context_and_artifacts(tmp_path):
    source = SourceRef("biopsy", "synthetic://biopsy", provenance="simulated")
    context = PatientContext(datetime(2026, 1, 1, tzinfo=timezone.utc), (
        ContextField("IDH", "mutant", source, available_at=datetime(2026, 1, 2, tzinfo=timezone.utc)),
    ))
    case = replace(create_synthetic_case((24, 25, 26)), context=context)
    case = revise_compartment(case, "enhancing", np.zeros(case.mri.shape, dtype=bool), reason="Test correction")
    path = save_case(case, tmp_path / "portable.rslcase", artifacts={"plans": [], "settings": {"opacity": 0.6}})
    loaded = load_case(path)
    assert loaded.semantic_hash == case.semantic_hash
    assert loaded.context.planner_values() == {}
    assert loaded.source_compartments["enhancing"].any()
    assert not loaded.compartments["enhancing"].any()
    np.testing.assert_array_equal(loaded.mri, case.mri)
    assert read_case_artifacts(path)["settings"]["opacity"] == 0.6
    with ZipFile(path) as archive:
        assert set(archive.namelist()) == {"manifest.json", "arrays.npz"}
        with np.load(BytesIO(archive.read("arrays.npz")), allow_pickle=False) as arrays:
            assert all(not arrays[key].dtype.hasobject for key in arrays.files)


@pytest.mark.parametrize("target", ["arrays", "manifest", "member"])
def test_bundle_tampering_is_reported(tmp_path, target):
    path = save_case(create_synthetic_case((24, 24, 24)), tmp_path / "case.rslcase")
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        payload = archive.read("arrays.npz")
    if target == "arrays":
        payload += b"corruption"
    if target == "manifest":
        manifest["case_id"] = "changed"
    with ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("arrays.npz", payload)
        if target == "member":
            archive.writestr("../../surprise", b"no extraction")
    expected = {"arrays": "BUNDLE_HASH_MISMATCH", "manifest": "CASE_HASH_MISMATCH", "member": "INVALID_BUNDLE_MEMBERS"}
    with pytest.raises(ImagingError, match=expected[target]):
        load_case(path)


def diffusion_inputs(tmp_path):
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    vectors = np.vstack(([0, 0, 0], vectors))
    dwi = write_image(tmp_path / "dwi.nii", np.ones((4, 5, 6, 7)))
    bvals = tmp_path / "bvals"
    bvecs = tmp_path / "bvecs"
    np.savetxt(bvals, [[0] + [1000] * 6])
    np.savetxt(bvecs, vectors.T)
    return dwi, bvals, bvecs


def test_diffusion_audit_distinguishes_numerical_and_spatial_checks(tmp_path):
    paths = diffusion_inputs(tmp_path)
    result = audit_diffusion(*paths)
    assert result["numerical_gradient_checks"] == "passed"
    assert not result["usable_for_reconstruction"]
    assert set(result["issues"]) == {"GRADIENT_FRAME_UNRESOLVED", "DIFFUSION_REGISTRATION_UNREVIEWED",
                                    "GRADIENT_CONVENTION_UNRESOLVED", "PREPROCESSING_UNREVIEWED", "GRADIENT_ROTATION_UNREVIEWED"}
    insufficient = audit_diffusion(*paths, gradient_frame="voxel", registration_reviewed=True)
    assert not insufficient["usable_for_reconstruction"]
    reviewed = audit_diffusion(*paths, gradient_frame="voxel", registration_reviewed=True,
                              gradient_convention="BIDS_FSL", preprocessing_reviewed=True,
                              gradient_rotation_reviewed=True, preprocessing_reference="synthetic://known-direction-test")
    assert reviewed["usable_for_reconstruction"]
    assert not reviewed["tracts_verified"]
    assert reviewed["clinical_deficit_probability"] is None


def test_diffusion_missing_wrong_counts_nonunit_and_degenerate_directions(tmp_path):
    paths = diffusion_inputs(tmp_path)
    assert audit_diffusion(paths[0], None, None)["issues"] == ["MISSING_GRADIENTS"]
    np.savetxt(paths[1], [[0, 1000]])
    assert "GRADIENT_COUNT_MISMATCH" in audit_diffusion(*paths)["issues"]
    paths = diffusion_inputs(tmp_path)
    vectors = np.loadtxt(paths[2])
    vectors[:, 1] *= 0.3
    np.savetxt(paths[2], vectors)
    assert "GRADIENT_NORM_INVALID" in audit_diffusion(*paths)["issues"]
    vectors[:, 1:] = np.array([[1], [0], [0]])
    np.savetxt(paths[2], vectors)
    assert "DEGENERATE_GRADIENT_DIRECTIONS" in audit_diffusion(*paths)["issues"]


def test_scalar_fa_is_not_raw_diffusion(inputs, tmp_path):
    _, bvals, bvecs = diffusion_inputs(tmp_path)
    assert "INVALID_DIMENSIONS" in audit_diffusion(inputs[0], bvals, bvecs)["issues"]


def test_fixture_is_deterministic_and_explicitly_simulated():
    a = create_synthetic_case((24, 24, 24))
    b = create_synthetic_case((24, 24, 24))
    assert a.semantic_hash == b.semantic_hash
    assert a.metadata["is_synthetic"]
    assert a.source_refs[0].provenance == "simulated"
    assert "not_a_patient" in a.unknowns


def test_fractional_annotation_reindex_is_explicit_and_original_preserved(tmp_path):
    shape = (8, 9, 10)
    structural = write_image(tmp_path / "t1.nii", np.indices(shape)[0])
    source = np.zeros(shape)
    source[1, 3, 4] = 0.8
    source[2, 3, 4] = 0.3
    affine = np.eye(4)
    affine[0, 0] = -1
    affine[0, 3] = shape[0] - 1
    annotation = write_image(tmp_path / "fractional.nii", source, affine)
    original_hash = file_sha256(annotation)
    case = load_fractional_annotation_case(structural, annotation, threshold=0.5,
                                           annotation_interpretation="Fractional manually derived source annotation")
    mask = case.compartments["source_target_threshold_scenario"]
    assert mask[6, 3, 4] and not mask[1, 3, 4]
    assert mask.sum() == 1
    assert file_sha256(annotation) == original_hash
    assert case.metadata["fractional_annotation"]["threshold_evidence"] == "declared_research_assumption"
    assert case.source_refs[-1].sha256 == original_hash
    lower = load_fractional_annotation_case(structural, annotation, threshold=0.25,
                                            annotation_interpretation="Fractional manually derived source annotation")
    assert lower.compartments["source_target_threshold_scenario"].sum() == 2
    with pytest.raises(ImagingError, match="INVALID_ANNOTATION_THRESHOLD"):
        load_fractional_annotation_case(structural, annotation, threshold=0,
                                        annotation_interpretation="Source annotation")


def test_fractional_annotation_never_hides_registration_requirement(tmp_path):
    structural = write_image(tmp_path / "t1.nii", np.indices((8, 9, 10))[0])
    affine = np.eye(4)
    affine[0, 3] = 0.4
    annotation = write_image(tmp_path / "offset.nii", np.ones((8, 9, 10)) * 0.8, affine)
    with pytest.raises(ImagingError, match="ANNOTATION_REGISTRATION_REQUIRED"):
        load_fractional_annotation_case(structural, annotation, threshold=0.5,
                                        annotation_interpretation="Source annotation")


def test_small_matrix_scale_drift_cannot_hide_large_native_grid_misalignment(inputs, tmp_path):
    structural, _ = inputs
    affine = np.eye(4)
    affine[0, 0] = 1.009  # Every matrix entry is within the former 0.01 tolerance.
    mask = write_image(tmp_path / "scaled_mask.nii", np.ones((8, 9, 10)), affine)
    with pytest.raises(ImagingError, match="MASK_AFFINE_MISMATCH"):
        load_nifti_case(structural, mask)
    affine[0, 0] = 1.0005  # Matrix residual <0.001, but accumulated grid error exceeds it.
    fractional = write_image(tmp_path / "scaled_fractional.nii", np.ones((8, 9, 10)) * 0.8, affine)
    with pytest.raises(ImagingError, match="ANNOTATION_REGISTRATION_REQUIRED"):
        load_fractional_annotation_case(structural, fractional, threshold=0.5,
                                        annotation_interpretation="Fractional source annotation")


def test_subdegree_qform_rotation_is_checked_across_the_image_extent(tmp_path):
    path = write_image(tmp_path / "qform_rotation.nii", np.ones((8, 9, 10)))
    image = nib.load(path)
    angle = np.deg2rad(0.5)
    rotation = np.eye(4)
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    image.set_qform(rotation, 1)
    nib.save(image, path)
    with pytest.raises(ImagingError, match="QFORM_SFORM_DISAGREEMENT"):
        inspect_nifti(path)
