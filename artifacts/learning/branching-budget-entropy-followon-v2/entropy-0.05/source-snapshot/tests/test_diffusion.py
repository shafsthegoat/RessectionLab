from __future__ import annotations

import numpy as np
import pytest

from resectionlab.diffusion import (
    DiffusionData, DiffusionError, fit_tensor, gradients_in_image_axes,
    image_axis_basis, load_diffusion, planning_gates, probabilistic_crop_audit,
    validate_gradients,
)


def synthetic_tensor(*, affine=None, shape=(4, 5, 6), high_shell=False):
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(40, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    bvals = np.r_[np.zeros(3), np.full(40, 1000)]
    world = np.vstack((np.zeros((3, 3)), vectors))
    if high_shell:
        bvals = np.r_[bvals, np.full(40, 2800)]
        world = np.vstack((world, vectors))
    affine = np.diag([2., 3., 4., 1.]) if affine is None else affine
    axis = np.array([1., 1., 0.]) / np.sqrt(2)
    tensor = 0.0003 * np.eye(3) + 0.0012 * np.outer(axis, axis)
    attenuation = np.einsum("ni,ij,nj->n", world, tensor, world)
    signal = np.broadcast_to(1000 * np.exp(-bvals * attenuation), (*shape, len(bvals))).astype(np.float32).copy()
    data = DiffusionData(signal, affine, bvals, gradients_in_image_axes(world, affine, "world_RAS"),
                         "world_RAS", {"dwi": "test-dwi", "bvecs": "test-bvecs"})
    return data, axis


def reviewed_evidence():
    return {"motion_corrected": True, "distortion_corrected": True,
            "gradient_rotation_reviewed": True, "brain_mask_reviewed": True,
            "registration_reviewed": True, "correction_provenance": "synthetic known geometry",
            "corrected_dwi_sha256": "test-dwi", "rotated_bvec_sha256": "test-bvecs",
            "dwi_RAS_mm_to_T1_RAS_mm": np.eye(4).tolist(), "structural_sha256": "test-T1"}


def test_bids_handedness_rule_preserves_world_direction_after_x_reorientation():
    # FSL vectors retain their radiological convention when image axes reverse.
    fsl = np.array([[1., 1., 0.]]) / np.sqrt(2)
    ras = np.diag([2., 3., 4., 1.])
    las = np.diag([-2., 3., 4., 1.])
    world_ras = gradients_in_image_axes(fsl, ras, "BIDS_FSL") @ image_axis_basis(ras).T
    world_las = gradients_in_image_axes(fsl, las, "BIDS_FSL") @ image_axis_basis(las).T
    np.testing.assert_allclose(world_ras, [[-1 / np.sqrt(2), 1 / np.sqrt(2), 0]])
    np.testing.assert_allclose(world_ras, world_las)


def test_oblique_anisotropic_world_conversion_rotates_without_scaling():
    angle = np.deg2rad(31)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([2., 3., 4.])
    ras = np.array([[0., 1., 0.]])
    image = gradients_in_image_axes(ras, affine, "world_RAS")
    np.testing.assert_allclose(image @ rotation.T, ras, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(image, axis=1), 1)
    lps = gradients_in_image_axes(ras * [-1, -1, 1], affine, "world_LPS")
    np.testing.assert_allclose(image, lps)


def test_shear_and_unknown_gradient_convention_are_named_failures():
    affine = np.eye(4)
    affine[0, 1] = 0.2
    with pytest.raises(DiffusionError, match="GRADIENT_FRAME_SHEARED"):
        gradients_in_image_axes(np.eye(3), affine, "image_axes")
    with pytest.raises(DiffusionError, match="GRADIENT_FRAME_UNRESOLVED"):
        gradients_in_image_axes(np.eye(3), np.eye(4), "scanner")


@pytest.mark.parametrize("defect,code", [
    ("count", "GRADIENT_COUNT_MISMATCH"), ("norm", "GRADIENT_NORM_INVALID"),
    ("rank", "DEGENERATE_GRADIENT_DIRECTIONS"), ("b0", "INSUFFICIENT_DIRECTIONS"),
    ("nan", "INVALID_GRADIENT_VALUES"),
])
def test_adversarial_gradient_contract(defect, code):
    data, _ = synthetic_tensor()
    bvals, bvecs = data.bvals.copy(), data.bvecs_image.copy()
    if defect == "count":
        bvecs = bvecs[:-1]
    elif defect == "norm":
        bvecs[-1] *= 2
    elif defect == "rank":
        bvecs[3:] = [1, 0, 0]
    elif defect == "b0":
        bvals[:3] = 1000
    else:
        bvals[-1] = np.nan
    with pytest.raises(DiffusionError, match=code):
        validate_gradients(bvals, bvecs, len(bvals))


def test_missing_gradients_block_before_image_load(tmp_path):
    with pytest.raises(DiffusionError, match="MISSING_GRADIENTS"):
        load_diffusion(tmp_path / "missing.nii.gz", None, None, gradient_convention="BIDS_FSL")


def test_preprocessing_evidence_cannot_be_a_ready_boolean_or_stale_hash():
    data, _ = synthetic_tensor()
    assert "DWI_MOTION_UNCORRECTED" in planning_gates(data, None)
    evidence = reviewed_evidence()
    assert planning_gates(data, evidence) == []
    evidence["corrected_dwi_sha256"] = "some-other-acquisition"
    assert "PREPROCESSING_EVIDENCE_STALE" in planning_gates(data, evidence)
    evidence = reviewed_evidence()
    evidence["dwi_RAS_mm_to_T1_RAS_mm"][0][0] = -1
    assert "DIFFUSION_REGISTRATION_TRANSFORM_UNRESOLVED" in planning_gates(data, evidence)


def test_raw_dwi_is_rejected_for_planning():
    pytest.importorskip("dipy")
    data, _ = synthetic_tensor()
    with pytest.raises(DiffusionError, match="DWI_MOTION_UNCORRECTED"):
        fit_tensor(data, np.ones(data.signal.shape[:3], dtype=bool))


def test_tensor_recovers_analytic_fa_md_and_ras_axis_with_unknown_coverage():
    pytest.importorskip("dipy")
    angle = np.deg2rad(29)
    affine = np.array([[np.cos(angle)*-2, -np.sin(angle)*3, 0, 10],
                       [np.sin(angle)*-2, np.cos(angle)*3, 0, -3], [0, 0, 4, 2], [0, 0, 0, 1]])
    data, axis = synthetic_tensor(affine=affine, high_shell=True)
    # Deliberately corrupt high-b signal: low-shell tensor must exclude it.
    data.signal[..., data.bvals > 1500] = 987.0
    mask = np.ones(data.signal.shape[:3], dtype=bool)
    mask[0] = False
    result = fit_tensor(data, mask, diagnostic_only=True, chunk_voxels=7)
    eigenvalues = np.array([0.0015, 0.0003, 0.0003])
    expected_fa = np.sqrt(1.5 * ((eigenvalues - eigenvalues.mean()) ** 2).sum() / (eigenvalues ** 2).sum())
    np.testing.assert_allclose(result["fa"][mask], expected_fa, atol=1e-6)
    np.testing.assert_allclose(result["md"][mask], eigenvalues.mean(), atol=1e-8)
    np.testing.assert_allclose(abs(result["principal_ras"][mask] @ axis), 1, atol=1e-6)
    assert np.isnan(result["fa"][~mask]).all()
    assert result["report"]["excluded_volume_count"] == 40
    assert result["report"]["clinical_deficit_probability"] is None
    assert result["report"]["motor_coverage"] == "unknown"
    assert not result["report"]["usable_for_tract_aware_planning"]


def test_reviewed_tensor_still_does_not_claim_functional_bundle_validation():
    pytest.importorskip("dipy")
    data, _ = synthetic_tensor()
    result = fit_tensor(data, np.ones(data.signal.shape[:3], bool), evidence=reviewed_evidence())
    assert result["report"]["planning_gate_failures"] == []
    assert not result["report"]["usable_for_tract_aware_planning"]
    assert result["report"]["language_coverage"] == "unknown"


def test_probabilistic_crop_uses_bounded_reproducible_native_world_geometry():
    pytest.importorskip("dipy")
    affine = np.diag([-2., 2., 2., 1.])
    affine[:3, 3] = [17, -8, 9]
    data, _ = synthetic_tensor(affine=affine, shape=(8, 8, 8), high_shell=True)
    tensor = fit_tensor(data, np.ones((8, 8, 8), bool), diagnostic_only=True)
    parameters = {"crop_start": (0, 0, 0), "crop_shape": (8, 8, 8), "seed_count": 8, "realizations": 2}
    result = probabilistic_crop_audit(data, tensor, **parameters)
    repeated = probabilistic_crop_audit(data, tensor, **parameters)
    np.testing.assert_allclose(result["points_ras_mm"], repeated["points_ras_mm"])
    assert result["offsets"][-1] == len(result["points_ras_mm"])
    assert np.isfinite(result["points_ras_mm"]).all()
    voxel = nib_inverse_points(affine, result["points_ras_mm"])
    assert voxel.min() > -1 and voxel.max() < 9
    assert np.all((result["run_visitation_fraction"] >= 0) & (result["run_visitation_fraction"] <= 1))
    assert result["report"]["functional_labels"] == []
    assert not result["report"]["usable_for_tract_aware_planning"]


def nib_inverse_points(affine, points):
    inverse = np.linalg.inv(affine)
    return points @ inverse[:3, :3].T + inverse[:3, 3]


def test_tracking_rejects_stale_patient_tensor_and_wrong_shell():
    pytest.importorskip("dipy")
    data, _ = synthetic_tensor(shape=(8, 8, 8), high_shell=True)
    tensor = fit_tensor(data, np.ones((8, 8, 8), bool), diagnostic_only=True)
    with pytest.raises(DiffusionError, match="CSA_SHELL_UNSUPPORTED"):
        probabilistic_crop_audit(data, tensor, crop_start=(0, 0, 0), crop_shape=(8, 8, 8), shell_b=1000)
    tensor["report"]["source_hashes"] = {"dwi": "different-patient"}
    with pytest.raises(DiffusionError, match="STALE_TENSOR_DERIVATIVE"):
        probabilistic_crop_audit(data, tensor, crop_start=(0, 0, 0), crop_shape=(8, 8, 8))
