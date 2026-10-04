"""Independent fixed roundoff gates on artificial grids; never patient use."""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from resectionlab.core import array_digest
from resectionlab.native_spatial_task import (
    NativeSpatialTask, make_native_opening_task, reconcile_native_grid_roundoff,
)

METHOD = "orthogonal_roundoff_1e-6mm"


def small_perturbation():
    affine = np.eye(4)
    affine[0, 1] = 1e-9
    return affine


def corner_worlds(affine, shape, *, cells):
    limits = [(-.5, size - .5) if cells else (0., size - 1.) for size in shape]
    homogeneous = np.array([(*point, 1.) for point in product(*limits)])
    return (homogeneous @ affine.T)[:, :3]


def test_whole_cell_corners_refuse_when_all_voxel_centers_would_pass():
    affine = small_perturbation()
    affine[:3, :3] *= 600.
    # Obtain the same polar frame on a smaller extent, then independently
    # measure both domains through homogeneous world-coordinate transforms.
    derived, _ = reconcile_native_grid_roundoff(affine, (1, 1, 1))
    centers = np.linalg.norm(corner_worlds(derived, (3, 3, 3), cells=False)
                            - corner_worlds(affine, (3, 3, 3), cells=False), axis=1).max()
    cells = np.linalg.norm(corner_worlds(derived, (3, 3, 3), cells=True)
                          - corner_worlds(affine, (3, 3, 3), cells=True), axis=1).max()
    assert centers < 1e-6 < cells
    with pytest.raises(ValueError, match="GRID_ROUNDOFF_DISPLACEMENT_EXCEEDED"):
        reconcile_native_grid_roundoff(affine, (3, 3, 3))


def test_normalized_gram_gate_rejects_shear_even_when_physical_motion_would_be_tiny():
    affine = np.diag([1e-6, 1e-6, 1e-6, 1.])
    affine[0, 1] = 1e-10  # Relative shear 1e-4 despite sub-nanometer movement.
    assert np.linalg.norm(affine[:3, :3] - np.eye(3) * 1e-6) < 1e-6
    with pytest.raises(ValueError, match="GRID_ROUNDOFF_MEANINGFUL_SHEAR"):
        reconcile_native_grid_roundoff(affine, (3, 3, 3))


def test_reflected_rotated_anisotropic_source_and_report_are_preserved_exactly():
    theta = .49
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                         [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([-.8, 1.9, 2.7]) @ small_perturbation()[:3, :3]
    affine[:3, 3] = [12., -8., 40.]
    before = affine.copy()
    derived, record = reconcile_native_grid_roundoff(affine, (160, 180, 90))
    np.testing.assert_array_equal(affine, before)
    np.testing.assert_array_equal(derived[:3, 3], before[:3, 3])
    original_lengths = np.linalg.norm(before[:3, :3], axis=0)
    np.testing.assert_allclose(np.linalg.norm(derived[:3, :3], axis=0), original_lengths, rtol=2e-15, atol=0)
    unit = derived[:3, :3] / original_lengths
    np.testing.assert_allclose(unit.T @ unit, np.eye(3), atol=1e-15)
    assert np.linalg.det(derived[:3, :3]) < 0
    assert record["original_affine_hash"] == array_digest(before)
    assert record["derived_affine_hash"] == array_digest(derived)
    assert record["maximum_allowed_gram_error"] == 1e-8
    assert record["maximum_allowed_corner_displacement_mm"] == 1e-6
    measured = np.linalg.norm(corner_worlds(derived, (160, 180, 90), cells=True)
        - corner_worlds(before, (160, 180, 90), cells=True), axis=1).max()
    assert record["maximum_corner_displacement_mm"] == pytest.approx(measured, abs=1e-13)
    assert not derived.flags.writeable and record["resampled"] is False
    with pytest.raises(TypeError):
        record["maximum_allowed_gram_error"] = 1.


def test_opt_in_seals_model_identity_without_changing_default_source_identity():
    source = make_native_opening_task().case
    unchanged = replace(source, native_grid_reconciliation="none")
    reconciled_identity = replace(source, native_grid_reconciliation=METHOD)
    assert source.source_hash == unchanged.source_hash
    assert source.source_hash != reconciled_identity.source_hash
    with pytest.raises(ValueError, match="orthogonal affine"):
        replace(source, affine_ras_mm=small_perturbation())
    with pytest.raises(ValueError, match="Unknown explicit"):
        replace(source, native_grid_reconciliation="auto")


def test_source_actor_native_action_certificate_and_auditor_share_one_derived_frame(monkeypatch):
    import resectionlab.evaluation as evaluation
    source = make_native_opening_task().case
    original = small_perturbation()
    reconciled = replace(source, affine_ras_mm=original, native_grid_reconciliation=METHOD, crop_shape=(5, 5, 5))
    task = NativeSpatialTask(reconciled, max_steps=2)
    np.testing.assert_array_equal(reconciled.affine_ras_mm, original)
    np.testing.assert_array_equal(reconciled.structural_intensity, source.structural_intensity)
    np.testing.assert_array_equal(reconciled.observed_support, source.observed_support)
    np.testing.assert_array_equal(reconciled.reference_target, source.reference_target)
    assert reconciled.tools == source.tools and reconciled.access is source.access
    assert reconciled._candidate_voxels == source._candidate_voxels
    np.testing.assert_array_equal(task._config.affine, reconciled._native_affine_ras_mm)
    expected_crop = reconciled._native_affine_ras_mm.copy()
    expected_crop[:3, 3] += expected_crop[:3, :3] @ reconciled._crop_origin
    observation = task.observation()
    np.testing.assert_array_equal(observation.affine_ras_mm, expected_crop)
    ledger = next(row for row in task.candidate_inventory()["ledger"] if row["feasible"])
    index = observation.action_ids.index(ledger["action_id"])
    tip = reconciled._native_affine_ras_mm[:3, :3] @ ledger["voxel"] + reconciled._native_affine_ras_mm[:3, 3]
    np.testing.assert_array_equal(observation.action_geometry[index, 4:7], tip)
    result = task.step(ledger["action_id"])
    np.testing.assert_array_equal(result.info["tip_mm"], tip)
    real_check, captured = evaluation.independent_check_native_history, []

    def capture(case, tools, history, **kwargs):
        captured.append(case.affine.copy())
        return real_check(case, tools, history, **kwargs)

    monkeypatch.setattr(evaluation, "independent_check_native_history", capture)
    assert task.independent_geometry_check().feasible
    np.testing.assert_array_equal(captured[0], reconciled._native_affine_ras_mm)
    metadata = task.metrics()["native_grid_reconciliation"]
    assert metadata["source_image_hash"] == array_digest(source.structural_intensity)
    assert metadata["support_hash"] == array_digest(source.observed_support)
    metadata["maximum_allowed_corner_displacement_mm"] = 1.
    assert task.metrics()["native_grid_reconciliation"]["maximum_allowed_corner_displacement_mm"] == 1e-6


@pytest.mark.parametrize("changed", ["derived_affine", "report", "mode"])
def test_reconciliation_authority_replacement_invalidates_model(changed):
    source = replace(make_native_opening_task().case, native_grid_reconciliation=METHOD)
    if changed == "derived_affine":
        shifted = source._native_affine_ras_mm.copy()
        shifted[0, 3] += 1e-9
        object.__setattr__(source, "_native_affine_ras_mm", shifted)
    elif changed == "report":
        object.__setattr__(source, "_grid_record", {**source._grid_record, "derived_affine_hash": "forged"})
    else:
        object.__setattr__(source, "native_grid_reconciliation", "none")
    with pytest.raises(RuntimeError, match="interpretation was replaced"):
        source.assert_intact()
