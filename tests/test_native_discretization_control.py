"""Small analytic numerical controls; no patient, optimizer or matrix experiment."""
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("discretization_control", ROOT / "scripts/probe_native_discretization.py")
PROBE = module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def clipped_hull_fraction(center_z, affine):
    """Separate geometric construction for the analytical fraction helper."""
    corners = np.asarray(list(product((-.5, .5), repeat=3)))
    points = corners @ affine[:3, :3].T + (0, 0, center_z)
    kept = [point for point in points if point[2] >= 0]
    if not kept:
        return 0.
    if len(kept) == 8:
        return 1.
    for i, a in enumerate(points):
        for j in range(i + 1, len(points)):
            if np.count_nonzero(corners[i] != corners[j]) != 1:
                continue
            b = points[j]
            if a[2] * b[2] < 0:
                kept.append(a + (b - a) * (-a[2] / (b[2] - a[2])))
    return ConvexHull(np.asarray(kept)).volume / abs(np.linalg.det(affine[:3, :3]))


def test_halfspace_fraction_has_exact_axial_and_symmetry_controls():
    np.testing.assert_allclose(PROBE.half_space_fractions(np.array([-1, -.5, -.25, 0, .25, .5, 1]), np.eye(4)),
                               [0, 0, .25, .5, .75, 1, 1], atol=1e-15, rtol=0)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [13, -17, 23], degrees=True).as_matrix()
    values = PROBE.half_space_fractions(np.array([-.7, -.2, 0, .2, .7]), matrix)
    np.testing.assert_allclose(values + values[::-1], np.ones(5), atol=2e-14, rtol=0)


@pytest.mark.parametrize("angles,spacing", [([0, 20, 0], [1, 1, 1]), ([13, -17, 23], [.5, 1., 1.5]), ([31, 27, -12], [1, 2, .25])])
def test_analytic_fraction_matches_independent_clipped_convex_hull(angles, spacing):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix() @ np.diag(spacing)
    centers = [-.7, -.3, -.1, 0, .2, .4, .8]
    actual = PROBE.half_space_fractions(centers, matrix)
    expected = [clipped_hull_fraction(z, matrix) for z in centers]
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)


def test_grid_revoxelization_keeps_physical_scene_and_tool_fixed():
    spec = PROBE.declaration()
    first_case, first = PROBE.grid_case(1., spec["grids"][0], spec)
    changed_case, changed = PROBE.grid_case(1., spec["grids"][3], spec)
    np.testing.assert_array_equal(first.access.center_mm, changed.access.center_mm)
    np.testing.assert_array_equal(first.access.normal_inward, changed.access.normal_inward)
    assert first.tools == changed.tools
    assert first.max_tip_step_mm == changed.max_tip_step_mm == .0625
    assert first.source_hash != changed.source_hash
    assert not np.array_equal(first.affine, changed.affine)
    for source, config in ((first_case, first), (changed_case, changed)):
        indices = np.indices(source.mri.shape).reshape(3, -1).T
        centers_z = indices @ config.affine[2, :3] + config.affine[2, 3]
        np.testing.assert_array_equal(config.tissue_mask.ravel(), centers_z + abs(config.affine[2, :3]).sum() / 2 > 0)


def test_phase_changes_voxel_boundaries_not_physical_entry_or_instrument():
    spec = PROBE.declaration()
    _, first = PROBE.grid_case(1., spec["grids"][0], spec)
    _, phase = PROBE.grid_case(1., spec["grids"][1], spec)
    center_first = np.linalg.solve(first.affine[:3, :3], -first.affine[:3, 3])
    center_phase = np.linalg.solve(phase.affine[:3, :3], -phase.affine[:3, 3])
    np.testing.assert_allclose(center_first % 1, 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(center_phase % 1, .5, atol=1e-12, rtol=0)
    assert first.tools == phase.tools


def test_one_coarse_control_keeps_partial_cells_and_brackets_analytic_sweep():
    spec = PROBE.declaration()
    row, _, _, history = PROBE.numerical_row(1., spec["grids"][0], spec)
    ideal = PROBE.ideal_volume()
    assert row["static_not_executable"]["contained_physical_lower_mm3"] < ideal
    assert row["static_not_executable"]["contact_physical_upper_mm3"] > ideal
    assert row["native"]["feasible"]
    assert row["native"]["partial_cells_kept_occupied"]
    assert row["native"]["removed_outer_voxelization_excess_mm3"] > 0
    assert history and all(step["contact_indices_native"] is not None for step in history[0]["microsteps"])


def test_declaration_is_bounded_numerical_control_not_training():
    spec = PROBE.declaration()
    assert spec["expected_rows"] == len(spec["grids"]) * len(spec["spacings_mm"]) == 16
    assert spec["caps"]["cooperative_worker_seconds"] == 170 < spec["caps"]["parent_wall_seconds"] == 180
    assert spec["caps"]["worker_peak_rss_bytes"] == 2 * 1024**3
    assert spec["role"] == "analytic_numerical_unit_control_not_training_data"
    assert min(spec["spacings_mm"]) / 2 == spec["max_tip_step_mm"]


def test_fresh_output_gate_cannot_overwrite_evidence(tmp_path):
    with pytest.raises(FileExistsError):
        PROBE.launch(tmp_path)


@pytest.mark.parametrize("worker_status,expected", [
    ("completed", "completed"),
    ("completed_numerical_rows_audits_incomplete", "completed_numerical_rows_audits_incomplete"),
    ("failed", "incomplete_or_failed"),
])
def test_launcher_retains_worker_outcome_instead_of_equating_exit_zero_with_audit_completion(tmp_path, monkeypatch, worker_status, expected):
    output = tmp_path / "fresh"
    spec = PROBE.declaration()
    source = PROBE.source_receipt()
    monkeypatch.setattr(PROBE, "source_receipt", lambda: source)
    class Worker:
        def __init__(self, *args, **kwargs):
            self.pid = 123
            PROBE.write_json(output / "report.json", {
                "source": PROBE.source_receipt(), "source_unchanged": True,
                "declaration_hash": PROBE.digest(spec), "completed_rows": 16,
                "rows": [{}] * 16, "status": worker_status})
        def poll(self):
            return 0
        def wait(self):
            return 0
    monkeypatch.setattr(PROBE.subprocess, "Popen", Worker)
    PROBE.launch(output)
    receipt = PROBE.json.loads((output / "launcher.json").read_text())
    assert receipt["status"] == expected
    assert receipt["independent_audits_complete"] == (worker_status == "completed")
