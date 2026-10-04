"""Independent integration adversaries; analytic histories, no patient runs."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab import evaluation, independent_geometry_batch as kernel
from resectionlab.geometry import AccessWindow, ToolGeometry


def _column_history():
    """Three removable cells plus a lateral partial contact that stays occupied.

    Coordinates and contact lists are constructed directly, without invoking a
    planning rasterizer or native rollout. All source cells are unit cubes.
    """
    tissue = np.zeros((7, 7, 8), bool)
    tissue[3, 3, 3:6] = True
    tissue[4, 3, 3] = True
    case = SimpleNamespace(mri=np.zeros(tissue.shape), affine=np.eye(4), frame="RAS+",
                           semantic_hash="independent-batch-column")
    tool = ToolGeometry("review-column", 1., .2, 8., tip_length_mm=2.)
    access = AccessWindow((3., 3., 2.), (0., 0., 1.), 3.)
    removals = [[], [[3, 3, 3]], [[3, 3, 4]], [[3, 3, 5]]]
    contacts = [[[3, 3, 3], [4, 3, 3]],
                [[3, 3, 3], [3, 3, 4], [4, 3, 3]],
                [[3, 3, 4], [3, 3, 5], [4, 3, 3]],
                [[3, 3, 5], [4, 3, 3]]]
    tips = [(2., 2.), (2., 3.), (3., 4.), (4., 5.)]
    microsteps = [dict(tip_start_mm=(3., 3., a), tip_end_mm=(3., 3., b),
        active_stroke_start_mm=(3., 3., a - 2.), active_stroke_end_mm=(3., 3., b),
        active_radius_mm=1., removed_indices_native=removed, contact_indices_native=contact)
        for (a, b), removed, contact in zip(tips, removals, contacts)]
    history = [dict(action_id="review-cut", source_hash=case.semantic_hash, source_shape=tissue.shape,
        native_affine=np.eye(4).tolist(), native_footprint="fully_contained_connected_cells_v1",
        tool_id=tool.tool_id, entry_mm=access.center_mm.tolist(), tip_mm=[3., 3., 5.],
        axis_unit=[0., 0., 1.], removed_indices_native=[[3, 3, 3], [3, 3, 4], [3, 3, 5]],
        removed_volume_mm3=3., microsteps=microsteps)]
    return case, tool, access, tissue, history


def _audit(fixture, **kwargs):
    case, tool, access, tissue, history = fixture
    return evaluation.independent_check_native_history(case, (tool,), history,
        tissue_mask=tissue, access=access, **kwargs)


def test_scalar_default_cannot_enter_opt_in_kernel(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Scalar default entered opt-in batch kernel")

    monkeypatch.setattr(kernel, "segment_box_contact_indices", forbidden)
    monkeypatch.setattr(kernel, "first_segment_box_contact", forbidden)
    fixture = _column_history()
    default = _audit(fixture)
    explicit = _audit(fixture, distance_backend="scalar")
    assert default.to_dict() == explicit.to_dict()
    assert default.feasible and default.contained_source_tissue_volume_mm3 == 3.


def test_disconnected_frontier_cannot_be_bypassed_by_batch_backend():
    case, tool, access, tissue, history = _column_history()
    tissue[:] = False
    tissue[1:6, 1:6, 1:7] = True
    # Retain only the first cut. Its cell is geometrically contained but every
    # neighbor is tissue; the surrounding free exterior is disconnected.
    cut = deepcopy(history[0]["microsteps"][1])
    history[0].update(microsteps=[cut], removed_indices_native=[[3, 3, 3]],
                      removed_volume_mm3=1., tip_mm=[3., 3., 3.])
    scalar = _audit((case, tool, access, tissue, history))
    batched = _audit((case, tool, access, tissue, history), distance_backend="batch", distance_batch_size=1)
    assert scalar.to_dict() == batched.to_dict()
    assert scalar.failures == ("disconnected_native_removal",)
    assert scalar.first_unsupported_source_voxel == (3, 3, 3)
    assert scalar.contained_source_tissue_volume_mm3 == 0.


def test_handbuilt_history_matches_analytic_contact_and_remaining_tissue_prefixes(monkeypatch):
    fixture = _column_history()
    original_contacts = evaluation._native_active_contacts
    original_extend = evaluation._extend_independent_free_space
    microsteps = fixture[4][0]["microsteps"]
    expected_before, expected_after = [], []
    remaining = fixture[3].copy()
    for micro in microsteps:
        expected_before.append(remaining.copy())
        for index in micro["removed_indices_native"]:
            remaining[tuple(index)] = False
        expected_after.append(remaining.copy())
    state = dict(contact=0, prefix=0)

    def contacts(current, *args, **kwargs):
        index = state["contact"]
        np.testing.assert_array_equal(current, expected_before[index])
        found = original_contacts(current, *args, **kwargs)
        assert found == {tuple(cell) for cell in microsteps[index]["contact_indices_native"]}
        state["contact"] += 1
        return found

    def extend(current, free, keys, **kwargs):
        index = state["prefix"]
        original_extend(current, free, keys, **kwargs)
        np.testing.assert_array_equal(current, expected_after[index])
        # In this sparse analytic scene every empty cell is exterior-connected.
        np.testing.assert_array_equal(free, ~current)
        assert current[4, 3, 3], "Lateral partial contact must remain tissue"
        state["prefix"] += 1

    monkeypatch.setattr(evaluation, "_native_active_contacts", contacts)
    monkeypatch.setattr(evaluation, "_extend_independent_free_space", extend)
    certificates = []
    for backend in ("scalar", "batch"):
        state.update(contact=0, prefix=0)
        certificates.append(_audit(fixture, distance_backend=backend, distance_batch_size=1))
        assert state == dict(contact=4, prefix=4)
    assert certificates[0].feasible
    assert certificates[0].to_dict() == certificates[1].to_dict()


@pytest.mark.parametrize("batch_size", [1, 2, 256])
def test_multiple_omitted_contacts_preserve_scalar_failure_witness(batch_size):
    fixture = _column_history()
    fixture[4][0]["microsteps"][0]["contact_indices_native"] = []
    scalar = _audit(fixture)
    batched = _audit(fixture, distance_backend="batch", distance_batch_size=batch_size)
    assert scalar.failures == ("unrecorded_partial_active_tissue_contact",)
    assert scalar.to_dict() == batched.to_dict()
    assert scalar.first_unsupported_source_voxel in {(3, 3, 3), (4, 3, 3)}


@pytest.mark.parametrize("reflect", [False, True])
@pytest.mark.parametrize("reverse_segment", [False, True])
def test_shaft_first_physical_witness_preserves_argwhere_order(reflect, reverse_segment):
    rotation = Rotation.from_euler("xyz", [22., 7., -11.], degrees=True).as_matrix()
    if reflect:
        rotation[:, 0] *= -1
    spacing = np.array([.5, 1.25, 2.])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag(spacing)
    affine[:3, 3] = [19., -12., 37.]
    occupied = np.zeros((5, 4, 4), bool)
    occupied[1:4, 1, 1] = True
    scene = SimpleNamespace(forbidden_mask=occupied, affine=affine)
    start = rotation @ np.array([0., 1.25, 2.]) + affine[:3, 3]
    end = rotation @ np.array([2., 1.25, 2.]) + affine[:3, 3]
    if reverse_segment:
        start, end = end, start
    scalar = evaluation._cell_collision(scene, start, end, .1)
    expected = affine[:3, :3] @ np.array([1, 1, 1]) + affine[:3, 3]
    np.testing.assert_allclose(scalar, expected, rtol=0, atol=1e-12)
    for size in (1, 2, 256):
        assert evaluation._cell_collision(scene, start, end, .1, _batch_size=size) == scalar


def test_batch_contact_helper_respects_supplied_order_without_sorting():
    cells = np.array([[3, 1, 1], [1, 1, 1], [2, 1, 1]], dtype=int)
    results = list(evaluation._batch_cell_contacts(
        np.array([0., 1., 1.]), np.array([4., 1., 1.]), cells,
        np.ones(3), .1, 1e-10, 2, None, first_only=True))
    assert len(results) == 1
    np.testing.assert_array_equal(results[0], [[3, 1, 1]])


def test_hard_exclusion_checks_remain_scalar_under_opt_in_contacts(monkeypatch):
    fixture = _column_history()
    hard = np.zeros(fixture[3].shape, bool)
    hard[3, 3, 2] = True
    original = evaluation._cell_collision
    hard_backend_calls = []

    def inspect(scene, *args, **kwargs):
        if np.array_equal(scene.forbidden_mask, hard):
            hard_backend_calls.append(kwargs.get("_batch_size"))
        return original(scene, *args, **kwargs)

    monkeypatch.setattr(evaluation, "_cell_collision", inspect)
    scalar = _audit(fixture, hard_exclusion=hard)
    batched = _audit(fixture, hard_exclusion=hard, distance_backend="batch", distance_batch_size=1)
    assert scalar.to_dict() == batched.to_dict()
    assert scalar.failures == ("native_full_tool_hard_constraint_failure",)
    assert hard_backend_calls and all(value is None for value in hard_backend_calls)


@pytest.mark.parametrize("cancel_stage", ["contact", "shaft"])
def test_mid_batch_cancellation_preserves_accepted_prefix_and_refuses_pending_cut(cancel_stage, monkeypatch):
    fixture = _column_history()
    original_tissue = fixture[3].copy()
    original_history = deepcopy(fixture[4])
    actual_contacts = evaluation._native_active_contacts
    actual_collision = evaluation._cell_collision
    actual_kernel = kernel._distances_chunk
    state = dict(microstep=0, stage=None, cancelled=False, triggered=False)

    def traced_contacts(*args, **kwargs):
        state["microstep"] += 1
        state["stage"] = "contact"
        return actual_contacts(*args, **kwargs)

    def traced_collision(*args, **kwargs):
        state["stage"] = "shaft" if kwargs.get("_batch_size") is not None else "hard"
        return actual_collision(*args, **kwargs)

    def cancel_after_kernel(*args):
        result = actual_kernel(*args)
        if state["microstep"] == 3 and state["stage"] == cancel_stage:
            state.update(cancelled=True, triggered=True)
        return result

    monkeypatch.setattr(evaluation, "_native_active_contacts", traced_contacts)
    monkeypatch.setattr(evaluation, "_cell_collision", traced_collision)
    monkeypatch.setattr(kernel, "_distances_chunk", cancel_after_kernel)
    result = _audit(fixture, distance_backend="batch", distance_batch_size=1,
                    cancelled=lambda: state["cancelled"])
    assert state["triggered"], "Fixture must reach the intended cancellation boundary"
    assert result.failures == ("independent_validation_cancelled",)
    assert not result.feasible and not result.complete_tool_checked and not result.frontier_checked
    assert result.claimed_source_tissue_volume_mm3 == 2.
    assert result.contained_source_tissue_volume_mm3 == 1.
    assert result.unsupported_source_tissue_volume_mm3 == 1.
    np.testing.assert_array_equal(fixture[3], original_tissue)
    assert fixture[4] == original_history
