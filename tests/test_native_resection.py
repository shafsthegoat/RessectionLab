"""Full-cell coverage, native mass, temporal legality, and tool sensitivity."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.geometry import AccessWindow, GeometryScene, ToolGeometry, capsule_voxel_indices
from resectionlab.native_resection import (
    NATIVE_GENERIC_TOOLS, NativeResectionConfig, NativeResectionEngine,
    _connected_surface_cells, contained_capsule_cells, native_config_from_case,
)


def config(*, tools=NATIVE_GENERIC_TOOLS, affine=None):
    tissue = np.zeros((21, 21, 30), bool)
    tissue[2:19, 2:19, 5:25] = True
    labels = np.zeros(tissue.shape, np.int16)
    labels[9:12, 9:12, 12:20] = 1
    affine = np.eye(4) if affine is None else affine
    entry = affine[:3, :3] @ [10, 10, 4.5] + affine[:3, 3]
    direction = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    return NativeResectionConfig(tissue, labels, affine, AccessWindow(entry, direction, 4),
                                 tools, "analytic-source", "explicit synthetic box")


def test_native_stroke_removes_only_contained_cells_and_charges_normal_access():
    engine = NativeResectionEngine(config())
    result = engine.execute_stroke(NATIVE_GENERIC_TOOLS[0].tool_id, [10, 10, 15])
    assert result.feasible, result.reason
    assert result.removed_volume_mm3 == 11
    metrics = engine.metrics()
    assert metrics["simulated_removed_target_mm3"] == 4
    assert metrics["simulated_removed_normal_mm3"] == 7
    assert metrics["contacted_but_not_removed_upper_bound_mm3"] > 0
    assert np.all(engine.remaining_mask[engine.contact_mask & ~engine.removed_mask])
    assert len(result.microsteps) == 43
    assert result.to_history_record()["clinical_deficit_probability"] is None


def test_preview_is_pure_and_commit_is_once_and_state_bound():
    engine = NativeResectionEngine(config())
    original = engine.state_hash
    preview = engine.preview_stroke(NATIVE_GENERIC_TOOLS[0].tool_id, [10, 10, 15])
    assert preview.feasible and not engine.removed_mask.any()
    assert engine.state_hash == original
    child = engine.clone()
    child.commit_preview(preview)
    assert child.removed_mask.any() and not engine.removed_mask.any()
    engine.commit_preview(preview)
    with pytest.raises(ValueError, match="stale"):
        engine.commit_preview(preview)


def test_forged_or_deserialized_preview_cannot_skip_geometry():
    engine = NativeResectionEngine(config())
    preview = engine.preview_stroke(NATIVE_GENERIC_TOOLS[0].tool_id, [10, 10, 15])
    with pytest.raises(ValueError, match="foreign"):
        engine.commit_preview(replace(preview))


def test_repeated_stroke_has_no_duplicate_removal_credit():
    engine = NativeResectionEngine(config())
    first = engine.execute_stroke(NATIVE_GENERIC_TOOLS[0].tool_id, [10, 10, 15])
    second = engine.execute_stroke(NATIVE_GENERIC_TOOLS[0].tool_id, [10, 10, 15])
    assert first.feasible and not second.feasible
    assert second.removed_volume_mm3 == 0
    assert engine.metrics()["completed_strokes"] == 1


def test_wider_active_tip_removes_more_source_cells_with_separate_tool_hash():
    fine, wide = (NativeResectionEngine(config(tools=(tool,))) for tool in NATIVE_GENERIC_TOOLS)
    first = fine.execute_stroke(NATIVE_GENERIC_TOOLS[0].tool_id, [10, 10, 15])
    second = wide.execute_stroke(NATIVE_GENERIC_TOOLS[1].tool_id, [10, 10, 15])
    assert first.feasible and second.feasible
    assert second.removed_volume_mm3 > first.removed_volume_mm3
    assert fine.decision_model_hash != wide.decision_model_hash


def test_old_coarse_tool_cannot_clear_wider_shaft_using_partial_tip_contacts():
    tool = ToolGeometry("coarse-tool-kept-unchanged", .8, 1.2, 120, 35, 1)
    engine = NativeResectionEngine(config(tools=(tool,)))
    result = engine.execute_stroke(tool.tool_id, [10, 10, 15])
    assert not result.feasible and result.reason == "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE"
    assert not engine.removed_mask.any() and not engine.contact_mask.any()
    assert result.removed_volume_mm3 == 0


def test_coarse_216_mm3_cell_touch_does_not_credit_216_mm3_removal():
    tissue = np.zeros((5, 5, 5), bool)
    tissue[2, 2, 1] = True
    tool = ToolGeometry("tiny", .8, .1, 20, 20, 1)
    cfg = NativeResectionConfig(tissue, tissue.astype(int), np.diag([6, 6, 6, 1]),
                                AccessWindow([12, 12, 2.5], [0, 0, 1], 2), (tool,),
                                "explicit-6mm-source", "synthetic coarse-source negative")
    result = NativeResectionEngine(cfg).preview_stroke("tiny", [12, 12, 2.75])
    assert not result.feasible
    assert result.removed_volume_mm3 == 0


def test_sphere_contact_and_full_voxel_containment_are_different():
    scene = GeometryScene(np.zeros((4, 4, 4), bool), np.eye(4))
    contact = capsule_voxel_indices(scene, [1, 1, 1], [1, 1, 1], .2)
    assert len(contact) == 1
    contained = contained_capsule_cells(scene, contact, np.array([1, 1, 1]), np.array([1, 1, 1]), .2)
    assert len(contained) == 0


def test_containment_uses_all_affine_corners_and_physical_mm():
    angle = .4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([2, 1, 1])
    affine[:3, 3] = [20, -10, 5]
    scene = GeometryScene(np.zeros((4, 4, 4), bool), affine)
    center = scene.voxel_to_world([1, 1, 1])
    cells = np.array([[1, 1, 1]])
    assert not len(contained_capsule_cells(scene, cells, center, center, 1.1))
    assert len(contained_capsule_cells(scene, cells, center, center, 1.3)) == 1


def test_enclosed_free_pocket_does_not_create_a_new_removal_frontier():
    connected_free = np.zeros((7, 7, 7), bool)
    connected_free[0, :, :] = True
    assert len(_connected_surface_cells(np.array([[3, 3, 3]]), connected_free)) == 0
    cells = np.array([[1, 3, 3], [2, 3, 3], [3, 3, 3]])
    assert len(_connected_surface_cells(cells, connected_free)) == 3


def test_microstep_never_borrows_removal_that_occurs_at_its_end():
    tissue = np.zeros((5, 5, 10), bool)
    tissue[2, 2, 5] = True
    tool = ToolGeometry("short-active", .9, .1, 10, 20, .1)
    for step in (.5, .25, .1):
        cfg = NativeResectionConfig(tissue, tissue.astype(int), np.eye(4),
                                    AccessWindow([2, 2, 4.49], [0, 0, 1], 2), (tool,),
                                    "temporal-counterexample", "single synthetic source cell", max_tip_step_mm=step)
        result = NativeResectionEngine(cfg).preview_stroke(tool.tool_id, [2, 2, 5.49])
        assert not result.feasible
        assert result.reason == "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE"


def test_native_config_and_certificate_arrays_are_immutable():
    cfg = config()
    result = NativeResectionEngine(cfg).preview_stroke(NATIVE_GENERIC_TOOLS[0].tool_id, [10, 10, 15])
    for array in (cfg.tissue_mask, cfg.affine, result.removed_indices_native,
                  result.contact_indices_native, result.microsteps[0].removed_indices_native):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_malformed_resolution_and_missing_provenance_are_rejected():
    cfg = config()
    with pytest.raises(ValueError):
        replace(cfg, max_tip_step_mm=.6)
    with pytest.raises(ValueError):
        replace(cfg, tissue_support_provenance="")
    with pytest.raises(ValueError):
        replace(cfg, source_hash="")


def test_failed_hard_exclusion_preview_does_not_change_state():
    cfg = config()
    hard = np.zeros(cfg.tissue_mask.shape, bool)
    hard[10, 10, 8] = True
    engine = NativeResectionEngine(replace(cfg, hard_exclusion=hard))
    result = engine.execute_stroke(NATIVE_GENERIC_TOOLS[0].tool_id, [10, 10, 15])
    assert not result.feasible and result.reason.startswith("HARD_GEOMETRY:")
    assert not engine.removed_mask.any() and engine.revision == 0


def test_parallel_entries_inside_same_aperture_remove_separate_paid_columns():
    engine = NativeResectionEngine(config())
    tool = NATIVE_GENERIC_TOOLS[0].tool_id
    first = engine.execute_stroke(tool, [10, 10, 15])
    second = engine.execute_stroke(tool, [12, 10, 15], entry_mm=[12, 10, 4.5])
    assert first.feasible and second.feasible
    assert not set(map(tuple, first.removed_indices_native)) & set(map(tuple, second.removed_indices_native))
    assert second.to_history_record()["entry_mm"] == (12., 10., 4.5)
    assert engine.metrics()["simulated_removed_normal_mm3"] > 7
    assert second.source_state_hash != first.source_state_hash


def test_shifted_entry_requires_declared_plane_and_full_aperture_clearance():
    engine = NativeResectionEngine(config())
    tool = NATIVE_GENERIC_TOOLS[0].tool_id
    with pytest.raises(ValueError, match="plane"):
        engine.preview_stroke(tool, [12, 10, 15], entry_mm=[12, 10, 5])
    rejected = engine.preview_stroke(tool, [14, 10, 15], entry_mm=[14, 10, 4.5])
    assert not rejected.feasible and "ACCESS_APERTURE" in rejected.reason


def test_native_case_helper_respects_explicit_anatomy_conflicts_and_source_frame():
    cfg = config()
    case = SimpleNamespace(mri=cfg.tissue_mask.astype(float), compartments={"target": cfg.target_labels > 0},
                           brain_mask=cfg.tissue_mask, affine=cfg.affine, metadata={},
                           case_id="analytic", semantic_hash="source-hash", frame="LPS+")
    converted = native_config_from_case(case, access=cfg.access)
    np.testing.assert_array_equal(converted.affine, np.diag([-1., -1., 1., 1.]))
    np.testing.assert_array_equal(converted.access.center_mm, [-10, -10, 4.5])
    np.testing.assert_array_equal(converted.target_labels, cfg.target_labels)
    case.brain_mask = np.zeros_like(cfg.tissue_mask)
    with pytest.raises(ValueError, match="outside"):
        native_config_from_case(case, access=cfg.access)
    case.brain_mask = None
    case.metadata = {"skull_stripped": False, "source_collection": {"name": "UCSF-PDGM"}}
    with pytest.raises(ValueError, match="brain"):
        native_config_from_case(case, access=cfg.access)
