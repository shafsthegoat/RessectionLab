"""Route-search invariants with analytic physical-grid anatomy."""

from dataclasses import replace
import json

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, StalePlanError
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.planning import (
    SearchConfig,
    classify_candidates,
    generate_candidate_routes,
    generate_hypothetical_windows,
    replay_route,
    sample_target_points,
)


def fixture_case(spacing=(1.0, 1.0, 1.0)):
    shape = (32, 32, 32)
    voxels = np.indices(shape)
    target = sum((voxels[i] - (20, 16, 16)[i]) ** 2 for i in range(3)) <= 16
    brain = np.zeros(shape, dtype=bool)
    brain[4:29, 4:29, 4:29] = True
    return CaseData(
        "synthetic-planning-fixture", brain.astype(np.float32), {"enhancing": target},
        np.diag((*spacing, 1.0)), (SourceRef("analytic", "synthetic:route-test", provenance="simulated"),),
        brain_mask=brain,
    )


def single_route(case=None, *, tool=None, critical_masks=None, window_radius=6.0):
    case = fixture_case() if case is None else case
    tool = tool or ToolGeometry("narrow", 0.75, 1.0, 40.0, tip_length_mm=2.0)
    entry = tuple(case.voxel_to_world((5, 16, 16)))
    window = AccessWindow(entry, (1, 0, 0), window_radius)
    return generate_candidate_routes(
        case, windows=(window,), tools=(tool,), critical_masks=critical_masks,
        config=SearchConfig(targets_per_compartment=1),
    )


def test_aperture_rejects_complete_shaft_and_retains_failure_pose():
    case = fixture_case()
    narrow = ToolGeometry("narrow", 0.7, 0.9, 40.0)
    wide = ToolGeometry("wide-shaft", 0.7, 2.8, 40.0)
    window = AccessWindow((5, 16, 16), (1, 0, 0), 2.0)
    result = generate_candidate_routes(case, windows=(window,), tools=(narrow, wide), config=SearchConfig(targets_per_compartment=1))
    first, second = result.candidates
    assert first.feasible
    assert second.category == "rejected"
    assert second.geometry.failures
    assert len(second.geometry.failures[0].position_mm) == 3
    assert second.accessible_target_volume_mm3 == 0
    assert second.simulated_removed_target_volume_mm3 is None


def test_missing_evidence_is_unassessed_and_never_zero_risk():
    route = single_route().candidates[0]
    assert route.feasible
    assert route.assessment == "incomplete"
    assert route.structure_contact_volume_mm3 == {"motor": None, "language": None, "vessels": None}
    assert {"motor_anatomy_unassessed", "language_anatomy_unassessed", "vessels_anatomy_unassessed"} <= set(route.unknowns)
    assert route.clinical_deficit_probability is None
    assert route.simulated_removed_target_volume_mm3 is None
    assert route.normal_tissue_exposure_mm3 > 0
    assert route.accessible_target_volume_mm3 > 0
    # Geometry tradeoffs are still inspectable, but have no safety interpretation.
    assert route.category == "pareto"
    assert "Unknown anatomy" in route.metric_definitions["pareto"]


def test_accessible_volume_uses_physical_units_and_union_not_removal():
    case = fixture_case(spacing=(2.0, 1.0, 1.0))
    tool = ToolGeometry("subvoxel", 0.1, 0.5, 60.0)
    route = single_route(case, tool=tool).candidates[0]
    # Five target centres x=16..20 lie on this half-sphere insertion; voxel=2mm3.
    assert route.accessible_target_volume_mm3 == pytest.approx(10.0)
    assert route.simulated_removed_target_volume_mm3 is None
    overlap = case.revised(compartments={"enhancing": case.compartments["enhancing"], "whole_target": case.compartments["enhancing"]})
    overlapping_route = single_route(overlap, tool=tool).candidates[0]
    assert overlapping_route.accessible_target_volume_mm3 == 10.0
    assert sum(overlapping_route.accessible_target_volume_mm3_by_compartment.values()) == 20.0


def test_search_replay_export_and_case_edit_invalidation():
    case = fixture_case()
    first = single_route(case)
    second = single_route(case)
    route = first.candidates[0]
    assert route.to_dict() == second.candidates[0].to_dict()
    assert first.planning_model_hash == second.planning_model_hash
    poses = replay_route(route, step_mm=1.0)
    assert np.allclose(poses[0].tip_mm, route.entry_mm)
    assert np.allclose(poses[-1].tip_mm, route.target_mm)
    assert np.allclose([p.axis_unit for p in poses], poses[0].axis_unit)
    assert np.max(np.linalg.norm(np.diff([p.tip_mm for p in poses], axis=0), axis=1)) <= 1.0 + 1e-9
    plan = route.to_plan()
    plan.assert_current(case)
    json.dumps(plan.to_dict(), allow_nan=False)
    assert "swept_voxel_indices" not in route.to_dict()["geometry"]
    with pytest.raises(StalePlanError):
        plan.assert_current(case.revised())


def test_cancellation_returns_partial_evaluated_candidates():
    case = fixture_case()
    progress_values = []
    result = generate_candidate_routes(
        case, config=SearchConfig(targets_per_compartment=3),
        progress=lambda done, total: progress_values.append((done, total)),
        cancel=lambda: bool(progress_values and progress_values[-1][0] >= 2),
    )
    assert result.cancelled
    assert len(result.candidates) == 2
    assert result.requested_candidates > 2
    assert progress_values[0][0] == 0
    assert progress_values[-1][0] == 2


def test_sampled_dominance_rejects_different_evidence_and_case():
    original = single_route().candidates[0]
    worse = replace(original, route_id="worse", route_length_mm=original.route_length_mm + 1)
    best, dominated = classify_candidates((original, worse))
    assert best.category == "pareto"
    assert dominated.category == "dominated"
    assert dominated.dominated_by == (original.route_id,)
    with pytest.raises(ValueError, match="different frozen"):
        classify_candidates((original, replace(worse, case_hash="changed")))
    with pytest.raises(ValueError, match="evidence coverage"):
        classify_candidates((original, replace(worse, structure_contact_volume_mm3={"motor": 0.0})))


def test_critical_masks_have_fixed_snapshot_and_are_hashed():
    case = fixture_case()
    vessels = np.zeros(case.mri.shape, dtype=bool)
    original_hash = single_route(case, critical_masks={"vessels": vessels}).planning_model_hash
    vessels[12, 16, 16] = True
    blocked = single_route(case, critical_masks={"vessels": vessels})
    assert blocked.planning_model_hash != original_hash
    assert not blocked.candidates[0].feasible
    assert blocked.candidates[0].geometry.failures
    with pytest.raises(ValueError, match="boolean mask"):
        single_route(case, critical_masks={"motor": np.zeros((2, 2, 2), dtype=bool)})


def test_hypothetical_windows_require_brain_support_and_target_samples_are_native():
    case = fixture_case()
    windows = generate_hypothetical_windows(case)
    assert len(windows) == 3
    assert all("hypothetical" in w.window_id for w in windows)
    for compartment, point in sample_target_points(case):
        index = np.rint(case.world_to_voxel(point)).astype(int)
        assert case.compartments[compartment][tuple(index)]
    with pytest.raises(ValueError, match="BRAIN_ENVELOPE_REQUIRED"):
        generate_hypothetical_windows(case.revised(brain_mask=None))
    no_brain = single_route(case.revised(brain_mask=None)).candidates[0]
    assert no_brain.normal_tissue_exposure_mm3 is None


def test_wider_active_tip_changes_static_accessibility():
    narrow = single_route(tool=ToolGeometry("narrow-tip", 0.5, 1.0, 40.0)).candidates[0]
    wide = single_route(tool=ToolGeometry("wide-tip", 2.0, 2.0, 40.0)).candidates[0]
    assert narrow.feasible and wide.feasible
    assert wide.accessible_target_volume_mm3 > narrow.accessible_target_volume_mm3
    assert wide.normal_tissue_exposure_mm3 >= narrow.normal_tissue_exposure_mm3


def test_explicit_estimated_support_never_becomes_brain_tissue():
    original = fixture_case()
    case = original.revised(brain_mask=None)
    provenance = {"source": "synthetic image", "method": "nonzero intensity support", "evidence_type": "estimated"}
    result = generate_candidate_routes(case, support_mask=original.mri != 0, support_provenance=provenance,
                                       config=SearchConfig(targets_per_compartment=1))
    assert result.access_support["evidence_type"] == "estimated"
    assert len(result.access_support["mask_sha256"]) == 64
    assert all(c.normal_tissue_exposure_mm3 is None for c in result.candidates)
    assert all("normal_tissue_exposure_unassessed" in c.unknowns for c in result.candidates)
    assert case.brain_mask is None
    with pytest.raises(ValueError, match="provenance"):
        generate_candidate_routes(case, support_mask=original.mri != 0)
    with pytest.raises(TypeError):
        result.access_support["method"] = "verified cortex"


def test_route_record_rejects_fabricated_results_and_mutable_metrics():
    route = single_route().candidates[0]
    with pytest.raises(ValueError, match="clinical deficit"):
        replace(route, clinical_deficit_probability=0.2)
    with pytest.raises(ValueError, match="simulated removal"):
        replace(route, simulated_removed_target_volume_mm3=20.0)
    with pytest.raises(ValueError, match="finite"):
        replace(route, structure_contact_volume_mm3={"motor": float("nan")})
    with pytest.raises(TypeError):
        route.accessible_target_volume_mm3_by_compartment["enhancing"] = 10000.0


@pytest.mark.parametrize("name,value", [("targets_per_compartment", 0), ("max_windows", 4), ("window_radius_mm", float("nan")), ("max_surface_step_mm", -1)])
def test_invalid_search_budgets_fail_before_work(name, value):
    with pytest.raises(ValueError):
        SearchConfig(**{name: value})
