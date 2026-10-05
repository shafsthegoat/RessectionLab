"""Cross-contract attacks on frozen assumptions and scientific output semantics.

These are independent regressions discovered during implementation review, rather
than duplicate happy-path checks of each module's own implementation.
"""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.core import CaseData, ContextField, PatientContext, Plan, SourceRef
from resectionlab.evaluation import EvaluationLedger, freeze_candidates
from resectionlab.geometry import (
    AccessWindow, GeometryScene, SphereObstacle, ToolGeometry, ToolPose,
    check_motion, check_pose,
)
from resectionlab.planning import SearchConfig, generate_candidate_routes
from resectionlab.worlds import (
    FrozenDecisionModel, WorldGenerator, WorldGeneratorConfig,
    WorldPartitionManifest, WorldRole, generate_partitions, physical_proximity_halo,
)


@pytest.mark.parametrize("mutable_value", [["CC-BY-4.0"], {"license": "CC-BY-4.0"}])
def test_source_license_cannot_smuggle_mutation_into_cached_case_hash(mutable_value):
    # A frozen SourceRef is retained by CaseData. Nested mutable license values
    # would let its manifest change after CaseData has cached its semantic hash.
    with pytest.raises((TypeError, ValueError)):
        SourceRef("source", "synthetic://adversarial", license=mutable_value)


@pytest.mark.parametrize("mutable_value", [["score"], {"unit": "score"}])
def test_context_unit_cannot_mutate_frozen_patient_context(mutable_value):
    with pytest.raises((TypeError, ValueError)):
        ContextField(
            "baseline_language", 1,
            SourceRef("source", "synthetic://adversarial"),
            available_at=datetime(2026, 10, 3, tzinfo=timezone.utc),
            unit=mutable_value,
        )


@pytest.mark.parametrize("field_name", ["version", "parameter_basis"])
def test_world_generator_string_metadata_cannot_hold_mutable_objects(field_name):
    with pytest.raises((TypeError, ValueError)):
        WorldGeneratorConfig(**{field_name: ["mutable"]})


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -1.0, 0.5])
def test_invalid_halo_mask_is_not_promoted_to_anatomical_evidence(invalid_value):
    structure = np.zeros((3, 3, 3), dtype=float)
    structure[1, 1, 1] = invalid_value
    with pytest.raises((TypeError, ValueError)):
        physical_proximity_halo(structure, np.eye(4), 1.0, source="invalid-fixture")


@pytest.mark.parametrize("array_name", ["tip", "axis", "access_center", "access_normal", "mask", "affine"])
def test_geometry_arrays_cannot_be_unfrozen_after_cached_checks(array_name):
    pose = ToolPose([1, 1, 2], [0, 0, 1])
    access = AccessWindow([1, 1, 0], [0, 0, 1], 1.0)
    scene = GeometryScene(np.zeros((3, 3, 3), bool), np.eye(4))
    arrays = {
        "tip": pose.tip_mm, "axis": pose.axis_unit,
        "access_center": access.center_mm, "access_normal": access.normal_inward,
        "mask": scene.forbidden_mask, "affine": scene.affine,
    }
    with pytest.raises(ValueError):
        arrays[array_name].setflags(write=True)


@pytest.mark.parametrize("x_mm, should_collide", [(2.8, True), (0.7, False)])
def test_rigid_reframing_preserves_anisotropic_whole_tool_clearance(x_mm, should_collide):
    """A rotation/translation of patient AND tool cannot change physical access."""
    forbidden = np.zeros((9, 9, 9), bool)
    forbidden[4, 4, 4] = True
    affine = np.diag([0.7, 1.3, 2.2, 1.0])
    tool = ToolGeometry("affine-test", 0.2, 0.3, 12.0, tip_length_mm=1.0)
    pose = ToolPose([x_mm, 5.2, 17.6], [0, 0, 1])
    original = check_pose(tool, pose, GeometryScene(forbidden, affine))

    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [17, -26, 33], degrees=True).as_matrix()
    transform[:3, 3] = [34.5, -12.25, 19.125]
    moved_pose = ToolPose(
        transform[:3, :3] @ pose.tip_mm + transform[:3, 3],
        transform[:3, :3] @ pose.axis_unit,
    )
    moved = check_pose(tool, moved_pose, GeometryScene(forbidden, transform @ affine))
    assert original.feasible is not should_collide
    assert moved.feasible == original.feasible
    assert moved.clearance_mm == pytest.approx(original.clearance_mm, abs=1e-8)
    np.testing.assert_array_equal(moved.swept_voxel_indices, original.swept_voxel_indices)


def test_sweep_rejects_small_obstacle_between_clear_endpoint_poses():
    affine = np.eye(4)
    affine[:3, 3] = -20
    scene = GeometryScene(
        np.zeros((41, 41, 41), bool), affine,
        sphere_obstacles=(SphereObstacle([0, 0, 1.2345], 0.02),),
    )
    tool = ToolGeometry("sweep-test", 0.1, 0.1, 6.0, tip_length_mm=0.2)
    start = ToolPose([-2, 0, 5], [0, 0, 1])
    end = ToolPose([2, 0, 5], [0, 0, 1])
    assert check_pose(tool, start, scene).feasible
    assert check_pose(tool, end, scene).feasible
    # Even a very coarse requested discretization needs a conservative sweep.
    result = check_motion(tool, start, end, scene, max_surface_step_mm=4.0)
    assert not result.feasible
    assert any(failure.reason == "SPHERE_COLLISION" for failure in result.failures)


def test_postoperative_metadata_changes_do_not_change_primary_world_realizations():
    cutoff = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)
    source = SourceRef("synthetic", "synthetic://time-cutoff", provenance="simulated")
    observed = ContextField("baseline_motor", "recorded", source, available_at=cutoff)
    postoperative = ContextField(
        "resection_idh", "mutant", source, available_at=cutoff + timedelta(days=3)
    )
    original = CaseData(
        "time-cutoff-fixture", np.zeros((3, 3, 3)),
        {"enhancing": np.ones((3, 3, 3), bool)}, np.eye(4), (source,),
        context=PatientContext(cutoff, (observed, postoperative)),
    )
    changed = original.revised(context=PatientContext(
        cutoff, (observed, replace(postoperative, value="wildtype")), version=2
    ))
    assert original.semantic_hash != changed.semantic_hash  # Full trace retains it.
    config = WorldGeneratorConfig(translation_scale_mm=(1, 1, 1))
    first = generate_partitions(original.semantic_hash, config, 7, planning_hash=original.planning_hash)
    second = generate_partitions(changed.semantic_hash, config, 7, planning_hash=changed.planning_hash)
    assert first.optimization.case_hash != second.optimization.case_hash
    for role in ("optimization", "selection", "final_evaluation", "stress"):
        assert getattr(first, role).seeds == getattr(second, role).seeds
    generator = WorldGenerator(config)
    for index in range(len(first.optimization.seeds)):
        np.testing.assert_array_equal(
            generator.sample(first.optimization, index).anatomy_transform_mm,
            generator.sample(second.optimization, index).anatomy_transform_mm,
        )


@pytest.mark.parametrize("second_seeds", [(102, 101), (102, 103)])
def test_repacking_revealed_worlds_cannot_evade_evaluation_ledger(second_seeds):
    config = WorldGeneratorConfig(translation_scale_mm=(1, 1, 1))
    case_hash = "sha256:cutoff-aware-case"
    model = FrozenDecisionModel.create(
        case_hash=case_hash, geometry={}, objectives={}, tools={},
        world_generator=config.to_dict(), action_primitives={},
    )
    optimization = WorldPartitionManifest(WorldRole.OPTIMIZATION, case_hash, config, (1, 2))
    selection = WorldPartitionManifest(WorldRole.SELECTION, case_hash, config, (3, 4))
    first_plan = Plan("route", case_hash, [[0, 0, 0], [0, 0, 1]], "generic")
    improved_after_reveal = replace(first_plan, route_points_mm=[[0, 0, 0], [1, 0, 1]])
    first_freeze = freeze_candidates(
        [first_plan], model, selection, "predeclared selection", optimization_manifest=optimization
    )
    second_freeze = freeze_candidates(
        [improved_after_reveal], model, selection, "predeclared selection", optimization_manifest=optimization
    )
    ledger = EvaluationLedger()
    ledger.claim(WorldPartitionManifest(WorldRole.FINAL_EVALUATION, case_hash, config, (101, 102)), first_freeze)
    with pytest.raises(ValueError, match="REVEALED"):
        ledger.claim(WorldPartitionManifest(WorldRole.FINAL_EVALUATION, case_hash, config, second_seeds), second_freeze)


@pytest.fixture
def route_candidate():
    target = np.zeros((5, 5, 5), bool)
    target[2, 2, 3] = True
    case = CaseData(
        "route-contract-fixture", np.ones(target.shape), {"enhancing": target},
        np.eye(4), (SourceRef("synthetic", "synthetic://route-contract", provenance="simulated"),),
        brain_mask=np.ones(target.shape, bool),
    )
    result = generate_candidate_routes(
        case, windows=(AccessWindow([2, 2, 0], [0, 0, 1], 1.0),),
        tools=(ToolGeometry("contract-tool", 0.1, 0.1, 10.0, tip_length_mm=0.2),),
        config=SearchConfig(targets_per_compartment=1),
    )
    assert result.candidates[0].feasible
    return result.candidates[0]


@pytest.mark.parametrize("field_name", ["clinical_deficit_probability", "simulated_removed_target_volume_mm3"])
def test_route_candidate_export_cannot_bypass_core_null_contract(route_candidate, field_name):
    # Both native route-card serialization and nested Plan.metadata must uphold
    # the same contract as Plan's top-level clinically unavailable fields.
    with pytest.raises((TypeError, ValueError)):
        replace(route_candidate, **{field_name: 0.25})


def test_route_metrics_cannot_change_after_pareto_classification(route_candidate):
    with pytest.raises(TypeError):
        route_candidate.accessible_target_volume_mm3_by_compartment["enhancing"] = 1e12
    with pytest.raises(TypeError):
        route_candidate.structure_contact_volume_mm3["motor"] = 0.0
