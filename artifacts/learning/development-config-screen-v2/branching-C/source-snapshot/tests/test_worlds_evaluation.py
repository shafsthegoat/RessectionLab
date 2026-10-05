"""Analytic geometry, physical distance, uncertainty and held-out contracts."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from resectionlab.evaluation import (
    EvaluationLedger, IndependentGeometryResult, WorldOutcome,
    evaluate_frozen_candidates, freeze_candidates, independent_check_motion,
    independent_check_pose, independent_check_sequence, segment_box_distance_sq, summarize_event,
    upper_tail_cvar, wilson_interval,
)
from resectionlab.geometry import (
    AccessWindow, GeometryScene, SphereObstacle, ToolGeometry, ToolPose,
)
from resectionlab.worlds import (
    FrozenDecisionModel, WorldGenerator, WorldGeneratorConfig,
    WorldPartitionManifest, WorldPartitions, WorldRole,
    anatomical_ensemble_support, generate_partitions, physical_proximity_halo,
    validate_training_partitions,
)


def assumptions(config=None):
    config = config or WorldGeneratorConfig(translation_scale_mm=(1, 2, 3))
    frozen = FrozenDecisionModel.create(case_hash="case-v1", geometry={"version": 1},
        objectives={"normal_tissue_cost": 2}, tools={"shaft_radius_mm": 1},
        world_generator=config, action_primitives=["REMOVE", "STOP"])
    partitions = generate_partitions("case-v1", config, 1234,
        optimization=4, selection=3, final_evaluation=6, stress=4)
    return frozen, partitions


def candidate(identifier="route-1", digest="geometry-v1", **kwargs):
    return SimpleNamespace(plan_id=identifier, semantic_hash=digest,
                           case_hash="case-v1", plan_type="route_only", **kwargs)


def frozen_candidates(config=None):
    model, partitions = assumptions(config)
    plans = [candidate()]
    freeze = freeze_candidates(plans, model, partitions.selection,
                               "prespecified nondominated candidates; retained STOP", 
                               optimization_manifest=partitions.optimization)
    return plans, freeze, model, partitions


def evaluate(plans, freeze, model, partition, **kwargs):
    return evaluate_frozen_candidates(plans, freeze, model, partition,
        event_definitions={"motor_structure_contact": "Full swept tool envelope intersects supplied motor structure"},
        cost_units={"motor_cost": "declared surrogate units"},
        world_evaluator=kwargs.pop("world_evaluator", lambda plan, world: WorldOutcome(
            {"motor_structure_contact": False}, {"motor_cost": 0.0})),
        geometry_checker=kwargs.pop("geometry_checker", lambda plan: IndependentGeometryResult(True)),
        ledger=kwargs.pop("ledger", EvaluationLedger()), **kwargs)


def test_frozen_model_copies_mutable_inputs_and_detects_changes():
    objective = {"motor": [1, 2]}
    config = WorldGeneratorConfig()
    model = FrozenDecisionModel.create(case_hash="case", geometry="g", objectives=objective,
        tools="t", world_generator=config, action_primitives="a")
    original = model.fingerprint
    objective["motor"][0] = 0
    assert model.fingerprint == original
    new = FrozenDecisionModel.create(case_hash="case", geometry="g", objectives=objective,
        tools="t", world_generator=config, action_primitives="a")
    with pytest.raises(ValueError, match="DECISION_MODEL_CHANGED"):
        model.assert_matches(new)
    with pytest.raises(FrozenInstanceError):
        model.case_hash = "altered"


def test_world_partitions_reproducible_disjoint_and_typed():
    model, worlds = assumptions()
    _, repeated = assumptions()
    assert worlds.to_dict() == repeated.to_dict()
    all_seeds = [seed for m in (worlds.optimization, worlds.selection, worlds.final_evaluation, worlds.stress) for seed in m.seeds]
    assert len(set(all_seeds)) == len(all_seeds)
    validate_training_partitions(worlds.optimization, worlds.selection, model)
    with pytest.raises(ValueError, match="only"):
        validate_training_partitions(worlds.optimization, worlds.final_evaluation)
    with pytest.raises(ValueError, match="overlap"):
        WorldPartitions(worlds.optimization,
            replace(worlds.selection, seeds=worlds.optimization.seeds),
            worlds.final_evaluation, worlds.stress)
    with pytest.raises(ValueError, match="withheld"):
        WorldPartitions(worlds.optimization, worlds.selection, worlds.final_evaluation,
                        replace(worlds.stress, generator=worlds.optimization.generator))


def test_seed_identity_uses_permitted_planning_inputs():
    config = WorldGeneratorConfig(translation_scale_mm=(1, 0, 0))
    first = generate_partitions("full-context-v1", config, 24, planning_hash="permitted-identical")
    future_change = generate_partitions("full-context-v2", config, 24, planning_hash="permitted-identical")
    assert first.optimization.seeds == future_change.optimization.seeds
    assert first.optimization.partition_hash != future_change.optimization.partition_hash
    generator = WorldGenerator(config)
    np.testing.assert_array_equal(generator.sample(first.optimization, 0).anatomy_transform_mm,
                                  generator.sample(future_change.optimization, 0).anatomy_transform_mm)


def test_hidden_world_is_episode_coherent_rigid_and_absent_from_actor_input():
    config = WorldGeneratorConfig(translation_scale_mm=(2, 2, 2), rotation_scale_deg=(2, 3, 4),
                                  rotation_center_mm=(10, 20, 30))
    worlds = generate_partitions("case", config, 444)
    generator = WorldGenerator(config)
    first = generator.sample(worlds.optimization, 0)
    second = generator.sample(worlds.optimization, 1)
    points = np.array([[0, 0, 0], [1, 2, 3], [-4, 5, 6]], float)
    transformed = first.transform_points(points)
    np.testing.assert_allclose(np.linalg.norm(transformed[1:] - transformed[0], axis=1),
                               np.linalg.norm(points[1:] - points[0], axis=1))
    np.testing.assert_array_equal(first.transform_points(points), transformed)
    np.testing.assert_array_equal(generator.sample_seed(worlds.optimization.seeds[0]).anatomy_transform_mm,
                                  first.anatomy_transform_mm)
    assert not np.allclose(first.anatomy_transform_mm, second.anatomy_transform_mm)
    assert first.actor_observation() == second.actor_observation()
    assert set(first.actor_observation()) == {"world_model_version", "uncertainty_mode"}
    with pytest.raises(ValueError):
        first.anatomy_transform_mm.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        generator.config = WorldGeneratorConfig()


@pytest.mark.parametrize("spacing", [(1, 1, 1), (0.5, 2, 4), (2, 3, 0.7)])
def test_h0_distances_match_physical_analytic_point(spacing):
    mask = np.zeros((9, 9, 9), bool)
    mask[4, 4, 4] = True
    affine = np.diag([*spacing, 1.0])
    # Rotate the patient frame; physical distance must not change.
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    affine[:3, :3] = rotation @ affine[:3, :3]
    halo = physical_proximity_halo(mask, affine, 2.0, source="analytic point fixture", qc_state="passed")
    grid = np.moveaxis(np.indices(mask.shape), 0, -1)
    distances = np.linalg.norm((grid - 4) * np.array(spacing), axis=-1)
    np.testing.assert_allclose(halo.distance_mm, distances)
    np.testing.assert_allclose(halo.values, np.exp(-distances**2 / 8))
    assert halo.distance_mm[4, 4, 4] == 0


def test_halo_missing_reconstruction_and_partial_coverage_stay_unknown():
    mask = np.zeros((4, 4, 4), bool)
    unknown = physical_proximity_halo(mask, np.eye(4), 2, source="failed tract reconstruction")
    assert np.isnan(unknown.values).all()
    assert not unknown.known_coverage.any()
    mask[1, 1, 1] = True
    coverage = np.ones_like(mask)
    coverage[2:] = False
    partial = physical_proximity_halo(mask, np.eye(4), 2, source="estimated tract", coverage_mask=coverage)
    assert np.isnan(partial.values[2:]).all()
    assert partial.values[1, 1, 1] == 1
    shear = np.eye(4)
    shear[0, 1] = 0.3
    with pytest.raises(ValueError, match="shear"):
        physical_proximity_halo(mask, shear, 2, source="sheared")


def test_ensemble_support_preserves_missing_world_coverage():
    first = np.zeros((3, 3, 3), bool)
    first[1, 1, 1] = True
    second = np.zeros_like(first)
    ensemble = anatomical_ensemble_support([first, second], source="two supplied structural realizations")
    assert ensemble.support[1, 1, 1] == 0.5
    missing = anatomical_ensemble_support([first, None], source="one failed realization")
    assert np.isnan(missing.support).all()
    assert missing.assessed_count[1, 1, 1] == 1
    assert missing.total_worlds == 2
    all_missing = anatomical_ensemble_support([None], source="missing reconstruction", shape=first.shape)
    assert np.isnan(all_missing.support).all()


def test_event_counts_monte_carlo_error_and_unknowns():
    _, worlds = assumptions()
    summary = summarize_event("motor_structure_contact", "tool envelope intersects motor structure",
                              [True, False, False, True, False, False], worlds.final_evaluation)
    assert summary.numerator == 2 and summary.denominator == 6
    assert summary.model_conditioned_frequency == pytest.approx(1 / 3)
    assert summary.monte_carlo_interval[0] < 1 / 3 < summary.monte_carlo_interval[1]
    assert summary.clinical_deficit_probability is None
    unknown = summarize_event("motor_structure_contact", "same defined contact event",
                              [None] * 6, worlds.final_evaluation)
    assert unknown.model_conditioned_frequency is None
    assert unknown.unknown_worlds == 6 and unknown.monte_carlo_interval is None
    with pytest.raises(ValueError, match="boolean"):
        summarize_event("motor_structure_contact", "contact", [0.1] * 6, worlds.final_evaluation)


def test_deterministic_replays_do_not_fabricate_uncertainty_validation():
    _, worlds = assumptions(WorldGeneratorConfig())
    summary = summarize_event("contact", "known capsule-cell intersection", [False] * 6, worlds.final_evaluation)
    assert summary.model_conditioned_frequency == 0
    assert summary.monte_carlo_interval is None
    assert "deterministic" in summary.interpretation


def test_wilson_boundary_counts_and_fractional_cvar():
    lo, hi = wilson_interval(0, 10)
    assert lo == pytest.approx(0, abs=1e-15)
    assert hi == pytest.approx(0.2775328, abs=1e-6)
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    assert upper_tail_cvar([0, 1, 2, 3], 0.625) == pytest.approx((3 + 0.5 * 2) / 1.5)
    assert upper_tail_cvar([0, 0, 0, 10], 0.5) == 5
    assert upper_tail_cvar([0, 1, 2, 3], 0) == 1.5


def test_final_evaluation_requires_frozen_set_and_isolation():
    plans, freeze, model, worlds = frozen_candidates()
    result = evaluate(plans, freeze, model, worlds.final_evaluation)
    assert result["candidates"][0]["clinical_deficit_probability"] is None
    assert result["candidate_freeze_hash"] == freeze.fingerprint
    with pytest.raises(ValueError, match="changed"):
        evaluate([candidate(digest="changed")], freeze, model, worlds.final_evaluation)
    with pytest.raises(ValueError, match="only"):
        evaluate(plans, freeze, model, worlds.selection)
    overlapping = replace(worlds.final_evaluation, seeds=worlds.optimization.seeds)
    with pytest.raises(ValueError, match="overlap"):
        evaluate(plans, freeze, model, overlapping)


def test_final_ledger_rejects_revealed_worlds_after_candidate_change_and_reorder():
    plans, freeze, model, worlds = frozen_candidates()
    ledger = EvaluationLedger()
    evaluate(plans, freeze, model, worlds.final_evaluation, ledger=ledger)
    # An exact replay is allowed for reproducibility.
    evaluate(plans, freeze, model, worlds.final_evaluation, ledger=ledger)
    revised = [candidate(digest="v2")]
    revised_freeze = freeze_candidates(revised, model, worlds.selection, "same predeclared rule",
                                      optimization_manifest=worlds.optimization)
    reordered = replace(worlds.final_evaluation, seeds=tuple(reversed(worlds.final_evaluation.seeds)))
    with pytest.raises(ValueError, match="REVEALED"):
        evaluate(revised, revised_freeze, model, reordered, ledger=ledger)


def test_failed_geometry_is_retained_and_route_removal_is_rejected():
    plans, freeze, model, worlds = frozen_candidates()
    result = evaluate(plans, freeze, model, worlds.final_evaluation,
        geometry_checker=lambda p: IndependentGeometryResult(False, ("shaft_collision",)))
    assert result["candidates"][0]["status"] == "rejected_geometry"
    assert result["candidates"][0]["model_events"] == []
    with pytest.raises(ValueError, match="accessibility"):
        evaluate(plans, freeze, model, worlds.final_evaluation,
            world_evaluator=lambda p, w: WorldOutcome({"motor_structure_contact": False},
                                                      simulated_removed_target_volume_mm3=100))


def test_segment_box_analytic_and_independent_numerical_comparison():
    lower, upper = np.array([-1, -1, -1]), np.array([1, 1, 1])
    assert segment_box_distance_sq([-5, 0, 0], [5, 0, 0], lower, upper) == 0
    assert segment_box_distance_sq([-5, 2, 2], [5, 2, 2], lower, upper) == pytest.approx(2)
    assert segment_box_distance_sq([2, 2, 2], [2, 2, 2], lower, upper) == 3
    rng = np.random.default_rng(512)
    for _ in range(50):
        start, end = rng.uniform(-5, 5, (2, 3))
        def objective(t):
            point = start + t * (end - start)
            difference = point - np.clip(point, lower, upper)
            return np.dot(difference, difference)
        reference = minimize_scalar(objective, bounds=(0, 1), method="bounded", options={"xatol": 1e-12})
        expected = min(objective(0), objective(1), reference.fun)
        assert segment_box_distance_sq(start, end, lower, upper) == pytest.approx(expected, abs=1e-7)


def scene_with_voxel(index, spacing=(1, 1, 1)):
    mask = np.zeros((24, 24, 24), bool)
    mask[index] = True
    return GeometryScene(mask, np.diag([*spacing, 1.0]))


def test_independent_checker_rejects_shaft_when_tip_is_clear():
    scene = scene_with_voxel((10, 12, 10))
    small = ToolGeometry("small", 0.2, 0.3, 10, tip_length_mm=1)
    large = ToolGeometry("wide_shaft", 0.2, 1.6, 10, tip_length_mm=1)
    pose = ToolPose((15, 10, 10), (1, 0, 0))
    assert independent_check_pose(small, pose, scene).feasible
    rejected = independent_check_pose(large, pose, scene)
    assert not rejected.feasible
    assert rejected.failures == ("shaft_envelope_collision",)


def test_voxel_cells_are_volumes_and_anisotropy_is_physical():
    scene = scene_with_voxel((5, 5, 5), (2, 3, 4))
    tool = ToolGeometry("tip", 0.25, 0.25, 2, tip_length_mm=0.5)
    # Obstacle spans physical z=18..22; center-only collision would miss z=21.9.
    pose = ToolPose((10, 15, 21.9), (1, 0, 0))
    assert not independent_check_pose(tool, pose, scene).feasible


def test_sweep_collisions_between_free_endpoints_are_rejected():
    scene = scene_with_voxel((10, 10, 10))
    tool = ToolGeometry("fixture", 0.2, 0.2, 4, tip_length_mm=1)
    start = ToolPose((12, 8, 10), (1, 0, 0))
    end = ToolPose((12, 12, 10), (1, 0, 0))
    assert independent_check_pose(tool, start, scene).feasible
    assert independent_check_pose(tool, end, scene).feasible
    result = independent_check_motion(tool, start, end, scene)
    assert not result.feasible
    assert result.failures[0].startswith("swept_")


def test_rotating_swept_shaft_hits_obstacle_with_clear_endpoint_poses():
    mask = np.zeros((30, 30, 30), bool)
    scene = GeometryScene(mask, np.eye(4), sphere_obstacles=(SphereObstacle((10, 10, 15), 0.3),))
    tool = ToolGeometry("fixture", 0.2, 0.2, 10, tip_length_mm=1)
    tip = np.array([15, 15, 15])
    start = ToolPose(tip, (1, 0, 0))
    end = ToolPose(tip, (0, 1, 0))
    assert independent_check_pose(tool, start, scene).feasible
    assert independent_check_pose(tool, end, scene).feasible
    assert not independent_check_motion(tool, start, end, scene).feasible


def test_nested_tool_envelopes_never_lose_known_contact():
    scene = scene_with_voxel((10, 12, 10))
    pose = ToolPose((15, 10, 10), (1, 0, 0))
    outcomes = [not independent_check_pose(ToolGeometry("tool", radius, radius, 10, tip_length_mm=1), pose, scene).feasible
                for radius in (0.2, 0.8, 1.5, 2.0, 3.0)]
    assert outcomes == sorted(outcomes)
    assert outcomes[0] is False and outcomes[-1] is True


def test_access_window_and_missing_proximal_coverage_are_explicit():
    scene = GeometryScene(np.zeros((24, 24, 24), bool), np.eye(4))
    tool = ToolGeometry("fixture", 0.2, 1.0, 30, tip_length_mm=1)
    pose = ToolPose((12, 12, 12), (1, 0, 0))
    result = independent_check_pose(tool, pose, scene)
    assert result.feasible
    assert "proximal_tool_outside_image_coverage_unassessed" in result.unknowns
    access = AccessWindow((0, 12, 12), (1, 0, 0), 0.9)
    assert not independent_check_pose(tool, pose, scene, access).feasible


def test_independent_sequence_checks_real_replay_accounting_and_termination():
    from resectionlab.simulation import greedy_search, make_synthetic_simulator

    sequence = greedy_search(make_synthetic_simulator()).actions
    checked = independent_check_sequence(make_synthetic_simulator, sequence)
    assert checked.feasible
    assert "native_resolution_removal_not_independently_verified" in checked.unknowns
    assert not independent_check_sequence(make_synthetic_simulator, ("STOP", "STOP")).feasible
    assert not independent_check_sequence(make_synthetic_simulator, ()).feasible
    assert not independent_check_sequence(make_synthetic_simulator, sequence, cancelled=lambda: True).feasible


def test_axial_motion_catches_obstacle_even_after_whole_tool_has_passed():
    scene = scene_with_voxel((12, 10, 10))
    tool = ToolGeometry("fixture", .2, .2, 4, tip_length_mm=1)
    start = ToolPose((5, 10, 10), (1, 0, 0))
    end = ToolPose((20, 10, 10), (1, 0, 0))
    assert independent_check_pose(tool, start, scene).feasible
    assert independent_check_pose(tool, end, scene).feasible
    assert not independent_check_motion(tool, start, end, scene).feasible
