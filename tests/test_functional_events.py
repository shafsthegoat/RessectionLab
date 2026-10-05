"""Analytical controls only; these are not synthetic training patients."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.core import SourceRef
from resectionlab.evaluation import (EvaluationLedger, IndependentGeometryResult,
                                     freeze_candidates, wilson_interval)
from resectionlab.functional_evidence import FunctionalEvidence
from resectionlab.functional_events import (
    AxialToolSweep, FunctionalEventConfig, FunctionalScenarioEvaluator,
    evaluate_functional_candidates, prepare_functional_exposure,
    sweeps_from_native_history, _sample_covered,
)
from resectionlab.geometry import ToolGeometry
from resectionlab.worlds import (FrozenDecisionModel, LatentWorld, WorldGenerator,
                                WorldGeneratorConfig, generate_partitions)


def evidence(motor, *, language=None, coverage=None, affine=None, uncertainty=None):
    source = SourceRef("analytical-control", "test:analytical-control", "a" * 64,
                       provenance="prior", native_frame="MNI")
    return FunctionalEvidence("analytical-control", motor, language,
        None if motor is None else (np.ones(motor.shape, bool) if coverage is None else coverage),
        None if language is None else (np.ones(language.shape, bool) if coverage is None else coverage),
        np.eye(4) if affine is None else affine, "sha256:" + "b" * 64, "sha256:" + "c" * 64,
        {"control": {"source": source.to_dict(), "mni_ras_to_patient_ras_mm": np.eye(4).tolist()}},
        uncertainty or WorldGeneratorConfig())


def world(translation=(0, 0, 0)):
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return LatentWorld("analytical-control", "control-v1", matrix.tobytes(), not any(translation))


def footprint(shape=(5, 5, 5), *, radius=.1, affine=None, tissue=None, sweeps=None):
    tissue = np.ones(shape, bool) if tissue is None else tissue
    tool = ToolGeometry("control-tool", .1, radius, 6., tip_length_mm=.2)
    sweeps = [AxialToolSweep(tool, (2., 2., -.5), (2., 2., 3.), (0., 0., 1.))] if sweeps is None else sweeps
    return prepare_functional_exposure(sweeps, tissue_mask=tissue,
        affine_ras_mm=np.eye(4) if affine is None else affine,
        candidate_hash="plan-v1", case_hash="case-v1")


def test_full_shaft_changes_event_while_tip_and_axis_are_identical():
    values = np.zeros((5, 5, 5))
    values[3, 2, 1] = 1.
    evaluator = FunctionalScenarioEvaluator(evidence(values))
    thin, wide = footprint(radius=.1), footprint(radius=.6)
    assert evaluator.world_outcome(thin, world()).events["motor_supplied_map_contact"] is False
    assert evaluator.world_outcome(wide, world()).events["motor_supplied_map_contact"] is True
    assert set(map(tuple, thin.indices)) < set(map(tuple, wide.indices))


def test_source_cell_intersection_not_tip_or_voxel_centre_only():
    tool = ToolGeometry("side-contact", .05, .05, 6., tip_length_mm=.2)
    sweep = AxialToolSweep(tool, (2.49, 2., -.5), (2.49, 2., 3.), (0., 0., 1.))
    result = footprint(sweeps=[sweep])
    assert (3, 2, 1) in set(map(tuple, result.indices))


def test_duplicate_strokes_do_not_multiply_contact_surrogate():
    tool = ToolGeometry("control", .1, .1, 6., tip_length_mm=.2)
    sweep = AxialToolSweep(tool, (2., 2., -.5), (2., 2., 3.), (0., 0., 1.))
    single, duplicate = footprint(sweeps=[sweep]), footprint(sweeps=[sweep, sweep])
    np.testing.assert_array_equal(single.indices, duplicate.indices)
    evaluator = FunctionalScenarioEvaluator(evidence(np.ones((5, 5, 5))))
    assert evaluator.world_outcome(single, world()).costs == evaluator.world_outcome(duplicate, world()).costs


def test_unknown_zero_is_not_negative_event_but_known_hit_remains_true():
    values = np.zeros((5, 5, 5))
    coverage = np.ones(values.shape, bool)
    coverage[2, 2, 1] = False
    result = FunctionalScenarioEvaluator(evidence(values, coverage=coverage)).world_outcome(footprint(), world())
    assert result.events["motor_supplied_map_contact"] is None
    assert result.costs["motor_contact_surrogate"] is None
    assert result.costs["motor_known_contact_lower_bound"] == 0
    assert result.costs["motor_unassessed_contact_volume"] == 1
    values[2, 2, 2] = 1.
    result = FunctionalScenarioEvaluator(evidence(values, coverage=coverage)).world_outcome(footprint(), world())
    assert result.events["motor_supplied_map_contact"] is True
    assert result.costs["motor_contact_surrogate"] is None


def test_absent_component_is_unknown_and_stop_has_no_contact():
    evaluator = FunctionalScenarioEvaluator(evidence(np.zeros((5, 5, 5))))
    assert evaluator.world_outcome(footprint(), world()).events["language_supplied_map_contact"] is None
    stopped = evaluator.world_outcome(footprint(sweeps=[]), world())
    assert all(v is False for v in stopped.events.values())
    assert all(v == 0 for v in stopped.costs.values())


def test_coverage_requires_every_nonzero_linear_interpolation_contributor():
    values = np.ones((3, 3, 3))
    coverage = np.ones(values.shape, bool)
    coverage[2, 1, 1] = False
    values[2, 1, 1] = 0
    sampled, known = _sample_covered(values, coverage, np.array([[1., 1., 1.], [1.1, 1., 1.]]))
    np.testing.assert_array_equal(known, [True, False])
    assert sampled[0] == 1 and np.isnan(sampled[1])


def test_world_moves_all_components_together_and_changes_real_support_encounter():
    values = np.zeros((5, 5, 5))
    values[3, 2, 1] = 1.
    evaluator = FunctionalScenarioEvaluator(evidence(values, language=values))
    nominal = evaluator.world_outcome(footprint(), world())
    shifted = evaluator.world_outcome(footprint(), world((-1., 0, 0)))
    assert all(v is False for v in nominal.events.values())
    assert all(v is True for v in shifted.events.values())
    assert shifted.costs["motor_contact_surrogate"] == shifted.costs["language_contact_surrogate"]


def test_external_shaft_is_not_missing_anatomical_coverage():
    # Most of this shaft lies outside the image, but its outside portion is air.
    result = FunctionalScenarioEvaluator(evidence(np.zeros((5, 5, 5)))).world_outcome(footprint(), world())
    assert result.events["motor_supplied_map_contact"] is False
    assert result.costs["motor_unassessed_contact_volume"] == 0.


@pytest.mark.parametrize("reflect", [False, True])
def test_oblique_reflected_physical_grid_keeps_whole_tool_contacts(reflect):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [11, 23, 7], degrees=True).as_matrix()
    if reflect:
        matrix[:3, 0] *= -1
    matrix[:3, 3] = (30, -9, 21)
    tool = ToolGeometry("oblique", .1, .1, 6., tip_length_mm=.2)
    move = lambda p: matrix[:3, :3] @ np.asarray(p) + matrix[:3, 3]
    sweep = AxialToolSweep(tool, tuple(move((2, 2, -.5))), tuple(move((2, 2, 3))), tuple(matrix[:3, 2]))
    changed = footprint(affine=matrix, sweeps=[sweep])
    np.testing.assert_array_equal(changed.indices, footprint().indices)
    result = FunctionalScenarioEvaluator(evidence(np.ones((5, 5, 5)), affine=matrix)).world_outcome(changed, world())
    assert result.events["motor_supplied_map_contact"] is True
    assert result.costs["motor_unassessed_contact_volume"] == 0.


def test_native_microsteps_collapse_to_one_exact_axial_sweep_and_reject_jump():
    tool = ToolGeometry("native-control", .1, .1, 6., tip_length_mm=.2)
    history = [{"tool_id": tool.tool_id, "entry_mm": [2, 2, -.5], "tip_mm": [2, 2, 3],
        "axis_unit": [0, 0, 1], "microsteps": [
            {"tip_start_mm": [2, 2, -.5], "tip_end_mm": [2, 2, 1]},
            {"tip_start_mm": [2, 2, 1], "tip_end_mm": [2, 2, 3]}]}]
    sweeps = sweeps_from_native_history(history, [tool])
    assert len(sweeps) == 1
    np.testing.assert_array_equal(footprint(sweeps=sweeps).indices, footprint().indices)
    history[0]["microsteps"][1]["tip_start_mm"] = [2, 2, 2]
    with pytest.raises(ValueError, match="discontinuous"):
        sweeps_from_native_history(history, [tool])


def test_nonaxial_motion_fails_instead_of_tip_only_approximation():
    tool = ToolGeometry("nonaxial", .1, .1, 6., tip_length_mm=.2)
    with pytest.raises(ValueError, match="axial"):
        AxialToolSweep(tool, (0, 0, 0), (1, 0, 1), (0, 0, 1))


def frozen_fixture(*, deterministic=False, missing=False):
    generator = WorldGeneratorConfig(family="rigid_uniform", translation_scale_mm=(0 if deterministic else .75, 0, 0))
    values = np.zeros((5, 5, 5))
    values[3, 2, 1] = 1
    coverage = np.ones(values.shape, bool)
    if missing:
        coverage[:, :, 2] = False
    item = evidence(values, language=values, coverage=coverage, uncertainty=generator)
    exposure = footprint()
    config = FunctionalEventConfig()
    model = FrozenDecisionModel.create(case_hash="case-v1",
        geometry={"functional_evidence_hash": item.fingerprint, "functional_event_config": config.to_dict(),
                  "tissue_support_hash": exposure.tissue_support_hash,
                  "functional_footprint_hashes": {"plan": exposure.fingerprint}}, objectives="fixed",
        tools="fixed", world_generator=generator, action_primitives="axial-sweep")
    partitions = generate_partitions("case-v1", generator, 143, optimization=2, selection=2,
                                     final_evaluation=64, stress=4)
    candidates = [SimpleNamespace(plan_id="plan", semantic_hash="plan-v1", case_hash="case-v1", plan_type="route_only")]
    freeze = freeze_candidates(candidates, model, partitions.selection, "fixed before evaluation",
                               optimization_manifest=partitions.optimization)
    kwargs = dict(evidence=item, footprints={"plan": exposure},
                  geometry_checker=lambda c: IndependentGeometryResult(True), ledger=EvaluationLedger(), config=config)
    return candidates, freeze, model, partitions.final_evaluation, kwargs


def test_report_real_nonidentical_worlds_counts_intervals_and_costs():
    candidates, freeze, model, partition, kwargs = frozen_fixture()
    report = evaluate_functional_candidates(candidates, freeze, model, partition, **kwargs)
    row = report["candidates"][0]
    event = row["model_events"][0]
    outcomes = [r["events"]["motor_supplied_map_contact"] for r in row["world_outcomes"]]
    assert report["distinct_anatomy_transforms"] == 64
    assert report["nonidentical_worlds_verified"]
    assert event["numerator"] == sum(v is True for v in outcomes)
    assert 0 < event["numerator"] < event["denominator"] == 64
    assert event["monte_carlo_interval"] == wilson_interval(event["numerator"], 64)
    costs = row["surrogate_costs"]["motor_contact_surrogate"]
    assert costs["upper_tail_cvar"] >= costs["mean"] > 0
    assert report["clinical_deficit_probability"] is None
    assert report["patient_function_assessed"] is False


def test_partial_coverage_preserves_total_denominator_and_suppresses_frequency():
    candidates, freeze, model, partition, kwargs = frozen_fixture(missing=True)
    report = evaluate_functional_candidates(candidates, freeze, model, partition, **kwargs)
    event = report["candidates"][0]["model_events"][0]
    assert event["denominator"] == 64
    assert event["unknown_worlds"] > 0
    assert event["model_conditioned_frequency"] is None and event["monte_carlo_interval"] is None
    costs = report["candidates"][0]["surrogate_costs"]
    assert costs["motor_contact_surrogate"]["mean"] is None
    assert costs["motor_unassessed_contact_volume"]["mean"] > 0


def test_different_seeds_of_zero_perturbations_cannot_claim_robustness():
    candidates, freeze, model, partition, kwargs = frozen_fixture(deterministic=True)
    report = evaluate_functional_candidates(candidates, freeze, model, partition, **kwargs)
    assert report["distinct_anatomy_transforms"] == 1 and not report["nonidentical_worlds_verified"]
    assert all(event["monte_carlo_interval"] is None for event in report["candidates"][0]["model_events"])


def test_geometry_rejection_kept_without_functional_safety_numbers():
    candidates, freeze, model, partition, kwargs = frozen_fixture()
    kwargs["geometry_checker"] = lambda c: IndependentGeometryResult(False, ("shaft_collision",))
    report = evaluate_functional_candidates(candidates, freeze, model, partition, **kwargs)
    assert report["candidates"][0]["status"] == "rejected_geometry"
    assert report["candidates"][0]["model_events"] == []


def test_changed_evidence_or_stale_footprint_rejected_after_freeze():
    candidates, freeze, model, partition, kwargs = frozen_fixture()
    kwargs["evidence"] = replace(kwargs["evidence"], motor=np.zeros((5, 5, 5)))
    with pytest.raises(ValueError, match="bind exact functional"):
        evaluate_functional_candidates(candidates, freeze, model, partition, **kwargs)
    candidates, freeze, model, partition, kwargs = frozen_fixture()
    kwargs["footprints"] = {"plan": replace(kwargs["footprints"]["plan"], candidate_hash="other")}
    with pytest.raises(ValueError, match="footprint content"):
        evaluate_functional_candidates(candidates, freeze, model, partition, **kwargs)
