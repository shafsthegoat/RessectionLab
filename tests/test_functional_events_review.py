"""Independent analytic review controls; no patient data or training fixtures.

Expected contacts below come from physical cell faces, linear functions, and
rigid point transforms, independently of the planner's contact rasterizer.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.core import SourceRef
from resectionlab.evaluation import EvaluationLedger, IndependentGeometryResult, freeze_candidates
from resectionlab.functional_evidence import FunctionalEvidence
from resectionlab.functional_events import (
    AxialToolSweep, FunctionalEventConfig, FunctionalScenarioEvaluator,
    _sample_covered, evaluate_functional_candidates, prepare_functional_exposure,
)
from resectionlab.geometry import ToolGeometry
from resectionlab.worlds import (
    FrozenDecisionModel, LatentWorld, WorldGenerator, WorldGeneratorConfig, generate_partitions,
)


def _evidence(values, *, affine=None, coverage=None, language=None, generator=None):
    source = SourceRef("independent-analytic-review", "test:independent-analytic-review",
                       "a" * 64, provenance="prior", native_frame="MNI")
    return FunctionalEvidence(
        "independent-analytic-review", values, language,
        np.ones(values.shape, bool) if coverage is None else coverage,
        None if language is None else np.ones(language.shape, bool),
        np.eye(4) if affine is None else affine, "sha256:" + "b" * 64,
        "sha256:" + "c" * 64,
        {"source": {"source": source.to_dict(),
                    "mni_ras_to_patient_ras_mm": np.eye(4).tolist()}},
        generator or WorldGeneratorConfig())


def _world(matrix=None):
    transform = np.eye(4) if matrix is None else np.asarray(matrix, dtype=np.float64)
    return LatentWorld("independent-analytic-world", "review-v1", transform.tobytes(),
                       bool(np.array_equal(transform, np.eye(4))))


def _exposure(sweeps, tissue, affine=None):
    return prepare_functional_exposure(
        sweeps, tissue_mask=tissue, affine_ras_mm=np.eye(4) if affine is None else affine,
        candidate_hash="review-route-hash", case_hash="review-case-hash")


@pytest.mark.parametrize("component", ["distal_tip_cap", "proximal_shaft_cap"])
@pytest.mark.parametrize("reflected", [False, True])
def test_anisotropic_oblique_cap_tangency_and_volume_units(component, reflected):
    # Native cell (1,1,1) has physical local bounds x=[1,3], y=[1.5,4.5],
    # z=[2,6]. Its volume is 24 mm3, independently of rotation/reflection.
    rotation = Rotation.from_euler("xyz", [31, -18, 9], degrees=True).as_matrix()
    if reflected:
        rotation[:, 0] *= -1
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([2., 3., 4.])
    affine[:3, 3] = [52., -37., 11.]
    tissue = np.zeros((3, 3, 3), bool)
    tissue[1, 1, 1] = True
    tool = ToolGeometry("analytic-caps", .5, .5, 2., tip_length_mm=.2)
    tip_z = 1.5 if component == "distal_tip_cap" else 8.5
    outward = -1 if component == "distal_tip_cap" else 1

    def stationary(z):
        tip = rotation @ np.array([2., 3., z]) + affine[:3, 3]
        return AxialToolSweep(tool, tuple(tip), tuple(tip), tuple(rotation[:, 2]))

    tangent = _exposure([stationary(tip_z)], tissue, affine)
    separated = _exposure([stationary(tip_z + outward * 1e-4)], tissue, affine)
    np.testing.assert_array_equal(tangent.indices, [[1, 1, 1]])
    assert len(separated.indices) == 0
    values = np.full(tissue.shape, .25)
    result = FunctionalScenarioEvaluator(_evidence(values, affine=affine)).world_outcome(tangent, _world())
    assert result.costs["motor_contact_surrogate"] == pytest.approx(6.)
    assert result.costs["motor_unassessed_contact_volume"] == 0
    assert result.events["motor_supplied_map_contact"] is False


def test_axial_retraction_and_adjacent_subsweeps_have_identical_physical_union():
    tissue = np.ones((9, 9, 9), bool)
    axis = np.array([1., 1., 1.]) / np.sqrt(3.)
    start, middle, end = (np.array([1., 1., 1.]) + distance * axis for distance in (0., 2., 5.))
    tool = ToolGeometry("diagonal", .17, .63, 4., tip_length_mm=.9)

    def sweep(a, b):
        return AxialToolSweep(tool, tuple(a), tuple(b), tuple(axis))

    forward = _exposure([sweep(start, end)], tissue)
    backward = _exposure([sweep(end, start)], tissue)
    split = _exposure([sweep(start, middle), sweep(middle, end)], tissue)
    np.testing.assert_array_equal(forward.indices, backward.indices)
    np.testing.assert_array_equal(forward.indices, split.indices)


def test_linear_interpolation_has_exact_weights_and_no_probability_sum():
    xyz = np.indices((4, 4, 4), dtype=float)
    values = .1 * xyz[0] + .2 * xyz[1] + .025 * xyz[2]
    point = np.array([[1.25, 1.5, .5]])
    sampled, known = _sample_covered(values, np.ones(values.shape, bool), point)
    assert known.tolist() == [True]
    assert sampled[0] == pytest.approx(.4375)
    # Twelve touched cells each have map support .25. The event remains false
    # at threshold .5 even though their summed surrogate is greater than one.
    tissue = np.zeros((3, 3, 12), bool)
    tissue[1, 1, :] = True
    tool = ToolGeometry("many-cells", .1, .1, 16., tip_length_mm=.2)
    exposure = _exposure([AxialToolSweep(tool, (1., 1., 11.), (1., 1., 11.), (0., 0., 1.))], tissue)
    assert len(exposure.indices) == 12
    outcome = FunctionalScenarioEvaluator(_evidence(np.full(tissue.shape, .25))).world_outcome(exposure, _world())
    assert outcome.events["motor_supplied_map_contact"] is False
    assert outcome.costs["motor_contact_surrogate"] == 3.


def test_singleton_axis_boundary_and_uncovered_zero_weight_neighbor():
    values = np.array([[[.2, .8], [.4, 1.]]])
    coverage = np.ones(values.shape, bool)
    coverage[0, 1, 1] = False
    values[0, 1, 1] = 0
    sampled, known = _sample_covered(values, coverage,
        np.array([[0., 0., 1.], [0., .5, 1.], [1e-6, 0., 1.], [-1e-6, 0., 1.]]))
    np.testing.assert_array_equal(known, [True, False, False, False])
    assert sampled[0] == .8
    assert np.isnan(sampled[1:]).all()


def test_nonorigin_world_rotation_uses_inverse_physical_transform_for_both_maps():
    affine = np.diag([2., 3., 4., 1.])
    affine[:3, 3] = [10., -20., 6.]
    values = np.zeros((5, 5, 5))
    values[3, 2, 2] = 1.
    language = values * .75
    # A 180 degree rotation around physical native center (2,2,2) moves
    # source cell (3,2,2) exactly onto target cell (1,2,2).
    center = affine[:3, :3] @ np.array([2., 2., 2.]) + affine[:3, 3]
    rotation = np.diag([-1., -1., 1.])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center - rotation @ center
    target = affine[:3, :3] @ np.array([1., 2., 2.]) + affine[:3, 3]
    tissue = np.zeros(values.shape, bool)
    tissue[1, 2, 2] = True
    tool = ToolGeometry("single-cell", .01, .01, .1, tip_length_mm=.02)
    exposure = _exposure([AxialToolSweep(tool, tuple(target), tuple(target), (0., 0., 1.))], tissue, affine)
    evaluator = FunctionalScenarioEvaluator(_evidence(values, language=language, affine=affine))
    nominal = evaluator.world_outcome(exposure, _world())
    rotated = evaluator.world_outcome(exposure, _world(transform))
    assert all(event is False for event in nominal.events.values())
    assert all(event is True for event in rotated.events.values())
    assert rotated.costs["motor_contact_surrogate"] == pytest.approx(24.)
    assert rotated.costs["language_contact_surrogate"] == pytest.approx(18.)


def _frozen_fixture(*, values=None, config=None, coverage=None):
    generator = WorldGeneratorConfig(family="rigid_uniform", translation_scale_mm=(.1, 0., 0.))
    if values is None:
        values = np.zeros((5, 5, 5))
        values[2, 2, 2] = 1.
    item = _evidence(values, generator=generator, coverage=coverage)
    tissue = np.zeros(values.shape, bool)
    tissue[2, 2, 1:3] = True
    tool = ToolGeometry("frozen-review-tool", .1, .1, 2., tip_length_mm=.2)
    exposure = _exposure([AxialToolSweep(tool, (2., 2., 1.), (2., 2., 2.), (0., 0., 1.))], tissue)
    config = config or FunctionalEventConfig()
    model = FrozenDecisionModel.create(case_hash="review-case-hash",
        geometry={"functional_evidence_hash": item.fingerprint, "functional_event_config": config.to_dict(),
                  "tissue_support_hash": exposure.tissue_support_hash,
                  "functional_footprint_hashes": {"review-plan": exposure.fingerprint}},
        objectives="fixed", tools="fixed", world_generator=generator, action_primitives="axial-sweep")
    partitions = generate_partitions("review-case-hash", generator, 731,
        optimization=2, selection=2, final_evaluation=4, stress=2)
    candidates = [SimpleNamespace(plan_id="review-plan", semantic_hash="review-route-hash",
        case_hash="review-case-hash", plan_type="route_only")]
    freeze = freeze_candidates(candidates, model, partitions.selection, "review fixed candidate",
        optimization_manifest=partitions.optimization)
    kwargs = dict(evidence=item, footprints={"review-plan": exposure},
        geometry_checker=lambda _: IndependentGeometryResult(True), ledger=EvaluationLedger(), config=config)
    return (candidates, freeze, model, partitions.final_evaluation), kwargs


def test_empty_exposure_substitution_cannot_relabel_frozen_route_as_contact_free():
    args, kwargs = _frozen_fixture()
    original = kwargs["footprints"]["review-plan"]
    assert len(original.indices) == 2
    kwargs["footprints"] = {"review-plan": replace(original, indices=np.empty((0, 3), dtype=np.int64))}
    with pytest.raises(ValueError, match="(?i)footprint|exposure"):
        evaluate_functional_candidates(*args, **kwargs)


@pytest.mark.filterwarnings("ignore:Setting the strides on a NumPy array has been deprecated:DeprecationWarning")
def test_same_shape_index_stride_mutation_cannot_change_frozen_encounter():
    args, kwargs = _frozen_fixture()
    exposure = kwargs["footprints"]["review-plan"]
    try:
        exposure.indices.strides = (0, 0)
    except (AttributeError, ValueError):
        return  # Blocking mutation at the array boundary is also valid.
    with pytest.raises(ValueError, match="(?i)metadata|layout|footprint|exposure"):
        evaluate_functional_candidates(*args, **kwargs)


def test_frozen_linear_map_report_matches_closed_form_mean_fractional_tail_and_wilson():
    values = .1 * np.indices((5, 5, 5), dtype=float)[0]
    args, kwargs = _frozen_fixture(values=values,
        config=FunctionalEventConfig(motor_threshold=.2, cvar_alpha=.7))
    partition = args[3]
    generator = WorldGenerator(partition.generator)
    shifts = np.array([generator.sample(partition, i).anatomy_transform_mm[0, 3] for i in range(4)])
    # Two unit-volume touched cells; the linear field at inverse-transformed
    # x=2 is .2-.1*translation. No image interpolation helper is used here.
    expected_costs = 2. * (.2 - .1 * shifts)
    count = int(np.count_nonzero(shifts <= 0))
    report = evaluate_functional_candidates(*args, **kwargs)
    row = report["candidates"][0]
    event = next(item for item in row["model_events"] if item["event"] == "motor_supplied_map_contact")
    costs = row["surrogate_costs"]["motor_contact_surrogate"]
    np.testing.assert_allclose([world["costs"]["motor_contact_surrogate"] for world in row["world_outcomes"]],
                               expected_costs, rtol=2e-7)
    assert costs["mean"] == pytest.approx(sum(expected_costs) / 4., rel=2e-7)
    ordered = sorted(expected_costs, reverse=True)
    assert costs["upper_tail_cvar"] == pytest.approx((ordered[0] + .2 * ordered[1]) / 1.2, rel=2e-7)
    assert (event["numerator"], event["denominator"], event["unknown_worlds"]) == (count, 4, 0)
    assert event["model_conditioned_frequency"] == count / 4
    z, probability = 1.959963984540054, count / 4
    center = (probability + z * z / 8) / (1 + z * z / 4)
    half = z * np.sqrt(probability * (1 - probability) / 4 + z * z / 64) / (1 + z * z / 4)
    np.testing.assert_allclose(event["monte_carlo_interval"], [max(0., center - half), min(1., center + half)])


def test_unknown_worlds_keep_full_denominator_and_known_unassessed_volume_statistics():
    values = np.zeros((5, 5, 5))
    coverage = np.ones(values.shape, bool)
    coverage[1, :, :] = False
    args, kwargs = _frozen_fixture(values=values, coverage=coverage,
                                  config=FunctionalEventConfig(cvar_alpha=.7))
    partition = args[3]
    generator = WorldGenerator(partition.generator)
    missing = [generator.sample(partition, i).anatomy_transform_mm[0, 3] > 0 for i in range(4)]
    assert 0 < sum(missing) < 4
    report = evaluate_functional_candidates(*args, **kwargs)
    row = report["candidates"][0]
    event = next(item for item in row["model_events"] if item["event"] == "motor_supplied_map_contact")
    assert (event["numerator"], event["denominator"], event["unknown_worlds"]) == (0, 4, sum(missing))
    assert event["model_conditioned_frequency"] is None
    assert event["monte_carlo_interval"] is None
    assert row["surrogate_costs"]["motor_contact_surrogate"]["mean"] is None
    assert row["surrogate_costs"]["motor_contact_surrogate"]["upper_tail_cvar"] is None
    unassessed = row["surrogate_costs"]["motor_unassessed_contact_volume"]
    assert unassessed["mean"] == 2. * sum(missing) / 4.
    ordered = sorted([2. * value for value in missing], reverse=True)
    assert unassessed["upper_tail_cvar"] == pytest.approx((ordered[0] + .2 * ordered[1]) / 1.2)


def test_callback_cannot_replace_validated_footprint_during_frozen_evaluation():
    # Independently reproduced before repair on functional_events.py SHA-256
    # 6d034dc823f367dd211cf8a11621e84d21dbbfa187f8296ac87bec6c532b2edf:
    # 12 controls passed; this control failed because the report used 0 touched
    # cells after the callback replaced the original two-cell frozen footprint.
    # Keep this regression as evidence of the actual negative result.
    args, kwargs = _frozen_fixture()
    original = kwargs["footprints"]["review-plan"]

    def replace_during_geometry_check(_):
        kwargs["footprints"]["review-plan"] = replace(original, indices=np.empty((0, 3), dtype=np.int64))
        return IndependentGeometryResult(True)

    kwargs["geometry_checker"] = replace_during_geometry_check
    try:
        report = evaluate_functional_candidates(*args, **kwargs)
    except ValueError as error:
        assert any(word in str(error).lower() for word in ("footprint", "exposure", "mutat"))
        return
    # An implementation may snapshot the original immutable mapping or reject
    # the change, but it must never issue a report for the substituted cells.
    row = report["candidates"][0]
    assert row["exposure"]["touched_source_cell_count"] == 2
    event = next(item for item in row["model_events"] if item["event"] == "motor_supplied_map_contact")
    assert event["numerator"] == event["denominator"] == 4
