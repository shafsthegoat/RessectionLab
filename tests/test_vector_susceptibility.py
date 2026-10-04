"""Independent physical and derivative checks for the experimental vector model."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("experimental_vector_susceptibility", SCRIPTS / "experimental_vector_susceptibility.py")
vector = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC.loader.exec_module(vector)
finally:
    sys.path.pop(0)

torch.set_num_threads(2)


def pair_inputs(shape=(16, 18, 14), angle=.8):
    affines = [vector.simulation.make_affine(shape), vector.simulation.make_affine(shape, angle)]
    directions = [vector.simulation.unit_basis(a)[:, 1] * sign for a, sign in zip(affines, [1, -1])]
    world = [vector.simulation.grid_world(shape, a) for a in affines]
    arrays = [100 + p @ np.array([.2, -.1, .15]) for p in world]
    return arrays, affines, directions


def test_exact_identity_has_independent_zero_objective_and_positive_jacobians():
    arrays, affines, directions = pair_inputs(angle=0)
    certificate = vector.exact_identity_certificate(arrays, affines, directions)
    assert certificate["total_objective"] == 0
    assert certificate["minimum_jacobian"] == 1
    assert certificate["optimizer_called"] is False
    assert certificate["anatomical_uniqueness_claim"] is False
    model = vector.PhysicalPair(arrays, affines, directions, maximum_displacement_mm=.5, dtype=torch.float64)
    loss, terms = model.objective(torch.zeros(arrays[0].shape, dtype=torch.float64), .1, .001)
    assert float(loss) < 1e-26
    assert float(terms["smoothness"]) == 0
    assert float(terms["barrier"]) == 0


def test_exact_identity_returns_before_any_numerical_optimizer(monkeypatch):
    arrays, affines, directions = pair_inputs(angle=0)
    monkeypatch.setattr(torch.optim, "LBFGS", lambda *a, **k: pytest.fail("Optimizer called for certified identity"))
    field, corrected, report = vector.optimize_pair(arrays, affines, directions, vector.CONFIG)
    assert np.count_nonzero(field) == 0
    assert all(np.array_equal(first, second) for first, second in zip(arrays, corrected))
    assert report["certificate"]["total_objective"] == 0


@pytest.mark.parametrize("alteration", ["affine", "almost_equal", "constant", "nan"])
def test_identity_certificate_never_ignores_geometry_or_noise(alteration):
    arrays, affines, directions = pair_inputs(angle=0)
    if alteration == "affine":
        affines[1][:3, 3] += [.1, 0, 0]
    elif alteration == "almost_equal":
        arrays[1] = arrays[1].copy()
        arrays[1][4, 4, 4] += 1e-9
    elif alteration == "constant":
        arrays = [np.ones_like(arrays[0])] * 2
    else:
        arrays[0][0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="Finite"):
            vector.exact_identity_certificate(arrays, affines, directions)
        return
    assert vector.exact_identity_certificate(arrays, affines, directions) is None


def test_native_affine_sampling_and_vector_jacobians_match_analytic_linear_functions():
    arrays, affines, directions = pair_inputs()
    model = vector.PhysicalPair(arrays, affines, directions, maximum_displacement_mm=1., dtype=torch.float64)
    world = vector.simulation.grid_world(arrays[0].shape, affines[0])
    world_gradient = np.array([.01, -.005, .003])
    field_np = .6 + (world - [12, -18, 32]) @ world_gradient
    corrected, jacobians, _ = model.correct(torch.tensor(field_np, dtype=torch.float64))
    interior = (slice(3, -3),) * 3
    for index, direction in enumerate(directions):
        expected_jacobian = np.linalg.det(np.eye(3) + np.outer(direction, world_gradient))
        warped_world = world + field_np[..., None] * direction
        expected_image = (100 + warped_world @ np.array([.2, -.1, .15])) * expected_jacobian / model.scale
        assert np.allclose(jacobians[index].numpy(), expected_jacobian, atol=1e-12)
        assert np.allclose(corrected[index].numpy()[interior], expected_image[interior], atol=1e-12)


def test_ignoring_second_affine_changes_physical_predictions():
    arrays, affines, directions = pair_inputs(angle=10)
    correct = vector.PhysicalPair(arrays, affines, directions, maximum_displacement_mm=.5, dtype=torch.float64)
    wrong = vector.PhysicalPair(arrays, [affines[0], affines[0]], directions, maximum_displacement_mm=.5, dtype=torch.float64)
    zeros = torch.zeros(arrays[0].shape, dtype=torch.float64)
    right_image = correct.correct(zeros)[0][1]
    wrong_image = wrong.correct(zeros)[0][1]
    assert float((right_image - wrong_image)[correct.mask].abs().max()) > .001


def test_reference_sampling_does_not_introduce_roundtrip_errors_at_interpolation_knots():
    arrays, affines, directions = pair_inputs()
    model = vector.PhysicalPair(arrays, affines, directions, maximum_displacement_mm=.5, dtype=torch.float64)
    expected = np.moveaxis(np.indices(arrays[0].shape, dtype=float), 0, -1)
    assert np.array_equal(model.sampling[0][0].numpy(), expected)


def test_autograd_matches_independent_finite_difference_for_nonparallel_pair():
    arrays, affines, directions = pair_inputs()
    model = vector.PhysicalPair(arrays, affines, directions, maximum_displacement_mm=.5, dtype=torch.float64)
    rng = np.random.default_rng(20261004)
    perturbation = torch.tensor(rng.normal(size=arrays[0].shape), dtype=torch.float64)
    field = torch.full(arrays[0].shape, .37, dtype=torch.float64, requires_grad=True)
    loss, _ = model.objective(field, .02, .001)
    loss.backward()
    analytic = float((field.grad * perturbation).sum())
    epsilon = 1e-5
    plus = model.objective(field.detach() + epsilon * perturbation, .02, .001)[0]
    minus = model.objective(field.detach() - epsilon * perturbation, .02, .001)[0]
    numeric = float((plus - minus) / (2 * epsilon))
    assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-10)


def test_global_rigid_coordinate_change_preserves_predictions():
    arrays, affines, directions = pair_inputs()
    first = vector.PhysicalPair(arrays, affines, directions, maximum_displacement_mm=.5, dtype=torch.float64)
    transform = np.eye(4)
    transform[:3, :3] = vector.simulation.rotation(0, 23) @ vector.simulation.rotation(2, -31)
    transform[:3, 3] = [32., -15., 8.]
    changed = vector.PhysicalPair(arrays, [transform @ a for a in affines],
        [transform[:3, :3] @ p for p in directions], maximum_displacement_mm=.5, dtype=torch.float64)
    field = torch.full(arrays[0].shape, .4, dtype=torch.float64)
    for one, two in zip(first.correct(field)[0], changed.correct(field)[0]):
        assert torch.allclose(one, two, atol=1e-12, rtol=0)
    assert torch.equal(first.mask, changed.mask)


def test_folded_field_is_visible_and_strongly_penalized():
    arrays, affines, directions = pair_inputs(angle=0)
    model = vector.PhysicalPair(arrays, affines, directions, maximum_displacement_mm=.5, dtype=torch.float64)
    field = torch.tensor(np.indices(arrays[0].shape)[1] * 5., dtype=torch.float64)
    _, jacobians, _ = model.correct(field)
    assert min(float(j.min()) for j in jacobians) < 0
    objective, terms = model.objective(field, .02, .001)
    assert torch.isfinite(objective)
    assert float(terms["barrier"]) > 1e5


def test_identical_phase_directions_and_invalid_vectors_are_rejected():
    arrays, affines, directions = pair_inputs()
    with pytest.raises(ValueError, match="opposite"):
        vector.validate_pair(arrays, affines, [directions[0], directions[0]])
    with pytest.raises(ValueError, match="unit"):
        vector.validate_pair(arrays, affines, [directions[0] * 2, directions[1]])
