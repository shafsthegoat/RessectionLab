"""Independent analytic transport checks; synthetic experiment only.

Fixtures deliberately do not call the experiment's image/field/affine generator.
These checks validate equations and coordinate bookkeeping, not patient correction.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("audited_vector_solver", SCRIPTS / "experimental_vector_susceptibility.py")
vector = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC.loader.exec_module(vector)
finally:
    sys.path.pop(0)


def independently_distorted_pair(handedness=1):
    """Invert the full affine distortion matrix and its determinant analytically."""
    shape = (18, 20, 16)
    axes, _ = np.linalg.qr(np.array([[1., .2, -.1], [.4, 1., .3], [-.2, .1, 1.]]))
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    axes[:, 0] *= handedness
    first = np.eye(4)
    first[:3, :3] = axes @ np.diag([2., 3.1, 1.7])
    first[:3, 3] = [-15., 7., -4.]
    angle = np.deg2rad(3.)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    second = first.copy()
    second[:3, :3] = rotation @ first[:3, :3]
    second[:3, 3] += [.4, -.3, .2]
    directions = [first[:3, 1] / np.linalg.norm(first[:3, 1]),
                  -second[:3, 1] / np.linalg.norm(second[:3, 1])]
    index = np.moveaxis(np.indices(shape, dtype=float), 0, -1)
    gradient = np.array([.002, -.003, .001])
    intercept = .7
    density_gradient = np.array([.1, -.05, .08])
    arrays = []
    for affine, direction in zip((first, second), directions):
        observed_world = index @ affine[:3, :3].T + affine[:3, 3]
        distortion = np.eye(3) + np.outer(direction, gradient)
        undistorted_world = np.linalg.solve(distortion,
            (observed_world - intercept * direction).reshape(-1, 3).T).T.reshape(*shape, 3)
        arrays.append((100. + undistorted_world @ density_gradient) / np.linalg.det(distortion))
    reference_world = index @ first[:3, :3].T + first[:3, 3]
    field = intercept + reference_world @ gradient
    truth = 100. + reference_world @ density_gradient
    return arrays, [first, second], directions, field, truth, gradient


@pytest.mark.parametrize("handedness", [1, -1])
def test_inverse_density_transport_recovers_truth_on_anisotropic_nonparallel_grids(handedness):
    images, affines, directions, field, truth, gradient = independently_distorted_pair(handedness)
    assert np.sign(np.linalg.det(affines[0][:3, :3])) == handedness
    model = vector.PhysicalPair(images, affines, directions, maximum_displacement_mm=2., dtype=torch.float64)
    corrected, jacobians, _ = model.correct(torch.tensor(field, dtype=torch.float64))
    support = model.mask.numpy()
    for image, jacobian, direction in zip(corrected, jacobians, directions):
        determinant = np.linalg.det(np.eye(3) + np.outer(direction, gradient))
        np.testing.assert_allclose(jacobian.numpy(), determinant, rtol=0, atol=2e-14)
        np.testing.assert_allclose(image.numpy()[support] * model.scale, truth[support], rtol=0, atol=2e-12)
    # Same-field transport corrects each native observation independently.
    np.testing.assert_allclose(corrected[0].numpy()[support], corrected[1].numpy()[support], rtol=0, atol=2e-14)


def test_reversing_scalar_field_and_both_pe_signs_preserves_density_predictions():
    images, affines, directions, field, _truth, _gradient = independently_distorted_pair(-1)
    positive = vector.PhysicalPair(images, affines, directions, maximum_displacement_mm=2., dtype=torch.float64)
    negative = vector.PhysicalPair(images, affines, [-direction for direction in directions],
                                   maximum_displacement_mm=2., dtype=torch.float64)
    values = torch.tensor(field, dtype=torch.float64)
    forward = positive.correct(values)
    backward = negative.correct(-values)
    for group in (0, 1):
        for one, two in zip(forward[group], backward[group]):
            torch.testing.assert_close(one, two, rtol=0, atol=0)
    torch.testing.assert_close(positive.objective(values, .02, .001)[0],
                              negative.objective(-values, .02, .001)[0], rtol=0, atol=0)


def test_reflecting_and_permuting_world_coordinates_preserves_physical_predictions():
    images, affines, directions, field, _truth, _gradient = independently_distorted_pair()
    change = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transform = np.eye(4)
    transform[:3, :3] = change
    transform[:3, 3] = [21., -11., 8.]
    original = vector.PhysicalPair(images, affines, directions, maximum_displacement_mm=2., dtype=torch.float64)
    reflected = vector.PhysicalPair(images, [transform @ affine for affine in affines],
        [change @ direction for direction in directions], maximum_displacement_mm=2., dtype=torch.float64)
    assert torch.equal(original.mask, reflected.mask)
    for one, two in zip(original.correct(torch.tensor(field))[0], reflected.correct(torch.tensor(field))[0]):
        torch.testing.assert_close(one, two, rtol=0, atol=4e-14)


def test_physical_length_scaling_preserves_jacobians_and_objective():
    images, affines, directions, field, _truth, _gradient = independently_distorted_pair(-1)
    factor = 3.7
    physical_scale = np.diag([factor, factor, factor, 1.])
    original = vector.PhysicalPair(images, affines, directions, maximum_displacement_mm=2., dtype=torch.float64)
    scaled = vector.PhysicalPair(images, [physical_scale @ affine for affine in affines], directions,
                                  maximum_displacement_mm=2. * factor, dtype=torch.float64)
    assert torch.equal(original.mask, scaled.mask)
    field_tensor = torch.tensor(field, dtype=torch.float64)
    for group in (0, 1):
        for one, two in zip(original.correct(field_tensor)[group], scaled.correct(field_tensor * factor)[group]):
            torch.testing.assert_close(one, two, rtol=0, atol=4e-14)
    torch.testing.assert_close(original.objective(field_tensor, .02, .001)[0],
                              scaled.objective(field_tensor * factor, .02, .001)[0], rtol=0, atol=1e-15)


def test_fixed_support_cannot_escape_native_images_at_either_displacement_bound():
    images, affines, directions, _field, _truth, _gradient = independently_distorted_pair()
    maximum = 2.
    model = vector.PhysicalPair(images, affines, directions, maximum_displacement_mm=maximum, dtype=torch.float64)
    original_support = model.mask.clone()
    for sign in (-1., 1.):
        for base, native_direction, shape in model.sampling:
            displaced = (base + sign * maximum * native_direction)[model.mask].numpy()
            assert np.all(displaced >= 0) and np.all(displaced <= np.asarray(shape) - 1)
        model.objective(torch.full(model.shape, sign * maximum, dtype=torch.float64), .02, .001)
        assert torch.equal(model.mask, original_support)


def test_nonzero_intensity_offset_remains_in_density_objective():
    images, affines, directions, _field, _truth, _gradient = independently_distorted_pair()
    images = [images[0], images[0] + 17.]
    affines = [affines[0], affines[0].copy()]
    directions = [directions[0], -directions[0]]
    assert vector.exact_identity_certificate(images, affines, directions) is None
    model = vector.PhysicalPair(images, affines, directions, maximum_displacement_mm=2., dtype=torch.float64)
    objective, terms = model.objective(torch.zeros(model.shape, dtype=torch.float64), .02, .001)
    assert float(terms["data"]) == pytest.approx(.5 * (17. / model.scale)**2, rel=1e-12)
    assert float(objective) > 0
