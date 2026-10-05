"""Independent continuum controls; no FEBio process or measured tissue data."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "mechanics_verification_review_target",
    Path(__file__).resolve().parents[1] / "scripts/mechanics_febio_verification.py",
)
verification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verification)


def principal_stretch_energy(F, mu):
    """Independent energy expressed through singular values, not target tensors."""
    stretches = np.linalg.svd(F, compute_uv=False)
    J = np.prod(stretches)
    isochoric = stretches / np.cbrt(J)
    bulk = (149.0 / 3.0) * mu
    return mu / 2 * (np.dot(isochoric, isochoric) - 3) + bulk / 4 * (
        J * J - 1 - 2 * np.log(J)
    )


def test_first_piola_is_energy_derivative_and_obeys_superposed_rotation():
    # All off-diagonals matter: a transpose error can survive diagonal stretches.
    F = np.array([[1.08, .13, -.04], [.02, .94, .07], [0., -.03, 1.01]])
    mu = 1000.
    response = verification.homogeneous_response(F, mu)
    derivative = np.empty((3, 3))
    step = 2e-6
    for i, j in np.ndindex(3, 3):
        direction = np.zeros((3, 3))
        direction[i, j] = step
        derivative[i, j] = (
            principal_stretch_energy(F + direction, mu)
            - principal_stretch_energy(F - direction, mu)
        ) / (2 * step)
    np.testing.assert_allclose(response["P_Pa"], derivative, rtol=3e-7, atol=2e-5)
    np.testing.assert_allclose(response["W_Pa"], principal_stretch_energy(F, mu), atol=1e-10)
    # A first-Piola/Cauchy substitution or missing transpose is detectable here.
    assert np.max(np.abs(derivative - response["sigma_Pa"])) > 100
    assert np.max(np.abs(derivative - response["P_Pa"].T)) > 100
    angle = .41
    Q = np.array([[np.cos(angle), -np.sin(angle), 0.],
                  [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    rotated = verification.homogeneous_response(Q @ F, mu)
    np.testing.assert_allclose(rotated["P_Pa"], Q @ derivative, rtol=3e-7, atol=2e-5)
    np.testing.assert_allclose(rotated["sigma_Pa"], Q @ response["sigma_Pa"] @ Q.T,
                               atol=1e-9, rtol=1e-12)
    np.testing.assert_allclose(rotated["W_Pa"], response["W_Pa"], atol=1e-10)


def test_signed_boundary_reactions_satisfy_reference_virtual_work():
    # An affine virtual displacement must recover the full nonsymmetric P tensor.
    X, _, _ = verification.cube_mesh()
    volume = .01 ** 3
    for case_id in verification.cases():
        for fraction in (.25, 1.):
            state = verification.expected_state(case_id, fraction)
            reactions = state["nodes"][:, 6:9]
            recovered_P = -(reactions.T @ X) / volume
            np.testing.assert_allclose(recovered_P, state["response"]["P_Pa"],
                                       atol=2e-12, rtol=1e-12)
    # Closed form checks the scale, sign and nonsymmetry without target P itself.
    shear = verification.expected_state("shear")
    recovered = -(shear["nodes"][:, 6:9].T @ X) / volume
    expected = np.array([[-10/3, 100., 0.], [301/3, -10/3, 0.], [0., 0., -10/3]])
    np.testing.assert_allclose(recovered, expected, atol=1e-10, rtol=0.)


def test_actual_position_reconstruction_preserves_full_affine_frame():
    X, _, _ = verification.cube_mesh()
    # Cyclic rotation, anisotropic stretch and mixed shear differ in every frame.
    Q = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    F = Q @ np.array([[1.03, .12, -.04], [.03, .96, .08], [0., .02, 1.02]])
    current = X @ F.T + np.array([.003, -.001, .0007])
    J, gradients = verification.reconstructed_jacobians(current)
    np.testing.assert_allclose(J, np.linalg.det(F), atol=3e-15, rtol=0.)
    np.testing.assert_allclose(gradients, np.broadcast_to(F, gradients.shape), atol=3e-15, rtol=0.)
    # Correct average-J output cannot hide a corrupted returned node coordinate.
    current[13, 0] += .02
    corrupted_J, _ = verification.reconstructed_jacobians(current)
    assert corrupted_J.min() < 0
