"""Independent analytic review controls. Never execute FEBio or read patient data.

The reference interpolation below is recovered by a polynomial Vandermonde
system, and the force oracle uses first Piola stress assembled over volume.
Neither calls the fixture's shape/response/energy/reaction implementation.
These controls establish software consistency, not a solved mechanics result.
"""
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "patient_constraints_review_target", ROOT / "scripts/mechanics_patient_constraints.py"
)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)
BUNDLE = ROOT / "artifacts/mechanics-patient-constraints-runtime-v1"
EXPONENTS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
             (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0),
             (0, 1, 1), (1, 0, 1))
# Pinned FEBio order, independently stated rather than inherited from v.EDGES.
REFERENCE_NODES = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [.5, 0, 0], [.5, .5, 0], [0, .5, 0],
    [0, 0, .5], [.5, 0, .5], [0, .5, .5],
])


def polynomial_basis(point):
    return np.array([np.prod(point ** p) for p in EXPONENTS])


COEFFICIENTS = np.linalg.inv(np.stack([polynomial_basis(p) for p in REFERENCE_NODES]))


def independent_derivatives(point):
    result = np.zeros((10, 3))
    for axis in range(3):
        for term, exponents in enumerate(EXPONENTS):
            if exponents[axis]:
                power = np.array(exponents)
                power[axis] -= 1
                result[:, axis] += exponents[axis] * np.prod(point ** power) * COEFFICIENTS[term]
    return result


def independent_material(F):
    """SVD principal energy and direct dW/dF; mu and K have SI pressure units."""
    mu, K = 1000., 29000. / 3
    stretches = np.linalg.svd(F, compute_uv=False)
    J = np.linalg.det(F)
    assert J > 0
    W = mu / 2 * (sum((stretches / J ** (1 / 3)) ** 2) - 3)
    W += K / 4 * (J ** 2 - 1 - 2 * np.log(J))
    invT = np.linalg.inv(F).T
    P = mu * J ** (-2 / 3) * (F - np.sum(F * F) / 3 * invT)
    P += K / 2 * (J ** 2 - 1) * invT
    return J, W, P


def independent_assembly(current, quadrature=None):
    X, elements, _ = v.fixture_mesh()
    # Separately verified quadrature moments below; exact centroid integration
    # is optionally used for the affine-force oracle (linear integrand only).
    points, weights = v.gauss_rule() if quadrature is None else quadrature
    internal = np.zeros_like(X)
    all_F, all_J, all_W, all_stress = [], [], [], []
    total_energy = 0.
    for element in elements:
        Fs, Js, Ws, stresses = [], [], [], []
        for point, weight in zip(points, weights):
            D = independent_derivatives(point)
            ref_jac = X[element].T @ D
            gradients = np.linalg.solve(ref_jac.T, D.T).T
            F = current[element].T @ gradients
            J, W, P = independent_material(F)
            dV = np.linalg.det(ref_jac) * weight
            internal[element] += dV * gradients @ P.T
            total_energy += W * dV
            Fs.append(F); Js.append(J); Ws.append(W)
            stresses.append(P @ F.T / J)
        all_F.append(Fs); all_J.append(Js); all_W.append(Ws); all_stress.append(stresses)
    return (internal, total_energy, np.array(all_F), np.array(all_J),
            np.array(all_W), np.array(all_stress))


def test_quadrature_moments_and_interpolation_match_reference_tetrahedron():
    points, weights = v.gauss_rule()
    assert points.shape == (8, 3) and np.all(weights > 0)
    # Rule constants are rounded in pinned upstream; retain those constants.
    for a, b, c in itertools.product(range(4), repeat=3):
        if a + b + c > 3:
            continue
        exact = math.factorial(a) * math.factorial(b) * math.factorial(c) / math.factorial(a + b + c + 3)
        assert weights @ np.prod(points ** [a, b, c], axis=1) == pytest.approx(exact, abs=5e-11)
    for point in [*REFERENCE_NODES, *points, [.12, .21, .31]]:
        shape, derivative = v.shape(point)
        np.testing.assert_allclose(shape, polynomial_basis(np.asarray(point)) @ COEFFICIENTS, atol=5e-16, rtol=0)
        np.testing.assert_allclose(derivative, independent_derivatives(np.asarray(point)), atol=2e-15, rtol=0)
    X, elements, boundary = v.fixture_mesh()
    for element in elements:
        affine = np.column_stack((np.ones(4), REFERENCE_NODES[:4]))
        mapping = np.linalg.solve(affine, X[element[:4]])
        np.testing.assert_allclose(np.column_stack((np.ones(10), REFERENCE_NODES)) @ mapping,
                                   X[element], atol=2e-18, rtol=0)
    assert set(range(27)) - set(boundary) == {11}
    assert all(11 in element for element in elements)
    assert sum(np.linalg.det((X[e[1:4]] - X[e[0]]).T) / 6 for e in elements) == pytest.approx(1e-6, rel=1e-14)


def test_finite_material_objectivity_pressure_and_initial_nu():
    nu = (3 * v.K_PA - 2 * v.MU_PA) / (2 * (3 * v.K_PA + v.MU_PA))
    assert nu == pytest.approx(.45)
    theta = .53
    Q = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    for F in [np.diag([1.1, .9, 1.04]), np.eye(3) * 1.03,
              np.array([[1.03, .09, -.02], [.01, .98, .03], [0, .01, 1.02]])]:
        J, W, P = independent_material(F)
        actualJ, actualW, sigma = v.response(F)
        np.testing.assert_allclose([actualJ, actualW], [J, W], rtol=1e-12, atol=1e-11)
        np.testing.assert_allclose(sigma, P @ F.T / J, rtol=2e-13, atol=2e-11)
        rotated = v.response(Q @ F)
        assert rotated[1] == pytest.approx(W, abs=2e-11)
        np.testing.assert_allclose(rotated[2], Q @ sigma @ Q.T, atol=3e-11, rtol=2e-13)


def test_nonuniform_actual_field_uses_pointwise_F_and_unweighted_primitive_means():
    X, _, _ = v.fixture_mesh()
    current = X.copy()
    # A conforming quadratic displacement field, not an equilibrium solution.
    current[:, 0] += 4 * X[:, 0] * X[:, 1]
    current[:, 1] -= 2 * X[:, 1] * X[:, 2]
    current[:, 2] += X[:, 0] ** 2
    _, energy, Fs, Js, Ws, stresses = independent_assembly(current)
    actual = v.state_mechanics(current)
    np.testing.assert_allclose(actual['F'], Fs, rtol=1e-13, atol=3e-15)
    assert actual['energy_J'] == pytest.approx(energy, rel=1e-12, abs=1e-16)
    means = stresses.mean(axis=1)
    expected = np.column_stack((means[:, 0, 0], means[:, 1, 1], means[:, 2, 2],
                                means[:, 0, 1], means[:, 1, 2], means[:, 0, 2],
                                Js.mean(axis=1), Ws.mean(axis=1)))
    np.testing.assert_allclose(actual['primitive'], expected, rtol=2e-12, atol=2e-11)
    # Primitive output means cannot replace the reference-volume integral.
    naive = Ws.mean(axis=1).sum() * 1e-6 / 6
    assert abs(energy - naive) > 1e-8


def test_all_72_finite_difference_directions_agree_with_independent_assembled_force():
    X, _, _ = v.fixture_mesh()
    H, parents, _, T = v.observation_operator()
    current = X.copy()
    current[parents] += np.linalg.solve(H[:, parents], v.targets('mpc_nonrigid'))
    internal, _, _, _, _, _ = independent_assembly(current)
    expected = (T.T @ internal).reshape(-1)
    assert expected.shape == (72,) and np.max(np.abs(expected)) > 1e-4
    audit = v.feasible_virtual_work(current)
    assert not audit['passed']
    measured = np.array(audit['directional_derivatives_N'])
    assert measured.shape == (72, 3)
    for step in range(3):
        np.testing.assert_allclose(measured[:, step], expected, atol=2e-9, rtol=1e-6)
    np.testing.assert_allclose(H @ (current - X), v.targets('mpc_nonrigid'), atol=1e-18, rtol=0)


def test_affine_signed_reactions_match_independent_volume_force_at_every_node():
    X, _, _ = v.fixture_mesh()
    for time in (.25, 1.):
        current = X @ v.affine_F(time).T
        internal = independent_assembly(current)[0]
        exact = independent_assembly(current, (np.array([[.25, .25, .25]]), [1 / 6]))[0]
        np.testing.assert_allclose(v.affine_reactions(time), -exact, atol=1e-14, rtol=1e-12)
        assert np.max(np.abs(exact[11])) < 1e-14
        # Pinned upstream rounded quadrature has a small nonzero first-moment
        # error. It remains below the original fixture's frozen force gate.
        np.testing.assert_allclose(internal, exact, atol=v.TOL['force_N'], rtol=0)
        assert np.max(np.abs(exact)) > 1e-4


@pytest.mark.parametrize('case', ['mpc_translation', 'mpc_nonrigid'])
def test_exported_mpc_reconstructs_original_H_for_nonzero_free_values(case):
    xml = ET.fromstring(v.deck_xml(case))
    H, parents, free, T = v.observation_operator()
    # Deterministic nonzero child values expose reversed offset/coefficient signs.
    free_values = np.sin(np.arange(72).reshape(24, 3) + .2) * 1e-4
    bcs = xml.findall('Boundary/bc')
    assert len(bcs) == 9 and all(b.attrib['type'] == 'linear constraint' for b in bcs)
    for time in (.25, .5, 1.):
        displacement = np.zeros((27, 3))
        displacement[free] = free_values
        for bc in bcs:
            node = int(bc.findtext('node')) - 1
            axis = 'xyz'.index(bc.findtext('dof'))
            offset = bc.find('offset')
            assert offset.attrib == {'lc': '1'}
            value = float(offset.text) * time
            for child in bc.findall('child_dof'):
                child_node = int(child.findtext('node')) - 1
                assert child_node not in parents
                assert child.findtext('dof') == bc.findtext('dof')
                coefficient = child.find('value')
                assert not coefficient.attrib  # Offset alone receives the ramp.
                value += float(coefficient.text) * displacement[child_node, axis]
            displacement[node, axis] = value
        np.testing.assert_allclose(H @ displacement, time * v.targets(case), atol=3e-20, rtol=0)
        np.testing.assert_allclose(displacement[free], free_values, atol=0, rtol=0)
    assert np.linalg.matrix_rank(np.kron(T, np.eye(3))) == 72


def test_prepared_caps_decks_and_source_bindings_are_complete():
    manifest = json.loads((BUNDLE / 'decks/manifest.json').read_text())
    prepared = json.loads((BUNDLE / 'preparation.json').read_text())
    assert manifest['cases'] == ['tet10_affine', 'mpc_translation', 'mpc_nonrigid']
    assert manifest['caps'] == {'aggregate_seconds': 60, 'process_group_rss_bytes': 3221225472,
                                'numerical_threads': 1, 'maximum_cases': 3, 'time_steps': 4, 'retries': 0}
    assert manifest['feasible_direction_count'] == 72
    assert manifest['finite_difference_steps_m'] == [2e-7, 1e-7, 5e-8]
    assert manifest['status'] == 'prepared_not_executed'
    for name, expected in prepared['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    for case in manifest['cases']:
        assert (BUNDLE / 'decks' / (case + '.feb')).read_text() == v.deck_xml(case)
        xml = ET.fromstring(v.deck_xml(case))
        assert xml.find('MeshDomains/SolidDomain').attrib['elem_type'] == 'TET10G8'
        assert xml.find('MeshDomains/SolidDomain').attrib['type'] == 'elastic-solid'
        assert xml.findtext('Control/time_stepper/max_retries') == '0'
        assert xml.findtext('Control/time_steps') == '4'
        assert xml.find('Control/solver/linear_solver').attrib == {'type': 'skyline'}
