"""Independent volume-functional controls; constructed coordinates only."""
import itertools
import math

import numpy as np
import pytest

from scripts import mechanics_patient_observation as m


def cube(side=.01):
    """Six Freudenthal tetrahedra, with independently constructed shared midsides."""
    points = []
    lookup = {}
    elements = []
    def index(x):
        key = tuple(x)
        if key not in lookup:
            lookup[key] = len(points)
            points.append(x)
        return lookup[key]
    for axes in itertools.permutations(range(3)):
        v = [np.zeros(3)]
        for axis in axes:
            step = v[-1].copy()
            step[axis] += side
            v.append(step)
        if np.linalg.det(np.stack(v[1:]) - v[0]) < 0:
            v[1], v[2] = v[2], v[1]
        row = [index(x) for x in v]
        row += [index((v[i] + v[j]) / 2)
                for i, j in [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]]
        elements.append(row)
    return np.array(points), np.array(elements)


@pytest.fixture(scope="module")
def octant():
    X, E = cube()
    return X, E, m.integrate_tent_observations(X, E, [[0., 0., 0.]])


@pytest.fixture(scope="module")
def small():
    X, E = cube(.002)
    centers = np.array([[.0004, .0004, .0004],
                        [.0016, .0005, .0006], [.0006, .0016, .0008]])
    return X, E, m.integrate_tent_observations(X, E, centers)


def test_octant_integrals_against_closed_form_radial_angular_moments(octant):
    X, _, op = octant
    radius = .005
    row = op.reports[0]
    assert row["weighted_volume_m3"] == pytest.approx(math.pi*radius**3/24, rel=1e-3)
    assert op.centroids_m[0] == pytest.approx(np.full(3, 3*radius/10), abs=5e-6)
    volume = math.pi*radius**3/6
    assert row["geometric_support_lower_m3"] <= volume <= row["geometric_support_upper_m3"]
    # E[r²]=2R²/5; octant E[nx²]=1/3 and E[nx*ny]=2/(3*pi).
    expected = radius**2*(2/15 + 3*4/(15*math.pi))
    quadratic = X[:, 0]**2 + 3*X[:, 0]*X[:, 1]
    assert op.apply(quadratic)[0] == pytest.approx(expected, rel=1e-3)
    assert op.apply(np.ones(len(X)))[0] == pytest.approx(1., abs=1e-12)


def test_confirmation_samples_count_toward_the_actual_hard_cap(octant):
    X, E, op = octant
    used = op.reports[0]["quadrature_points"]
    with pytest.raises(m.ObservationRefusal, match="quadrature_point_cap") as error:
        m.integrate_tent_observations(X, E, [[0., 0., 0.]],
            limits=m.Limits(quadrature_points_per_row=used - 1))
    assert error.value.details["points_used"] <= used - 1


@pytest.mark.parametrize("change", ["support", "convergence"])
def test_report_evidence_mutation_invalidates_operator(change):
    X, E = cube(.002)
    op = m.integrate_tent_observations(X, E, [[.001, .001, .001]])
    if change == "support":
        op.reports[0]["weighted_support_fraction"] = 1.
    else:
        op.reports[0]["refinement_ledger"][-1]["higher_order_cross_check"]["passes"] = False
    with pytest.raises(m.ObservationRefusal, match="operator_content_changed"):
        op.apply(np.ones(len(X)))
    with pytest.raises(m.ObservationRefusal, match="operator_content_changed"):
        m.prepare_elimination(op)


def test_input_snapshots_and_returned_numerical_storage_are_independent():
    X, E = cube(.002)
    centers = np.array([[.001, .001, .001]])
    original_field = X.copy()
    op = m.integrate_tent_observations(X, E, centers)
    identity = op.operator_hash
    X[:] = 50
    E[:] = 0
    centers[:] = 80
    assert op.operator_hash == identity
    assert op.apply(original_field)[0] == pytest.approx([.001]*3, abs=5e-6)
    for array in [op.weights, op.active_node_indices, op.centers_m, op.centroids_m]:
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_pivot_system_reproduces_all_rows_for_a_nonzero_constructed_right_side(small):
    X, _, op = small
    result = m.prepare_elimination(op)
    assert len(set(result.parent_node_indices)) == 3
    assert not set(result.parent_node_indices) & set(result.child_node_indices)
    field = np.zeros((len(X), 3))
    field[result.child_node_indices] = np.column_stack([
        np.arange(len(result.child_node_indices))*1e-7,
        np.full(len(result.child_node_indices), -2e-7),
        np.full(len(result.child_node_indices), 3e-7)])
    # Constructed algebraic RHS only: no patient destination supplied to operator.
    rhs = np.array([[1., -2., 3.], [2., 3., -1.], [-1., 2., 4.]])*1e-6
    field[result.parent_node_indices] = (np.linalg.solve(result.pivot_block, rhs)
        + result.child_coefficients @ field[result.child_node_indices])
    assert op.apply(field) == pytest.approx(rhs, abs=1e-18)
    again = m.prepare_elimination(op)
    np.testing.assert_array_equal(result.parent_node_indices, again.parent_node_indices)


def test_independent_rows_with_collinear_centroids_cannot_remove_all_rigid_modes():
    X, E = cube(.002)
    op = m.integrate_tent_observations(X, E,
        [[.0003]*3, [.001]*3, [.0017]*3])
    singular = np.linalg.svd(op.weights, compute_uv=False)
    assert singular[-1] > 1e-8*singular[0]
    with pytest.raises(m.ObservationRefusal, match="rigid_mode_rank_failure"):
        m.prepare_elimination(op)
