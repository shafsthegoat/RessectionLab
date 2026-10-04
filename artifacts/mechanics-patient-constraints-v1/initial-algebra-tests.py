"""Executable algebra specifications, not FEBio or patient-model verification.

The source-reviewed XML equation is evaluated literally. No anatomy, material
solver, landmark data, patient destinations or model runs are used here.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('fraction', [0.0, 0.25, 0.5, 1.0])
def test_declared_xml_nonzero_offset_has_correct_sign_and_meters(fraction):
    root = ET.parse(ROOT / 'artifacts/mechanics-patient-constraints-v1/linear-constraint-example.xml').getroot()
    bc = root.find('Boundary/bc')
    assert bc.attrib == {'type': 'linear constraint'}
    assert bc.findtext('node') == '1' and bc.findtext('dof') == 'x'
    assert bc.find('offset').attrib == {'lc': '1'}
    children = bc.findall('child_dof')
    assert len(children) == 1 and children[0].findtext('node') == '2'
    for independent_m in [-0.002, 0.0, 0.007]:
        dependent_m = float(bc.findtext('offset')) * fraction + float(children[0].findtext('value')) * independent_m
        original_average_m = .25 * dependent_m + .75 * independent_m
        assert original_average_m == pytest.approx(.001 * fraction, abs=2e-18)
        if fraction:
            assert not np.isclose(original_average_m, 1.0 * fraction, rtol=0, atol=1e-9)


def test_overlapping_rows_preserve_both_targets_without_chained_parents():
    H = np.array([[.5, .5, 0], [0, .5, .5]])
    D = np.array([[.001, -.002, .003], [-.004, .005, .006]])
    pivots = [0, 2]
    free = [1]
    coefficients = -np.linalg.solve(H[:, pivots], H[:, free])
    offset = np.linalg.solve(H[:, pivots], D)
    np.testing.assert_array_equal(coefficients, [[-1], [-1]])
    np.testing.assert_array_equal(offset, 2 * D)
    assert not set(pivots).intersection(free)
    for value in [-.002, 0, .008]:
        U = np.zeros((3, 3))
        U[free] = [value, -value, value / 3]
        U[pivots] = offset + coefficients @ U[free]
        np.testing.assert_allclose(H @ U, D, rtol=0, atol=2e-18)
    # Naive sequential rows using parent0 then parent1 violate FEBio's rule.
    assert set([0, 1]).intersection([1, 2]) == {1}


def test_feasible_directions_leave_original_constraints_unchanged():
    H = np.array([[.5, .5, 0, 0], [0, .25, .25, .5]])
    pivots, free = [0, 3], [1, 2]
    T = np.zeros((4, 2))
    T[free] = np.eye(2)
    T[pivots] = -np.linalg.solve(H[:, pivots], H[:, free])
    np.testing.assert_array_equal(H @ T, np.zeros((2, 2)))
    # Pure virtual-work coordinate identity; no constitutive/force engine.
    force = np.array([.01, -.02, .03, -.04])
    direction = np.array([.2, -.3])
    assert force @ (T @ direction) == pytest.approx((T.T @ force) @ direction, abs=2e-18)
    # Adding constraint-normal force changes no feasible virtual work.
    normal_force = H.T @ np.array([.7, -.6])
    np.testing.assert_allclose(T.T @ normal_force, 0, rtol=0, atol=1e-16)


def test_redundant_observation_rows_have_no_invertible_complete_pivot_block():
    H = np.array([[.2, .8, 0], [.2, .8, 0]])
    assert np.linalg.matrix_rank(H) == 1
    for pivots in ([0, 1], [0, 2], [1, 2]):
        assert not np.isfinite(np.linalg.cond(H[:, pivots])) or np.linalg.cond(H[:, pivots]) > 1e15
    # Equal support with different target is incompatible, not an anchor request.
    assert np.linalg.matrix_rank(np.column_stack([H, [.001, .002]])) == 2


def rigid_mode_operator(centroids_m):
    characteristic_m = .005
    blocks = []
    for x, y, z in np.asarray(centroids_m) / characteristic_m:
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        blocks.append(np.column_stack([np.eye(3), -skew]))
    return np.vstack(blocks)


@pytest.mark.parametrize('centroids,rank', [
    ([[0, 0, 0]], 3),
    ([[0, 0, 0], [.005, 0, 0], [.01, 0, 0]], 5),
    ([[0, 0, 0], [.005, 0, 0], [0, .005, 0]], 6),
])
def test_rigid_mode_rank_is_separate_from_observation_row_rank(centroids, rank):
    assert np.linalg.matrix_rank(rigid_mode_operator(centroids)) == rank


def test_poisson_ratio_point_four_five_is_not_specimen_bulk_modulus():
    ratio = 2 * (1 + .45) / (3 * (1 - 2 * .45))
    assert ratio == pytest.approx(29 / 3, rel=1e-15)
    assert ratio != pytest.approx(149 / 3)
