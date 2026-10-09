"""Synthetic-only controls for unreleased source-support preparation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/mechanics_case4_source_support.py"
spec = importlib.util.spec_from_file_location("mechanics_case4_source_support", SOURCE)
support = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = support
spec.loader.exec_module(support)


def domain(mask, origin=(0, 0, 0), matrix=None):
    affine = np.eye(4)
    affine[:3, 3] = origin
    if matrix is not None:
        affine[:3, :3] = matrix
    return support.VoxelDomain(mask, affine)


def test_signed_distance_to_full_cell_boundary_including_oblique_face():
    d = domain(np.ones((3, 3, 3), dtype=bool))
    assert d.signed_distance_mm((1, 1, 1)) == pytest.approx(1.5)
    assert d.signed_distance_mm((3, 1, 1)) == pytest.approx(-0.5)
    assert d.signed_distance_mm((2.5, 1, 1)) == pytest.approx(0)
    assert d.contains((2.49, 1, 1))
    assert d.within_full_cell_bounds((2.5, 1, 1))
    assert not d.within_full_cell_bounds((2.51, 1, 1))
    matrix = np.array([[1, 0.2, 0], [0, 1, 0.1], [0, 0, 1]], dtype=float)
    oblique = domain(np.ones((3, 3, 3), dtype=bool), matrix=matrix)
    point_on_face = matrix @ np.array([2.5, 1.0, 1.0])
    assert oblique.signed_distance_mm(point_on_face) == pytest.approx(0, abs=1e-10)
    normal = np.cross(matrix[:, 1], matrix[:, 2])
    normal /= np.linalg.norm(normal)
    assert oblique.signed_distance_mm(point_on_face + 0.2 * normal) == pytest.approx(-0.2)


def test_binary_union_internal_face_is_not_boundary():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    d = domain(mask)
    assert d.signed_distance_mm((2.5, 2, 2)) == pytest.approx(1.0)
    assert d.signed_distance_mm((0.5, 2, 2)) == pytest.approx(0)


def test_full_ball_analytic_mass_first_moment_and_two_rule_convergence():
    d = domain(np.ones((21, 21, 21), dtype=bool), origin=(-10, -10, -10))
    coarse = d.integrate_tent((0, 0, 0), order=4)
    fine = d.integrate_tent((0, 0, 0), order=8)
    other = d.integrate_tent((0, 0, 0), order=8, method="gauss")
    assert support.numerical_convergence(coarse, fine, other)
    assert fine.weighted_mass_fraction == pytest.approx(1, rel=0.01)
    assert fine.unweighted_volume_mm3 == pytest.approx(support.FULL_BALL_MM3, rel=0.02)
    assert fine.first_moment_mm4 == pytest.approx((0, 0, 0), abs=1e-9)
    assert fine.centroid_offset_mm == pytest.approx((0, 0, 0), abs=1e-9)
    assert fine.active_component_count == 1 and fine.centre_component_present


def test_half_ball_analytic_mass_and_centroid_are_geometric_only():
    mask = np.zeros((22, 21, 21), dtype=bool)
    mask[11:] = True
    d = domain(mask, origin=(-10.5, -10, -10))
    row = d.integrate_tent((0, 0, 0), order=10)
    assert row.weighted_mass_fraction == pytest.approx(0.5, rel=0.02)
    assert row.centroid_offset_mm[0] == pytest.approx(1.5, abs=0.1)
    assert row.centroid_offset_mm[1:] == pytest.approx((0, 0), abs=1e-9)
    assert d.signed_distance_mm((0, 0, 0)) == pytest.approx(0)


def test_disconnected_support_and_nonconvergence_hold():
    mask = np.zeros((17, 17, 17), dtype=bool)
    mask[8, 8, 8] = True
    mask[11, 8, 8] = True
    d = domain(mask, origin=(-8, -8, -8))
    row = d.integrate_tent((0, 0, 0), order=4)
    assert row.active_component_count == 2
    assert support.RowSupport(1, d.signed_distance_mm((0, 0, 0)), row, True, True, True).disposition == "unsupported_disconnected"
    assert support.RowSupport(1, 1.0, row, False, True, True).disposition == "hold_nonconverged"
    assert support.RowSupport(1, 1.0, row, True, None, True).disposition == "hold_surface_disagreement_or_unchecked"
    assert support.RowSupport(1, 0.0, row, True, True, True).disposition == "hold_boundary_ambiguous"
    assert support.RowSupport(1, 1.0, row, True, True, None).disposition == "hold_full_cell_bounds_unchecked"
    with pytest.raises(ValueError):
        support.numerical_convergence(row, row, row)


def test_tiny_positive_mass_is_not_admitted_as_operator_support():
    mask = np.zeros((11, 11, 11), dtype=bool)
    mask[5, 5, 5] = True
    d = domain(mask, matrix=np.diag([0.1, 0.1, 0.1]))
    centre = (0.5, 0.5, 0.5)
    row = d.integrate_tent(centre, order=8)
    assert d.signed_distance_mm(centre) == pytest.approx(0.05)
    assert 0 < row.weighted_mass_fraction < support.MINIMUM_WEIGHTED_SUPPORT_FRACTION
    assert support.RowSupport(1, d.signed_distance_mm(centre), row, True, True, True).disposition == "unsupported_insufficient_mass"


def test_fragmented_mask_rejected_before_face_materialization():
    checkerboard = np.indices((100, 100, 100)).sum(axis=0) % 2 == 0
    with pytest.raises(ValueError, match="exposed voxel faces exceed"):
        domain(checkerboard)


def test_dense_fine_mask_rejected_before_candidate_or_quadrature_materialization():
    dense = domain(np.ones((70, 70, 70), dtype=bool),
                   matrix=np.diag([0.1, 0.1, 0.1]))
    with pytest.raises(ValueError, match="candidate voxel box exceeds"):
        dense.integrate_tent((3.5, 3.5, 3.5), order=8)
    moderate = domain(np.ones((25, 25, 25), dtype=bool),
                      matrix=np.diag([0.4, 0.4, 0.4]))
    with pytest.raises(ValueError, match="quadrature work exceeds"):
        moderate.integrate_tent((5, 5, 5), order=8)


def test_fixed_denominator_rank_and_prerequisite_gate():
    integral = support.SupportIntegral(1, 1, 1, (0, 0, 0), (0, 0, 0), 1, True, "midpoint", 8)
    rows = {i: support.RowSupport(i, 1, integral, True, True, True) for i in range(1, 20)}
    centres = np.array([(0, 0, 0), (10, 0, 0), (0, 10, 0),
                        (0, 0, 10), (10, 10, 0), (10, 0, 10)])
    rank, ratio = support.rigid_average_rank(centres, np.zeros((6, 3)))
    assert rank == 6 and ratio > 1e-8
    line_rank, _ = support.rigid_average_rank(np.column_stack((np.arange(6),
                                                               np.zeros(6), np.zeros(6))),
                                               np.zeros((6, 3)))
    assert line_rank == 5
    with pytest.raises(PermissionError):
        support.fixed_partition_decision(rows, frame_accepted=False,
                                         anatomy_accepted=True, rigid_rank=rank)
    complete = support.fixed_partition_decision(rows, frame_accepted=True,
                                                 anatomy_accepted=True, rigid_rank=rank)
    assert complete["V_fixed_denominator"] == 13
    assert complete["V_geometrically_supported"] == 13
    assert complete["all_13_RMS_geometrically_estimable"]
    rows[2] = support.RowSupport(2, -1, integral, True, True, True)
    partial = support.fixed_partition_decision(rows, frame_accepted=True,
                                                anatomy_accepted=True, rigid_rank=rank)
    assert partial["V_fixed_denominator"] == 13
    assert partial["V_geometrically_supported"] == 12
    assert not partial["all_13_RMS_geometrically_estimable"]
    with pytest.raises(ValueError):
        support.fixed_partition_decision({1: rows[1]}, frame_accepted=True,
                                         anatomy_accepted=True, rigid_rank=rank)
