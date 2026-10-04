"""Independent algebra-only V2 checks. No simulator, policy or gradient import."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.native_spatial_features_v2 import (DeclaredAccessFrame, DeclaredTangent,
    FrameConditioningError, MIN_TANGENT_SINE, ROUNDOFF_REJECTION_HALF_WIDTH,
    declared_access_frame, candidate_coordinates, cavity_residual_summary)

SOURCE = "synthetic-independent-spatial-v2"


def reference(vector=(1., 0., 0.), label="RAS+"):
    return DeclaredTangent(vector, SOURCE, label, "fixed-external-reference")


def frame(normal=(0., 0., 1.), tangent=(1., 0., 0.), center=(0., 0., 0.), label="RAS+"):
    return declared_access_frame(center, normal, reference=reference(tangent, label),
                                 expected_source_hash=SOURCE, coordinate_frame=label)


@pytest.mark.parametrize("scale", [np.nextafter(0., 1.), 1e-300, 1e300, np.finfo(float).max / 2])
def test_extreme_finite_direction_scales_preserve_orthonormal_right_handed_basis(scale):
    base = frame(tangent=(1., 1., 0.))
    actual = frame(normal=(0., 0., scale), tangent=(scale, scale, 0.))
    np.testing.assert_allclose(actual.basis_columns, base.basis_columns, rtol=0, atol=2e-15)
    basis = np.asarray(actual.basis_columns)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), rtol=0, atol=2e-15)
    assert np.linalg.det(basis) == pytest.approx(1.)
    assert np.isfinite(actual.project([[1., 2., 3.]])).all()
    json.dumps(actual.to_dict(), allow_nan=False)


@pytest.mark.parametrize("normal,tangent", [((0., 0., 0.), (1., 0., 0.)),
    ((0., 0., 1.), (0., 0., 0.)), ((0., 0., np.inf), (1., 0., 0.)),
    ((0., 0., 1.), (np.nan, 0., 0.)), ((0., 0., 1.), (0., 0., -1.))])
def test_invalid_or_antiparallel_directions_do_not_receive_accepted_receipts(normal, tangent):
    with pytest.raises(ValueError):
        frame(normal, tangent)


@pytest.mark.parametrize("field,value", [
    ("basis_columns", np.zeros((3, 3))),
    ("basis_columns", np.diag([1., -1., 1.])),
    ("basis_columns", np.diag([-1., -1., 1.])),
    ("normal_inward", (0., 1., 0.)),
    ("tangent_sine", .5),
])
def test_direct_frame_construction_cannot_forge_accepted_basis_or_condition(field, value):
    declared = frame()
    with pytest.raises(ValueError):
        replace(declared, **{field: value})


def test_direct_frame_owns_immutable_numeric_data_not_caller_lists():
    origin = [0., 0., 0.]
    basis = np.eye(3)
    normal = [0., 0., 1.]
    declared = DeclaredAccessFrame(origin, basis, normal, reference(), 1.)
    original = declared.to_dict()
    origin[0] = 99.; basis[:] = 0.; normal[:] = [1., 0., 0.]
    assert declared.to_dict() == original
    np.testing.assert_array_equal(declared.project([[1., 2., 3.]]), [[1., 2., 3.]])


def test_projection_rejects_finite_inputs_whose_subtraction_overflows():
    declared = frame(center=(1e308, 0., 0.))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite|overflow|range|represent"):
            declared.project([[-1e308, 0., 0.]])


@pytest.mark.parametrize("many_cells", [False, True])
def test_summary_rejects_nonrepresentable_affine_or_aggregate_volume(many_cells):
    shape = (3, 3, 3) if many_cells else (1, 1, 1)
    tissue = np.ones(shape, dtype=bool)
    scale = 10 ** (307 / 3) if many_cells else 1e200
    affine = np.diag([scale, scale, scale, 1.])
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite|overflow|volume|represent|determinant"):
            cavity_residual_summary(tissue, tissue, ~tissue, affine, frame())


@pytest.mark.parametrize("where", ["normal", "reference", "point", "candidate", "affine"])
def test_complex_physical_inputs_are_rejected_instead_of_silently_discarding_imaginary_parts(where):
    complex_vector = np.array([1. + 1.j, 0., 0.])
    with pytest.raises((ValueError, TypeError)):
        if where == "normal":
            frame(normal=complex_vector, tangent=(0., 1., 0.))
        elif where == "reference":
            reference(complex_vector)
        elif where == "point":
            frame().project([complex_vector])
        elif where == "candidate":
            candidate_coordinates(("test-cut",), [complex_vector], [[1., 0., 0.]], frame())
        else:
            affine = np.eye(4, dtype=complex); affine[0, 0] += 1.j
            tissue = np.ones((1, 1, 1), dtype=bool)
            cavity_residual_summary(tissue, tissue, ~tissue, affine, frame())


@pytest.mark.parametrize("sine", [.099, MIN_TANGENT_SINE - .5 * ROUNDOFF_REJECTION_HALF_WIDTH,
    MIN_TANGENT_SINE + .5 * ROUNDOFF_REJECTION_HALF_WIDTH, .101, .9])
def test_rejection_band_transforms_with_the_explicit_physical_reference(sine):
    saved = json.loads((ROOT / "artifacts/native-spatial-feature-independent-v1/conditioning-counterexample.json").read_text())
    q = np.asarray(saved["threshold_rotation_counterexample"]["rotation"])
    n = np.array([1., 0., 0.]); tangent = np.array([np.sqrt(1 - sine * sine), sine, 0.])
    accepted = sine in (.101, .9)
    frames = []
    for normal, reference_vector, label in ((n, tangent, "RAS+"), (q @ n, q @ tangent, "ROTATED")):
        if not accepted:
            with pytest.raises(FrameConditioningError) as caught:
                frame(normal, reference_vector, label=label)
            assert caught.value.receipt["status"] == "rejected"
            assert caught.value.receipt["automatic_axis_selection"] is False
            json.dumps(caught.value.receipt, allow_nan=False)
        else:
            frames.append(frame(normal, reference_vector, label=label))
    if accepted:
        points = np.array([[1., 2., 3.], [-2., 4., .5]])
        np.testing.assert_allclose(frames[0].project(points), frames[1].project(points @ q.T), rtol=0, atol=1e-12)


def test_reference_vector_is_bound_by_its_hash_even_when_declaration_id_is_reused():
    first, second = frame(), frame(tangent=(0., 1., 0.))
    assert first.reference.declaration_id == second.reference.declaration_id
    assert first.to_dict()["reference_hash"] != second.to_dict()["reference_hash"]
    assert not np.array_equal(first.project([[1., 0., 0.]]), second.project([[1., 0., 0.]]))


def test_signed_sheared_affine_and_full_source_reindex_keep_the_same_physical_summary():
    tissue = np.ones((3, 4, 5), dtype=bool)
    target = np.zeros_like(tissue); target[1:3, 1:4, 2:5] = True
    removed = np.zeros_like(tissue); removed[1, 2, :3] = True
    affine = np.array([[2., .3, 0., 7.], [0., 3., .2, -4.], [0., 0., -4., 8.], [0., 0., 0., 1.]])
    declared = frame(center=(7., -4., 8.))
    expected = [72., 2.6, 6.2, -4., 408., 2 * 26 / 17 + .6, 6 + .2 * 52 / 17, -4 * 52 / 17]
    original = cavity_residual_summary(tissue, target, removed, affine, declared)
    np.testing.assert_allclose(original, expected, rtol=0, atol=1e-12)
    # New indices(a,b,c) address old(nx-1-b,c,nz-1-a): both a cyclic
    # permutation and two flips. Reference/access are fixed in physical space.
    index_map = np.array([[0., -1., 0., 2.], [0., 0., 1., 0.], [-1., 0., 0., 4.], [0., 0., 0., 1.]])
    reindexed = [mask.transpose(2, 0, 1)[::-1, ::-1, :] for mask in (tissue, target, removed)]
    transformed = cavity_residual_summary(*reindexed, affine @ index_map, declared)
    np.testing.assert_allclose(transformed, expected, rtol=0, atol=1e-12)
