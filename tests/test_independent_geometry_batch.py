"""Numerical controls only: no patient cases, policies, or evaluator wiring."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab import evaluation


PATH = Path(__file__).resolve().parents[1] / "scripts/independent_geometry_batch_prototype.py"
SPEC = importlib.util.spec_from_file_location("independent_geometry_batch_prototype", PATH)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def scalar(start, end, lower, upper):
    return np.array([evaluation.segment_box_distance_sq(start, end, lo, hi)
                     for lo, hi in zip(lower, upper)])


@pytest.mark.parametrize("start,end,expected", [
    ([0, 0, 0], [0, 0, 0], 3.0),
    ([0, 1.5, 1.5], [0, 1.5, 1.5], 1.0),
    ([0, 0, 1.5], [0, 0, 1.5], 2.0),
    ([0, 1.5, 1.5], [3, 1.5, 1.5], 0.0),
    ([1, 1, 1], [2, 2, 2], 0.0),
    ([0, 0, 1], [3, 3, 1], 0.0),
    ([1.5, 1.5, 1.5], [1.5, 1.5, 1.5], 0.0),
])
def test_analytic_distances(start, end, expected):
    lower, upper = np.ones((1, 3)), np.full((1, 3), 2.0)
    actual = batch.segment_box_distances_sq(start, end, lower, upper)
    np.testing.assert_array_equal(actual, scalar(start, end, lower, upper))
    assert actual[0] == expected


@pytest.mark.parametrize("seed", [0, 41, 2026])
def test_random_distances_and_contacts_preserve_scalar(seed):
    rng = np.random.default_rng(seed)
    for _ in range(8):
        start, end = rng.uniform(-20, 20, size=(2, 3))
        centers = rng.uniform(-25, 25, size=(173, 3))
        half_sizes = rng.uniform(0, 4, size=(173, 3))
        lower, upper = centers - half_sizes, centers + half_sizes
        expected = scalar(start, end, lower, upper)
        actual = batch.segment_box_distances_sq(start, end, lower, upper, batch_size=37)
        # Report bitwise equality in the probe, but portable tests allow small
        # implementation-level dot-product rounding differences across NumPy.
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
        for tolerance in (0, 1e-10, 1e-9):
            expected_indices = np.flatnonzero(expected <= 2.5**2 + tolerance)
            np.testing.assert_array_equal(batch.segment_box_contact_indices(
                start, end, lower, upper, 2.5, tolerance_sq=tolerance, batch_size=37), expected_indices)
            first = int(expected_indices[0]) if len(expected_indices) else None
            assert batch.first_segment_box_contact(start, end, lower, upper, 2.5,
                tolerance_sq=tolerance, batch_size=37) == first


@pytest.mark.parametrize("direction", [[0, 0, 0], [1e-16, 0, 0], [1e-15, -1e-15, 0], [1.1e-15, 0, 0], [1, 0, 0]])
def test_degenerate_segments_point_boxes_and_face_crossings(direction):
    lower = np.array([[0, 0, 0], [.25, 0, 0], [2, 1, 0], [-1, -1, -1]], dtype=float)
    upper = np.array([[0, 0, 0], [.75, 0, 0], [3, 2, 0], [1, 1, 1]], dtype=float)
    np.testing.assert_array_equal(batch.segment_box_distances_sq(np.zeros(3), direction, lower, upper),
                                  scalar(np.zeros(3), direction, lower, upper))


@pytest.mark.parametrize("radius", [1.0, np.sqrt(2), np.sqrt(3)])
@pytest.mark.parametrize("tolerance", [0.0, 1e-10, 1e-9])
def test_tangency_one_ulp_radii_and_original_tolerances(radius, tolerance):
    lower = np.array([[1, -1, -1], [1, 1, -1], [1, 1, 1]], dtype=float)
    upper = lower + 2.0
    expected = scalar(np.zeros(3), np.zeros(3), lower, upper)
    for r in (np.nextafter(radius, 0), radius, np.nextafter(radius, np.inf)):
        np.testing.assert_array_equal(batch.segment_box_contact_indices(
            np.zeros(3), np.zeros(3), lower, upper, r, tolerance_sq=tolerance, batch_size=1),
            np.flatnonzero(expected <= r*r + tolerance))


def test_near_threshold_decision_calls_preserved_scalar(monkeypatch):
    lower = np.array([[1, -1, -1], [4, -1, -1]], dtype=float)
    upper = lower + 1
    original = evaluation.segment_box_distance_sq
    calls = []
    def recorded(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(evaluation, "segment_box_distance_sq", recorded)
    assert batch.first_segment_box_contact([0, 0, 0], [0, 0, 0], lower, upper, 1, tolerance_sq=0) == 0
    assert len(calls) == 1


@pytest.mark.parametrize("shift", [0.0, 1e6, 1e12])
def test_large_common_translation_compares_same_representable_coordinates(shift):
    rng = np.random.default_rng(217)
    lower = rng.uniform(-10, 10, size=(67, 3)) + shift
    upper = lower + rng.uniform(0.001, 5, size=(67, 3))
    start, end = np.array([-7.4, .25, -3]) + shift, np.array([9.3, -8, 7.7]) + shift
    expected = scalar(start, end, lower, upper)
    np.testing.assert_allclose(batch.segment_box_distances_sq(start, end, lower, upper), expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(batch.segment_box_contact_indices(start, end, lower, upper, .75),
                                  np.flatnonzero(expected <= .75**2 + 1e-10))


@pytest.mark.parametrize("mirrored", [False, True])
def test_anisotropic_oblique_grid_uses_same_independent_local_frame(mirrored):
    spacing = np.array([.31, 2.4, 5.7])
    rotation = Rotation.from_euler("xyz", [13, -17, 23], degrees=True).as_matrix()
    if mirrored:
        rotation[:, 1] *= -1
    translation = np.array([145.5, -831.2, 99.3])
    world_start = rotation @ np.array([-1.2, 5.6, 18.0]) + translation
    world_end = rotation @ np.array([9.2, 19.4, -2.0]) + translation
    local_start, local_end = (rotation.T @ (point - translation) for point in (world_start, world_end))
    indices = np.argwhere(np.ones((8, 5, 4), dtype=bool))
    centers = indices * spacing
    lower, upper = centers - spacing / 2, centers + spacing / 2
    expected = scalar(local_start, local_end, lower, upper)
    np.testing.assert_allclose(batch.segment_box_distances_sq(local_start, local_end, lower, upper), expected,
                               rtol=2e-14, atol=2e-14)
    matches = np.flatnonzero(expected <= .7**2 + 1e-10)
    first = batch.first_segment_box_contact(local_start, local_end, lower, upper, .7, batch_size=9)
    assert first == (int(matches[0]) if len(matches) else None)
    if first is not None:
        np.testing.assert_array_equal(indices[first], indices[matches[0]])


@pytest.mark.parametrize("batch_size", [1, 7, 256, 4096])
def test_chunk_sizes_preserve_distances_and_first_input_order(batch_size):
    lower = np.zeros((71, 3)); lower[:, 0] = np.arange(71)[::-1]
    upper = lower + .5
    expected = scalar(np.zeros(3), np.zeros(3), lower, upper)
    np.testing.assert_array_equal(batch.segment_box_distances_sq([0, 0, 0], [0, 0, 0], lower, upper, batch_size=batch_size), expected)
    assert batch.first_segment_box_contact([0, 0, 0], [0, 0, 0], lower, upper, 1, batch_size=batch_size) == 69


def test_working_chunks_bounded_even_for_noncontiguous_inputs(monkeypatch):
    original = batch._distances_chunk
    seen = []
    def recorded(start, end, lo, hi):
        seen.append(len(lo))
        return original(start, end, lo, hi)
    monkeypatch.setattr(batch, "_distances_chunk", recorded)
    lower = np.zeros((1003, 6))[:, ::2]
    upper = np.ones((1003, 6))[:, ::2]
    result = batch.segment_box_distances_sq([2, 2, 2], [3, 3, 3], lower, upper, batch_size=63)
    assert len(result) == 1003 and max(seen) == 63 and sum(seen) == 1003


def test_empty_contact_chunks_do_not_accumulate_output_fragments(monkeypatch):
    def no_empty_concatenation(*args, **kwargs):
        raise AssertionError("Empty batch results must not accumulate")
    monkeypatch.setattr(batch.np, "concatenate", no_empty_concatenation)
    lower, upper = np.full((91, 3), 10.0), np.full((91, 3), 11.0)
    assert batch.segment_box_contact_indices([0, 0, 0], [0, 0, 0], lower, upper, 1, batch_size=7).size == 0


@pytest.mark.parametrize("function", [batch.segment_box_distances_sq, batch.segment_box_contact_indices, batch.first_segment_box_contact])
def test_cancellation_after_chunk_never_returns_partial_result(function, monkeypatch):
    original = batch._distances_chunk
    calls = []
    def recorded(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(batch, "_distances_chunk", recorded)
    lower, upper = np.zeros((11, 3)), np.ones((11, 3))
    arguments = ([0, 0, 0], [0, 0, 0], lower, upper)
    if function is not batch.segment_box_distances_sq:
        arguments += (1.0,)
    with pytest.raises(batch.IndependentBatchCancelled):
        function(*arguments, batch_size=3, cancelled=lambda: bool(calls))
    assert len(calls) == 1


def test_cancelled_empty_request_and_callback_exception_propagate():
    empty = np.empty((0, 3))
    with pytest.raises(batch.IndependentBatchCancelled):
        batch.segment_box_distances_sq([0, 0, 0], [0, 0, 0], empty, empty, cancelled=lambda: True)
    def broken():
        raise RuntimeError("callback failed")
    with pytest.raises(RuntimeError, match="callback failed"):
        batch.first_segment_box_contact([0, 0, 0], [0, 0, 0], empty, empty, 1, cancelled=broken)
    assert batch.segment_box_distances_sq([0, 0, 0], [0, 0, 0], empty, empty).shape == (0,)
    assert batch.segment_box_contact_indices([0, 0, 0], [0, 0, 0], empty, empty, 1).shape == (0,)
    assert batch.first_segment_box_contact([0, 0, 0], [0, 0, 0], empty, empty, 1) is None


@pytest.mark.parametrize("batch_size", [0, -1, 4097, 1.5, True])
def test_bad_batch_sizes_refused(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        batch.segment_box_distances_sq([0, 0, 0], [0, 0, 0], np.zeros((1, 3)), np.ones((1, 3)), batch_size=batch_size)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_boxes_after_first_collision_still_rejected(bad):
    lower, upper = np.zeros((2, 3)), np.ones((2, 3))
    lower[1, 0] = bad
    with pytest.raises(ValueError, match="finite ordered"):
        batch.first_segment_box_contact([0, 0, 0], [0, 0, 0], lower, upper, 1, batch_size=1)


@pytest.mark.parametrize("kind", ["dtype", "shape", "length", "list", "reversed"])
def test_malformed_boxes_refused(kind):
    lower, upper = np.zeros((2, 3)), np.ones((2, 3))
    if kind == "dtype": lower = lower.astype(np.float32)
    elif kind == "shape": lower = np.zeros((2, 4))
    elif kind == "length": lower = lower[:1]
    elif kind == "list": lower = lower.tolist()
    else: lower[1, 0] = 2
    with pytest.raises(ValueError):
        batch.segment_box_distances_sq([0, 0, 0], [0, 0, 0], lower, upper)


@pytest.mark.parametrize("radius,tolerance", [(-1, 0), (1, -1), (np.inf, 0), (1, np.nan), (True, 0), (np.array(1), 0), (1e308, 0)])
def test_invalid_threshold_refused(radius, tolerance):
    with pytest.raises(ValueError):
        batch.first_segment_box_contact([0, 0, 0], [0, 0, 0], np.zeros((1, 3)), np.ones((1, 3)), radius, tolerance_sq=tolerance)


def test_finite_arithmetic_overflow_fails_closed():
    with pytest.raises(ValueError, match="arithmetic"):
        batch.segment_box_distances_sq([-1e308, 0, 0], [1e308, 0, 0], np.zeros((1, 3)), np.ones((1, 3)))


def test_no_input_mutation_or_shared_output():
    start, end = np.zeros(3), np.ones(3)
    lower, upper = np.zeros((5, 3)), np.ones((5, 3))
    originals = [value.copy() for value in (start, end, lower, upper)]
    actual = batch.segment_box_distances_sq(start, end, lower, upper)
    actual[:] = 9
    for current, original in zip((start, end, lower, upper), originals):
        np.testing.assert_array_equal(current, original)
