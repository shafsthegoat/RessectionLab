"""Independent analytic controls for the unwired numerical prototype.

Written during the PAT05 quiet window. Running these requires the orchestrator's
release; this file neither invokes the timing probe nor loads patient cases.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab import evaluation


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


batch = _load("independent_batch_review", ROOT / "scripts/independent_geometry_batch_prototype.py")


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_arbitrary_segment_to_degenerate_point_boxes_has_closed_form_distance(reverse, order):
    start = np.array([-2., 1., -3.])[list(order)]
    end = np.array([4., -2., 6.])[list(order)]
    points = np.array([[-2., 1., -3.], [4., -2., 6.], [1., 2., 0.],
                       [-5., 4., -9.], [8., -4., 13.], [0., 0., 0.]])[:, list(order)]
    direction = end - start
    parameters = np.clip(((points - start) @ direction) / np.dot(direction, direction), 0., 1.)
    residuals = points - (start + parameters[:, None] * direction)
    expected = np.sum(residuals * residuals, axis=1)
    if reverse:
        start, end = end, start
    actual = batch.segment_box_distances_sq(start, end, points, points, batch_size=2)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_collapsed_plane_and_line_boxes_have_internal_stationary_minima():
    # Segment is y=x,z=0. The closest points on the boxes are respectively
    # (1,3,0), (2,0,1), and (2,2,0), giving squared distances 2, 3, and 0.
    lower = np.array([[0., 3., 0.], [2., 0., 1.], [2., 2., 0.]])
    upper = np.array([[1., 4., 0.], [3., 0., 1.], [2., 2., 0.]])
    actual = batch.segment_box_distances_sq([0, 0, 0], [4, 4, 0], lower, upper, batch_size=2)
    np.testing.assert_array_equal(actual, [2., 3., 0.])


def test_readonly_negative_strides_and_duplicate_boxes_preserve_input_witness():
    # The order below is intentionally unrelated to geometric proximity. The
    # reversed borrowed view starts with a miss then two identical contacts.
    lower = np.array([[0., 0., 0.], [8., 8., 8.], [0., 0., 0.],
                      [0., 0., 0.], [4., 4., 4.]])[::-1]
    upper = (lower[::-1] + .25)[::-1]
    lower.flags.writeable = False
    upper.flags.writeable = False
    np.testing.assert_array_equal(batch.segment_box_contact_indices(
        [0, 0, 0], [0, 0, 0], lower, upper, 0., tolerance_sq=0., batch_size=2), [1, 2, 4])
    assert batch.first_segment_box_contact(
        [0, 0, 0], [0, 0, 0], lower, upper, 0., tolerance_sq=0., batch_size=2) == 1


@pytest.mark.parametrize("radius", [np.nextafter(np.sqrt(2.), 0.), np.sqrt(2.),
                                    np.nextafter(np.sqrt(2.), np.inf)])
def test_arbitrary_direction_tangency_defers_exact_contact_to_scalar(radius, monkeypatch):
    start, end = np.full(3, -1.), np.ones(3)
    point = np.array([[0., 1., -1.]])  # perpendicular to the segment at its midpoint
    expected = evaluation.segment_box_distance_sq(start, end, point[0], point[0]) <= radius * radius
    original = evaluation.segment_box_distance_sq
    calls = []

    def recorded(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(evaluation, "segment_box_distance_sq", recorded)
    actual = batch.segment_box_contact_indices(start, end, point, point, radius, tolerance_sq=0.)
    assert bool(len(actual)) == expected
    assert calls == [True]


@pytest.mark.parametrize("name", ["segment_box_distances_sq", "segment_box_contact_indices",
                                  "first_segment_box_contact"])
def test_cancellation_during_validation_prevents_kernel_or_partial_first_witness(name, monkeypatch):
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 3

    def forbidden_kernel(*args):
        raise AssertionError("A cancelled validation pass must not enter the kernel")

    monkeypatch.setattr(batch, "_distances_chunk", forbidden_kernel)
    lower, upper = np.zeros((9, 3)), np.ones((9, 3))
    args = ([0, 0, 0], [0, 0, 0], lower, upper)
    if name != "segment_box_distances_sq":
        args += (0.,)
    with pytest.raises(batch.IndependentBatchCancelled):
        getattr(batch, name)(*args, batch_size=2, cancelled=cancelled)
    assert calls == 3


@pytest.mark.parametrize("module_name", ["evaluation", "prototype"])
def test_probe_refuses_receipting_a_different_loaded_source_file(module_name):
    probe = _load("independent_batch_probe_review", ROOT / "scripts/probe_independent_geometry_batch.py")
    if module_name == "evaluation":
        probe.evaluation = SimpleNamespace(__file__=str(ROOT / "other-checkout/evaluation.py"))
    else:
        probe.batch = SimpleNamespace(__file__=str(ROOT / "other-checkout/prototype.py"))
    with pytest.raises((ValueError, RuntimeError), match="(?i)source|origin|import|path"):
        probe._source_files()
