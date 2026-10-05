"""Research descriptors only; no registered profile, patient or learned policy."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.native_spatial_features import (access_source_frame, candidate_coordinates,
    cavity_residual_summary, descriptor_contract)


def masks():
    tissue = np.ones((7, 7, 8), bool)
    target = np.zeros(tissue.shape, bool); target[1:6, 1:6, 2:7] = True
    removed = np.zeros(tissue.shape, bool); removed[3, 3, :5] = True
    return tissue, target, removed


def test_coordinates_expose_known_different_entries_and_permute_with_rows():
    frame = access_source_frame(np.eye(4), (3, 3, -.5), (0, 0, 1))
    ids = ("STOP", "left", "right")
    entries = np.asarray(((3, 3, -.5), (3, 3, -.5), (4, 3, -.5)))
    tips = np.asarray(((3, 3, -.5), (3, 3, 6), (4, 3, 6)))
    features = candidate_coordinates(ids, entries, tips, frame)
    np.testing.assert_array_equal(features[0], np.zeros(6))
    np.testing.assert_array_equal(features[1], (0, 0, 0, 0, 0, 6.5))
    np.testing.assert_array_equal(features[2], (1, 0, 0, 1, 0, 6.5))
    order = np.asarray((2, 0, 1))
    changed = candidate_coordinates(tuple(ids[i] for i in order), entries[order], tips[order], frame)
    np.testing.assert_array_equal(changed, features[order])


@pytest.mark.parametrize("rotation,shift", [
    (np.eye(3), np.asarray((71., -18., 3.))),
    (np.diag((-1., -1., 1.)), np.zeros(3)),
    (np.asarray(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.))), np.asarray((5., 8., -12.))),
])
def test_rigid_coordinate_change_preserves_candidate_and_state_values(rotation, shift):
    affine = np.diag((2., 3., 4., 1.))
    center, normal = np.asarray((3., 3., -.5)), np.asarray((0., 0., 1.))
    frame = access_source_frame(affine, center, normal)
    entries = np.asarray(((3, 3, -.5), (5, 4, -.5)))
    tips = np.asarray(((4, 5, 6), (6, 3, 9)))
    before = candidate_coordinates(("a", "b"), entries, tips, frame)
    state = cavity_residual_summary(*masks(), affine, frame)
    transform = np.eye(4); transform[:3, :3] = rotation; transform[:3, 3] = shift
    new_affine = transform @ affine
    new_frame = access_source_frame(new_affine, rotation @ center + shift, rotation @ normal)
    np.testing.assert_allclose(candidate_coordinates(("a", "b"), entries @ rotation.T + shift,
                                                    tips @ rotation.T + shift, new_frame), before, atol=1e-12)
    np.testing.assert_allclose(cavity_residual_summary(*masks(), new_affine, new_frame), state, atol=1e-10)


def test_distinct_occupancy_can_still_alias_the_entire_proposed_state_summary():
    tissue, target, _ = masks()
    a, b = np.zeros_like(tissue), np.zeros_like(tissue)
    a[2:4, 2:4, 3] = True
    b[1, 2:4, 3] = True; b[4, 2:4, 3] = True
    frame = access_source_frame(np.eye(4), (3, 3, -.5), (0, 0, 1))
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(cavity_residual_summary(tissue, target, a, np.eye(4), frame),
                                  cavity_residual_summary(tissue, target, b, np.eye(4), frame))
    # Descriptor counterexample only: no legal-native-cavity reachability claim.


def test_empty_regions_and_physical_volume_are_explicit():
    tissue, target, _ = masks()
    affine = np.diag((2., 3., 4., 1.))
    frame = access_source_frame(affine, (0, 0, 0), (0, 0, 1))
    summary = cavity_residual_summary(tissue, target, np.zeros_like(tissue), affine, frame)
    np.testing.assert_array_equal(summary[:4], np.zeros(4))
    assert summary[4] == pytest.approx(125 * 24)
    all_removed = cavity_residual_summary(tissue, target, tissue.copy(), affine, frame)
    np.testing.assert_array_equal(all_removed[4:], np.zeros(4))
    assert all_removed[0] == pytest.approx(np.count_nonzero(tissue) * 24)


def test_source_axis_order_is_declared_and_not_falsely_invariant():
    original = np.eye(4)
    reordered = original.copy(); reordered[:3, [0, 1]] = reordered[:3, [1, 0]]
    a = access_source_frame(original, (3, 3, -.5), (0, 0, 1))
    b = access_source_frame(reordered, (3, 3, -.5), (0, 0, 1))
    args = (("right",), ((4, 3, -.5),), ((4, 3, 6),))
    assert not np.array_equal(candidate_coordinates(*args, a), candidate_coordinates(*args, b))
    assert descriptor_contract()["markov_completeness_claimed"] is False
    assert descriptor_contract()["production_input_profiles_changed"] is False


def test_malformed_geometry_or_removed_tissue_is_rejected():
    with pytest.raises(ValueError, match="nonzero"):
        access_source_frame(np.eye(4), (0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError, match="invertible"):
        access_source_frame(np.zeros((4, 4)) + np.diag((0, 1, 1, 1)), (0, 0, 0), (0, 0, 1))
    frame = access_source_frame(np.eye(4), (0, 0, 0), (0, 0, 1))
    with pytest.raises(ValueError, match="Finite"):
        candidate_coordinates(("x",), ((np.nan, 0, 0),), ((0, 0, 0),), frame)
    tissue, target, removed = masks(); tissue[0, 0, 0] = False; removed[0, 0, 0] = True
    with pytest.raises(ValueError, match="declared tissue"):
        cavity_residual_summary(tissue, target, removed, np.eye(4), frame)
