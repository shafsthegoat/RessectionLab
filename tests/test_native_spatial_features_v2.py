"""Isolated explicit-tangent checks; never construct a simulator or policy."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.native_spatial_features_v2 import (DeclaredTangent, FrameConditioningError,
    MIN_TANGENT_SINE, ROUNDOFF_REJECTION_HALF_WIDTH, declared_access_frame,
    candidate_coordinates, cavity_residual_summary, descriptor_contract)


SOURCE = "synthetic-spatial-conditioning-v2"


def reference(vector, frame="RAS+"):
    return DeclaredTangent(vector, SOURCE, frame, "explicit-test-reference")


def frame(normal, tangent, center=(0, 0, 0), label="RAS+"):
    return declared_access_frame(center, normal, reference=reference(tangent, label),
                                 expected_source_hash=SOURCE, coordinate_frame=label)


def counterexample():
    return json.loads((ROOT / "artifacts/native-spatial-feature-independent-v1/conditioning-counterexample.json").read_text())["threshold_rotation_counterexample"]


def test_saved_rotation_counterexample_has_stable_explicit_y_reference():
    saved = counterexample()
    q, n, p = (np.asarray(saved[key]) for key in ("rotation", "normal", "points_mm"))
    tangent = np.asarray((0., 1., 0.))
    original, changed = frame(n, tangent), frame(q @ n, q @ tangent, label="ROTATED")
    np.testing.assert_allclose(original.project(p), changed.project(p @ q.T), rtol=0, atol=1e-12)
    for actual in (original, changed):
        assert actual.to_dict()["conditioning"]["tangent_normalization_factor"] < 10
        assert actual.to_dict()["conditioning"]["automatic_axis_selection"] is False


def test_saved_near_parallel_x_reference_rejects_before_any_axis_switch():
    saved = counterexample()
    q, n = np.asarray(saved["rotation"]), np.asarray(saved["normal"])
    for normal, tangent, label in ((n, (1, 0, 0), "RAS+"), (q @ n, q @ np.asarray((1, 0, 0)), "ROTATED")):
        with pytest.raises(FrameConditioningError) as caught:
            frame(normal, tangent, label=label)
        receipt = caught.value.receipt
        assert receipt["rejection_reason"] == "declared_tangent_below_engineering_condition_limit"
        assert receipt["reference"]["coordinate_frame"] == label
        assert receipt["automatic_axis_selection"] is False


@pytest.mark.parametrize("offset", [-0.5, 0., 0.5])
def test_engineering_roundoff_band_is_explicit_and_rejects(offset):
    sine = MIN_TANGENT_SINE + offset * ROUNDOFF_REJECTION_HALF_WIDTH
    with pytest.raises(FrameConditioningError, match="roundoff_rejection_band"):
        frame((1, 0, 0), (np.sqrt(1-sine*sine), sine, 0))


def test_well_separated_boundary_cases_reject_or_accept_without_fallback():
    with pytest.raises(FrameConditioningError, match="below_engineering"):
        frame((1, 0, 0), (np.sqrt(1-.099**2), .099, 0))
    accepted = frame((1, 0, 0), (np.sqrt(1-.101**2), .101, 0))
    assert accepted.tangent_sine == pytest.approx(.101)
    assert accepted.to_dict()["conditioning"]["tangent_normalization_factor"] < 10


def test_reference_required_bound_and_immutable_without_inferred_axis():
    with pytest.raises(TypeError):
        declared_access_frame((0, 0, 0), (0, 0, 1), expected_source_hash=SOURCE, coordinate_frame="RAS+")
    for source, label in (("wrong-source", "RAS+"), (SOURCE, "LPS+")):
        with pytest.raises(ValueError, match="source/frame"):
            declared_access_frame((0, 0, 0), (0, 0, 1), reference=reference((1, 0, 0)),
                                  expected_source_hash=source, coordinate_frame=label)
    vector = [1., 0., 0.]
    declared = reference(vector); vector[0] = 0
    assert declared.vector == (1., 0., 0.)
    with pytest.raises(ValueError, match="zero"):
        reference((0, 0, 0))


def test_reference_and_normal_scaling_does_not_change_frame():
    base = frame((0, 0, 1), (1, 2, 0))
    for scale in (1e-250, 1e250):
        altered = frame((0, 0, scale), (scale, 2*scale, 0))
        np.testing.assert_allclose(altered.basis_columns, base.basis_columns, rtol=0, atol=1e-15)


def test_translation_lps_and_row_permutation_keep_declared_reference():
    ids = ("STOP", "left", "right")
    center = np.asarray((3., 3., -.5))
    entries = np.asarray((center, center, center+(1, 0, 0)))
    tips = entries + (0, 0, 6.5)
    n, tangent = np.asarray((0., 0., 1.)), np.asarray((1., 0., 0.))
    original = candidate_coordinates(ids, entries, tips, frame(n, tangent, center))
    q, shift = np.diag((-1., -1., 1.)), np.asarray((17., -8., 2.))
    changed = frame(q @ n, q @ tangent, q @ center + shift, "LPS+")
    transformed = candidate_coordinates(ids, entries @ q.T + shift, tips @ q.T + shift, changed)
    np.testing.assert_array_equal(transformed, original)
    order = [2, 0, 1]
    np.testing.assert_array_equal(candidate_coordinates(tuple(ids[i] for i in order), entries[order], tips[order],
                                                       frame(n, tangent, center)), original[order])


def test_source_reindexing_is_invariant_only_when_physical_reference_is_preserved():
    tissue = np.ones((3, 4, 5), bool)
    target = np.zeros_like(tissue); target[1:3, 1:4, 2:5] = True
    removed = np.zeros_like(tissue); removed[1, 2, :3] = True
    affine = np.diag((2., 3., 4., 1.)); affine[:3, 3] = (7, -4, 8)
    declared = frame((0, 0, 1), (1, 0, 0), (7, -4, 8))
    original = cavity_residual_summary(tissue, target, removed, affine, declared)
    reordered = affine.copy(); reordered[:3, [0, 1]] = reordered[:3, [1, 0]]
    actual = cavity_residual_summary(tissue.transpose(1, 0, 2), target.transpose(1, 0, 2),
                                    removed.transpose(1, 0, 2), reordered, declared)
    np.testing.assert_allclose(actual, original, rtol=0, atol=1e-10)


def test_lossy_state_summary_remains_lossy_and_scope_stays_explicit():
    tissue = np.ones((7, 7, 8), bool)
    target = np.zeros_like(tissue); target[1:6, 1:6, 2:7] = True
    a, b = np.zeros_like(tissue), np.zeros_like(tissue)
    a[2:4, 2:4, 3] = True; b[1, 2:4, 3] = True; b[4, 2:4, 3] = True
    declared = frame((0, 0, 1), (1, 0, 0), (3, 3, -.5))
    np.testing.assert_array_equal(cavity_residual_summary(tissue, target, a, np.eye(4), declared),
                                  cavity_residual_summary(tissue, target, b, np.eye(4), declared))
    contract = descriptor_contract()
    assert set(("partial_contact_history", "current_tool", "remaining_action_budget")) <= set(contract["omitted_state"])
    assert not contract["global_cross_patient_tangent_convention_claimed"]
    assert not contract["production_input_profiles_changed"] and not contract["markov_completeness_claimed"]
