"""Independent tiny spatial descriptor checks; no policy, gradients or patients."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.native_spatial_features import access_source_frame, candidate_coordinates, cavity_residual_summary
from scripts.probe_native_spatial_features import fixture


def physical_fixture():
    # Invertible left-handed and sheared source grid. Physical cell volume is
    # 24 mm3; the declared access basis is exactly (world+y, world-x, world+z).
    affine = np.array([[0., -3., .5, 10.], [2., 0., 0., 20.],
                       [0., 0., -4., 30.], [0., 0., 0., 1.]])
    center = np.array([10., 20., 30.])
    normal = np.array([0., 0., 5.])
    tissue = np.ones((2, 2, 2), dtype=bool)
    target = np.zeros_like(tissue)
    target[0, 0, 0] = target[1, 1, 1] = target[1, 0, 1] = True
    removed = np.zeros_like(tissue)
    removed[0, 0, 0] = removed[0, 1, 0] = True
    return affine, center, normal, tissue, target, removed


def test_signed_sheared_affine_uses_physical_centers_and_absolute_cell_volume():
    affine, center, normal, tissue, target, removed = physical_fixture()
    frame = access_source_frame(affine, center, normal)
    expected_basis = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    np.testing.assert_array_equal(frame.basis_columns, expected_basis)
    assert np.linalg.det(affine[:3, :3]) == pytest.approx(-24.)
    np.testing.assert_allclose(cavity_residual_summary(tissue, target, removed, affine, frame),
                              [48., 0., 1.5, 0., 48., 2., 1., -4.], rtol=0, atol=1e-12)
    # Voxel(1,1,1) has independently calculated physical position(7.5,22,26).
    result = candidate_coordinates(("physical-cut",), [center], [[7.5, 22., 26.]], frame)
    np.testing.assert_array_equal(result, [[0., 0., 0., 2., 2.5, -4.]])


def test_oblique_three_dimensional_rigid_transform_of_sheared_source_matches_hand_basis():
    affine, center, normal, tissue, target, removed = physical_fixture()
    axis = np.array([1., 2., 3.]); axis /= np.linalg.norm(axis)
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    angle = .64
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    transform = np.eye(4); transform[:3, :3] = rotation; transform[:3, 3] = [17., -21., 9.]
    changed_affine = transform @ affine
    frame = access_source_frame(changed_affine, rotation @ center + transform[:3, 3], rotation @ normal)
    expected_basis = rotation @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    np.testing.assert_allclose(frame.basis_columns, expected_basis, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(frame.basis_columns).T @ frame.basis_columns, np.eye(3), rtol=0, atol=1e-14)
    assert np.linalg.det(frame.basis_columns) == pytest.approx(1.)
    np.testing.assert_allclose(cavity_residual_summary(tissue, target, removed, changed_affine, frame),
                              [48., 0., 1.5, 0., 48., 2., 1., -4.], rtol=0, atol=1e-12)


def test_normal_magnitude_and_source_column_scale_do_not_change_physical_coordinate_basis():
    affine, center, normal, *_ = physical_fixture()
    base = access_source_frame(affine, center, normal)
    scaled = affine.copy(); scaled[:3, :3] *= np.array([7., .2, 3.])
    changed = access_source_frame(scaled, center, normal * 73.)
    np.testing.assert_array_equal(base.basis_columns, changed.basis_columns)
    np.testing.assert_array_equal(base.project([[1., 3., 4.]]), changed.project([[1., 3., 4.]]))


def test_zero_centroid_at_access_origin_is_distinguished_from_empty_by_volume():
    tissue = np.ones((1, 1, 1), dtype=bool)
    affine = np.diag([2., 3., 4., 1.]); affine[:3, 3] = [7., 8., 9.]
    frame = access_source_frame(affine, [7., 8., 9.], [0., 0., 1.])
    empty_cavity = cavity_residual_summary(tissue, tissue, ~tissue, affine, frame)
    full_cavity = cavity_residual_summary(tissue, tissue, tissue, affine, frame)
    np.testing.assert_allclose(empty_cavity, [0., 0., 0., 0., 24., 0., 0., 0.], atol=1e-12)
    np.testing.assert_allclose(full_cavity, [24., 0., 0., 0., 0., 0., 0., 0.], atol=1e-12)


def test_near_parallel_basis_has_a_retained_conditioning_counterexample():
    # Deliberate limitation, not an invariance assertion: two almost identical
    # access normals can reverse both tangent coordinates near a parallel axis.
    positive = access_source_frame(np.eye(4), [0., 0., 0.], [1., 2e-8, 0.])
    negative = access_source_frame(np.eye(4), [0., 0., 0.], [1., -2e-8, 0.])
    point = [[0., 1., 1.]]
    assert positive.source_tangent_axis == negative.source_tangent_axis == 0
    assert np.max(np.abs(positive.project(point) - negative.project(point))) > 1.99


def test_frozen_evidence_archive_and_report_match_captured_source_bytes():
    directory = ROOT / "artifacts/native-spatial-feature-probe-v1"
    capture = json.loads((directory / "source-capture.json").read_text())
    verification = json.loads((directory / "archive-verification.json").read_text())
    report = json.loads((directory / "probe/report.json").read_text())
    assert hashlib.sha256((directory / "source.tar.gz").read_bytes()).hexdigest() == verification["archive_sha256"]
    assert hashlib.sha256((directory / "probe/report.json").read_bytes()).hexdigest() == verification["report_sha256"]
    with tarfile.open(directory / "source.tar.gz") as archive:
        members = archive.getmembers()
        assert len(members) == verification["verified_file_count"]
        assert all(member.isfile() for member in members)
        hashes = {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest() for member in members}
    assert hashes == capture["files"]
    for path, digest in report["source_receipt"]["files"].items():
        assert hashes[path] == digest


def test_reward_alias_difference_is_independently_explained_by_new_partial_contact_cells():
    sim = fixture()
    cfg = sim.native_config
    initial = sim.observation()
    np.testing.assert_array_equal(initial.action_features[1], initial.action_features[3])
    assert sim.config.evidence_available == (False, False)
    best_totals, first_rewards, partial_normals = [], [], []

    def execute_and_check(branch, action_id):
        if action_id == "STOP":
            transition = branch.step(action_id)
            assert transition.reward == 0.
            return transition
        action = next(a for a in branch.proposed_actions() if a.action_id == action_id)
        removed = {tuple(row) for row in action.removal_indices}
        new_partial = {tuple(row) for row in action.swept_indices
            if not branch.engine.removed_mask[tuple(row)] and not branch._partial_contact_mask[tuple(row)]} - removed
        target = sum(cfg.target_labels[index] > 0 for index in removed)
        normal = len(removed) - target
        partial = sum(cfg.target_labels[index] == 0 for index in new_partial)
        tool_change = branch._current_tool is not None and branch._current_tool != action.tool_id
        expected = target - .2 * normal - .05 * .2 * partial - .02 - .02 * tool_change
        transition = branch.step(action_id)
        assert transition.reward == pytest.approx(expected, abs=1e-12)
        assert transition.info["target_removed_mm3"] == target
        assert transition.info["normal_removed_mm3"] == normal
        assert transition.info["partial_normal_contact_mm3"] == partial
        return transition

    for index in (1, 3):
        branch = sim.clone()
        first = execute_and_check(branch, initial.action_ids[index])
        first_rewards.append(first.reward)
        assert first.info["target_removed_mm3"] == 5
        assert first.info["normal_removed_mm3"] == 2
        assert first.info["partial_normal_contact_mm3"] == 25
        candidates = []
        for action_id in branch.observation().action_ids:
            last = execute_and_check(branch.clone(), action_id)
            assert last.terminated
            candidates.append(last)
        best = max(candidates, key=lambda item: item.reward)
        assert best.info["target_removed_mm3"] == 40 and best.info["normal_removed_mm3"] == 21
        partial_normals.append(best.info["partial_normal_contact_mm3"])
        best_totals.append(first.reward + best.reward)
    np.testing.assert_allclose(first_rewards, [4.33, 4.33], rtol=0, atol=1e-12)
    np.testing.assert_allclose(best_totals, [39.43, 39.68], rtol=0, atol=1e-12)
    assert partial_normals == [66., 41.]
    assert best_totals[1] - best_totals[0] == pytest.approx(.05 * .2 * (66 - 41))
