"""Independent small-grid ingress checks; never patient data or stroke previews."""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from resectionlab.core import array_digest
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_ingress import IngressCandidate, screen_axis_accesses
from resectionlab.native_proposals import NominalCavityProposalConfig, PreparedNominalCavityProposer
from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine


TOOLS = (ToolGeometry("narrow", .7, .2, 16., 35., 1.),
         ToolGeometry("broad", 1.2, .8, 16., 35., 1.5))


def exits(*, tissue=None, offsets=((0, 0),), affine=None, tools=TOOLS,
          max_candidates=96, entries=None, index_affine=None, frame_record=None):
    if tissue is None:
        tissue = np.zeros((13, 13, 13), bool)
        tissue[4:9, 4:9, 4:9] = True
    affine = np.eye(4) if affine is None else affine
    original = affine if index_affine is None else index_affine
    nominal = np.zeros(tissue.shape, np.float32)
    nominal[5:8, 5:8, 5:8] = .75
    result = []
    for axis, sign in product(range(3), (-1, 1)):
        point = np.array([6., 6., 6.])
        point[axis] = 3.25 if sign == -1 else 8.75
        if entries and (axis, sign) in entries:
            point = np.array(entries[axis, sign], float)
        normal = -sign * original[:3, axis] / np.linalg.norm(original[:3, axis])
        access = AccessWindow(original[:3, :3] @ point + original[:3, 3], normal, 5.)
        config = NativeResectionConfig(tissue, np.zeros(tissue.shape, np.int16), affine,
            access, tools, "analytical-permitted-source", "analytical support only")
        proposer = PreparedNominalCavityProposer(config, nominal,
            nominal_provenance={"source_hash": config.source_hash,
                "nominal_target_hash": array_digest(nominal), "source_kind": "supplied_annotation",
                "derivation": "explicit analytical nominal"},
            config=NominalCavityProposalConfig(offsets, max_candidates),
            index_affine=index_affine, index_frame_record=frame_record)
        result.append(IngressCandidate(axis, sign, 3., NativeResectionEngine(config), proposer))
    return result


def test_six_labels_cannot_disguise_one_physical_access():
    actual = exits()[0]
    mislabeled = [replace(actual, axis=axis, outward_sign=sign,
        distance_mm=1. if (axis, sign) == (2, 1) else 3.)
        for axis, sign in product(range(3), (-1, 1))]
    result = screen_axis_accesses(mislabeled)
    assert result.status == "failed", result.to_dict()
    assert result.selected_exit is None and not result.complete


def test_opposite_metadata_sign_is_not_the_screened_access():
    candidates = exits()
    candidates[0], candidates[1] = (
        replace(candidates[1], outward_sign=-1), replace(candidates[0], outward_sign=1))
    result = screen_axis_accesses(candidates)
    assert result.status == "failed", result.to_dict()
    assert result.selected_exit is None


def row_for(result, key):
    return next(row for row in result.to_dict()["exits"]
                if (row["axis"], row["outward_sign"]) == key)


@pytest.mark.parametrize("transformed", [False, True])
def test_step_zero_whole_cell_shaft_matches_independent_axis_segment_box_distance(transformed):
    tissue = np.zeros((13, 13, 13), bool)
    tissue[4:9, 4:9, 4:9] = True
    tissue[6, 6, 0] = tissue[7, 7, 0] = tissue[12, 6, 6] = True
    affine = np.eye(4)
    if transformed:
        angle = .31
        affine[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0],
                                  [np.sin(angle), np.cos(angle), 0], [0, 0, -1]]) @ np.diag([1.2, 1.4, .9])
        affine[:3, 3] = [31., -17., 9.]
    candidates = exits(tissue=tissue, affine=affine)
    result = screen_axis_accesses(candidates)
    assert result.complete, result.error
    occupied = np.argwhere(tissue)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    inverse = np.linalg.inv(affine[:3, :3])
    for candidate in candidates:
        for pose in row_for(result, candidate.key)["poses"]:
            tool = next(tool for tool in TOOLS if tool.tool_id == pose["tool_id"])
            entry, direction = np.asarray(pose["entry_mm"]), np.asarray(pose["axis_unit"])
            endpoints = np.array([entry - tool.working_length_mm * direction,
                                  entry - tool.tip_length_mm * direction])
            voxel_endpoints = (endpoints - affine[:3, 3]) @ inverse.T
            # Axial line-to-box distance, independently of geometry's capsule routine.
            segment_min, segment_max = voxel_endpoints.min(0), voxel_endpoints.max(0)
            gaps = np.maximum(np.maximum(occupied - .5 - segment_max,
                                         segment_min - occupied - .5), 0) * spacing
            expected = occupied[np.sum(gaps ** 2, axis=1) <= tool.shaft_radius_mm ** 2]
            assert pose["blocked_cell_count"] == len(expected)
            assert pose["first_blocked_cell"] == (None if not len(expected) else expected[0].tolist())
            assert ("SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE" in pose["reasons"]) == bool(len(expected))
            if len(expected):
                np.testing.assert_allclose(pose["first_blocked_cell_mm"],
                    affine[:3, :3] @ expected[0] + affine[:3, 3], rtol=0, atol=1e-12)


def test_any_noncentral_entry_can_keep_shortest_exit_eligible():
    tissue = np.zeros((13, 13, 13), bool)
    tissue[4:9, 4:9, 4:9] = True
    tissue[6, 6, 0] = True
    candidates = [replace(c, distance_mm=.5 if c.key == (2, -1) else 3.)
                  for c in exits(tissue=tissue, offsets=((0, 0), (2, 0)))]
    result = screen_axis_accesses(candidates)
    chosen = row_for(result, (2, -1))
    assert result.selected_exit == (2, -1), result.error
    assert chosen["eligible"]
    center = [p for p in chosen["poses"] if p["entry_mm"][0] == 6.]
    offset = [p for p in chosen["poses"] if p["entry_mm"][0] == 8.]
    assert center and offset and all(not p["admissible"] for p in center)
    assert any(p["admissible"] for p in offset)


def test_any_one_tool_is_sufficient_without_majority_vote():
    tissue = np.zeros((13, 13, 13), bool)
    tissue[4:9, 4:9, 4:9] = True
    tissue[7, 6, 0] = True
    candidates = [replace(c, distance_mm=.5 if c.key == (2, -1) else 3.)
                  for c in exits(tissue=tissue)]
    result = screen_axis_accesses(candidates)
    poses = row_for(result, (2, -1))["poses"]
    assert result.selected_exit == (2, -1), result.error
    assert all(p["admissible"] for p in poses if p["tool_id"] == "narrow")
    assert all(not p["admissible"] for p in poses if p["tool_id"] == "broad")


def test_exact_pose_partition_not_number_of_endpoints_and_no_preview_or_commit(monkeypatch):
    import resectionlab.native_ingress as ingress
    candidates = exits(offsets=((0, 0), (1, 0)))
    before = [(c.engine.state_hash, c.engine.config.fingerprint,
               *(array_digest(getattr(c.engine, key)) for key in
                 ("remaining_mask", "removed_mask", "contact_mask", "connected_free_mask"))) for c in candidates]
    calls = []
    real = ingress.check_pose
    def counted(tool, pose, scene, access):
        calls.append((tool.tool_id, tuple(pose.tip_mm)))
        return real(tool, pose, scene, access)
    def forbidden(*args, **kwargs):
        raise AssertionError("Full native preview/commit is outside the ingress screen")
    monkeypatch.setattr(ingress, "check_pose", counted)
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", forbidden)
    monkeypatch.setattr(NativeResectionEngine, "commit_preview", forbidden)
    result = screen_axis_accesses(candidates)
    expected_count = 0
    for candidate in candidates:
        rays = candidate.proposer.propose(candidate.engine).proposals
        signatures = set()
        for ray in rays:
            entry = np.asarray(ray.entry_mm, np.float64)
            axis = np.asarray(ray.tip_mm, np.float64) - entry
            axis /= np.linalg.norm(axis)
            signatures.add((ray.tool_id, entry.tobytes(), axis.tobytes()))
        expected_count += len(signatures)
        assert len(signatures) < len(rays)
    assert result.complete and result.screen_count == len(calls) == expected_count, result.error
    after = [(c.engine.state_hash, c.engine.config.fingerprint,
              *(array_digest(getattr(c.engine, key)) for key in
                ("remaining_mask", "removed_mask", "contact_mask", "connected_free_mask"))) for c in candidates]
    assert before == after
    assert all(c.engine.revision == 0 and c.engine.history == [] for c in candidates)


def test_unknown_outside_geometry_remains_unknown_and_cannot_authorize_removal():
    result = screen_axis_accesses(exits())
    receipt = result.to_dict()
    assert result.complete and result.selected_exit is not None
    assert not receipt["complete_stroke_certified"] and not receipt["removal_authorized"]
    assert "outside-image" in receipt["scope"]
    assert all("tool_geometry_outside_image_unassessed" in p["geometry"]["unknowns"]
               for r in receipt["exits"] for p in r["poses"])


@pytest.mark.parametrize("change", ["remaining", "nominal", "metadata", "provider_axis"])
def test_final_callback_cannot_publish_selection_after_source_or_state_mutation(monkeypatch, change):
    import resectionlab.native_ingress as ingress
    candidates = exits()
    count = 0
    real = ingress.check_pose
    def counted(*args, **kwargs):
        nonlocal count
        count += 1
        return real(*args, **kwargs)
    monkeypatch.setattr(ingress, "check_pose", counted)
    def mutate():
        if count == 12:
            if change == "remaining":
                candidates[0].engine.remaining_mask[6, 6, 6] = False
            elif change == "nominal":
                candidates[0].proposer._nominal = np.zeros((13, 13, 13), np.float32)
            elif change == "metadata":
                object.__setattr__(candidates[0], "distance_mm", .01)
            else:
                candidates[0].proposer._axis = 1
        return False
    result = screen_axis_accesses(candidates, cancelled=mutate)
    assert count == 12
    assert result.status == "failed" and result.selected_exit is None, result.to_dict()


def test_cancel_after_all_geometries_still_has_no_partial_selection(monkeypatch):
    import resectionlab.native_ingress as ingress
    count = 0
    real = ingress.check_pose
    def counted(*args, **kwargs):
        nonlocal count
        count += 1
        return real(*args, **kwargs)
    monkeypatch.setattr(ingress, "check_pose", counted)
    result = screen_axis_accesses(exits(), cancelled=lambda: count == 12)
    assert count == 12 and result.status == "interrupted"
    assert not result.complete and result.selected_exit is None


@pytest.mark.parametrize("kind", ["slot_bound", "candidate_cap", "tools"])
def test_incomplete_late_inventory_invalidates_earlier_clear_results(kind):
    candidates = exits()
    if kind == "slot_bound":
        # 14 columns * 2 tools * 3 families, including out-of-image slots.
        bad = exits(offsets=tuple((i, 0) for i in range(14)))
    elif kind == "candidate_cap":
        bad = exits(max_candidates=1)
    else:
        third = ToolGeometry("third", .6, .1, 16., 35., 1.)
        bad = exits(tools=TOOLS + (third,))
    candidates[-1] = bad[-1]
    result = screen_axis_accesses(candidates)
    assert result.status == "failed" and result.selected_exit is None
    assert any(row["eligible"] for row in result.exit_records)


def test_unrelated_reward_private_reference_crop_objects_are_never_consulted():
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Forbidden evaluator or crop array inspected")
        def __getattr__(self, name):
            raise AssertionError("Forbidden evaluator or reward object inspected")
    candidates = exits()
    baseline = screen_axis_accesses(candidates)
    for candidate in candidates:
        for name in ("reference_target", "reward", "crop", "policy", "actor_inputs"):
            setattr(candidate.engine, name, Poison())
    changed = screen_axis_accesses(candidates)
    assert baseline.to_dict() == changed.to_dict()


def reconciled_fixture():
    original = np.eye(4)
    original[0, 1] = 4e-10
    spacing = np.linalg.norm(original[:3, :3], axis=0)
    left, _, right = np.linalg.svd(original[:3, :3] / spacing)
    native = original.copy()
    native[:3, :3] = (left @ right) * spacing
    record = {"method": "orthogonal_roundoff_1e-6mm", "shape": [13, 13, 13],
              "original_affine_hash": array_digest(original), "derived_affine_hash": array_digest(native),
              "original_affine_ras_mm": original.tolist(), "derived_affine_ras_mm": native.tolist(),
              "resampled": False}
    return original, native, record


def test_axis_binding_uses_authenticated_original_frame_after_native_roundoff():
    original, native, record = reconciled_fixture()
    assert not np.array_equal(original, native)
    candidates = exits(affine=native, index_affine=original, frame_record=record)
    result = screen_axis_accesses(candidates)
    assert result.complete and result.selected_exit is not None, result.error
    assert {(row["axis"], row["outward_sign"]) for row in result.exit_records} == set(product(range(3), (-1, 1)))


def test_same_native_grid_cannot_hide_different_original_index_frame():
    original, native, record = reconciled_fixture()
    candidates = exits(affine=native, index_affine=original, frame_record=record)
    candidates[-1] = exits(affine=native)[-1]
    result = screen_axis_accesses(candidates)
    assert result.status == "failed" and result.selected_exit is None, result.to_dict()


def test_unknown_provider_schema_fails_before_static_geometry(monkeypatch):
    import resectionlab.native_ingress as ingress
    candidates = exits()
    monkeypatch.setattr(ingress, "NOMINAL_CAVITY_PROPOSAL_VERSION", "unreviewed-provider-v2")
    result = screen_axis_accesses(candidates)
    assert result.status == "failed" and result.screen_count == 0
    assert result.selected_exit is None and "schema" in result.error
