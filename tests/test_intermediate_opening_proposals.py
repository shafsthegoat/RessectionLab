"""Generated physical/source contracts; no models or acquired anatomy."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.geometry import AccessWindow, GENERIC_TOOLS
from resectionlab.native_proposals import NominalCavityProposalConfig, INTERMEDIATE_OPENING_FAMILY
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask, make_native_opening_task


def slab(*, spacing=(.5, .5, .5), sign=1, cap=96, advance=1., target_plane=None, offsets=None):
    support = np.zeros((13, 13, 14), bool); support[2:11, 2:11, 2:12] = True
    target = np.zeros(support.shape, np.float32)
    target[2:11, 2:11, 9:11] = 1
    if target_plane is not None:
        target[:] = 0; target[2:11, 2:11, target_plane] = 1
    # Rotation, translation and anisotropy are explicit; no RAS-axis assumption.
    affine = np.array([[0., -spacing[1], 0., 10.], [spacing[0], 0., 0., -7.],
                       [0., 0., spacing[2], 20.], [0., 0., 0., 1.]])
    entry = np.array([6., 6., 1.5 if sign == 1 else 11.5])
    access = AccessWindow(affine[:3, :3] @ entry + affine[:3, 3], (0, 0, sign), 8.)
    kwargs = {} if offsets is None else {"offsets_source_voxels": offsets}
    config = NominalCavityProposalConfig(max_candidates=cap, intermediate_opening_mm=advance, **kwargs)
    source = NativeSpatialCase(support.astype(float), support, target, affine, access, GENERIC_TOOLS,
        nominal_target=target, target_derivation="public generated slab annotation",
        crop_shape=(13, 13, 14), proposal_mode="nominal_cavity_v1", proposal_config=config)
    return NativeSpatialTask(source, max_steps=6)


def test_default_exact_prechange_generated_identity_and_outputs():
    # Captured from canonical prechange source before running this candidate.
    config = NominalCavityProposalConfig()
    source = replace(make_native_opening_task().case, proposal_mode="nominal_cavity_v1", proposal_config=config)
    task = NativeSpatialTask(source, max_steps=2)
    assert config.fingerprint == "sha256:a6700e187fcfab7446ed75067c29033ed8d0e230ecdf93f90355916525ac6261"
    assert source.source_hash == "sha256:02690713a888b3f2927a5048d1f7b874a5bbfaf3fff68c32e7b446f9aa8186d2"
    assert semantic_digest(task.candidate_inventory()) == "sha256:fb7079f59a20c14c37e3a1c172f297c79eb63d06aa73dbf0c1720e012c88e5e2"
    assert task.observation().fingerprint == "sha256:ae47ded22af9ad5d420f05f5119887e69038667e904f652a4282e2809ee4993c"
    assert semantic_digest(source._nominal_proposer.propose(task._engine).to_dict()) == "sha256:abb133e4d4528585d80696a85f2ccb9639e1fcbe9a1f4da113c09df4f7b242e4"


def test_original_geometry_order_precedes_extras_and_96_cap_is_explicit():
    old, new = slab(advance=None), slab()
    a, b = old.candidate_inventory(), new.candidate_inventory()
    physical = lambda row: (row["tool_id"], row["family"], row["voxel"], row["entry_mm"], row["tip_mm"])
    assert [physical(row) for row in a["emitted"]] == [physical(row) for row in b["emitted"][:78]]
    assert b["declared_slots"] == 104 and len(b["ledger"]) == 104
    assert b["emitted_count"] == 96 and b["omitted_count"] == 8
    assert all(row["family"] != INTERMEDIATE_OPENING_FAMILY for row in b["ledger"][:78])
    assert all(row["family"] == INTERMEDIATE_OPENING_FAMILY for row in b["ledger"][78:])
    assert not b["complete"] and b["ledger_complete"]
    assert new.case.source_hash != old.case.source_hash
    np.testing.assert_array_equal(new.case.observed_support, old.case.observed_support)
    np.testing.assert_array_equal(new.case.nominal_target, old.case.nominal_target)


@pytest.mark.parametrize("spacing,count,actual", [(.4, 2, .8), (.5, 2, 1.), (1.6, 1, 1.6)])
@pytest.mark.parametrize("sign", [1, -1])
def test_requested_lookahead_snaps_to_actual_anisotropic_source_cell(spacing, count, actual, sign):
    task = slab(spacing=(.45, .65, spacing), sign=sign, offsets=((0, 0),))
    for row in task.candidate_inventory()["ledger"]:
        if row["family"] != INTERMEDIATE_OPENING_FAMILY: continue
        assert row["requested_opening_advance_mm"] == 1.
        assert row["advance_source_axis_voxels"] == count
        assert row["actual_opening_advance_mm"] == pytest.approx(actual)
        anchor, endpoint = np.array(row["opening_anchor_voxel"]), np.array(row["voxel"])
        np.testing.assert_array_equal(endpoint-anchor, (0, 0, sign*count))
        if row["proposal_reason"] == "PROPOSED_UNCERTIFIED":
            affine = task.case._native_affine_ras_mm
            np.testing.assert_allclose(row["tip_mm"], affine[:3, :3]@endpoint+affine[:3, 3], atol=1e-12)
            displacement = affine[:3, :3]@(endpoint-anchor)
            assert displacement @ task.case.access.normal_inward == pytest.approx(actual)


def test_new_endpoints_get_full_preview_and_can_make_deeper_certified_opening(monkeypatch):
    calls = []
    original = NativeResectionEngine.preview_stroke
    def counted(self, tool_id, tip_mm, **kwargs):
        calls.append((tool_id, tuple(tip_mm), tuple(kwargs["entry_mm"])))
        return original(self, tool_id, tip_mm, **kwargs)
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", counted)
    task = slab(offsets=((0, 0),))
    inventory = task.candidate_inventory()
    assert len(calls) == inventory["emitted_count"]
    assert set(calls) == {(r["tool_id"], tuple(r["tip_mm"]), tuple(r["entry_mm"])) for r in inventory["emitted"]}
    depth = lambda row: np.linalg.norm(np.subtract(row["tip_mm"], row["entry_mm"]))
    extra = [r for r in inventory["emitted"] if r["family"] == INTERMEDIATE_OPENING_FAMILY and r["feasible"]]
    old = [r for r in inventory["emitted"] if r["family"] != INTERMEDIATE_OPENING_FAMILY and r["feasible"]]
    assert extra and max(map(depth, extra)) > max(map(depth, old))
    outcome = task.step(extra[0]["action_id"])
    assert outcome.info["normal_removed_mm3"] > 0 and outcome.info["target_removed_mm3"] == 0
    assert outcome.reward < 0 and task.independent_geometry_check().feasible
    # Deep target rays remain subject to the same strict wider-shaft blockage.
    assert any("SHAFT_BLOCKED" in row["reason"] for row in inventory["emitted"] if row["family"] == "distal_nominal")


def test_dedup_and_true_endpoint_crop_metadata():
    task = slab(target_plane=4, offsets=((0, 0),))
    extra = [r for r in task.candidate_inventory()["ledger"] if r["family"] == INTERMEDIATE_OPENING_FAMILY]
    assert all(r["proposal_reason"] == "DUPLICATE_GEOMETRY" for r in extra)
    cropped = NativeSpatialTask(replace(slab(offsets=((0, 0),)).case, crop_shape=(3, 3, 3)), max_steps=6)
    origin, shape = np.array(cropped.case._crop_origin), np.array(cropped.case._crop_shape)
    inside = lambda voxel: bool(np.all(np.array(voxel) >= origin) and np.all(np.array(voxel) < origin+shape))
    assert any(inside(row["opening_anchor_voxel"]) != inside(row["voxel"])
        for row in cropped.candidate_inventory()["emitted"] if row["family"] == INTERMEDIATE_OPENING_FAMILY)
    for row in cropped.candidate_inventory()["emitted"]:
        assert row["endpoint_center_in_actor_crop"] == bool(np.all(np.array(row["voxel"]) >= origin)
            and np.all(np.array(row["voxel"]) < origin+shape))


def test_out_of_source_endpoint_is_reported_without_preview(monkeypatch):
    calls = []
    original = NativeResectionEngine.preview_stroke
    def counted(self, tool_id, tip_mm, **kwargs):
        calls.append(tuple(tip_mm))
        return original(self, tool_id, tip_mm, **kwargs)
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", counted)
    task = slab(spacing=(.5, .5, .08), offsets=((0, 0),))
    inventory = task.candidate_inventory()
    extras = [row for row in inventory["ledger"] if row["family"] == INTERMEDIATE_OPENING_FAMILY]
    assert extras and all(row["proposal_reason"] == "ENDPOINT_OUT_OF_IMAGE" for row in extras)
    assert all(row["action_id"] is None and row["voxel"][2] == 14 for row in extras)
    assert len(calls) == inventory["emitted_count"]


def test_private_reference_cannot_change_new_candidates_or_planning_rewards():
    first = slab(offsets=((0, 0),))
    second = NativeSpatialTask(replace(first.case, reference_target=np.ones(first.case.reference_target.shape)), max_steps=6)
    assert first.candidate_inventory() == second.candidate_inventory()
    assert first.observation().fingerprint == second.observation().fingerprint
    action = next(r["action_id"] for r in first.candidate_inventory()["emitted"]
                  if r["family"] == INTERMEDIATE_OPENING_FAMILY and r["feasible"])
    a, b = first.planning_clone(), second.planning_clone()
    assert a.advance_planning(action).info == b.advance_planning(action).info


@pytest.mark.parametrize("value", [True, 0., 2., float("nan")])
def test_option_is_one_declared_requested_distance(value):
    with pytest.raises(ValueError, match="explicit fixed 1 mm"):
        NominalCavityProposalConfig(intermediate_opening_mm=value)
