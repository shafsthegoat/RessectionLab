"""Independent analytic proposal/diagnostic contracts; no patient or training run."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.geometry import AccessWindow
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask, make_native_opening_task
from resectionlab.observed_search import observed_beam_search
from resectionlab.simulation import InvalidActionError


def tiny_task(*, cap=96, private=False, affine=None):
    original = make_native_opening_task().case
    changes = dict(proposal_mode="nominal_cavity_v1",
        proposal_config=NominalCavityProposalConfig(((0, 0), (1, 1), (30, -30)), cap))
    if private:
        changes["reference_target"] = np.ones(original.reference_target.shape, np.float32)
    if affine is not None:
        changes.update(affine_ras_mm=affine,
            access=AccessWindow(affine[:3, :3] @ original.access.center_mm,
                affine[:3, 2] / np.linalg.norm(affine[:3, 2]), original.access.radius_mm),
            native_grid_reconciliation="orthogonal_roundoff_1e-6mm")
    return NativeSpatialTask(replace(original, **changes), max_steps=2)


def opening(task):
    return next(row["action_id"] for row in task.candidate_inventory()["emitted"]
        if row["feasible"] and row["family"] == "exposed_opening"
        and row["tool_id"] == task.case.tools[0].tool_id and row["voxel"] == [4, 4, 1])


def test_hidden_reference_swap_preserves_actual_bounded_search_and_every_actor_input():
    first, other = tiny_task(), tiny_task(private=True)
    assert first.case.reference_hash != other.case.reference_hash
    for stage in range(2):
        a, b = first.observation(), other.observation()
        assert a.fingerprint == b.fingerprint and a.action_ids == b.action_ids
        assert first.candidate_inventory() == other.candidate_inventory()
        for name in ("image_channels", "coverage", "action_geometry", "state_features", "action_mask"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        left, left_counts = observed_beam_search(first, max_calls=6, beam_width=2, seconds=20.)
        right, right_counts = observed_beam_search(other, max_calls=6, beam_width=2, seconds=20.)
        assert left == right
        assert {k: v for k, v in left_counts.items() if k != "planning_seconds"} == {
            k: v for k, v in right_counts.items() if k != "planning_seconds"}
        if stage == 0:
            action = opening(first)
            assert first.step(action).reward != other.step(action).reward
    assert not first._config.target_labels.any() and not other._config.target_labels.any()


def test_branches_share_proposer_but_cavity_and_stale_actions_do_not_cross_authority():
    task = tiny_task()
    branches = (task.clone(), task.planning_clone(), task.fresh())
    assert all(branch.case._nominal_proposer is task.case._nominal_proposer for branch in branches)
    provider, initial = task.case._nominal_proposer, task.candidate_inventory()
    batch = provider.propose(task._engine)
    branch = branches[0]
    old_action = opening(branch)
    branch.step(old_action)
    assert task.candidate_inventory() == initial
    provider.validate_batch(batch, task._engine)
    with pytest.raises(ValueError, match="stale"):
        provider.validate_batch(batch, branch._engine)
    with pytest.raises(InvalidActionError, match="stale"):
        branch.step(old_action)
    assert not task._engine.removed_mask.any() and branch._engine.removed_mask.any()


def test_emitted_preview_order_and_complete_ledger_survive_cap_and_outside_columns(monkeypatch):
    calls = []
    preview = NativeResectionEngine.preview_stroke
    def counted(engine, tool_id, tip_mm, **kwargs):
        calls.append((tool_id, tuple(kwargs["entry_mm"]), tuple(tip_mm)))
        return preview(engine, tool_id, tip_mm, **kwargs)
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", counted)
    # The helper creates the legacy fixture first; isolate the dynamic engine.
    source = tiny_task(cap=2).case
    calls.clear()
    task = NativeSpatialTask(source, max_steps=2)
    inventory = task.candidate_inventory()
    assert calls == [(r["tool_id"], tuple(r["entry_mm"]), tuple(r["tip_mm"])) for r in inventory["emitted"]]
    assert inventory["emitted_count"] == 2 and inventory["omitted_count"] > 0
    assert len(inventory["ledger"]) == inventory["declared_slots"] == 18
    assert inventory["disposition_counts"]["COLUMN_OUT_OF_IMAGE"] == 6
    assert sum(inventory["disposition_counts"].values()) == 18
    assert not inventory["complete"] and inventory["ledger_complete"]
    assert task.observation().action_ids[0] == "STOP"
    assert len(task.observation().action_ids) <= inventory["candidate_cap"] + 1 < 128
    task.step("STOP")
    terminal = task.candidate_inventory()
    assert terminal["terminated_slots"] == 18 and terminal["emitted"] == []
    assert terminal["remaining_steps"] == 1 and terminal["terminal"]


def test_duplicate_families_are_distinct_ledger_slots_but_one_native_preview():
    base = tiny_task().case
    nominal = np.zeros_like(base.nominal_target)
    nominal[4, 4, 1] = 1.
    task = NativeSpatialTask(replace(base, nominal_target=nominal))
    inventory = task.candidate_inventory()
    for tool in task.case.tools:
        rows = [r for r in inventory["ledger"] if r["column_index"] == 0 and r["tool_id"] == tool.tool_id]
        assert [r["proposal_reason"] for r in rows] == ["PROPOSED_UNCERTIFIED", "DUPLICATE_GEOMETRY", "DUPLICATE_GEOMETRY"]
        assert len({r["action_id"] for r in rows}) == 1
    assert len({r["action_id"] for r in inventory["emitted"]}) == inventory["emitted_count"]


def test_maximum_dynamic_batch_is_bounded_below_native_certificate_capacity(monkeypatch):
    base = make_native_opening_task().case
    support = np.ones(base.observed_support.shape, bool)
    source = replace(base, observed_support=support, nominal_target=support,
        proposal_mode="nominal_cavity_v1",
        proposal_config=NominalCavityProposalConfig(tuple((a, b) for a in range(-4, 5) for b in range(-4, 5))))
    engine = NativeResectionEngine(source._native_config)
    def forbidden(*args, **kwargs):
        raise AssertionError("Endpoint proposal generation must not certify geometry")
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", forbidden)
    batch = source._nominal_proposer.propose(engine)
    assert len(batch.proposals) == 96
    assert len(batch.proposals) + 1 < 128  # Leading STOP consumes no native certificate.
    assert len(batch.ledger) == 81 * 2 * 3
    assert any(row.reason == "CANDIDATE_CAP" for row in batch.ledger)
    assert not batch.to_dict()["geometry_certified"]


def test_full_native_geometry_and_contact_remain_independent_of_small_actor_crop():
    base = tiny_task().case
    task = NativeSpatialTask(replace(base, crop_shape=(3, 3, 3)), max_steps=2)
    inventory = task.candidate_inventory()
    distal = [r for r in inventory["emitted"] if r["family"] == "distal_nominal"]
    assert distal and all(not r["endpoint_center_in_actor_crop"] for r in distal)
    assert all(not r["feasible"] for r in distal)
    task.step(opening(task))
    retained = task._engine.contact_mask & ~task._engine.removed_mask
    assert retained.any() and task._engine.remaining_mask[retained].all()
    assert task.metrics()["partial_contact_weight"] == 0.
    inventory = task.candidate_inventory()
    action = next(r["action_id"] for r in inventory["emitted"] if r["family"] == "distal_nominal"
        and r["tool_id"] == task.case.tools[1].tool_id and r["feasible"])
    result = task.step(action)
    removed = np.asarray(result.info["removed_indices_native"])
    origin = np.asarray(task.case._crop_origin)
    assert np.any((removed < origin) | (removed >= origin + 3))
    assert task.independent_geometry_check().feasible
    # The second cut fully removes the previously contacted cell; the earlier
    # retained-contact assertion belongs to the intermediate cavity only.
    assert not (task._engine.contact_mask & ~task._engine.removed_mask).any()


def test_explicit_original_index_roundoff_frame_drives_native_tips_and_report():
    affine = np.eye(4)
    affine[0, 2] = 8e-10
    task = tiny_task(affine=affine)
    np.testing.assert_array_equal(task.case.affine_ras_mm, affine)
    native = task.case._native_affine_ras_mm
    assert not np.array_equal(native, affine)
    inventory = task.candidate_inventory()
    for row in inventory["emitted"]:
        np.testing.assert_array_equal(row["tip_mm"], native[:3, :3] @ row["voxel"] + native[:3, 3])
        assert abs((np.asarray(row["entry_mm"]) - task.case.access.center_mm) @ task.case.access.normal_inward) < 1e-13

