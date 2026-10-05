"""Tiny analytic checks for the separate dynamic proposal action model."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef
from resectionlab.evaluation import independent_check_native_history
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_simulation import (
    AXIS_ADAPTER_VERSION, AxisColumnNativeSimulator, CommittedTransitionInterrupted,
)
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.native_simulation import NativeSequentialSimulator, native_beam_search
from resectionlab.simulation import InvalidActionError, RewardSpec
from resectionlab.worlds import WorldGeneratorConfig


def fixture(*, hard_distal=False):
    tissue = np.ones((7, 7, 8), bool)
    target = np.zeros(tissue.shape, bool)
    target[1:6, 1:6, 2:7] = True
    case = CaseData("axis-adapter-analytic", tissue.astype(float), {"target": target}, np.eye(4),
                    (SourceRef("analytic", "simulated:axis-adapter", provenance="simulated"),),
                    brain_mask=tissue)
    hard = np.zeros_like(tissue)
    if hard_distal:
        hard[3, 3, 5] = True
    config = NativeResectionConfig(tissue, target.astype(np.int16), np.eye(4),
        AccessWindow((3, 3, -.5), (0, 0, 1), 4), NATIVE_GENERIC_TOOLS,
        case.semantic_hash, "explicit synthetic support", case_id=case.case_id,
        hard_exclusion=hard)
    return case, config


def make_sim(**kwargs):
    case, config = fixture()
    return case, AxisColumnNativeSimulator(config,
        proposal_config=AxisColumnProposalConfig(((0, 0), (1, 0)), 4), max_steps=2, **kwargs)


def test_dynamic_inventory_is_bounded_certified_and_shared_by_search_and_raw_policy():
    _, sim = make_sim()
    original = sim.native_config.target_labels.copy()
    search_state, actor_state = sim.clone(), sim.fresh()
    search_actions = search_state.proposed_actions()
    actor = actor_state.observation()
    assert actor.action_ids == tuple(action.action_id for action in search_actions)
    np.testing.assert_array_equal(actor.action_features, search_state.observation().action_features)
    assert actor.action_features.shape[1] == 15 and actor.state_features.shape == (6,)
    assert len(actor.action_ids) <= sim.config.max_actions == 5
    assert actor.action_ids[0] == "STOP" and actor.action_mask.all()
    assert all(len(action_id) == 31 for action_id in actor.action_ids[1:])
    receipt = sim.metrics()["inventory_receipts"][0]
    assert receipt["status"] == "complete"
    assert receipt["batch"]["slot_count"] == 4
    assert len(receipt["attempts"]) <= 8
    assert not receipt["batch"]["geometry_certified"]
    assert not sim.removed_mask.any()
    np.testing.assert_array_equal(sim.native_config.target_labels, original)
    assert sim.metrics()["simulation_version"] == AXIS_ADAPTER_VERSION
    assert sim.metrics()["motor_surrogate"] is None


def test_primary_rejection_precedes_fallback_and_both_checks_are_accounted():
    _, config = fixture(hard_distal=True)
    sim = AxisColumnNativeSimulator(config,
        proposal_config=AxisColumnProposalConfig(((0, 0),), 2), max_steps=1)
    receipt = sim.metrics()["inventory_receipts"][0]
    assert [attempt["phase"] for attempt in receipt["attempts"]] == ["primary", "fallback", "primary", "fallback"]
    assert all(not row["feasible"] for row in receipt["attempts"] if row["phase"] == "primary")
    assert len(sim.proposed_actions()) > 1
    assert sim.metrics()["proposal_accounting"]["preview_calls"] == 4
    assert all(row["phase"] == "fallback" for row in sim._action_provenance.values())


def test_feasible_primary_does_not_try_fallback_even_when_reward_is_negative():
    _, sim = make_sim(reward=RewardSpec(action_cost=1e6))
    assert all(row["phase"] == "primary" for row in sim.metrics()["inventory_receipts"][0]["attempts"])
    assert all(sim.nominal_action_value(action) < 0 for action in sim.proposed_actions()[1:])


def test_search_replays_exactly_and_has_an_independent_native_certificate():
    case, sim = make_sim()
    result = native_beam_search(sim, beam_width=2, max_expansions=8, max_wall_seconds=10)
    assert result.actions[0] != "STOP"
    replay = sim.fresh()
    replay.replay(result.actions)
    metrics = replay.metrics()
    assert metrics["total_reward"] == pytest.approx(result.nominal_score)
    assert metrics["simulated_removed_target_volume_mm3"] > 0
    audit = independent_check_native_history(case, sim.native_config.tools, metrics["history"],
        tissue_mask=sim.native_config.tissue_mask, access=sim.native_config.access,
        hard_exclusion=sim.native_config.hard_exclusion)
    assert audit.feasible and audit.complete_tool_checked and audit.frontier_checked
    assert not sim.removed_mask.any()
    replay.reset()
    assert replay.observation().action_ids == sim.observation().action_ids


def test_stale_actions_clone_isolation_and_frozen_rules():
    _, sim = make_sim()
    stale = sim.proposed_actions()[1].action_id
    branch = sim.clone()
    branch.step(stale)
    assert not sim.removed_mask.any() and branch.removed_mask.any()
    with pytest.raises(InvalidActionError):
        branch.step(stale)
    assert sim.fresh().decision_model_hash == sim.decision_model_hash
    assert sim.fresh(max_steps=3).decision_model_hash != sim.decision_model_hash
    sim.proposal_config = replace(sim.proposal_config, max_primary_rays=3)
    with pytest.raises(RuntimeError, match="decision model"):
        sim.step("STOP")


def test_nominal_inventory_ignores_hidden_world_seed():
    _, config = fixture()
    motor = np.zeros(config.tissue_mask.shape)
    motor[2, :, :] = 1
    sim = AxisColumnNativeSimulator(config, nominal_motor=motor,
        proposal_config=AxisColumnProposalConfig(((0, 0),), 2),
        world_generator=WorldGeneratorConfig(translation_scale_mm=(.3, .3, .3)))
    first = sim.reset(10)
    first_world = sim.metrics()["episode_world_hash"]
    second = sim.reset(11)
    assert first_world != sim.metrics()["episode_world_hash"]
    assert first.action_ids == second.action_ids
    np.testing.assert_array_equal(first.action_features, second.action_features)
    assert sim.fresh().config.evidence_available == (True, False)


def test_existing_exact_class_profiles_remain_closed():
    from resectionlab.learning import _validate_simulator_profile
    from resectionlab.procedural_learning import native_observation_schema
    _, sim = make_sim()
    _validate_simulator_profile(sim, "RAW")
    with pytest.raises(ValueError, match="actual native"):
        _validate_simulator_profile(sim, "FEATURE_UNITS")
    with pytest.raises(ValueError, match="exact native"):
        native_observation_schema(sim)
    assert type(sim) is not NativeSequentialSimulator


def test_cancelled_inventory_is_not_published_and_retry_is_valid(monkeypatch):
    flag = {"cancelled": False}
    _, sim = make_sim(cancelled=lambda: flag["cancelled"])
    original = sim.engine.preview_stroke
    sim._proposals = None
    def cancel_after_preview(*args, **kwargs):
        result = original(*args, **kwargs)
        flag["cancelled"] = True
        return result
    monkeypatch.setattr(sim.engine, "preview_stroke", cancel_after_preview)
    with pytest.raises(InterruptedError):
        sim.proposed_actions()
    assert sim._proposals is None and not sim.removed_mask.any()
    assert sim._inventory_receipts[-1]["status"] == "cancelled"
    assert len(sim._inventory_receipts[-1]["attempts"]) == 1
    flag["cancelled"] = False
    monkeypatch.setattr(sim.engine, "preview_stroke", original)
    assert len(sim.proposed_actions()) > 1
    sim.step(1)
    assert sim.removed_mask.any()


def test_cancel_after_commit_reports_executed_transition_without_rollback(monkeypatch):
    flag = {"cancelled": False}
    _, sim = make_sim(cancelled=lambda: flag["cancelled"])
    original = sim.engine.commit_preview
    def cancel_after_commit(preview):
        result = original(preview)
        flag["cancelled"] = True
        return result
    monkeypatch.setattr(sim.engine, "commit_preview", cancel_after_commit)
    with pytest.raises(CommittedTransitionInterrupted) as caught:
        sim.step(1)
    assert caught.value.committed and sim.removed_mask.any()
    assert len(sim.engine.history) == len(sim._history) == 1
    assert sim.total_reward == caught.value.reward
    assert sim._proposals is None


def test_native_capacity_is_a_constructor_bound():
    _, config = fixture()
    with pytest.raises(ValueError, match="128"):
        AxisColumnNativeSimulator(config, proposal_config=AxisColumnProposalConfig(max_primary_rays=129))
