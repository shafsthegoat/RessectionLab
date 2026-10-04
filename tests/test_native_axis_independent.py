"""Independent adversaries for the experimental dynamic native adapter."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef
from resectionlab.evaluation import independent_check_native_history
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator, CommittedTransitionInterrupted
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.procedural_learning import native_observation_schema
from resectionlab.simulation import InvalidActionError, RewardSpec, _readonly
from resectionlab.worlds import WorldGeneratorConfig


def fixture(*, barrier=False, reward=None, worlds=None, motor=None, cancelled=None, max_steps=2):
    tissue = np.ones((9, 9, 10), bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[:, :, 3:9] = 1
    hard = np.zeros(tissue.shape, bool)
    if barrier:
        hard[:, :, 5] = True
    case = CaseData("axis-independent", tissue.astype(np.float32), {"target": labels > 0}, np.eye(4),
                    (SourceRef("analytic", "synthetic:axis-independent", provenance="simulated"),), brain_mask=tissue)
    native = NativeResectionConfig(tissue, labels, case.affine, AccessWindow((4, 4, -.5), (0, 0, 1), 6),
        tuple(replace(tool) for tool in NATIVE_GENERIC_TOOLS), case.semantic_hash,
        "explicit synthetic support", case_id=case.case_id, hard_exclusion=hard)
    simulator = AxisColumnNativeSimulator(native,
        proposal_config=AxisColumnProposalConfig(offsets_source_voxels=((0, 0),), max_primary_rays=2),
        reward=RewardSpec() if reward is None else reward, world_generator=worlds,
        nominal_motor=motor, max_steps=max_steps, cancelled=cancelled)
    return case, simulator


def same_actor_observation(one, two):
    assert one.action_ids == two.action_ids
    np.testing.assert_array_equal(one.action_features, two.action_features)
    np.testing.assert_array_equal(one.state_features, two.state_features)
    np.testing.assert_array_equal(one.action_mask, two.action_mask)


def test_hidden_worlds_do_not_change_actor_or_inventory_after_identical_cut():
    motor = np.broadcast_to(np.linspace(0, 1, 9)[:, None, None], (9, 9, 10)).copy()
    worlds = WorldGeneratorConfig(family="rigid_uniform", translation_scale_mm=(.6, .6, .6))
    _, simulator = fixture(worlds=worlds, motor=motor)
    other = simulator.fresh()
    same_actor_observation(simulator.reset(11), other.reset(23))
    action = simulator.proposed_actions()[1].action_id
    first, second = simulator.step(action), other.step(action)
    same_actor_observation(first.observation, second.observation)
    assert first.reward != second.reward
    np.testing.assert_array_equal(simulator.cavity_mask, other.cavity_mask)
    assert simulator._hidden_world_hash != other._hidden_world_hash
    assert simulator.config.evidence_available == other.config.evidence_available == (True, False)


def test_feasible_primary_with_negative_reward_never_tries_fallback():
    _, simulator = fixture(reward=RewardSpec(target_per_mm3=0., normal_per_mm3=10.))
    actions = simulator.proposed_actions()
    assert len(actions) == 3 and all(simulator.nominal_action_value(action) < 0 for action in actions[1:])
    receipt = simulator.metrics()["inventory_receipts"][-1]
    assert len(receipt["attempts"]) == 2
    assert all(row["phase"] == "primary" and row["feasible"] for row in receipt["attempts"])
    assert len(receipt["batch"]["ledger"]) == 2


def test_barrier_rejection_is_recorded_before_any_certified_fallback():
    _, simulator = fixture(barrier=True)
    receipt = simulator.metrics()["inventory_receipts"][-1]
    attempts = receipt["attempts"]
    assert len(attempts) == 4
    for primary, fallback in zip(attempts[::2], attempts[1::2], strict=True):
        assert primary["proposal_id"] == fallback["proposal_id"]
        assert primary["phase"] == "primary" and not primary["feasible"]
        assert fallback["phase"] == "fallback"
        assert "FORBIDDEN" in primary["reason"]
    assert any(row["phase"] == "fallback" and row["feasible"] for row in attempts)
    assert len(receipt["certified_action_ids"]) == len(simulator.proposed_actions()) - 1


def test_fresh_resets_state_preserves_backend_and_missing_evidence():
    _, simulator = fixture()
    initial = simulator.observation()
    old = simulator.proposed_actions()[1].action_id
    simulator.step(old)
    with pytest.raises(InvalidActionError):
        simulator.step(old)
    fresh = simulator.fresh()
    assert type(fresh) is AxisColumnNativeSimulator
    assert fresh.engine is not simulator.engine and not fresh.removed_mask.any()
    assert fresh.config.evidence_available == (False, False)
    assert fresh.decision_model_hash == simulator.decision_model_hash
    same_actor_observation(initial, fresh.observation())
    changed = simulator.fresh(max_steps=1)
    assert changed.decision_model_hash != simulator.decision_model_hash
    same_actor_observation(initial, simulator.reset())
    with pytest.raises(ValueError):
        native_observation_schema(fresh)  # Existing procedural checkpoint gate remains closed.


@pytest.mark.parametrize("field", ["_partial_contact_mask", "remaining_mask"])
def test_reward_accounting_and_cavity_tampering_reject_even_cached_inventory(field):
    _, simulator = fixture()
    array = getattr(simulator, field)
    array[4, 4, 0] = ~array[4, 4, 0]
    with pytest.raises(RuntimeError):
        simulator.observation()


def test_derived_reward_array_replacement_cannot_change_frozen_objective():
    _, simulator = fixture()
    simulator._normal_fraction = _readonly(np.zeros_like(simulator._normal_fraction))
    with pytest.raises(RuntimeError):
        simulator.observation()


def test_cached_action_cost_cannot_differ_from_registered_preview():
    _, simulator = fixture()
    action = simulator.proposed_actions()[1]
    object.__setattr__(action, "insertion_distance_mm", action.insertion_distance_mm + 100.)
    with pytest.raises(RuntimeError):
        simulator.step(action.action_id)


def test_external_transient_cache_cannot_bypass_action_preview_binding():
    _, simulator = fixture()
    actions = simulator.proposed_actions()
    forged = replace(actions[1], removal_indices=_readonly(np.argwhere(simulator.config.target_labels > 0), np.int64))
    simulator._observation_inventory = (actions[0], forged)
    with pytest.raises(RuntimeError):
        simulator.step(forged.action_id)
    assert not simulator.removed_mask.any() and simulator.total_reward == 0.


def test_clone_diagnostics_and_committed_tissue_are_isolated():
    _, simulator = fixture()
    clone = simulator.clone()
    clone._inventory_receipts[0]["attempts"][0]["reason"] = "diagnostic edit"
    assert simulator.metrics()["inventory_receipts"][0]["attempts"][0]["reason"] != "diagnostic edit"
    clone.step(clone.proposed_actions()[1].action_id)
    assert clone.removed_mask.any() and not simulator.removed_mask.any()
    assert simulator.engine.history == []


def test_cancelled_preview_never_publishes_partial_inventory_and_retry_works(monkeypatch):
    state = {"cancelled": False, "armed": False}
    _, simulator = fixture(cancelled=lambda: state["cancelled"])
    initial = simulator.observation()
    original = simulator.engine.preview_stroke

    def preview(*args, **kwargs):
        result = original(*args, **kwargs)
        if state["armed"]:
            state.update(cancelled=True, armed=False)
        return result

    monkeypatch.setattr(simulator.engine, "preview_stroke", preview)
    for _ in range(2):
        state.update(cancelled=False, armed=True)
        with pytest.raises(InterruptedError):
            simulator.reset()
        assert simulator._proposals is None and simulator._previews == {}
        assert simulator.engine.history == [] and not simulator.removed_mask.any()
        assert simulator._inventory_receipts[-1]["status"] == "cancelled"
        assert len(simulator._inventory_receipts[-1]["attempts"]) == 1
    state.update(cancelled=False, armed=False)
    same_actor_observation(initial, simulator.reset())
    simulator.step(simulator.proposed_actions()[1].action_id)
    assert len(simulator.engine.history) == 1


@pytest.mark.parametrize("tamper", ["partial_contact", "total_reward", "hidden_world"])
def test_interrupted_reset_cannot_disable_episode_integrity(monkeypatch, tamper):
    state = {"cancelled": False}
    _, simulator = fixture(cancelled=lambda: state["cancelled"])
    original = simulator.engine.preview_stroke

    def preview(*args, **kwargs):
        result = original(*args, **kwargs)
        state["cancelled"] = True
        return result

    monkeypatch.setattr(simulator.engine, "preview_stroke", preview)
    with pytest.raises(InterruptedError):
        simulator.reset()
    state["cancelled"] = False
    monkeypatch.setattr(simulator.engine, "preview_stroke", original)
    if tamper == "partial_contact":
        simulator._partial_contact_mask[4, 4, 0] = True
    elif tamper == "total_reward":
        simulator.total_reward = 999.
    else:
        simulator._hidden_motor = _readonly(np.ones_like(simulator._hidden_motor))
    with pytest.raises(RuntimeError):
        simulator.observation()


def test_raising_preview_still_has_one_attempt_in_failure_denominator(monkeypatch):
    _, simulator = fixture()
    simulator._proposals = None
    prior_calls = simulator._preview_calls

    def preview(*args, **kwargs):
        raise ValueError("injected preview failure")

    monkeypatch.setattr(simulator.engine, "preview_stroke", preview)
    with pytest.raises(ValueError, match="injected preview"):
        simulator.proposed_actions()
    receipt = simulator._inventory_receipts[-1]
    assert simulator._preview_calls == prior_calls + 1
    assert receipt["status"] == "failed" and len(receipt["attempts"]) == 1
    assert receipt["attempts"][0]["phase"] == "primary"
    assert not simulator.removed_mask.any()


def test_incomplete_reset_blocks_all_rollout_exports_until_explicit_reset(monkeypatch):
    state = {"cancelled": False}
    _, simulator = fixture(cancelled=lambda: state["cancelled"])
    prior_action = simulator.proposed_actions()[1]
    original = simulator.engine.preview_stroke

    def preview(*args, **kwargs):
        result = original(*args, **kwargs)
        state["cancelled"] = True
        return result

    monkeypatch.setattr(simulator.engine, "preview_stroke", preview)
    with pytest.raises(InterruptedError):
        simulator.reset()
    state["cancelled"] = False
    monkeypatch.setattr(simulator.engine, "preview_stroke", original)
    operations = [simulator.proposed_actions, simulator.observation, simulator.metrics,
                  simulator.clone, simulator.fresh, lambda: simulator.step("STOP"),
                  lambda: simulator.nominal_action_value(prior_action)]
    for operation in operations:
        with pytest.raises(RuntimeError):
            operation()
    observation = simulator.reset()
    assert len(observation.action_ids) == 3 and simulator.total_reward == 0.
    simulator.step(observation.action_ids[1])
    assert len(simulator.engine.history) == 1


def test_cancellation_after_value_calculation_stops_before_commit(monkeypatch):
    state = {"cancelled": False}
    _, simulator = fixture(cancelled=lambda: state["cancelled"])
    original = simulator._action_value

    def value(*args, **kwargs):
        result = original(*args, **kwargs)
        state["cancelled"] = True
        return result

    monkeypatch.setattr(simulator, "_action_value", value)
    with pytest.raises(InterruptedError) as error:
        simulator.step(simulator.proposed_actions()[1].action_id)
    assert not isinstance(error.value, CommittedTransitionInterrupted)
    assert not simulator.removed_mask.any() and simulator.engine.history == []
    assert simulator.total_reward == 0


def test_cancellation_after_commit_preserves_explicit_executed_transition(monkeypatch):
    state = {"cancelled": False}
    _, simulator = fixture(cancelled=lambda: state["cancelled"])
    action_id = simulator.proposed_actions()[1].action_id
    original = simulator.engine.commit_preview

    def commit(preview):
        result = original(preview)
        state["cancelled"] = True
        return result

    monkeypatch.setattr(simulator.engine, "commit_preview", commit)
    with pytest.raises(CommittedTransitionInterrupted) as error:
        simulator.step(action_id)
    assert error.value.committed is True and error.value.info["action_id"] == action_id
    assert error.value.reward == simulator.total_reward
    assert simulator.removed_mask.any() and len(simulator.engine.history) == len(simulator._history) == 1
    assert simulator._proposals is None and simulator._previews == {}
    state["cancelled"] = False
    monkeypatch.setattr(simulator.engine, "commit_preview", original)
    assert simulator.metrics()["total_reward"] == error.value.reward
    assert simulator.observation().action_ids[0] == "STOP"


def test_executed_dynamic_history_retains_independent_native_certificate():
    case, simulator = fixture(max_steps=1)
    simulator.step(simulator.proposed_actions()[1].action_id)
    certificate = independent_check_native_history(case, simulator.native_config.tools,
        simulator.engine.history, tissue_mask=simulator.native_config.tissue_mask,
        hard_exclusion=simulator.native_config.hard_exclusion, access=simulator.native_config.access,
        geometry_frame="RAS+")
    assert certificate.feasible and certificate.unsupported_source_tissue_volume_mm3 == 0
    assert certificate.complete_tool_checked and certificate.frontier_checked
    assert certificate.claimed_source_tissue_volume_mm3 == simulator.removed_mask.sum()
