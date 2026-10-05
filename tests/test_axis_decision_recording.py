"""Decision-to-native-transition joins over tiny synthetic RAW training only."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

import resectionlab.native_axis_accounting as accounting_module
from resectionlab.learning import MaskedPatientPolicy
from resectionlab.native_axis_accounting import AccountingReceiptError, train_axis_policy
from resectionlab.native_axis_simulation import CommittedTransitionInterrupted
from test_native_axis_accounting import fixture, manager, settings


def test_recorded_native_training_matches_disabled_weights_and_joins_every_step(tmp_path):
    template = fixture()
    config = replace(settings(), episodes_per_update=2, max_environment_steps=16)
    plain, _ = manager(tmp_path / "plain", template, factory=template.fresh)
    recorded, _ = manager(tmp_path / "recorded", template, factory=template.fresh)
    before = torch.random.get_rng_state().clone()
    baseline = train_axis_policy(plain, config=config, output_dir=tmp_path / "plain/learner")
    observed = train_axis_policy(recorded, config=config, output_dir=tmp_path / "recorded/learner",
                                 record_decisions=True)
    assert torch.equal(before, torch.random.get_rng_state())
    assert baseline.gradient_steps == observed.gradient_steps == 1
    assert baseline.latest_checkpoint_hash == observed.latest_checkpoint_hash
    assert baseline.selected_checkpoint_hash == observed.selected_checkpoint_hash
    for name in ("optimization_environment_steps", "selection_environment_steps", "initial_selection_return", "selected_selection_return"):
        assert getattr(baseline, name) == getattr(observed, name)
    plain_checkpoint = torch.load(tmp_path / "plain/learner/checkpoint.pt", weights_only=True)
    recorded_checkpoint = torch.load(tmp_path / "recorded/learner/checkpoint.pt", weights_only=True)
    assert torch.equal(plain_checkpoint["random_state"], recorded_checkpoint["random_state"])
    for key, value in plain_checkpoint["policy"].items():
        assert torch.equal(value, recorded_checkpoint["policy"][key])
    receipt = json.loads(recorded.receipt_path.read_text())
    decisions = {event["decision_id"]: event for event in receipt["events"] if event["kind"] == "decision"}
    transitions = [event for event in receipt["events"] if event.get("executed_transition")]
    assert len(decisions) == len(transitions) > 0
    assert len({event["decision_id"] for event in transitions}) == len(transitions)
    assert {event["payload"]["panel"] for event in decisions.values() if event["role"] == "selection"} == {0, 1}
    for event in transitions:
        decision = decisions[event["decision_id"]]
        payload = decision["payload"]
        assert (decision["episode"], decision["role"], decision["seed"]) == (event["episode"], event["role"], event["seed"])
        assert payload["selected_action_id"] == event["action_id"]
        assert payload["inputs"]["action_ids"][payload["selected_index"]] == event["action_id"]
        assert decision["status"] == "step_returned"
        if payload["forward_evaluated"]:
            assert payload["inputs"]["actor_action_features"] == payload["inputs"]["action_features"]
    assert not any(event["kind"] == "decision" for event in plain.snapshot()["events"])
    shape_probe = receipt["episodes"][0]
    assert shape_probe["status"] == "unfinished_at_learner_return"
    assert not any(event["episode"] == shape_probe["episode"] for event in decisions.values())


def force_initial_cut(monkeypatch):
    original = MaskedPatientPolicy.forward
    def first_cut(policy, observation, **kwargs):
        logits, value = original(policy, observation, **kwargs)
        forced = logits * 0
        forced[1 if len(forced) > 1 else 0] = 10
        return forced, value
    monkeypatch.setattr(MaskedPatientPolicy, "forward", first_cut)


def test_decision_export_failure_after_prior_commit_keeps_prior_truth(tmp_path, monkeypatch):
    force_initial_cut(monkeypatch)
    template = fixture()
    instances = []
    def factory():
        raw = template.fresh()
        instances.append(raw)
        return raw
    accounting, _ = manager(tmp_path, template, factory=factory)
    original = accounting_module._atomic_receipt
    def export(path, receipt):
        if (any(event.get("native_commit") for event in receipt["events"])
                and receipt["events"][-1]["kind"] == "decision"):
            raise OSError("synthetic next-decision export failure")
        original(path, receipt)
    monkeypatch.setattr(accounting_module, "_atomic_receipt", export)
    with pytest.raises(AccountingReceiptError) as caught:
        train_axis_policy(accounting, config=settings(), output_dir=tmp_path / "learner", record_decisions=True)
    receipt = caught.value.axis_accounting_receipt
    totals = receipt["totals"]["selection"]
    assert totals["native_commits"] == totals["returned_transitions"] == 1
    assert receipt["status"] == "failed" and not receipt["candidate_eligible"]
    decisions = [event for event in receipt["events"] if event["kind"] == "decision"]
    assert len(decisions) == 2 and decisions[0]["status"] == "step_returned"
    assert decisions[1]["status"] == "recorded_pre_step"
    assert instances[-1].engine.revision == 1
    assert not (tmp_path / "learner/result.json").exists()


def test_recorded_decision_joins_actual_committed_interruption(tmp_path, monkeypatch):
    force_initial_cut(monkeypatch)
    template = fixture()
    instances = []
    def factory():
        raw = template.fresh()
        instances.append(raw)
        commit = raw.engine.commit_preview
        def commit_then_cancel(preview):
            result = commit(preview)
            raw._cancelled = lambda: True
            return result
        raw.engine.commit_preview = commit_then_cancel
        return raw
    accounting, _ = manager(tmp_path, template, factory=factory)
    with pytest.raises(CommittedTransitionInterrupted) as caught:
        train_axis_policy(accounting, config=settings(), output_dir=tmp_path / "learner", record_decisions=True)
    receipt = json.loads(accounting.receipt_path.read_text())
    decision = next(event for event in receipt["events"] if event["kind"] == "decision")
    transition = next(event for event in receipt["events"] if event.get("native_commit"))
    assert decision["decision_id"] == transition["decision_id"]
    assert decision["payload"]["selected_action_id"] == caught.value.info["action_id"]
    assert transition["returned_to_learner"] is False
    assert decision["status"] == "executed_unreturned"
    assert instances[-1].engine.revision == 1
    assert receipt["totals"]["selection"]["committed_but_unreturned"] == 1
    assert not (tmp_path / "learner/result.json").exists()
