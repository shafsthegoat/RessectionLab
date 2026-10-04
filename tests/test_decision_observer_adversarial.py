"""Independent decision-journal checks on tiny synthetic episodes only."""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import resectionlab.learning as learning
import resectionlab.native_axis_accounting as accounting_module
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_accounting import AccountingReceiptError, AxisTrainingAccounting
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator, CommittedTransitionInterrupted
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.worlds import WorldGeneratorConfig, WorldPartitionManifest, WorldRole
from resectionlab.worlds import generate_partitions


class CountingObservation:
    """Mutable source buffers expose extra reads or copies taken after step."""

    def __init__(self, stage, reads):
        self.reads = reads
        self.actions = np.diag([.1, .2, .3]).astype(np.float64)
        self.state = np.array([float(stage), .1], dtype=np.float64)
        self.mask = np.array([True, stage == 0, stage == 1])
        self.ids = ("STOP", "OPEN", "FINISH")

    @property
    def action_features(self):
        self.reads["action_features"] += 1
        return self.actions

    @property
    def state_features(self):
        self.reads["state_features"] += 1
        return self.state

    @property
    def action_mask(self):
        self.reads["action_mask"] += 1
        return self.mask

    @property
    def action_ids(self):
        self.reads["action_ids"] += 1
        return self.ids


class TraceSimulator:
    decision_model_hash = "synthetic-decision-observer-model-v1"
    case_hash = "synthetic-decision-observer-case-v1"
    world_generator_fingerprint = WorldGeneratorConfig().fingerprint

    def __init__(self, trace, reads):
        self.trace, self.reads = trace, reads
        self.stage = 0
        self.total = 0.

    def reset(self, seed=0):
        self.trace.append(("reset", int(seed)))
        self.stage, self.total, self.seed = 0, 0., int(seed)
        return self.observation()

    def observation(self):
        self.reads["observation"] += 1
        self.last = CountingObservation(self.stage, self.reads)
        return self.last

    def step(self, action):
        action = int(action)
        assert action == 0 or action == self.stage + 1
        self.trace.append(("step", self.seed, self.stage, action))
        # A simulator may reuse buffers after consuming them. The observer must
        # already have detached the exact policy inputs before this mutation.
        self.last.actions.fill(99.)
        self.last.state.fill(88.)
        self.last.mask.fill(False)
        reward = 0. if action == 0 else .25
        self.stage += int(action != 0)
        self.total += reward
        return SimpleNamespace(observation=self.observation(), reward=reward,
            terminated=action == 0 or self.stage == 2, info={"synthetic_only": True})

    def metrics(self):
        self.reads["metrics"] += 1
        return {"return_surrogate": self.total, "fixture": "observer_synthetic_only"}


def panels():
    generator = WorldGeneratorConfig()
    return (
        WorldPartitionManifest(WorldRole.OPTIMIZATION, TraceSimulator.case_hash, generator, (7, 13)),
        WorldPartitionManifest(WorldRole.SELECTION, TraceSimulator.case_hash, generator, (107, 113)),
    )


def settings(**kwargs):
    return replace(learning.TrainingConfig(seed=17, hidden_features=8,
        max_gradient_steps=2, max_environment_steps=20, max_episode_steps=3,
        max_wall_seconds=60., episodes_per_update=2, checkpoint_interval=1), **kwargs)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_observer_preserves_actual_updates_rng_forward_calls_and_simulator_reads(tmp_path, monkeypatch):
    actual_forward = learning.MaskedPatientPolicy.forward
    forwards = []

    def traced_forward(self, observation, **kwargs):
        result = actual_forward(self, observation, **kwargs)
        forwards.append(tuple(t.detach().clone() for t in result))
        return result

    monkeypatch.setattr(learning.MaskedPatientPolicy, "forward", traced_forward)
    outcomes = []
    for observed in (False, True):
        trace, reads, records = [], Counter(), []
        forwards.clear()
        directory = tmp_path / ("observed" if observed else "default")
        options = {"decision_observer": records.append} if observed else {}
        before_rng = torch.get_rng_state().clone()
        result = learning.train_patient_policy(lambda: TraceSimulator(trace, reads), *panels(),
            config=settings(), output_dir=directory, **options)
        assert torch.equal(before_rng, torch.get_rng_state()), "Logging or learning consumed global RNG"
        checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        outcomes.append((result, trace, reads, records, copy.deepcopy(forwards), checkpoint))
    left, right = outcomes
    assert left[0].gradient_steps == right[0].gradient_steps == 2
    assert left[0].latest_checkpoint_hash == right[0].latest_checkpoint_hash
    assert left[0].selected_checkpoint_hash == right[0].selected_checkpoint_hash
    assert left[1] == right[1], "The observer changed reset/transition execution"
    assert left[2] == right[2], "The observer reread observations or called simulator methods"
    assert len(left[4]) == len(right[4])
    for a, b in zip(left[4], right[4]):
        assert_nested_equal(a, b)
    for field in ("policy", "selected_policy", "optimizer", "random_state"):
        assert_nested_equal(left[5][field], right[5][field])
    assert right[3]
    assert len(right[3]) == sum(row[0] == "step" for row in right[1])
    for record in right[3]:
        json.dumps(record.to_dict(), allow_nan=False)


class TensorProbeSimulator:
    decision_model_hash = TraceSimulator.decision_model_hash

    def __init__(self):
        self.steps = 0
        self.observation_reads = 0
        self.source_actions = np.arange(45, dtype=np.float64).reshape(3, 15) / 7
        self.source_state = np.arange(6, dtype=np.float64) / 11

    def reset(self, seed=0):
        self.observation_reads += 1
        self.observation = SimpleNamespace(action_features=self.source_actions.copy(),
            state_features=self.source_state.copy(), action_mask=np.array([True, True, False]),
            action_ids=("STOP", "CUT_SYNTHETIC", "BLOCKED_SYNTHETIC"))
        return self.observation

    def step(self, action):
        assert action in (0, 1)
        self.steps += 1
        self.observation.action_features.fill(99.)
        self.observation.state_features.fill(88.)
        self.observation.action_mask.fill(False)
        return SimpleNamespace(observation=self.observation, reward=0., terminated=True, info={})

    def metrics(self):
        return {"synthetic_only": True}


@pytest.mark.parametrize("profile", ["RAW", "FEATURE_UNITS"])
def test_record_contains_same_forward_tensors_not_a_second_inference(profile):
    simulator = TensorProbeSimulator()
    with torch.random.fork_rng():
        torch.manual_seed(123)
        policy = learning.MaskedPatientPolicy(15, 6, 8, input_profile=profile)
    actor_inputs, value_inputs, outputs, records = [], [], [], []
    policy.actor.register_forward_pre_hook(lambda _m, args: actor_inputs.append(args[0].detach().clone()))
    policy.value.register_forward_pre_hook(lambda _m, args: value_inputs.append(args[0].detach().clone()))
    policy.register_forward_hook(lambda _m, _args, output: outputs.append(tuple(v.detach().clone() for v in output)))

    def observe(record):
        assert simulator.steps == 0, "The decision was recorded after execution"
        records.append(record)

    context = learning.DecisionContext("selection", 7, 4, 2)
    result = learning.rollout_policy(policy, simulator, seed=107, max_steps=2,
        decision_observer=observe, decision_context=context)
    assert result.environment_steps == simulator.steps == 1
    assert simulator.observation_reads == 1
    assert len(actor_inputs) == len(value_inputs) == len(outputs) == len(records) == 1
    record = records[0]
    assert record.context == context and record.seed == 107 and record.step == 0
    assert record.forward_evaluated and record.decision_rule == "deterministic_argmax"
    assert record.inputs.source_action_features.dtype == np.float64
    assert record.inputs.source_state_features.dtype == np.float64
    assert record.inputs.action_features.dtype == record.inputs.state_features.dtype == np.float32
    np.testing.assert_array_equal(record.inputs.source_action_features, simulator.source_actions)
    np.testing.assert_array_equal(record.inputs.source_state_features, simulator.source_state)
    np.testing.assert_array_equal(record.inputs.action_features, simulator.source_actions.astype(np.float32))
    np.testing.assert_array_equal(record.inputs.actor_action_features, actor_inputs[0].numpy()[:, :15])
    np.testing.assert_array_equal(record.inputs.state_features, value_inputs[0].numpy())
    np.testing.assert_array_equal(record.inputs.action_mask, [True, True, False])
    np.testing.assert_array_equal(record.logits, outputs[0][0].numpy())
    assert record.value == float(outputs[0][1])
    assert record.selected_index == int(outputs[0][0].argmax())
    assert record.selected_action_id == result.actions[0] == record.inputs.action_ids[record.selected_index]
    serialized = record.to_dict()
    assert serialized["logits"]["values"][2] == "-inf"
    json.dumps(serialized, allow_nan=False)
    serialized["inputs"]["action_features"]["values"][0][0] = 999.
    assert record.to_dict()["inputs"]["action_features"]["values"][0][0] != 999.


def captured_record():
    with torch.random.fork_rng():
        policy = learning.MaskedPatientPolicy(15, 6, 8)
    records = []
    learning.rollout_policy(policy, TensorProbeSimulator(), seed=107, max_steps=2,
        decision_observer=records.append)
    return records[0]


@pytest.mark.parametrize("field", ["source_action_features", "action_features", "actor_action_features", "action_mask", "logits"])
@pytest.mark.parametrize("mutation", ["shape", "dtype"])
def test_metadata_reinterpretation_is_rejected_before_journal_export(field, mutation):
    record = captured_record()
    array = record.logits if field == "logits" else getattr(record.inputs, field)
    before = record.to_dict()
    try:
        if mutation == "shape":
            array.shape = (array.size, 1) if array.ndim == 1 else (array.size,)
        else:
            array.dtype = np.uint8
    except (ValueError, AttributeError, TypeError):
        assert record.to_dict() == before
    else:
        with pytest.raises((ValueError, RuntimeError), match="mutat|chang|layout|integrity|metadata"):
            record.to_dict()


def test_observer_cannot_make_live_tensor_buffers_writable():
    record = captured_record()
    for array in (record.inputs.action_features, record.inputs.state_features,
                  record.inputs.actor_action_features, record.inputs.action_mask, record.logits):
        with pytest.raises(ValueError):
            array.flags.writeable = True


@pytest.mark.parametrize("baseline", [False, True])
def test_forced_rollout_stop_does_not_invent_uncomputed_logits_or_value(monkeypatch, baseline):
    policy = None if baseline else learning.MaskedPatientPolicy(3, 2, 8)
    def forbidden_forward(*_args, **_kwargs):
        raise AssertionError("Forced rollout STOP must not compute a new policy forward")
    monkeypatch.setattr(learning.MaskedPatientPolicy, "forward", forbidden_forward)
    records, trace, reads = [], [], Counter()
    result = learning.rollout_policy(policy, TraceSimulator(trace, reads), seed=107,
        max_steps=1, decision_observer=records.append,
        decision_context=learning.DecisionContext("selection", 0, 0, 0))
    assert result.actions == ("STOP",) and len(records) == 1
    record = records[0]
    assert record.logits is None and record.value is None and not record.forward_evaluated
    assert record.inputs.action_features is None and record.inputs.actor_action_features is None
    assert record.inputs.state_features is None and record.inputs.source_state_features.dtype == np.float64
    assert record.decision_rule == ("stop_baseline" if baseline else "forced_stop")
    assert record.forced_reason == (None if baseline else "episode_step_limit")
    assert record.selected_index == 0 and record.selected_action_id == "STOP"


def test_optimization_forced_stop_retains_existing_forward_and_budget_override(tmp_path):
    trace, reads, records = [], Counter(), []
    result = learning.train_patient_policy(lambda: TraceSimulator(trace, reads), *panels(),
        config=settings(max_environment_steps=1, max_gradient_steps=1, max_episode_steps=1,
                        episodes_per_update=1), output_dir=tmp_path, decision_observer=records.append)
    optimization = [r for r in records if r.context.role == "optimization"]
    assert result.gradient_steps == result.optimization_environment_steps == len(optimization) == 1
    decision = optimization[0]
    assert decision.forward_evaluated and decision.logits is not None and decision.value is not None
    assert decision.decision_rule == "forced_stop" and decision.selected_index == 0
    assert set(decision.forced_reason.split("+")) == {"episode_step_limit", "optimization_transition_budget"}
    assert decision.context.update == decision.context.episode == 0
    assert decision.seed == panels()[0].seeds[0]
    for record in records:
        if record.context.role == "selection":
            assert not record.forward_evaluated
            assert record.context.update in (0, 1)
            assert record.seed == panels()[1].seeds[record.context.episode]


def test_observer_failure_precedes_execution_and_cannot_publish_complete_selection(tmp_path):
    trace, reads, records = [], Counter(), []
    failure = OSError("synthetic decision journal disk failure")
    def observe(record):
        records.append(record)
        raise failure
    with pytest.raises(OSError) as caught:
        learning.train_patient_policy(lambda: TraceSimulator(trace, reads), *panels(),
            config=settings(), output_dir=tmp_path, decision_observer=observe)
    assert caught.value is failure
    assert len(records) == 1 and records[0].context.role == "selection"
    assert not any(row[0] == "step" for row in trace)
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "checkpoint.pt").exists()
    assert json.loads((tmp_path / "failures.jsonl").read_text().splitlines()[-1])["status"] == "failed"


def test_partial_selection_journal_does_not_supply_a_missing_panel_score(tmp_path):
    trace, reads, records, cancelled = [], Counter(), [], {"value": False}
    class InterruptAfterOne(TraceSimulator):
        def step(self, action):
            transition = super().step(action)
            cancelled["value"] = True
            return transition
    result = learning.train_patient_policy(lambda: InterruptAfterOne(trace, reads), *panels(),
        config=settings(), output_dir=tmp_path, decision_observer=records.append,
        cancelled=lambda: cancelled["value"])
    assert result.status == "cancelled" and result.gradient_steps == 0
    assert result.initial_selection_return is result.selected_selection_return is None
    assert result.selection_environment_steps == len(records) == 1
    assert records[0].context.role == "selection" and records[0].context.panel == 0
    assert json.loads((tmp_path / "result.json").read_text())["selection_history"] == []


def native_session(tmp_path):
    tissue = np.ones((7, 7, 8), dtype=bool)
    labels = np.zeros(tissue.shape, dtype=np.int16)
    labels[1:6, 1:6, 2:7] = 1
    config = NativeResectionConfig(tissue, labels, np.eye(4),
        AccessWindow((3, 3, -.5), (0, 0, 1), 4), NATIVE_GENERIC_TOOLS,
        "synthetic-decision-accounting-audit-v1", "explicit synthetic support",
        case_id="synthetic_decision_accounting_audit")
    cancelled = {"value": False}
    def factory():
        return AxisColumnNativeSimulator(config,
            proposal_config=AxisColumnProposalConfig(((0, 0),), 2), max_steps=2,
            cancelled=lambda: cancelled["value"])
    template = factory()
    worlds = generate_partitions(template.case_hash, template.config.world_generator, 20261004,
        optimization=1, selection=1, final_evaluation=1, stress=1)
    owner = AxisTrainingAccounting(factory, worlds.optimization, worlds.selection,
        receipt_path=tmp_path / "axis-decision-audit.json", expected_model_hash=template.decision_model_hash)
    owner.enable_decision_recording()
    simulator = owner.recorded_factory()
    # A real deterministic actor prefers every available cut over STOP. No
    # gradients, source labels, or optimization are used to choose this policy.
    with torch.random.fork_rng():
        policy = learning.MaskedPatientPolicy(15, 6, 4)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.actor[0].weight[0, 0] = -1.
        policy.actor[2].weight[0, 0] = 1.
    return owner, simulator, policy, worlds, cancelled


def native_rollout(owner, simulator, policy, worlds, observer=None):
    return learning.rollout_policy(policy, simulator, seed=worlds.selection.seeds[0],
        max_steps=2, expected_model_hash=owner.expected_model_hash,
        decision_observer=observer or owner.observe_decision,
        decision_context=learning.DecisionContext("selection", 0, 0, 0))


def test_axis_journal_links_cut_and_stop_without_treating_decisions_as_executions(tmp_path):
    owner, simulator, policy, worlds, _cancel = native_session(tmp_path)
    result = native_rollout(owner, simulator, policy, worlds)
    receipt = owner.snapshot()
    decisions = [r for r in receipt["events"] if r["kind"] == "decision"]
    transitions = [r for r in receipt["events"] if r["kind"] == "transition"]
    assert result.environment_steps == len(decisions) == len(transitions) == 2
    assert all(not row["executed_transition"] for row in decisions)
    assert all(row["status"] == "step_returned" for row in decisions)
    assert [row["decision_id"] for row in decisions] == [row["decision_id"] for row in transitions]
    assert transitions[0]["native_commit"] and not transitions[1]["native_commit"]
    assert decisions[0]["payload"]["forward_evaluated"]
    assert not decisions[1]["payload"]["forward_evaluated"]
    assert receipt["totals"]["selection"]["executed_transitions"] == 2
    assert receipt["totals"]["selection"]["returned_transitions"] == 2
    assert receipt["candidate_eligible"] is False and not receipt["independent_geometry_evaluation"]


def test_axis_postcommit_interrupt_links_used_decision_to_executed_unreturned_cut(tmp_path, monkeypatch):
    owner, simulator, policy, worlds, cancelled = native_session(tmp_path)
    commit = simulator._raw.engine.commit_preview
    def commit_then_interrupt(preview):
        result = commit(preview)
        cancelled["value"] = True
        return result
    monkeypatch.setattr(simulator._raw.engine, "commit_preview", commit_then_interrupt)
    with pytest.raises(CommittedTransitionInterrupted) as caught:
        native_rollout(owner, simulator, policy, worlds)
    receipt = owner.snapshot()
    decision = next(row for row in receipt["events"] if row["kind"] == "decision")
    transition = next(row for row in receipt["events"] if row["kind"] == "transition")
    assert decision["status"] == "executed_unreturned"
    assert decision["decision_id"] == transition["decision_id"]
    assert decision["payload"]["selected_action_id"] == transition["action_id"]
    assert transition["native_commit"] and transition["executed_transition"]
    assert not transition["returned_to_learner"] and not transition["next_observation_complete"]
    assert receipt["totals"]["selection"]["committed_but_unreturned"] == 1
    assert receipt["totals"]["selection"]["returned_transitions"] == 0
    assert caught.value.axis_accounting_receipt["receipt_hash"] == receipt["receipt_hash"]
    assert receipt["status"] == "failed" and receipt["candidate_eligible"] is False


@pytest.mark.parametrize("failure_stage", ["decision", "transition"])
def test_axis_export_failure_retains_last_durable_receipt_and_honest_attempt_status(tmp_path, monkeypatch, failure_stage):
    owner, simulator, policy, worlds, _cancel = native_session(tmp_path)
    atomic = accounting_module._atomic_receipt
    last_durable = {"bytes": owner.receipt_path.read_bytes()}
    def failing_atomic(path, receipt):
        if any(row["kind"] == failure_stage for row in receipt["events"]):
            raise OSError("synthetic decision receipt export failure")
        atomic(path, receipt)
        last_durable["bytes"] = path.read_bytes()
    monkeypatch.setattr(accounting_module, "_atomic_receipt", failing_atomic)
    with pytest.raises(AccountingReceiptError) as caught:
        native_rollout(owner, simulator, policy, worlds)
    receipt = caught.value.axis_accounting_receipt
    assert owner.receipt_path.read_bytes() == last_durable["bytes"]
    decision = next(row for row in receipt["events"] if row["kind"] == "decision")
    totals = receipt["totals"]["selection"]
    assert totals["returned_transitions"] == 0
    assert totals["executed_transitions"] == totals["native_commits"] == int(failure_stage == "transition")
    assert decision["status"] == ("executed_unreturned" if failure_stage == "transition" else "recorded_pre_step")
    assert receipt["status"] == "failed" and receipt["candidate_eligible"] is False
    assert receipt["learner_result"] is None
    with pytest.raises(RuntimeError, match="closed"):
        owner.recorded_factory()


def test_axis_consumer_rejects_forged_world_role_before_any_step(tmp_path):
    owner, simulator, policy, worlds, _cancel = native_session(tmp_path)
    def forged(record):
        owner.observe_decision(replace(record, context=replace(record.context, role="final_evaluation")))
    with pytest.raises(ValueError, match="bind|declared|role"):
        native_rollout(owner, simulator, policy, worlds, forged)
    receipt = owner.snapshot()
    assert receipt["totals"]["selection"]["executed_transitions"] == 0
    assert simulator._raw.engine.revision == 0
    assert receipt["candidate_eligible"] is False and receipt["status"] == "failed"


@pytest.mark.parametrize("field", ["source_action_features", "source_state_features", "action_features",
                                   "state_features", "actor_action_features", "action_mask"])
def test_axis_consumer_binds_inputs_to_the_served_observation_not_just_action_ids(tmp_path, field):
    owner, simulator, policy, worlds, _cancel = native_session(tmp_path)
    def forged(record):
        altered = getattr(record.inputs, field).copy()
        if field == "action_mask":
            altered[-1] = not altered[-1]
        else:
            altered.flat[0] += 1.
        # Replacing dataclasses creates fresh layout signatures. A self-valid
        # record still must match the observation served by this exact episode.
        inputs = replace(record.inputs, **{field: altered})
        owner.observe_decision(replace(record, inputs=inputs))
    with pytest.raises(ValueError, match="observ|source|input|mask|tensor|RAW|bind"):
        native_rollout(owner, simulator, policy, worlds, forged)
    receipt = owner.snapshot()
    assert receipt["totals"]["selection"]["executed_transitions"] == 0
    assert simulator._raw.engine.revision == 0
    assert receipt["candidate_eligible"] is False and receipt["status"] == "failed"
