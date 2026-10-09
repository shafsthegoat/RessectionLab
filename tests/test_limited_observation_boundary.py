"""Generated fixture boundary only: no training, patient files, or hidden anatomy."""
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest
import torch

import resectionlab.limited_observation_boundary as boundary
from resectionlab.core import semantic_digest, thaw_json
from resectionlab.limited_observation_boundary import (
    VERSION, SCOPE, FrozenLimitedPlan, HeldOutTargetBinding, HeldOutTargetReference,
    LimitedPlanningSpec, build_limited_planning_task, evaluate_frozen_limited_plan,
    freeze_limited_plan,
)
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
from resectionlab.native_spatial_task import make_native_opening_task
from resectionlab.observed_search import observed_beam_search
from resectionlab.spatial_policy import SpatialPolicy


SEARCH_CONFIG = dict(max_calls=128, beam_width=16, seconds=5., transition_mode="lazy_planning")


def permitted_spec():
    # Fixture setup explicitly selects permitted fields. The rich task is then
    # discarded; no adapter accepting a rich task exists in the boundary API.
    fixture = make_native_opening_task()
    case = fixture.case
    return LimitedPlanningSpec(VERSION, SCOPE, case.structural_intensity, case.affine_ras_mm,
        case.observed_support, case.nominal_target,
        {"source_kind": case.support_source_kind, "derivation": case.support_derivation},
        {"source_kind": case.target_source_kind, "derivation": case.target_derivation},
        case.access, case.tools, case.crop_shape, case.intensity_normalization,
        case.proposal_mode, fixture.max_steps, fixture.reward_spec)


def method_record(cost=None, method="SEARCH"):
    cost = cost or {}
    return {"method": method, "parameter_hash": semantic_digest({"search": "unguided"}),
        "configuration_hash": semantic_digest(SEARCH_CONFIG),
        "model_transition_calls": cost.get("model_transition_calls", 0),
        "actor_forward_calls": cost.get("actor_forward_calls", 0),
        "completed": True, "time_cap_reached": cost.get("time_cap_reached", False)}


def binding(spec, kind="original_generated_target"):
    return HeldOutTargetBinding(VERSION, SCOPE, spec.fingerprint, kind)


def load_generated_reference(descriptor):
    # This helper is evaluator-owned and is only invoked after sealing.
    target = np.zeros((9, 9, 7), np.float32)
    if descriptor.reference_kind == "original_generated_target":
        target[4, 4, 4:6] = 1
    return HeldOutTargetReference(descriptor, target)


@pytest.fixture(scope="module")
def sealed_search():
    spec = permitted_spec()
    task = build_limited_planning_task(spec)
    sequence, cost = observed_beam_search(task, **SEARCH_CONFIG)
    assert len(sequence) == 2 and "STOP" not in sequence
    for action in sequence:
        task.advance_planning(action)
    return spec, freeze_limited_plan(task, spec=spec, method_record=method_record(cost)), cost


def test_same_seal_independently_scores_original_and_zero_private_target(sealed_search, monkeypatch):
    spec, plan, cost = sealed_search
    before = plan.to_record()
    independent_tasks = []
    actual_evaluate = boundary.evaluate_native_spatial_episode

    def audit(task):
        assert task.metrics()["planning_estimator_only"] is False
        assert tuple(row["action_id"] for row in task.metrics()["history"]) == plan.action_ids
        independent_tasks.append(task)
        return actual_evaluate(task)

    monkeypatch.setattr(boundary, "evaluate_native_spatial_episode", audit)
    reports = []
    for kind in ("original_generated_target", "zero_target_counterfactual"):
        descriptor = binding(spec, kind)

        def loader():
            assert plan.to_record() == before
            return load_generated_reference(descriptor)

        report = evaluate_frozen_limited_plan(plan, spec=spec, reference_binding=descriptor,
                                             load_reference=loader)
        reports.append(report)
        assert report["status"] == "evaluated"
        assert report["independent_episode"]["geometry"]["feasible"]
        assert report["independent_episode"]["geometry"]["complete_tool_checked"]
        assert report["plan_seal_hash"] == plan.seal_hash
        assert report["method_record"]["model_transition_calls"] == cost["model_transition_calls"]
        assert report["clinical_deficit_probability"] is report["patient_generalization"] is None
        assert not report["hidden_support_validated"] and not report["hidden_hazards_validated"]
        assert not report["file_access_isolation"]
        serialized = thaw_json(report)
        seal = serialized.pop("report_hash")
        assert seal == semantic_digest(serialized)
        with pytest.raises(TypeError):
            report["status"] = "other"
    first, second = (report["independent_episode"] for report in reports)
    assert [first["outcomes"]["target_removed_mm3"], second["outcomes"]["target_removed_mm3"]] == [2., 0.]
    assert first["outcomes"]["total_reward"] != second["outcomes"]["total_reward"]
    assert first["geometry"] == second["geometry"]
    assert first["source_hash"] == second["source_hash"]
    assert first["decision_model_hash"] == second["decision_model_hash"]
    assert first["reference_hash"] != second["reference_hash"]
    assert reports[0]["reference_hash"] != reports[1]["reference_hash"]
    assert independent_tasks[0] is not independent_tasks[1]
    assert plan.to_record() == before


def test_private_evaluation_never_requests_another_actor_or_search_decision(sealed_search, monkeypatch):
    import resectionlab.observed_search as search
    spec, plan, _ = sealed_search
    descriptor = binding(spec)
    def forbidden(*args, **kwargs):
        pytest.fail("A private evaluator must not request planner decisions")
    monkeypatch.setattr(search, "observed_beam_search", forbidden)
    monkeypatch.setattr(SpatialPolicy, "forward", forbidden)
    report = evaluate_frozen_limited_plan(plan, spec=spec, reference_binding=descriptor,
        load_reference=lambda: load_generated_reference(descriptor))
    assert report["status"] == "evaluated"


def test_planning_inputs_masks_logits_nominal_cavity_and_cost_are_reference_invariant(sealed_search):
    spec, plan, cost = sealed_search
    # Evaluator objects are now available, but neither can be passed to planning.
    refs = [load_generated_reference(binding(spec, kind)) for kind in
            ("original_generated_target", "zero_target_counterfactual")]
    assert refs[0].fingerprint != refs[1].fingerprint
    tasks = [build_limited_planning_task(spec) for _ in refs]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        policy = SpatialPolicy().eval()
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for step in range(3):
            left, right = (task.observation() for task in tasks)
            assert left.fingerprint == right.fingerprint
            assert left.action_ids == right.action_ids
            assert tasks[0].candidate_inventory() == tasks[1].candidate_inventory()
            for name in ("image_channels", "coverage", "action_geometry", "action_mask", "state_features"):
                np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
            with torch.no_grad():
                logits_a, value_a = policy(left)
                logits_b, value_b = policy(right)
            torch.testing.assert_close(logits_a, logits_b, atol=0, rtol=0)
            torch.testing.assert_close(value_a, value_b, atol=0, rtol=0)
            assert tasks[0].metrics() == tasks[1].metrics()
            if step < 2:
                for task in tasks:
                    task.advance_planning(plan.action_ids[step])
        sequence, repeated_cost = observed_beam_search(build_limited_planning_task(spec), **SEARCH_CONFIG)
    finally:
        torch.set_num_threads(old_threads)
    assert sequence == plan.action_ids
    for key in ("model_transition_calls", "actor_forward_calls", "expanded_nodes", "completed_layers",
                "evaluated_transition_prefixes", "lazy_planning_transition_calls"):
        assert repeated_cost[key] == cost[key]
    for ref in refs:
        with pytest.raises(ValueError, match="LimitedPlanningSpec"):
            build_limited_planning_task(ref)


def test_spec_roundtrip_is_detached_and_only_exact_fixture_is_admitted():
    spec = permitted_spec()
    record = json.loads(json.dumps(spec.to_record()))
    copy = LimitedPlanningSpec.from_record(record)
    record["structural_intensity"][4][4][4] = 0
    assert copy.fingerprint == spec.fingerprint
    assert copy.structural_intensity[4, 4, 4] == np.float32(.8)
    for value in (copy.structural_intensity, copy.nominal_support, copy.nominal_target, copy.affine_ras_mm):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(TypeError):
        copy.support_provenance["derivation"] = "changed"
    with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
        copy.private_reference = object()
    for rich in (make_native_opening_task(), make_native_opening_task().case, record):
        with pytest.raises(ValueError, match="LimitedPlanningSpec"):
            build_limited_planning_task(rich)


@pytest.mark.parametrize("key", ["reference_target", "hidden_support", "hard_exclusion", "patient_id", "metadata"])
def test_spec_extra_fields_rejected(key):
    record = permitted_spec().to_record()
    record[key] = "forbidden"
    with pytest.raises(ValueError, match="exact fields"):
        LimitedPlanningSpec.from_record(record)


@pytest.mark.parametrize("mutation", ["missing_target", "none_target", "patient_scope", "affine", "tool",
    "image", "support", "nominal", "provenance", "crop", "normalization", "proposal", "reward", "horizon"])
def test_spec_cannot_relabel_new_geometry_anatomy_or_configuration_as_fixture(mutation):
    record = permitted_spec().to_record()
    if mutation == "missing_target": del record["nominal_target"]
    elif mutation == "none_target": record["nominal_target"] = None
    elif mutation == "patient_scope": record["scope"] = "patient-linked-limited-v1"
    elif mutation == "affine": record["affine_ras_mm"][0][3] = 1
    elif mutation == "tool": record["tools"][0]["tip_radius_mm"] += .1
    elif mutation == "image": record["structural_intensity"][4][4][4] = .7
    elif mutation == "support": record["nominal_support"][0][0][0] = True
    elif mutation == "nominal": record["nominal_target"][4][4][4] = 0
    elif mutation == "provenance": record["support_provenance"]["derivation"] = "patient mask"
    elif mutation == "crop": record["crop_shape"] = [9, 9, 7]
    elif mutation == "normalization": record["intensity_normalization"] = "support_percentile_1_99"
    elif mutation == "proposal": record["proposal_mode"] = "nominal_cavity_v1"
    elif mutation == "reward": record["reward"]["normal_per_mm3"] = .1
    elif mutation == "horizon": record["max_steps"] = 3
    with pytest.raises(ValueError):
        LimitedPlanningSpec.from_record(record)


def test_incomplete_or_non_nominal_history_cannot_seal():
    spec = permitted_spec()
    task = build_limited_planning_task(spec)
    with pytest.raises(ValueError, match="completed"):
        freeze_limited_plan(task, spec=spec, method_record=method_record())
    action = next(row["action_id"] for row in task.candidate_inventory()["ledger"] if row["feasible"])
    task.advance_planning(action)
    with pytest.raises(ValueError, match="completed"):
        freeze_limited_plan(task, spec=spec, method_record=method_record())
    task.advance_planning("STOP")
    plan = freeze_limited_plan(task, spec=spec, method_record=method_record())
    assert plan.terminal_reason == "STOP" and plan.action_ids == (action, "STOP")
    actual = make_native_opening_task()
    actual.step("STOP")
    with pytest.raises(ValueError, match="nominal"):
        freeze_limited_plan(actual, spec=spec, method_record=method_record())
    with pytest.raises(ValueError, match="planning clone"):
        evaluate_native_spatial_episode(task)


@pytest.mark.parametrize("method", ["SEARCH", "IL", "RL", "HYBRID", "MANUAL", "STOP"])
def test_all_method_records_seal_the_same_nominal_stop_model(method):
    spec = permitted_spec()
    task = build_limited_planning_task(spec)
    task.advance_planning("STOP")
    plan = freeze_limited_plan(task, spec=spec, method_record=method_record(method=method))
    assert plan.action_ids == ("STOP",) and plan.terminal_reason == "STOP"
    descriptor = binding(spec)
    report = evaluate_frozen_limited_plan(plan, spec=spec, reference_binding=descriptor,
        load_reference=lambda: load_generated_reference(descriptor))
    assert report["status"] == "evaluated"
    assert report["independent_episode"]["outcomes"]["target_removed_mm3"] == 0


@pytest.mark.parametrize("mutation", ["action", "prefix", "early_stop", "physical", "input", "tool", "observation",
    "decision", "extra", "physical_extra", "method_extra", "horizon", "reason", "time_cap", "incomplete_method"])
def test_plan_tampering_and_incomplete_methods_reject_before_loader(sealed_search, mutation):
    spec, plan, _ = sealed_search
    values = plan.to_record()
    if mutation == "action": values["action_ids"][0] = "foreign"
    elif mutation == "prefix":
        values["action_ids"] = values["action_ids"][:1]
        values["physical_actions"] = values["physical_actions"][:1]
    elif mutation == "early_stop": values["action_ids"][0] = "STOP"
    elif mutation == "physical": values["physical_actions"][0]["entry_mm"][0] += .1
    elif mutation in {"input", "tool", "observation", "decision"}:
        name = {"input": "limited_input_hash", "tool": "tool_catalog_hash",
                "observation": "initial_observation_hash", "decision": "decision_model_hash"}[mutation]
        values[name] = semantic_digest("changed")
    elif mutation == "extra": values["reference_target"] = "private"
    elif mutation == "physical_extra": values["physical_actions"][0]["private_reward"] = 8
    elif mutation == "method_extra": values["method_record"]["reference_hash"] = "private"
    elif mutation == "horizon": values["horizon"] = 3
    elif mutation == "reason": values["terminal_reason"] = "TIME_CAP"
    elif mutation == "time_cap": values["method_record"]["time_cap_reached"] = True
    elif mutation == "incomplete_method": values["method_record"]["completed"] = False
    reads = []
    with pytest.raises(ValueError):
        changed = FrozenLimitedPlan.from_record(values)
        evaluate_frozen_limited_plan(changed, spec=spec, reference_binding=binding(spec),
                                    load_reference=lambda: reads.append(True))
    assert reads == []


@pytest.mark.parametrize("mutation", ["input", "tool", "physical", "action"])
def test_resealed_invalid_plan_still_replays_and_rejects_before_private_read(sealed_search, mutation):
    spec, plan, _ = sealed_search
    values = plan.to_record()
    values.pop("seal_hash")
    if mutation == "input": values["limited_input_hash"] = semantic_digest("other")
    elif mutation == "tool": values["tool_catalog_hash"] = semantic_digest("other")
    elif mutation == "physical": values["physical_actions"][0]["tip_mm"][0] += .1
    else:
        values["action_ids"][0] = "foreign"
        values["physical_actions"][0]["action_id"] = "foreign"
    changed = FrozenLimitedPlan(**values)
    reads = []
    with pytest.raises(ValueError):
        evaluate_frozen_limited_plan(changed, spec=spec, reference_binding=binding(spec),
                                    load_reference=lambda: reads.append(True))
    assert reads == []


def test_object_level_mutation_and_reference_source_mismatch_reject_before_loader(sealed_search):
    spec, plan, _ = sealed_search
    reads = []
    descriptor = binding(spec)
    object.__setattr__(descriptor, "limited_input_hash", semantic_digest("foreign"))
    with pytest.raises(ValueError, match="source binding"):
        evaluate_frozen_limited_plan(plan, spec=spec, reference_binding=descriptor,
                                    load_reference=lambda: reads.append(True))
    changed = FrozenLimitedPlan.from_record(plan.to_record())
    object.__setattr__(changed, "action_ids", ("STOP",))
    with pytest.raises(ValueError, match="seal mismatch"):
        evaluate_frozen_limited_plan(changed, spec=spec, reference_binding=binding(spec),
                                    load_reference=lambda: reads.append(True))
    changed_spec = LimitedPlanningSpec.from_record(spec.to_record())
    object.__setattr__(changed_spec, "access", replace(spec.access, radius_mm=3.))
    with pytest.raises(ValueError, match="lineage"):
        evaluate_frozen_limited_plan(plan, spec=changed_spec, reference_binding=binding(spec),
                                    load_reference=lambda: reads.append(True))
    assert reads == []


@pytest.mark.parametrize("key", ["reference_support", "hazards", "patient_id", "clinical_probability"])
def test_reference_metadata_cannot_claim_unsupported_worlds(sealed_search, key):
    spec, _, _ = sealed_search
    record = binding(spec).to_record()
    record[key] = "unsupported"
    with pytest.raises(ValueError, match="exact fields"):
        HeldOutTargetBinding.from_record(record)
    target = np.zeros((9, 9, 7), np.float32)
    target[0, 0, 0] = 1
    with pytest.raises(ValueError, match="declared generated"):
        HeldOutTargetReference(binding(spec), target)


@pytest.mark.parametrize("failure", ["load", "wrong_type", "wrong_binding", "changed_reference", "independent", "rejected_audit"])
def test_private_load_and_replay_failures_return_sealed_non_success(sealed_search, monkeypatch, failure):
    spec, plan, _ = sealed_search
    descriptor = binding(spec)
    reads = []
    def loader():
        reads.append(True)
        if failure == "load": raise OSError("private source details are not exported")
        if failure == "wrong_type": return make_native_opening_task()
        reference = load_generated_reference(binding(spec, "zero_target_counterfactual")
            if failure == "wrong_binding" else descriptor)
        if failure == "changed_reference":
            object.__setattr__(reference, "reference_target", np.zeros((9, 9, 7), np.float32))
        return reference
    if failure == "independent":
        def fail_audit(task): raise ValueError("audit failed")
        monkeypatch.setattr(boundary, "evaluate_native_spatial_episode", fail_audit)
    elif failure == "rejected_audit":
        monkeypatch.setattr(boundary, "evaluate_native_spatial_episode",
            lambda task: evaluate_native_spatial_episode(task, cancelled=lambda: True))
    report = evaluate_frozen_limited_plan(plan, spec=spec, reference_binding=descriptor, load_reference=loader)
    assert reads == [True]
    assert report["status"] == ({"load": "reference_load_failed", "independent": "evaluator_replay_failed",
                                 "rejected_audit": "independent_audit_failed"}
                                .get(failure, "reference_rejected"))
    if failure == "rejected_audit":
        assert report["independent_episode"]["accepted"] is False
        assert report["independent_episode"]["outcomes"] is None
        assert report["independent_episode"]["geometry"]["failures"] == ("independent_validation_cancelled",)
    else:
        assert report["independent_episode"] is None
    assert "private source details" not in json.dumps(thaw_json(report))
    assert report["plan_seal_hash"] == plan.seal_hash
