"""Independent orchestration adversaries; no patient, checkpoint or model reads."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import math

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("prepared_comparison_independent", ROOT / "scripts/compare_prepared_training_planners.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def valid_arm(name="STOP"):
    binding = {"source_hash": "source", "model_hash": "model", "initial_state_hash": "state", "max_steps": 3}
    return {
        "name": name, "status": "complete", "initial_binding": binding,
        "final_initial_binding": deepcopy(binding), "initial_parameter_hash": runner.frozen.POLICY_HASH,
        "final_parameter_hash": runner.frozen.POLICY_HASH, "closure_verified": True,
        "independent_evaluation_accepted": True, "terminal_history_sha256": "a" * 64,
        "episode_sha256": "b" * 64, "outcomes": {"total_reward": 0.},
        "planning_budget": {
            "status": "complete_history_awaiting_independent_audit", "failure": None,
            "history_complete_caller_attestation": True, "counting_reliable": True,
            "native_preview_entries": 0, "blocked_preview_attempts": 0,
            "limits": {"native_preview_entries": 468, "planning_execution_seconds": 90.},
            "elapsed_seconds": 1., "time_overshoot_seconds": 0.,
        },
    }


def write_patient(output, *, mutate=None, supervisor=None):
    subject = "sub-PAT22"
    directory = output / subject
    directory.mkdir(parents=True)
    row = {
        **runner.blank(subject), "status": "complete", "closure_verified": True,
        "initial_binding": valid_arm()["initial_binding"],
        "arms": {name: valid_arm(name) for name in runner.METHODS},
    }
    if mutate:
        mutate(row)
    (directory / "receipt.json").write_text(json.dumps(row))
    (directory / "closure-check.json").write_text(json.dumps({"sources_inputs_checkpoint_unchanged": True}))
    (directory / "supervisor.json").write_text(json.dumps(supervisor or {
        "status": "complete", "returncode": 0, "timed_out": False, "termination_reason": None,
    }))
    return directory


def test_constructed_complete_three_arm_authority_retains_six_denominator(tmp_path):
    write_patient(tmp_path)
    result = runner.summarize(tmp_path)
    assert result["patients_prescribed"] == 6 and len(result["patients"]) == 6
    assert result["complete_comparisons"] == 1


@pytest.mark.parametrize("change", ["empty", "missing_stop", "wrong_subject"])
def test_missing_arm_or_wrong_patient_cannot_publish_complete_comparison(tmp_path, change):
    def mutate(row):
        if change == "empty":
            row["arms"] = {}
        elif change == "missing_stop":
            del row["arms"]["STOP"]
        else:
            row["subject"] = "sub-PAT26"
    write_patient(tmp_path, mutate=mutate)
    assert runner.summarize(tmp_path)["complete_comparisons"] == 0


@pytest.mark.parametrize("supervisor", [
    {"status": "failed", "returncode": 1, "timed_out": False, "termination_reason": "worker_error"},
    {"status": "failed", "returncode": 0, "timed_out": True, "termination_reason": "parent_wall_budget_exceeded"},
])
def test_failed_parent_authority_cannot_publish_even_if_worker_payload_completed(tmp_path, supervisor):
    write_patient(tmp_path, supervisor=supervisor)
    assert runner.summarize(tmp_path)["complete_comparisons"] == 0


def test_arm_dictionary_keys_must_match_actual_method_identity(tmp_path):
    def mutate(row):
        row["arms"]["greedy_search"]["name"] = "STOP"
    write_patient(tmp_path, mutate=mutate)
    assert runner.summarize(tmp_path)["complete_comparisons"] == 0


def test_historical_block_cannot_be_promoted_into_an_attempted_patient(tmp_path):
    directory = write_patient(tmp_path)
    row = json.loads((directory / "receipt.json").read_text())
    row["subject"] = "sub-PAT16"
    (directory / "receipt.json").write_text(json.dumps(row))
    directory.rename(tmp_path / "sub-PAT16")
    assert runner.summarize(tmp_path)["complete_comparisons"] == 0


@pytest.mark.parametrize("key,value", [
    ("failure", "native_preview_entry_limit"),
    ("blocked_preview_attempts", 1),
    ("time_overshoot_seconds", .001),
])
def test_completed_budget_label_cannot_override_recorded_resource_violation(key, value):
    row = valid_arm()
    row["planning_budget"][key] = value
    assert not runner.validate_completed_arm(row, row["initial_binding"])


@pytest.fixture
def real_budget_arm(tmp_path):
    """Actual guard around toy class calls; no simulator/torch/model construction."""
    from resectionlab.planning_budget import PlanningBudget
    clock = SimpleNamespace(value=0.)
    events = []

    class Engine:
        def preview_stroke(self, label):
            events.append(label)
            return SimpleNamespace(feasible=True, microsteps=(), reason="")

    original = Engine.preview_stroke
    budget = PlanningBudget(Engine, _clock=lambda: clock.value)
    controls = SimpleNamespace(planning_seconds=20., execution_seconds=30., planning_calls=2,
                               execution_calls=3, audit_seconds=400., malformed_terminal=False,
                               saved_mutation=None)
    binding = valid_arm()["initial_binding"]

    class Base:
        def observed_greedy_search(self, *, seconds):
            assert seconds <= 90.
            for _ in range(controls.planning_calls):
                Engine().preview_stroke("planning")
            clock.value += controls.planning_seconds
            return ("cut", "STOP"), {"complete": True, "evaluated_nonstop_actions": 1234}

    base, policy = Base(), object()

    def episode(base_, policy_, generator, *, mode, name, output, guard, audit, sequence):
        assert base_ is base and policy_ is policy and generator is None
        guard()
        for _ in range(controls.execution_calls):
            Engine().preview_stroke("execution")
        clock.value += controls.execution_seconds
        guard()
        metrics = {"terminated": True, "decision_model_hash": "model", "history": [{"action_id": "STOP"}],
                   "crop": {"origin_voxels": (1, 2, 3), "shape": (4, 5, 6)},
                   "full_precision_field": 1.25, "nested": {"geometry": [(0.25, 1.5, 2.75)]}}
        task = SimpleNamespace(terminated=True, metrics=lambda: deepcopy(metrics))
        record = {"status": "awaiting_independent_check", "metrics": metrics,
                  "decisions": [{"action_id": "STOP", "decision_seconds": 0., "transition_seconds": 0.}]}
        if controls.malformed_terminal:
            record["metrics"] = {**metrics, "history": []}
        runner.write_json(output / (name + ".json"), record)
        if controls.saved_mutation is not None:
            path = output / (name + ".json")
            saved = json.loads(path.read_text())
            controls.saved_mutation(saved["metrics"])
            # Deliberately permit a nonfinite malformed JSON fixture for that negative control.
            path.write_text(json.dumps(saved))
        result = audit(task, metrics=metrics)
        guard()
        record.update(status="complete", independent_evaluation=result)
        runner.write_json(output / (name + ".json"), record)
        return (), record

    def auditor(task, *, metrics):
        assert Engine.preview_stroke is original
        events.append("audit")
        clock.value += controls.audit_seconds
        # A separate fake work call proves that online counting was restored.
        Engine().preview_stroke("audit_only")
        return {"accepted": True, "outcomes": {"total_reward": 0.}, "target_access_success": False}

    kwargs = {"whole_guard": lambda: None, "auditor": auditor, "closure_check": lambda: None,
              "budget_factory": lambda: budget, "episode_runner": episode,
              "policy_hasher": lambda _: runner.frozen.POLICY_HASH,
              "binding_reader": lambda _: deepcopy(binding)}
    return SimpleNamespace(output=tmp_path, controls=controls, events=events, kwargs=kwargs,
                           base=base, policy=policy, budget=budget, binding=binding, engine=Engine, original=original)


def test_actual_guard_one_clock_covers_planning_and_replay_but_not_audit(real_budget_arm):
    s = real_budget_arm
    row = runner.run_arm(s.base, s.policy, "greedy_search", s.output, **s.kwargs)
    assert runner.validate_completed_arm(row, s.binding)
    assert row["planning_budget"]["elapsed_seconds"] == 50.
    assert row["planning_budget"]["native_preview_entries"] == 5
    assert row["search_accounting"]["evaluated_nonstop_actions"] == 1234
    assert s.events == ["planning"] * 2 + ["execution"] * 3 + ["audit", "audit_only"]


def test_actual_guard_cumulative_planning_plus_execution_timeout_is_not_reset(real_budget_arm):
    s = real_budget_arm
    s.controls.planning_seconds = 60.
    s.controls.execution_seconds = 31.
    row = runner.run_arm(s.base, s.policy, "greedy_search", s.output, **s.kwargs)
    assert row["status"] == "failed" and row["outcomes"] is None
    assert row["planning_budget"]["elapsed_seconds"] == 91.
    assert "audit" not in s.events and s.engine.preview_stroke is s.original


def test_actual_guard_refuses_entry_469_before_engine_and_keeps_partial_unscored(real_budget_arm):
    s = real_budget_arm
    s.controls.planning_calls = 469
    row = runner.run_arm(s.base, s.policy, "greedy_search", s.output, **s.kwargs)
    assert row["status"] == "failed" and row["outcomes"] is None
    assert s.events == ["planning"] * 468
    assert row["planning_budget"]["native_preview_entries"] == 468
    assert row["planning_budget"]["blocked_preview_attempts"] == 1
    assert s.engine.preview_stroke is s.original


def test_actual_guard_mismatched_saved_history_never_reaches_audit(real_budget_arm):
    s = real_budget_arm
    s.controls.malformed_terminal = True
    row = runner.run_arm(s.base, s.policy, "frozen_il", s.output, **s.kwargs)
    assert row["status"] == "failed" and row["outcomes"] is None
    assert "audit" not in s.events and s.engine.preview_stroke is s.original


@pytest.mark.parametrize("mutation", ["added_field", "deleted_field", "one_ulp", "integer", "nested", "nonfinite"])
def test_canonical_roundtrip_does_not_hide_any_field_or_numeric_change(real_budget_arm, mutation):
    s = real_budget_arm
    def change(metrics):
        if mutation == "added_field":
            metrics["unexpected_field"] = 0
        elif mutation == "deleted_field":
            del metrics["nested"]["geometry"]
        elif mutation == "one_ulp":
            metrics["full_precision_field"] = math.nextafter(metrics["full_precision_field"], math.inf)
        elif mutation == "integer":
            metrics["crop"]["shape"][0] += 1
        elif mutation == "nested":
            metrics["nested"]["geometry"][0][2] += .01
        else:
            metrics["full_precision_field"] = float("nan")
    s.controls.saved_mutation = change
    row = runner.run_arm(s.base, s.policy, "frozen_il", s.output, **s.kwargs)
    assert row["status"] == "failed" and row["outcomes"] is None
    assert "audit" not in s.events and s.engine.preview_stroke is s.original


def test_canonical_json_is_full_record_without_rounding_or_field_projection():
    original = {"crop": {"shape": (64, 64, 64)}, "nested": [(1.25, 2., 3.)], "unknown": None}
    serialized = json.loads(json.dumps(original))
    assert runner.canonical_json(original) == runner.canonical_json(serialized)
    assert runner.canonical_json({"v": 1.25}) != runner.canonical_json({"v": math.nextafter(1.25, math.inf)})
    assert runner.canonical_json({"v": 1}) != runner.canonical_json({"v": 1.0})
    assert runner.canonical_json(original) != runner.canonical_json({**original, "extra": 0})
    with pytest.raises(ValueError):
        runner.canonical_json({"v": float("inf")})


def test_runtime_source_inventory_uses_frozen_paths_without_git_discovery(tmp_path, monkeypatch):
    paths = {"scripts/" + name for name in runner.SCRIPT_CLOSURE}
    paths |= {"src/resectionlab/" + name for name in (
        "planning_budget.py", "native_access_preparation.py", "native_spatial_task.py",
        "native_spatial_evaluation.py", "spatial_policy.py")}
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen source fixture\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: pytest.fail("Archive runtime queried Git"))
    result = runner.source_inventory(paths)
    assert set(result) == paths and all(len(value) == 64 for value in result.values())


def test_imported_resectionlab_module_outside_frozen_root_is_rejected(tmp_path, monkeypatch):
    from types import ModuleType
    root = tmp_path / "archive"
    root.mkdir()
    external = tmp_path / "mutable_checkout" / "intruder.py"
    external.parent.mkdir()
    external.write_text("# outside captured source root\n")
    module = ModuleType("resectionlab._independent_external_probe")
    module.__file__ = str(external)
    monkeypatch.setattr(runner, "ROOT", root)
    # Limit inspection to this explicit test module; no real module is imported.
    monkeypatch.setattr(runner, "sys", SimpleNamespace(modules={module.__name__: module}))
    with pytest.raises(ValueError):
        runner.assert_imported_source_closure({"source_sha256": {}})
