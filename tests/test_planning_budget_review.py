"""Independent mocked budget controls; no native geometry, patient or policy run."""
from types import SimpleNamespace
import math

import pytest

import resectionlab.planning_budget as module
from resectionlab.planning_budget import PlanningBudget, PlanningBudgetViolation


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value


def fake_engine():
    calls = []
    class Engine:
        def preview_stroke(self, label):
            calls.append(label)
            return SimpleNamespace(feasible=True, reason="", microsteps=())
    return Engine, calls


def test_deleted_inherited_instrumentation_restores_exception_and_releases_scope():
    base, _ = fake_engine()
    class Child(base):
        pass
    guard = PlanningBudget(Child, _clock=Clock())
    caught = None
    try:
        with guard:
            Child().preview_stroke("first")
            del Child.preview_stroke
    except BaseException as error:
        caught = error
    try:
        assert module._active_guard is None, "failed cleanup leaked the process-exclusive budget"
        assert isinstance(caught, PlanningBudgetViolation), repr(caught)
        assert guard.snapshot()["native_preview_entries"] is None
        assert "preview_stroke" not in vars(Child)
        with PlanningBudget(base, _clock=Clock()) as fresh:
            base().preview_stroke("separate later arm")
            fresh.complete(history_complete=True)
    finally:
        # Keep the intentionally broken pre-repair control from poisoning pytest.
        if module._active_guard is guard:
            module._active_guard = None


def test_469th_preview_cannot_enter_after_multiple_branches_and_replay():
    engine, calls = fake_engine()
    original = engine.preview_stroke
    budget = PlanningBudget(engine, _clock=Clock())
    branches = [engine() for _ in range(3)]
    with pytest.raises(PlanningBudgetViolation, match="entry_limit"):
        with budget:
            with budget.phase("planning"):
                for branch_index, branch in enumerate(branches):
                    for action in range(78):
                        branch.preview_stroke(("branch", branch_index, action))
            with budget.phase("execution"):
                replay = engine()
                for action in range(234):
                    replay.preview_stroke(("replay", action))
                try:
                    replay.preview_stroke("469th must never reach engine")
                except PlanningBudgetViolation:
                    pass
                with pytest.raises(PlanningBudgetViolation):
                    budget.complete(history_complete=True)
    snapshot = budget.snapshot()
    assert len(calls) == snapshot["native_preview_entries"] == 468
    assert snapshot["blocked_preview_attempts"] == 1
    assert snapshot["native_preview_profile"]["phases"]["planning"]["started"] == 234
    assert snapshot["native_preview_profile"]["phases"]["execution"]["started"] == 234
    assert engine.preview_stroke is original and module._active_guard is None


def test_actual_native_entries_and_explicit_requests_are_different_denominators():
    engine, calls = fake_engine()
    budget = PlanningBudget(engine, _clock=Clock())
    with budget:
        with budget.request("fresh_inventory"):
            engine().preview_stroke("first")
            engine().preview_stroke("second")
        with budget.request("already_available_inventory"):
            pass
        with budget.request("STOP_only"):
            pass
        # A scored candidate is not another native request and is not inferred.
        scores = [1.0, 2.0]
        assert max(scores) == 2.0
        engine().preview_stroke("direct replay call without request wrapper")
        budget.complete(history_complete=True)
    snapshot = budget.snapshot()
    assert len(calls) == snapshot["native_preview_entries"] == 3
    assert snapshot["requests"]["fresh_inventory"]["preview_admissions"] == 2
    assert snapshot["requests"]["fresh_inventory"]["zero_preview_returns"] == 0
    assert snapshot["requests"]["already_available_inventory"]["zero_preview_returns"] == 1
    assert snapshot["requests"]["STOP_only"]["zero_preview_returns"] == 1
    assert snapshot["cache_hits"] is None and snapshot["score"] is None


def test_clock_includes_clone_replay_and_retention_but_freezes_before_external_audit():
    engine, calls = fake_engine()
    clock = Clock()
    engine().preview_stroke("shared preparation outside arm")
    clock.value = 1000.
    with PlanningBudget(engine, _clock=clock) as budget:
        clock.value += 13.  # Mock clone/setup work, still inside online arm.
        with budget.phase("planning"):
            engine().preview_stroke("branch")
            clock.value += 29.
        with budget.phase("execution"):
            engine().preview_stroke("replay")
            clock.value += 37.
        clock.value += 10.  # Mock durable history retention/serialization.
        budget.complete(history_complete=True)
    engine().preview_stroke("independent audit outside arm")
    clock.value += 500.
    snapshot = budget.snapshot()
    assert snapshot["elapsed_seconds"] == 89.
    assert snapshot["native_preview_entries"] == 2 and len(calls) == 4
    assert snapshot["status"] == "complete_history_awaiting_independent_audit"
    assert snapshot["independent_audit"] == "outside_this_guard_not_verified"


def test_inclusive_finish_and_first_representable_overrun_have_distinct_status():
    engine, calls = fake_engine()
    clock = Clock()
    with PlanningBudget(engine, seconds=90., _clock=clock) as completed:
        clock.value = 90.
        completed.complete(history_complete=True)
    assert completed.snapshot()["elapsed_seconds"] == 90.
    clock.value = 0.
    with pytest.raises(PlanningBudgetViolation, match="time_exceeded"):
        with PlanningBudget(engine, seconds=90., _clock=clock) as failed:
            clock.value = math.nextafter(90., math.inf)
            engine().preview_stroke("overrun must not enter")
    assert calls == [] and failed.snapshot()["native_preview_entries"] == 0
    assert failed.snapshot()["time_overshoot_seconds"] > 0


@pytest.mark.parametrize("attestation", [None, False, "complete", 1])
def test_incomplete_history_does_not_become_complete_or_receive_zero_score(attestation):
    engine, _ = fake_engine()
    with pytest.raises(PlanningBudgetViolation):
        with PlanningBudget(engine, _clock=Clock()) as budget:
            engine().preview_stroke("uncommitted analytical preview")
            if attestation is not None:
                budget.complete(history_complete=attestation)
    snapshot = budget.snapshot()
    assert snapshot["status"] == "failed" and snapshot["score"] is None
    assert snapshot["history_complete_caller_attestation"] is not True


def test_swallowed_native_exception_is_sticky_and_retains_original_error_identity():
    sentinel = ValueError("constructed native refusal")
    class Engine:
        def preview_stroke(self, label):
            raise sentinel
    budget = PlanningBudget(Engine, _clock=Clock())
    with pytest.raises(PlanningBudgetViolation):
        with budget:
            try:
                Engine().preview_stroke("one actual entry")
            except ValueError as error:
                assert error is sentinel
            # Finishing the arm cannot erase a swallowed failure.
    snapshot = budget.snapshot()
    assert snapshot["native_preview_entries"] == 1
    assert snapshot["native_preview_profile"]["phases"]["unclassified"]["raised"] == 1
    assert snapshot["failure"] == "preview_or_profiler_raised"


def test_caught_wrapper_displacement_makes_total_unknown_not_the_observed_prefix():
    engine, calls = fake_engine()
    original = engine.preview_stroke
    budget = PlanningBudget(engine, _clock=Clock())
    with pytest.raises(PlanningBudgetViolation):
        with budget:
            engine().preview_stroke("recorded prefix")
            engine.preview_stroke = original
            engine().preview_stroke("unobserved after external displacement")
            try:
                budget.check()
            except PlanningBudgetViolation:
                pass
    snapshot = budget.snapshot()
    assert len(calls) == 2 and snapshot["observed_admissions_before_accounting_failure"] == 1
    assert snapshot["native_preview_entries"] is None and not snapshot["counting_reliable"]
    assert engine.preview_stroke is original


def test_unsupported_audit_phase_cannot_hide_work_in_online_clock():
    engine, calls = fake_engine()
    with pytest.raises(PlanningBudgetViolation, match="unsupported_online_phase"):
        with PlanningBudget(engine, _clock=Clock()) as budget:
            with budget.phase("independent_audit"):
                engine().preview_stroke("not reached")
    assert calls == []


def test_completion_seals_further_work_and_snapshot_is_detached():
    engine, calls = fake_engine()
    with pytest.raises(PlanningBudgetViolation, match="work_after_history"):
        with PlanningBudget(engine, _clock=Clock()) as budget:
            with budget.request("retained"):
                engine().preview_stroke("before complete")
            saved = budget.snapshot()
            saved["requests"]["retained"]["started"] = -1
            saved["native_preview_profile"]["phases"].clear()
            assert budget.snapshot()["requests"]["retained"]["started"] == 1
            budget.complete(history_complete=True)
            engine().preview_stroke("after declared completion")
    assert calls == ["before complete"]


def test_existing_external_profiler_is_restored_instead_of_silently_removed():
    from resectionlab.spatial_policy_diagnostics import NativePreviewProfiler
    engine, calls = fake_engine()
    original = engine.preview_stroke
    with NativePreviewProfiler(engine) as external:
        prior_wrapper = engine.preview_stroke
        with PlanningBudget(engine, _clock=Clock()) as budget:
            engine().preview_stroke("both observers")
            budget.complete(history_complete=True)
        assert engine.preview_stroke is prior_wrapper
        engine().preview_stroke("external only after arm")
    assert len(calls) == 2
    assert budget.snapshot()["native_preview_entries"] == 1
    assert external.snapshot()["phases"]["unclassified"]["started"] == 2
    assert engine.preview_stroke is original


@pytest.mark.parametrize("entry", ["native_preview", "request"])
def test_no_new_work_may_start_at_exact_deadline(entry):
    engine, calls = fake_engine()
    clock = Clock()
    budget = PlanningBudget(engine, seconds=90., _clock=clock)
    with pytest.raises(PlanningBudgetViolation):
        with budget:
            clock.value = 90.
            if entry == "native_preview":
                engine().preview_stroke("must refuse before original method")
            else:
                with budget.request("no remaining time"):
                    calls.append("request body must not execute")
            budget.complete(history_complete=True)
    assert calls == []
