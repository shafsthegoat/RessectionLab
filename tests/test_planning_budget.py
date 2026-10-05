"""Standalone mock controls: no patient, checkpoint, native geometry or learning."""
from types import SimpleNamespace
import threading

import pytest

from resectionlab.planning_budget import PlanningBudget, PlanningBudgetViolation


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value


@pytest.fixture
def fake():
    clock, calls = Clock(), []
    result = SimpleNamespace(feasible=True, microsteps=(1, 2), reason="")

    class Engine:
        def preview_stroke(self, tag=None, *, failure=None, delay=0., feasible=True):
            calls.append(tag)
            clock.value += delay
            if failure is not None:
                raise failure
            return result if feasible else SimpleNamespace(feasible=False, microsteps=(), reason="blocked")

    return Engine, clock, calls, result


def test_exact_common_ceiling_counts_all_branches_and_replay_not_shared_or_audit(fake):
    engine, clock, calls, result = fake
    engine().preview_stroke("shared_initial_preparation")
    original = engine.preview_stroke
    guard = PlanningBudget(engine, _clock=clock)
    with guard:
        with guard.phase("planning"):
            for number in range(230):
                with guard.request("inventory"):
                    assert engine().preview_stroke(number) is result
            with guard.request("inventory"):
                cached = result
                assert cached is result
        clock.value = 40.
        with guard.phase("execution"):
            for number in range(238):
                engine().preview_stroke(number)
        clock.value = 89.
        # Durable history recording is inside the same uninterrupted clock.
        clock.value += 1.
        guard.complete(history_complete=True)
    engine().preview_stroke("independent_audit")
    clock.value += 500.
    report = guard.snapshot()
    assert report["native_preview_entries"] == 468
    assert report["elapsed_seconds"] == 90.
    assert report["status"] == "complete_history_awaiting_independent_audit"
    assert report["score"] is None
    assert report["native_preview_profile"]["phases"]["planning"]["started"] == 230
    assert report["native_preview_profile"]["phases"]["execution"]["started"] == 238
    assert report["requests"]["inventory"] == {"started": 231, "returned": 231, "raised": 0,
        "preview_admissions": 230, "zero_preview_returns": 1}
    assert report["cache_hits"] is None
    assert len(calls) == 470 and engine.preview_stroke is original


def test_next_entry_refused_before_engine_and_caught_failure_stays_fatal(fake):
    engine, clock, calls, _ = fake
    guard = PlanningBudget(engine, max_native_previews=2, _clock=clock)
    with pytest.raises(PlanningBudgetViolation, match="entry_limit"):
        with guard:
            engine().preview_stroke("one")
            engine().preview_stroke("two")
            try:
                engine().preview_stroke("must_not_execute")
            except PlanningBudgetViolation:
                pass
    assert calls == ["one", "two"]
    report = guard.snapshot()
    assert report["native_preview_entries"] == 2
    assert report["blocked_preview_attempts"] == 1
    assert report["history_complete_caller_attestation"] is None and report["score"] is None


def test_stop_can_complete_with_zero_previews_and_zero_cap(fake):
    engine, clock, calls, _ = fake
    with PlanningBudget(engine, max_native_previews=0, _clock=clock) as guard:
        with guard.phase("execution"):
            with guard.request("STOP"):
                pass
        guard.complete(history_complete=True)
    assert guard.snapshot()["native_preview_entries"] == 0 and calls == []


@pytest.mark.parametrize("complete", [None, False, 0, "true"])
def test_incomplete_or_unknown_history_never_becomes_zero_score(fake, complete):
    engine, clock, _, _ = fake
    guard = PlanningBudget(engine, _clock=clock)
    with pytest.raises(PlanningBudgetViolation):
        with guard:
            if complete is not None:
                guard.complete(history_complete=complete)
    assert guard.snapshot()["status"] == "failed"
    assert guard.snapshot()["score"] is None


def test_deadline_prevents_next_entry_even_without_any_previews(fake):
    engine, clock, calls, _ = fake
    guard = PlanningBudget(engine, seconds=3., _clock=clock)
    with pytest.raises(PlanningBudgetViolation, match="time_exceeded"):
        with guard:
            clock.value = 3.001
            engine().preview_stroke("not entered")
    assert calls == [] and guard.snapshot()["native_preview_entries"] == 0


@pytest.mark.parametrize("kind", ["preview", "request", "phase"])
def test_new_work_cannot_start_exactly_at_deadline(fake, kind):
    engine, clock, calls, _ = fake
    entered = []
    guard = PlanningBudget(engine, _clock=clock)
    with pytest.raises(PlanningBudgetViolation, match="deadline_exhausted"):
        with guard:
            clock.value = 90.
            if kind == "preview":
                engine().preview_stroke("not admitted")
            elif kind == "request":
                with guard.request("new work"):
                    entered.append(True)
            else:
                with guard.phase("execution"):
                    entered.append(True)
    assert calls == [] and entered == []


def test_inflight_time_overrun_records_finished_preview_and_overshoot(fake):
    engine, clock, calls, _ = fake
    guard = PlanningBudget(engine, seconds=3., _clock=clock)
    with pytest.raises(PlanningBudgetViolation, match="time_exceeded"):
        with guard:
            engine().preview_stroke("completed native call", delay=4.)
    report = guard.snapshot()
    assert calls == ["completed native call"]
    assert report["native_preview_entries"] == 1
    assert report["native_preview_profile"]["phases"]["unclassified"]["returned"] == 1
    assert report["time_overshoot_seconds"] == 1.


def test_context_exit_checks_nonpreview_serialization_time(fake):
    engine, clock, _, _ = fake
    guard = PlanningBudget(engine, seconds=3., _clock=clock)
    with pytest.raises(PlanningBudgetViolation, match="time_exceeded"):
        with guard:
            guard.complete(history_complete=True)
            clock.value = 4.
    assert guard.snapshot()["history_complete_caller_attestation"] is True
    assert guard.snapshot()["status"] == "failed"


def test_native_exception_keeps_identity_and_partial_entry_accounting(fake):
    engine, clock, _, _ = fake
    error = KeyboardInterrupt("analytic cancellation")
    guard = PlanningBudget(engine, _clock=clock)
    with pytest.raises(KeyboardInterrupt) as caught:
        with guard:
            engine().preview_stroke("raised", failure=error)
    assert caught.value is error
    assert error.planning_budget_accounting["native_preview_entries"] == 1
    assert guard.snapshot()["native_preview_profile"]["phases"]["unclassified"]["raised"] == 1


def test_swallowed_native_exception_cannot_finish_as_success(fake):
    engine, clock, _, _ = fake
    guard = PlanningBudget(engine, _clock=clock)
    with pytest.raises(PlanningBudgetViolation, match="preview_or_profiler_raised"):
        with guard:
            try:
                engine().preview_stroke(failure=ValueError("analytic"))
            except ValueError:
                pass
            guard.complete(history_complete=True)


def test_committed_unreturned_interruption_keeps_durable_record(fake):
    engine, clock, _, _ = fake

    class Committed(InterruptedError):
        def __init__(self, info):
            self.info = info

    info = {"action_id": "declared_cut", "committed": True}
    guard = PlanningBudget(engine, seconds=3., _clock=clock)
    with pytest.raises(Committed) as caught:
        with guard:
            with guard.phase("execution"):
                try:
                    clock.value = 4.
                    guard.check()
                except InterruptedError as error:
                    raise Committed(info) from error
    assert caught.value.info is info
    assert isinstance(caught.value.__cause__, PlanningBudgetViolation)
    assert caught.value.planning_budget_accounting["failure"] == "planning_execution_time_exceeded"


@pytest.mark.parametrize("mutation", ["replace", "delete", "counters", "delegate"])
def test_instrumentation_loss_is_sticky_and_actual_total_becomes_unknown(fake, mutation):
    engine, clock, calls, _ = fake
    original = engine.preview_stroke
    guard = PlanningBudget(engine, _clock=clock)
    with pytest.raises(PlanningBudgetViolation):
        with guard:
            engine().preview_stroke("one")
            if mutation == "replace":
                engine.preview_stroke = original
            elif mutation == "delete":
                del engine.preview_stroke
            elif mutation == "counters":
                guard._profiler.records.clear()
            else:
                guard._profiler.original = lambda *a, **k: None
            try:
                guard.check()
            except PlanningBudgetViolation:
                pass
    report = guard.snapshot()
    assert report["native_preview_entries"] is None
    assert report["observed_admissions_before_accounting_failure"] == 1
    assert report["counting_reliable"] is False
    assert engine.preview_stroke is original and calls == ["one"]


def test_profiler_error_before_underlying_entry_is_not_silently_disabled(fake):
    engine, clock, calls, _ = fake
    guard = PlanningBudget(engine, _clock=clock)

    class BrokenRecords(dict):
        def setdefault(self, *args):
            raise RuntimeError("observer failed")

    with pytest.raises(PlanningBudgetViolation):
        with guard:
            guard._profiler.records = BrokenRecords()
            try:
                engine().preview_stroke("not reached")
            except RuntimeError:
                pass
    assert calls == []
    assert guard.snapshot()["native_preview_entries"] is None


def test_snapshot_failure_poisoning_and_returned_snapshot_is_a_copy(fake, monkeypatch):
    engine, clock, _, _ = fake
    guard = PlanningBudget(engine, _clock=clock)
    with pytest.raises(PlanningBudgetViolation, match="snapshot_failed"):
        with guard:
            with guard.request("inventory"):
                engine().preview_stroke()
            report = guard.snapshot()
            report["requests"]["inventory"]["started"] = 200
            report["native_preview_profile"]["phases"].clear()
            assert guard.snapshot()["requests"]["inventory"]["started"] == 1
            def fail():
                raise ValueError("observer snapshot failed")
            monkeypatch.setattr(guard._profiler, "snapshot", fail)
            assert guard.snapshot()["failure"] == "profiler_snapshot_failed"


def test_foreign_thread_refuses_entry_and_poisoning_is_visible_to_owner(fake):
    engine, clock, calls, _ = fake
    guard, errors = PlanningBudget(engine, _clock=clock), []
    with pytest.raises(PlanningBudgetViolation, match="cross_thread"):
        with guard:
            def run():
                try:
                    engine().preview_stroke()
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=run); thread.start(); thread.join()
    assert len(errors) == 1 and isinstance(errors[0], PlanningBudgetViolation)
    assert calls == []


@pytest.mark.parametrize("clock_value", [-1., float("nan"), float("inf"), True])
def test_clock_failure_cannot_create_free_budget(fake, clock_value):
    engine, clock, calls, _ = fake
    guard = PlanningBudget(engine, _clock=clock)
    with pytest.raises(PlanningBudgetViolation, match="clock_failed"):
        with guard:
            clock.value = clock_value
            engine().preview_stroke()
    assert calls == []


def test_single_use_and_exclusive_scope_restore_inherited_method(fake):
    engine, clock, _, _ = fake
    class Child(engine):
        pass
    guard = PlanningBudget(Child, _clock=clock)
    with guard:
        with pytest.raises(RuntimeError, match="exclusive"):
            with PlanningBudget(engine, _clock=clock):
                pass
        guard.complete(history_complete=True)
    assert "preview_stroke" not in vars(Child)
    with pytest.raises(RuntimeError, match="single-use"):
        with guard:
            pass


def test_deleted_inherited_wrapper_still_releases_exclusive_scope(fake):
    engine, clock, _, _ = fake
    class Child(engine):
        pass
    with pytest.raises(PlanningBudgetViolation, match="wrapper_replaced"):
        with PlanningBudget(Child, _clock=clock):
            del Child.preview_stroke
    with PlanningBudget(engine, _clock=clock) as later:
        later.complete(history_complete=True)


def test_failed_or_nested_requests_and_work_after_completion_are_not_success(fake):
    engine, clock, _, _ = fake
    for action in ("nested", "raised", "after_complete"):
        guard = PlanningBudget(engine, _clock=clock)
        with pytest.raises((PlanningBudgetViolation, ValueError)):
            with guard:
                if action == "after_complete":
                    guard.complete(history_complete=True)
                with guard.request("inventory"):
                    if action == "nested":
                        with guard.request("another"):
                            pass
                    if action == "raised":
                        raise ValueError("request failure")
        assert guard.snapshot()["status"] == "failed"


@pytest.mark.parametrize("options", [{"max_native_previews": True}, {"max_native_previews": -1},
    {"max_native_previews": 1.5}, {"seconds": 0.}, {"seconds": True},
    {"seconds": float("nan")}, {"seconds": float("inf")}])
def test_invalid_limits_refused_before_instrumentation(fake, options):
    engine, _, _, _ = fake
    original = engine.preview_stroke
    with pytest.raises(ValueError):
        PlanningBudget(engine, **options)
    assert engine.preview_stroke is original
