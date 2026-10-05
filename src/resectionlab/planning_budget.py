"""Opt-in per-arm accounting; no task, policy, source or scoring changes."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from functools import wraps
import math
import threading
import time


VERSION = "native-planning-budget-v1"
_active_guard = None


class PlanningBudgetViolation(InterruptedError):
    """Sticky refusal, compatible with committed-transition cancellation handling."""

    def __init__(self, reason, accounting):
        super().__init__(reason)
        self.reason = reason
        self.accounting = accounting


class PlanningBudget:
    """One single-threaded arm, including all ordinary class-dispatched previews.

    Compose the existing transparent NativePreviewProfiler instead of installing
    a CPython profile observer. A caught violation remains fatal at scope exit.
    The caller must retain the terminal history before declaring completion and
    exit this context before independent audit. This is not a hard-timeout or a
    geometry/history validator; a process supervisor remains necessary.
    """

    def __init__(self, engine_type, *, max_native_previews=468, seconds=90., _clock=None):
        if type(max_native_previews) is not int or max_native_previews < 0:
            raise ValueError("Preview ceiling must be a nonnegative integer")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Time ceiling must be finite and positive")
        if not isinstance(engine_type, type) or not callable(getattr(engine_type, "preview_stroke", None)):
            raise ValueError("An engine class with preview_stroke is required")
        self.engine_type = engine_type
        self._limits = (max_native_previews, float(seconds))
        self._clock = time.perf_counter if _clock is None else _clock
        self._status = "not_started"
        self._failure = None
        self._active = False
        self._history_complete = None
        self._counting_reliable = True
        self._admitted = self._blocked = self._inflight = 0
        self._requests = {}
        self._request_active = False
        self._profiler = None
        self._started = self._last = None
        self._elapsed = 0.

    def _fail(self, reason, *, unreliable=False):
        if self._failure is None:
            self._failure = reason
        self._status = "failed"
        if unreliable:
            self._counting_reliable = False

    def _time(self):
        try:
            now = self._clock()
            if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
                raise ValueError("invalid clock")
            if self._last is not None and now < self._last:
                raise ValueError("clock moved backward")
        except BaseException:
            self._fail("clock_failed")
            return
        self._last = float(now)
        if self._started is None:
            self._started = self._last
        self._elapsed = self._last - self._started
        if self._elapsed > self._limits[1]:
            self._fail("planning_execution_time_exceeded")

    def _health(self):
        if threading.get_ident() != self._thread:
            self._fail("cross_thread_use", unreliable=True)
        if getattr(self.engine_type, "preview_stroke", None) is not self._wrapper:
            self._fail("preview_wrapper_replaced", unreliable=True)
        if getattr(self._profiler, "original", None) is not self._original:
            self._fail("profiler_original_changed", unreliable=True)
        try:
            records = self._profiler.records.values()
            total = sum(row["started"] for row in records)
            if type(total) is not int or total != self._admitted:
                raise ValueError("counter mismatch")
        except BaseException:
            self._fail("preview_accounting_changed", unreliable=True)

    def _raise_failure(self):
        if self._failure is not None:
            raise PlanningBudgetViolation(self._failure, self.snapshot())

    def check(self):
        """Cooperative checkpoint, also usable inside a task cancellation callback."""
        if not self._active:
            raise RuntimeError("Planning budget is not active")
        self._health()
        self._time()
        self._raise_failure()

    def _work_allowed(self, *, finalizing=False):
        self.check()
        if self._history_complete is not None:
            self._fail("work_after_history_completion")
            self._raise_failure()
        if not finalizing and self._elapsed >= self._limits[1]:
            self._fail("planning_execution_deadline_exhausted")
            self._raise_failure()

    def __enter__(self):
        global _active_guard
        if self._status != "not_started" or _active_guard is not None:
            raise RuntimeError("Planning budgets are single-use and process-exclusive")
        self._thread = threading.get_ident()
        self._time()
        self._raise_failure()
        from .spatial_policy_diagnostics import NativePreviewProfiler
        self._had_own_method = "preview_stroke" in vars(self.engine_type)
        self._original = self.engine_type.preview_stroke
        self._profiler = NativePreviewProfiler(self.engine_type)
        measured = self._profiler.__enter__().engine_type.preview_stroke

        @wraps(self._original)
        def guarded(instance, *args, **kwargs):
            self._work_allowed()
            if self._admitted >= self._limits[0]:
                self._blocked += 1
                self._fail("native_preview_entry_limit")
                self._raise_failure()
            self._admitted += 1
            self._inflight += 1
            try:
                result = measured(instance, *args, **kwargs)
            except BaseException:
                self._fail("preview_or_profiler_raised")
                raise
            finally:
                self._inflight -= 1
                self._health()
                self._time()
            self._raise_failure()
            return result

        self._wrapper = guarded
        self.engine_type.preview_stroke = guarded
        self._active, self._status = True, "active"
        _active_guard = self
        return self

    def __exit__(self, exc_type, exc, traceback):
        global _active_guard
        try:
            self._health()
            self._time()
            if exc is not None:
                self._fail("arm_raised_" + exc_type.__name__)
            elif self._history_complete is not True:
                self._fail("complete_history_not_declared")
            if self._request_active or self._inflight:
                self._fail("unfinished_guarded_work", unreliable=True)
        finally:
            # Restore even after replacement/failure; do not depend on an
            # exception-prone observer callback for restoration or refusal.
            try:
                if self._had_own_method:
                    self.engine_type.preview_stroke = self._original
                elif "preview_stroke" in vars(self.engine_type):
                    delattr(self.engine_type, "preview_stroke")
            finally:
                self._active = False
                _active_guard = None
        if self._failure is None:
            self._status = "complete_history_awaiting_independent_audit"
        if exc is not None:
            # Preserve committed-transition interruption details and original
            # exception identity. Accounting is also always retained on self.
            try:
                exc.planning_budget_accounting = self.snapshot()
            except (AttributeError, TypeError):
                pass
            return False
        self._raise_failure()
        return False

    @contextmanager
    def phase(self, name):
        self._work_allowed()
        if name not in ("planning", "execution"):
            self._fail("unsupported_online_phase")
            self._raise_failure()
        with self._profiler.phase(name):
            try:
                yield
                self.check()
            except BaseException:
                self._fail("phase_raised")
                raise

    @contextmanager
    def request(self, label):
        """Optional request count; zero previews do not establish a cache hit."""
        self._work_allowed()
        if not isinstance(label, str) or not label or len(label) > 64 or self._request_active:
            self._fail("invalid_or_nested_request")
            self._raise_failure()
        row = self._requests.setdefault(label, {"started": 0, "returned": 0, "raised": 0,
            "preview_admissions": 0, "zero_preview_returns": 0})
        row["started"] += 1
        before = self._admitted
        self._request_active = True
        try:
            yield
            self.check()
        except BaseException:
            row["raised"] += 1
            self._fail("request_raised")
            raise
        else:
            row["returned"] += 1
            row["zero_preview_returns"] += int(self._admitted == before)
        finally:
            row["preview_admissions"] += self._admitted - before
            self._request_active = False

    def complete(self, *, history_complete):
        """Caller attests that a terminal history was durably retained; no audit implied."""
        self._work_allowed(finalizing=True)
        if type(history_complete) is not bool or not history_complete:
            self._history_complete = False if history_complete is False else None
            self._fail("incomplete_or_unknown_history")
            self._raise_failure()
        if self._request_active or self._inflight:
            self._fail("completion_during_guarded_work")
            self._raise_failure()
        self._history_complete = True

    def snapshot(self):
        """Copy retained accounting even after failure; never manufacture a score."""
        if self._active:
            self._health()
            self._time()
        try:
            profile = None if self._profiler is None else self._profiler.snapshot()
        except BaseException:
            self._fail("profiler_snapshot_failed", unreliable=True)
            profile = None
        return {"version": VERSION, "status": self._status, "failure": self._failure,
            "limits": {"native_preview_entries": self._limits[0], "planning_execution_seconds": self._limits[1]},
            "native_preview_entries": self._admitted if self._counting_reliable else None,
            "observed_admissions_before_accounting_failure": self._admitted,
            "counting_reliable": self._counting_reliable,
            "blocked_preview_attempts": self._blocked, "elapsed_seconds": self._elapsed,
            "time_overshoot_seconds": max(0., self._elapsed - self._limits[1]),
            "history_complete_caller_attestation": self._history_complete,
            "score": None, "independent_audit": "outside_this_guard_not_verified",
            "requests": copy.deepcopy(self._requests), "cache_hits": None,
            "request_interpretation": "Zero-preview returns may be cached, empty or STOP-only; cache hits are not instrumented.",
            "native_preview_profile": profile,
            "excluded_costs": "Shared original preparation, ingress screening, selected initial inventory and independent audit must be reported separately.",
            "wall_limit": "Cooperative checks; in-flight native/C work cannot be forcibly stopped here. External process supervision is required."}
