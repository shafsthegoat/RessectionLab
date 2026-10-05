# Equal online planning budget

`resectionlab.planning_budget.PlanningBudget` is an opt-in accounting guard for one comparison arm. Existing task, policy, planner, objective and default behavior are unchanged. The prospective common ceilings are **468 actual native preview entries and 90 seconds of planning plus execution**. Equal ceilings permit unequal actual usage; neither number is an expected count.

Start the scope before the arm's clone, policy calls or search. Keep one clock through all search branches, replay, and durable terminal-history recording. The shared original preparation, static ingress screen and selected initial inventory are measured once outside this scope. Independent full-history audit follows scope exit and remains subject to the separate whole-run supervisor. There is no pause/reset API.

```python
with PlanningBudget(NativeResectionEngine) as budget:
    with budget.phase("planning"):
        # Optional: count a request without calling it a cache hit.
        with budget.request("observation"):
            observation = branch.observation()
        # Frozen actor or permitted-model search uses the same guard.
    with budget.phase("execution"):
        # Replay, check terminal history, and retain it durably here.
        pass
    budget.complete(history_complete=True)
accounting = budget.snapshot()
# Independent audit is separate; it is still necessary before accepting a result.
```

The example omits task execution; calling `complete` is a caller attestation, not a substitute for a terminal-history check. The existing `run_episode` performs audit before returning, so a new runner must explicitly separate this boundary. It cannot simply wrap the existing whole call and claim audit was excluded. `budget.check()` can be composed into the task's cancellation callback without adding preview counts. Serialization inside the online scope is charged.

The guard composes `NativePreviewProfiler` with a class-method wrapper. Every ordinary `preview_stroke` entry on that engine class is charged across all branch instances and replay, including native rejection or exception. The next entry after the count ceiling is refused before native work; finishing exactly at a ceiling is allowed. New previews, requests and phases cannot start at the time deadline. `blocked_preview_attempts` counts entry-count-ceiling refusals specifically; earlier time/health refusals remain in the failure record rather than this counter. Cached certificates and candidate scoring incur no invented native calls. Optional request counters retain starts, returns, exceptions and preview admissions; a zero-preview request may be cached, empty or STOP-only, so `cache_hits` remains null. Greedy `evaluated_nonstop_actions` is a separate scoring count.

Violations are sticky. Catching an exception does not restore budget authority: later checks and normal scope exit refuse completion. `PlanningBudgetViolation` derives from `InterruptedError`, allowing the task's existing committed-transition interruption path to retain a committed action. An escaping exception retains its identity and receives `planning_budget_accounting`; the guard also retains a copied snapshot. Lost wrapper/delegate/counter authority makes the actual total null while preserving observed admissions. Incomplete or undeclared histories never produce a zero score: this module always returns `score: null`. Successful budget completion is labelled `complete_history_awaiting_independent_audit`.

This is single-use, process-exclusive and single-threaded instrumentation, not a security boundary against arbitrary Python. Current native task inventories dispatch through `self._engine.preview_stroke`; pre-captured method aliases, subclasses overriding that method, other processes and direct invocation of the original function are outside this contract. The runner must use the declared engine path. No CPython profile callback is installed, so a callback exception cannot silently unset this wrapper. Wrapper replacement and accounting corruption are checked explicitly.

Time checks occur before and after native calls, requests, completion and scope exit. They do not forcibly interrupt native/C work. Overrun duration is reported and the arm fails; a hard process deadline and durable partial receipts remain caller obligations. No patient files, checkpoints, models or scientific experiments are needed by the guard's tests.

Initial owner controls exposed an inherited-method cleanup defect: deleting the installed wrapper led to a second `delattr` failure and left the exclusive slot occupied (32 passed, 2 failed; draft source `b1c1ee95c0b5d83fd110d072e699659877694b6d69aa45eb7107da28c30e3eb9`). The independent reviewer preserved the source and reproduced negative before repair. Restoration is now conditional, with unconditional exclusive-slot cleanup. The first corrected owner suite passed 34 controls in 1.21 seconds. Independent controls then found that new work could start at the exact time deadline; the preserved two-case negative led to separate entry (`>=`) and completion (`>`) checks without changing either ceiling. The final owner suite passed 37 controls in 0.99 seconds. Independent final review is separate.
