# Decision-system progress and next dependency

Branch: `shaf/decision-system-v1`; baseline `53c49019`. Work remains in the
primary checkout. This is an implementation milestone, not closure of the
patient-resection goal or clinical validation.

Implemented:

- Explicit protected/critical source masks survive categorical target overlap,
  bind automatically in the authoritative engine, and cannot be erased by a
  supplemental protection contract. Both tools, shaft, continuous sweep,
  withdrawal and supported indirect effects are checked. A protected proposal
  rolls back its entire tentative action while preserving accepted history.
- Reviewed unknown-coverage task boxes now also apply to direct engine actions
  and native checked skills. Existing autonomous-task admission remains closed
  for the five projects; its restrictions were not removed.
- Bounded two/three-skill staged lookahead retains preparation states and checked
  exits, reports Pareto outcomes and explicit preferences, and meters discarded
  work. The original horizon-1 controller remains reproducible.
- Shared sequence accounting includes original-target residuals, cumulative
  treatment, modeled time, exchanges, withdrawal and missing evidence. The
  desktop exposes this within the existing outcome summary on demand. Native
  cancellation now targets the request itself, including before progress.

Evidence is retained under `testcases/decision-system-v1/` (ignored). The
independent case review verified source planes, units/transforms, originals,
TRAIN identities, target recipes and fresh native replay for all five TopCoW
candidates (004/006/012/014/018). Their compact/full manifests and exact commands
are in `case-qualification/`. All five require repair or narrowed task scope.
The original reported brainstem incident remains unidentified.

The strongest existing baseline found is CTA014 stock kerf preparation followed
by CUSA: 15.810416 mm³ target (5.8524%), 272.901532 mm³ cumulative non-target,
280.04 modeled seconds and completed withdrawal, reproduced by independent
cold replay. This is retained as baseline evidence, not a new improvement.

The explicit protected-overlap before/after control changes actual manual/bridge
execution from accepted forbidden removal to zero-effect rejection. The native
fixed-radius planning control changes STOP to clearance → target → withdrawal,
with one mm³ target and one mm³ non-target. The full controller has another
solution on that generated geometry; this does not meet the material patient
improvement requirement. Exact source hashes, calls, timing and replay evidence
are in `execution-audit/` and `planning-trace/`.

Validation: 257 selected backend tests passed; a subsequent 106-test overlapping
admission/cache suite passed after preserving existing empty-coverage snapshot
identity. All 1,272 desktop tests and the desktop build passed (existing bundle
size warning). Independent review reproduced the safety counterexample,
original-target overlap accounting and strongest CTA014 baseline. It also
retained an obsolete training test which fails before cancellation because its
synthetic-case API is disabled; no bypass was introduced.

Hidden Electron checks pass on UCSF-0007 and the UCSF-0005 no-route case, including
edits, missing-evidence refusal, actual backend cancellation, preview/installation,
save and fresh-process reopen. The 0007 cold run also plays/pauses recorded native
frames, returns to an action boundary/planning checkpoint, and recalculates the
original-target ledger. Every window stayed hidden and source archives stayed
unchanged. `milestone-results.json` indexes results and the retained failed
verifier attempts. Screenshots were inspected; native actions and state assertions,
not the screenshots, establish behavior.

Reproduce focused checks:

```sh
.venv/bin/python -m pytest -q tests/test_declared_surgical_protection.py \
 tests/test_surgical_sequence_outcomes.py tests/test_surgical_sequence_search.py \
 tests/test_surgical_staged_planning.py tests/test_surgical_staged_skills.py \
 tests/test_surgical_staged_exit.py tests/test_shared_surgical_guards.py \
 tests/test_surgical_task_domain.py tests/test_surgical_modeled_outcomes.py
pnpm --dir desktop build
pnpm --dir desktop test
DECISION_SYSTEM_CHECK=1 \
 INTERACTIVE_PLANNING_OUTPUT="$PWD/testcases/decision-system-v1/desktop-repeat" \
 desktop/node_modules/.bin/electron desktop/electron/verify-interactive-planning.cjs
```

Next substantive dependency: qualify a source-bound task with the required
critical/coverage evidence and whole-tool opening geometry, then freeze
size-appropriate target/spatial/burden/exit/computation goals before comparing
the strongest existing controller with a search that jointly branches over
opening, preparation, work configuration and target order. Preserve the CTA014
positive and all five negative cases. Do not turn missing labels into clear
space, change original splits, shrink instruments, or treat correct STOP as
completed treatment. The unified Case → Plan → Compare → Rehearse workflow,
three meaningful completed patient tasks, uncertainty evaluation and matched
patient improvement remain open.
