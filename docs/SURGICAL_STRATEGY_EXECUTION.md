# Surgical strategy foundation — current execution record

Updated October 31, 2026. Goal: one source-bound, reproducible system for
comparing executable patient-specific strategies. The pasted October 31 steering
supersedes the older training-first priorities. No new training or acquisition
was started. Work began at `72c81843` on `shaf/data-foundation-rebuild`; unrelated
data and experimental retractor changes in the primary checkout are preserved.

## Completed: explicit shared physical clock v1

The current-code reproducer confirmed the audit's CUSA defect: at a fixed
generated contact and an assumed rate of 0.01 mm³/s, a latched left CUSA received
0 versus 0.002 mm³ capacity during 0.2 s of left versus right inspection.
The new mode supplies 0.002 mm³ in either case and gives identical complete
physical states. Two prechange legacy state hashes remain exact regression gates.

`surgical_clock.py` schedules both instruments in left/right order, validates
their contacts before effects, and counts each tool's effect once per interval.
The engine advances elapsed time, deformation, drilling cooling, bipolar heat,
vascular evolution and fluid fields once. Contact/exposure accounting includes
the persistent inserted tools rather than only the inspection owner.

Enable through **Rehearse → Environment and source setup → Enable shared
physical clock**, with both instruments withdrawn/inactive and no holds.
The button records `configure` with `clockRevision: 1`; the version is hashed in
physical state and displayed as `view.physicalClock`. It does not reset anatomy,
pending work, accumulated burdens, mechanics selection or elapsed time. Like
historical mechanics configuration, this is a zero-time setup operation; its
positive action duration is a required envelope field, not a dwell.

No old recording opts in automatically. Replaying or seeking before that action
uses historical behavior, including the earlier thermal-scene exception. Native
save/reopen and verified/observed replay retain the boundary. Scene identity
alone does not identify the clock: compare source plus state/action identity.
No clock downgrade operation is exposed.

Activation/deactivation occurs before the first accepted interval. `apply`,
`hold`, `grasp` and explicit mapping inspection have transient activation;
latched switches continue through other timed actions. Release, deactivation
and withdrawal turn off the selected switch before their interval. Ordinary
motion retains the existing spatial subdivision and maximum 0.1 s temporal
step. Complete microsteps commit atomically; a failed microstep rolls back both
hands and every field, leaving only the accepted prefix. Cancellation discards
the whole unpublished action candidate.

This is deterministic operator splitting for sequential motion with a persistent
stationary other hand. It does not implement continuously simultaneous paths.
Matched stationary time grids are tested for exact state equality. Uneven CUSA
partitions use absolute 1e-14 tolerances for floating quantities, retaining exact
binary capacity thresholds; no tolerance authorizes whole-cell clearance.
Nonlinear coupled fields retain their existing numerical discretization and
qualification limits. Experimental planar-maneuver and surface-retractor clocks
are not admitted to this new mode.

## Executed checks and evidence

- 22 generated scheduling tests pass: legacy hashes, activation, two active
  CUSAs, fixed and uneven partitions, one-shot/latched equivalence, irrigation/
  suction accounting, retractor relaxation, thermal/vascular evolution,
  rollback, cancellation, branching, verified caches and observed frames.
- The first focused regression run passes 281 tests. The broader integration
  run passes 337 with one optional compiled-prefix test skipped. These suites
  overlap; counts are not additive. The final clock run adds the later controls.
- Desktop build and all 1,212 existing desktop tests pass (322 main, 634 renderer,
  256 viewer). Build retains its existing large-bundle warning.
- Actual hidden Electron controls enable the clock and mechanics, activate a
  CUSA, and fork own/opposite 0.2 s inspection from the same boundary. Each
  supplies 0.4 mm³ and reaches the exact same state. Both branches save and
  cold-replay independently; a fresh hidden Electron process verifies the saved
  endpoint. The source archive is unchanged and no window becomes visible.
- Ignored evidence: `testcases/shared-clock-v1/`. The first desktop attempt is
  retained: its comparison also wrote previously absent left-tool parameter
  defaults. The corrected contrast binds both parameter records before the
  timed fork. No simulator tolerance was weakened to obtain a matching state.

Reproduce from the repository root:

```sh
.venv/bin/python -m pytest tests/test_surgical_clock.py -q
.venv/bin/python scripts/prepare_clock_verification.py
cd desktop
pnpm build
pnpm test
CLOCK_OUTPUT="$PWD/../testcases/shared-clock-v1/desktop-fresh" node_modules/.bin/electron electron/verify-shared-clock.cjs
```

Choose a fresh output directory to preserve prior evidence. The preparer reuses
its identical native generated control on subsequent runs. For a cold desktop
reopen, run the same verifier with `CLOCK_REOPEN_ONLY=1`, `CLOCK_CASE` set to the
saved workspace, `CLOCK_EXPECTED_STATE` from its report, and a fresh `CLOCK_OUTPUT`.

## Remaining milestones and next dependency

The scheduling defect is repaired in an explicit opt-in mode. This is software
and numerical scheduling evidence, not empirical mechanics accuracy or clinical
validation. Legacy episodes intentionally retain the historical defect.

Next: declarative patient/task admission for two independently prepared acquired
cases and a missing-evidence control, with the clock/physics versions and public
observation contract bound to each task. Preserve the frozen assessment failures
and missing-anatomy refusals; do not merely remove the UCSF-0007 name check.
The common planner information boundary, complete-tool sequence checks,
source-linked functional endpoint, end-to-end profiling and frozen uncertainty
comparison remain open. Older lattice resolution dependence and unresolved
continuum empirical/domain gates remain qualification holds. No speed, learned
planning advantage, neurological probability or clinical-readiness claim is made.
