# Repaired Accelerate runtime: eight existing controls

**Prepared only. No solver execution is authorized by this document.**
The final private runtime/profile pins are pending the separately reviewed
CSC and matrix-attribute initialization repairs. Its actual identity
must exist and be independently accepted before a separate root release. Missing
or mismatched identity is a refusal, never an invitation to use the old runtime.

`scripts/mechanics_accelerate_controls.py` prepares five original HEX8 controls
(`zero`, `translation`, `finite_stretch`, `shear`, `shear_double_stiffness`), then
three original tet10/MPC controls (`tet10_affine`, `mpc_translation`,
`mpc_nonrigid`). The new files live in
`artifacts/mechanics-accelerate-controls-v1/decks`. Original Skyline decks,
launchers, material laws, element domains, four-step grids, boundary conditions,
primitive outputs, numerical tolerances and independent checkers are unchanged.
The original deck manifests and both checker implementations are hash-pinned.

The shared backend adapter replaces exactly one literal solver subtree:
`accelerate`, `iterative=0`, `factorization=4`, `order_method=0`,
`print_condition_number=0`. Reversing this replacement recovers every original
deck byte. The repaired runtime uses the reviewed existing private OpenMP
library; this preparation installs or copies no runtime/dependency.

## Generic commit boundary

This checkpoint has **no active `declaration.json`**. The eight transformed deck
bytes are indexed by `software-preparation-manifest.json`, which cannot authorize
execution. The initial stale draft is preserved as
`initial-unreleased-declaration.json` for provenance only. Final runtime/profile
pins must wait for the combined adapter controls and accepted build; a later
reviewed declaration must then be generated and committed before any release.
The runner currently refuses because its active declaration is absent.

## One separately released attempt

The release template is deliberately invalid and has `authorized=false`.
A future release must bind a full Git commit, its immutable source archive
inside the same repository,
repository, one fresh absolute attempt path inside that repository, the actual
runtime-identity path/SHA and root's explicit authorization. The fixed source
closure is exported as `CLOSURE` by the runner. Every archived file is compared
against its exact Git object, then the declaration, original/adapted decks and
runtime executable/library/repair bytes are verified. Setup performs no solves.

One existing `febio_runtime.supervise` call controls the worker process group:
**60 seconds aggregate, 3 GiB sampled group RSS, one numerical thread, eight
maximum solver calls and zero retries**. This includes the worker's repeated
source checks, solver calls and output checks. Setup in the parent is recorded
separately. Sampling can miss brief between-sample memory peaks; this is not a
new resource monitor. The shared private environment removes injected build and
library settings and fixes the original numerical thread environment.

The worker initializes all eight statuses. At the first solver, backend,
primitive-checker, stiffness-scaling, source-integrity or output-inventory
failure, it stops invoking the solver and preserves every unexecuted case.
Stiffness scaling is checked immediately after the fifth case, before any MPC
case. An existing attempt directory or already-started worker receipt cannot be
reused. No fallback solver, changed tolerance, extra case or automatic retry is
available.

## Actual evidence required

Each executed case must contain the actual console message selecting the
Accelerate solver. A default-solver announcement alone is insufficient;
conflicting selection messages fail. This is combined with the exact model XML,
executed deck SHA and the same actual runtime identity. It establishes the
selected implementation and declared settings, not an instrumented count of
internal factorization calls.

The unchanged HEX8 and tet10/MPC checkers consume actual node/element/solver logs.
The original shear/doubled-stiffness checker verifies scaling. Parent success
additionally requires complete worker evidence, successful supervision and all
bound input bytes unchanged before and after execution. A zero process exit
alone is insufficient.

Two summaries preserve the previous consumer shapes:

- `hex8-summary.json`: `status=passed_all_five_fixed_patch_controls`, `rows` with
  five cases, `stiffness_scaling.passed`, and five solver invocations.
- `tet10_mpc-summary.json`: `status=three_actual_fixed_software_controls_passed`,
  `case_rows` with three cases and three solver invocations.

Both bind `runtime_identity_sha256`, `solver_backend=accelerate`, the same parent
execution/results/baseline records and complete per-case evidence inventories.
The latter bind original and executed decks, console, primitive logs, solver log,
checker source and checked result. The backend profile consumer independently
verifies those saved files and re-evaluates the unchanged checks; summary booleans
alone confer no authority. The parent receipt is written before summaries, so
there is no circular digest. A publication failure rewrites that receipt to
failed and invalidates any earlier summary binding. Failed summaries explicitly
recheck both original checker hashes rather than assuming they stayed unchanged.

These are small software-verification controls only. Even eight successful
cases would not establish patient mesh accuracy, tissue properties, material
calibration, clinical validity or surgical safety. No patient or measured curve
input is accepted by this runner.
