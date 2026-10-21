# Coordinated shallow-burring result

The successful script and all three learned Q policies select **BURR → CLEAR**.
Each removes 34 mm³ of bone and aspirates all 34 mm³ of generated debris. Either
instrument alone and CLEAR → BURR → STOP fail. The drill-only repeat reaches
64 mm³ and fails the 40 mm³ bone cap; that failure is retained.

| Assessment | Success | Mean return |
| --- | ---: | ---: |
| Script | 1/1 | 1.2916 |
| Learned, all seeds | 3/3 | 1.2916 |
| Initialized Q policy | 0/3 | -0.0504 |
| Seeded random scheduling | 8/30 | 0.2516733 |

All successful learned trajectories have zero healthy/tumor removal, vessel
damage, generated bleeding and protected removal. Bone heating is recorded and
uncalibrated; this is not a clinical safety result. The anatomy, rendering and
engine safeguards were not changed.

Actual temporal-difference updates: **92 / 93 / 93**. A separate read-only
reconstruction exactly matches every Q table from the recorded transitions.
Every seed passes fresh public-action replay, native save/fresh-process reopen,
source-cell bone-only accounting and debris mass conservation. Actual Electron
scrubbing shows 34 mm³ of debris at state 5 and zero at state 10, with source
slices, completed anatomy surfaces, both tools and all 11 strict exclusions.

Validation: **148 focused Python tests pass**; the desktop TypeScript check and
production build pass. No renderer or anatomy implementation was edited.

This is learned **sequential scheduling of scripted spatial macros**, on one
fixed development anatomy. It matches the successful script and shows useful
behavior relative to initialization/random scheduling. It establishes neither
continuous bimanual control nor patient generalization. The shallow pocket is
not a completed craniotomy. Scope stops here.

The [complete numerical summary](summary.json) retains every assessment row and
measured cost scope. [Task/protocol documentation](../../docs/coordinated-burring.md)
describes reproduction, reward, failures and limitations. Complete generated
datasets, checkpoints, telemetry and native demonstrations remain under ignored
`testcases/coordinated-burring-run03/`.

An initial motion variant failed the baseline and ran zero updates. Subsequent
plain-case export errors were repaired; the one repeated frozen fit produced
identical hashes for all three checkpoints. Final export recovery performed zero
extra optimizer updates. Fit/assessment: 69 measured engine calls, 27.998 s;
export recovery: 100 calls including replay/telemetry, 54.429 s. Interrupted
export cost is unavailable, so these are not full campaign totals. Earlier
attempts and the prior surgical pilot's negative result are preserved.
