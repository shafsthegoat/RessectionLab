# Conditional state-driven CUSA/continuum pilot

The [pre-effect study](continuum-pre-effect-qualification.md) finds two genuine
conditional removal discrepancies. Native future removal masks cannot qualify a
coupled action sequence. This next stage advances the actual CUSA ledger and
full continuum history together from an explicitly declared nonzero seed.

`CusaContinuumEpisode` owns source masks, current displacement and every Maxwell
field, spring targets/holds, tool poses, exact CUSA pending work and clocks.
Every physical microtick queries the current deformation, limits the requested
CUSA pose, checks all tool shafts/materials and conservative tool-pair covers,
then calls the original CUSA allocator. Its actual new owner mask rebuilds the
source graph. Retained source-cell/corner histories transfer bit-for-bit;
removed tissue energy is exported once. The next fixed steps use that graph.
There is no interface accepting a prescribed future removal event.

The unchanged four-fraction source-center obstacle/FOV guard runs on every BE
increment. Gradient, displacement, reaction, full free-force, actual spring/Maxwell
energy and cumulative balance checks remain active, without clipping. Nonzero
initial energy uses `E-E0+removed-contactJump+loss-work=0`; release has a signed
spring-potential jump. CUSA capacity is not assigned mechanical work or injury.
Held-owner and outside-domain removal refuse; multiple completed owners transfer
atomically. Configured protection, rigid fragments, cuts, gravity and another
active tool require separate support and refuse this restricted pilot.

Failed microticks restore every physical field, ledger, clock and exact numeric
binding. A batch becomes observable only after full commit, through a detached
pull API. Later evidence-writing failures are execution failures, not retroactive
physics rejections. Resource costs of attempted preparation remain charged.

Generated controls preload nonzero exterior displacement and memory. They compare
all nine ticks/108 BE steps against independently solved full equations, check
fresh exact replay, release/relaxation, multiple removals, outside-domain and
held-owner refusals, late/cancelled-step rollback and batch isolation. These are
algebraic/software controls, not patient or device validation.

## Frozen source cost profile

The seed is short-hold step 853's complete coupled pre-event field, reconstructed
losslessly and anchored to its full displacement/memory hash. The native CUSA
before-ledger is an explicit initial condition. The seed has 5,501 previously
removed source owners; no historical removed energy is invented for them.

Six explicit 0.1-second physical CUSA microticks repeat the captured requested
pose; a 0.1-second stationary release and two stationary passive ticks follow.
The actual seed clock remainder is preserved. These interventions are not six
public actions: the public scheduler may subdivide them differently when free
to advance. Future masks and poses remain state-driven. Exact initial pending
work plus six fixed budgets bound new removal at two unit source cells, without
prescribing their identities or requiring that both complete.

`continuum-closed-loop-profile-v1.json` first allows a read-only 15-second/1-GiB
seed preflight, then profiles two algebraic partitions under 120 seconds, 3 GiB
and 16 MiB. Both use the same full 20 mm mesh, materials, supports and histories.
The first retains the original 8 mm algebraic core. The second selects every
supported-free coordinate touched by the actual spring matrix as a smaller
core, mapped to root source lineage and frozen through release. All exterior
DOFs remain present and counted. This is not a new physical domain or boundary.

The profile measures the initial preparation, one actual state-driven tick,
and a separately labeled hypothetical deletion of the leading contact owner.
That deletion tests rebuild/transfer cost, not a CUSA event or a future outcome.
Its preparation is rolled back without modifying physical fields or ledgers.
Retaining an exterior requires exact current exterior and both coupling blocks;
otherwise the default refuses. The explicit fresh-update profile charges both
old rollback and new factors concurrently. Earlier factors are released between
variants. A first-deletion peak does not bound a second-deletion peak.

The full short pilot remains conditional on the measured profile. It must keep
its own hard resource cap, compare two fresh complete histories exactly, export
selected first/last BE checkpoints and all-step hashes/scalars, and pass an
independent source Q1/ledger checker. No full 900-second replay is justified by
an unmeasured extrapolation. Existing live caps and failed outer-radius evidence
remain unchanged; physical validation is false and both RL holds remain intact.

## Completed cost profile

The frozen profile at `50ca6c0f` completes in **102.83 seconds supervised**
(101.90 seconds worker), at 2,360,492,032 sampled RSS bytes and 2,362,540,032 OS
peak bytes. Metadata output is 29,486 bytes. The preceding seed-only preflight
takes 2.36 seconds at 390,086,656 sampled bytes, with no episode or solve.

| Algebraic core | Root core free DOFs | Initial preparation | Candidate rebuild | First microtick |
| --- | ---: | ---: | ---: | ---: |
| Original 8 mm core | 5,496 | 58.04 s | 4.26 s, retained exterior | 1.18 s |
| Spring coordinates | 96 | 18.18 s | 16.57 s, fresh exterior | 1.17 s |

The smaller core reduces Schur-column time from 48.42 to 2.96 seconds, while
its larger exterior factor increases factor time from 5.61 to 12.19 seconds.
Both partitions retain all 33,360 supported-free and 49,260 total source DOFs.
Their complete first-microtick displacement/memory difference is 1.87e-16 mm,
within the unchanged 1e-9 mm comparison budget. No physical model was reduced.

For the profiled nine-tick intervention and at most two new removals, measured
two-pass worker estimates are 156.42 seconds for the original core and 125.69
seconds for the spring-coordinate core. These are estimates, not unseen-step
or second-removal peak guarantees. The source profile executes only one real
microtick per partition; its next-owner deletion is a matrix/history cost probe.

## Prospective complete short pilot

The next frozen intervention adds one active CUSA microtick after the profiled
six, to exercise advance into a newly cleared corridor. Seven active ticks,
one release and two passive ticks total ten microticks/120 BE steps. The exact
initial pending work plus seven budgets still bounds new whole-owner removal
at two cells. No owner identity or removal timing is prescribed.

Use the measured spring-coordinate partition, keep all physical DOFs and old/new
factor overlap charged, and retain a hard 180-second combined, 3-GiB, 32-MiB
ceiling. The worker has at most 145 seconds and the independent checker at most
30 seconds from the remaining combined wall budget. Two fresh histories must
match every full-field/scalar hash and causal tick receipt exactly. Twenty first
and last BE sample worlds, propagated capacity decisions, transfer/removal energy,
release, relaxation, clocks, source ownership and advance require independent
export verification. Only committed batches publish observations; partial or
failed execution retains its receipts and cannot admit the pilot.

## Preserved first pilot failure and targeted revision

Frozen `dac5247e` runs the complete first history in 73.2141 seconds, including
54.0802 seconds in its three source operator preparations. Actual removals are
`3403863` on tick 5 and `3404173` on tick 6; tick 7 advances the CUSA tip by
0.322821 mm. Release and both passive ticks complete in that first history.
Twenty committed field checkpoints are preserved.

The second fresh history reaches its final passive tick, then the worker refuses
at its unchanged 140-second cooperative threshold inside the pre-tick geometry
check. Supervision ends at 140.8036 seconds; sampled group RSS is 2,613,821,440
bytes and OS peak RSS is 2,693,857,280 bytes, both under 3 GiB. PID 87131 is
terminal and absent. There is no recorded physics failure, memory-cap kill or
tool/approval failure. The independent checker never starts. The pilot is
**incomplete**, and its failure receipts remain under `closed-loop-pilot-v1/`.
The first launcher did not persist its partial second-pass replay digests; reaching
the final tick is not a substitute for the missing complete replay artifact.

An isolated, read-only profile reconstructs all 100 before-sweep/after-endpoint
Q1 capsule queries from the first history's exports, with no source solve or MRI
decode. All post-endpoint Q1 memberships match their committed receipts. The
geometry queries take 2.548 seconds without the execution callback; with the
original output-directory guard they take 6.322 seconds. Its 10,939 callbacks
spend 3.713 seconds rescanning the output directory, including about 580,000
filesystem stat calls. At most seven of 13,210 source boxes are contact candidates,
but changing the geometry algorithm is unnecessary to remove this guard cost.

Execution revision 2 routes every owned artifact write through a byte ledger.
It checks the exact prospective total before writing, counts replacements once,
and audits the complete directory at exit. The same 64 KiB worker reserve and
32 MiB combined ceiling remain. Every physics checkpoint still checks elapsed
time. Generated controls require pre-write cap refusal without file corruption,
replacement accounting, and refusal of an unexpected file at the final audit.

Committed progress is recorded only after a whole physical tick succeeds.
The first-pass scalars/checkpoints and partial second-pass replay digests persist
on interruption. A generated second-pass cancellation verifies that exactly the
previous committed tick is reported and incomplete status remains false for
qualification. No physical equation, contact/rank rule, numerical threshold,
intervention, fresh-replay requirement or resource cap changes in revision 2.

The revised wall-only callback takes 0.004665 seconds across those same 10,939
calls. Complete query time is 2.5573 seconds, a 2.47x isolated speedup over the
original guarded version, and every full query-result digest is identical.
These are instrumented geometry measurements, not a full-pilot completion claim.
The final acceptance file is included in the global prospective byte total.

Failure receipts distinguish the last actual physical commit from the last
complete export/verified replay batch. A generated failure while exporting tick 2
must report physical tick 2 and exported tick 1, retain only tick 1's complete
scalar/sample batch, and remain unqualified. Partial file-write failures reconcile
actual bytes and propagate; they cannot turn into successful qualification.

## Completed conditional source pilot

Execution revision 2, frozen at `97e9a23e`, passes in **142.2078 seconds combined**.
Worker time is 132.2352 seconds (133.1127 supervised); the independent checker
takes 8.7165 seconds. Both worker and checker exit 0. Sampled group peak RSS is
2,910,568,448 bytes; OS worker peak is 2,916,483,072 bytes, below the unchanged
3-GiB ceiling. Output is 25,065,953 bytes, below 32 MiB. Both processes are terminal
and absent. The first and second fresh histories take 66.3638 and 64.0568 seconds.

All **120 complete field/scalar records and 10 causal microtick receipts replay
exactly**. Twenty first/last BE sample worlds, plus the initial payload, are
exported. The independent checker folds every scalar/clock/hash record, reconstructs
the complete seed and dynamic source graph, and independently integrates all 20
sampled Q1 force/Maxwell/spring/energy worlds. The other 100 intermediate fields
have scalar/hash/fresh-replay coverage, not independent full quadrature.

The propagated coupled ledger removes tumor owners `3403863` and `3404173` on
ticks 5 and 6, in that order. Tick 7 advances the CUSA tip by 0.322821 mm and
leaves exactly 0.2 mm³ pending work on owner `3441217`; it does not remove that
owner. Release and two passive ticks leave the CUSA ledger unchanged and preserve
nonzero Maxwell memory. New removal is 2 mm³ of target tissue and zero non-target
tissue. The declared initial condition already includes 561 mm³ target and
4,940 mm³ non-target removal; those costs remain in the complete source ledger.
This one-second intervention is not the original short-hold program outcome.

Removed tissue/history energy is 7.9276621e-9 N mm; release contributes a signed
spring-model jump of -1.0664364e-5 N mm. Initial and final stored energies are
3.1653995e-5 and 3.1552053e-6 N mm. Mechanical work is zero; viscous plus BE loss
is 1.7826498e-5 N mm. Maximum cumulative event/step balance residual is 3.95e-20
N mm. Maximum independent sampled free-force residual is 2.87e-19 N; sampled
step energy balance is 8.91e-21 N mm. All original thresholds remain unchanged.

The checker also proves separation of 40 sampled full-source shaft covers and
all associated conservative tool-pair covers. This does not validate unsampled
contacts, unannotated critical anatomy, patient/device response or physiology.
Across execution revisions, all 10 first-history causal receipts (excluding
artifact-container metadata) and all 160 arrays in the 20 checkpoint worlds are
byte-exact. The unchanged live engine, continuum, history solve, deformation and
CUSA files still match accepted commit `0d66dd7e`.

The expanded suite passes **265 regression tests in 22.41 seconds**, alongside
12 independent generated checker controls. Exact launch/source/input hashes,
resource receipts, partial-failure evidence and cross-execution comparisons are
in the ignored `continuum-exterior-events-v1/` collection, summarized by
`closed-loop-stage-v1.json`.

## Remaining executable gate

Six fresh source preparations consume 97.947 seconds, about 74% of worker time.
The physical root still contains 13,211 cells, 49,260 total and 33,360 free DOFs;
the 96-coordinate algebraic core does not reduce that model or bypass live caps.
Profile conservative tool-corridor algebraic partitions next, checking exact
exterior/coupling identity and full-field parity before using retained updates.
Then qualify the actual public scheduler from a prefix evolved by the coupled
model, with state-driven geometry, capacity and masks throughout. The present
hybrid declared seed does not establish that complete original 20 mm trajectory.

The failed 16-to-20 mm stress-stability gate, mixed/negative donor mechanics,
patient/tumor response, source-supported vascular consequences and neurological
consequences remain unresolved. Physical and larger-domain public admission are
false; physiological and large-scale RL stay on hold. The scientific goal remains
active.
