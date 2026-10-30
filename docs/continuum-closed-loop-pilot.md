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
