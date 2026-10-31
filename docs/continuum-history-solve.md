# Source-owned continuum history solve

An opt-in solver advances the voxel continuum with full distributed Maxwell
memory, including contact changes, cuts, removal and release. It reuses an
equilibrium compliance operator instead of refactorizing the complete contact
system. This changes the computation of the declared research equations; it
does not qualify their material, contact or artificial-boundary assumptions.

The [larger-domain reference](continuum-contact-reference.md) remains useful
for boundary diagnosis. Its two frozen 16-to-20 mm stress-stability gates still
fail. The [donor response comparison](continuum-donor-transfer.md) remains
mixed/negative. Neither result is overridden by numerical solver agreement.

## Algebra and admissible domain

For equilibrium stiffness `K0`, Maxwell stiffnesses `Kj = αj K0`, source-cell
corner-average contact operator `C` and positive contact springs `D`, define:

```text
rj = τj / (τj + h)             β = 1 + Σ rj αj
g = K0_ff⁻¹ bodyForce_f         B = K0_ff⁻¹ C_fᵀ
v = (g + Σ rj αj zj_old) / β    S = C B
(β I + D S) q = D(target - C v)
u = v + B q
zj_new = rj zj_old + (1 - rj) u
```

Unlike the zero-initial-field offline reduction, `zj_old` here is the complete
source-owned nodal field. It need not lie in the contact basis. Changing the
body force recomputes its equilibrium field. Removing all contacts still
advances distributed memory and body loading. The engine retains its existing
memory update arithmetic, free-residual, energy, gradient, displacement,
collision and source-occupancy checks.

Each material must share the same proportional relaxation spectrum. Matrix
proportionality is checked independently to 64 machine epsilons of coefficient
scale. The equilibrium stiffness must be supported independently of contact
springs; springs cannot hide a free rigid component. Constrained memory must
remain zero. Unsupported configurations refuse explicitly rather than fitting
a reduced basis, projecting strain or resetting history.

Topology changes rebuild operators through the retained source-cell, cut,
contact and support signature. Every retained node recovers the first available
source owner's corner value; inconsistent shared copies refuse. Removed
elements export their pre-removal equilibrium/Maxwell stored energy once, with
their source IDs. Cached matrices and bases contain no episode memory.

## Support identity and computation versions

An adversarial test found that removing a soft neighbor outside the active
patch can change its constrained faces. The prior cache signature omitted that
removal. Both newly tagged detailed and proportional variants include current
removed source IDs in support identity, so a warm cache agrees with a fresh
system. This support correction is shared by both variants; it is not counted
as a solver speedup. Untagged recordings retain their historical convention.

The `continuumLinearSolveV1` property accepts complete versioned declarations.
Revision 1 retains the original tensor/history arithmetic. Explicit revision 2
uses a contiguous BLAS tensor contraction and vectorized first-owner unpacking.
Its identity is distinct because floating-point contraction order can vary
between numerical builds. Existing untagged and revision-1 recordings do not
silently select revision 2. Loaded-code replay fingerprints bind the new solver.

Exact graph copying also has a builtin-float 8-by-3 corner fast path. A shared
memo preserves aliases and cycles, and unusual values keep ordinary fallback.
Occupancy hashes use the existing synchronously owned canonical JSON path.
Canonical bytes, mutation detection and custom-type fallback remain checked.

## Frozen acquired-source workload

The [first protocol](../manifests/experiments/continuum-history-qualification-v1.json)
was frozen at `93e56e4b`. It reached the 890-second cooperative wall limit inside
the 900-second supervisor budget. Eight paired rows were recorded, but observer,
legacy and native-case completion were not reached. That attempt remains
**incomplete**; its timings do not establish a throughput improvement. Small
development profiles overlapped it and further limit timing interpretation.

The [second protocol](../manifests/experiments/continuum-history-qualification-v2.json)
was frozen at `2e760043`, after measured hotspot changes and 377 passing tests
(one skip). It preserves all three complete programs, two alternating-order
repetitions, numerical disagreement tolerances, separate observer replay and
six original exact replays. Its caps remain 900 seconds, 2 GiB process-group
RSS, 32 MiB output, one numerical thread and no retry.

That second execution also remains **incomplete**. All twelve paired runs
finished in 711.61 seconds, with every accepted action, repeated receipt and
physical-state comparison passing. It then reached the 890.07-second cooperative
cap during sustained-hold observer replay. The supervisor recorded exit 1,
no kill or cleanup error, and 660,258,816 bytes sampled peak group RSS. Both
recorded worker PIDs were absent when work resumed; no active experiment was
restarted. The evidence identifies a runtime-budget failure, with no numerical
test failure, approval rejection or observed memory-cap exhaustion. Nineteen
preserved system exports independently pass geometry, ownership, constrained
history and equilibrium checks (maximum residual 3.17e-19 N), but missing event
metadata prevents removal-ledger acceptance of that incomplete attempt.

An isolated revision-2 withdrawal profile measured 7.67 of 19.07 seconds in
occupancy lookup, including 5.40 seconds in canonical cache-key validation and
encoding. A targeted private-key change snapshots sorted source keys and exact
finite builtin-double bytes. It detects in-place edits and signed zero; other
types retain canonical validation. Public state digests, serialized values,
collision construction and all mechanics remain unchanged. The same isolated
profile measured 3.64 seconds in occupancy and 14.75 seconds overall afterward.
These instrumented development timings are not an end-to-end throughput claim.
Observer phase/completion markers and partial export indexes now survive a
cooperative failure. The frozen protocol, numerical thresholds and caps are
unchanged.

The subsequent isolated source preflight completes all 29 short-hold actions in
57.92 seconds, matching every saved revision-2 action receipt bit-exactly. An
original untagged sustained-hold recording also reconstructs its entire snapshot
exactly in 25.58 seconds. Together with the measured paired/observer costs, this
supports a separately frozen execution under the same 900-second protocol; it
does not convert either prior timeout into a completed qualification.
The subsequent engine, continuum, CUSA, rollback, replay/cache, vascular, thermal
and RL-hold regression selection passes 612 tests with one skip. No training
updates are performed.

## Completed revision-2 qualification

The distinct `execution-v3-snapshot` attempt uses the unchanged revision-2
protocol from source commit `0d66dd7e`. It completes in **874.76 seconds**
(875.38 seconds including supervision), with **693,895,168 bytes** sampled peak
group RSS and **8,632,110 bytes** of retained output. Exit status is zero;
numerical execution acceptance is true. The cooperative 890-second margin is
only 15.24 seconds, so this is completion evidence with limited timing headroom.

All twelve paired runs finish in 620.68 seconds. This includes 300 detailed/new
solver public transitions, preserving accepted and partial actions, retraction,
CUSA work/removal, relaxation, release and checked STOP. All repeated per-action
receipts and final snapshots match bit-exactly within each variant. Three
separate observer replays also match their complete snapshots exactly, and all
six original untagged snapshots reconstruct bit-exactly. Original arrays,
archive SHA256 and frozen source-code hashes remain unchanged.

| Full-state quantity | Maximum paired disagreement | Frozen absolute limit |
| --- | ---: | ---: |
| Displacement | 3.33e-16 mm | 1e-9 mm |
| Force | 9.76e-19 N | 1e-10 N |
| Energy | 3.05e-20 N mm | 1e-10 N mm |
| Strain-squared volume-time | 1.67e-16 mm³ s | 1e-9 mm³ s |
| Stress-squared volume-time | 1.16e-9 Pa² mm³ s | 0.001 Pa² mm³ s |
| Other floating state | 1.33e-14 | 1e-9 |

Discrete state remains exact. Both resection branches remove the expected
additional 8 target mm³ and zero additional non-target mm³, export seven active
source elements' stored energy once, and retain the full 4,940 mm³ non-target
access cost. Residual target is 18,800 mm³. The complete exports contain 22
systems and fourteen source-owned removal events.

Median complete action times from the two alternating-order repetitions are:

| Program | Detailed | Proportional history | Detailed / history |
| --- | ---: | ---: | ---: |
| Short hold | 67.66 s | 65.47 s | 1.033 |
| Sustained hold | 73.64 s | 71.53 s | 1.029 |
| Checked STOP | 13.60 s | 13.36 s | 1.018 |

These are full accepted/partial action-program costs, excluding comparison and
observer exports, with only two repetitions. The solver's complete-episode
advantage is modest; private key work benefits both variants. They do not
establish interactive latency or RL planning throughput. The remaining isolated
profile costs include continuum checks/integration, rollback copying and
occupancy reconstruction.

The independent primitive checker passes all 22 exported systems and fourteen
removal events. Its maximum free-force residual is 3.17e-19 N; separately
integrated removal energy differs from the source-owned ledger by at most
6.62e-23 N mm. A separate provenance audit passes 23 completion/hash/hold checks.
This numerical gate is complete. Both prior timeouts retain their original
failed/incomplete receipts and are not relabeled.

Both variants reuse the same DEVELOPMENT UCSF-PDGM-0007 source archive by SHA256
`0098232cceb946cbc5cccf00ca81b3b8b5c768c726d379fc8e8dbb503cd13127`.
The 8 mm patch has 1,203 elements and 5,508 corner DOFs, within unchanged live
caps of 2,000 elements and 24,000 DOFs. Geometry established this choice before
the new response: seven of the original eight removed source cells are inside
this patch, whereas none are inside the historical 4 mm patch. An 8 mm patch
is not a physically adequate boundary merely because it can run.

The programs are the original short hold, sustained hold and checked STOP.
The continuation removes eight additional target mm³ and zero additional
non-target mm³, while retaining the full prior 4,940 mm³ non-target access cost.
No material, relaxation, contact or tool parameter is fitted or changed. Source
arrays, patient role, masks and affines remain fixed; no new data are acquired.

## Independent controls and reproduction

A bounded independent numerical reviewer checked 36 dense/contact/branch
problems through 432 steps. Maximum displacement disagreement was 6.83e-14,
free-equation residual 5.68e-13 and energy-balance residual 2.84e-14 in the
generated controls' declared units. A heterogeneous oblique source coupon
exercised nonzero distributed history, changed targets, an actual split plane,
removal, release, reapplication and JSON rebuild. Its maximum full-state
disagreement was 1.07e-14; removed stored energy agreed with separately
integrated Q1 stress/strain energy to 2.59e-25 Nmm.

Additional independent controls cover support-cache changes, loaded-code
fingerprints, observer noninterference, canonical hashing, aliased/cyclic
copies, cut/partial history ownership, tensor layout and malformed declarations.
These generated controls are numerical verification, not measured tissue data.

Run from the primary checkout with a fresh output directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/qualify_continuum_history.py \
  --protocol manifests/experiments/continuum-history-qualification-v2.json \
  testcases/outputs/simulator-qualification-20261029/continuum-history-v1/NEW-RUN
```

Ignored local `testcases/outputs/simulator-qualification-20261029/continuum-history-v1/`
retains all three attempts, source/launch hashes, raw timing samples, independent
checker scripts and scored validation. The next
[fixed-domain exterior preparation](continuum-exterior-coupling.md) begins the
boundary-mechanics work; it does not promote the failed radius gates.
Physiological RL remains on hold:
boundary adequacy, measured transient/device response, accepted acquired shift
prediction and patient vascular/functional evidence remain open.
