# Contact and topology history before larger-domain program replay

The [fixed-domain exterior witness](continuum-exterior-coupling.md) passes, but
does not admit changing topology or larger-domain instrument actions. This next
stage preserves the original three programs and develops explicit source-owned
transfer and model-change energy accounting before another full source solve.
No live engine property, historical recording or physical threshold changes.

## Resource and input admission

The previously qualified native receipts contain 1,938, 2,154 and 126
post-prefix fixed steps for short hold, sustained hold and checked STOP. The
fixed-domain verifier's approximately 0.256 seconds per step would imply about
1,080 seconds for those 4,218 steps before topology rebuilds, detailed controls
or independent verification. This estimate does not admit a 900-second full
program run. It motivates bounded capture and isolated kernel/rebind profiling.

`capture_continuum_programs.py` freezes a separate
[native input capture](../manifests/experiments/continuum-program-capture-v1.json).
It reuses the qualified 8 mm archive and exact action receipts, and requires
every public outcome and final snapshot to remain bit-exact. The observer
stages continuum inputs until the complete physical microstep succeeds;
candidate clones own independent observer append lists. Post-physics failures,
rejected actions and cancellation cannot publish failed microsteps. Contact
source IDs, stiffness/targets, physical dt, fixed-clock remainders and the whole
source removal/cut/rigid state remain inspectable. Source IDs are not replaced
by local node numbers.

All 15 prefix actions and their costs remain in the capture. The frozen 20 mm
mesh starts **after** that exposure and excludes 5,500 already removed owners.
Its activation requires demonstrated zero displacement/Maxwell fields and the
explicit clock remainder, rather than assigning fake zero removed energy to
unmodeled prefix tissue. The continuation must count all eight additional
target owners: seven in the old core and one in the exterior. No source event
outside the frozen reference may silently disappear. Native active/anchor lists
are control metadata, not substitutes for the larger domain's supports.

The capture runs within 300 seconds, 2 GiB sampled group RSS and 32 MiB outputs,
with one numerical thread and no retry. It captures qualified source-world
inputs only; prescribing those events later is not larger-domain public-action
or collision acceptance.

## Exact advancement and contact rebinding

Fixed tissue factors can be shared across contact bindings. A full contact
compliance basis is recovered once per binding, and an exact full equilibrium
body-force field is cached only while every load value remains identical. A
changed body array, including in-place edits, invalidates that derived cache.
The step uses

```text
base = (K0⁻¹ bodyForce + Σ αj rj zj_old) / β
(β I + D C K0⁻¹ Cᵀ) q = D(target − C base)
u = base + K0⁻¹ Cᵀ q
```

`zj_old` remains the complete distributed source field. It is not projected or
fitted to the contact basis. Exterior/core support admission, full residual and
actual spring/tissue/Maxwell work-energy checks remain unchanged. Contact
rebinding reuses the same admitted Schur/core factor but rebuilds its contact
basis; a changed tissue operator cannot use another operator's prepared factor.

## Instantaneous source-history events

`SourceHistoryLedger.transfer` takes full corner fields before re-equilibration.
Each new node selects the lowest available source-cell/corner owner. Every
other retained owner must agree **bit-exactly**, including signed zero, for
displacement and every Maxwell memory. Splits inherit their parent's fields;
unequal joins refuse. There is no averaging, history reset or new unloaded
source activation. Retained material coefficients and source grid must match.

New constraints require already zero displacement **and** memory at those DOFs;
otherwise the event refuses without an explicit constraint-work/transfer law.
Removed source-cell stored energy is integrated once, by source ID. Duplicate
exports and resurrection of an already exported owner refuse transactionally.
A cut/split preserving every source corner has no instantaneous tissue-energy
jump. Removed tissue and changes to a contact binding use separate ledgers:

```text
ΔΨ_event = −E_removed_tissue + J_contact
ΔΨ_total + E_removed_tissue − J_contact + D_viscous + D_numerical = W_BE
```

`J_contact` is the new minus old actual spring potential at the transferred
state; it is signed model energy, not removed tissue, viscous loss or measured
device work. A fixed-binding backward-Euler step starts from the transferred,
possibly unequilibrated fields. No instantaneous uncharged equilibrium solve is
inserted between the event and that step.

The source-operator identity binds complete source ownership, affine, corner
connectivity, supports, materials/spectrum, outside-domain removals/cuts and the
**ordered core free-DOF map**. A key for the full stiffness alone cannot identify
its Schur partition. This identity does not authorize reuse of a changed
exterior factor; that needs a separately verified update.

The frozen native capture at `8eab7f14` completed all three programs in
137.56 worker seconds (138.19 supervised), with 461,815,808 bytes sampled peak
group RSS. Every action receipt and full final snapshot is exact. The resulting
1,938 / 2,154 / 126 suffix steps preserve the seven core and one exterior
additional removals, prefix source costs and checked STOP. This passes input
capture only; larger-domain public actions remain unadmitted. Local receipts,
input hashes and commands are retained under
`testcases/outputs/simulator-qualification-20261029/continuum-exterior-events-v1/`.

The [isolated source step profile](../manifests/experiments/continuum-step-profile-v1.json)
compares the frozen `2e2de647` step with the revised step on the exact same
admitted operators and one previously qualified full source state. It measures
setup once, alternates eight repeated calls per kernel, requires byte-exact
within-kernel repetition and unchanged field/force/energy tolerances, and
profiles both calls. Contact release and rebinding must share the tissue
preparation. This is a separate 180-second / 3 GiB / 8 MiB experiment with no
retry. Its kernel extrapolation excludes topology, field guards, controls and
independent checking, so cannot admit a complete-program run by itself.

That profile passed at `8173a7e8` in 48.93 worker seconds (49.27 supervised),
with 1,626,750,976 bytes sampled peak RSS. Median fixed-state step cost fell
from 138.28 to 7.13 ms (19.40 times faster); maximum displacement / memory
disagreement was 2.78e-17 / 3.47e-18 mm, reaction 4.07e-20 N and energy
6.78e-21 N mm. Both kernels repeat byte-exactly. Release binding cost 0.78 ms;
holding-contact rebinding cost 396.71 ms and reused the same preparation.
The kernel-only 4,218-step estimate is now 30.06 seconds, excluding every
unmeasured cost listed above. Setup remains dominated by 40.36 seconds of
Schur columns, plus 4.71 seconds exterior factor and 1.07 seconds core factor.

A separate source-sized field-check profile measured 52 of 58 ms in generic
tensor contraction. Selecting NumPy's optimized contraction preserves all
16 gradient guard points, eight integration points and Maxwell stresses;
the source gradient tensor was bit-exact in the discovery check. Oblique,
non-affine controls verify the full corner bound and 0/1/2 memory branches
against direct element/point sums. The selected guard/event/capture/exterior
regression passes 38 tests. That discovery alone does not admit a complete
program; the following comparisons supply the next evidence.

The frozen guard comparison at `8350c3fb` passes all six source worlds (four
distinct payloads): strain/stress rates and center bounds match exactly;
maximum gradient-bound difference is 6.94e-18. Median cost falls from 54.72 to
9.19 ms, a 5.95 times improvement, with byte-exact within-kernel repetition.
Its 30-second/1 GiB supervisor completes in 2.34 seconds. Combining the isolated
step and guard medians suggests 68.83 seconds for 4,218 calls, **excluding**
setup, topology, transfer, controls, replay and independent verification.

## Source-owned geometry and exact block admission

Independent capture/geometry inspection finds nine tissue masks, eight
single-owner removals, one contact release and no cut/rigid/support changes.
The fourth removal, source owner `3441217`, contributes to 12 free exterior
DOFs. Recomputing I from the surviving core elements would promote four old I
nodes into E, growing E from 27,864 to 27,876 DOFs even for core-only events.
Such a partition change cannot silently retain an exterior factor.

The offline `SourceDomain` instead declares original I lineage explicitly.
Every descendant retains that algebraic membership even when its last core
element is removed. The existing source mesher rebuilds uncut-face sectors;
original source corners supply supports and coefficients. It validates the
complete baseline cut graph, permits splits and further cuts, refuses implicit
joins, and owns no episode fields. The mesher's export bound remains separate
from the unchanged live 24k-DOF limit. Eleven generated controls plus the
selected exterior/partition/transfer/capture suite pass 56 tests.

The [bounded topology identity audit](../manifests/experiments/continuum-program-topology-v1.json)
assembles all nine frozen masks within 45 seconds/1 GiB/8 MiB, with no solve or
factorization. It compares complete canonical CSR patterns and coefficient
bits under source-free-DOF lineage. Exterior source removal requires a new
block group. Core-only events qualify as reuse candidates only if both KEE
and KIE identities are exact; even a one-bit assembly-order discrepancy is
retained as a failed identity. Numerical factor/Schur updates and source
history/energy qualification are still separate required gates.

The version-1 audit at `c693edec` completes all nine states in 6.94 worker
seconds (7.25 supervised), with 699,744,256 bytes sampled peak group RSS. It
correctly **fails** exact reuse: core-only deletions shift the retained-row
256-element buckets, changing up to 726 exterior coefficients by at most
8.67e-19 N/mm. I/E ties also change inside COO-to-CSR duplicate reduction.
The actual exterior owner changes KEE by up to 7.48e-4 N/mm. Geometry, supports
and ordered lineage match independent source-face graph reconstruction in all
nine worlds; E stays at 27,864 DOFs, and four orphan I nodes disappear.

The [version-2 audit](../manifests/experiments/continuum-program-topology-v2.json)
targets that measured assembly issue. Its optional assembly order pins buckets
to the original full owner list and explicitly reduces row/column/owner ties
before CSR construction. The earlier default path remains unchanged. Generated
deletion and input-row permutation controls require exact untouched KEE/KIE
coefficients; the oblique heterogeneous coupon also matches independent Q1
assembly at its existing tolerance. The source audit additionally checks the
previously qualified full source state against unchanged force/energy limits
of 1e-10 N / 1e-10 N mm. The reuse criterion remains bit identity; no failed
identity is replaced by a tolerance. No factor or Schur reuse is yet admitted.

Version 2 at `5b6f820a` passes all nine masks in 13.03 worker seconds (13.36
supervised), with 845,742,080 bytes sampled peak RSS. All seven core-only
comparisons have bit-exact KEE and KIE patterns/coefficients. The exterior
removal changes 144 KEE coefficients by up to 7.48e-4 N/mm and correctly starts
the second block group. Canonical baseline assembly differs from the older
summation order by at most 8.67e-19 N/mm; the previously qualified source
world's full free-force residual is 5.27e-19 N, and its energy-accounting
disagreement is zero. The 57 selected tests pass, and independent source-face
graph verification matches all nine exported geometries, materials, supports
and ordered free-DOF lineages.

A separate full-source parity check confirms that the default assembly remains
bit-exact to `2e2de647`: all 2,914,565 coefficients and the CSR pattern of the
49,260-total-DOF matrix match. The live engine, continuum, history solver,
deformation and CUSA source files also remain identical to the accepted
`0d66dd7e` qualification. The optional canonical path changes summation order,
with its numerical source checks stated above; it has no live admission.

The retained-operator/Schur gate requires an exact update with
explicit bijective exterior mapping, complete coefficient checks, a fresh core
support certificate, and generated full-field/energy controls. It must reject
one-bit coefficient changes, non-bijective exterior sector maps, and an
unsupported new core. The implementation and bounded two-block numerical pilot
below cover these checks before freezing all 4,218 source program steps.
Neither a future cut nor a split constrained sector may inherit a factor or
physical support claim from this no-cut source audit. The 16-to-20 mm stress
stability failure, larger-domain public actions and physical/physiological RL
evidence gates remain open.

The focused event/capture/continuum/clone/cancellation selection passes 119
tests. Independent code checks 45 bindings and 180 dense-control steps,
including arbitrary exterior memory, contactless/rebound states and changing
body loads. Maximum displacement disagreement is 4.08e-15; work-energy balance
residual is 1.67e-15 in the generated controls' declared units. Independent
per-element Q1 energy agrees within 2.12e-21 N mm. These are numerical controls,
not source-program or physical acceptance. Topology/history and larger-domain
public actions remain unqualified. Physiological RL remains on hold.

## Retained factor and core support update

`ExteriorCoreFactor.rebuild` accepts a new supported-free stiffness, explicit
core order and injective new-free-to-old-free source lineage. Exterior lineage
must cover the old exterior bijectively. Complete canonical CSR coefficient
bits must match for KEE and both directed couplings KIE/KEI under that map.
The exterior solve composes permutations back to the original factor's order.
One-bit changes, missing exterior DOFs, split/merged lineage or changed coupling
refuse this fast path; ordinary full construction remains available.

The new Schur matrix subtracts the mapped **directly measured** exterior
correction `KIE KEE^-1 KEI` from the current KII. The correction is immutable
and shared; each new core owns its Schur matrix, Cholesky factor and empty
body-load cache. Only 64 correction rows are indexed at once, and weak parent
references let obsolete operators/Schurs/factors be collected. The unchanged
Schur symmetry and positive-definite core certificate run on every rebuild.
Reconstructing the correction as `KIIold - Sold` was rejected before commit:
the independent `KIIold=1e16, correction=1` control loses that correction to
roundoff and can falsely admit an unsupported new core. Direct retention
preserves it exactly. The dense source correction adds about 241 MB and must
be included in source memory admission.

Seventy selected tests pass, including generated two-block removal/release
exports checked by independent quadrature and a separate split-sector refusal.
Independent dense controls cover 17 updates and 204 steps, chained I/E
permutations, arbitrary exterior memory/body forces, contactless advancement
and actual spring/Maxwell energy. Maximum displacement disagreement is
7.04e-16 and energy-balance residual 3.05e-17 in the generated controls' units.
One ill-conditioned control correctly refuses the unchanged symmetry bound.

The [retained source pilot](../manifests/experiments/continuum-retained-pilot-v1.json)
starts from the previously qualified nonzero step-30 source state. It traverses
the nine frozen tissue masks at one physical step per mask, then releases and
relaxes for two more steps. It declares two full exterior builds, seven retained
core updates and eight removal-energy transfers. A changed exterior must refuse
retention, and the old operator must be released before the second full build.
Current-operator Schur probes, source gradients/reactions and cumulative
removal/contact/BE energy stay within the existing budgets. Every full field
is exported for source-owned continuity, force, quadrature energy and stress
checks using the frozen independent checker. Worker and checker share 180
seconds and 32 MiB outputs, each with 3 GiB sampled process-group RSS and one
numerical thread. No run retry or full-program/physical acceptance is implied.

The frozen pilot at `e75599e8` passes in 131.07 worker seconds plus 5.33
independent-supervisor seconds (136.75 combined). Sampled peak process-group
RSS is 2,329,575,424 bytes; the worker's OS-reported peak is 2,383,200,256
bytes. Outputs total 11,945,241 bytes. Both worker/checker exit 0 with no kill,
approval issue or retry. The first/second complete tissue-state builds take
50.60 / 52.22 seconds; seven retained state updates take 3.33–4.67 seconds
(median 4.02), including source assembly, transfer, contact basis, checks and
field export. Retained Schur construction alone takes 0.48–0.80 seconds instead
of the measured 42.99 / 43.98 seconds of fresh Schur columns. The final root
correction occupies 241,120,800 bytes.

All 11 full source fields pass independent Q1 equilibrium, spring/Maxwell
energy, stress/probe, support and source-corner continuity checks. Maximum
independent free-force residual is 7.25e-19 N; step energy balance 1.05e-20
N mm and cumulative event/BE balance 7.25e-21 N mm. All eight removal exports
are positive and sum to 5.92448e-8 N mm in this pilot; release changes the
declared spring-model potential by -1.48817e-5 N mm. These values belong to
the prescribed step-30 pilot sequence, not the complete captured programs or
measured device work. The exterior event refuses retention and the previous
operator is collected before the new exterior build. Deterministic full
program replay is not covered by this pilot.

Input preflight verifies every suffix clock transition exactly for all three
captures, including the native rounding rule, with no zero-fixed-step ticks.
The post-prefix contact targets are initially zero. All three monotone stage
sequences map into the same nine tissue states; short/sustained/STOP preserve
1,938 / 2,154 / 126 fixed steps. A full harness can therefore interleave three
**independent episode fields** by tissue stage while sharing immutable
operators, then repeat from fresh inputs for deterministic replay. Two full
exterior builds and seven retained core updates suffice per pass; keeping
every historical dense factor resident is unnecessary.

Using the measured 131.07-second two-block pilot plus the earlier 68.83-second
kernel/guard estimate per pass gives about 399.8 seconds for two passes,
excluding orchestration and the independent checker. Twenty-six proposed
sample worlds with 24 expected byte-distinct field payloads would occupy about
26.0 MB at the largest measured pilot payload size, before metric/provenance
overhead. These are admission estimates, not a completed experiment. The next
executable milestone is the tested full captured-input harness, per-branch
removal/contact/clock accounting, exact replay hashes and a frozen 900-second
protocol. These were preflight estimates; the frozen run is reported below.
The physical boundary-stability,
larger-domain public-action and physiological RL holds remain unchanged.

## Complete captured-input qualification protocol

The [full program protocol](../manifests/experiments/continuum-program-qualification-v1.json)
freezes the complete 1,938 / 2,154 / 126 fixed-step schedules. Each branch has
independent full displacement/Maxwell fields, modeled spring targets, source
removal and work/energy ledgers. Tissue-stage scheduling shares only immutable
operators/contact bases. STOP ends at stage 0 and is never transferred through
the later removals. The second pass reconstructs the operators and episodes
from fresh inputs; every full-field/scalar record and final branch chain must
replay exactly. Numerical/material/gradient/reaction limits remain unchanged.

The last prefix hold supplies the initial seven zero-target springs. A moving
target contributes actuator work on the first BE step of its captured tick;
later steps use that target as the previous target. Contact removal has a
separate signed potential jump. The captured programs contain no simultaneous
topology/target or topology/binding changes; the parser refuses to invent an
event-work policy for such a case. Native fixed-clock counts, remainders and
elapsed times are checked against the frozen records. The source case's native
mapping identifies target label 2; the full reference source labels separately
verify the eight additional target owners and unchanged non-target cost.

All 4,218 records retain residual, actual spring/tissue/Maxwell energy,
dissipation, probe rates, cumulative exposure, source-event and full-array
hashes. Twenty-six field sample worlds cover initial actuation, stage endpoints,
every source event, release and final relaxation. Samples store the vanished
old-node values alongside the current before-state. Independent source-corner
mapping then reconstructs the complete pre-event field and checks its prior
step hash, preserving nonzero orphan-node memory without exporting a second
full state. No retained source history is projected onto a contact basis.

Independent checking derives the exact unique sample/event set from the native
inputs, validates physical h and scope/RL flags, folds all scalar/clock/source
ledgers, checks final summaries, and computes full Q1 force/energy/stress for
every sampled world. Generated controls exercise both fresh replay passes,
reversed branch order, nonzero vanished-node recovery and checker refusals for
checksummed one-ULP orphan corruption, duplicate samples, negative h, missing
events, altered final energy and promoted RL flags. The selected suite passes
75 tests. The pure independent quadrature functions are now tracked code;
generated tests no longer depend on an ignored source-case checker file.

Worker and checker share one 900-second / 32 MiB budget, each with 3 GiB sampled
group RSS and one numerical thread. A one-second finalization reserve is
charged before accepting the combined wall time. Source edits, changed inputs,
failed replay or limits preserve an incomplete receipt and partial metrics;
the launcher never retries or overwrites a previous output path. This protocol
qualifies prescribed inputs on the larger mesh; actual larger-domain public
actions/collisions, outer-radius stability and physiological validation remain
separate gates.

## Complete source result

The frozen protocol at `a6b32ddb` passes in **449.72 seconds combined**:
434.30 worker seconds (434.82 supervised) and 14.90 independent-supervisor
seconds. The two fresh passes take 215.81 / 217.75 seconds. Sampled group RSS
peaks at 2,466,758,656 bytes; OS-reported worker peak is 2,469,756,928 bytes.
All outputs total 26,169,468 bytes. Worker and checker exit 0, with no kill,
approval issue, retry, changed threshold or changed source input.

All 4,218 complete supported-free fields and scalar records replay exactly
through source removal, target motion, release and relaxation. Twenty-six
sample worlds use 24 byte-distinct payloads. Independent Q1 checks pass every
sample, including lossless pre-event fields and all 16 removal events; the
checker also folds every one of the 4,218 scalar/clock records. It performs no
source-sized mechanics solve and does not claim independent full-field checks
at every step. Maximum sampled free-force residual is 6.61e-19 N and sampled
step energy balance is 4.74e-20 N mm. Across all worker steps, maximum residual
is 1.07e-18 N and cumulative event/BE balance is 1.29e-19 N mm.

| Prescribed program | Fixed steps | Suffix duration | Additional target owners | Final target removed / residual (mm³) | Fixed-probe stress² integral (Pa² mm³ s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short hold | 1,938 | 16.15 s | 8 | 568 / 18,800 | 763,054.95 |
| Sustained hold | 2,154 | 17.95 s | 8 | 568 / 18,800 | 861,310.08 |
| Checked STOP | 126 | 1.05 s | 0 | 560 / 18,808 | 0 |

Both resection histories preserve the same seven core plus one exterior source
owner removal. Removed stored energy is 6.11443e-8 / 6.11467e-8 N mm for
short/sustained hold, respectively. Each releases the held spring model and
relaxes with nonzero retained memory. STOP releases the zero-target initial
hold, remains at the baseline geometry and has zero subsequent modeled
mechanical exposure. Every branch still includes **4,940 mm³ non-target access
removal** in its preserved prefix; zero later mechanics does not erase that
cost or establish absence of injury.

Maximum full-program gradient norm is 0.066588, center displacement 0.075834 mm
and handle reaction 0.78539 mN, within the unchanged declared model limits.
The physical interpretation of that stress integral and the boundary/material
law remains unvalidated. These programs prescribe the already qualified
native 8 mm inputs to the larger world; they do not reevaluate its complete
public action/collision responses. Full dimensions remain 33,360 initial free
and 49,260 total DOFs, exceeding the unchanged live 24k cap.

The original revision-2 native detailed/history qualification remains intact;
the live engine, continuum, history-solve, deformation and CUSA files are
unchanged from `0d66dd7e`. Numerical progression now supports a separately
bounded larger-domain public-action/collision feasibility test. That next test
must account for every exterior DOF and preserve model refusals before any live
resource admission. The original 16-to-20 mm stress-stability failures,
mixed/negative donor-force evidence, patient action-response, source-supported
vascular/neural physiological evidence and both RL holds remain open. The
scientific goal is not complete.
