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

The focused event/capture/continuum/clone/cancellation selection passes 119
tests. Independent code checks 45 bindings and 180 dense-control steps,
including arbitrary exterior memory, contactless/rebound states and changing
body loads. Maximum displacement disagreement is 4.08e-15; work-energy balance
residual is 1.67e-15 in the generated controls' declared units. Independent
per-element Q1 energy agrees within 2.12e-21 N mm. These are numerical controls,
not source-program or physical acceptance. Measured source capture and kernel
costs are the next admission evidence. Physiological RL remains on hold.
