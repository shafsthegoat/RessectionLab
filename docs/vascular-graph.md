# Declared vascular graph: numerical qualification

`vascularGraphV1` opts a V2 scene into a separate RC network. Existing scenes and
recordings retain their original wound model. This extension can distinguish
intact occlusion from hemorrhage, arterial delivery loss from venous drainage
loss, and wound sealing from reconnection. It is not a patient perfusion model.

The [frozen protocol](../manifests/experiments/vascular-graph-v1.json) records
parameters, evidence scope, tests and open acceptance gates. The acquired tumor
development task remains UCSF-PDGM-0007. It has no supplied vascular imaging or
vessel labels, so patient-specific vascular evaluation remains unsupported.
The separate local TOPBRAIN CTA has partial arterial and venous annotations;
it has not been merged with the glioma. Registered centerlines/connectivity,
coverage review and measured boundaries remain open work on acquired cases.
The separate [acquired-source geometry workflow](native-vascular-geometry.md)
now checks CTA001 masks, source IDs, class semantics and connectivity; its
representative subgraph remains unadmitted for physiological simulation.

## Model and action consequences

Each node has explicit compliance `C` in mm³/Pa and pressure-supporting storage
`V` in mm³, with gauge pressure `P=V/C`. Zero unstressed volume and zero operative
field pressure are assumptions. Storage is not measured anatomical blood volume.
Declared fixed boundary pressures must be nonnegative relative to that field.
Internal flow is signed `G(Pa-Pb)`; artery/vein names do not enforce a direction.
Boundary inflow and outflow are counted separately, including reversals.

Physical artery, vein, sinus or uncertain edges bind explicit original vessel
source cells and one oriented six-neighbor source face. The endpoints of that
face identify the two wound outlets. These bindings do not estimate lumen area,
conductance, compliance or territory. Abstract assumed links have no acquired
geometry. A cut outside the declared graph is refused; an encounter outside its
coverage is recorded as unknown. An additional cut inside a lumped segment
requires refining its source-face graph first.

Closed bipolar jaws use the existing working capsules and electrode contact
convention. Applied force divided by a declared full-compression force reduces
conductance reversibly, including a contacted wound port. This is an uncalibrated
force proxy, not wall mechanics. Contact is present with the energy off. Opening
or withdrawing restores only the reversible compression factor. Energy-induced
occlusion persists after withdrawal and can occur without a wound. Jaw closing
time is explicit: energy applied while opening/closing is not a full interval of
closed-electrode contact.

Transection disconnects the edge and opens two separately supplied ports. A
distal stump can bleed from its own storage and surviving connections. Coagulation
seals only contacted stump cells, never reconnects the cut. Earlier intact
occlusion does not automatically seal a subsequently created stump. A declared
exponential natural-clot rate reduces port conductance; none of these injury,
compression, coagulation or clot coefficients is fitted to a device or patient.
Other force contact is an encounter, without an invented forcedamage law.

The solver integrates node storage, boundary in/out, per-port wound volumes and
regional flow deficits together. An exactly-once pending token reserves each
wound interval. Source ID, model, kind, cell and finite per-port cumulative
ledgers are checked before any field mutation. The ordinary field emits that
volume, routes fluid, and retains blocked outlets outside vascular storage.
Suction and irrigation retain the existing mixed-fluid budgets. No second
legacy vascular reservoir participates. Blockage does not create tamponade:
there is no field-to-vessel pressure feedback.

Regions require explicit signed delivery and drainage interfaces, positive
reference flows and storage nodes. Deficit integrates
`max(0,Qref-max(0,Qsigned))` in mm³; dividing by `Qref` gives deficit seconds.
Upstream storage change is a separate congestion signal. These quantities do not
measure oxygen delivery, imaging ischemia, infarction, paralysis or aphasia.
Indirect compression from tissue stress, vessel collapse, autoregulation,
pulsatility, heart-height adjustments and patient ICP are not modeled.

## Evidence and independent checks

[Olufsen et al.](https://pubmed.ncbi.nlm.nih.gov/11792673/) used a two-resistor,
one-capacitor model with measured finger pressure and MCA Doppler responses in ten
healthy subjects during posture change. It supports the RC abstraction, not
local surgical wound calibration. [Nguyen et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3210348/)
measured flow reductions, reversals and collateral-dependent responses after
targeted rat venular occlusion. Their injury-producing induction is not a clean
occlusion blood-loss experiment. [Blinder et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4141079/)
combined reconstructed mouse vascular topology with declared pressure boundaries
and resistances; their results and modeling limitations support keeping collateral
and boundary uncertainty explicit. No numeric coefficient is transferred from
these papers, and no quantitative agreement with their observations is claimed.

The controls in `tests/test_surgical_vascular_graph.py` include independent
closed-form RC storage and wound integrals, two-node exchange, two-stump drainage,
boundary reversal within one interval, and natural clotting. Arterial and venous
occlusion, reversed collateral flow, field handoff failures, source substitution,
unknown cuts, affine invariance, real bipolar contact and replay are tested
separately. LSODA solves the coupled system at tight tolerances. Failed solves,
meaningfully negative storage and mass residuals refuse the transition.
Sub-tolerance negative storage roundoff alone is set to zero and counted in
`roundoffCorrectionMm3`; no other ledger is repaired to force conservation.

The generated app case contains three isolated source-bound arterial, venous and
sinus segments, a declared RC chain and a superficial numerical target. It is a
post-exposure engineering coupon; no anatomical exposure is claimed. Four plans
preserve flow, occlude an artery, occlude a vein, or transect and seal an artery.
All use identical CUSA revision2 actions at the unchanged assumed 2 mm³/s rate,
remove the same nine original target cells (9 mm³), and remove no non-target
cells. All end with withdrawn tools at a common 30-second modeled horizon.

At every accepted substep, a separate matrix-exponential system checks storage,
wound and boundary integrals. An independent eight-point Gaussian quadrature
checks cumulative regional flow deficits. Ordinary full-state replay and source
accounting are additional checks. These establish numerical Level A behavior;
they are not Level B specimen/device or Level C acquired anatomy validation.
Exact local results and failed/intermediate runs stay under `testcases/`.
Authenticated replay caches bind loaded continuum and vascular graph code as
well as disk sources; runtime replacement disables cache admission.

```sh
.venv/bin/python -m pytest tests/test_surgical_vascular_graph.py -q
.venv/bin/python scripts/qualify_vascular_graph.py \
  testcases/outputs/simulator-qualification-20261029/vascular-v1/new-run
pnpm --dir desktop build
VASCULAR_CASE="$PWD/testcases/outputs/simulator-qualification-20261029/vascular-v1/new-run/vascular-rehearsal.ressectionlab" \
VASCULAR_OUTPUT="$PWD/testcases/outputs/simulator-qualification-20261029/vascular-v1/new-hidden-check" \
desktop/node_modules/.bin/electron desktop/electron/verify-vascular-rehearsal.cjs
```

The native check stays hidden, checks matching GPU/HUD/state identities, fixes
microscope optics across all four branches, and saves the workspace. A separate
fresh-process reopen can use `VASCULAR_REOPEN_EXPECT` pointing to its report,
`VASCULAR_REPORT` pointing to the generated qualification report and
`VASCULAR_CASE` pointing to its saved workspace. Temporary browser profiles are
disposable after the process exits; evidence, source cases and reservations are
preserved.

The viewer exposes graph assumptions and separate storage, flow-deficit and
cut/occlusion/compression summaries. Outcome export and telemetry use graph terms.
The existing RL reward explicitly refuses graph scenes because its old harm
vector omits supply and drainage consequences. Measured vascular response,
acquired connectivity, neurological validation and meaningful vascular RL remain
on hold.
