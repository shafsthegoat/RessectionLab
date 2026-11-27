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
Search and the opt-in generated-graph reward integrate the largest declared
supply and drainage deficits separately over simulated time. A generated
arterial occlusion produced 2.319 supply-deficit seconds and 0.244
drainage-deficit seconds with zero bleeding. Venous occlusion reversed these
measurements. The default RL configuration still refuses graph scenes. The
opt-in reward requires generated coverage and declared regions. Acquired
connectivity, measured physiology and neurological outcome calibration remain open.

Partial vascular labels leave unsampled vessels unknown. Compare topology on
the original source across cropped boundaries. clDice describes a
centreline-sensitive segmentation metric. The stereotactic trajectory study
motivates retaining acquisition uncertainty. Its distances apply to its own
procedure. Postoperative diffusion imaging motivates tracking ischemic effects
separately from bleeding. [clDice](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html)
· [Vascular segmentation and SEEG trajectories](https://discovery.ucl.ac.uk/id/eprint/10082723/)
· [DWI and acquired deficits after glioma resection](https://pubmed.ncbi.nlm.nih.gov/24992634/).

## Cut-vessel bleeding (`vesselBleedingV1`)

The opt-in scene property `vesselBleedingV1`
(`src/resectionlab/surgical_vessel_bleeding.py`) shows where bleeding comes
from on a scene whose vessels came from imaging. Scenes without it keep their
exact state and state identity. It requires `tissuePhysicsV2`, which keeps the
Python reference engine. When present it supersedes the single compliant V2
compartment and zeroes graph wound ports on its segments, so blood is not
emitted twice.

`declare(labels, regions, affine)` makes one segment per 26-connected
component of each vessel region, with run-length cell lists. A segment's kind
comes from the region's `vesselKind` or its name: artery, vein, sinus, else
uncertain. Angiography-only masks (`declare_mask`, or
`qualifying_properties(..., vessel_bleeding=True)` for a native wall) stay
uncertain. An uncertain wound is a bracketed estimate: it bleeds at the
arterial law (the conservative magnitude, since angiography mostly shows
arteries), records `kindStatus: 'UNKNOWN'` and keeps the venous-law value as
`venousLawLowerBoundMm3PerSecond` and `venousLawLowerBoundLossMm3`. Calibre is the local thickness at the wound (Hildebrand and
Rüegsegger 1997, J Microsc 185:67-75): the largest inscribed sphere of the
segment containing the cell, as `2R - s` with `s` the voxel spacing. The
estimate is floored at `s` and labelled `grid-floor`. A 1 mm grid cannot
resolve a sub-millimetre vessel, and the floor overstates its flow.

A wound opens when a segment cell is removed or a face between two segment
cells is transected, cut or torn by contact. Its opening fraction is the breach
area over the lumen area `π d²/4`, capped at 1. Arterial flow is
`Q = 717 mL/min × 0.21 × (d / 2.8 mm)³ × opening` (mL/min × 1000/60 = mm³/s),
modulated by `1 + (PI/2) sin(2π f t)` and integrated exactly over each substep.
A 1 mm artery bleeds 114 mm³/s on average. Murray scaling is not extrapolated
past the reference vessel: the calibre used for flow is capped at 2.8 mm (the
wound is marked `calibre: 'above-reference'`), the arterial wounds of one
segment together are capped at the reference flow (0.21 × 717 mL/min), and all
concurrent vessel bleeding at total cerebral blood flow (717 mL/min); a wound
scaled down records `flowCapFactor`. Veins and sinuses well steadily at that
flow times the venous-to-arterial pressure ratio (10/90). The site-minus-heart
hydrostatic pressure of a declared `patientPosition` adds to the 10 mmHg venous
pressure, floored at zero. A site above the heart therefore gives zero venous
flow; the open venous wound and the metrics then carry
`venousAirEntryRiskUnmodelled: true`, because a dry field there is not a safe
one and air entry is not modelled.
Exposed tumour faces ooze `0.004 mm³ s⁻¹ mm⁻²`, decaying with a 1800 s time
constant. For a 14 cm³ spherical cavity this gives about 40 mL/h on the smooth
2,800 mm² surface. The grid surface is about 1.5 times larger, giving about
60 mL/h, falling to about 26 mL/h averaged over the first hour.

Blood is deposited into the ordinary per-cell `fluid` inventory, next to the
wound or within two voxels of it. Suction, irrigation, overflow and gravity
pooling therefore act on it unchanged. A wound with no free outlet holds its
blood in a reservoir until one opens. `bloodLossMm3` equals the deposited
volume plus the held volume, and losses are kept per segment and per kind.
Under an active `brainShiftV1` scenario the wound, ooze outlet and dressing
coverage are taken at their current positions: each source cell is mapped by
`surgical_brain_shift.current_cells` to its shifted destination before the
outlet search, so blood appears where the cut vessel now lies.

The bipolar seals a wound only where the electrodes touch it, after
`2 × max(1, (d/1.5)²)` activation-seconds. A vessel no wider than 1.5 mm seals
fully. A wider vessel is capped at a seal fraction of `(1.5/d)²`. A sinus does
not seal. With at least a quarter-voxel of free fluid within one voxel of the
tip, the required dose rises 1.5 times and char sticking falls by 0.5 per
activation-second. Dry coagulation adds 0.25 per activation-second. When
sticking reaches 1, moving the tip more than 0.5 mm, opening the jaws or
withdrawing re-opens the wound, once. Ooze stops after 0.5 activation-seconds
on a cell or one of its neighbours, or under a placed haemostatic dressing,
which also reduces venous wound flow by 0.7.

Metrics appear as `metrics()['vesselPhysicsV2']['vesselBleedingV1']`: blood
loss total, by kind and by segment, tumour ooze, open wounds by kind, the
largest uncontrolled calibre, the current rate, the conservation residual,
`vesselKindUnknown` (emitted loss and rate of uncertain wounds with their
venous-law lower bounds and `bloodLossLowerBoundMm3`), `aboveReferenceCalibreWounds`
and `venousAirEntryRiskUnmodelled`. The physics descriptor lists every
parameter with its source, the segments above the reference calibre and the
two flow caps.

| Parameter | Value | Source |
| --- | --- | --- |
| Total cerebral blood flow | 717 mL/min | Zarrinkoob et al. 2015, J Cereb Blood Flow Metab 35:648-654 |
| MCA share of that flow | 0.21 | Zarrinkoob et al. 2015 |
| Flow proportional to d³ | exponent 3 | Murray 1926, Proc Natl Acad Sci USA 12:207-214 |
| Reference M1 diameter | 2.8 mm | assumed-research |
| Heart rate | 70 /min | assumed-research |
| Pulsatility index (Gosling-King definition) | 0.9 | assumed-research |
| Mean arterial pressure | 90 mmHg | assumed-research |
| Cortical venous pressure | 10 mmHg | assumed-research |
| Full bipolar seal limit | 1.5 mm | assumed-research |
| Seal dose | 2 activation-s | assumed-research |
| Irrigation cooling factor | 1.5 | assumed-research |
| Wet threshold | 0.25 voxel of fluid | assumed-research |
| Dry sticking / wet cleaning | 0.25 / 0.5 per activation-s | assumed-research |
| Pull-off distance | 0.5 mm | assumed-research |
| Tumour ooze rate | 0.004 mm³ s⁻¹ mm⁻² | assumed-research |
| Ooze decay time constant | 1800 s | assumed-research |
| Ooze seal dose | 0.5 activation-s | assumed-research |
| Dressing venous tamponade | 0.7 | assumed-research, matching the glioma-materials dressing |
| Outlet search / calibre window | 2 / 10 voxels | assumed-research numerical bounds |

Limits: the model has no ischaemic territory, so a cut vessel starves no
tissue. Calibres below the grid resolution are unknown. There is no
coagulopathy, systemic blood volume or haemorrhagic shock. Arterial flow is
the normal segment flow, capped as above, not an orifice law, and there is no
distal back-bleeding. Air entry at a venous site above the heart is flagged,
not modelled. Blood and irrigation share one inventory. Sealing uses delivered
activation, not tissue temperature. Only tumour, not brain, oozes.
`SurgicalEngine.metrics()` and `view()` do not yet expose the property at top
level.
