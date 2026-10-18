# Surgical research physics V2

V2 is an explicit, uncalibrated research surrogate for controlled simulator and RL experiments. It does not infer tissue properties, vessel caliber, perfusion, bone density, or patient prognosis from segmentation labels. Its parameters are declared assumptions, not measurements. Existing V1 scenes remain V1.

## Opt-in and patient identity

The complete profile lives at `scene.properties.tissuePhysicsV2`, with schema `resectionlab.tissue-physics.v2`, `provenance.kind = assumed-research`, and explicit `brain`, `bone`, and `vessel` sections. `surgical_physics_v2.default_profile()` supplies a baseline. Validation rejects simultaneous `tissuePhysics` and `tissuePhysicsV2` declarations.

The preparation script writes a derivative patient bundle and a preparation receipt. It compares MRI, affine, compartment arrays, source compartments, brain mask where present, source references, and material ownership labels; it also checks that the source bundle file hash remains unchanged. The derivative semantic identity changes because its simulation metadata changes.

```sh
PYTHONPATH=src .venv/bin/python scripts/prepare_patient_physics_v2.py \
  outputs/surgical-patient-sub06/patient.ressectionlab \
  outputs/surgical-patient-v2

PYTHONPATH=src .venv/bin/python scripts/verify_patient_physics_v2.py \
  outputs/surgical-patient-v2/patient.ressectionlab \
  outputs/surgical-patient-v2 \
  --source outputs/surgical-patient-sub06/patient.ressectionlab \
  --reference-directory outputs/surgical-patient-sub06
```

The verification script requires existing reference action receipts `verification.json`, `vessel-verification.json`, and `retraction-verification.json`. Their poses are candidates, rechecked through public engine transitions. Separate initially intact episodes exercise vessel injury/coagulation, drilling, and retraction/release on the same patient scene. Each action compares preview against commit; episode snapshots are serialized to JSON and restored by action replay. The script checks volume and work ledgers, source hashes, and unchanged anatomy. Generated receipts, rather than this document, establish whether a particular run passed. This is software verification, not clinical validation.

## Work-limited drilling

The bone model is `work-limited-drill-v2`. For force F in N, cutter radius r in mm, RPM n, assumed tangential-force fraction f, and duty d, the available power is

`P = d min(Pmax, F f (r/1000) (2π n/60))` in W.

Inputs are nonnegative; RPM, force, radius, and duty are capped at the declared model bounds. Zero force or zero RPM supplies no work. Contact cells share the single `P dt` budget equally, consistent with their equal physical voxel volumes. An additional contact does not create additional power. Each cell retains accumulated work across strokes and becomes removed only when its work reaches `specificEnergyJPerMm3 × voxelVolumeMm3`. Remaining budget after a cell saturates is recorded as unused rather than cutting inaccessible deeper material. Each removed cell releases exactly its full voxel volume as debris. Partial erosion remains occupied and generates no premature debris.

| Bone parameter | Baseline | Units / meaning |
| --- | ---: | --- |
| specificEnergyJPerMm3 | 0.5 | J/mm³ required for clearance |
| tangentialForceFraction | 0.1 | dimensionless force-to-torque assumption |
| maxRpm | 80000 | revolutions/min |
| maxForceN | 20 | N |
| maxPowerW | 30 | W |
| maxRadiusMm | 10 | mm |
| heatFraction | 0.35 | fraction of consumed cutting work converted to heat |
| heatCapacityJPerMm3K | 0.002 | J/(mm³ K) |
| coolingTimeSeconds | 8 | s |
| ambientTemperatureC | 37 | °C |
| doseThresholdC | 43 | °C |

For each heated cell, `C = heatCapacityJPerMm3K × voxelVolumeMm3` and

`d(T−Tamb)/dt = Q/C − (T−Tamb)/tau`.

Temperature uses the exact constant-power exponential update. Thermal dose is the exact integral of `max(T−doseThresholdC, 0)` in °C·s; it is separate from destructive cutting and is not a necrosis prediction. Heat is a partition of cutting work, not additional input energy. The ledgers satisfy `input = cutting + unused` and `heatGenerated = storedHeat + coolingLoss`, to floating-point tolerance. There is no spatial conduction or irrigation cooling model.

Before moving an active drill, `contact_limited_pose` searches for first bone contact and bisects its location. Existing contact holds the tool at that position while work accumulates. The engine commits dwell time and work; a request ending short of its intended endpoint reports `BONE_CUTTING_IN_PROGRESS`. Shaft, tool, and incompatible-material checks still apply. Removal is voxel-quantized, so intermediate work does not expose a fractional geometric cavity.

For simultaneous drills, the engine calls `begin_step`, accumulates each tool's power through `apply(..., cool=False)`, and calls `end_step` once with the shared interval. This integrates summed heat and idle cooling once. The transient accumulator is not serialized; persistent energy, temperature, and work are serialized. Standalone module use may call `apply` with its default cooling behavior, or `cool` for an idle interval.

## Coupled local deformation

The deformation surrogate uses continuous source-cell displacements u in mm and a bounded patch of neighboring soft tissue/tumor nodes. A vector spring Laplacian couples adjacent nodes. An additional discrete-divergence penalty approximately resists volume change, and a weak foundation represents unresolved surrounding tissue. Cuts remove lattice bonds; removed cells no longer participate.

With stiffness matrix K, held-node stiffness H, drag c, fixed step h, and target forcing f, the implicit update is

`(K + H + c/h I) u_next = c/h u_previous + f`.

Patch-interface nodes are anchored. Held targets are displacement-clamped; forcing is capped per node. A global displacement rescaling bounds neighboring displacement differences relative to bond length. Swept samples reject coupled increments that enter declared bone, vessel, unknown material, or leave the source grid. Release removes the target constraint and permits relaxation under the same dynamics.

| Brain parameter | Baseline | Units / meaning |
| --- | ---: | --- |
| radiusMm | 4 | mm local patch radius |
| maxNodes | 2000 | maximum active nodes |
| fixedStepSeconds | 1/120 | s; residual time is retained |
| brainStiffnessNPerMm | 0.0006 | N/mm |
| tumorStiffnessNPerMm | 0.0012 | N/mm |
| dragNsPerMm | 0.0003 | N·s/mm |
| holdStiffnessNPerMm | 0.02 | N/mm |
| bulkPenaltyRatio | 2 | dimensionless |
| maxDisplacementMm | 3 | mm per node |
| maxEdgeStrain | 0.65 | bound on neighbor displacement difference / rest length |
| maxForceN | 0.08 | N per-node target forcing cap |

This is a small-strain, local Kelvin–Voigt lattice surrogate, not an incompressible finite-element brain model. It omits gravity, CSF pressure, contact friction, tissue fracture, and patient material calibration. The strain bound is a numerical constraint, not an injury threshold.

## Vessel damage and finite fluid supply

The vessel model is `compliant-wounds-v2`. Unique neighboring vessel-cell faces carry physical area from affine basis-vector cross products, injury severity s, patency, clot fraction c, and coagulation dose. Adjacent damaged faces cluster into one local outlet while preserving their distinct area contributions and emitted/reservoir volumes.

Direct transection sets severity to one and permanently clears face patency. Contact accumulates mechanical dose `F dt / (footprintArea × mechanicalDoseNs)` and, when explicitly supplied to the module, thermal dose `max(T−43,0) dt / thermalDoseCs`. Severity rises after combined normalized dose exceeds one. `mechanicalDoseNs` is a historical field name: its actual threshold units are N·s/mm². The integrated engine supplies mechanical contact force; it does not currently pass drill temperature into vessel contact. Thus thermal vascular injury is a module capability, not a coupled drill-to-vessel effect.

The shared compartment has capacity Vmax, remaining volume V, and pressure `P = P0 V/Vmax`. A wound's conductance is

`G = leakConductanceMmPerPaS × Σ(areaMm2 × s² × (1−c))`.

The compartment obeys `dV/dt = Grefill (P0−P) − ΣG P`. Refill and generated fluid have separate conserved volume ledgers. Clot approaches one with a first-order rate; coagulation also increases persistent clot according to integrated declared power exposure. Sealing does not restore patency. The ODE is integrated with tight numerical tolerances; generated volume is handed to the shared fluid system as an interval-average source rate exactly once. A blocked outlet retains fluid in its reservoir.

| Vessel parameter | Baseline | Units / meaning |
| --- | ---: | --- |
| pressurePa | 1000 | Pa full-compartment pressure |
| reservoirMm3 | 100 | mm³ capacity |
| refillConductanceMm3PerPaS | 0.001 | mm³/(Pa·s) |
| leakConductanceMmPerPaS | 0.0001 | mm/(Pa·s), multiplied by face area |
| mechanicalDoseNs | 2 | N·s/mm² contact-dose threshold |
| thermalDoseCs | 200 | °C·s above 43 °C |
| coagulationDose | 2 | declared power-proxy × seconds |
| clotRatePerSecond | 0.02 | 1/s |

A single finite compartment is not an anatomical vascular network. Neighboring outlets may cluster despite unknown anatomical connectivity. Caliber, collateral circulation, pulsatility, territory perfusion, and patient blood loss remain unknown. Grid-face area is a staircase approximation.

## Shared grid, transitions, and RL use

Source cell identities persist through cutting, deformation, and removal. Continuous displacement is mapped through the inverse scene affine and rounded onto the common source grid. Collision occupancy keeps all source memberships when soft cells compress into the same raster cell; visible occupancy is their union with deterministic region precedence. View `moved` pairs use that same raster. This avoids separate collision and display geometry interpretations, but it does not preserve incompressible volume or resolve subvoxel surfaces.

Actions remain the public engine operations with a slot, optional pose/path, duration, and parameter dictionary. V2 adds `rpm`, `forceN`, and `duty`; pose motion and hold/release drive deformation, scissors create transections, and bipolar `power` supplies coagulation exposure. Public validation currently permits RPM 0–80000, force 0–20 N, and duty 0–1. Geometry and material checks can reject requests; a successful drilling dwell can return partial completion. Use committed elapsed time and actual slot pose, not requested duration or endpoint alone, when constructing transition records.

`observe()` exposes the declared profile, capability limits, slot state, and metrics. `view()` additionally exposes `drillingV2`, `vesselsV2`, continuous `deformation`, `deformationModel`, debris, source reservoirs, removal IDs, and raster movement pairs. Useful research observables include pending drill work, input/cutting/unused energy, generated heat, maximum temperature, removed volume, brain maximum displacement and elastic energy, neighbor displacement ratio, wound area, active wounds, pressure, vascular reserve, generated fluid, and conservation residuals. These are simulator observables, not measured patient outcomes.

Each microstep is checkpointed. An actual collision/error rolls back that microstep, retaining earlier completed microsteps. Work-limited bone dwell itself does not trigger that rollback. Preview operates on a cloned state; restore replays actions and verifies the resulting state identity and checkpoint. Fixed deformation residual time and all persistent physics ledgers are part of state. Solver tolerances and voxel approximations should be retained in any reproducibility claim.

The focused drilling tests exercise zero-input behavior, force/RPM response, persistent work, conservative contact power, exact debris volume, energy balance, deterministic caps, frontier stopping, thermal partition invariance, and simultaneous-source thermal integration. End-to-end patient receipts are required in addition to these module-level tests.

## Patient run on this Mac (18 October 2026)

The existing Zenodo 17486320 sub-06 derivative was used unchanged: 220 × 220 × 181 cells at 1 mm. Brain, meningioma, CT bone and vessel regions are the existing unreviewed segmentation estimates. The inherited CT registration uncertainty and 8.9% brain/dense-bone overlap remain visible. No anatomy or initial surgical exposure was added for these tests. This does not establish segmentation accuracy.

`outputs/surgical-patient-v2/verification.json` records passing public-action episodes, preview/commit identity checks, conserved ledgers, and fresh-process native save/reopen replay. Five vessel faces totaling 5 mm² clustered into one outlet; generated fluid stopped increasing after bipolar sealing, while severed patency stayed false. The drill removed 34 mm³ of bone, spent 17.28 J cutting work, and retained unfinished work. The uncalibrated maximum temperature proxy reached approximately 124 °C; this value illustrates the declared heat model and must not be treated as a tissue injury prediction. The retraction episode moved 36 nodes, with peak displacement 0.588 mm and nonheld-neighbor displacement 0.176 mm; after release and relaxation the maximum was 0.000496 mm.

Run the additional parameter checks with:

```sh
PYTHONPATH=src .venv/bin/python scripts/verify_patient_physics_v2_sensitivity.py
```

The saved `sensitivity.json` compares the same patient geometry. Pressure 500→1500 Pa increased generated fluid 0.250→0.749 mm³; 30000 rpm/2 N→60000 rpm/5 N increased removal 7→34 mm³; tissue stiffness 0.5×→2× decreased displacement 0.615→0.540 mm. Repeated open-jaw mechanical contact produced zero, intermediate, and full modeled injury with no geometric transection. This demonstrates parameter sensitivity, not validated physiological response.

Desktop verification opened the native drill episode, changed force and target coordinates, previewed 17 mm³ additional removal, applied it, undid to 34 mm³ total removal, redid to 51 mm³, and saved `patient-desktop-v2-verified.ressectionlab`. A separate Python process replayed that native artifact exactly. `desktop-verification.json` records its state identity; `desktop-drilling.png` shows the workspace. The desktop process tree sampled at 2543 MiB RSS (shared pages may be counted twice; this is not peak memory). The scripted patient run recorded preview plus commit times of roughly 0.01–2.7 seconds per action, including a long approach; these are backend timings, not frame latency. Surface rebuilding remains asynchronous and can take longer.

The regression run passed 230 backend tests covering surgical tools, adapters, physics, rehearsal, imaging and workspace imaging. The final geometry-radius guard also passed with the integration suite (14/14 tests). The desktop suites passed 681 tests (including 16 surgical renderer tests). The desktop production build passed with its existing large-chunk warning. No training run was started.

Open `patient-brain-retracted-verified.ressectionlab` to inspect the persistent hold, `patient-brain-verified.ressectionlab` for the relaxed state, `patient-vessel-verified.ressectionlab` for the sealed wound and fluid ledger, or `patient-desktop-v2-verified.ressectionlab` for the manually verified drilling state. All are inside `outputs/surgical-patient-v2/`. In Rehearse, expand “Recorded V2 mechanics and fluid budget”, scrub the timeline, then use Preview/Apply for a further action. Source images stay unchanged beneath the modeled overlays.

The retracted patient artifact was also opened in the desktop. Applying a one-second Release changed the displayed maximum displacement from 0.58832 mm to 0.01688 mm with an accepted action; `desktop-brain-release.png` records the relaxed state. The 69 active solver nodes include anchored nodes; 36 nodes had nonzero displacement in the scripted receipt.

A drawn drill path stops at an unfinished waypoint instead of skipping ahead to a later path leg. Ordinary work-limited dwell is excluded from the collision metric. A requested drill radius beyond the declared bone-profile limit is rejected before execution so cutting work and collision geometry cannot use different burr radii.
