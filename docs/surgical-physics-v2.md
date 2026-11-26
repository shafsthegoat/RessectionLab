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

## Generated glioma material assignments

The generated convexity walkthrough opts into `gliomaMaterialsV1` alongside V2. Its source file records each assignment, the modeled pressure boundaries, and the vessel-to-tissue attachment pairs. The original categorical array stays immutable. An action queries current source membership through the same displacement raster used for collisions, field outlets, and saved scene state.

| Quantity or mechanism | Walkthrough assignment and interpretation |
|---|---|
| Equilibrium shear modulus | Brain 300–750 Pa and tumor 250–1,200 Pa. A smooth 8 mm source-coordinate wavelength selects within the overlapping ranges. These are generated scenarios. In human ex vivo rheology, Budday et al. fitted equilibrium shear moduli of 300 and 700 Pa in two regions. Cieśluk et al. found heterogeneous tumor indentation responses under a different loading protocol. [Human rheology](https://pubmed.ncbi.nlm.nih.gov/28658600/) · [Fresh tumor indentation](https://doi.org/10.2147/IJN.S270147) |
| Hold and recovery | A Maxwell branch in parallel with the equilibrium lattice uses a 10 s time constant and a transient-to-equilibrium stiffness ratio of 2.8. Lattice drag damps tool motion. The same spectrum applies to local distortion and volume resistance. This reduced single-branch model is informed by measured relaxation. [Human rheology](https://pubmed.ncbi.nlm.nih.gov/28658600/) · [Human multiaxial loading](https://pubmed.ncbi.nlm.nih.gov/27989920/) |
| Topology and volume | Cuts omit surviving axial bonds. Removal deletes source nodes and their material memory. Each retained node estimates the local deformation gradient from its available face neighbors. Missing directions have zero estimated slope. The numerical domain limits the gradient norm to 0.2 and the estimated local volume ratio to 0.82–1.18. The artificial patch edge anchors neighboring tissue. A weak foundation represents unresolved support. The walkthrough has no imposed gravity or interstitial fluid load. Fluid redistribution and loading rate matter in human quasi-static measurements. [In vivo positional loading](https://pubmed.ncbi.nlm.nih.gov/36514891/) |
| Aspiration | Current accessible tip contact receives one shared finite reference-volume budget. A completed brain voxel costs 1.0–1.25 reference volumes and a tumor voxel 1.0–1.4, with spatial variation. Both are susceptible. Work expended overcoming resistance, physical solid volume, pending work, discarded work and unused capacity are separate ledger terms. The transfer coefficients are engineering assumptions. [Experimental aspirator feedback in tissue models](https://pmc.ncbi.nlm.nih.gov/articles/PMC9463293/) |
| Vessel mechanics | Explicit attachments transfer motion to the cortical artery and vein. Positive axial stretch increases an assigned tether load with stiffness 0.06 N for the artery and 0.025 N for the vein per unit strain. Uncut stretched segments use a fixed-lumen-volume cylindrical approximation in which conductance scales as the inverse cube of stretch. The generated injury threshold is 0.18 tensile strain at the bound face. The artery/vein contrast and thresholds lack specimen calibration. Human passive cerebral artery tests support distinguishing axial loading and pressure. [Human artery biaxial loading](https://pubmed.ncbi.nlm.nih.gov/18855141/) |
| Pressure, bleeding and hemostasis | The cerebral network assigns inlet/bed/outlet pressures of 2,400/1,400/400 Pa initially. Storage over compliance determines subsequent node pressures. The normal closing force includes gauge pressure times assumed lumen area, 0.8 mm² for artery and 1.2 mm² for vein. Cut faces open source-owned ports, a seal changes that port’s leakage, and the cut edge remains disconnected. The separate scalp branch has an assigned supply of 1,800 Pa and a blind distal bed. A cut crossing its mapped margin opens it. A retained local Raney clip suppresses inflow until retrieval. Flow outputs remain regional engineering proxies. |

The biopsy neck binds a separate generated microvascular bed to two tissue cells. It starts with 8 mm³ of blood at 1,000 Pa and has no external inflow. Separating that specific tissue bond exposes the finite bed. Bipolar contact can seal the retained stump. A hemostatic dressing at the same stump reduces leakage while it exerts local pressure. The cortical artery and vein keep their own source identities and circuits.

Run the generated indentation, recovery, timestep, connectivity, aspiration, vessel traction and cold-replay controls with `.venv/bin/python -m pytest -q tests/test_surgical_glioma_materials.py`. The tests establish numerical behavior of the assigned models. Physical measurements from the generated case do not establish agreement with a surgical specimen or a patient.

### Distributed source-bound vessel motion

Set `gliomaMaterialsV1.vesselMotionModel` to `distributed-source-rods-v1` for an explicit source-chain rehearsal. Consecutive vascular graph sites share axial springs, and branches that share a source site share a mechanical junction. The bending solve preserves the magnitude of each rest angle between incident segments. Parallel springs attach each vessel site to its declared tissue supports. Removing a support or cutting its attachment removes that spring. The vessel retains its accepted position and velocity and continues evolving under the remaining loads. A local compliant contact law transfers working-tip force from the dissector and the unpowered bipolar forceps.

The fixed integration step is 1/120 s. The assigned per-site mass and drag are 0.000035 N·s²/mm and 0.0025 N·s/mm. The support stiffness is 0.035 N/mm, the tip contact stiffness is 0.18 N/mm and the radial contact stiffness is 0.25 N/mm. Local radius reduction is bounded to 25% of rest radius. The declared axial stiffness and traction limit remain engineering parameters. These values support reproducible research controls. Patient wall stiffness, tissue attachment strength, and flow are unresolved.

An optional `vesselRodGeometry` supplies a source description and a `sites` map of mapped source cell IDs to RAS-mm center and radius. With the `centerline-tubes` wall geometry (see [native vascular geometry](native-vascular-geometry.md#centerline-tube-model)) the site positions, calibres and wall come from one stored tube model (which sites exist, and their bonds, still come from the local source chains); an edit cuts that model rather than re-extracting the wall. Contact is compliant as below, then a hard shell projection: after each compliant tick a damped least-squares step, smoothed along rod bonds, moves the rods so that every transported wall vertex inside a working shell leaves it (0.01 mm margin). The projected pose must pass the same section-motion and continuous wall-sweep checks. When it cannot, the tool stops as at the compliant limit and the step is `partial` or rejected; the limited tick is charged to the hand whose shell the push could not clear. Contact, compression and the certified fraction use the projected pose, and the projection stays out of the carried velocity (only the accepted share of the compliant solve is inertial). The shell then overlaps the displayed and collision wall by at most `SHELL_TOLERANCE_MM` (0.05 mm). Other wall geometries keep the axis-only limit. A case with retained native scores uses the same owned 0.5-isosurface triangles for display and instrument contact. The engine supplies accepted wall-owner displacements through `view().vesselMechanics`. Removing a vessel cell re-extracts the score wall with that positive excluded. A transection whose local wall forms two closed contours creates capped stumps by clipping the native triangles. An unresolved cut uses displaced source cuboids for the affected connected ownership group and retains the native score wall for other groups. The graph preserves severed source bonds. The coupled contact profile below handles local soft-tissue loads.

Run the native-wall, passive contact, support-release, cut-topology and replay controls with `.venv/bin/python -m pytest -q tests/test_surgical_vessel_rods.py`.

#### Native annular section mechanics

The native annular section profiles require explicit source-bound sites, vertex transport, and continuous wall clearance. Saved scenes retain the response of their recorded profile. The model interprets the estimated site radius `r` as a lumen-radius proxy and uses an assumed wall thickness `t = 0.15r`. The rod uses `EA` for extension and `EI` for bending, with `A = π[(r+t)²−r²]` and `I = π[(r+t)⁴−r⁴]/4`. Half segments contribute compliance in series through a bend. Branch sites share positional degrees of freedom.

| Assigned quantity | Default | Basis |
| --- | --- | --- |
| Axial tangent modulus | 1.8 MPa artery, 0.7 MPa vein, 1.0 MPa unclassified | Passive human pial arteries measured 1.8–5.5 MPa around the approximate in vivo state. Pial arteries were stiffer than pial veins in uniaxial tests. The venous and unclassified values are sensitivity assumptions. [Biaxial experiments](https://pubmed.ncbi.nlm.nih.gov/18855141/), [axial experiments](https://pubmed.ncbi.nlm.nih.gov/12751292/) |
| Wall thickness to radius | 0.15 | Assumed annular section with unresolved wall thickness |
| Effective radial relaxation | 0.08 s | Assigned radial response time. Bridging-vein tension experiments measured viscoelastic effects under distinct loading conditions. [Relaxation experiments](https://pmc.ncbi.nlm.nih.gov/articles/PMC8472519/) |
| Contact friction coefficient | 0.15 | Assigned sliding parameter with a regularized Coulomb bound |

Closest points on the current indexed wall define a force patch per working-tool capsule. Relative feature areas and penetration distribute the bounded load, and barycentric weights transfer it through the same vertex transport used by collision and display. The radial load and effective axial area loss drive bounded lumen narrowing. Accepted lumen radii and segment lengths scale the assigned hydraulic resistance by integrated inverse fourth-power radius along each taper. An open cut uses the accepted endpoint lumen area to scale its modeled outlet conductance. Source-only ordinary rehearsals can declare zero flow when pressure and vascular boundaries are unavailable.

The v2 profile can resolve normal contact between disconnected native walls. If the original step fails the continuous wall check, the solver finds nearby triangle pairs on distinct connected components. A separating direction from the last accepted pose defines a unilateral constraint. The solver transfers opposing normal loads through each triangle corner's vertex-to-rod weights and the factorized damped rod stiffness. An assigned separation of 0.0002 mm provides numerical clearance for the projected constraint. The same translation field moves the visible wall and its collision triangles. This uses the rod's projection-based integration with a bounded contact solve. [Projective dynamics](https://doi.org/10.1145/2601097.2601116)

The continuous guard checks local facet collapse, folds across shared edges, and crossing between nonincident triangles. It rechecks the projected increment and can try smaller increments if the correction fails. The prior pose stays in force if no trial passes. The working instrument axis also needs clearance from retained vessel anatomy. Same-component wall contact, inter-vessel friction, pressure-dependent narrowing from neighboring vessels, and pulsatile coupling remain unresolved. Vessel material and support values are assigned research parameters.

New ordinary patient preparations record `continuous-source-triangles-v2` in the scene and native-wall receipt. Pairs of faces that share just one vertex must stay on opposite sides of a plane through that vertex, with both rays of each face strictly on its own side. The sweep checks the signed projections at both ends of a linear interval and subdivides when one plane cannot certify the whole interval. A crossing or an unresolved interval fails the clearance check. The reference implementation and native backend apply the same certificate. Previously saved scenes with `continuous-source-triangles-v1` use their original check during replay.

Ordinary native-wall preparations declare `passive-dissector-rigid-tick-v2` for contact timing. During a passive move or withdrawal with fixed tool dimensions, each completed 120 Hz rod tick samples the working capsule at the corresponding translation and orientation on the checked path. Friction uses the tip position and angular velocity at that same tick. The contact search bounds the entire swept translation and rotation before evaluating the sampled capsules at their physical radius. The saved fractional remainder carries the phase across movement increments and pauses. Saved `passive-dissector-fixed-tick-v1` scenes retain axial sampling and use the original endpoint sampling for other motions. Tissue reactions are still averaged over each engine increment, so differences with waypoint density can remain in the coupled response.

#### Native wall contact with surrounding tissue

The v3 contact model associates near-rest native wall triangle samples with exposed
soft-cell faces. A bounded, compressive normal force acts on the wall and an
opposing average reaction advances the soft-cell lattice in the same 120 Hz
increment. The normal stiffness uses an effective modulus of three times the
assigned equilibrium shear modulus, weighted by projected contact area and
cell width. The existing assigned shear range remains a research input.
Vacated support cells cease to carry force. Existing saves keep the model
stored in their physics profile.

In the generated paired control, peak tissue motion was 0.022 mm on initial
loading and 0.054 mm on continued loading. The same 12-action TopCoW-014
requests accepted and cold-replayed with both contact profiles. Peak tissue
motion with the new profile was 0.177 µm. The paired controls and M2 Pro timings
are indexed under the [private storage ledger](local-storage-ledger.md).
The v4 profile added a translational slip force opposing relative tangential velocity,
bounded by the normal force times an assigned friction coefficient of 0.15.
The force uses the trial rod velocity and the preceding lattice velocity, with
a 0.5 mm/s regularization scale. V3 and v4 with zero friction reproduce the
same state. Static sticking, inter-vessel friction and continuous collision
against a deforming tissue wall remain outside this approximation.

In the paired synthetic control, v4 increased mean tissue displacement along
the two prescribed slide directions by 0.37 and 0.47 µm. The TopCoW-014 slide
increased peak tissue displacement by less than 0.002 µm. The matched 240 ms
slide took 302 ms with v3 and 316 ms with v4 on the M2 Pro. These are
implementation controls with assigned parameters. The ordinary v4 preparation
retained the original CTA wall binding and all 12 actions cold-replayed.

New cursor-local patient preparations use `linear-annular-research-contact-v5`.
The lattice holds up to 512 slots, or one third of its node budget for smaller
domains, in reserve during optional contact-neighborhood expansion. Loaded
cells can use the full declared budget. The solver may omit a reaction on a
new receiver below 10⁻¹⁰ N, with the sum of omitted force magnitudes bounded
to 10⁻⁸ N per coupling update. Existing receivers retain their reactions.
The same checked action path enforces shaft clearance, working-axis clearance,
wall sweep, and the hard node limit. Earlier profiles retain their recorded
activation behavior.

New ordinary native-wall rehearsals with matching 120 Hz rod and tissue
steps declare `reciprocal-source-springs-v1` in both the scene and the
native-wall receipt. An intact spring pulls its assigned tissue cell toward
the accepted rod displacement over the current engine increment. Its target
is the mean of the wall-checked rod ticks. Each spring contributes 0.035 N/mm
to its receiver's implicit lattice diagonal. Several rods on one tissue
cell contribute through the sum of their spring forces. Cutting the source
attachment or removing either endpoint ends that contribution. Rigid holders
prescribe their support displacement while they own a cell.

The rod and tissue use the same spring law, with the other side held fixed
during each solve. Their evaluated forces can differ within an increment
because of this lag. Wall contact and sliding friction remain separate
loads. The tissue budget and wall clearance still apply. These stiffnesses
and attachments are model assignments. Older saves keep their original
one-way support behavior.

New native-wall rehearsals also declare `native-passive-working-axis-v1`.
Each sampled rod tick checks the working axes of inserted dissectors and
unpowered bipolar tips against the same moved vessel triangles used for
rendering. The working shell can apply a load outside the axis. If recoil
would cross an axis, the solver tries successively smaller displacement and
caliber increments and retains a clear pose. Axes are checked for both hands
again at the accepted action boundary. The state records limited ticks by
hand. Earlier saves replay with their declared policy.
The operative view draws the accepted instruments from the same local
contact envelopes, including the working length and radius configured in
the saved action.

On the reviewed local TopCoW-012 CTA/MRA crop, v4 reached the 2,000-node limit
before meaningful contact. The ordinary v5 preparation retained the exact
source crop and wall binding and completed a checked approach, contact, and
withdrawal. Peak modeled wall displacement was 0.00644 mm. Withdrawal cleared
both contacted source cells and left 0.00566 mm of modeled wall displacement.
The loaded and withdrawn states contained 1,549 and 1,807 active tissue cells.
The largest sum of omitted reaction magnitudes was 8.33 × 10⁻⁹ N.
Cold replay reproduced every boundary, including the archived v4 refusal.
On the M2 Pro, the accepted 380 ms contact request and 500 ms withdrawal
took 2.89 s and 7.18 s through the bridge. The nearby support survey found no
full-shaft clear placement for a soft-only release. [Evidence and receipts](local-storage-ledger.md)

The ordinary TopCoW-014 v5 case retains the same native wall in two
18-action runs. Aspiration removed three estimated brain cells and released
two assigned supports. The intact arm waited an equal interval at the checked
pose because retained tissue blocked the inactive approach. Both arms then
used identical contact, hold, withdrawal and settling requests. At the same
native wall vertex, motion after the 0.60 s hold was 0.05222 mm with the
supports intact and 0.05570 mm after release. After 1.20 s of recovery, the
modeled residuals were 0.000177 and 0.000275 mm. Cold replay reproduced every
accepted boundary. [Paired experiment and film](local-storage-ledger.md)

The parallel-vessel control uses two independently supported 13-site rods and a 784-triangle native score wall. The same 0.4 s dissector stroke rolls back under v1 and advances under v2. The passive vessel moves 0.133 mm, reaches 0.250 mm with continued loading, and recoils below one quarter of that displacement after withdrawal and relaxation. The accepted wall clears the guard and cold replay reproduces its pose and subsequent receipts. Run `.venv/bin/python -m pytest -q tests/test_surgical_vessel_rods.py -k inter_vessel_contact`.

On an Apple M2 Pro with the native sweep guard, the accepted contact stroke took 470 ms at the median of five trials. The projection accounted for 67 ms across 26 correction calls. A well-separated vessel control took 293 ms with v1 and 292 ms with v2 and required no projection. These timings cover the engine action, including collision checks and state publication.

#### Continuous swept cover for axial dissector motion

New ordinary cursor-local native-wall preparations bind
`toolSweepPolicy: axial-dissector-capsule-v2` into the scene identity. Saved
scenes with no sweep declaration use the original midpoint capsule cover.
The second policy applies to passive motion and withdrawal of the fixed
generic dissector. It uses the original cover when instrument extent is
specified in the action, or when a substep exceeds the qualified range. The
absolute RAS coordinates are at most 4096 mm, substep travel is between
10⁻⁵ mm and 0.5 mm, lateral travel is at most 10⁻⁶ mm and rotation is at most
10⁻⁷ radians. Operative tools, active dissection and other instruments use
their established cover.

For each capsule, the midpoint-pose centerline has unit direction `u`.
The displacement between the two tips has axial component `αu` and
transverse component `d⊥`. A point on the centerline would move by at most
`|α|/2` along `u` from its midpoint location if orientation were fixed.
Extending both ends by that distance covers the projected axial travel.
The remaining translation is at most `‖d⊥‖/2`. Shortest-arc interpolation
with endpoint angle `θ` moves each point at most `84 sin(θ/4)` mm from
its orientation at the midpoint, since the centerline is within 42 mm of
the tip. Adding these two distances and outward numerical padding to the
capsule radius encloses every interpolated capsule throughout the substep.
The material guards still use swept cell coverage. The continuous wall
intersection, local fold and facet compression tests are independent.

On the historical TopCoW-012 field with reviewed local registration, a single 380 ms endpoint request completed
contact and a single 500 ms endpoint request withdrew the tool. The prepared
case retained the native score, mask, source crossings and wall triangles.
Contact moved the modeled wall up to 0.00765 mm and loaded the same two
source owners as the recorded fine-waypoint run. Withdrawal cleared both
contacts and left 0.00649 mm of wall displacement. All 105 assigned supports
remained attached. The two requests took 2.11 s and 3.93 s through the bridge
on the M2 Pro. The original direct request still replays its earlier partial
stop at step 8 of 11. Both old histories and all four new action boundaries
cold-replayed exactly. [Evidence and receipts](local-storage-ledger.md)

New local native-wall preparation requires verified positive brain-tissue
labels on their exact acquired grid. The historical TopCoW-012 and TopCoW-014
records remain available under their stored tissue assumptions. Their archived
brain envelopes need additional tissue evidence for new local preparation.
