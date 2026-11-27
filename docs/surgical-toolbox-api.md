# Surgical toolbox engine and adapter contract

The API is a deterministic engineering model. All coordinates used by engine actions are RAS+ millimeters. Geometry, attachments, sources, and observations must come from explicit inputs; the API does not infer patient anatomy or clinical response.

## Scene and adapters

`SurgicalScene(labels, affine, regions, source, properties)` takes a nonnegative integer 3D label array, a nonsingular 4×4 affine, declared region records, source provenance, and mechanism properties. Zero denotes free space. Each occupied label requires exactly one region with an integer `id`, a `name`, and a `material`. Supported materials are `unknown`, `bone`, `soft_tissue`, `tumor`, `membrane`, `vessel`, `fluid`, and `functional_region`. Region records may carry a color and target flag. Material behavior is determined by the explicit material, not the display name.

Source cell IDs use NumPy C-order flattening of the label array. They remain stable when the model removes or displaces cells. Spatial cell IDs use the same grid indexing but denote current locations, so they must not be confused with source identity after movement.

Adapters in `surgical_adapters.py` include:

- `from_nifti(path, mapping, *, frame, units=None, properties=None)`: categorical 3D labels with coded spatial affine; mapping covers every nonzero source value exactly once. Explicit units cannot conflict with known header units.
- `from_mesh(path, material, *, resolution, frame, units, topology='solid', thickness_mm=None, name=None, properties=None)`: solid surface, sheet, or strand import. Sheets/strands require explicit thickness. Solid meshes require closed manifold geometry. The declared resolution determines the voxel-center approximation.
- `from_surface_buffers(...)`: in-memory geometry import using the same explicit coordinate/material/topology conventions.
- `from_native_collection(collection, mapping=None, *, frame, units=None, properties=None)`: registered source collection adaptation.

Accepted frames are `RAS+` and `LPS+`; units are `mm`, `meter`, or `micron`. The resulting scene operates in RAS millimeters. Native mesh vertices, faces, and lines remain in source metadata with their declared original frame and units. Case conversion preserves adapter metadata under `source.adapter`.

## Actions

```ts
type Pose = {
  position: [number, number, number];       // RAS millimeters
  quaternion: [number, number, number, number]; // normalized xyzw
};
type Operation = 'place' | 'move' | 'activate' | 'deactivate' | 'apply'
  | 'hold' | 'grasp' | 'release' | 'exchange' | 'withdraw' | 'extract' | 'inspect';
type Action = {
  slot: 'left' | 'right';
  operation: Operation;
  instrument?: string;                     // catalog ID, for exchange
  pose?: Pose;
  path?: Pose[];                           // up to 512 poses; use instead of pose
  duration?: number;                       // default 1; (0, 30] seconds
  parameters?: Record<string, number>;
};
```

`step(action)` returns `accepted`, `partial`, or `rejected`, plus a reason, completed fraction, state identity, and events. Successful transitions include elapsed time, deltas, and metrics. Partial transitions retain completed safe microsteps and report the contact stop. A rejected transition leaves the live state unchanged. `preview(action)` runs on a clone and includes its candidate view.

Native preview consumes one privately prepared transition candidate; public
`step` publishes that same preparation before computing committed metrics.
Public `clone()` still detaches state, actions and history. Subclasses and
patched public methods retain their clone/step dispatch; rejected previews
render a clean detached source. The bridge reuses its admitted engine only for
the native pure preview and retains stale-state and result-budget checks.
See [paired backend measurements](shared-surgical-guards.md#preview-candidate-ownership-2026-10-30).

`place` requires one free-space pose and a withdrawn/inactive instrument. `exchange` requires deactivation, release, and withdrawal. `hold` is specific to the retractor; `grasp` and `extract` are specific to tumor forceps. Exchange, release, deactivate, activate, inspect, and extract cannot also move the instrument. `apply` performs a finite operation; `activate` enables continuing activity and applies exposure during its own duration. An active stationary instrument continues acting while actions on the other slot advance the shared clock. `deactivate` stops that exposure. Administrative actions also use the simulated action duration; they are not wall-clock waits.

Release removes the reversible hold. It does not restore cells that another instrument has removed. Two holds cannot own the same source cells. Extraction requires a held fragment with no remaining nonbreakable declared attachment.

| Parameter | Allowed range | Interpretation |
| --- | --- | --- |
| `radiusMm` | 0.1–12 | Working neighborhood |
| `depthMm` | 0.1–20 | Cutting depth or irrigation outlet offset |
| `flowMm3PerSecond` | 0–1000 | Irrigation/suction flow budget |
| `power` | 0–10 | Uncalibrated local bipolar exposure multiplier |
| `jawOpeningMm` | 0–12 | Jaw geometry; cutting/grasp effects require ≤0.5 |
| `amplitudeMa` | 0–20 | Recorded mapping input; no physiological prediction |
| `frequencyHz` | 0–500 | Recorded mapping input |
| `pulseWidthUs` | 0–1000 | Recorded mapping input |

Bounds are engineering API limits, not clinical recommendations. A parameter only affects the mechanisms that consume it. Mapping inputs accompany supplied evidence; changing them does not synthesize a response model.

## Explicit properties

```ts
type Properties = {
  attachments?: Array<{
    id: string; cells: number[];
    kind: 'periosteal' | 'soft' | 'tumor'; breakable?: boolean;
  }>;
  strands?: Array<{ id: string; cells: number[] }>;
  boneAnchors?: number[];
  bleedingSources?: Array<{
    id: string; cell: number; rateMm3PerSecond: number;
    coagulationExposure: number; patent?: boolean;
  }>;
  functionalEvidence?: Array<{
    id: string; cell: number; probe: 'cortical' | 'subcortical';
    result: unknown; uncertainty: unknown;
  }>;
};
```

An empty attachment list explicitly declares no bonds; a missing required list produces an error. The elevator releases periosteal bonds, and the dissector releases soft bonds. Forceps can break only bonds explicitly marked breakable. Micro scissors require declared strand cells or native strand topology. Bone anchors distinguish anchored from separable retained components; without anchors, separability is unknown. Source energy thresholds and flow rates must be explicitly supplied engineering parameters. Mapping without matching probe/site evidence returns unknown.

## State, observation, and replay

`view()` includes scene description, catalog, slots, observations, history, metrics, and these geometry/field records:

| Field | Representation |
| --- | --- |
| `removed` | Persistent source cell IDs |
| `moved` | `[sourceCellId, currentSpatialCellId]` pairs |
| `cutEdges`, `transectedEdges` | Source-cell adjacency pairs |
| `separatedAttachments` | Declared attachment IDs |
| `flapComponents` | Retained cells with separability status |
| `meshCutEdges` | Original mesh vertex pairs |
| `meshSeams` | Original face/edge identity and RAS-mm seam coordinates |
| `fluid`, `debris` | Spatial cell ID string → volume in mm³ |
| `thermal` | Source cell ID string → uncalibrated accumulated exposure |
| `sources` | Source state, separate bleeding activity/patency, exposure, reservoir volume |

Fluid conservation includes present spatial fluid and enclosed source reservoirs, aspirated fluid, and explicitly escaped/spilled volume. Bipolar treatment changes bleeding activity independently of vessel patency.

`snapshot()` stores schema, scene identity, action sequence, state identity, and `checkpoint: {position, stateId, state}`. The checkpoint contains the full current state at the recorded action count. `SurgicalEngine.restore(scene, record)` replays and verifies identity, then checks the checkpoint position, identity, and full state against that replay; it rejects a mismatched checkpoint instead of trusting it as an unchecked shortcut. `branch(position)` replays a prefix; `replay(actions)` constructs a fresh result. The desktop bridge additionally binds state changes to `caseHash` and `expectedStateId`, rejecting stale edits. Its rehearsal commands include create, inspect, preview, step, undo, redo, seek, branch, selectBranch, reset, fullCase, and seekFrame.

`fullCase` takes `scenario: 'primary' | 'vascular'` and `maxActions` in 1–64
(default 2). It advances a resumable deterministic director through checked public
actions. Each committed transition must equal its preview. The source-bound
director records phases, attempts and specific stop reasons outside engine state;
it does not change source labels or bypass tool, shaft or protection checks.
The engine permits at most 8,192 accepted/partial actions per branch.

Full rehearsal resolves protection from the current engine's saved
`resectionlab.surgical-protection-contract.v1` contract. There is no implicit
patient-name, filename or local dataset lookup. Ordinary editable responses
include `fullCaseReadiness` (`resectionlab.full-case-readiness.v1`): bound case,
scene and state IDs, `ready | blocked`, available actions, concise blockers with
optional technical details, a recovery action when it can resolve the blockers,
and the prospective strict protection contract ID. The desktop checks this
binding before enabling either start control; the bridge rechecks before every
slice, including resumed directors. Read-only archive/prefix/frame views do not
grant execution readiness. Inspection and supported replay remain available.

At an intact checkpoint, **Attach matching protection** opens a JSON picker for
an explicit `surgical-rl-protection-v1` manifest and its adjacent hashed NPZ.
The bridge `attachProtection` command takes `protectionPath` and
`expectedStateId`; desktop paths come from the main-process picker. It checks
source identity, label/grid digest, exact affine, array integrity, nonempty
brainstem and declared required coverage, and the unchanged strict displacement
policy. Existing clearance, required structures and protected cells cannot be
reduced. A changed binding creates a new branch and preserves the old actions
and contract. After actions, seek to the initial checkpoint before attaching;
full rehearsal can continue an action checkpoint only with its identical strict
contract. Native Save/Open embeds that contract and requires no external
manifest on reopen. Missing targets or unsupported director grids remain blocked.

The CLI accepts `--protection PATH` for explicit initial attachment, otherwise
it reuses the saved contract. It also rechecks resumed histories. Reproduce the
bounded source-backed integration check (fresh output directory required):

```sh
.venv/bin/python scripts/check_full_case_readiness.py \
  --output testcases/outputs/full-case-readiness-20261030/source-check-fresh
.venv/bin/python -m pytest tests/test_full_case_readiness.py \
  tests/test_full_case_protection_bridge.py tests/test_full_case_rehearsal.py -q \
  --basetemp=testcases/outputs/full-case-readiness-20261030/pytest-fresh
READINESS_OUTPUT="$PWD/testcases/outputs/full-case-readiness-20261030/electron-fresh" \
  desktop/node_modules/.bin/electron desktop/electron/verify-full-case-readiness.cjs
```

The local check uses existing sub-06, UCSF-PDGM-0004 and sub-13 source cases,
one action per compatible target case, immutable rejection checks and fresh
canonical save/reopen replay. Missing local inputs are reported explicitly.
Generated adversarial tests cover source/label/affine changes and action-boundary
restrictions. These checks establish software binding, not physiological
validation; planning and training admission rules are unchanged.
Local verification on 2026-10-30 passed all three source cases in 23.13 seconds,
87 focused Python tests, 575 renderer tests, 317 main-process tests and the
desktop build. Hidden Electron attachment and cold-reopen checks also passed;
no full episode or physiological verification was performed.

`seekFrame` takes a recorded action `position`, `frameIndex`, and optional
`branchId`. Frames observe successful engine microsteps at 0.2-second intervals
plus action start/final boundaries; no pose or tissue interpolation is used.
The observed finalized outcome and state must equal canonical replay before
frames are returned. The response's `playback` identifies the read-only frame,
effective tool activation and observed time. Its top-level `position` remains the
editable action cursor. Return to a recorded action boundary before editing,
branching or resuming. Disposable replay caches preserve this distinction;
native saves retain the complete canonical action log and endpoint checkpoint.

Removal metrics use material ownership in the supplied simulation grid.
Overlapping raw source masks are separate evidence: zero removed owned
`soft_tissue` cells does not establish zero raw brain-mask membership or clinical
healthy-tissue injury. Full source annotation and owned target denominators must
be reported separately.

## Opt-in finite planar blade maneuver

An explicitly bound `planarBladeManeuverV1` scene admits only four ordered native
actions: `{slot:'left', operation:'bladeManeuver', maneuverPhase, duration}`.
Phases are `approach`, `indent`, `hold`, `withdraw`, with fixed durations
0.1/0.2/0.2/0.2 seconds and 0.025-second kernel increments. The declaration binds
source geometry, material profile, finite pad/shaft, exposed faces and independent
support faces. Missing declarations and unsupported geometry/history refuse;
ordinary historical scenes keep their previous actions and behavior.

Each phase uses the existing preview/step transaction and observer microsteps.
Canonical JSON owns acquisition/release, Q1 displacement, Maxwell memory and
energy/work ledgers; measured wall time stays outside state/history. The typed
`view.planarBladeManeuver` carries concise status/next actions, declared rigid
primitives and source-owned deformed boundary corners. The same data drive the
desktop pad/shaft and Q1 surface, with a fixed initial optical frame. Source/MPR
arrays remain original. Unsupported planning/full-case commands stay blocked.

See [supported domain, measured results and reproduction](planar-blade-maneuver.md).
This integration retains the standalone kernel's failed 5% cycle-work sensitivity
gates and establishes no physiological or patient-prediction qualification.

## Opt-in selective CUSA and suction traction

Two scene properties make the CUSA and the Frazier suction behave more like the
instruments a neurosurgeon holds. Each is opt-in; scenes without them keep
identical state ids, events and refusals. Scenes with either run the Python
reference transition. Both add the action parameters `amplitude` (0–1),
`aspiration` (0–1), `irrigation` (0–30 mL/min) and `duty` (0–1); other scenes
still refuse them as unsupported.

**`cusaV4`** (`surgical_cusa_v4.py`) replaces CUSA V3 when declared: amplitude-,
aspiration- and irrigation-dependent fragmentation that is fastest in tumour
core and slowest in white matter and tracts, spares vessel walls and membranes
below a sparing amplitude and breaches them after a calibre-dependent dwell
above it, leaves debris when aspiration cannot keep up and loses efficiency
when the tip runs dry. Constants, provenance, the law and its limits are in
[the CUSA contact model](cusa-contact-model.md#tissue-selective-cusa-v4).

**`suctionTractionV1`** (`surgical_suction_traction.py`, requires
`tissuePhysicsV2`) lets an active `frazier_suction` work as the second hand.

- When the active tip touches soft tissue or tumour, and no placed cottonoid
  covers it, the touched (unprotected, unheld) cells become a coupled hold
  through `surgical_deformation_v2.begin_hold`, the brain retractor's mechanism.
  Moving the active suction (`move` with the switch on) moves the hold target, so
  the gripped patch follows as gentle traction or counter-traction.
- Hold force = `aspiration × maxVacuumMmHg × 133.322 Pa/mmHg × π (tipLumenDiameterMm/2)²`:
  0.084 N at full aspiration, 0.042 N at the default 0.5.
- After each lattice step the hold load is the hold-spring load,
  `holdStiffnessNPerMm × Σ(asked − current displacement)` over the held cells.
  It releases the patch with `suction_hold_released` when the load exceeds the
  hold force (`hold-force-exceeded`) or the switch is off (`vacuum-off`). A torn-off
  patch is not re-gripped in the same action until the tip has lost contact.
  `release` also lets go, and `withdraw` requires it first.
- In `surgicalInjuryV1` injury mode a `hold-force-exceeded` release records a
  traction avulsion (`surgical_injury.tear`, code `SUCTION_HOLD_EXCEEDED`,
  attachment `suction-patch:<slot>`) and marks the gripped cells contused.
- Over a placed cottonoid (an `operativeExposureV1` object of kind `cottonoid`,
  not held, within its radius plus half a voxel diagonal) the suction takes
  fluid and debris through it and holds nothing (`suction_through_cottonoid`).
- Fluid and debris uptake keep the ordinary suction law (`flowMm3PerSecond`).
- **Reaction estimates.** On these scenes every action ends with one read-only
  `tissue_reaction_estimate` event per coupled hold (retractor or suction):
  `reactionN` (vector), `magnitudeN`, the slot's `inputForceN` and the basis.
  It reads the hold-spring load and changes no dynamics or state. Tools whose
  force is only an input (`forceN` for dissector, bipolar or vessel contact) have
  no lattice reaction to report, so they get no estimate.

| Constant | Value | Provenance |
| --- | ---: | --- |
| `maxVacuumMmHg` | 200 mmHg at full aspiration | assumed-research; not a device specification |
| `tipLumenDiameterMm` | 2 mm (fine, about 10 French, Frazier lumen) | assumed-research |
| `defaultAspiration` | 0.5 | assumed-research |
| Hold stiffness, lattice and caps | `tissuePhysicsV2.brain` profile | existing uncalibrated lattice (`holdStiffnessNPerMm` 0.02, `maxDisplacementMm` 3) |

State lives in `state['suctionTractionV1']` (`holds`, `releases`,
`regripBlocked`, `throughCottonoidSeconds`, `peakLoadN`); metrics add
`suctionHolds`, `suctionReleases`, `suctionThroughCottonoidSeconds` and
`suctionPeakLoadN`. Limits: no seal quality, porosity or tip angle; the suction
never aspirates brain itself; release is a force threshold on the lattice load,
not pial tearing mechanics; on the generated lattice the load rises steeply once
the patch nears its strain limit, so release is close to a traction distance of
about 1.5 mm and full vacuum holds only slightly further than half vacuum; a
cottonoid is a centre and radius with no absorbency. The reaction estimate
describes the action's end state, not its peak (`peakLoadN` keeps the suction
peak).

## Current limits

Mechanics are conservative capsule contact, whole-cell removal, graph cuts, and lattice displacement. Energy is a local scalar, not a calibrated thermal or physiological model. Mesh seams retain connectivity records without remeshing faces or opening a physical aperture. Native mesh seam cutting after lattice displacement rejects explicitly because vertices are not remapped by that deformation model. These outputs support inspection and deterministic engineering tests, not clinical safety or outcome estimates.

### Generated operative instrument profiles

The generated `operativeExposureV1` scene selects 21 fixed profiles. Instrument geometry revision 2 includes the entire rigid instrument from working tip to the end of the handle, connector, or handpiece. Earlier sealed scenes retain revision 1 with its 40 mm distal section. The original catalogue remains available to historical cases. Each accepted slot supplies its local capsules (`contactGeometry`) in millimeters. Its catalogue row supplies the corresponding render forms in `physicalProfile.parts`. Local +Z points into the field. The recorded quaternion rotates the shaft, guard, edges and jaws together. Insulated arms, the pivot, and the craniotome foot and post count as passive obstructions. Jaw aperture changes the native cover, including motion between accepted values. The rendered bevels, flutes, lumen walls, and gripping faces lie within that cover.

The 4 mm cranial burr and 1.5 mm craniotome cutter use nominal diameters in the Medtronic MR8 catalogue. The simulated drill makes a burr hole using the spherical burr. A commercial paired, self-stopping perforator has distinct inner and outer diameters and a clutch. The rehearsal does not model that mechanism. The guarded craniotome has a separate cutter and passive foot. [MR8 product catalogue](https://www.medtronic.com/content/dam/medtronic-wide/public/western-europe/medical-specialties/spine-surgery/midas-rexmr8-catalogue-produits.pdf) · [MR8 perforator dimensions](https://www.medtronic.com/content/dam/medtronic-wide/public/western-europe/medical-specialties/spine-surgery/midas-rex-mr8-wall-chart-poster.pdf)

The Frazier suction cannula has a 3 mm outer diameter. The bipolar profile has two exposed tips of 0.5 mm diameter. The clip applier's reusable jaws and the temporary scalp clip are distinct objects. [Stryker Frazier suction tube](https://www.stryker.com/hk/en/ent/products/stryker-ent-navigation-system.html) · [Aesculap neurosurgical consumables and forceps](https://www.bbraun.com/content/dam/b-braun/master/website-6/en/01_products-and-solutions/0102_therapies/010207-neurosurgery/neurosurgery-consumables-catalog.pdf)

The numerical seal binds the local coordinates, radii, contact roles and render forms at both ends of the permitted jaw range. A revision 2 angular sweep uses the reach of each individual capsule from the rotation center. Opposing handles, finger loops, and insulated limbs can obstruct a motion while both tips remain in free space. Rings use conservative filled capsule envelopes. A bent suction shank and paired spring grips move rigidly with the instrument.

| Instrument | Modeled axial extent, mm | Nominal measurement or assignment |
| --- | ---: | --- |
| Frazier suction | 180 | B. Braun GF923R specifies 180 mm total, 100 mm working length and 3 mm outer diameter. The proximal bend is assigned. [GF923R](https://surgical-instruments.bbraun.com/en-01/p/GF923R/frazier-suction-cannula-180-mm-7-curved-9fr-3-mm-work-length-100-mm) |
| Bipolar forceps | 195 | Representative 195 mm instrument with 75 mm working length. The 0.5 mm tip and body combine catalogue dimensions from the same instrument family. [Bipolar forceps catalogue](https://www.bbraun.com/content/dam/catalog/bbraun/bbraunProductCatalog/S/AEM2015/en-01/b182/electrosurgical-instrumentsanddevices.pdf) |
| Needle driver | 180 | A straight 180 mm Castroviejo needle holder supplies the length. Spring profile, grip width and jaw shape are approximations. [BM004R](https://surgical-instruments.bbraun.com/en-01/p/BM004R/castroviejo-durogrip-micro-needle-holder-tc-straight-180-mm-7-jaw-with-0-2-mm-pitch-of-serration-flat-handle-with-ratchet-for-suture-6-0-10-0) |
| Scalpel | 143 | A 125 mm No. 3 handle plus an assigned 18 mm exposed blade and mount. The active bevel occupies the distal 5.8 mm. [BB073R](https://surgical-instruments.bbraun.com/en-01/p/BB073R/scalpel-handle-no-3-blade-size-no-3-125-mm-5) |
| Cranial burr and craniotome | 172 each | Assigned complete handpiece geometry including the motor grip, with a 70 mm shaft-to-grip junction. MR8 documents 70 mm attachments and footed craniotomy options. [MR8 system](https://www.medtronic.com/en-us/healthcare-professionals/products/neurological/powered-surgical-instruments/high-speed-pneumatic-surgical-drills/midas-rex-mr8-high-speed-drill-system.html) |
| CUSA, retractor, periosteal elevator | 194, 185, 165 | Assigned research dimensions. The CUSA working sphere stays at 1.5 mm radius with a 0.5 mm rear distal shaft. |
| Dural scissors, micro scissors, dissector | 170, 155, 175 | Assigned ring, spring and pencil handles respectively. |
| Tumor, bone-flap, membrane and dressing forceps | 160, 165, 150, 155 | Assigned spring or ring handles and distinct working jaws. |
| Irrigator, cortical and subcortical probes | 155, 165, 165 | Assigned valve grip or pencil grip. |
| Scalp clip applier and bone fixation driver | 165, 136 | Assigned ring applier and driver grip. |

Instrument geometry revision 3 is the revision 2 whole-device catalogue with two anatomy-scale distal profiles, appended as profile ids `scalpel_long_blade` and `craniotome_router` so revisions 1 and 2 keep their profile indices, catalogue rows and `operativeInstrumentCatalogHash`. Host rehearsals declare it. The long scalpel blade carries a 22 mm cutting edge (assumed-research: a No. 10 or No. 22 class blade presents roughly 20-30 mm of edge; 22 mm puts the full host scalp, up to 11 mm on the 1 mm grid, on the edge with the shaft outside tissue); its bevel pair is the revision 1 pair scaled along the edge. The craniotome router carries a 14 mm side-cutting flute with its guard post lengthened to match and the unchanged 0.7 mm foot (assumed-research: footed routers are long enough for the full calvarial thickness; 14 mm clears the 3-10 mm host skull with the post inside the cut trough). Both keep the revision 2 handle and handpiece. The catalogue row of a revision 3 tool names its `profileId`, and the desktop draws the same capsules, with one flute band per 1.6 mm of cutter.

The guard offset, gripping faces, bevels, shaft bends, hinges, insulation and handle transverse dimensions are engineering assumptions. Dimensions stay fixed when the camera or opening changes. The rigid collision model ends at the proximal connector and excludes external hoses, cables, console and operator hands. Conservative capsules can report obstruction before the displayed metal surfaces touch. The closure instruments apply scenario-declared approximation or fixation constraints. Thread passage and screw torque are outside this model.

### Generated scalp, bone and dura exposure

A generated scene can declare `operativeExposureV1`. Its three bodies retain their original material-cell IDs through cutting, holding, parking and closure. The public view reports accepted RAS-mm centers, cut neighbor pairs, remaining attachment counts, grip position and held tip offset. The engine uses the same transformed centers for material occupancy and inspects the moving body against both instrument assemblies, retained tissue, fixtures and declared protected anatomy. The occupancy grid resolves whole voxels. The body-to-tool check uses a bounded source-center model.

A scalpel or closed dural-scissor edge must intersect an actual material bond in its cutting plane. The remaining hinge stays attached. A closed forceps grip drives the free edge around a finite bend in the curved source sheet. The outer strip preserves longitudinal arc length within each local column. Through-thickness shear increases smoothly away from the attached edge. The resisting moment is $k\theta+c\dot{\theta}$, limited by applied grip force and lever arm. The scalp and dura use assigned stiffnesses of 1.8 and 0.45 N·mm/rad and damping of 0.15 and 0.04 N·mm·s/rad. Human tensile and tissue-structure measurements motivate distinct materials. The fold coefficients, anisotropy approximation and grip force limit are engineering assumptions. [Scalp tensile measurements](https://pubmed.ncbi.nlm.nih.gov/29793156/) · [Dura microstructure and mechanics](https://pubmed.ncbi.nlm.nih.gov/37598794/)

A second hand can secure a reflected margin with a temporary retaining stay. Its fixed RAS point and reaction moment represent local external support. The first hand must regrasp the sheet before the stay comes out. The model resolves one folding degree of freedom and omits full membrane strain, thread routing, tear propagation and a surgical frame.

A small, uncalibrated scalp clip applies pressure to the wounded arterial stump. Its pressure jaw, neck and external backing have the same geometry in the collision checks and the public display. The operator regrasps the backing with the scalp reflected, disengages the jaw through the divided margin and carries the entire clip outside for counting. Removing pressure can restore local flow. Durable sealing requires a separate bipolar dose on the registered stump.

A host rehearsal (`hostRehearsalV1`, built by `scripts/build_host_glioma_rehearsal.py` from any prepared study with scalp or skin, skull, cortex, white matter and tumour layers) declares three opt-in additions. Each scalp and dural body may set `foldReference: hinge-deep-face-v1`: each slice across the hinge bends about the deep-face height of its own hinge column, so a thick sheet on a curved head rolls back with its curvature instead of driving deep layers into bone, and the flap is first lifted 2 mm off its bed over the first 6 degrees (assumed-research). The host scalp declares a 25 mm roll and the dura 6 mm. `accessDepthFootprintMm` bounds the closed-access depth test to the craniotomy footprint, since the curved scalp beyond it lies below the dural plane. The bone body's `bladeCorridorReachMm` (4.05 mm on the host, default 2.55) lets the cutter trough reach outer bone lying below a thin flap edge. The host source also carries assumed anatomy, each recorded in the declaration: a one-cell dura, a two-cell epidural plane in the annulus under the kerf and burr hole (from 2.5 mm inside to 6.5 mm outside the 22 mm flap, following the locally deepest bone within 3 mm) and a one-cell plane inside it, the dura stripped from the flap's inner table so that dural sutures clear the replaced flap (the cortex there sits one cell deeper than estimated), a subgaleal dissection plane under the scalp flap, the outer 10 mm of scalp in the flap, filled dropouts and pockets of the T1 skull estimate under the craniotomy and reassigned bone flecks inside the scalp. Its CUSA V3 profile declares 40 mm³/s (assumed-research: clinical ultrasonic aspirators debulk soft glioma on the order of cm³ per minute; the 2 mm³/s default would take about two simulated hours for 14 cm³). Operative exposure no longer depends on source class; only an inspection-only source is refused.

The burr must create a local through-perforation before the guarded side cutter extends a connected kerf. Both tools spend finite cutting work and account for heat and debris. Outer table, diploë and inner table are separately identified in the source and use the same drilling-energy coefficients. The rigid flap releases only when every load-bearing perimeter bond is open. Its exterior parking rest supports three inferior contact points. Replacement requires seating in the original pose and local fixation bridging the kerf to retained skull.

Counted cottonoids leave through the exterior in a gripped instrument. A retained hemostat can contact a bound microvascular outlet. Durally and cutaneously placed sutures constrain paired material margins at declared attachment sites. Their spans and the bone fixation spans participate in contact checks. `closureSecured` reports achieved closure and object accounting. `complete` also requires both instruments withdrawn, inactive and empty. Residual target, pressure and source flow remain separate measurements.
