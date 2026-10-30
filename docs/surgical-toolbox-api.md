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

## Current limits

Mechanics are conservative capsule contact, whole-cell removal, graph cuts, and lattice displacement. Energy is a local scalar, not a calibrated thermal or physiological model. Mesh seams retain connectivity records without remeshing faces or opening a physical aperture. Native mesh seam cutting after lattice displacement rejects explicitly because vertices are not remapped by that deformation model. These outputs support inspection and deterministic engineering tests, not clinical safety or outcome estimates.
