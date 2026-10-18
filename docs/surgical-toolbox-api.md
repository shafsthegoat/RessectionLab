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

`snapshot()` stores schema, scene identity, action sequence, state identity, and `checkpoint: {position, stateId, state}`. The checkpoint contains the full current state at the recorded action count. `SurgicalEngine.restore(scene, record)` replays and verifies identity, then checks the checkpoint position, identity, and full state against that replay; it rejects a mismatched checkpoint instead of trusting it as an unchecked shortcut. `branch(position)` replays a prefix; `replay(actions)` constructs a fresh result. The desktop bridge additionally binds state changes to `caseHash` and `expectedStateId`, rejecting stale edits. Its rehearsal commands are create, inspect, preview, step, undo, redo, seek, branch, selectBranch, and reset.

## Current limits

Mechanics are conservative capsule contact, whole-cell removal, graph cuts, and lattice displacement. Energy is a local scalar, not a calibrated thermal or physiological model. Mesh seams retain connectivity records without remeshing faces or opening a physical aperture. Native mesh seam cutting after lattice displacement rejects explicitly because vertices are not remapped by that deformation model. These outputs support inspection and deterministic engineering tests, not clinical safety or outcome estimates.
