# Optional CUSA contact models v3 and v4

Version 3 (below) is a contact- and time-limited removal law. Version 4
(`cusaV4`, [last section](#tissue-selective-cusa-v4)) adds amplitude, aspiration,
irrigation and tissue selectivity on top of the same exact-progress scheme.

The historical CUSA law removes every compatible source cell intersected by an
active swept tip immediately. That remains the law for every existing scene
without an explicit `scene.properties['cusaV3']` profile. Saved rehearsals retain
their original interpretation. The new opt-in model supplies bounded contact
and time dependence for generated engineering experiments; it is uncalibrated
and does not qualify a clinical procedure or an RL policy.

The physical motivation is limited. Bond and Cimino's original experiments
examined fragmentation rate with respect to stroke, suction, frequency, tissue
type and tip area. Their reported mechanism combines tip impact and mechanical
and hydrodynamic forces. This supports requiring active contact and finite
capacity; it does **not** establish the numerical rate or linear law used here.
[Bond and Cimino, *Ultrasonics* 34 (1996), 579–585,
DOI 10.1016/0041-624X(96)00039-X](https://pubmed.ncbi.nlm.nih.gov/8701547/).

The manufacturer describes suction maintaining tissue contact, adjustable
amplitude and aspiration, and multiple tips for different tissue consistencies.
Its catalog lists differing dimensions, including 1.6 and 2.0 mm **inner
diameters**. These are not the same quantity as our generic catalog radius.
Our fixed 1.5 mm tip radius and 0.5 mm shaft radius preserve this application's
existing catalog geometry; they do not claim to reproduce a specified device.
[Integra CUSA Clarity brochure, revision 10/2019, pages 3–6](https://app.sales.integralife.com/advanced-energy/tissue-ablation/cusa-clarity/cusa-clarity-brochure-ca.pdf).

## Explicit profile and activation

Use `surgical_cusa_v3.default_profile()` to obtain a detached complete profile:

```json
{
  "schema": "resectionlab.cusa-contact.v3",
  "provenance": {
    "kind": "assumed-research",
    "description": "Uncalibrated contact-capacity surrogate; rate is an engineering assumption, not device or patient calibration."
  },
  "tipRadiusMm": 1.5,
  "shaftRadiusMm": 0.5,
  "capacityMm3PerSecond": 2.0,
  "maxActivation": 1.0
}
```

The example rate of **2 mm³/s is an arbitrary engineering assumption**. It has
not been fitted to manufacturer data, a phantom, patient tissue, tumor class,
or any preserved rehearsal result. Changing it changes the source scene and
its identity. A complete explicit profile is required; no legacy scene opts in
automatically.

For CUSA v3 only, the existing generic `power` parameter means a dimensionless
activation fraction, not watts or a manufacturer's amplitude setting. Both
`power` and `duty` must be in `[0,1]`. Values outside the interval and any
`radiusMm` other than 1.5 mm are rejected instead of silently clamped. Effective
activation comes from the engine operation: a one-shot `apply` is active even
if the persistent instrument switch is off. Inactive motion supplies no capacity.

## Contact and accounting law

For each accepted physical interval `dt`, one total available capacity is:

```text
capacity_mm3 = capacity_mm3_per_second × power × duty × dt_seconds
```

The integrator first limits tip travel to the first retained compatible tissue.
The application then checks source/material/protection/shaft/tool collisions
using the shared engine. `apply` independently intersects its candidate IDs with
the actual current fixed-radius tip contact query. Remote or swept-only IDs
cannot receive capacity. Current source-cell centers determine nearest-contact
order; source ID breaks exact distance ties. Capacity is shared across contacts,
so adding cells cannot multiply the interval budget.

Each source cell retains sparse pending `progressMm3`, `progressExact`, and
`contactSeconds` entries. A cell disappears only after its entire affine voxel
volume has been supplied. Pending work has no partial removal, positive removal
reward, or aspirated mass. Exact binary-rational integer pairs authorize the
threshold comparison; rounded float display values and completion tolerances
cannot force a voxel to clear. At clearance, the complete source voxel is
credited to `aspiratedSolidMm3`, and its pending entries are deleted. This model
assumes complete fragment capture upon clearance. Liquid suction and drilling
debris remain separate ledgers.

Capacity with no eligible contact, or beyond the currently contacted material,
is unused; it does not carry forward to remove material at depth. If a different
mechanism removes a pending source cell, its work proxy becomes
`discardedCapacityMm3`. That is discarded capacity, **not** CUSA-aspirated mass.
The checkable cumulative partition, within floating ledger roundoff, is:

```text
inputCapacityMm3 = aspiratedSolidMm3 + sum(progressMm3)
                 + unusedCapacityMm3 + discardedCapacityMm3
```

For a fixed contact set, splitting the same supplied capacity preserves the
threshold law. Representably different floating durations can supply tiny
different capacities; no epsilon is added to hide that difference. Movement,
changing contact sets, concurrent tools and voxel quantization remain explicit
discretization effects.

## Integration contract

- Validate the optional profile when constructing a source scene, initialize
  state only when enabled, and include that profile in the normal scene digest.
- Validate CUSA action parameters before motion; use
  `contact_limited_pose(engine, start, requested, radius)` before sweeps/effects.
  It changes no state and returns the reachable tip plus retained contacts.
- Pass instantaneous compatible contacts to
  `apply(engine, cells, dt, params, position=tip_position, active=effective_active)`.
  Explicit finite RAS-mm position is mandatory. The return value is events;
  the sparse cumulative ledger is in `engine.state['cusaV3']`.
- Retain the limited pose, elapsed time and partial progress when tissue is not
  yet clear. Report `TISSUE_ASPIRATION_IN_PROGRESS` for an unreached requested
  endpoint. Never jump over the retained contact or present this as a collision.
- Call `reconcile_removed(engine)` during action finalization so other tools'
  removals discard pending capacity immediately. Existing transactional rollback
  must include the new state just as it includes other sparse effect state.

The generated tests cover activation, duration, fixed geometry, contact order,
capacity sharing, whole-voxel thresholds, an adversarial rounded-to-full display
value, timestep partitions, conservation and protected membership. They execute
no patient action, training, optimizer step or policy selection. They establish
the surrogate's implementation contract only. Acoustic fields, measured force,
suction pressure, irrigation coupling, thermal injury, tissue selectivity and
patient calibration remain unmodeled.

## Explicit patient-derived experiment

`scripts/prepare_patient_cusa_v3.py inspect --output testcases/outputs/new-inspection`
checks one fixed source line without stepping physics. Its `evaluate` mode uses
a separate new output directory and constructs a distinct case/scene with this
explicit profile. MRI, masks, affine, material ownership and source references
remain identical. Identical protection memberships are rebound to the new scene
with unchanged exclusion settings. Historical branches stay in the original
source file; no old checkpoint is restored under new physics.

The bounded experiment publicly executes five archival bone-access actions,
requiring their exact recorded receipts, then compares short, long and matched
partitioned CUSA dwells at a fixed surface contact. Historical oversized burrs
and durations are explicit native setup, not raw-RL admission. Source-preserving
native save/reopen checks replay all three new branches from intact anatomy.
This is a contact-law experiment that can remove non-target tissue; it does not
qualify tumor access, resection planning or a learned policy.

Electron exposes fixed-radius, dimensionless activation/duty controls only when
the selected scene explicitly enables v3. Its **CUSA contact and aspiration**
panel separates supplied, pending, aspirated, unused and discarded capacity.
Pending capacity is work toward voxel clearance, not removed tissue or fluid.

## Tissue-selective CUSA v4

A scene that declares `scene.properties['cusaV4']` (module
`surgical_cusa_v4.py`) uses this law for the `cusa` instrument instead of V3.
It serves the surgeon question "how will tumour and brain behave under the
tip". Scenes without the property keep identical state ids; the tests pin
captured V3 and suction/retractor sequences to their pre-V4 ids.

The only rule taken from practice is the well-known rationale for ultrasonic
aspirators: they fragment high-water-content parenchyma and relatively spare
collagen- and elastin-rich vessel walls and membranes. Every number below is an
assumed research value; none is fitted to a device, phantom or patient.

### Declaration and settings

`surgical_cusa_v4.default_profile()` returns the complete profile. A scene may
declare `{"schema": "resectionlab.cusa-selective.v4"}` and override any known
field; unknown fields, other schemas and non-assumed provenance are refused.
`regionKinds` maps a region name or id to a tissue kind; a region may also carry
`cusaTissueKind`. Otherwise the kind is inferred from the region name and
material (tumour `core` → `tumour_core`, `rim`/`margin`/`infiltrat` →
`tumour_rim`, other tumour → `tumour`; `tract` → `tract`, `white`/`brainstem` →
`white_matter`, `cortex`/`grey`/`gray`/`cerebellum`/`deep` → `grey_matter`,
other soft tissue → `soft_tissue`; membrane named `dura` → `dura`, other
membrane → `membrane`; vessel → `vessel`). A declared kind must match the
region material.

Action parameters, accepted only on scenes with `cusaV4` or
`suctionTractionV1`: `amplitude` 0–1 (tip stroke), `aspiration` 0–1 (vacuum),
`irrigation` 0–30 mL/min and `duty` 0–1. `power` is not used by V4. The tip keeps
the fixed 1.5 mm radius and 0.5 mm shaft of V3.

### Constants and provenance

| Constant | Value | Provenance |
| --- | ---: | --- |
| `baseRateMm3PerSecond` | 40 mm³/s at full amplitude on tumour core | assumed-research: clinical ultrasonic aspirators debulk soft glioma on the order of cm³ per minute (2.4 cm³/min here) |
| `relativeRate` | tumour core 1, tumour 1, tumour rim 0.6, grey matter 0.5, unspecified soft tissue 0.4, white matter 0.3, tract 0.3 | assumed-research ordering; tracts behave as white matter |
| `sparingAmplitude` | 0.3 | assumed-research: at or below it vessel walls and membranes take no dose |
| `vesselReferenceSeconds`, `vesselReferenceCalibreMm` | 1 s at 1 mm | assumed-research |
| `vesselCalibreExponent` | 2 | assumed-research: wall strength ~ diameter × wall thickness, thickness ~ diameter |
| `duraFailureSeconds` | 3 s | assumed-research (collagen-dense dura) |
| `membraneFailureSeconds` | 1 s | assumed-research (arachnoid and other thin membranes) |
| `aspirationCapacityMm3PerSecond` | 150 mm³/s at full vacuum (9 mL/min) | assumed-research |
| `irrigationRequiredMlPerMinute` | 2 mL/min at full amplitude | assumed-research |
| `tipHeatingPerSecond`, `tipCoolingPerSecond` | 1 /s, 0.5 /s | assumed-research efficiency proxy |
| `defaultAmplitude`, `defaultAspiration`, `defaultIrrigationMlPerMinute` | 0.5, 0.5, 4 mL/min | assumed-research defaults when an action omits them |

### Law

Each accepted interval `dt` at the current tip position:

1. **Fragmentation.** `capacity = baseRate × amplitude × duty × dt / (1 + tipHeat)`
   is shared across current tip contacts nearest first, exactly as V3. A cell of
   kind `k` needs `voxelVolume / relativeRate[k]` of capacity. The same sparse
   `progressMm3`, `progressExact` (integer rational pairs) and `contactSeconds`
   fields live under `state['cusaV4']`, so a renderer reading progress fraction
   per cell works unchanged. Only exact rational comparison clears a cell.
2. **Selectivity.** Vessel and membrane cells in contact take no capacity. Above
   the sparing amplitude they accumulate a dose
   `(amplitude − sparing)/(1 − sparing) × duty × dt` (exact rationals in
   `sparedDoseExact`). A cell fails when the dose reaches its threshold:
   `vesselReferenceSeconds × (calibre / 1 mm)^2` for a vessel, 3 s for dura,
   1 s for other membranes. Calibre comes from the `vesselBleedingV1` segment
   estimate when declared, else a region `calibreMm`, else a local-thickness
   estimate on the region mask (2R − s, floored at one voxel). A failed cell is
   added to `state['removed']`, recorded in `sparedFailures` and emitted as
   `cusa_spared_failure`, so `vesselBleedingV1` opens a wound on it.
3. **Irrigation** adds `irrigation × 1000/60 × dt` mm³ of fluid to the nearest
   free cell in the tip sphere; with none free it runs off as escaped fluid.
4. **Aspiration** carries away up to `aspirationCapacity × aspiration × dt`:
   fragmented solid first, the excess left as debris in the cells it came from;
   remaining capacity takes fluid, then debris, in the tip sphere, with the
   suction fields `fluidRemovedMm3`, `debrisRemovedMm3` and `debrisMm3`.
5. **Tip heat.** `tipHeat ← (tipHeat + heating × amplitude × duty × dry × dt) / (1 + cooling × dt)`
   with `dry = max(0, 1 − irrigation / (2 mL/min × amplitude))`. Idle time
   between applications cools the tip by the same implicit decay. Efficiency is
   `1 / (1 + tipHeat)`. No thermal injury is claimed.

The ledger partition is
`inputCapacityMm3 = fragmentedSolidMm3 + selectivityCapacityMm3 + Σ progressMm3 + unusedCapacityMm3 + discardedCapacityMm3`
and `fragmentedSolidMm3 + sparedFailedMm3 = aspiratedSolidMm3 + debrisLeftMm3`.

### Engine integration and rules

The engine calls V4 at the two V3 points: `contact_limited_pose` in the
reference substep (the tip stops at the first fragmentable **or** spared cell)
and `apply` in the remove branch of `_effect`. While the V4 tip is active,
vessel and membrane contacts are permitted contacts, so the engine does not
refuse them as unsupported material; V4 decides whether they survive. Bone,
unknown and other materials are refused as in V3. A protected contact raises
`PROTECTED_STRUCTURE` as in V3; in `surgicalInjuryV1` injury mode it is
recorded against the named structure instead. In injury mode a breached vessel
or membrane is also recorded as an `ultrasonic` laceration (and `vessel_injury`
with cell ids). V4 scenes run the Python reference transition (no compiled
ledger). Metrics carry `cusaV4*` keys, `view()` the `cusaV4` state and
`observe()['physics']['cusaV4']` the profile, sources and limits.

With the shared physical clock enabled (`clockRevision`), `surgical_clock.advance_instruments`
still refuses a CUSA touching a vessel or membrane
(`ACTIVE_TOOL_UNSUPPORTED_MATERIAL`); spared contacts there need a one-line
permit in that module.

### Limits

- Fixed generic 1.5 mm tip; no tip catalogue, frequency or tip angle.
- Rates, selectivity, sparing amplitude, failure doses, aspiration capacity and
  tip heat are uncalibrated; the ordering is the only claim.
- Whole source voxels; no partial cell, cavity wall smoothing or tissue tags.
- No acoustics, cavitation, tissue temperature or thermal injury.
- Calibre is a grid estimate; vessels thinner than a voxel read as one voxel.
- Spared tissue below the sparing amplitude never fails, however long the dwell;
  a real low-amplitude tip still carries some risk.
- The dose ignores contact pressure; pushing harder does not breach sooner.

`tests/test_surgical_cusa_v4.py` covers the absent-property ids, amplitude and
kind scaling, sparing and calibre-dependent failure, the aspiration cap and
debris, irrigation and tip heat, exact progress, protected contact, replay and
snapshot restore, and V4 with `vesselBleedingV1` and `surgicalInjuryV1` together.
