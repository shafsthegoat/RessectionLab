# Optional CUSA contact model v3

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
