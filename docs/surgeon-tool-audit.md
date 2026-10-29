# Surgical instrument and enabled-physics audit

Audit baseline: `2e7477bb` on `shaf/surgeon-physics-rehearsal`, 2026-10-28. This records the inspected implementation before the follow-on fixes; it is not a clinical validation. No engine or renderer files were changed for this audit. Tests below exercise synthetic research coupons, not patients.

The catalog explicitly says “generic research geometry” and “no calibrated clinical mechanics.” V2 additionally declares assumed parameters, local coupled deformation, work-limited drilling and compliant vascular wounds. Those are useful, bounded approximations. A missing causal mechanism is distinguished below from a defect within the stated approximation.

Severity: **P1** changes whether a commanded surgical action succeeds or removes tissue; **P2** can misrepresent contact, injury or a supported mechanism; **P3** is a disclosed research limitation that should remain visible. Test selectors are relative to the repository root and are executable as `.venv/bin/python -m pytest -q <selector>`.

## What is actually enabled

| Scene selection | Active behavior | Important boundary |
|---|---|---|
| No explicit physics property | Legacy cell effects, rigid lattice holds, instantaneous drill/CUSA removal | Catalog membership does not select a physical model. Preserve historical replay interpretation. |
| `tissuePhysicsV2` | Local Kelvin–Voigt brain lattice; persistent energy-limited drill work and lumped cooling; finite compliant wound reservoir with refill, area-dependent leak and persistent coagulation | `surgical_environment.standardize()` and `prepare_surgical_environment()` select this profile for new environments. No gravity, CSF pressure, anatomical perfusion, clinical necrosis, or patient calibration. |
| `cusaV3` | Fixed 1.5-mm tip and 0.5-mm shaft; instantaneous current-tip contacts; one shared capacity budget; persistent exact pending work; advance stops until complete source voxels clear | This property is independently opt-in. Standardizing/preparing V2 does **not** add it. Without it, even a V2 scene uses legacy instantaneous CUSA removal and ignores CUSA `power`/`duty`. |
| V2 vessel thermal injury | `surgical_vessels_v2.contact(..., temperature=...)` supports accumulated temperature exposure in isolation | Engine call sites do not supply `temperature`. This is **not an integrated thermal injury mechanism**; bipolar `thermal` values are exposure proxies, not temperatures. |

## All 15 instruments

The reference column either links primary evidence for the meaningful mechanism or explicitly identifies an engineering assumption. References justify qualitative mechanisms only; no catalog dimension, numerical threshold, rate, force, temperature or clinical setting is calibrated from them.

| Instrument | Current backend behavior | Rendered geometry | Discrepancy / declared approximation | Primary reference or assumption | Severity and intended correction | Executable existing test |
|---|---|---|---|---|---|---|
| Cranial drill | Bone-only. V2 divides one torque/speed-derived energy budget across frontier cells, retains work until whole-cell clearance, releases equal debris volume, limits advance, integrates heat/cooling once per shared interval. | Sphere burr, flutes and RPM-derived rotation. | V2 is work-limited rather than an eraser. No irrigation cooling, heat diffusion, bit type, breakthrough sensing or patient-specific bone response. The heat is attached to drilled spatial cells; nearby neurovascular heat transfer is absent. | [Kondo et al., primary bone-flap drilling experiment](https://pubmed.ncbi.nlm.nih.gov/10807249/) supports dependence on intermittency, cutter and irrigation; all model coefficients remain assumptions. | P3: keep disclosed; a future explicit thermal version should couple irrigant and neighboring tissue with conservation and sensitivity checks. Do not infer a clinical safe threshold. | `tests/test_surgical_physics_v2_integration.py::test_drill_pending_work_blocks_advance_and_preview_replays_energy`; `tests/test_surgical_drilling_v2.py::test_two_simultaneous_drills_order_independent_shared_thermal_step` |
| CUSA / ultrasonic aspirator | Legacy: every eligible swept soft/tumor cell disappears immediately. V3: contact/capacity-limited solid aspiration with whole-cell thresholds, exact pending ledger and no advance through retained tissue. | Tube, annular mouth and short central lumen. | New standard V2 environments omit V3, so zero power/duty still removes tissue there. V3 explicitly omits acoustics, suction pressure, irrigation, cavitation, heating and differential tissue selectivity. Material rejection protects declared vessels; it does not simulate selectivity. | [FDA CUSA Clarity device description](https://www.accessdata.fda.gov/cdrh_docs/pdf25/K251162.pdf) identifies ultrasonic fragmentation combined with irrigation and aspiration. V3 rate and capture rule are declared engineering assumptions. | P1: explicitly select/version V3 for new rehearsal environments and save that selection; preserve old scenes. P3: retain V3 limits in UI and reports. | `tests/test_surgical_cusa_v3.py::test_public_motion_stops_at_tissue_and_persists_pending_contact_before_clearance`; `tests/test_surgical_cusa_v3.py::test_inactive_zero_activation_zero_duty_and_no_contact_cannot_remove` |
| Brain retractor | Holds compatible soft/tumor cells. V2 imposes a bounded target on a local coupled lattice; neighbors move and release relaxes. Cuts remove bonds; removal deletes nodes. | Flat blade with curved distal lip. | Tissue response is displacement-driven; no brain perfusion or pressure-time injury model. Soft-cell raster overlap preserves source identities but does not establish incompressibility. | [Yokoh et al., primary animal experiment](https://pubmed.ncbi.nlm.nih.gov/6854385/) supports force/time and intermittent-release effects; its numerical values are not transferred to this model. | P3: display displacement/force proxies and elapsed exposure without translating them to clinical injury risk. | `tests/test_surgical_physics_v2_integration.py::test_public_brain_hold_couples_neighbors_and_release_recovers_without_source_edits` |
| Scalpel | Roll-dependent plane severs neighbor bonds inside a tip sphere/depth window; tissue remains. | Thin extruded blade. | No cutting force, crack propagation, local opening or incision-generated bleeding. Missing opening displacement is disclosed topology-only research behavior. Visible blade plane and backend local x/z cut plane should be checked together. | Assumption: a deterministic retained-cell cut seam, not a constitutive fracture model. | P3 for mechanics; P2 for any display/contact mismatch. Use the shared working-envelope specification and plane-orientation tests. | `tests/test_surgical_tools.py::test_blade_roll_changes_cut_plane_and_retains_tissue`; `tests/test_surgical_tools.py::test_blade_cut_follows_displaced_source_cells` |
| Periosteal elevator | Releases only intersected declared `periosteal` attachment IDs; refuses missing attachment properties; removes no bone. | Broad paddle and rounded edge. | No force-dependent peeling or subperiosteal interface. Any touched attachment is released as a complete declared object. | Assumption: supplied discrete attachment graph with categorical release, not inferred anatomy. | P3: retain explicit graph provenance; do not create missing periosteum automatically. | `tests/test_surgical_tools.py::test_attachment_tools_require_properties_and_release_only_declared_kind` |
| Craniotome | Instantaneous roll-dependent thin bone kerf; emits debris; recomputes anchored/separable bone components. V2 does not route this instrument through drill energy/heat. | Cutter plus offset guard/footplate. | Guard is not separately represented by the generic `kerf` backend envelope; no burr-hole entry, footplate-under-bone rule, depth protection, powered cutting rate or flap-removal action. A separable component is retained. | [Primary surgical technique describing footplate use and entry](https://doi.org/10.1093/ons/opz246). Kerf width of half a minimum-spacing voxel and instantaneous cutting are explicit engineering assumptions. | P2: represent guard/cutter in a shared envelope and enforce any declared entry rule; label instantaneous kerf clearly or version a rate law. P3: retained flap is correct topology-only behavior until a separate removal action exists. | `tests/test_surgical_tools.py::test_bone_kerf_retains_separated_component`; `tests/test_surgical_tools.py::test_kerf_roll_changes_thin_cut_and_defers_component_work` |
| Dural scissors | Membrane-only. Opening above 0.5 mm produces no cut; closed jaws sever planar neighbor bonds and supported native mesh seams; sheet remains. | Two blades, pivot and protective distal feature. | No closing stroke, contact force, sheet mechanics or actual membrane opening. Default jaw opening is 2 mm, so activation alone does not cut. | Assumption: binary closed-jaw topology edit at supplied membrane labels; threshold is not a device specification. | P3: retain explicit jaw-state requirement; P2: align blade extent and cut plane with backend contacts. | `tests/test_surgical_tools.py::test_blade_roll_changes_cut_plane_and_retains_tissue`; `tests/test_surgical_tools.py::test_retained_mesh_sheet_seam_uses_native_lps_microns` |
| Microsurgical dissector | Releases intersected declared `soft` attachments only; no removal. | Small angled shaft with rounded tip. | No spreading/peeling force, tissue-plane detection or rupture threshold; release concerns complete declared attachment IDs. | Assumption: categorical release of an explicit soft-attachment graph. | P3: preserve missing-property refusal; keep this distinct from cutting and retraction. | `tests/test_surgical_tools.py::test_attachment_tools_require_properties_and_release_only_declared_kind` |
| Frazier suction | One flow budget takes spatial fluid before debris in deterministic cell order; does not remove solid tissue. Free-field movement checks avoid entering new solid tissue. | Cannula, mouth ring and vent-like feature. | No pressure, lumen blockage, thumb-hole control, tissue capture, or physical multiphase flow. Fluid-before-debris priority is an engineering allocation rule. V2 mechanical vessel contact uses default force even though suction UI does not expose force. | Assumption: conservative local ledger uptake; tip and vent are illustrative, not a pressure solution. | P3: retain local mass ledger; P2: avoid hidden force-driven injury or expose an explicit contact-force assumption consistently. | `tests/test_surgical_tools.py::test_fluid_mass_conservation_suction_local_and_no_tissue_removal` |
| Bipolar forceps | Adds `power × dt` to every eligible spherical-tip cell; reduces bleeding-source activity or progressively seals existing V2 wound faces. Does not restore vessel patency. Stationary activation shares the other hand's clock. | Two insulated-looking arms and tips; activation animation varies jaw separation. | `jawOpeningMm` has no effect on coagulation: even 12-mm-open jaws heat cells between the tips. No opposing electrode contact, current path, impedance, temperature, thermal spread or irrigation cooling. Exposure is a dimensionless proxy. | [Aesculap primary instrument description](https://www.aesculapusa.com/en/healthcare-professionals/or-solutions/bipolar-forceps.html) describes controlled coagulation and insulation; [manufacturer instrument catalog](https://www.bbraun.ie/content/dam/catalog/bbraun/bbraunProductCatalog/S/AEM2015/en-ie/b111/electrosurgical-instrumentsanddevices.pdf) explains the relevance of tip/tissue current contact. | P1: use a declared jaw/electrode contact rule before adding energy or sealing wounds; do not claim that merely touching the two metal tips is sufficient. P3: keep exposure explicitly uncalibrated. | `tests/test_surgical_tools.py::test_bipolar_closes_only_local_declared_source_and_irrigation_does_not`; `tests/test_surgical_tools.py::test_stationary_active_bipolar_shares_other_slot_clock_and_deactivation` |
| Micro scissors | Requires closed jaws. Cuts declared strand edges, native strand seams, or V2 neighboring vessel edges; retains cut pieces and damages V2 vessel faces. Catalog also allows soft tissue/tumor. | Two short blades and pivot. | In ordinary V2 soft/tumor labels without declared strands, an active closed-jaw pass is accepted but records **zero cuts**. The profile suppresses the missing-strand error but supplies only vessel edges. | Assumption: retained pieces and planar neighbor cuts; the catalog's advertised soft/tumor support is the implementation contract, not a claim of calibrated fracture. | P1: cut appropriate soft/tumor neighbor bonds with the same closed-jaw planar rule, or reject unsupported input explicitly; test accepted material effects, not admission alone. | `tests/test_surgical_tools.py::test_micro_scissors_require_strands_and_retain_pieces`; `tests/test_surgical_tools.py::test_open_jaws_do_not_transect_and_source_reservoir_conserves` |
| Tumor forceps | Closed jaws seed the full connected retained tumor component, respecting prior cut/transected edges. Nonbreakable attachments block motion/extraction; breakable attachments release immediately. `extract` deletes all held cells at the current pose. | Two grasping arms and toothed/cupped tips. | Grasp has no local purchase/fragment-size limit. Extraction requires no checked outward motion or outlet and can delete a remote connected mass. Additionally, `extract` repeats in each microstep: after first deletion the missing hold produces `GRASP_BEFORE_EXTRACTION`, yielding a misleading partial result. | [Sklar primary tumor-forceps description](https://www.sklarcorp.com/p/sklar-hunt-tumor-grasping-forceps/98-7771) supports paired cup-tip grasping/removal. Fragment graph, breakability and extraction completion are explicit simulator assumptions; the source does not calibrate them. | P1: make extraction a single lifecycle transition; require an explicit cleared removal condition or document a symbolic removal command visibly. Prefer retained-fragment transport through checked poses before removal; never silently erase a newly grasped whole connected target. | `tests/test_surgical_tools.py::test_forceps_fragment_identity_attachment_and_extraction`; `tests/test_surgical_tools.py::test_two_forceps_cannot_hold_same_connected_fragment` |
| Irrigation | Deposits a directed volume at a free outlet or records out-of-grid escape. Shared field ticks redistribute fluid and carry a bounded debris fraction with mass accounting. | Outlet tube with annular tip. | No jet momentum, pressure, viscosity, thermal cooling or tissue displacement; no coupling to drill or bipolar heat. | Assumption: conservative deposition/transport ledger. [Primary drilling experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC4108487/) supports treating actual irrigant cooling as a separate meaningful mechanism, not something these visual droplets establish. | P3: keep flow and heat claims separate; add cooling only in an explicit energy-conserving version. | `tests/test_surgical_tools.py::test_fluid_mass_conservation_suction_local_and_no_tissue_removal`; `tests/test_surgical_tools.py::test_irrigation_transports_debris_with_conserved_bounded_fluid` |
| Cortical stimulation probe | Returns only matching supplied cortical site evidence, otherwise `unknown`; stores declared stimulation parameters. No removal or thermal effect. | Two distal electrodes. | Amplitude/frequency/pulse width do not generate a response model or filter supplied evidence. No spread of stimulation, electrophysiology or negative mapping inference. | Assumption: evidence lookup over explicitly supplied sites, not simulated functional mapping. | P3: make supplied-evidence/unknown status visible; do not infer functional safety from an unknown result. P2: avoid implying parameter-dependent physiology. | `tests/test_surgical_tools.py::test_probes_distinct_supplied_evidence_or_unknown` |
| Subcortical stimulation probe | Same lookup semantics, restricted to supplied subcortical evidence. | Single rounded distal electrode. | No modeled current-distance relation, fiber activation or tract response. Geometry alone does not make the evidence lookup physiological. | Assumption: separately tagged subcortical evidence lookup. | P3: preserve distinct coverage and unknown results; show stimulation parameters as recorded inputs. | `tests/test_surgical_tools.py::test_probes_distinct_supplied_evidence_or_unknown` |

## Shared geometry and integration findings

1. **P2 — Display and collision geometry are separately authored.** `desktop/src/viewer/surgicalGeometry.ts` draws fixed 9-mm forceps arms, 6/4-mm scissors blades, a 7-mm CUSA body, retractor/elevator paddles and a craniotome footplate. `surgical_motion.envelopes()` uses default `depthMm=2`, 0.35-mm-radius straight jaw capsules, two blade capsules, and at most 3 mm of axial working body. Renderer shaft spans approximately z=-43 to -3, while the backend shaft begins at -40. Some displayed metal lies outside the collision envelope, especially open jaws behind their short backend capsules and the offset footplate. A shared shape specification should drive render dimensions and conservative envelopes; test representative mesh vertices against the union of declared physical parts for every tool, jaw state, radius and orientation. Existing `test_open_jaw_sweep_stops_at_unsupported_material_outside_tip_sphere` proves one envelope case, not render/envelope agreement.
2. **P2 — Visual jaw animation is illustrative.** `updateSurgicalToolAnimation()` changes displayed jaw separation by interval progress even when the recorded `jawOpeningMm` is constant; the backend continues using the recorded value. The comment discloses this, but it must not be shown as recorded physical closure. Prefer recorded jaw transitions, or keep a visible illustrative-motion label. `desktop/src/viewer/surgicalRenderer.test.mjs` currently asserts this visual oscillation and root-pose invariance; it does not assert a corresponding backend jaw event.
3. **P2 — V2 vessel mechanical contact depends on which hand advances the clock.** The operated hand passes zero force when inactive; the stationary hand is considered only when active and receives `forceN` even for field tools. No thermal temperature is supplied. Explicitly document this scope or integrate consistent per-tool contact state. Do not feed dimensionless bipolar exposure into a temperature argument.
4. **P3 — Material filters are categorical.** CUSA/drill cannot touch unsupported labels because the engine rejects them; this creates useful safe research constraints, but not partial biological damage/selectivity. Deformation is local and bounded; one finite vascular compartment is not an anatomical vessel network; all whole-voxel effects retain discretization dependence.

## Verified tests and reproducible gaps

The following passed at the audited baseline:

```sh
.venv/bin/python -m pytest -q \
  tests/test_surgical_tools.py \
  tests/test_surgical_physics_v2_integration.py \
  tests/test_surgical_cusa_v3.py \
  tests/test_surgical_vessels_v2.py \
  tests/test_surgical_deformation_v2.py \
  tests/test_surgical_drilling_v2.py
# 152 passed in 5.72s

cd desktop
node --experimental-strip-types --test src/viewer/surgicalRenderer.test.mjs
# 26 passed
```

This inline coupon writes no dataset or app state. Run from the repository root. At the audited baseline it prints the four gaps; following fixes, convert these into public-API assertions for the version that enables the corrected mechanisms. The existing direct `apply_effect` forceps test misses extraction microstep repetition, and the 15-tool admission test does not assert soft/tumor micro-scissors cuts.

```sh
.venv/bin/python - <<'PY'
import numpy as np
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import standardize

def coupon(material):
    labels = np.zeros((24, 24, 20), np.uint16)
    labels[8, 7:10, 8] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'Engineering coupon', 'material': material}],
        {'kind': 'generated mechanism coupon'}, {'attachments': []})
    return SurgicalEngine(standardize(scene))

def command(e, operation, **kw):
    return e.step(dict(slot='left', operation=operation, duration=.2, **kw))

def pose(z):
    return dict(position=[8., 8., z], quaternion=[0., 0., 0., 1.])

for tool, material in [('cusa', 'tumor'),
                       ('micro_scissors', 'soft_tissue'),
                       ('micro_scissors', 'tumor'),
                       ('bipolar_forceps', 'vessel'),
                       ('tumor_forceps', 'tumor')]:
    e = coupon(material)
    command(e, 'exchange', instrument=tool)
    command(e, 'place', pose=pose(4.), parameters={
        'radiusMm': 1.5,
        'jawOpeningMm': 12. if tool == 'bipolar_forceps' else 0.})
    extra = {'parameters': {'power': 0., 'duty': 0.}} if tool == 'cusa' else {}
    result = command(e, 'grasp' if tool == 'tumor_forceps' else 'apply',
                     pose=pose(7.), **extra)
    print(tool, material, result['status'],
          'v3=', 'cusaV3' in e.scene.properties,
          'removed=', len(e.state['removed']),
          'cut/transect=', len(e.state['cutEdges']), len(e.state['transectedEdges']),
          'heated=', len(e.state['thermal']))
    if tool == 'tumor_forceps':
        start = e.state['slots']['left']['position'][:]
        result = command(e, 'extract')
        print('extract:', result['status'], result.get('reason'),
              'removed=', len(e.state['removed']),
              'pose unchanged=', start == e.state['slots']['left']['position'])
PY
```

Required new regression cases: explicit-new-environment CUSA version persisted through save/reopen with zero activation causing no removal; closed micro scissors cutting soft/tumor bonds while retaining cells; open bipolar producing no energy/seal without a declared two-electrode tissue-contact condition; forceps extraction completing exactly once across durations above and below one microstep, preserving preview/step/replay parity and attachment/corridor guards; all 15 rendered working shapes conservatively enclosed by the same physical envelope used for collision. Thermal integration tests should be added only if an actual explicit temperature/heat-transfer model is introduced.

## Implemented follow-on revision

`configure` is an explicit, recorded action selected by **Enable revised instrument
mechanics** after both hands are withdrawn. It leaves source arrays/properties
unchanged and enables `toolMechanicsRevision=1` in the episode. Historical
prefixes without that action retain their old interpretation and replay.

- CUSA now selects the existing finite-capacity V3 model even when the standard
  V2 source did not originally include it. Zero activation/duty cannot remove
  or advance through tissue; source voxels clear only at complete capacity.
- All revised non-round working parts use native capsule contacts, including the
  stationary second hand. Incision and scissors sever compatible neighbor bonds
  only where their recorded working plane intersects the bounded blade region.
- Bipolar requires closed jaws and local electrode contact before energy/sealing.
  Thermal exposure is still an uncalibrated dose surrogate, not temperature.
- Forceps retain source ownership and in-field volume through translation,
  release and regrasp. Rigid purchase transfers uniform snapped displacement
  out of the elastic lattice; a nonrigid fragment purchase is explicitly blocked.
  Rigid fragments are excluded from gravity reseeding. Rotation of a rigid
  grid fragment remains unsupported and blocked.
- Extraction is a single terminal evacuation after actual collision-checked
  transport beyond a common face of the original occupied-anatomy bounds.
  It cannot auto-break attachments or remove protected membership. This
  conservative exterior criterion may refuse a clinically usable exit; no
  anatomical exit or force threshold is invented.
- Jaw aperture interpolates in native microsteps. Blocked articulation retains
  its last accepted aperture; the rendered metal consumes the same native
  capsules and has no independent close/release oscillator. Full generic shafts
  remain at their recorded poses. These are conservative instrument proxies,
  not manufacturer CAD, hand models or calibrated contact mechanics.

`tests/test_surgical_revised_tools.py` exercises all 15 instruments, with positive,
zero/open, unsupported-material, partial, source-ownership, two-hand, articulation,
preview/replay and extraction checks. Existing V1/V2/V3, drilling, deformation,
vessel, guard and frame suites remain applicable. No calibrated thermal injury,
acoustic CUSA model, closed-loop suction pressure, membrane FEM, universal
attachment inference or physiological stimulation-response model is claimed.

## Operative geometry revision 2

The integrated real-imaging raster exposed a CUSA discrepancy: revision 1's
1.5 mm rear shoulder required tissue outside its aspirating tip sphere to be
cleared first, so even prolonged accepted work stalled at that shoulder.
Explicit `configure(mechanicsRevision=2)` uses the unchanged 1.5 mm working
sphere and a continuous checked 0.5 mm shaft up to the tip. It removes the
unaspirated shoulder; shaft, swept rotation, protected-material, two-tool and
finite-work admission remain enforced. This is a declared conservative research
proxy, not measured CUSA geometry. V3 capacity remains 2 mm³/s at unit power/duty.
Old actions default to revision 1 and retain exact state hashes.

`tests/test_operative_cusa_geometry.py` preserves a revision-1 golden replay,
checks revision-2 source ownership and work budget, rotational/shaft/material/
protection/two-tool guards, and exercises positive effects across all 15 tools.
It also covers actual accepted/blocked jaw geometry, retained forceps ownership
through release/regrasp and checked exterior extraction, and separate conserved
fluid/debris ledgers. Cannula rendering is a hollow illustrative mesh enclosed
by the native working envelope, at the exact recorded pose.
