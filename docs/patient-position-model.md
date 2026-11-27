# Patient posture research model, version 1

Position actions explicitly opt an episode into `patientPosition.modelVersion=1`.
Episodes without this field retain their original behavior. The position state,
including all estimates, is serialized with the action/state and deterministic
replay. Position never changes a camera or an image affine.

The neutral supine room frame has +Y upward and +Z toward the head. Patient RAS
means +R right, +A anterior and +S superior. Preset roll about superior is 0°
supine, 180° prone, +90° left lateral, −90° right lateral, and +110° park bench.
Park bench is an explicit left-dependent, slightly prone research approximation;
it does not encode the joints, limbs, supports or a clinical positioning protocol.
Semisitting uses supine roll and defaults to 60° head-up table tilt. Other presets
default to zero tilt. `bodyTiltDeg` is signed head-up tilt, bounded −45° to 85°.
These bounds limit the numerical domain; they do not certify positioning safety.

`headRotationRasDeg=[x,y,z]` applies extrinsic rotations about body R/A/S axes,
in x then y then z order, relative to the body preset. Each component is bounded
to ±90°. The resulting body and head RAS-to-room matrices and gravity vectors
are returned explicitly. Gravity in image/head physical RAS is the inverse head
orientation applied to room downward acceleration, 9.80665 m/s². Voxel-axis
directions are always transformed through the affine; camera orientation is
irrelevant.

`heartToSiteRasMm` is an **estimated** heart-to-operative-site vector in body RAS,
defaulting to the unmeasured research assumption `[0,0,300]` mm. This is not
extracted from cranial imaging. The signed room-vertical component estimates
site height above heart; `siteHeightAboveHeartMm` may explicitly override it.
The provenance string `heightEstimateSource` travels with replay. Head rotation
does not move this body-relative site/heart estimate. A future measured head/neck
kinematic model would be needed to estimate its displacement under head motion.

The reported static-column offset is `p_site − p_heart = −ρ g h`, with height
converted from mm to m, density in kg/m³, and pressure in Pa. Divide by 133.3224
for mmHg. Default fluid density 1060 kg/m³ and tissue density 1040 kg/m³ are
uncalibrated configurable research assumptions, bounded 500–2000 kg/m³.
These offsets are neither venous pressure nor ICP, CPP, perfusion, or an air
embolism risk estimate. No heart pressure or patient physiology is inferred.

Gravity body force is `ρ × abs(det(affine[:3,:3])) × 10⁻⁹ × gravity`, N per voxel,
because affine voxel volume is mm³. The existing local Kelvin–Voigt lattice
receives these forces only on its active local nodes and keeps its displacement,
strain, collision and force bounds. This uses full tissue weight, with no modeled
CSF buoyancy, incompressible FEM, vascular pressure or calibration. Activation
considers tissue exposed beside resected source cells, scanning at most 512
removed-cell entries and 128 seeds, growing at most a 12 mm local radius and
2000 active nodes. Intact anatomy and image background do not activate mechanics.
Cuts block propagation; the outer attached tissue interface anchors the patch.

The optional fluid hook moves subcapacity fluid down physical gravity into free
neighbor cells, using a fixed-microstep snapshot to prevent multiple-edge travel.
It transfers at most `min(1,4 × dt)` of a cell's initial fluid per microstep, with
one-voxel receiving capacity, bounded to 2000 source fluid cells. This rate is an
engineering assumption, not viscosity. Mixed debris follows carrier concentration.
Fluid/debris mass is conserved, and this hook never adds boundary escape. Existing
overflow/source accounting remains responsible for its own explicit escape.

Primary evidence informs qualitative context only:

- [Calliauw et al., 1987](https://pubmed.ncbi.nlm.nih.gov/3591477/) measured
  posture-related cardiovascular, intracranial and venous parameters in posterior
  fossa surgery; the module imports no effect size or patient-specific conclusion.
- [Effects of neck position and head elevation, 2000](https://pubmed.ncbi.nlm.nih.gov/10636614/)
  observed ICP responses in 15 anesthetized surgical patients; this supports
  retaining head/neck orientation as context, without modeling jugular patency.
- [Lawley et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5350445/)
  directly measured posture/gravity-related pressure changes and referenced
  pressure to appropriate anatomical levels. Static hydrostatic correction does
  not reproduce the observed physiological response.
- [Wang et al., 2020](https://pubmed.ncbi.nlm.nih.gov/33222920/) observed an
  association between subatmospheric jugular bulb pressure and air embolism in
  semisitting surgery; only the qualitative mechanism is retained here.
- [NIST pressure conversion](https://www.nist.gov/document/00-20-h133-apdxefinal-17pdf)
  supplies the mmHg–Pa conversion.

The most useful next extension is a measured free-surface/CSF support boundary
coupled to local tissue mechanics and cavity collapse, calibrated against serial
intraoperative imaging. This is a proposed research direction, not implemented
physiology. [Nabavi et al., 2001](https://pubmed.ncbi.nlm.nih.gov/11322439/)
measured serial deformations in 25 patients: gravity largely explained surface
shift, while subsurface shift during resection included cavity collapse and
complex internal changes. Their observations support evaluating that boundary
model against repeated images; they do not supply a universal displacement,
safety threshold, ICP or perfusion predictor.

## Brain shift after dural opening and navigation (scenario, opt-in)

A scene may declare `brainShiftV1` (`src/resectionlab/surgical_brain_shift.py`).
It is a labelled **scenario**, not a prediction: it shows what navigation gets
wrong once the dura is open and the brain sags. Scenes without it keep their
state, state ids and transitions unchanged.

- **Onset and ramp.** When operative access first opens (scalp and dura
  reflected at least 90°, bone flap parked), the engine stores compact
  parameters in `state['brainShiftV1']`: onset time, gravity direction and its
  source, peak, settling time and a settling level. The sag ramps to the peak in
  `settlingSteps` equal steps over `settlingSeconds` (default 6 steps over 60 s).
  Direction is the declared `gravityRasUnit`, else gravity from an applied
  patient position, else the inward normal of the opening.
- **Field.** Peak (default 5 mm) at the exposed cortex in the centre of the
  opening, halving every `depthHalfLengthMm` of depth (default: the depth of the
  deepest original target margin, so that margin moves about half the peak),
  decaying beyond the opening edge with `lateralDecayMm` (15 mm), and tapering
  to zero within one cell plus `boundaryTaperMm` (3 mm) of rigid support. Bone,
  dura still attached to bone and every non-declared region never move; a
  shifting cell within one cell of them (falx, tentorium, skull base) does not
  move either. Default shifting regions are the soft, tumour and vessel regions
  that lie inside the cranial shell as seen from the target.
- **Raster.** The field is never written to `state['deformation']`, so the
  protected-displacement guard, which reads only that local lattice field, does
  not refuse actions because of it. For each settling level a destination map is
  built once and cached on the scene: each shifting source has one rounded
  destination, pulled back along the displacement if it would be rigid; several
  sources in one cell keep all identities as memberships with the existing
  region precedence; vacated cells are filled from the inverse field when that
  source landed adjacent, and any vacated cell enclosed on all six faces is
  filled from its nearest face neighbour. The CSF gap therefore opens under the
  dura, free space downstream closes, and no one-cell hole appears. Instruments
  meet tissue at its shifted position through the shared occupancy; removal and
  every ledger still name the original source cells. Local lattice displacement
  composes with the shift in `surgical_physics_v2.raster`.
- **Readout.** `view()['brainShiftV1']` holds the parameters, current peak, a
  coarse lattice of at most 512 displacement samples and the planned and
  shifted original-target centroids; `metrics()` adds `brainShiftPeakMm` and
  `targetShiftMm`. `view()['moved']` does not list the shifted cells.

`navigationV1` (`src/resectionlab/surgical_navigation.py`) adds a
`navigation_pointer`. Inspecting with it records, in `observations`, what a
navigation screen shows: the tip mapped into the preoperative image through a
rigid registration error drawn once from a declared seed (default 2 mm RMS
target error, 0.5° rotation SD about the target centroid; assumed research,
error model after Fitzpatrick, West and Maurer, IEEE Trans Med Imaging 1998),
the image label there and the distance to the planned tumour. It knows nothing
about brain shift. The latent truth (tip, registration error, the shifted
tissue under the tip and its preoperative position, the tumour-edge
discrepancy) is kept in `state['navigationV1']` and shown only in
`view()['navigationV1']['privilegedTruth']`, never in `observe()`.

**Magnitudes.** Intraoperative series report cortical surface displacement of
several millimetres up to about a centimetre after dural opening, mostly along
gravity, with smaller displacement at the deep tumour margin (Roberts et al.,
Neurosurgery 1998; Nimsky et al., Neurosurgery 2000; Nabavi et al. 2001 above).
No coefficient is fitted to them; the 5 mm peak and half-peak deep margin are
assumed research. For comparison, the repository's RESECT ultrasound
correspondence records give the unchanged-image (no update) error between
preoperative and intraoperative landmarks on single patients: Case4 held-out
RMS 4.04 mm (maximum 5.22 mm), Case2 3.35 mm (maximum 5.19 mm) and Case3
2.03 mm (maximum 3.02 mm) (`artifacts/resect-case4-sparse-update-v1`,
`artifacts/resect-case2-sparse-update-v1`,
`artifacts/poc-resect-case3-image-registration-v1`). These sparse, deep
landmark errors bracket the default scenario: the target-centroid shift on the
host head is about 3.4 mm at a 5 mm cortical peak. They are not a validation of
the field shape.

What a surgeon would still find wrong: the shift is a fixed analytic field that
does not grow with resection, cavity collapse, retraction, CSF drainage volume,
mannitol or ventilation; it never reverses on closure; outward bulging through
the opening is clamped by the reflected flaps; and the pointer has no tracking
jitter, line-of-sight loss or reference-frame slip.
