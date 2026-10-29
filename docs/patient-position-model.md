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
