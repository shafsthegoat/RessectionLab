# RESECT Case4: conditional displacement proof of concept

Prospective design, 2026-10-04. No patient image arrays or landmark coordinates
were opened for this design. Case4 is permanently **DEVELOPMENT**, selected by
the acquisition agent's smallest complete-file metadata rule. Published
case-level summary distances were incidentally consulted after that selection;
this is not an unconsulted clinical test. BTC, UPenn and existing patient roles
remain unchanged.

The question is whether elastic interpolation, conditioned on six measured
internal motions, predicts other retained-landmark motions better than simple
interpolation. It does not identify the surgical action or force causing motion.
The [original RESECT release](https://archive.sigma2.no/dataset/5D6BFC33-F58D-4F56-88E8-C40AF269D6F2)
is CC BY 4.0. Its [creator instructions](https://yimingxiao.weebly.com/data-repositories.html)
resolve the paired columns as reference MRI/before-US followed by destination,
in world millimetres; corresponding MINC/NIfTI world coordinates agree.
Actual headers must still establish anatomical axis convention and pair binding.

## Minimal inputs and permitted access

All paths below are relative to `RESECT/NIFTI/Case4/`; acquisition pins provider
fixity and measured local SHA256 separately. Six files total **35,068,010 bytes**.

| File | Bytes | Role |
| --- | ---: | --- |
| `MRI/Case4-T1.nii.gz` | 8,763,797 | Real structural domain input |
| `MRI/Case4-FLAIR.nii.gz` | 6,550,243 | MRI reference for baseline alignment |
| `US/Case4-US-before.nii.gz` | 9,955,302 | Baseline geometry and alignment QC |
| `US/Case4-US-during.nii.gz` | 9,796,312 | Sealed from fitting; later independent visual QC |
| `Landmarks/Case4-beforeUS-duringUS-full.tag` | 1,304 | Source-only partition, then six input pairs and sealed remaining outcomes |
| `Landmarks/Case4-MRI-beforeUS.tag` | 1,052 | Baseline alignment input, never an independent displacement outcome |

The original inventory supplies no Case4 brain/cavity mask or tetrahedral mesh.
US coverage is not a brain boundary. Derive a separately versioned, unreviewed
brain envelope from T1 under a frozen extraction protocol, retaining the native
affine and source/model hashes. Meshing that real estimated surface requires
closure, orientation, component, element-quality and physical approximation QC.
No artificial patient anatomy, nonzero-MRI brain substitute, or landmark-driven
mask dilation is allowed.

## Outcome separation and baseline frame

The trusted partitioner alone reads the paired file. It exposes source columns
without displaying, logging or deriving selection features from destinations.
Use one-based original row IDs, finite coordinates, at least twelve rows and no
duplicate source coordinates; ambiguity is a recorded block, not a merge across
roles. Select **exactly six B inputs** by farthest-point sampling: start at the
lowest row ID, then maximize minimum squared physical distance to the selected
set; exact ties choose the lower ID. All other destinations are **V**, at least
six. Require centered B to have smallest/largest singular-value ratio greater
than `1e-6`. This rank-three volumetric/affine-baseline eligibility rule is
stronger than rigid registration needs; it does not prove spatial coverage.
Retain failures without case replacement.
Freeze selected IDs, source coordinates, raw-file/pair/frame hashes and the
algorithm before exposing B destinations. The solver receives B only; V sources
and destinations stay with the independent evaluator. No separate C or tuning
on V is permitted. Overlapping historical landmark sets are not new samples.
Spatially spread B supports conditional interpolation; correlated nearby V
points do not justify independent-point confidence intervals.

Use one proper least-squares rigid fit from all permitted MRI-before pairs,
FLAIR-world to before-US-world, with no outcome-driven outlier rejection or
nonrigid correction. The paper reports upstream T1-to-FLAIR rigid alignment;
verify the delivered grids before applying the baseline transform to physical
mesh nodes. Preserve original arrays/affines. Record every baseline fit residual,
leave-one-out baseline diagnostic, transform and source-plane overlay; these
diagnostics are not independent physical truth. Freeze alignment/QC before B
displacements or V outcomes are released. During-US intensities, registration
warps and V destinations cannot improve that fit. Initial brain shift and shared
tracking/annotation errors remain unresolved uncertainty.

## Observation and boundary assumptions

Use an established FEM solver with homogeneous elasticity, zero body load and
a traction-free exterior. These are interpolation
assumptions, not observed skull, dura or retractor conditions. Set Poisson ratio
0.45 prospectively; absolute stiffness is a scale convention, not measured
patient stiffness. No patient material fit is attempted.

Prefer the source-verified FEBio `Ogden` alpha-two candidate shared with the
[specimen controls](hbe-specimen-mechanics-poc.md): `c1=2μ`, `m1=2`, other
coefficients zero, `pressure_model=1`, and `K/μ=29/3` for the declared ratio 0.45.
Its finite-strain energy is `μ/2 (Ibar1−3) + K/4 (J²−1−2 ln J)`.
Pin source revision `32ae206ff4881dfb54f62296cd1558e58ed9fcc6` and the eventual
runtime/deck; source verification is not completed solver verification.
No fitted specimen stiffness transfers. Positive Jacobians, reported principal
stretch range and convergence are required, without inventing an injury limit.
The final model/element choice must be declared before V; no V-directed switch.

Avoid isolated point clamps. Define each observation as a normalized volume
average with the fixed physical kernel `max(0, 1 - distance/5 mm)` over the real
domain, and constrain that average to the measured B displacement. The 5-mm
support is an explicit smoothing assumption; the measured point does not prove
that its surrounding tissue moved uniformly. Hold the support fixed during
mesh refinement and report point and average residuals separately.
Record truncated support volume and kernel first moments: averages near a domain
edge need not equal the displacement at the landmark. Independently reproduce
constant/affine field averages and rigid-mode rank; do not hide this approximation
with nearest-node snapping.

Require independent observation rows and rank six when the observation operator
acts on the three translation and three rotation modes. Failure blocks the
solve; do not add invented skull anchors. FEBio's existing
[linear multipoint constraint](https://febiosoftware.github.io/febio-docs/features/features/core_bc_linear_constraint/)
supports an offset plus weighted nodal degrees of freedom: rank-checked row
reduction can encode these averages without an additional force engine.
The exact pinned runtime's offsets, dependent-DOF handling and constraint
conditioning still need analytic verification. Reactions at these
constraints are numerical conditioning forces, not measured contact forces.

## Evaluation and remaining gates

Give identical B to no-shift, proper rigid-fit and fixed simple-interpolation
baselines. Freeze the interpolation specification, FEM settings, three mesh
scales, source-only alignment acceptance and numerical budgets before any V
reveal. Independently interpolate FEM predictions at V source positions and
report millimetre RMS, median, maximum, paired differences, B residuals, coverage,
numerical convergence and every failure. No bootstrap population claim from one
patient. Follow the [validation contract](tissue-mechanics-validation.md).

Before/during resection can remove tissue. With no accepted cavity domain, the
preoperative mesh is an interpolation domain, not a verified retained-tissue
mechanical model. Initially publish retained-landmark predictions only; do not
present a dense warp through an unknown cavity as tissue motion. Count unsupported
or outside-domain measurements under frozen rules. There is no synchronized
tool pose/load, contact boundary, removal history or force reference here, so
this experiment cannot validate cutting, retraction, injury or action-response.
The [creator methods](https://users.encs.concordia.ca/~hrivaz/Xiao_Database_MedPhys.pdf)
support correspondence assessment; the distinct measured specimen experiment
remains necessary for force response. Acquisition can proceed once its pins and
access rules are committed; anatomy estimation, meshing and patient solves each
still require their bounded declarations.
