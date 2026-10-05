# Case4 MRI domain readiness

**Assessment only; no new patient arrays, paired coordinates, inference, meshing
or solver execution.** The baseline alignment declaration is unchanged. This
is preparation for retrospective intraoperative displacement updating: manual
motion observations are unavailable from preoperative MRI alone.

The narrow feasible next domain step is **one existing SynthStrip main-v1
estimate on the original Case4 T1**, independent of all motion landmarks.
Use the unchanged `run_synthstrip` entry point explicitly with `model="main"`,
`device="mps"`, the existing compatible interpreter, and downloads disabled.
The general CLI defaults to other work and is inappropriate for this scope.
No alternative model, no-CSF comparison, threshold tuning, motion-guided mask
correction or automatic CPU fallback is proposed.

The [header receipt](../artifacts/mechanics/resect-case4-baseline-header-qc-v1/header-receipt.json)
binds the real T1: 256×256×192, 1 mm nominal spacing, oblique native geometry,
qform absent and sform present. Its original affine remains the authority.
T1 and FLAIR have different grids; the pending baseline views must assess their
placement before a T1-domain transform is carried into the before-US frame.
Neither equal shapes nor overlapping physical bounds establish anatomical
alignment. No voxel-index copying or affine replacement is permitted.

The previously used [main-only configuration](../manifests/experiments/brain-extraction-btc-spatial-main-v1.json)
provides a prospective bound: one child, two CPU threads, 300 seconds,
6 GiB sampled child RSS, and MPS allocator fraction at most 0.7 with a 6 GiB
allocator ceiling. Runtime probing, validation and persistence need separately
declared overhead. A failing attempt is retained without retry. Earlier BTC
timings and memory observations do not establish this scan's resource use;
native dimensions alone do not determine SynthStrip's conformed workload.

The local wrapper, main checkpoint, upstream script, license and interpreter
were rehashed during this metadata-only assessment and match the existing
declaration: wrapper `702f5df2…`, weights `37417f80…`, upstream script
`291c253a…`, license `632d404e…`, interpreter `b33be71b…`. These are distinct
identities. Before release, bind a committed source snapshot, recheck runtime
versions/import paths and the marked MPS adapter. The existing isolated
NumPy 2.2.6 environment avoids the documented project-NumPy/Surfa conflict.
Historical package-binary equivalence is not attested, and upstream training
overlap with RESECT is unresolved. This remains a development case.

Save mask and predicted signed-distance map as **estimated, review-required
derivatives**. Check finite values, native shape and full physical corner
agreement; independently reconstruct the upstream SDT<1 mm, largest-component,
hole-fill mask. Inspect all three native planes and report boundary contact,
components and volume without invented anatomical thresholds. Preserve every
original. The model's envelope is not a reviewed pial surface, cerebral tissue
partition, cortical access surface or cavity mask. Its distance map includes
upstream outside-fill values and is not calibrated spatial uncertainty. Later
point containment may flag missing support; it cannot justify union, dilation,
clipping points or selecting another model to satisfy measurements.

The remaining mechanics path is bounded but **not execution-ready**:

1. A private [Gmsh 4.15.2 runtime](../artifacts/mechanics/gmsh-4.15.2-runtime-v1/runtime-receipt.json)
   exists; its module/library hashes were rechecked (`f4b0935c…`/`def0f5c1…`).
   Three specimen hex meshes have been prepared. This does not validate
   tetrahedral brain meshing. First inspect the unmodified mask's surface
   topology, orientation and closure in original physical coordinates. Do not
   smooth away sulci, fill anatomical holes or change the domain merely to make
   a mesher succeed. Any proposed simplification needs a separate fixed
   approximation and fidelity report; surface extraction alone is insufficient.
2. Freeze an element formulation, three mesh scales, element-count/resource
   caps and independent quality checks before patient solving. Check positive
   element Jacobians, volume/surface fidelity and convergence of the observation
   operator. Mask agreement measures discretization fidelity, not true anatomy.
3. Existing [FEBio affine patch controls](../artifacts/mechanics-febio-patch-run-v1/summary.json)
   passed for the fixed Ogden law and stiffness scaling. They do not establish
   anatomy-element convergence or MPC correctness. Separately verify fixed
   5 mm volume-average constraints, their rank and rigid-mode control, offsets,
   units and affine-field reproduction with the actual chosen elements. Retain
   the [declared interpolation assumptions](resect-conditional-displacement-poc.md):
   ν=0.45, arbitrary stiffness gauge, no inferred surgical contact or skull clamp.

Preoperative anatomy contains no verified during-resection cavity. Even after
these gates, the first result is conditional retained-landmark displacement;
it cannot validate cutting, retractor forces or dense tissue continuity through
an unknown cavity. B motion and V outcomes remain closed during domain creation.
