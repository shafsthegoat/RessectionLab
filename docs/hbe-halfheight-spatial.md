# Finer axial spatial convergence

The previous N8/N12 half-height equivalence result permits a narrower, cheaper
numerical refinement experiment. It does not calibrate human brain material.
The original spatial-convergence failure and full-domain N24 timeout remain
unchanged. This experiment must pass before measured-response fitting is opened.

The [frozen declaration](../manifests/experiments/hbe-01-03-halfheight-spatial-v1.json)
retains the original specimen geometry, 60 steps, 75 physical probes, numerical
gauge modulus, boundary conditions and acceptance thresholds. It reuses five
complete full-domain results and adds exactly three native half-domain solves:
compression N24, tension N16 and tension N24. Both N12–16–24 and N8–16–24
triplets must pass in both branches. Reconstructed full fields are labeled as
reflections; they are not full-domain native measurements or physical validation.

Preparation and solving have separate source-bound execution records. The full
archive contains14 modules and four declarations. No new mesher calls are needed.
Limits:60 seconds preparation;1200 seconds inclusive solve/readout allowance;
420 seconds per native call; one thread;3 GiB sampled process-family memory;
512 MiB per active run and2 GiB retained output. One final clock reading controls
both acceptance and the recorded duration. Final retained-size checking includes
the exact serialized result receipt. Receipt publication itself is outside that
clock measurement. Failure never authorizes retries or deletion of partial logs.

Independent source review and geometry/deck controls passed before execution.
Nine pure file/clock controls pass, including equality at the time limit, exact
byte-cap boundaries, leftover temporary files, symlinks and publication failures.
All11 inherited source pins are unchanged. The reviewer initially asserted the
wrong bottom-constraint representation; comparison with the original grouped
xyz constraints corrected the review, with no production-code change.

Run the archived runner with `--phase prepare` or `--phase solve`, `--root` set
to the primary checkout, and the respective release path/hash plus `--execute`.
The release binds the committed archive, interpreter and repaired FEBio runtime.
Execution results and complete raw-log review must be recorded separately.
The [executed result](../artifacts/mechanics/hbe-halfheight-spatial-execution-v1/RESULT.md)
retains a failed compression-force trend despite all three individually passing
native runs and improved displacement differences. Independent replay matches
all eight readouts. No measured response, fitting, RL or surgical-validity claim
follows from these software checks or numerical outputs.

## Next diagnostic after the failed study

Read-only specialist inspection of the saved compression endpoint found growing
plate-edge concentration with more stable core values. AcrossN8/12/16/24, maximum
rim nodal reaction divided by integrated tributary area was2.523/3.286/3.970/5.161kPa;
plate-core maxima were680.1/671.2/667.5/663.8Pa. The former is a mesh-dependent
traction proxy, not measured or exact pointwise stress. Element-averaged stress
diagnostics show a similar edge trend. These are posthoc numerical diagnostics,
not changes to the frozen acceptance result. They motivate a controlled next
test; neither raw stress peaks nor existing interior displacement probes prove
continuum accuracy. Simple polygon-area scaling has the opposite sign to the
N16→24 force change, but curved-boundary error remains unbounded.

The existing material already uses nearly incompressible hex8 three-field
elements with zero augmentation. [FEBio documentation](https://febiosoftware.github.io/febio-docs/features/features/solid_soliddomain_three-field-solid/)
supports that formulation choice; it does not certify this specimen result.
Independent review supports **preparing**, not yet running, this diagnostic:

| Variant | Change to accepted N24 half geometry | Native nodes/cells | Reconstructed nodes/cells |
|---|---|---:|---:|
| P2 | Split first slab,0≤z/H≤1/12, into two equal layers |14,216/12,096|26,655/24,192|
| I2 | Split interior slab,1/4≤z/H≤1/3, into two equal layers |14,216/12,096|26,655/24,192|
| P4 | Split first slab into four equal layers |17,770/15,552|33,763/31,104|

P2 versus I2 is the primary matched-size comparison. It measures sensitivity
to plate-layer versus interior refinement, not uniquely a rim singularity or
every bulk/formulation error. Preserve the96-sided polygon, all original x/y
coordinates and z planes, unaffected cells, material, constraints andS60 loading.
Preparation must verify shared-face conformity and parent–child volume
conservation. Exact expanded geometry limits belong in a new study wrapper;
do not relax the historical25,000-item full-geometry limits globally.

The secondary P1/P2/P4 sequence estimates only boundary-normal refinement on
the fixed radial/circumferential/bulk grid. Prespecify signed endpoint force
increments at−15% strain and retain all61-state differences. Report
`p=log2(abs(delta12/delta24))` only for same-sign, non-negligible, decreasing
increments; otherwise leave order null. Any inferred limit remains a local
fixed-grid indicator. Freeze physical regions and area/volume weights for
regional summaries; changing element rings are not fixed physical regions.

Proposed stopping limits:60seconds preparation; three native calls,420seconds
each and1,200seconds aggregate including readout; one thread;3GiB sampled family
RSS;512MiB active and2GiB new retained output; no automatic retry. Native logs
are estimated at about0.74GB before ancillary files. Thin-cell conditioning and
factorization fill make runtime uncertain, and the aggregate cap cannot promise
three full per-call allowances. The original force/motion/physics checks remain.

The [new implementation](../artifacts/mechanics/hbe-halfheight-boundary-implementation-v1/RESULT.md)
now passes18focused controls and independent geometry/XML plus runner review.
No study mesh preparation or solve has occurred. Regional energy is allocated
using element-average density and fixed physical bin volumes. Next archive the
committed source and separately release preparation and solving. This diagnostic cannot
release measured curves, establish total continuum accuracy, replace finest-mesh
time convergence, or validate patient-specific material or clinical force.

## Completed fixed-N24 boundary diagnostic

The [three native solves and independent raw replay](../artifacts/mechanics/hbe-halfheight-boundary-execution-v1/RESULT.md)
passed. Primary P2 minus I2 reaches 55.308 µN across the loading history and
0.657 µm at the fixed probes. Plate endpoint increments decrease from 46.540 to
11.280 µN, with conditional local order 2.0447. The 3.608 µN remaining indicator
applies only to plate-normal refinement at fixed transverse/bulk resolution.
It is not a continuum-error bound or a calibration release.

The earlier failed global comparison remains authoritative. Another plate split
has limited value; a separately declared N32 uniform compression check can test
the unresolved global trend before selecting the finest load-step checks. No
new mesh or native call is authorized merely by this interpretation.
