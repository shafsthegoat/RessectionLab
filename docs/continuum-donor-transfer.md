# Donor-held-out specimen response comparison

This experiment compares the source-voxel continuum with the unchanged default
lattice against acquired human brain force measurements. It separates the
already exposed development donor from one prospectively selected evaluation
donor. It does not establish patient mechanics, tool response, injury, or
viscoelastic time constants.

The result is mixed: the calibrated continuum improves on the default lattice
in both branches, while the zero-force baseline wins in tension. No physical
acceptance or physiological RL gate is promoted.

## Held-out result

Only the original rows within the frozen 2% strain window are scored: four
compression and four tension points from 30 rows per source branch. Signed
released values and zeros are retained. No parameters, mesh, offset, point
window or stopping rule were changed after this reveal.

| Finest-mesh force RMSE (mN) | Compression | Tension |
| --- | ---: | ---: |
| Development-calibrated continuum | **0.355** | 0.590 |
| Unchanged default lattice | 4.992 | 5.601 |
| Lattice with matched nominal strain | 4.409 | 5.025 |
| Zero-force baseline | 0.882 | **0.485** |

Continuum maximum absolute error is 0.574 mN in compression and 0.853 mN in
tension; signed farthest-point error is +0.574 and +0.128 mN, respectively.
Its tension RMSE exceeds zero force at every prespecified mesh. Some positive
released displacements have negative measured force; those values were kept,
without inventing a preload or force-zero correction. This experiment cannot
determine whether the mismatch comes from material behavior, processing,
reference-zero convention, fixture or other unmeasured factors.

The lattice's force slope rises from 0.0460 to 0.1387 N/mm as resolution
increases, while the calibrated continuum changes from 0.01506 to
0.01392 N/mm. The default lattice's known grid dependence remains visible.
An advantage against that baseline alone would be insufficient physical
validation.

## Source and prospective selection

[Hinrichsen et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10511383/)
tested cylindrical postmortem human brain specimens between bonded plates.
The [CC BY 4.0 source release](https://zenodo.org/records/8095559) provides
filtered, load/unload-averaged quasi-static force curves, rather than raw time
histories. Axial displacement is in metres and force in newtons. The nominal
radius is 4 mm; specimen height was determined by the creator from the test
data. No independent geometry uncertainty or clinical error tolerance is
supplied.

Before any new response access, the existing metadata inventory selected the
first eligible motor-cortex specimen from the next numeric donor:
**HBE_02_03**, height 3.56882 mm. HBE_01 remains DEVELOPMENT; HBE_02 is
MEASUREMENT_EVAL, including its siblings and derivatives. Only HBE_02_03's
third-cycle compression and tension are admitted. These are correlated
branches from one specimen, not two independent validation units. Other
specimens, torsion and first-cycle response members remain unopened.

The existing whole-cohort published parameter estimate was excluded from this
test because it includes the evaluation donor. Instead, exactly one positive
continuum shear scale was fitted to four compression and four tension points
already exposed in HBE_01_03's historical development receipt. Each original
row has equal weight. The fit uses signed released zeros without preload,
offset or sign adjustment, and only |displacement/height| ≤ 0.02.

## Models and numerical admission

The continuum uses the unchanged fully integrated source-voxel hex8 matrices,
ν=0.45, no Maxwell branches, and zero lateral displacement on both physical
end faces. Remaining faces have zero traction. Its fitted **229.26226449 Pa**
is an effective scale conditional on this fixture, geometry and compressibility,
not an independently measured intrinsic or patient shear modulus.

The default lattice retains its original coefficients, divergence penalty and
weak foundation. Its first/last source-centre planes are prescribed, with a
shorter span than the physical plate separation. Both equal physical
displacement and a prespecified equal-nominal-strain sensitivity are reported.
The latter multiplies imposed displacement by `(height − dz) / height`.
Both models use equilibrium stiffness, without drag, handles or strain
projection. An error difference combines calibration, constitutive and boundary
differences; it cannot isolate the benefit of a particular discretization.

The [first frozen protocol](../manifests/experiments/continuum-donor-transfer-v1.json)
failed its 5% force-slope stability limit: the final change was 7.03% in
development and 7.12% in evaluation. No new donor response was read and no
archive was acquired in that attempt. Its numerical results are preserved.

The separately committed
[numerical refinement](../manifests/experiments/continuum-donor-transfer-v2.json)
uses cross-section counts 12/20/28 and height counts 4/7/10. It retains the
same 5% limit and additionally checks the slope range across the final two
new meshes and the original finest mesh. This guards against coincidentally
similar voxelized cylinder areas. Before response access:

| Check | Development | Evaluation |
| --- | ---: | ---: |
| Final consecutive force-slope change | 3.04% | 2.84% |
| Range including original finest mesh | 4.95% | 4.63% |
| Finest nominal-cylinder volume error | 0.0403% | 0.0403% |

The maximum element-gradient norm at the declared strain window was 0.07864,
below the unchanged 0.2 model-domain cap. Maximum free-force residual was
2.94×10⁻¹⁹ N. Preparation took 31.57 s and peak process RSS was 1.34 GB.
This is a limited three-mesh stability check near its tolerance, not proof of
asymptotic convergence. Static specimen fixtures exceed the runtime local
source-cell cap; the live solver cap is unchanged.

The [committed parameter and prediction freeze](../manifests/experiments/continuum-donor-transfer-v2-freeze.json)
binds protocol, code, preparation hashes and every resolution's force slope.
Scoring checks these bindings before opening exactly two declared ZIP members.
A durable access ledger records each attempt before member decoding. The
13 MB source archive is verified against the creator MD5 and historical
SHA256, following a same-size/content duplicate search. No full archive is
extracted and no other donor response is decoded.

Independent code reconstructed Q1 gradients, reactions and strain energy from
all four finest numerical bases for both attempts, without importing the
engine's element operators. It reproduced the development fit and passed 13
isolated scoring/access controls using synthetic data. The 55 targeted
continuum/qualification regression tests passed. Numerical controls and
synthetic scoring tests are kept separate from the acquired measurements.

## Reproduction and claim limits

All acquired data, numerical bases, results and independent review receipts
live in ignored
`testcases/outputs/simulator-qualification-20261029/donor-transfer-v1/` and
`testcases/downloaded-datasets/HBE-8095559/`. The local README provides exact
commands and the source citation. Run `prepare` before `score`; scoring requires
the committed parameter freeze and matching source archive. Preparing a new
result does not silently overwrite the frozen attempt.

The result is a descriptive donor-held-out material response comparison on one
motor-cortex specimen. Missing metrology and application tolerance prevent an
automatic physical acceptance claim. It says nothing about surgical contact,
relaxation, cutting, cavity collapse, acquired patient brain shift, vascular
physiology or neurological outcome. Patient mechanics and physiological RL
remain on hold. The existing replayable retraction/resection branches retain
their original explicitly assumed material scenarios.
