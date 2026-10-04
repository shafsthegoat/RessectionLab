# HBE_01_03: one-scale specimen mechanics baseline

Prospective, 2026-10-04; no response curves opened and no specimen solver run. The [experiment declaration](../manifests/experiments/hbe-01-03-mechanics-poc-v1.json) is the executable-work specification, pending root review, source freeze and separate release. The [role manifest](../manifests/experiments/hbe-01-03-specimen-roles-v1.json) fixes this development specimen without outcome-based replacement.

## Question and observations

Can one stiffness scale calibrated to processed third-cycle compression and tension predict the same specimen's previously unopened low-amplitude third-cycle torsion? Only torsion is held-out validation; axial errors describe calibration fit. All first-cycle, higher-amplitude and other-specimen curves remain sealed. This is a restricted constitutive baseline, with no model sweep or patient-property claim.

The [creator experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC10511383/) bonded the tissue to both plates. Its released curves are processed quasi-static approximations; torsion followed axial testing and relaxation, so a shared cycle suffix does not establish identical loading history. The [released geometry](https://zenodo.org/records/8095559) gives radius 4 mm and test-inferred height 4.89159 mm. Geometry/force/torque uncertainty is unavailable. Do not fit height, offset, signs, glue compliance or a pressure parameter.

## One fixed mechanics problem

Use SI units, FEBio 4.13 at the exact source revision in the [runtime declaration](../artifacts/febio-runtime-investigation-v1/prospective-runtime.json), a three-field hex8 solid and `laugon=0`. Fix the Ogden exponent at 2: `c1=2μ`, other coefficients zero, `pressure_model=1`, `K=149μ/3`. The energy is

`W = μ/2 (Ī₁ − 3) + K/4 (J² − 1 − 2 ln J)`.

This follows the pinned [Ogden implementation](https://raw.githubusercontent.com/febiosoftware/FEBio/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioMech/FEOgdenMaterial.cpp) and [volumetric convention](https://raw.githubusercontent.com/febiosoftware/FEBio/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioMech/FEUncoupledMaterial.h). It is an uncoupled neo-Hookean special case, not reproduction of the paper's free-exponent fit. Fixed initial Poisson ratio .49 is an assumption; no measured volume change identifies it here.

The cylinder spans z=0 to H. Both faces use prescribed x/y/z displacements, including prescribed zeros so reactions can be recovered. The bottom is stationary. Axial loading fixes top x/y and prescribes z. Torsion rotates every top node exactly around the z axis, fixes top z, and leaves the side traction-free. There are no rigid bodies, contact, gravity, inertia or glue elements. Use exact sine/cosine displacement values at every declared step; a linear interpolation between final rotated coordinates is not the prescribed rotation trajectory.

Four branches start separately from the same rest state: axial strain −.15/+.15 and torsion nominal shear −.15/+.15 (`θ=γH/R`). Use 60 equal pseudo-time steps; the fine-mesh repeat uses 120. Pseudo-time is not physical loading rate. Coordinate/units/sign ambiguity or measured inputs outside this fixed loading range stop the comparison; do not extrapolate or change endpoints after inspecting torque.

## Established meshing and numerical gates

Use the separately pinned Gmsh 4.15.2 arm64 utility. Its [transfinite and extrusion interfaces](https://gmsh.info/doc/texinfo/gmsh.html) produce a five-block disk: central square corners (±R/2,±R/2), four surrounding quadrilateral sectors terminating on quarter-circle arcs, followed by a recombined axial extrusion. There are N edge/arc cells, N/2 radial-connector cells and N/2 axial layers. N=4,8,12 gives nominal 96,768,2592 hex8 cells. These are prospective counts; generation and Skyline cost remain unmeasured. The separate GPL runtime is not bundled into the desktop app.

Before solving, verify shared interfaces, explicit top/bottom/side tags, unique node IDs, hex8-only cells and the Gmsh-to-FEBio ordering. Independently compute element Jacobians at corners, center and eight Gauss points. Require positive determinants and minimum scaled Jacobian >.1. At the finest mesh require volume discrepancy ≤.5% from πR²H and maximum radial boundary sag ≤.005R. Keep measured mesh quality at every level; do not claim ideal CAD geometry is the mesh actually solved.

First pass the separate zero/rigid/finite-strain/shear analytical controls. The specimen uses static symmetric Skyline, BFGS, maximum 15 reformations and relative displacement/energy/residual tolerances 1e−8; residual floor is `(1e−10 μR²)²` N². All tolerances scale consistently in the modulus check. No automatic time cutback, formulation switch or failed-run retry is allowed.

Reference μ=1000 Pa is a numerical gauge, not a measured tissue property. Define L₀=R, F₀=μR², T₀=μR³ and E₀=μR²H. The declaration fixes absolute floors and relative limits: finest/medium reactions ≤2%, load-step reactions ≤.2%; displacement changes ≤.002R and .0002R respectively at 75 fixed interior probes. Require refinement differences to decrease unless already below their absolute floor. Record every violation. These are engineering numerical error gates, unrelated to an unreported instrument precision or clinical tolerance.

Repeat fine-mesh compression and positive torsion at 2μ: displacements must agree and force/torque must double within the declared numerical tolerance. A finite K, positive J and mesh convergence do not by themselves prove global material stability or general locking immunity.

## Calibration, freeze and independent readout

After runtime/mesh/analytical gates, open only the two calibration members and validate the source schema. Retain every finite, unique input coordinate and original signed response; do not subtract a first sample or fit an intercept. Interpolate computed reaction curves linearly at measured coordinates within their solved range. Each axial mode receives total weight one half, distributed by normalized trapezoidal coordinate weights. With reference predictions gᵢ and measured forces yᵢ, fit `a=Σwᵢgᵢyᵢ / Σwᵢgᵢ²`, `μ̂=a μref`. Reject an unidentified denominator or a nonpositive scale; no clipping or nonlinear search.

Confirm the calibrated scale in two fine-mesh axial reruns. Save immutable calibration inputs, weights, predictions, residuals, μ̂, mesh/deck/runtime/source hashes, numerical receipts and predicted torsion curves. Only then may the two held-out torsion responses be read. Held-out values cannot choose the mesh, model, load schedule, offsets, fit, sign or stopping rule. A failed torsion result remains a failure of this baseline; a later model needs a new declaration and cannot reuse this mode as unseen validation.

The independent reader consumes rest/current coordinates, node IDs, prescribed DOFs, raw nodal reactions and solved load values. Verify the raw-reaction sign against the analytical extension/shear controls and an independent known-force-pair moment calculation first: canonical externally applied tissue force/torque is the negative of FEBio's raw support reaction sum. Compute Fz and Tz using current lever arms about the declared axis; preserve raw values as well. Do not fit a sign from measured response. Sum all boundary force/moment about a common origin, reconstruct deformation Jacobians and compare integrated work with the three-field energy (volumetric energy at the cell-averaged J, with zero augmentation).

Report calibration force RMSE/MAE in N and validation torque RMSE/MAE in Nm, each signed branch separately. Also report signed endpoint bias, branch-balanced RMSE divided by each branch's observed RMS, every denominator, and observed/predicted odd-symmetry discrepancies. A near-zero observed RMS yields null normalized error with a reason. Processed curve rows are not independent replicates: no row-bootstrap confidence interval or patient-accuracy claim. There is no defensible empirical pass tolerance yet; measured agreement is descriptive with unknown metrology, even when numerical gates pass.

## Bounds and claims

Maximum 20 specimen solves: 12 for four branches on three meshes, four fine load-step repeats, two doubled-modulus controls and two calibrated axial confirmations. Cap each solve at 90 s, all specimen work at 900 s and sampled process-family RSS at 3 GiB, with one thread, ≤256 MiB active-run output and ≤1 GiB total generated output. A metadata-only log-size estimate is about 790 MiB before headers/IDs; actual storage remains guarded, with lossless archival compression and decompressed hashes. Gmsh acquisition/generation has separate declared caps. Keep immutable raw primitives under ignored `outputs/mechanics/hbe-01-03-poc-v1`, with hashes; commit compact receipts and source rather than large solver logs. Stop and retain partial receipts on any cap, nonconvergence, nonpositive Jacobian, missing output, hash drift or gate failure; no automatic budget expansion or replacement specimen.

This first baseline cannot establish donor transfer, patient retraction/cutting forces or a real-anatomy displacement result. The latter remains a separate obligation under the [validation contract](tissue-mechanics-validation.md). Mechanics-based search/imitation/RL comparisons wait for the relevant task's evidence gate; hidden loads/material truth remain outside deployment observations.
