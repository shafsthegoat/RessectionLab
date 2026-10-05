# Actual tet10 and multipoint-constraint software controls

All three fixed cases passed in the single released attempt from commit `ddaa7e352282d159f454562cfa091cdad0a284d3`. The 27-node analytic cube used six tet10 elements, ordinary `elastic-solid`, initial Poisson ratio 0.45 and an arbitrary reference modulus 1000 Pa. No patient inputs or measured specimen curves were accessed.

| Case | Outcome | Maximum original constraint error | Maximum finest feasible energy derivative |
| --- | --- | ---: | ---: |
| Mixed affine stretch/shear | Free interior motion, signed reactions, stress, energy and Jacobians passed | Not applicable | Not applicable |
| Translation through overlapping nodal averages | Entire field translated with effectively zero energy | 2.71e-20 m | 7.07e-12 N |
| Nonrigid overlapping nodal averages | Constraints, actual deformation and all virtual-work gates passed | 2.00e-16 m | 2.41e-11 N |

Both MPC cases retained all 72 feasible directions at each of the initial and four load states, evaluated at the three predeclared finite-difference steps. Every step-convergence gate passed. The minimum sampled Jacobian in the nonrigid case was 0.9966851963, and its final integrated energy was 7.63920962669e-8 J. Logged nonlinear residuals also passed their fixed gates. Ordinary parent reaction outputs were retained but never used as evidence of eliminated constraint forces.

The affine case's final integrated energy was 5.30767629087e-6 J. The translation case retained negative energy roundoff of −4.43e-20 J without clamping. All three consoles explicitly selected Skyline. Each case includes complete initial and four prescribed-state primitive outputs.

The supervisor completed in 0.781202 s under the fixed 60 s / 3 GiB / one-thread limits; sampled peak process-group RSS was 40,108,032 bytes. Brief between-sample peaks may be missed. Source/runtime verification preceded supervision; the full launcher observed 1.012472 s. The HBE experiment was running concurrently, as recorded in the immutable release. These are one-attempt timing observations, not a performance comparison. There was no retry, cutback or tolerance/material change.

All 37 bound input files, the nine-file source archive and 18 original per-case output files were rehashed successfully. The raw outputs and original receipts remain unchanged. [summary.json](summary.json) gives exact derived measurements; [archive-binding.json](archive-binding.json), [release.json](release.json), [execution](attempt-01/execution.json) and [original results](attempt-01/results.json) preserve authority and execution evidence. Independent saved-output review is recorded separately.

This establishes the chosen FEBio software path on three analytical fixtures. The nodal averages here are not the future 5 mm volume-tent observation operator. No anatomical mesh accuracy, mesh convergence, fitted human material law, patient prediction, global uniqueness or clinical safety is established. Positive Jacobians were checked at the declared sample points, not every point of arbitrary curved elements. No solver or checker was rerun to generate this report.
