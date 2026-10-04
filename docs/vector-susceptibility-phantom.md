# Physical-vector susceptibility prototype: bounded feasibility result

Executed locally October 4, 2026. An independently written experimental solver now represents the two native image affines and their separate physical PE vectors. On a smooth 0.8-degree rotated synthetic pair, image relative RMSE fell from **9.68% to 1.07%**, with **0.315 mm field RMSE**. A checked exact-identity branch avoids the previous numerical zero-gradient failure. Thirteen new tests and five final full-size phantom cases pass their declared numerical checks.

This is a small feasibility experiment, **not an accepted patient correction method**. Production preprocessing gates are unchanged. No patient, FSL, motion, eddy-current, outlier or tract computation ran. The nonzero optimizations reach the fixed 40-iteration limit; no convergence proof, clinical calibration or equivalence to an established correction pipeline is claimed.

## Mathematical model and independent implementation

The primary [Macdonald–Ruthotto formulation, equations 1–6](https://arxiv.org/pdf/1607.00531), describes mass-preserving EPI correction, spatial smoothness and positive-Jacobian requirements; it also explains why unrestricted similarity fitting is ill-posed. The earlier [Andersson–Skare–Ashburner paper](https://pubmed.ncbi.nlm.nih.gov/14568458/) motivates estimating distortion from reversed acquisitions. Those papers do not validate this new prototype.

For this experiment, let `x` denote physical millimeters and `p_i` each image's physical unit PE vector. A shared displacement scalar `d(x)` gives

```text
T_i(x) = x + d(x) p_i
J_i(x) = det(I + p_i grad(d)^T) = 1 + p_i · grad(d)
C_i(x) = I_i(T_i(x)) J_i(x)
```

Applying the single-image physical formula separately to nonparallel `p_1` and `p_2` is **our experimental extension**. It assumes the same static anatomy, susceptibility field, contrast and displacement scaling in both acquisitions. Equal physical displacement scaling is an assumption of these phantoms, not something inferred from image similarity. Different readouts/voxel scales would require explicit per-image scaling. Real head motion and time-varying susceptibility require additional models.

The independently written `scripts/experimental_vector_susceptibility.py` minimizes a squared difference between `C_1` and `C_2`, plus gradient-energy smoothness and a nonnegative Jacobian penalty. It uses a 12 × 12 × 8 control grid, trilinear field interpolation, native 3D image sampling and standard Torch L-BFGS/autograd. It contains no copied or imported GPL solver implementation and is not integrated into the app.

The field is bounded to ±12 mm through the parameterization. Training support is a fixed geometric intersection with enough margin for that bound; it does not use truth or shrink dynamically around errors. A finite penalty continuation permits recovery from a folded trial step; final grid Jacobians must exceed 0.2. This finite-grid check does not prove continuous global diffeomorphism between samples. Images are scaled together without subtracting an intensity offset, preserving the density equation.

Physical units, coordinate transforms, nonparallel Jacobians and autograd derivatives have independent linear-function, matrix-determinant and finite-difference tests. PE direction is attached to each stored native image's voxel axes, then transported through its actual affine. The second image is sampled directly in native space; no preliminary PA regridding or header replacement is required by the optimizer.

## Exact identity: a proof for a narrow mathematical case

The analytic branch requires finite, nonconstant arrays that are **exactly equal**, exactly equal affines, and valid broadly opposite PE directions. At `d=0`, it independently computes a zero displacement gradient, Jacobians of one, identical corrected arrays and zero data, smoothness and barrier objectives. Every objective component is nonnegative, so zero is a global minimum for this stated objective. Tests independently check that the numerical objective at zero agrees and that no numerical optimizer is called.

Almost-equal/noisy arrays, unequal geometry, nonfinite values, constant images and invalid directions cannot enter this branch. This is not a claim that identical acquisitions prove undistorted anatomy: duplicated input files or unidentifiable anatomy are possible. Acquisition QC remains separate. The original unmodified PyHySCO release still logs its zero-gradient NaN when invoked directly; its failed result in `pyhysco-phantom.md` remains retained. The new branch establishes its own solution before optimization and does not ignore or relabel that NaN.

## Frozen experiment and measured results

The same analytic generator, oblique 96 × 96 × 60 grid, 2.5 mm spacing, smooth field and 5%-of-truth-peak evaluation support from `pyhysco-phantom.md` were retained. Truth is held by the evaluator; workers receive only distorted images, affines/PE metadata and frozen optimizer parameters. The worker manifest contains no true field, amplitude, evaluation mask or truth error threshold, and truth image files are written only after each worker exits.

The final run uses float64 model/optimization arithmetic, two Torch CPU threads, 40 maximum L-BFGS iterations, at most 60 nominal function evaluations, gradient regularization `300/256²` after common input normalization, and Jacobian weight 0.001. Each worker has a 55-second watchdog and 4 GiB sampled-RSS ceiling. Three experimental revisions consumed approximately **92 seconds of solver wall time total**; all generated images and optional runtimes stayed local.

Final run: `outputs/vector_susceptibility/v3-native-pe`. Error figures are over the declared evaluation support, not low-signal background.

| Case | Mean image relative RMSE before → after | Field RMSE | Wall time | Result |
|---|---:|---:|---:|---|
| Exact zero | 2.49e−8 → 2.49e−8 | 0 mm | 1.20 s | Independent zero-objective identity certificate |
| Same grid, smooth field | 9.6788% → 1.0733% | 0.3137 mm | 7.42 s | Numerical checks pass |
| Native 0.8° pair | 9.6792% → 1.0682% | 0.3154 mm | 7.41 s | Actual nonparallel vectors represented |
| Negative scalar field | 9.6791% → 1.0799% | 0.3223 mm | 7.39 s | Sign case passes |
| Entire physical coordinate frame reoriented | 9.6792% → 1.0682% | 0.3154 mm | 7.39 s | Maximum field difference 0.000570 mm; preset 0.02 mm bound passes |

The final five cases used approximately 30.8 seconds total worker wall time and less than 500 MiB sampled peak RSS. A 0.1-second RSS sampler can miss brief peaks. Timings were measured during other development activity and are not controlled throughput benchmarks.

The prior PyHySCO same-grid baseline yielded 0.4032% image RMSE and 0.3948 mm field RMSE in 9.34 seconds. This prototype's image recovery is worse while its field RMSE is lower on this one smooth phantom. That tradeoff, different precision/discretizations and limited iteration budgets prevent a superiority or equivalence claim. Both improve on the zero-field baseline of 2.2420 mm. The established baseline artifact's exact hash is retained with the comparison.

## Coordinate failure found and repaired

All preliminary truth-error checks passed, but the first end-to-end coordinate-invariance test failed: changing only the global physical coordinate frame changed the optimized field by as much as **0.2498 mm**. Switching from float32 to float64 alone still left **0.1044 mm**. This was a separate engineering failure even though images looked improved.

Inspection identified inconsistent numerical geometry around interpolation knots. The reference image was being taken through an unnecessary world-to-voxel round trip, adding tiny signed offsets at integer voxel locations. Its physical PE vector was also computed before NIfTI affine serialization, while sampling used the rounded stored header. Trilinear derivatives have slope changes at voxel knots, and the limited-iteration nonconvex solve amplified these differences.

The final revision uses exact integer indices when an image is its own reference, full relative affines for other images, and physical PE directions derived from the authoritative stored native headers. It does not snap a second image into the first image's grid or discard a transverse PE component. A regression test checks the reference's exact sampling indices. With the same hyperparameters, truth and thresholds, the final reorientation difference is **0.000570 mm**, passing the preset bound. This is evidence that the repair resolves the observed test, not a guarantee for every acquisition or interpolation condition.

All three runs, failed coordinate checks, frozen configs and source hashes remain under `artifacts/vector-susceptibility-v1/`. Ignored local output folders also retain snapshots of the first two source revisions. The final generated-file hash manifest covers raw images, affine/PE manifests, estimates, corrected images and evaluation-only truth. Numerical improvement never erased the earlier failures.

## Reproduce and remaining work

```sh
.venv/bin/python scripts/experimental_vector_susceptibility.py \
  --output outputs/vector_susceptibility/new-run
.venv/bin/python -m pytest \
  tests/test_vector_susceptibility.py tests/test_pyhysco_phantom.py -q
```

The directory must be new. No additional package installation or GPL runtime is needed for this prototype; the existing NumPy, nibabel, SciPy and Torch dependencies suffice. The combined focused suite has **27 passing tests**. None of these scripts are app entry points or alter patient preprocessing acceptance.

This resolves the narrow identical-input numerical failure and establishes a useful direction-aware phantom prototype. Remaining research includes convergence/sensitivity checks, sharper/noisier fields, unequal acquisition scaling, signal dropout, different anatomy and independent reference acquisitions. Patient execution still requires a validated complete correction chain with motion, eddy-current, gradient-rotation and outlier provenance. No current result licenses clinical probabilities, lesion accuracy claims or accepted tract evidence.
