# Same-specimen Ogden shape comparison

The predeclared change from α=2 to α=−8 reduced the combined force RMSE from **8.3611 to 4.2211 mN** on the same 60 exposed HBE_01_03 DEVELOPMENT observations. Compression improved strongly; tension remains poorly explained. This is a measured model-adequacy improvement on fitted data, not independent specimen validation or a patient material assignment.

| Quantity | Matched α=2 | Predeclared α=−8 |
|---|---:|---:|
| Profiled common μ, Pa | 714.3671 | 602.2739 |
| Equal-mode combined force RMSE, mN | 8.3611 | 4.2211 |
| Compression force RMSE, mN | 8.0877 | 2.7386 |
| Tension force RMSE, mN | 8.6259 | 5.3042 |
| Compression normalized RMSE | 0.37027 | 0.12538 |
| Tension normalized RMSE | 2.83681 | **1.74441** |
| Compression signed endpoint bias, mN | +21.6924 | +6.3719 |
| Tension signed endpoint bias, mN | +12.2888 | +6.2516 |

Residuals are prediction minus measurement. The positive compression endpoint bias means insufficient predicted compressive-force magnitude; the positive tension bias means excessive tensile force. Normalized RMSE divides by the branch's weighted measured RMS, not its peak: 21.84285 mN in compression and 3.04070 mN in tension. Thus the remaining tension error exceeds the measured RMS. All 30 signed values per mode, including the four negative early tension forces, remain in the calculation.

## Fixed experiment

The only shape candidates were 2 and −8, chosen before the six new native runs; no exponent optimization or follow-on sweep occurred. Each candidate used one positive common modulus from the unchanged analytical least-squares estimator, normalized trapezoidal weights with equal total weight for each loading mode, and no intercept, sign adjustment, row removal or coordinate change. These are processed cycle-average observations, not 60 independent specimens. No row-IID confidence interval or empirical physical pass threshold was declared.

The original specimen geometry and constraints stayed fixed: radius 4 mm, height 4.89159 mm, original bonded plates and inherited half-height reflection, K/μ=149/3, compression endpoint −0.73726 mm and tension endpoint +0.73601 mm. Reference solves used μ=1000 Pa. The matched force comparison uses N24/S60 for both candidates. The earlier α=2 fit on compression N36/S120 and tension N24/S120 was separately reproduced exactly at μ=715.361571139082 Pa; it was not silently replaced by the matched-grid baseline.

The fitted μ values are scaled reference responses; a new native solve at μ=602.2739 Pa was not performed. Torsion and other donors remained closed. The result does not qualify living-brain retraction, cutting, fracture, or tool-contact forces.

## Numerical evidence and limits

All six α=−8 streams passed the existing per-stream numerical checks. The retained mesh/time screen passed at 61 common states and 75 probes. At the reference μ=1000 Pa, maximum signed-force differences and probe-vector norm differences were:

| Comparison | Force change / limit, mN | Probe change / limit, μm |
|---|---:|---:|
| Compression N16→N24, S60 | 0.454776 / 1.375730 | 3.351811 / 8 |
| Tension N16→N24, S60 | 0.028709 / 0.431723 | 2.932903 / 8 |
| Compression N24, S60→S120 | 1.43×10⁻⁹ / 0.137573 | 1.42×10⁻⁸ / 0.8 |
| Tension N24, S60→S120 | 7.08×10⁻⁹ / 0.043172 | 2.83×10⁻⁸ / 0.8 |

These are fixed diagnostic mesh/time limits, not a continuum-error bound or an empirical accuracy tolerance. The α=−8 experiment does not inherit the full original v5 numerical-qualification claim. Physical validation remains null. The full numerical screen and all six numerical readouts stay at their original local paths and hashes; this compact archive retains scalar summaries and bindings.

## Cost and publication recovery

Six serial native calls consumed 448.956 s in their stage timers, with maximum sampled process-group RSS 494,927,872 B. Preparation, six readouts, intermediate screen and the failed final profile brought the sum of all original stage timers to 678.062 s; this sum excludes parent overhead and is not an independent end-to-end wall measurement. The original run's 1,476,136,960 allocated bytes remain locally retained. Sampled RSS can miss brief peaks.

The original attempt failed only when the final comparison writer exceeded its 512 KiB cap by embedding duplicate baseline probe arrays. All six solves, readouts and mesh/time screens had passed. The traceback places this failure after two candidate profiles and one original-baseline reproduction; the parent's zero counter counted completed worker reports, not attempted arithmetic.

The separate saved-readout recovery compacted those duplicated baseline arrays to scalar decisions plus exact original references. It did not rerun FEBio, re-evaluate numerical screens, reread raw measured members or change candidates. It repeated the same two profiles and one baseline reproduction: **four profile calls and two baseline reproductions cumulatively**, with no exponent optimizer. Recovery completed in 1.5864 s, its worker in 0.8917 s, maximum sampled RSS 89,718,784 B, exit zero, reaped, no fallback or remaining group members. The accepted comparison is 43,535 B. The failed first publication remains preserved.

## One next physical bottleneck

The unresolved issue is the tensile response: even after the shape change, its weighted RMSE is 1.744 times the measured RMS and the endpoint is overpredicted by 6.252 mN. These averaged curves do not identify whether the remaining discrepancy belongs to an equilibrium constitutive law or to the force reference and loading-history effects collapsed by processing. The needed discriminating evidence is a reference- and history-resolved tensile response for this specimen, with documented force-free/preload convention and paired loading/unloading timing. The present result supports neither fitting an arbitrary offset nor selecting a viscoelastic explanation. No additional sweep, data access or acquisition is part of this result.

Exact numerical results and provenance are in `execution/comparison.json`; the compact archive manifest binds the original failed attempt, recovered publication, releases, generated controls and retained local numerical references. The independent saved-result audit passed 113,059 checks, reproducing interpolation, weights, both common-modulus profiles, all predictions/residuals and metrics, baseline reproduction, and saved numerical comparisons without rerunning FEBio or importing the canonical fitter. Its first reader-only metadata-literal mismatch and corrected second audit are both retained; no experimental values or decisions changed. See `independent/RECOVERY-RESULT.txt` and `independent/recovery-audit-index.json`.
