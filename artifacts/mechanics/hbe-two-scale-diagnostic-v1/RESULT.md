# Shared stiffness cannot match both exposed calibration branches

The existing fixed-exponent, fixed-bulk/shear-ratio model has only one positive stiffness scale. On the already exposed HBE_01_03 DEVELOPMENT calibration records, independently fitting one scale for compression and one for tension yields a 5.78046 ratio. The original shared coefficient reproduces exactly within floating-point tolerance.

| Same-data diagnostic | Shared-scale baseline | Separate branch scales |
| --- | ---: | ---: |
| Compression RMSE | 0.00812920 N | 0.00446590 N |
| Tension RMSE | 0.00864191 N | 0.000843598 N |
| Compression normalized RMSE | 0.372168 | 0.204456 |
| Tension normalized RMSE | 2.842080 | 0.277436 |

This isolates a common-amplitude limitation conditional on the existing forward model. It does not identify whether constitutive shape, geometry, bonding or processing caused the mismatch. Compression retains substantial shape error. Two fitted amplitudes are a diagnostic, not an admitted mode-dependent material law; no patient or simulator property changed. The next physical-model comparison needs both a consistent numerical energy oracle and eligible independent specimen/loading measurements. Existing axial branches were already fitted, so neither is held-out validation.

Exactly two closed-form weighted least-squares coefficients use the original sixty samples, signed forces and quadrature weights. There are no offsets, sign changes, sweeps or native solves. The saved source input remains hash-identical. A separate root calculation using fifty-digit Decimal arithmetic reproduces both coefficients and residual metrics without importing the diagnostic implementation. This independently checks the arithmetic, not physical generalization.

Recorded calculation time is 0.001440 s, with 21,135,360 B self peak memory; this is not end-to-end solver cost. Source, metrics and receipt originally total 30,357 B. No archive, torsion, other donor, model weight, dependency or dataset was opened or acquired. Specimen measurements represent processed quasi-static ex-vivo force/displacement; they establish neither living-brain retraction, cutting forces, injury thresholds nor patient-specific material properties.

Original fit SHA256: `b6b23a297f81b9481a6a0c7b584c8d1d84907baf86dd4ecb62b78474caf9666a`. Full numeric records, formula, original baseline reproduction and the separate arithmetic audit are retained with hashes.
