# FEBio numerical patch verification — one actual attempt

All five frozen cases passed, including signed reactions, free-center motion, stress, energy, reconstructed Jacobians, force/moment balance and logged convergence residuals. Each case contains the initial state and four prescribed load states. Doubling stiffness preserved motion/J and doubled reactions, stress and energy within the predeclared tolerances.

| Case | Final minimum sampled J | Final logged energy (J) | Maximum force imbalance (N) |
| --- | ---: | ---: | ---: |
| zero | 1 | -2.94902990916e-19 | 1.06e-45 |
| translation | 1 | -9.36750677028e-20 | 3.11e-27 |
| finite_stretch | 1.045 | 6.08037845547e-05 | 1e-14 |
| shear | 1 | 5e-06 | 1.46e-16 |
| shear_double_stiffness | 1 | 1e-05 | 3.99e-16 |

The zero and translation cases retain tiny negative energy roundoff (magnitude below 3e-19 J); values were not clamped. The finite stretch has expected final J=1.045. The simple-shear cases (shear strain 0.1) have expected J=1 and final energy 5e-6 J and 1e-5 J. All five solver consoles explicitly select Skyline.

The single attempt used archived commit `0632559217b8cdff8080fdee34c5f785dbd58a45` and the accepted local FEBio 4.13 runtime. All 36 bound inputs and 35 per-case outputs were rehashed successfully. The archive includes only the committed checker, process supervisor, decks and manifests; Python imported the checker from that archive. Original source/decks, raw logs, xplt files, checker receipts and runtime identity remain unchanged.

The enforced patch cap was **56.49 s**, after counting the 3.502 s informational version probe inside the original 60 s allowance. The supervisor observed **0.524696 s**, the worker 0.429380 s and the five solver calls together 0.203194 s. Sampled peak process-group RSS was 48,775,168 bytes; brief between-sample peaks may be missed. One numerical thread, a 3 GiB cap and zero retries were used. The original result's nominal 60 s field is retained; the baseline and supervision receipt document the tighter enforced cap.

This verifies an eight-element homogeneous patch with a free center node and an arbitrary 1000 Pa reference modulus. It does not validate a human tissue property, locking or mesh convergence, nonuniform specimen mechanics, damage, contact, cutting or patient mechanics. For these affine controls, logged element energy and reconstructed energy agree; that does not establish an energy rule for nonuniform three-field specimens.

[Saved measurements](summary.json), [original result](results.json), [execution receipt](execution.json), [prospective baseline](execution-baseline.json), [archive binding](archive-binding.json). The independent saved-output audit is recorded separately; no solver or checker was rerun to write this report.
