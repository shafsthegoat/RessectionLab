# Retain solved mechanics while repairing oversized reporting

All six original solves, independent readouts, and fixed mesh/time-step checks passed. The final comparison writer exceeded its 512 KiB cap because it duplicated full baseline diagnostic arrays. The original failed attempt remains unchanged.

The repair stores scalar baseline-pair results and exact references to the existing full arrays. Four generated controls pass, including production-sized 61-state/75-probe fixtures. A separate bounded recovery reuses saved readouts and the same exposed measurements, estimator and comparison. It runs no solver and does not recompute numerical screens. Both the original and repeated analytic fit calls are counted. The original physical-validation decision remains unset.
