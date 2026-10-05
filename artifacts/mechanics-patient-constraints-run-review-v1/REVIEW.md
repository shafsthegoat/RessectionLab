# Independent audit of the completed mechanics controls

The saved single attempt from `ddaa7e352282d159f454562cfa091cdad0a284d3` passes independent review. No solver was rerun and no patient data were accessed.

An independent parser confirmed 27 nodes, six elements, and the initial plus four required load states in every case. The archived checker reproduced its saved results exactly. A separately reviewed polynomial interpolation and first-Piola force calculation verified actual deformation, stress, Jacobians, integrated energy, and all 72 feasible directions at all three declared finite-difference steps for both MPC cases.

| Case | Largest constraint error | Largest independently assembled reduced force | Other check |
| --- | ---: | ---: | --- |
| Affine stretch/shear | Not applicable | Not applicable | All-node position error 2.60e-13 m; signed-reaction difference 7.51e-13 N |
| Translation MPC | 2.71e-20 m | 2.70e-16 N | Whole-field translation and negligible energy |
| Nonrigid MPC | 2.00e-16 m | 2.10e-11 N | Minimum sampled Jacobian 0.9966851963 |

The largest difference between an independently assembled reduced force and any retained finite-difference derivative was 1.12e-10 N. All original convergence and virtual-work gates passed; ordinary eliminated-parent reaction outputs were not used as equilibrium evidence. Final nonrigid energy was 7.63920962668e-8 J. Near-zero translation energy differs in sign between algebraically equivalent calculations at approximately 1e-19 J; this numerical roundoff was retained without clamping.

All 37 bound inputs and all original run files remained unchanged across the audit. Nine archived source files also matched their exact Git objects. Receipts show three ordered Skyline invocations, zero retries, and one released attempt directory. The supervisor recorded 0.781202 s and a sampled peak process-group RSS of 40,108,032 bytes under the fixed 60 s / 3 GiB limits, with one numerical thread. The disclosed concurrent HBE workload prevents treating this as a performance comparison; brief memory peaks between samples may be missed.

This supports the mechanics software path for these three analytical fixtures. It does not establish the future 5 mm volume-tent observation operator, anatomical mesh accuracy, mesh convergence, patient material properties, global solution uniqueness, or clinical safety. Jacobian positivity was sampled rather than proven over every curved element.

The full [numerical receipt](numerical-verification.json), [Git verification](git-source-verification.json), and [reproducible saved-output audit](audit_saved_outputs.py) retain the evidence. The audit script reads saved files and computes checks; it contains no solver-launch path.
