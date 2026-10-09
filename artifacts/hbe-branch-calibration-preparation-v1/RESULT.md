# Branch-specific specimen calibration: prepared, first execution stopped

Commit `de420c3ba97d26044e7bd73d5274903702412946` implements one common positive stiffness scale fitted equally to the released compression and tension curves. Its frozen numerical references are N36/S120 compression, N24/S120 tension and N12/S120 full-domain torsion. Fixed material exponent, compressibility ratio, geometry, loading, solver and error limits remain unchanged. The original failed aggregate remains failed; the new branch qualification is conditional.

The experiment requires two actual fitted axial solves and paired primitive checks before parameter/prediction freeze and withheld torque access. Its inclusive one-hour cap covers preflight, fitting, preparation, both solves, comparisons, reveal and publication, with one CPU thread, 3 GiB sampled family RSS and 2 GiB new output. This is an established FEBio-based specimen test, not a new engine or a patient-specific mechanics claim.

Twenty-seven owner controls and 24 independent controls passed. Earlier metadata assumptions and a real floating-point operation-order defect are preserved with the corrected controls; no tolerance was widened. Mocked failure-path controls do not establish new OS resource-enforcement evidence.

Root archived 38 committed files and bound release SHA-256 `05aecaadbf55a7e0c5917712ed85e42e5adbbc6f47111d6c47b182511c4e8957`. The exact-source metadata preflight passed in 4.388249 seconds. The first actual launch then failed in its supervised child with `Interpreter identity differs`, before any measured member access, fitting or native solve. The child exited with status 1; supervision reported 0.538435 seconds and 71,745,536 bytes peak sampled group RSS. The outer harness recorded 1.043516 seconds. Failed attempt files remain immutable; no automatic retry occurred.

The runtime-launch discrepancy is the next repair. Physical agreement and held-out prediction remain unmeasured, and `physical_validation_pass` remains null. The interpretation note explains what same-specimen torque agreement could establish and which patient contact/displacement/load measurements are still missing.

All files here are compact source, software-review or execution metadata. Original biological curves, native bulk logs and the reproducible source tar remain local.
