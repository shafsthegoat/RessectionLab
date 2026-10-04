# Native grid roundoff: independent core review

Nine focused analytic checks passed in 0.64 seconds against the exact source hashes in `receipt.json`. The core adapter delta has no blocking finding. The native engine, common geometry and independent evaluator match committed source `8aeb3be`; their guards were not relaxed.

The independent extent counterexample passes every voxel-center corner but exceeds 1e-6 mm at a full-cell corner, and is rejected. A separate tiny-spacing example has negligible absolute movement but excessive relative shear and is rejected by the fixed 1e-8 normalized-Gram gate. Reflected, rotated, anisotropic inputs retain original source bytes, spacing, handedness and origin. The case adapter requires an explicit option and leaves the source case unchanged.

A tiny analytic transition verifies that the actor grid, native configuration, action tip and independent committed-history checker use the same derived matrix. Source/support/affine hashes and the reconciliation method remain bound. Immutable matrix replacement and forged report/method metadata invalidate the source identity.

The first run had seven passes and one test-only mismatch: replacing the derived affine with a writable array correctly failed with an earlier ValueError, while the test expected a later RuntimeError. That test and log are retained. The final regression uses immutable replacement bytes to reach the identity guard; production source stayed unchanged.

Review also found a reporting integration mismatch: the old diagnostics still used original coordinates for physical depth and crop inversion. That owner is applying a separate minimal fix; this core receipt does not clear that unfinished reporting delta. No patient case was opened, and no patient task, preview, training or evaluation ran. There is no anatomy approval or clinical claim.
