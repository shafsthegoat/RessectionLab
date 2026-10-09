# HBE_01_03 v5 deck adapter preparation

This separately versioned, **non-executable** slice specifies twelve fresh
same-endpoint reference schedules across compression and tension. A pure
adapter checks a supplied old-domain FEBio Skyline deck, changes only the
declared time controls and both zero/axial load-controller point arrays,
checks the resulting Skyline XML, then applies the existing Accelerate backend
transformation. Full-native and reconstructed-half boundary factors, exact
endpoint float representations, and S60/S120 even-state coordinates are
explicit. The source specimen geometry, material law and previous failures
remain unchanged.

The [independent review](source-review.txt) accepted this preparation after
correcting an initial manifest omission of the zero-controller change. All
243 focused v3–v5 controls pass in a fresh checkout. Fixture tests are software
checks; the actual twelve source decks still have null hash bindings. A
generated deck missing an entire top-node boundary set is not detected without
binding the true mesh and source deck, so full boundary-topology authentication
is mandatory before any solve. Native primitive reconstruction, numerical
qualification, resource release, fitting and held-out torsion access remain
closed. No solver, mesher, measured response or patient data was opened here.

The [scientific scope review](scientific-scope-review.md) identifies this as
specimen numerical/material groundwork. It does not validate live patient
retraction, deformation around a cavity, cutting forces or surgical injury.
The separate RESECT patient-anatomy displacement track needs an accepted mesh;
even there, brain-shift landmarks can validate displacement interpolation only.

Source: [manifest](../../manifests/experiments/hbe-01-03-branch-calibration-v5.json),
[adapter](../../scripts/mechanics_hbe_branch_calibration_v5.py), and
[tests](../../tests/test_mechanics_hbe_branch_calibration_v5.py).
