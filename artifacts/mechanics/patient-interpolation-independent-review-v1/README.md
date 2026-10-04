# Independent saved-field interpolation review

The repaired evaluator passes 57 focused checks in 0.41 seconds: 17 independent, 26 owner and 14 existing comparison checks. All six bound source/test files remain unchanged. The source is `ce5b1639f542d1ac515ec10b8a7492f628cd704c6b011473d6c950e13772bd00`. No patient landmarks, images, meshing, solver execution or runtime installation occurred.

Three original failures are preserved against source `2caf861d…`. Two nested straight tetrahedra sharing only a vertex accepted a proven interior-overlap query 1e-13 m from that vertex as supported, because positive off-simplex coordinates fit the contact band. The portable loader also accepted a missing or malformed forward identity. Original source and test gzip, logs and receipts remain in this directory.

The narrow repair requires zero off-simplex coordinates to establish shared-simplex contact. Contact inferred only from the 1e-12 m band returns uncertainty with null displacement; wider overlap remains ambiguous. Forward SHA syntax is checked before loading artifacts. No numerical tolerance or physical model was changed. Conservative uncertainty on rounded boundary coordinates is intentional.

Independent controls additionally verify an oblique quadratic field with distinct point/vector transforms, three-cell shared-edge continuity and element-order invariance, all six statuses through complete temporary NPZ bundles, exact query binding, and coherently rehashed foreign source rejection before point location. An isolated basis-return corruption introduces three traces where each differs from the first by 0.75e-12 m but the farthest pair differs by 1.5e-12 m; the all-pair check correctly refuses. This is an aggregation control, not a physical observation. Duplicate archive members, truncated arrays, object dtype and Fortran storage reject before NumPy array loading.

This layer evaluates frozen fields; it does not validate global mesh geometry, deformation, alignment or solver evidence. Numerical-evidence bytes are bound, not promoted to solver proof. Full source/field freeze chronology and actual data release remain the comparison workflow's responsibility. The 64 MiB limit is per encoded/expanded artifact, not an aggregate-memory or performance claim. Interpolation support is not a tissue-mechanics accuracy claim.

`verification.json` binds exact sources, the initial negatives, repair and final results. Current source remains in the normal repository; only the original uncommitted failure snapshot is duplicated here for reproducibility.
