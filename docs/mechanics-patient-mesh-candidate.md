# One prospective graded-mesh candidate

Preparation only. [The candidate helper](../scripts/mechanics_patient_mesh_candidate.py) and [fixed declaration](../manifests/experiments/resect-case4-patient-mesh-graded-v2.json) introduce no image reader, CLI, Gmsh/VTK initialization, solver or training. No patient arrays, landmarks or native mesh operations were used in this preparation. The helper requires a future separately released caller to verify exact source/runtime/case bindings, supply the previously retained native source surface and initialized native API, and enforce the declared 180-second / 3-GiB / one-numerical-thread limits. This preparation is not that execution release.

The completed v1 source, declaration and terminal failed attempt remain untouched. The new helper imports v1 geometry validators only after verifying SHA256 `d8aae815c66ae017e925c1b4ba16f1e6a64c1d2ee314153ef9889c165d64b74d`.

## Fixed candidate

There is one candidate, one generation call, no additional levels and no retry. Its existing discrete-surface Gmsh workflow requests a 12 mm near-boundary size and 24 mm interior size. A `Distance` field includes every classified discrete surface, with `Sampling=100`; `Threshold` transitions from 2 mm to 24 mm distance, with unequal global 12–24 mm size clamps. Existing algorithms, linear tet10 reference edges and numerical-thread options remain unchanged. [Gmsh documents these size-field controls](https://gmsh.info/doc/texinfo/gmsh.html#t10); sampled field distances are sizing heuristics, separate from the independent source-fidelity calculation.

Root set the new mesh-count eligibility ceiling to 6,000 nodes and 6,000 tet10 elements before any candidate execution. The earlier 2,065-node result motivates more room for geometric assessment: boundary refinement may increase counts, but changing the size field guarantees neither monotonic counts nor a fit. This prospective change does not rescue the failed v1 attempt or guarantee that this candidate will fit.

All previous geometry gates remain unchanged: closed connected manifold and matching topology, straight shared midsides, positive Jacobians, independent overlap/quality checks, full-triangle bidirectional surface bound at most 2 mm, and relative volume difference at most 3%. No smoothing, dilation, enlargement, anatomy repair or B/V landmark-driven adjustment is introduced. Actual geometry may fail any gate.

The 6,000-node cap is a **mesh-count gate, not solver admission**. At 18,000 displacement degrees of freedom, three worst-case Skyline symmetric value arrays alone require 3,888,216,000 bytes, above 3 GiB. No patient backend or solve is authorized; any geometry-pass result retains `solver_admitted=false`. Operator suitability, refinement convergence and mechanics accuracy remain separate work.

## Complete diagnostics before count rejection

`preserve_diagnostic` stores four complete arrays: float64 physical coordinates, int64 zero-based FEBio tet10 connectivity, and the original native node/element IDs. The bounded manifest records the permutation, units/frame, declared source identity, actual supplied source-array digests, complete-array shapes and hashes. It expressly confers no candidate acceptance.

Limits are 10,000 nodes, 20,000 tet10 elements and **2 MiB for this returned-candidate packet**, including NPY headers, IDs and its manifest. This is not a total-attempt-output cap: the existing 2,621,465-byte native source surface, logs and result receipts are separate. At maximum counts, payloads use `32N + 88E = 2,080,000` bytes; four 128-byte headers plus an 8,192-byte metadata reservation give 2,088,704 bytes, still below 2,097,152. Exact sizes are checked before writes, and metadata has its own bound.

Each written array must hash identically to its header plus the complete original array bytes. Partial or same-length corrupt writes cannot become a complete diagnostic. Writes use a separate partial directory; a complete marker appears only after all payload, metadata and budget checks. Failures preserve incomplete evidence without a complete marker. Existing attempts are never overwritten. A count/byte/metadata excess omits the entire packet with original counts and an explicit reason; arrays are never truncated to fit.

The helper saves diagnostics **before** checking count eligibility. A count-rejected candidate remains rejected even with a complete packet, and never enters geometry/fidelity or another candidate. Missing complete diagnostics also prevents validation continuation. Already-computed validation numbers are retained before subsequent rejection. The existing failed v1 mesh cannot be recovered retrospectively; this behavior applies only to a future reviewed/released attempt.
