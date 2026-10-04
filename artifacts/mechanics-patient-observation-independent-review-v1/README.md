# Independent reference-volume tent review

The repaired source passes 36 focused tests (7 independent and 29 owner) in 7.94 seconds. The five bound source/test files stayed unchanged. No patient arrays, measured destinations, mesh generation or solver execution were used.

The independent mesh is a Freudenthal cube with six tetrahedra and shared quadratic midsides, constructed without the owner's mesh fixture. A corner-centered kernel has exact octant integrals: tent mass `pi*R^3/24`, sphere-support volume `pi*R^3/6`, centroid components `3R/10`, and mixed quadratic average `R^2*(2/15+12/(15*pi))` for `x^2+3xy`. These pass the declared numerical policy. Further controls cover actual sample-budget refusal, independent input snapshots, immutable numerical storage, deterministic elimination with a nonzero constructed algebraic right side, and independent rows whose collinear centroids leave a rigid mode.

Two initial tests failed against original source `a7e75153…`: changing reported support or a nested convergence flag did not invalidate the operator. The exact original source/test gzip, log and receipt are retained. The owner repaired only report inclusion in both generation and validation hashes; the integration rule is unchanged. Final source is `d085b114b7d38ec9ebc65e624137cea1c2d66140129be7a4b44e1f260e373a25`. Reports can still be edited as Python containers, but such edits now fail validation before application or constraint preparation.

The five other independent controls passed separately before the final combined run; their log is retained. There were no additional observed test failures. All tests are analytical software checks, not synthetic training or a performance benchmark.

The h/refinement and degree-five comparisons are empirical convergence evidence, not certified integral error bounds. The geometric support interval separately assumes the mesh producer has established conformity, nonoverlap and a valid closed domain. This module's positive Jacobians and connected node graph do not establish those facts. Weighted fractions are not clinical probabilities. The fixed 5 mm volume functional approximates point-picked motion; it is not a measured volume-average response. Patient construction and solver use remain separately released.

`verification.json` binds sources, controls, negative evidence and limitations. The preserved initial source is for interpreting the reproduced failure; current source remains in the normal repository and is not duplicated here.
