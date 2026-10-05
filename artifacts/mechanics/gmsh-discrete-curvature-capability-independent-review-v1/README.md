# Saved analytical curvature-pair review

**Passed within the declared analytical scope.** No remaining reporting blocker was found. The frozen owner index (`ab6d3b39…`) binds ten small records; all seven raw outputs and thirteen source/runtime/declaration bindings match. Twelve inspected source files also match their exact member paths in the pinned 18,496,049-byte official source archive. The first reviewer locator used ambiguous filename suffixes and matched nested `CMakeLists.txt` files; its assertion failure is preserved and the locator now uses exact archive paths. This was a reviewer-control error, not a mesher failure.

The saved log and receipts agree on exactly two generation calls, one predeclared pair and no retry. The supervisor recorded 2.795098500 s, 171,065,344 bytes sampled group peak RSS and exit zero under 30 s / 1 GiB limits. Final raw output is 145,510 bytes under 16 MiB. Source inspection confirms requested numerical thread limits, sequential VTK, count/sample limits and no patient reader, solver or training path. These are sampled resource observations, not performance predictions.

Independent NumPy arithmetic on the three saved **analytical** arrays verified the 1,986-vertex / 3,968-triangle source, ellipsoid equation and source volume; both returned node/tet counts; positive corner Jacobians; exact straight reference midsides; tet volumes; boundary Euler characteristic and edge quantiles; minimum mean-ratio quality; and both-direction sampling counts. None of the mesher, VTK distance functions or shared validation functions was called during this review.

| Saved/recomputed observation | Curvature off, minimum 12 mm | Curvature 24/2π, minimum 3 mm |
| --- | ---: | ---: |
| Nodes / tet10 elements | 269 / 116 | 2,806 / 1,340 |
| Median boundary edge, independently recomputed | 10.749434 mm | 3.167301 mm |
| Source → mesh full covering bound, saved | 5.302167 mm | 0.936373 mm |
| Mesh → source full covering bound, saved | 4.934613 mm | 0.919818 mm |

The nearest-triangle distances above were checked against source logic and bound output records; they were **not independently recomputed with VTK**. The source triangles themselves approximate the exact ellipsoid, with independently recomputed volume error of 0.400938%. The reported distances therefore concern those triangles, not exact smooth-ellipsoid Hausdorff error.

The pinned [official Gmsh 4.15.2 source archive](https://gmsh.info/src/gmsh-4.15.2-source.tgz) contains the relevant path: `GModelParametrize.cpp` computes discrete curvatures during classification; `discreteFace.cpp` carries them into the parametrization and exposes a curvature query; `BackgroundMeshTools.cpp` includes curvature-derived size in its minimum before global clamps. This supports the availability of the capability. It is not a proof that estimated curvature equals analytic curvature, nor proof of binary reproducibility from that source.

The two profiles changed **both curvature and the permitted minimum size**. The evidence supports the observed joint profile response on this one analytical input, not an isolated curvature/floor effect, a general accuracy guarantee or a patient node/memory estimate. Both profiles' 64 nonzero curvature-query values cover only the first 32 classified nodes per patch, establishing limited availability rather than surface-wide coverage or estimator accuracy. No patient arrays, mesh regeneration, native Gmsh/VTK call, solver or training occurred in this review. Any patient candidate remains a separate declaration and release.
