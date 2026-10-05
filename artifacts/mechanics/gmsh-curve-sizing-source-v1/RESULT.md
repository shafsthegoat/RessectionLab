# Source-only investigation: discrete-curve sizing and model units

**A metre/mm conditioning control is justified; patient bottleneck causality remains unmeasured.** The source exposes repeated surface projection within curvature-enabled curve sizing, including a fixed search length expressed in model coordinates. No native call, patient read, runtime rebuild, parameter change or new candidate was made in this investigation.

The authority is the official Gmsh 4.15.2 source archive, SHA256 `be3f66f225d27ba9fa014f07e83169285da8a051b0e8ab7103d88066b39bdd3e`. Exact inspected-file hashes and locations are retained in `source-receipt.json`; line numbers below refer to that archive.

## Verified call path

- `src/mesh/meshGEdge.cpp:838` integrates `F_Lc`; lines218–234 evaluate `BGM_MeshSize` and curve derivative. The adaptive trapezoidal rule at407–434 samples midpoints until its local error criterion passes, with a minimum recursion depth greater than6 and maximum greater than25. Thus even a constant ordinary sizing integrand gets at least129 evaluations, before endpoint and other sizing work. Actual patient counts were not recorded.
- `src/mesh/BackgroundMeshTools.cpp:219–222` invokes curvature before the background field and final size clamp. At57–60, curve curvature is combined with incident-surface curvature. Lines15–25 reparameterize onto every incident face for each query; no cache is present in this path.
- `src/geo/discreteEdge.cpp:93–99` projects the curve point using the face's `closestPoint` overload. `src/geo/discreteFace.cpp:265–268` ignores the initial guess and starts with a fixed **0.1 model-unit** search half-width. Lines219–228 query an axis-aligned R-tree box and enlarge it only when no triangle is found; callback185–207 evaluates triangle distances and continues through the hits. The parametric curvature lookup at333–358 performs another spatial lookup and uses the located triangle's first stored vertex curvature.
- This is a plausible repeated-work mechanism, not proof that projection, integration depth or any particular curve caused the observed170.079 s. The saved logs have no individual curve durations, evaluation counts, triangle-visit counts or profiler samples.

`Mesh.LcIntegrationPrecision` is an established numerical option (default1e-9, `src/common/DefaultOptions.h:1297`); it controls the above integration and could reduce extra evaluations. It cannot remove the minimum sampling depth or make individual projection queries cheaper. It also changes curve integration/node placement and has no certified relationship to the2 mm physical fidelity gate. **It is held fixed in the recommended unit control.** No exposed cache/search-radius setting was found in this exact path. A size callback is invoked only after curvature (`BackgroundMeshTools.cpp:244–248`), so returning a prescribed size does not bypass this work; an identity-returning callback could count sizing calls in a separately declared diagnostic, with its own overhead.

## Coherent metre/mm hypothesis

Explicitly supply Gmsh coordinates multiplied by1000, then divide all returned nodes by1000 before the unchanged SI validators. The fixed projection box would then have a physical half-width of **0.1 mm instead of100 mm**. This could reduce candidate-triangle visits. Counts and timing improvements are unknown. `Mesh.ScalingFactor` only scales a saved mesh (`DefaultOptions.h:1606–1607`) and does not implement this internal-unit change.

| Quantity | Metre arm → millimetre arm |
|---|---|
| All input vertex coordinates | ×1000; connectivity and physical source unchanged |
| Global min/max | .003/.024 →3/24 |
| Distance/Threshold SizeMin/SizeMax | .012/.024 →12/24 |
| Distance/Threshold DistMin/DistMax | .002/.024 →2/24 |
| Explicit absolute geometry tolerance | Geometry.Tolerance1e-8 →1e-5 |
| Geometry/mesh matching tolerance, if configured | 1e-6 →.001; matching remains disabled |
| Absolute minimum edge-length tolerance | 0 →0; a nonzero value would scale×1000 |
| Returned nodes / reported curvature | nodes ÷1000; native inverse-length curvature ×1000 to SI |

Keep dimensionless controls fixed: curvature24, field Sampling100, classification angles/flags, algorithms6/1, order2/straight midsides, optimization flags, random seeds/factors, mesh-size factors, thread/count limits and integration precision1e-9. `Mesh.ToleranceReferenceElement` concerns a unit reference cell. The initial Delaunay tolerance is passed as TetGen epsilon (`meshGRegionBoundaryRecovery.cpp:149`); distance/area/volume checks normalize by bounding-box powers (`tetgenBR.cxx:9083–9138`), so that relative setting stays fixed. Zero/disabled boolean, matching and extrusion controls remain disabled. The external SI surface bounds, covering radius, midpoint tolerance, Jacobian/quality/topology/nonoverlap checks and relative-volume gate are unchanged and evaluated after restoring SI coordinates.

This is **not guaranteed numerical scale invariance**. Native fixed lengths also include1e-14 discrete-edge endpoint matching (`discreteEdge.cpp:196–201`) and another1e-6 projection start (`discreteFace.cpp:274`). Curvature code has fixed small numerical thresholds; floating-point predicates and topology can change. Geometry.Tolerance is not uniformly dimensional in every Gmsh branch: unused extrusion paths multiply it by model size, whereas the closed-line check uses it directly (`meshGEdge.cpp:790`). The proposed control must stay in the present discrete, non-CAD, non-extruded path; it cannot establish behavior for all kernels.

**Narrow next control:** one separately declared paired analytical metre/mm run of the same frozen ellipsoid triangulation, both using curvature24 and the same physical3–24 mm sizing profile. Keep integration precision fixed; compare stage times, returned counts, SI geometry/quality/fidelity outcomes, and preserve failure. Do not demand byte-identical meshes or infer patient feasibility. Root selected preparation of this control; execution remains unreleased.
