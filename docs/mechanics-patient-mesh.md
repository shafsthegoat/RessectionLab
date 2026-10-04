# Prepared Case4 estimated-domain mesh slice

Preparation only: no patient arrays, landmark coordinates, native surface extraction, Gmsh generation or mechanics solve have been executed for this slice. The implementation is [mechanics_patient_mesh.py](../scripts/mechanics_patient_mesh.py), with the fixed [declaration](../manifests/experiments/resect-case4-patient-mesh-v1.json). Existing analytical software controls are separate from patient validation.

The source is the unchanged native Case4 main-v1 mask, SHA256 `7902cfbcb5fad15144181c06883bac7f4800e842f7ff05a540f56bac4eb38759`. Existing independent QC reports 1,186,021 positive voxels, one component, no image-boundary contact, exact native geometry and exact reconstruction of the upstream extraction. Root's fixed-slice review observed inferior/cerebellar structures outside the estimated envelope. It is an estimated computational domain, not whole-brain coverage, a pial surface or anatomical ground truth. The related SDT hash is `21545d1b70fcce791ddb9d6144c8534cd4c4d2d2616c21e99bfd969781b11195`; this utility does not use that map as a distance-to-anatomy or uncertainty measure.

The header receipt supplies the 256×256×192 native grid, nominal 1 mm sampling and oblique sform. Coordinates remain in original T1 RAS. Apply the full affine to extracted voxel coordinates and divide millimeters by 1000 once. No before-US transform, motion observation, fitting destination or validation landmark enters meshing. The release's baseline flag means accepted **provisional research diagnostic**, preserving false anatomical-registration and clinical-validation claims; it does not upgrade the baseline to anatomical truth.

## Existing libraries and fixed mesh sizes

Use scikit-image's [Lewiner marching cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes) at level 0.5, native step size 1, without smoothing, dilation, hole filling or component replacement. An image-edge contact fails instead of adding exterior padding. A global face-winding reversal is permitted solely to express the same surface with outward orientation.

The already acquired private Gmsh 4.15.2 provides the [tutorial 13 discrete-surface workflow](https://gmsh.info/doc/texinfo/gmsh.html#t13): classify into parametrizable patches, create discrete geometry, create one enclosed volume and generate tetrahedra. Retaining every dense source-surface vertex in a volume mesh would defeat the small Skyline budget; Gmsh remeshing is therefore assessed against the unchanged source surface. No separate simplifier or new physics engine is introduced. Gmsh remains a separate research utility with its existing license scope.

| Level | Target edge length | Maximum tet10 nodes | Maximum elements | Worst-case symmetric value array |
| --- | ---: | ---: | ---: | ---: |
| Coarse | 24 mm | 2,000 | 2,000 | 144,024,000 bytes |
| Medium | 20 mm | 3,000 | 4,000 | 324,036,000 bytes |
| Fine | 16 mm | 4,500 | 6,000 | 729,054,000 bytes |

These are prospective targets and rejection caps, not predicted counts or verified converged resolutions. Use ordinary `TET10G8`/`elastic-solid` downstream, preserving the existing finite-compressibility model. `Mesh.ElementOrder=2` and `Mesh.SecondOrderLinear=1` produce genuine ten-node displacement elements with straight reference edges. Reference coordinates from Gmsh establish its local permutation into FEBio order: four corners, then edges 12, 23, 31, 14, 24, 34. Retain and verify actual returned midside nodes; do not label a tet4 mesh tet10.

The fine cap gives at most 13,500 unreduced displacement degrees of freedom. Three worst-case dense-profile symmetric value arrays need 2,187,162,000 bytes, leaving approximately 1.03 GB beneath 3 GiB for other allocations. This calculation omits additional solver storage and factorization work. Skyline feasibility and runtime remain unmeasured; neither the previous small fixtures nor an unvalidated alternate backend clears a patient solve.

## Geometry and approximation gates

Both source and returned boundary require a single connected, consistently oriented closed 2-manifold: two incident faces per edge, one connected incident fan per vertex, no duplicate/degenerate faces, and positive enclosed volume. Preserve Euler characteristic. Do not change topology to make a mesher succeed.

The surface approximation gate is **2 mm in both directions**, relative to the estimated native mask isosurface. It is an engineering discretization limit rather than anatomical accuracy. Existing [VTK nearest-triangle distance](https://vtk.org/doc/nightly/html/classvtkImplicitPolyDataDistance.html) evaluates samples. For each triangle with largest edge length `L`, choose `n=ceil(L/1 mm)` and include its full barycentric lattice. That lattice partitions the whole triangle into `n²` subtriangles with diameter at most `L/n`. Every interior or edge point is within that distance of a sample. Since distance to a fixed surface is 1-Lipschitz, the greatest sample distance plus the covering radius bounds distance over the entire triangle. Repeat in the reverse direction. Vertex-only agreement cannot pass this gate. Nearest-point arithmetic is finite precision; this is a geometric covering bound using the library's numerical distances, not an interval-arithmetic proof.

At most four million samples per direction per level are allowed, computed in bounded chunks. A cap failure does not permit a sparser check. Require at most 3% volume difference from the original isosurface as an additional check, without interpreting voxel or surface volume as anatomical truth.

Tet10 validation checks indices, shared midside identity, midpoint error at most 1e-12 m, positive corner/reference determinants, actual returned geometry at eight Gauss points plus vertices, edge midpoints and centroid, mean-ratio quality at least 0.05, closed boundary and consistent volume integral. An independent separating-axis test rejects positive-volume overlap between candidate tetrahedra; mere face/edge/vertex contact is permitted within a fixed 1e-10 m numerical tolerance. Candidate pairs are capped at two million. No skull clamps or other boundary conditions are added.

## Single released attempt and downstream handoff

Root must commit/archive the four-file closure, bind exact mask/QC/runtime/source hashes and one fresh attempt directory, and separately release execution. The two prerequisite receipts are independent mask QC and the provisional baseline diagnostic. The existing process-group supervisor limits the complete worker to 180 seconds, 3 GiB sampled RSS and one numerical thread. Run the three fixed levels once in order and stop at the first failure. Preserve bounded raw meshes before quality evaluation, partial records and all unexecuted statuses. No retries, alternate resolutions, repair operations or solver calls are available. Memory peaks between supervisor samples may be missed.

Each level writes a native `.msh` and an `.npz` containing `nodes_m: float64[N,3]` and `tet10_indices: int64[E,10]`, zero based, in native T1 RAS meters and FEBio local node order. The separate boundary file represents the same straight reference surface. Source, units, grid, runtime, mesh-level and fidelity records accompany the output.

The geometry agent's fixed 5 mm volume-tent operator consumes this contract. These coarse target sizes may not resolve its compact support adequately; bounded operator integration and mesh-refinement checks must decide that independently. The mask must never be enlarged, smoothed or changed to contain a source or held-out landmark. A failed containment or fidelity result remains a negative result.

## Executed software control and retained failure

The independently reviewed source passed 37 analytical/mocked checks. One separately released native-library cube control then ran in 1.537 seconds with 157.4 MB sampled process-group memory. Its 510-node, 249-element tet10 mesh passed topology, reference Jacobian, overlap and both whole-surface distance bounds (1.600/1.577 mm). It failed the unchanged 3% volume gate with 8.499% volume loss. The result demonstrates a working library path and a useful rejection; it is not a passing fidelity control. No retry, patient meshing or mechanics solve occurred. See `artifacts/mechanics-patient-mesh-independent-review-v1/receipt.json` and the preserved native output index.
