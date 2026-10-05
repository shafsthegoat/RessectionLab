# Native Gmsh curvature sizing works on this discrete analytical surface

The single declared analytical pair completed with exactly two generation calls and no retry. Both used the same triangulated ellipsoid (35/20/10 mm semi-axes; 1,986 vertices and 3,968 source triangles), current `classifySurfaces(pi, True, True, pi)` / `createGeometry` workflow, 12–24 mm Distance/Threshold field, and straight-reference tet10 volume elements. No patient data, image, mask, landmark, measured curve, physics solve or training was used.

| Observation | Curvature off, minimum 12 mm | Curvature 24 elements/2π, minimum 3 mm |
| --- | ---: | ---: |
| Nodes / tet10 elements | 269 / 116 | 2,806 / 1,340 |
| Median boundary edge | 10.749 mm | 3.167 mm |
| Source → mesh maximum sample distance | 4.803 mm | 0.437 mm |
| Mesh → source maximum sample distance | 4.436 mm | 0.420 mm |
| Source → mesh full covering bound | 5.302 mm | 0.936 mm |
| Mesh → source full covering bound | 4.935 mm | 0.920 mm |

This demonstrates an actual **joint profile effect**. Curvature and the compatible minimum-size floor changed together; it is not an isolated factorial estimate of either setting. The 10.43-fold node increase is specific to this small analytical input and cannot be multiplied into a patient node or memory prediction. Fidelity is measured against the dense input triangulation, whose volume differs from the exact ellipsoid by 0.401%; it is not an exact analytical Hausdorff distance.

The official [Gmsh 4.15.2 source archive](https://gmsh.info/src/gmsh-4.15.2-source.tgz), SHA256 `be3f66f225d27ba9fa014f07e83169285da8a051b0e8ab7103d88066b39bdd3e`, establishes the relevant path:

- `src/geo/GModelParametrize.cpp:492` computes discrete curvatures during classification; its subsequent routine uses `CurvatureRusinkiewicz`.
- `src/geo/discreteFace.cpp:578` transfers this data during geometry creation; `curvatureMax` reads the stored values.
- `src/mesh/BackgroundMeshTools.cpp:64` obtains surface curvature. The resulting size participates in the background minimum and is subsequently clamped by the global minimum and maximum.

These are numerical curvature estimates on a discrete surface. The source does not turn the sizing request into a fidelity guarantee. The manual documents the [curvature and global sizing controls](https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes). The raw GitLab endpoint returned a bot-check page and was not used as source evidence; selected files were inspected from the official published archive instead. `official-source-inspection.json` retains its size/hash and inspected-file hashes; the archive is local under `build/validation/` and is not duplicated in Git.

Native curvature queries were nonzero at all 64 predetermined sample locations (up to the first 32 classified input nodes per patch), in both profiles. Those locations are evidence that the discrete geometry carries usable curvature data; their reported min/median/max do not cover every point of the surface. The observed size and fidelity changes supply the separate empirical effect check.

The pair used 2.795 s supervised wall time and 171,065,344 bytes sampled peak group RSS, below the fixed 30 s / 1 GiB limits. One numerical thread and sequential VTK were required. All 13 source/runtime/declaration bindings remained unchanged. The compact raw output index records every file; source and original patient helpers were not edited. These resource observations do not predict a larger anatomy mesh or a mechanics solve.

A next patient candidate, if separately declared and released, can use this tested curvature-24 / minimum-3-mm profile while retaining the existing source, 12–24 mm interior sizing, straight tet10 and all physical-fidelity gates. Strong folds and voxel-scale curvature may substantially increase the mesh or still fail the 2 mm bound. Mesh-count, full-diagnostic-packet and actual wall/RSS limits need their own prospective values; this successful analytical control grants no patient execution or solver admission.
