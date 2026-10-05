# Saved Case 4 geometry: straight surface chords miss local detail

The one released diagnostic completed from exact commit `afd0226be1b2619780cc9f8b6f5668ef0ba2b6fa`. It reproduced the prior sample counts, maximum distances and full-surface covering bounds without generating another mesh. Both failed candidates remain rejected and unchanged.

All **732 boundary corner vertices** lie on the retained source surface within `1.40e-17 m` in the nearest-triangle calculation. However, **32 of 2,196 straight midside nodes** exceed 2 mm distance, with a maximum of **3.473 mm**. **25 of 1,464 face centroids** exceed 2 mm, with a maximum of **3.432 mm**. This supports coarse straight chords between on-source vertices as a concrete source of local approximation error. It does not establish that every excursion has the same cause.

| Direction | Reproduced maximum | Faces with a sampled distance >2 mm | Area fraction of those whole faces | Connected face clusters |
| --- | ---: | ---: | ---: | ---: |
| Source → mesh | 5.460 mm | 7,590 / 183,902 | 4.135% | 104 |
| Mesh → source | 3.834 mm | 101 / 1,464 | 9.442% | 30 |

These area fractions count each entire face containing a sampled excursion; they are not exact areas where distance exceeds 2 mm. The source and mesh triangle resolutions differ, so their cluster counts should not be treated as matched anatomical regions. The saved report retains native RAS coordinates and counterpart-triangle IDs for ten worst witnesses per direction. No new anatomical labels were assigned.

On the mesh, 60.36% of the area of faces with excursions adjoins an edge with at least 30° normal change, compared with 17.35% of all mesh-face area. At 60°, the corresponding fractions are 3.70% and 0.757%. These are descriptive associations with a discrete geometric proxy, not an independent smooth-curvature estimate or a causal test. The original packet contains no Gmsh patch/curve identities, so it cannot locate seams.

The small **1.208% total-volume error** did not bound these local discrepancies. The current size field enforces a 12 mm requested-size floor and disables curvature sizing. A concrete established-framework follow-up is a separately declared Gmsh curvature/feature-aware boundary size field with a compatible lower minimum, keeping straight-reference tet10 and every existing geometry/count gate. The [Gmsh size-field documentation](https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes) describes curvature sizing and global clamps. No such change or new candidate was executed; lower local sizes may exceed the existing mesh-count cap and do not guarantee the 2 mm bound.

The worker made **1,152,670 distance queries** in 2.583 s; supervision elapsed was **2.902 s**, with **258,113,536 bytes** sampled peak group RSS. Raw output totaled **998,481 bytes**. These are one-run resource observations, not a speed benchmark. All 17 source/input/release bindings and all ten files of the prior graded attempt remained unchanged. The six-file archive matches the exact Git objects. No image/mask reload, B/V landmark access, Gmsh call, solve, training or retry occurred. The original estimated-envelope omissions and lack of anatomical validation remain limitations.

`raw-output-index.json` binds all original outputs, including the complete per-triangle diagnostic arrays retained locally. `saved-records/` contains byte-identical compact receipts. `post-execution-verification.json` records input, archive and original-attempt preservation. This result is diagnostic evidence only and grants no mesh or solver admission.
