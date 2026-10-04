# Patient imaging viewer

One WebGL2 context shares original MRI and annotation textures across the 3D scene and three linked MRI views. Source arrays use NumPy XYZ C order (Z fastest); texture dimensions reverse this order. Source affine transforms are retained, including oblique spacing. LPS inputs convert once to the RAS display frame. Cursor positions and complete instrument dimensions are millimeters.

Source binary annotations produce display surfaces in a worker. The 0.5 isosurface rounds voxel corners; quantitative volumes always come from native voxel cells, never mesh volume. No generic brain surface is substituted for missing patient anatomy. The optional MRI plane starts hidden in 3D to expose the annotation surfaces; the linked MRI views remain visible.

## Modeled removal

The host first verifies the independent native replay certificate. `validateReplay` then checks the accepted effect's case version, grid, RAS affine, binary cells, and target/normal/residual cell accounting. It is a display integrity gate, not an independent trajectory checker.

- Original MRI and source annotations remain unchanged.
- Colored replay surfaces show source annotations minus accepted removed cells.
- The mint mesh and MRI overlay show only accepted modeled removal. They do not show accessible volume or partially contacted tissue.
- Candidate approach tools are hidden during replay because their terminal poses do not establish the replayed motion.
- A new replay cancels pending surface work. Failed or cleared replay restores source anatomy; stale worker results cannot replace current geometry.

`coordinates.test.mjs` covers interpolation, orientation, physical bounds, watertight surface geometry, union accounting with overlapping annotations, immutable source data, LPS replay alignment, rejected invalid effects, and zero-removal STOP.

The October 4, 2026 integration check used `outputs/desktop-bridge/native-ucsf0004-v2/candidate.json` and the UCSF-PDGM-0004 source arrays: two accepted strokes, 249 mm³ target plus 17 mm³ normal modeled removal, 41,670 mm³ residual target. Viewer checks agreed exactly. On the local development machine, validation took 79 ms, residual copies 42 ms, and removal mesh generation 39 ms for 8,928,000 source voxels. These are one-run implementation timings, not clinical or throughput benchmarks. The underlying geometric and tissue assumptions remain those declared in the saved replay.
