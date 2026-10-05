# Independent saved-geometry diagnostic review

The frozen diagnostic passed 24 focused controls (18 owner and six independent cases, 0.42 s). Review covered deterministic sample equality and global witness indices across chunk boundaries, area-weighted face summaries, edge-connected components, fixed normal-change proxies and execution guards. All tests used analytical arrays or mocked execution. No saved geometry, images, landmarks or native geometry API were accessed.

Two static guard findings were repaired before independent reproduction: final acceptance must become failed when its own publication breaches a cap, and VTK must confirm its Sequential backend. The corresponding independent controls passed. No failing test run is claimed for either finding.

The exact original lattice and group-cover arithmetic are retained. A separately released real diagnostic must reproduce the saved sample counts and extrema within declared tolerances. Whole-face area reports measure faces containing sampled witnesses, not the exact area where distance exceeds 2 mm. Adjacent-normal changes are discretization proxies; no seam IDs survive, so seam attribution and causal claims are unsupported.

Preparation retains the declared one-attempt 120/110-second, 2-GiB, one-thread, 8-MiB and 1.2-million-query limits. RSS and aggregate output checks are sampled, and the parent bounds long operations. Passing this source review provides no execution permission, mesh acceptance, anatomical validation or mechanics-solver admission. Both prior candidates remain rejected.
