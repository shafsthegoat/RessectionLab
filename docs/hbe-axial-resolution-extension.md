# Fixed axial mesh-resolution extension

The completed Accelerate experiment remains failed at its mesh-convergence gate. This separate numerical study adds only N16 and N24 compression/tension at the original 60 equal load steps and μ=1000 Pa. It performs no calibration, measured CSV access, torsion run, policy training, or material selection. The original N4/N8/N12 results and their budgets remain immutable.

The fixed [declaration](../manifests/experiments/hbe-01-03-axial-resolution-v1.json) requires **both** N12→N16→N24 and N8→N16→N24 comparisons to pass the original reaction, displacement and refinement-trend criteria. The common N16→N24 absolute differences are reported in each group; neither triplet can be selected after seeing results. N8 and N12 use the exact previously accepted saved axial readouts. The [saved diagnosis](../artifacts/mechanics/hbe-cross-mesh-diagnosis-v1/DIAGNOSIS.md) found actual field changes rather than a probe interpolation defect.

Preparation generates two meshes with the existing Gmsh geometry: N16 has 6,144 hex8 cells/7,209 nodes; N24 has 20,736 cells/23,101 nodes. Both must pass the original finest-mesh geometry gates. It writes exactly two original Skyline decks per level and applies the existing verified solver-token-only Accelerate transformation. The geometry, constitutive energy, fixed modulus, boundaries, loads, tolerances and probe positions are unchanged. The new explicit readout/parser entry points admit at most 25,000 records; the original study still uses its 20,000-record bound and original allowed mesh levels.

The [thin runner](../scripts/mechanics_hbe_resolution.py) reuses existing solver execution, process-group supervision, output monitoring and numerical comparison functions. Preparation and solution require separate root-issued releases. Preparation has a 120-second aggregate wall limit; solution has four calls at most, 420 seconds per call and 1,200 seconds total including worker preflight/readout/comparisons. Both use one numerical thread, a sampled 3 GiB process-group RSS cap, 512 MiB active output and 2 GiB total new-study output. Partial failure is terminal and retained. No compression savings or runtime extrapolation is assumed. Historical raw evidence stays outside the new output tree and is never duplicated or charged to its new-output cap.

The new namespace is `outputs/mechanics/hbe-01-03-resolution-v1`, with `mesh-preparation` and `experiment` subdirectories. Exclusive phase markers prevent retries. The worker cannot advance from meshing into solving. A passed numerical extension would not establish material fidelity or authorize subsequent fitting.

Each final release must have schema `hbe-resolution-release-v1`, `authorized: true`, the exact `phase` (`prepare` or `solve`), `study` path/SHA binding, `source_commit`, `source_archive`, ten `source_bindings`, exact interpreter identity, the existing accepted `backend_profile`, and verified `gmsh_runtime`. The solve release additionally binds the actual accepted preparation `result.json`. Draft or missing releases fail closed.

The minimal prefix-free Git archive contains the ten imported script files (the existing nine-module HBE execution closure plus `mechanics_hbe_resolution.py`) and the original protocol and new declaration: twelve files in total. Loaded source locations and file hashes must match that archive and its Git commit comment. External runtime, eight-control evidence, Gmsh, prior run and prepared-input dependencies remain separately hash-bound. No complete repository archive is required.

After separate commitment and release, the CLI is:

```text
<bound-python> <archived-script>/mechanics_hbe_resolution.py --root <repository> \
  --phase prepare --release <preparation-release.json> --release-sha256 <exact-sha> --execute
<bound-python> <same-archived-script>/mechanics_hbe_resolution.py --root <repository> \
  --phase solve --release <solve-release.json> --release-sha256 <exact-sha> --execute
```

Omitting `--execute` performs metadata checks only. These commands have not been run for this extension. Focused preparation tests use analytic arrays and mocked native operations; they are not completed mesh or solver evidence.
