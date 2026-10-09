# HBE v5 N36 generated-fixture readout profile

**Status:** performance evidence only, 2026-10-09 UTC. No native FEBio execution, saved native output, measured HBE response, patient record, or physical validation was used. A generated rest frame and two-frame text fixture exercise readout code; they do not establish mechanical fidelity, numerical convergence, or a released 121-frame run.

## Bound inputs and method

The profile used the frozen v5 schedule and existing N36 lower-half specimen geometry (39,610 nodes; 34,992 hex8), reconstructed to the full specimen (75,259 nodes; 69,984 hex8). Each mesh/reconstruction file digest matched the declared v4 binding. The generated rest frame set current nodes equal to rest nodes, displacements/reactions to zero, and element `J=1`; no response was fitted. Python 3.12.14 and repository `.venv` NumPy 2.5.3 were used on the local Mac. The baseline frame source was copied from the then-current committed version before a separate writer's in-progress cache edits.

| Input/source at profile time | SHA-256 |
| --- | --- |
| `scripts/mechanics_hbe_v5_frame.py` baseline bytes | `b69d256db617bd7d68117df588b92b018e8260ac90d6b4dfa8f7a8199a31848d` |
| `scripts/mechanics_hbe_v5_stream.py` baseline bytes (inspected; full stream **not** run) | `268470228fe962c0a20eee1fe3e68c84f9ad5927655167cc6c16c524d6417e99` |
| `scripts/mechanics_hbe_physics.py` | `00b8b5c87696609178a673ed98bbbe706ec524b4a8a09d0f0a55a85f53fb345d` |
| `scripts/mechanics_hbe_halfheight_readout.py` | `8ffe2742f20e0d426beb93bb9fa8f24d9766a705b9154d90cc83e8bd5a7efa30` |
| `scripts/mechanics_hbe_outputs.py` | `f64d4f499644177a846e66a55bc9264eb7d8997f346ffa0ba27d09a9d7b19b19` |
| `scripts/mechanics_hbe_branch_calibration_v5.py` (schedule) | `1f183287e65d008095f004815a1a2344a74f71d935cedddc7388d42d1dbdc016` |
| v5 / v4 experiment declarations | `50c5dbc8c45279245a8dcd9b92d16e9499bad6bc61c6dfbd5b5e307af49a71ce` / `85ed9ca8cb1a9048e885f27424678e97f20cb6dd2e2cbb5cfd6dccfe6dafb52b` |
| N36 full / half mesh manifests | `6ba012cc042ec0a537a6c61283d444488ccf69de0bc9c385f319a827d69c8e74` / `8e7ad0e636151e007ab9fd8af0d043abfcf889b960c53e726766126691c3c075` |
| N36 reconstruction wrapper | `9e0265495f47ece2d315746f79f40e71f717e0720f6f1621fa9c7f9d2d9eba8f` |
| Generated two-frame node / element text fixtures (ignored local files) | `68c4cf89037a995cb35fb64451cb9d452e7c24a2111504ec2711db1c30ba8ecc` / `01d18a1c93f884d98440c33c4b33ffa5e588162b5a303e2457fd6fdd5ba6a618` |

Reproduction helpers and full profiler stdout are under ignored `build/hbe-v5-stream-profile/`; the timing helper SHA-256 is `4083d3083c68ff21c3b5f83bb23e9afea5fd479b712280614463cf709c98e94d`, parser helper `feda4b9b2e5c852df2cb1ab0c6758db79364f4037507d6afac7e780d83f94146`, and exact-result comparison helper `7790d2a0a8bc0783a838c8a3091c65e28d62938447058e582dd13a65ae4b8c88`. These helpers and raw generated text remain untracked.

## Measurements

Manifest load was 0.21–0.23 s, with 187 MB RSS afterward. The generated N36 rest frame took **4.31–4.45 s** under cProfile and **4.18/4.19 s** in two uninstrumented repeats. `/usr/bin/time -l` observed 760 MB peak RSS on the first frame and about 796 MB after repeats. It returned `fixture_passed=true`, which is only an internal generated-fixture check.

In one 4.31 s profile, **three `HexMesh.from_manifest` calls took 2.832 s cumulative**: the frame rebuilt its native mesh, then `HalfHeightReconstruction` rebuilt both half and full meshes. These calls repeat fixed rest-map inversions, Jacobians, fingerprints, and mapping validation. Two per-frame deformation calls took 0.904 s; fixed probe inverse mapping took 0.160 s. JSON serialization/hashing took 0.451 s total, partly for unchanged mapping/geometry identity. Profiler cumulative categories overlap.

A **profile-only** monkeypatch reused one prevalidated immutable reconstruction/mesh pair without changing production source. Construction cost 2.396 s once; three repeated frames took **1.116/1.109/1.115 s** each. Uncached and reused result dictionaries compared exactly equal, and their canonical sorted JSON SHA-256 digests were both `b0fcc2f4b6ce890e90b015514191d90673b8a7b746c970b0e05736b1244354a6`. Peak RSS in the retained-profile process was 802 MB; this is not a minimized-memory benchmark or validation of the writer's later source edit.

A separate strict parser trial used only two generated text frames: 6,774,158 node bytes and 2,637,280 element bytes. Parsing both took **0.254 s nodes + 0.177 s elements**, or ~0.216 s per generated frame; peak RSS was 88 MB in that separate process. Historical N36/S120 native logs were much larger (about 775 MB nodes and 518 MB elements), so this is not a native-log throughput guarantee. The full reader's pre/post file hashing and 121-frame processing were not timed.

## Decision boundary

For the same generated-rest-frame workload, 121 evaluation calls **project** to ~8.4–8.5 minutes with repeated geometry construction versus ~2.3 minutes with one-time preparation and reuse. This is extrapolation, not an observed full-stream result; actual parsed file length, I/O, work checks, non-rest states, and native solve cost remain unknown. The older 90 s / 256 MiB limits cannot be carried over to this N36 readout or historical N36/S120 output sizes. The measured readout RSS is below a *tentative* 3 GiB process-family design cap; that cap is not validated for a solver or a full run.

A safe implementation would prepare stream-local, hash-bound full/half meshes, reconstruction mapping, fingerprints, and 75-probe map once after authenticating the six input bindings. It must retain every per-frame primitive ID/time/shape/finiteness check, boundary/reaction check, sampled Jacobian and energy calculation, solver residual and work checks, and pre/post output hashes. Reuse must never cross changed source, mesh, or mapping identities. This result supports prioritizing that narrow optimization; it does not release native execution, fitting, measured comparison, or clinical claims.
