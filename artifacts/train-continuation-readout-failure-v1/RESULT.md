# Incomplete comparison: cache handoff failure

All 36 original TRAIN traces and 497 rows were reconstructed. The authenticated
670,261-byte feature cache was saved and reloaded successfully. The runner then
attempted a second exclusive write to `logged-targets.json`, raising
`FileExistsError`. This is a runner integration defect missed by the preceding
controls. No head update, calibration or fresh comparison route ran, so this
attempt establishes no policy improvement or completed comparison.

The original output directory, source index and source code remain unchanged.
In particular, the earlier target metadata was not refreshed after the cache
save. Recovery must use the separately pinned cache receipt, external original
binding and reconstruction evidence; it must not silently reinterpret or replace
the failed report. No cache payload is committed in this directory.

Costs remain paid: 1,285.243 seconds parent, 1,284.667 seconds worker,
45,560 native previews,497 encoder forwards and 497 reconstruction transitions;
peak sampled process-group RSS1,463,156,736 bytes. The child exited 1 and was
reaped; cleanup is verified with no remaining owned process. This is scientific
execution failure with successful supervision, not a successful endpoint.

The derived cache has SHA256
`2ebd7bd2ad777492bec95fa910ee2fb1c4405da04a30f69219b448ae20d236f7`.
Its external binding and reconstruction evidence are included as compact scalar
records. The original source index remains in
`../train-continuation-readout-source-v1/runner/source-index.json`.

A separately declared recovery will reuse that exact cache, write each output
once, reproduce the original Q64 head and perform the same 12 fresh
GREEDY/Q64/CALIBRATED routes. All rewards, worlds, TRAIN identities, information
boundaries and success criteria remain unchanged. It will save compact
per-row predictions so an independent checker can recompute fitting errors.
This avoids repeating successful reconstruction; the original failed cost is
still included. Patient images, features and weights remain outside Git.
