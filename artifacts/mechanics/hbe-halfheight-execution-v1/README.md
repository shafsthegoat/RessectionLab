# Half-height numerical-equivalence execution

October 8, 2026. Root releases **preparation only** under the existing real-data
goal. This numerical model uses HBE_01_03's creator-provided specimen dimensions
and existing saved full-model meshes. It does not create patient examples,
learning transitions or empirical validation data. The 1000-Pa modulus remains
an explicit numerical gauge, not a measured patient material property. No
response curve, calibration or withheld torsion data is released.

[Preparation release](prepare-release.json) SHA256:
`62243e3d9e54a2ba263c884e35e73fcf248ec743f7859aae77ff903a9528c68f`.
The exact 14-file source archive comes from commit
`437bbeddb9ea110644315a10d11846a0d52c4a0d` and has SHA256
`6db8fe2156148f9b45bc768f479e53ad56dee5b284aca662d4471236062914c9`.
Its member list is the unchanged
`artifacts/mechanics/hbe-halfheight-runner-preparation-v1/source-files.txt`.
All 12 imported modules bind to the extracted files under
`build/hbe-halfheight-v1/source/`; raw sources/archive remain ignored local files.

Read-only independent review checked the prepared hashes, 22 unique full-model
primitive files, 13 installed runtime files, private OpenMP and interpreter.
[Actual preflight](preflight.json) initially failed because invoking the bare
Python binary did not expose SciPy. Invoking the existing `.venv/bin/python`
resolved to the identical bound binary and succeeded with NumPy2.5.3/SciPy1.18.1.
No installation or source change was needed. Preflight rechecks saved numerical
controls; it is not merely metadata inspection, despite the legacy CLI message.

The released command is:

```sh
.venv/bin/python build/hbe-halfheight-v1/source/scripts/mechanics_hbe_halfheight_experiment.py \
  --root "$PWD" --phase prepare \
  --release artifacts/mechanics/hbe-halfheight-execution-v1/prepare-release.json \
  --release-sha256 62243e3d9e54a2ba263c884e35e73fcf248ec743f7859aae77ff903a9528c68f \
  --execute
```

It extracts whole lower-half cells at N8/N12 and prepares four decks under the
unchanged 60-second/3-GiB cap. No solver or mesher call is permitted. The phase
creates an exclusive attempt marker; do not silently restart it. Solver
equivalence requires a separate release bound to the accepted preparation.
Original mesh-convergence failure and N24 timeout remain unchanged.

## Completed preparation and bounded solve release

Preparation completed once; [independent saved-output review](preparation-review.md)
accepted the exact meshes, decks and source bindings. Compact original
[supervision result](preparation-result.json) and [worker state](preparation-state.json)
are retained. No preparation retry is permitted.

Root releases the separately declared solve phase under the existing mechanics
validation goal. The [solve release](solve-release.json), SHA256
`55e1b221a2619a6ba99a24b1579b01bfb05d53fcf853d4eafb8e29aab61c399b`, binds
preparation result `34fbd84994440a098c0b510a90e27b8b72a7cee8f20113d5e1d10ce944c5a629`.
Its preflight completed with exit 0 using the same existing virtual environment.
This authorizes the original four comparisons, at most 90 seconds each and
420 seconds overall, one numerical thread, 3 GiB sampled process-group RSS,
128 MiB active and 512 MiB total new output. Stop on first failure; no retries,
curve access, calibration, finer-mesh promotion or relaxed acceptance criteria.

```sh
.venv/bin/python build/hbe-halfheight-v1/source/scripts/mechanics_hbe_halfheight_experiment.py \
  --root "$PWD" --phase solve \
  --release artifacts/mechanics/hbe-halfheight-execution-v1/solve-release.json \
  --release-sha256 55e1b221a2619a6ba99a24b1579b01bfb05d53fcf853d4eafb8e29aab61c399b \
  --execute
```
