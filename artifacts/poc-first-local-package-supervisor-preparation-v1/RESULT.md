# Local package supervisor preparation

Ready for ROOT source commit and stage releases. **No build, install, download, image-array read, or model run occurred.** Thirteen generated controls pass. The initial `41ce325c` source snapshot and review history are preserved; the reported missing import did not reproduce on that exact snapshot. The final revision adds explicit `issuer: ROOT` enforcement and verifies a refused stage produces a scalar failure receipt without launching a worker.

Current runner: `scripts/poc_first_local_package_build.py`, SHA256 `923d8cc873643942d142c692d78eac21493cbe0e42c53ba72265bab9aa8c5788`.

Current tests: `tests/test_poc_first_local_package_build.py`, SHA256 `6607531a47d45a1435aa8baf43fbb0e92d0bbeef5b7bf70748e03e77ea2a9a40`.

Use `advisory-description-root-review-v2.json`, `generated-controls-root-review-v2.json` and `unreleased-stage-templates-root-review-v2.json`. The description is advisory because ROOT still owns the final source commit. Templates deliberately remain `NOT_RELEASED`.

After committing the intended source, refresh the description into a new output file:

```sh
.venv/bin/python scripts/poc_first_local_package_build.py describe --output outputs/poc-first-local-package-release-description.json
```

ROOT must bind each release to the refreshed source commit/manifest hash, supervisor hash, fixed plan hash, executable hash and environment digest. The runner refuses dirty or untracked build sources. The electron release additionally binds the successful private sidecar receipt, original baseline/growth ledger, exact archive path/bytes/hash and the single approved `RESECTIONLAB_ELECTRON_ZIP_DIR` addition. No other inherited environment values are passed, apart from the sidecar PATH permitted by ROOT; no environment values appear in receipts.

```sh
.venv/bin/python scripts/poc_first_local_package_build.py sidecar --release ROOT_SIDECAR_RELEASE.json
.venv/bin/python scripts/poc_first_local_package_build.py electron --release ROOT_ELECTRON_RELEASE.json
```

Each stage uses its exact program, argv and cwd from plan `a7a3e918e519882e115714d0624eb4ed8e9f6952598cca505029dcbe0940a7a5`. Existing builders remain responsible for building and capturing their source manifests. The supervisor adds no build architecture and never imports the image-registration runner.

The sampled stop limits are 3 GiB for supervisor plus recursive worker descendants, 1 GiB minimum available memory, 600/300 seconds for sidecar/electron, 8 GiB cumulative positive file growth, and 50 GiB free disk reserve. Sampling targets 0.2 seconds and reports the largest observed gap; these are not OS hard caps. The first-stage baseline covers the four disjoint plan output/cache/temp roots. Per-path growth high-water accounting persists into the electron stage, so deletion and stage transition do not reset it. Logical regular-file and symbolic-link lengths are counted without following link targets; repeated hard-link paths are conservatively counted separately.

Private session directories hold stdout/stderr, the original baseline, growth ledger and stage receipt. The release chooses the scalar artifact receipt path. Stage directories are exclusive and never reused, with no automatic retry. Stops, cleanup errors and failed bindings cannot become successful builds. A zero exit is still `completed_unverified` and requires a fresh build report; package and GUI validation remain ROOT's later work.

Generated controls cover success observations, RSS and timeout stops, nested-child cleanup while an unrelated process survives, PermissionError and observer failures, symlink non-traversal, retained cumulative baseline, deleted-file credit refusal, prelaunch disk reserve, environment isolation, source-binding refusal and issuer rejection. The frozen plan, desktop code and shared runtime were not edited; no commits were made.
