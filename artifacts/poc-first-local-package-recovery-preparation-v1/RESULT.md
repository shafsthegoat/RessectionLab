# One explicit sidecar recovery

Ready for ROOT's dependency-audit/source commit and separate release. No build, failed-output move/copy, dependency change, or biomedical payload read was performed here. The original `923d8cc8` supervisor and its 13 generated controls are preserved alongside the original failure metadata. All **28** current generated controls pass.

The actual first attempt was `worker_failed`, exit1, reaped, after 48.3338 seconds. Its sampled peak supervisor/family RSS was 2,233,909,248 bytes; its original ledger charges 1,671,846,180 bytes across 3,318 paths. The failure receipt and original accounting remain unchanged:

| Record | SHA256 |
|---|---|
| `outputs/poc-first-local-package-build-v1/sidecar/receipt.json` | `5c03bcfffb832de6a907a6e3f94044c815cdd0cd345bb66ed14c08c9f9dd42ac` |
| `outputs/poc-first-local-package-build-v1/sidecar/growth-ledger.json` | `8ade65f73cd9c411256272f4037abc34a052a579142022c0fa1e3a6f413f73d3` |
| `outputs/poc-first-local-package-build-v1/baseline.json` | `34e8f9cdafdfac30f30b5621c1d51cfb569d71d7764cac0e74f2b5b62f5923c3` |

Only `sidecar-recovery-01` is added. It uses the existing plan's **sidecar** program, arguments, cwd and environment; the `describe` maps therefore retain the `sidecar` key for its program/environment hashes. ROOT must issue a new release bound to the corrected committed sources and these exact three predecessor hashes. The recovery uses the original session and baseline, loads the complete old growth ledger, and never overwrites the first receipt or ledger. The 8 GiB cumulative limit and every other plan limit remain unchanged. Recovery 02, arbitrary subdirectories, successful/unreaped predecessors and reuse of recovery 01 are refused.

Before release, ROOT must preserve the failed trees intact at:

- `build/poc-first-local-package-v1/failed-first-sidecar/electron-sidecar`
- `build/poc-first-local-package-v1/failed-first-sidecar/ressectionlab-engine`

The original `build/electron-sidecar` and `desktop/sidecar/ressectionlab-engine` targets must be absent. The runner checks this but does not perform the moves or replace ROOT's before/after hash audit. Preserved trees remain inside the monitored roots; their new paths are conservatively charged in addition to the carried failure ledger.

After ROOT commits the lock/supervisor corrections, refresh the description into a new file:

```sh
.venv/bin/python scripts/poc_first_local_package_build.py describe --output outputs/poc-first-local-package-recovery-release-description.json
```

`unreleased-recovery-and-electron-templates.json` remains `NOT_RELEASED`. ROOT fills the refreshed commit/source/runner hashes and sets `decision` to `release_one_stage` for the authorized attempt. For recovery, take the program/environment hash from `description.program_sha256.sidecar` and `description.environment_sha256.sidecar`.

```sh
.venv/bin/python scripts/poc_first_local_package_build.py sidecar-recovery-01 --release artifacts/poc-first-local-package-build-v1/sidecar-recovery-01-release.json
```

Electron must now explicitly select `sidecar_subdirectory` as either `sidecar` or `sidecar-recovery-01`, and provide `sidecar_binding` with the selected receipt, ledger and original baseline hashes. The selected receipt must be successful, reaped and bound to the current committed source manifest. It cannot select the failed first attempt. The existing exact local ZIP-directory addition and archive pin remain required.

```sh
.venv/bin/python scripts/poc_first_local_package_build.py electron --release artifacts/poc-first-local-package-build-v1/electron-release.json
```

No automatic recovery or retry occurs. A successful worker still means `completed_unverified`; frozen-engine and packaged-GUI validation remain separate ROOT work.
