# Independent TRAIN validity launcher review

GO for the exact disabled candidate below, for a later root-owned one-shot release. This review does not release execution. It used source, saved structure metadata, and generated controls only; no original MAT, response values, or protected numerical arrays were opened.

Reviewed preparation directory: `build/menichetti-train-validity-launch-preparation-v1/`.

| File | SHA-256 |
| --- | --- |
| launch.py | 9b172b3e93375fb2dfc918da3f67896e0b4ea977520e3aa30f5d1b25b958d689 |
| worker_entry.py | 3f76e56598199b0d84ec66c79d14828336adebfe0b8ecc7ba5250ee9c5735693 |
| runtime-closure.json | a8584f66e51cdf45a36cdbe879c210c36577bb31a5ab651bcd6fa18367b8a6fc |
| authorized-template.json | 915e05dd613cefbc1bc0fd3928e1c92b68570b2e1ba644781caae9ea6ff440cc |
| test_launch_generated.py | b32abd972c7a4946b9e403a65594426ea52e25cfd9f46e2c9bb52617c2c40a8d |

All 14 source bindings were independently rehashed. They retain the reviewed reader `2eddae70e5e38d79bf1325dcb9fce574b49ef30098278cd055090c0067e41d6c`, prepared false protocol `f055e22a73f858bc3d4923b5c3c0a0d0b2ff8e949a5c42af5b25c4e39c85982a`, frozen roles, and saved metadata tree. The child protocol changes only the release bit and must hash to `68ba439afb73700c908a8e52b786e626601e6523c10aea6a2442c8ef3e383978`.

The release gate binds launcher bytes, sources, scope, caps, output directory, and exact six TRAIN aliases. Development/test access and fitting remain false. Historical Git blob checks retain the three existing supervisor helpers; current HEAD may advance through unrelated evidence commits. The reviewed OwnedWorker implementation is reused unchanged, including owned-child cleanup, direct-child fallback, reaping, and process-group confirmation. Fallback/errors cannot become accepted success. A freshly reserved output directory enforces one attempt.

The initial stale-bytecode gap was repaired: `-B` alone does not prevent reading existing caches. Accepted parent execution now requires `-B -X pycache_prefix=<absolute preparation directory>/launcher-unused-pycache`, with that exact prefix absent and not a symlink. The child receives its own fresh absent prefix under the new attempt directory. Both prefixes must remain absent before success. This root invocation requirement is material to this GO.

The entry checks its pinned loaded runtime before and after the worker. The closure covers 411 module origins, 406 unique module files, three explicit executing sources, four local native runtime files, interpreter bytes/version, and NumPy/SciPy versions. OS shared-cache libraries are platform-bound rather than byte-pinned; this limitation is explicit. A cold generated compressed MAT control exercises one TRAIN decode and one protected opaque skip without changing the recorded runtime.

The external 35-second worker stage covers cold imports, runtime checks, original fixity, QC, and final runtime checks. The 45-second lifecycle reserves 10 seconds for cleanup/finalization; sampled process-group RSS is capped at 512 MiB, aggregate output at 4 MiB, and the QC document at 1 MiB. RSS/output sampling can miss brief peaks and is not a kernel memory limit. Final output size and lifecycle checks prevent semantic success after an observed cap failure.

Semantic validation requires the exact 72 TRAIN matrices and 357 trial columns, protected paths, provenance, output whitelist, count partitions, boolean flags, and dispositions. An exit-zero worker reporting QC refusal becomes `qc_refused` and a nonzero launcher result. Nonfinite trials are counted as quarantined. Constants remain review flags. No raw samples, fitting, physical-validation pass, timing alignment, padding causality, or sign-convention claim is admitted.

Independent verification: the author's 15 generated controls passed in 2.46 seconds using a fresh reviewer cache prefix and fresh ignored test directory. Additional generated semantic controls confirmed one nonfinite trial is quarantined and an incorrect usable disposition is refused. All source pins, false template, and absent scientific attempt/cache directories were checked again. `controls.json` saves the exact evidence. Original MAT open attempts were zero. No actual scientific worker was launched; no tracked file was edited.
