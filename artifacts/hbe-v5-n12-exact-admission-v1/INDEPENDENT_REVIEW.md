# Independent exact N12 supplement-admission source review

Decision: **GO for this narrow exact numerical predecessor/input exception; no native run or completed twelve-row comparison is authorized or claimed.** I made no tracked edits and launched no native or replay worker.

Frozen reviewed SHA-256s:

| File | SHA-256 |
| --- | --- |
| `scripts/mechanics_hbe_v5_n12_admission.py` | `29bbd47b1df7bd08efbca9cea13172e9ba220dec8dd9dbc22915285dffb9b70a` |
| `scripts/mechanics_hbe_v5_n12_saved_replay.py` | `c35c3f122599c6415607d12c77c70e5c29070256b9bd2c5b13440513773a020d` |
| `scripts/mechanics_hbe_v5_remaining_one_shot.py` | `90d855b1e9518793480688afbfb133b586a6ccbe5b974be30b6b40624336216b` |
| `tests/test_mechanics_hbe_v5_n12_admission.py` | `929b8805f51e1e36481dec6e904326fac29ede1e7f401fb92ffe8e4dc600b880` |
| `tests/test_mechanics_hbe_v5_n12_saved_replay.py` | `b1305a71f73054153f9468f0efd4efbfa4724a6371696f9f04d622e1a4527ce3` |
| `tests/test_mechanics_hbe_v5_remaining_one_shot.py` | `20e12c7d830e5cbc821ab66a0143ad26e3739deb6601b0d3589a80342818d887` |
| `manifests/experiments/hbe-v5-n12-exact-supplement-admission-v1.json` | `84cd645a10d70c07613e12d9548076336326b8e97510cfe83e39e261ec0eaa5e` |
| `docs/hbe-v5-n12-exact-supplement-admission.md` | `e3bc0229ea18ef5a91388250b51bc8f5867e73bc2ba32f57da2d02b0af08eee6` |

The exception is keyed only on ordinal 1 and original failed receipt SHA `9020e6c7ea1f01360dc8d02e5efd16e4a29591c2f566d0194415b03d8b68bbc2`. The generic branch still requires `passed_numerical_software_only`, so other failed receipts are rejected. The original receipt remains `failed_or_incomplete` and its SHA was unchanged after tests. The new verifier authenticates the exact original failed receipt, original release, historical source closure, nine original files, the specific supplemental receipt SHA `8985a0e23b5d3a3d0029c5d69b48caaa6aff54c0628e40dabd1ea33aa6f3bb24`, replay release/commit, replay files, independent review artifacts, and byte-identical saved numerical JSON. It compares original frozen runtime/profile to the caller's expectations. The original missing post-run source/runtime guard is preserved as a disclosed historical gap, not retrospectively passed.

Historical source closures remain separate from the expanded future release closure. This matters because the original release cannot contain newly added admission modules; future one-call releases now bind those modules. The saved-replay validator uses its historical closure for original Git-blob checks, and the exact verifier uses the historical replay closure for the supplement's release.

Accounting is explicit. N8 plus the original N12 produce exactly two cumulative native calls, 15.259143584175035 s native time, 2.8276635000947863 s original readout, 1.348830624949187 s original N12 preparation, and 66,576,514 B original closed output. The supplemental replay separately contributes one replay call, 2.8474308329168707 s worker time, 3.4362154591362923 s preparation, and 557,392 B. Combined N8-plus-N12 totals are 25.71928400127217 s and 67,133,906 B. The separate supplemental caps remain checked; the pre-existing prospective twelve-row 18,000 s/6 GiB limits were not silently rewritten. Future receipts carry prior supplement and combined ledgers, while the failed original is charged once.

Verification: independent `.venv` runs completed 29 focused admission/replay/supervisor tests in 14.98 s and 14 adjacent generated-comparator tests in 1.09 s; `git diff --check` was clean. Source and actual receipt SHA-256 values above were independently rechecked. The tests include forged original source binding, same-size original-file tamper, missing release/token, wrong frozen runtime/profile, changed predecessor hash, and exact accounting.

Remaining gates: the admission only makes this one numerical readout eligible as a future native comparator input. It does not run row 2, complete the twelve-row native comparator, fit material parameters, inspect measured responses or patients, or establish physical/clinical validity. Any future one-call row still needs its own reviewed exact release and resource preflight.
