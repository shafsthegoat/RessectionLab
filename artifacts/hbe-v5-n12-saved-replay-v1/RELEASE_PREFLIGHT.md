# Independent N12 saved-attempt replay release preflight

Checked 2026-10-09 06:37 UTC. **GO for exactly one Python-only saved-output replay**, conditional on immediately rechecking unchanged HEAD, release and original-receipt hashes, and the still-unused target directory. The permitted outcome is a supplemental numerical software classification; the original failed attempt remains failed. This preflight did not run the replay, FEBio, or a test that internally replays saved logs. It did not access measured HBE response or patient data.

## Release and source identity

- Frozen checkout/release source commit: `f335975d4ed331eb33fc9806846cc19193037c1e`.
- Ignored one-use release: `build/hbe-v5-n12-saved-replay-release/release.json`, SHA256 `250961266042ffde26fb779743bbf669551b65d8b976e57df74cf80ff3018326`. Its status field is a required wire value, not an independent approval.
- All **19** bound executing Python files independently matched on-disk SHA256 and exact blobs in that commit. Static direct `scripts.*` imports stayed inside this closure, and the read-only validator's dynamic import audit passed.
- Pinned preparation: `manifests/experiments/hbe-v5-n12-saved-attempt-replay-preparation-v1.json`, SHA256 `45c6728ee2ce183990a1110c4ca333e22ed175b597d9e5378a131d2d60577329`; it retains `release: null` and closes native execution, generic failed-predecessor admission, comparator admission, fitting, held-out torque, physical validation, and patient/measured response access.
- Safe regular-file release read and read-only `validate_release()` passed in 1.656 s. The validator left `outputs/mechanics/hbe-v5-n12-saved-attempt-replay-v1/attempt-01` absent and did not launch a subprocess for replay or FEBio.

## Immutable original and runtime

- The original N12 release SHA256 `b9e765529e98668e8fe675b61bc7084d48efddb1f431221174801d3bfe50acad` and original failed receipt SHA256 `9020e6c7ea1f01360dc8d02e5efd16e4a29591c2f566d0194415b03d8b68bbc2` matched. I independently rehashed all **nine** whitelisted saved original files: total closed size **50,501,397 bytes**, including the legitimate zero-byte `readout-console.txt`. The receipt still states `failed_or_incomplete` with `ValueError: Native output absent or above bound`; its post-run source/runtime guards were not recorded after the parent packaging failure.
- The original separately supervised native and saved readout stages recorded exit zero within their prior caps. The saved `readout.json` reports a numerical pass over 61 frames. That saved JSON does not change the original failed receipt; the planned replay must re-evaluate the same bound logs in its own supervised Python child and preserve exact output equality.
- Current repaired `accelerate_csc_v1` runtime/profile verification passed with 115 bound profile/runtime inputs. All 13 installed executable/library bytes independently matched the runtime inventory; executable SHA256 `d7d4855d1d541a6407f22efd4381e23d7057cc49ec726144f06599aa5184f7b6`. This is a **current** recheck, not evidence that the failed original parent recorded those guards after its error.

## Execution and resource boundary

- The replay source fixes **zero native calls, one replay child, one attempt, no retry**, 600 s replay wall, 3 GiB sampled process-family RSS, 64 MiB active output, 150 s preparation wall, and one numerical thread. Its one subprocess command is the bound Python `--replay-worker` module with a one-time work-order token; the path does not invoke FEBio. Resource sampling can miss short peaks.
- The target output directory was absent; no `febio4` process appeared at preflight. Free disk was **695,832,514,560 bytes**, versus **1,140,850,688 bytes** required by the release validator (64 MiB output plus 1 GiB reserve). The private release and target outputs are ignored by Git.
- Entry point from repository root: `.venv/bin/python -B -m scripts.mechanics_hbe_v5_n12_saved_replay --execute --release build/hbe-v5-n12-saved-replay-release/release.json`. Direct-file invocation is unsupported. A resulting supplement requires independent saved-result audit and a separate explicit chain-admission decision before any later v5 row or comparator work.
