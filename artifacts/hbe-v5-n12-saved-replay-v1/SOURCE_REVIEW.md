# Independent N12 saved-attempt replay source review

Decision: **GO for source-only preparation, not for release or downstream admission.** Reviewed the four frozen files at the exact SHA-256 values below, with no tracked edits, no replay launch, and no native call.

| File | SHA-256 |
| --- | --- |
| `scripts/mechanics_hbe_v5_n12_saved_replay.py` | `55cbd636032fe08c41c6acabda9a535505cb70249def7c2ee7d997be7a263fc2` |
| `tests/test_mechanics_hbe_v5_n12_saved_replay.py` | `8c0a54c9ed25be6fe2f0714df37a907de941560f74095c774157ff4f6d312588` |
| `manifests/experiments/hbe-v5-n12-saved-attempt-replay-preparation-v1.json` | `45c6728ee2ce183990a1110c4ca333e22ed175b597d9e5378a131d2d60577329` |
| `docs/hbe-v5-n12-saved-attempt-replay.md` | `95e19a5b191aab887114a363b67424558f3c3980c752c23f6798af513f60fc20` |

The manifest pins exactly one failed `compression:N12:S60:reference` attempt, ordinal 1, its failed receipt/release/source commit, and all nine closed files by byte count and SHA-256 (50,501,397 bytes). The checker rejects an added/missing file, symlink, wrong length, or same-length byte tampering. It permits a zero-byte file only for the original `readout-console.txt`; mandatory native logs, deck, work order, readout and failed receipt remain bound and nonempty. The original receipt is checked to remain `failed_or_incomplete`; no path rewrites it.

The original release is checked against its historical Git source blobs, N8 predecessor, source deck, mesh, adapted deck, schedule, and repaired Accelerate runtime/profile. Both original native logs are checked for a single Accelerate selection without fallback. The original failed receipt records preflight source hashes but not post-run source/runtime verification; the module explicitly labels this historical gap and records current runtime rechecks as current, not retrospective observations. The worker calls `mechanics_hbe_v5_stream.read_bound_run` on the six exact bound inputs, which verifies input hashes before and after parsing every saved frame. The focused test independently obtains full JSON equality with the saved readout, 61 frames/60 steps and a numerical pass. These are numerical software checks, not physical, patient, or clinical validation.

The supervised parent path requires a separately reviewed exact release, the new committed import closure, a fresh output path, and one Python-only replay process with 600-second wall, 3-GiB sampled process-group RSS, 64-MiB active-output, 150-second preparation, and no-retry constraints. The inherited supervisor kills remaining descendants on failure. The only child command is this module's Python replay worker; it does not run FEBio. Parent rechecks original hashes/release, replay work order and result equality, current source bytes and runtime/profile before recording only `supplemental_saved_attempt_numerical_pass_only`. Explicit receipt fields keep predecessor and comparator admission false. The preparation manifest has `release: null`, and the replay output directory did not exist at review time.

Verification: `.venv/bin/python -m pytest -q tests/test_mechanics_hbe_v5_n12_saved_replay.py` completed **7 passed in 5.31 s**. The tests include exact-whitelist, forged Git blob, same-size file tamper, symlink, absent-release and absent-worker-token checks. The writer's reported broader HBE suite (144 passed) was not independently rerun here.

Wording review: the final document now accurately limits the release gate to the supervised `--execute` path and states that direct invocation of the private worker cannot produce an accepted supplement receipt. This resolves the wording concern raised during review.

Still closed: no actual replay supplement has been launched or independently reviewed; the original failed receipt is not a normal predecessor; row 2 and the twelve-row comparator require a separate explicit exact-hash admission declaration; specimen numerical replay does not establish force-measurement agreement or patient mechanics.
