# N12 saved-attempt numerical replay

The first native `compression:N12:S60:reference` attempt ran once. FEBio and
its separate complete-stream worker exited zero within their caps, but the
parent rejected the worker's legitimate zero-byte `readout-console.txt`.
The original receipt remains `failed_or_incomplete` with SHA-256
`9020e6c7ea1f01360dc8d02e5efd16e4a29591c2f566d0194415b03d8b68bbc2`.
It cannot be retried, rewritten or treated as a passing predecessor. The
fixed supervisor permits an empty file **only** for that ancillary console;
the deck, native logs, work order and readout remain nonempty.

`manifests/experiments/hbe-v5-n12-saved-attempt-replay-preparation-v1.json`
whitelists the nine original files by exact SHA-256 and byte count, original
release, original source commit, one failure signature and a fresh replay
output path. Its release is null. The separate
`scripts/mechanics_hbe_v5_n12_saved_replay.py` can only launch a supervised
Python saved-log worker through its `--execute` path after an independently
reviewed replay-only release binds a new committed source closure. The
private worker entry point can be called separately, but cannot produce an
accepted supplement receipt. Neither path can launch FEBio. One released replay is
bounded to 600 seconds, 3 GiB sampled process-family RSS and 64 MiB output,
with a 150-second preparation cap and no automatic retry.

Before a replay, the script rehashes the entire original closed directory,
requires the original failed receipt and release bytes, verifies all source
bindings against the original immutable Git commit, regenerates the exact
adapted deck and schedule from the frozen source/mesh, rechecks the repaired
Accelerate runtime/profile **at the current time**, and checks both original
logs for the actual backend selection. The original failure stopped before
its post-run source/runtime guards; those historical observations do not
exist and the supplement must disclose that gap. It does not infer that
current rechecks occurred at the original run time.

The worker replays all 61 saved frames with the frozen v5 stream reader and
the original six bound inputs. A prospective supplement can pass only if the
replayed result exactly equals the originally saved `readout.json`, both
original and current source/output hashes survive post-run checks, and the
current runtime/profile still verifies. Its classification is
`supplemental_saved_attempt_numerical_pass_only`; it has no physical,
measured-response, patient, fitting or clinical meaning. Its original failed
receipt stays failed. This supplement alone does **not** admit N12 to the
twelve-row comparator or allow row 2: those require independent review and a
separate, explicitly named chain-admission declaration. Generic predecessor
validation continues to reject failed receipts.

`tests/test_mechanics_hbe_v5_n12_saved_replay.py` checks the exact whitelist,
read-only original release/source/runtime evidence, complete independent
saved-stream equality, tamper rejection, and missing-release/token failure
before any replay output is created. No native run or replay supplement is
authorized by these source-only tests.
