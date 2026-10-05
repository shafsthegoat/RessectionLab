# Independent axis input-profile review

The final independent run passed **25 checks in 14.22 seconds**, with one CPU
thread and a constructed 7×7×8 anatomy with 1×1×1.5 mm voxels. No public imaging,
population model, paid service, or full-suite execution was used in this review.

The independent tests calculate removed and newly contacted retained cell sets
directly, checking target, normal, motor and language feature volumes before and
after a cut. They confirm that the historical `depth` field is action-budget
usage, adjacent-target occupancy uses the supplied labels, and partial contact
does not double-charge previously contacted retained cells. The existing fixed
divisors remain unchanged; all six critic inputs remain raw.

Other checks verify equal paired trainable initial tensors, distinct behavioral
hashes, the actual same-forward actor transform, row permutations and masked
padding, stale action rejection, exact backend types, malformed observations,
declared-profile mismatches, and coherently resealed unregistered divisors. A
second set of guards reconstructs otherwise well-formed decision records with
raw actor inputs, altered critic inputs, or invented action IDs; accounting
rejects each before a transition. Both registered profiles refuse axis transfer
and resume before checkpoint deserialization or Adam creation. The existing
procedural-pool and PPO boundaries remain closed to this new axis profile path.

One tiny FEATURE_UNITS update changed actor parameters and completed 3
optimization transitions and 6 selection transitions. Initial and updated
selection returns both measured −152.98, so the initial checkpoint remained
selected under the existing tie rule. This demonstrates integration and honest
accounting, with no observed selection improvement in this constructed example.

`verification.json` binds tested source, final documentation, test logs, and the
compressed actual receipt, contract and result in `tiny-update/`. All 34 numerical
module hashes in that contract matched the final working source. Source copies
preserve both the original 15-check test and the final 25-check test. The owner's
separate `../native-axis-input-profile-prerequisites-v1/` archive contains its
broader tested source; this independent folder does not substitute for a complete
release archive or the root-owned integrated suite.

The initial negative is retained: attempt 01 passed 10 checks and had 5 pytest
setup errors because the parent of the requested `--basetemp` did not exist.
Creating that parent was the only change before attempt 02, which passed all 15
checks in 9.40 seconds. Attempt 03 added the receipt and transfer boundary checks;
the production source remained unchanged across all three attempts. Registry
equality for resealed divisors was an early review suggestion that the owner
implemented before these frozen test attempts; no other production defect was
found in this review.

To rerun from a source checkout containing the tested revision and its installed
development dependencies:

```sh
mkdir -p build/validation/native-axis-input-profile-review-rerun
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_native_axis_input_profiles_review.py --basetemp=build/validation/native-axis-input-profile-review-rerun/attempt-01
```

Read the saved JSON evidence with Python's `gzip.open(path, "rt")`; decompression
is sufficient for inspecting the actual receipt and results. Checkpoint bytes
remain only in ignored build storage. Their recorded hashes do not recover those
bytes. A rerun creates fresh test checkpoints and environment-dependent timings;
this record does not claim byte-identical training replay from hashes alone.
