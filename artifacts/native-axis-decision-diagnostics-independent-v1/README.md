# Independent review of saved decision diagnostics

The final v2 diagnostic passes independent saved-array arithmetic and provenance
checks. All six initial/updated pairs have exactly equal ordered IDs, 15-feature
RAW float32 rows, six state features and legal masks. They describe three unique
states, each repeated across two deterministic selection seeds. The 26, 24 and
22 non-STOP rows are all byte-distinct within their respective states; there are
no exact observed feature aliases. The chosen action is unchanged at each step,
while 8, 17 and 12 ordinal ranks change, including STOP in those denominators.

`audit.py` uses only the standard library. It independently counts pairwise
ordinal ranks, verifies first-index tie behavior, checks every saved action's
logit and change, margins and values, and recomputes feature equivalence with
exact float32 bytes. Centered deltas also agree with exact rational arithmetic
within binary64 rounding: the largest difference is 2.7755575615628914e-17.
No tolerance is applied to matching states, features, masks, pair slots or ranks.

The source receipt, completed launcher/worker/pilot authorities, candidate bytes,
contract, journal, profile and initial/latest checkpoint file hashes agree.
The prior independently executed forward audit is pinned to its exact bytes and
matches these same saved decisions and their chosen indices. Its tensor/forward
verification is retained evidence; this review does not repeat it or instantiate
a policy. The selected checkpoint remains the initial checkpoint under the tied
441.6 selection return. These records establish no improvement or clinical
efficacy, and synthetic alias examples do not show aliases on this patient path.

One helper validation gap was corrected before final approval: the original v1
helper allowed the same `decision_id` in distinct seed/step/update slots. Actual
saved decision IDs were already unique, so the observed results were unaffected.
The revised helper requires nonempty unique IDs. Its rejection is independently
reproduced in `duplicate-id-and-version-recheck.json`, which also confirms every
v1/v2 scientific field is exactly equal; only the diagnostic script hash differs.
The original executed v1 script and outputs remain in their original namespace.

Fifteen independent constructed tests pass in 0.09 seconds. They cover repeated
states, event order, alias-group sizes, signed zero and one-unit float changes,
exact ties, common-logit-shift centering, masked/nonboolean rows, duplicate pair
slots and decision IDs, and raw/gzip or pinned-hash disagreements. The owner also
ran the combined 26-test suite; its receipt belongs to the v2 diagnostic folder.

The independent artifact audit initially stopped on a reviewer harness error:
it used the journal's spaced JSON convention for the profile's compact JSON
fingerprint. `audit-v2.log` and `audit-before-profile-hash-fix.py.txt` preserve
that attempt. Correcting the independently implemented serialization convention
produced `audit-v2.json` and `audit-v2-attempt-02.log`; no source experiment or
diagnostic values changed. Every input file was byte-identical after the audit.

The repeatable read-only command is:

```sh
.venv/bin/python artifacts/native-axis-decision-diagnostics-independent-v1/audit.py \
  --report artifacts/native-axis-decision-diagnostics-v2 \
  --source scripts/report_native_axis_decision_diagnostics.py \
  --output /path/to/a/new-audit-receipt.json
```

The output must be a new file. No model forwards, simulator calls, random draws,
gradients, geometry checks, final worlds or stress worlds were executed. The
test and audit source identities are recorded with the receipts.
