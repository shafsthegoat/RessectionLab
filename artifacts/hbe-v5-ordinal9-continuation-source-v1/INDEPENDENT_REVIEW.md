# Independent source review: proposed HBE v5 ordinal-9 continuation

**GO for the ignored generated-tested source candidate; HOLD any release or native execution.** This review did not launch FEBio, issue a release, run the full predecessor validator, read bulk predecessor output, or access patient or measured specimen data.

Frozen SHA-256s reviewed:

- Wrapper `hbe_v5_ordinal9_continuation_v1.py`: `8ea24abd2918bc36c289979e4ff15e1d7b0b5614d0ddfe14820254a7d81478b4`
- Wrapper tests `test_generated_continuation.py`: `d650f3edea369d6edcdce7ba1aed31b79648c6c602f1f9fecfbeda290022ebdb`
- `PROPOSAL.md`: `78b6f16be0eb92064cc460fabdedfef289f57d8596d71329097c6cca26cde21c`
- Updated admission source `admission.py`: `20d2aab7cfe22001c255c88f4b3ffd9023eb6ba134bdd08a0aab9c05b3d52295`
- Admission tests: `b819d0bc94b86349b80456ea94728419ab82e7fb3917519610360b197aaeaf7b`

The admission change binds tracked `RESULT.md` and `RESULT_METADATA.json` whose bytes match the previously reviewed ignored originals (SHA-256 `6fb89ad0e7b132796554f285ece490d5a20c45c9f031a067174d7b8ca4abd56e` and `6c048fcaa61c19b65a9f3799e69f3bf5b0bc81f2a0bdc890aa90cfb77be81345`). Both copied hint/host helpers are byte-identical to their current tracked counterparts (SHA-256 `96c71d0d9b8d896b713aef7305e3e86d5b12b9d2dca5fcc55515f6b2eb9cf3fc` and `2ff8f3e4d30ed0a8976fb01da99403ff5b2b2ffd6e750791fe7a954946d0a00f`). The proposed 16 selected predecessor opens add ordinal-8 `nodes.log` (29,311,268 bytes) and `elements.log` (18,500,496 bytes) to the independently reviewed prior 14-open, 2,801,621,755-byte hint total: 2,849,433,519 bytes before native and twice that after.

The outer envelope requires a separately issued status and binds one ordinal-9 inner release, one commit, all four executing source bytes, policy, and sidecar path. The unchanged old validator remains the full nine-receipt source/runtime/deck/output gate. The wrapper authenticates the ordinal-8 v2 event, charges its 1.1413705407176167-second preparation delta and 1-MiB sidecar reserve once, and carries that adjusted ledger through pre-native and final aggregate checks. It repeats source/release and v2 checks immediately before native supervision; retains/reaps owned native/readout children; bounds host pressure, wall, RSS, output, thread count through the existing old runner and wrapper; rebinds the saved native receipt and persists an ordinal-9 supplemental ledger without modifying old receipts. Terminal success requires a regular receipt-only sidecar under the reserved 1-MiB cap. Indices 10/11 remain closed to the admission API pending later wrapper accounting.

Independent generated-suite run in `.venv` with an ignored temporary directory: **10 tests passed; pytest reported 25 subtests in 0.33 seconds**. This includes fake successful one-call flow, host/ancestry/hint/source/release refusals before the fake native stage, source binding, native receipt stability, sidecar inventory/symlink/cap negatives, and no-double-charge controls. The tests exercise control flow; they do not establish real ordinal-9 host feasibility, actual numerical behavior, convergence, specimen force agreement, or patient benefit.

Before any native call: promote the two proposed sources together in one reviewed commit, bind the four exact executing sources and unchanged old runtime/deck in independently reviewed inner/outer release bytes, obtain a separately reviewed ordinal-9 read-only F_NOCACHE feasibility/host measurement, and verify the full predecessor chain and live host/storage caps. Preserve the old numerical thresholds and one-shot limits. There is no ordinal-9 release or attempt in this source candidate.
