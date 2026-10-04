# Proposal integrity regression record

Independent review found that the first standalone proposer trusted cached
native configuration and cavity fingerprints. Replacing immutable source arrays
before preparation, or directly editing live cavity masks after preparation,
could therefore change proposals while retaining an old advertised identity.

`initial-native_proposals.py` preserves that initial implementation.
`initial-review-tests.txt` records eight failing and four passing independent
regressions. The initial failing cases remain in
`tests/test_native_proposals_review.py`, alongside later batch-staleness and
coordinated-mask-edit regressions.

The revised standalone module reconstitutes native configuration contents at
preparation, verifies the native V2 committed-history digest chain on every
proposal call, and reconstructs all four live masks from that history. It rejects
changed data instead of refreshing identities to accept it. Exact batch
revalidation rejects altered or stale proposal records. No protected native
engine file is changed, no geometry certificate is cached, and no clinical
clearance is claimed. The whole-grid integrity work is explicit and should be
measured separately before any later adapter integration.

After the separate feature-unit study's timing hold was released, the focused
proposer, independent review, native-engine and adversarial tests passed:
66 tests in 0.98 seconds. `repaired-focused-tests.json` records the command,
tested-source hashes and hashes of the retained initial failing evidence.
The independent reviewer separately reran all 15 mutation and batch-binding
cases: 15 passed in 0.29 seconds. `independent-review.json` and
`independent-review-tests.txt` preserve that separate check and its source hash.
The native-patient cost observation remains pending a separate timing window;
these small regression timings are not a patient performance benchmark.
