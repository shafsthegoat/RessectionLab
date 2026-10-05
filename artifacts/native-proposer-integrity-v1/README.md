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

`actual-case-observation.json` records one local UCSF-PDGM-0004 observation in a
brief coordinated quiet window. Native factory setup took 2.324 seconds;
provider source-content verification and static preparation took 0.044 seconds.
The initial integrity-only, propose and validate calls took 0.288, 0.287 and
0.285 seconds respectively. After one paid native stroke, these took 0.300,
0.302 and 0.302 seconds. One full native preview took 0.281 seconds and its
certificate commit took 0.028 seconds. Initial and updated proposed rays matched
the frozen experimental prototype exactly, and the old batch was rejected after
the cavity changed.

The process cumulative peak resident set reached 1,245,593,600 bytes (about
1.16 GiB), including case loading and native factory setup. This is not an
isolated allocation estimate for the provider. Each phase was observed once;
there are no repeated distributions, cross-model timing comparisons or claims
of controlled hardware conditions. No RL, final or stress worlds were used.
Whole-grid/history verification is a material cost, comparable here to the
single measured native preview. It remains enabled; any future adapter should
account for integrity calls separately and avoid a redundant validation of a
batch it just regenerated within one uninterrupted access path. Checking source
and cavity integrity does not independently certify geometric clearance.
