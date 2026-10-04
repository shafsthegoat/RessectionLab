# Independent HBE backend integration review

This is a source and constructed-file review. It does not execute FEBio, read
measured specimen curves or patient data, or establish mechanics accuracy.

The initial three independent controls all failed because the verifier accepted:

- Eight case names and pass flags without execution records, primitive outputs
  or checker evidence. This test isolates runtime verification so a runtime
  refusal cannot conceal the missing control-evidence gate.
- A runtime inventory containing the executable but none of its twelve linked
  private libraries.
- A worker candidate marked `candidate_pending_parent_build_acceptance` as if it
  were a final parent-accepted runtime.

`initial-negative.json` records the tested source hashes, command and timing;
`initial-negative-pytest.log` preserves the three failures. The compressed initial
source and test files reproduce the uncommitted negative snapshot. Toy binary
and source files are deliberately not executable. The fixture substitutes only
the expected patched-source digest to authenticate these constructed bytes.

The owner's earlier OpenMP contract mismatch is separately preserved in
`../hbe-backend-preparation-v1/openmp-contract-negative.json` and its accompanying
pre-repair source snapshots. That mismatch concerns the build's declared reuse
of the original private OpenMP library. It is separate from the three failures
observed here after that narrow repair.

The repair requires all thirteen installed runtime files, a final accepted-build
identity and its six bound artifacts, both groups from the same eight-case
attempt, exact declared input decks, and replayed checker results and stiffness
scaling. It retains the explicit release profile through every specimen run,
full twenty-run replay and the dependency inventory frozen before held-out access.
Default Skyline source and runtime contracts remain separate.

A fourth negative was found after the first repair: identical checker bytes
could be loaded from an undeclared location. The real tet checker imports a
sibling helper by location, so content identity alone was insufficient.
`checker-origin-negative.json` and its source/test snapshots preserve that
failure. The final repair checks both pinned sibling files at their declared
source locations before importing either.

Final generic integration: **127 passed in 2.03 seconds**, including fourteen
independent controls; all fourteen bound source/test files were unchanged.
`verification.json` binds that result to backend source `e98a51a5…`, experiment
source `385fa5eb…`, and access source `cfc7a6c1…`. The earlier 126-pass run and the
subsequently discovered negative are retained without rewriting them.

The fourteen independent controls cover the original three gaps, exact original
OpenMP path/hash/byte reuse, one complete eight-case receipt graph, stale parent
evidence, another attempt directory, an undeclared same-byte original deck,
relocated checker code, rehashed raw-output disagreement and claimed stiffness
scaling disagreement. The eight-case graph uses clearly constructed checker
modules with explicitly substituted expected checker hashes: it tests evidence
wiring, not physical correctness or actual solver execution. The OpenMP cases
isolate build acceptance and use toy library bytes; the accepted-build graph has
separate owner controls. Legacy orchestration tests retain their documented
numerical and access stubs. Harmless watchdog child processes run in those
existing tests; no FEBio process is launched.

**Runtime patch identities are not cleared.** A separate review found that the
prospective adapter also needs matrix-attribute and ownership corrections.
The old patch/source constants in this generic source snapshot are provisional
and must be replaced after that repair is independently reviewed. No sparse
runtime, eight actual sparse-solver controls, specimen experiment, clinical
result, or execution authorization is established by this review.
