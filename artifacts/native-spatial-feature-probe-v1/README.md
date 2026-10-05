# Synthetic spatial-feature evidence

This record uses only the established 7×7×8 analytic native fixture. The probe
executed nine native transitions and no gradients, policy evaluations or patient
loads. Eight focused tests passed in 0.06s; the probe took 0.363s. Numerical
thread limits were one. No public or production result was changed.

`probe/report.json` contains exact observations, coordinate descriptors,
post-cut summaries, all final legal continuations, model/source identities,
coordinate-check errors and both counterexamples. The two first cuts have equal
RAW inputs and first reward 4.33, but best two-cut returns 39.43 and 39.68. The
candidate coordinates and proposed state summary distinguish this pair.

Translation, RAS-to-LPS and row-order errors were zero; rotation/translation
candidate error was 1.11e-15 mm. These are descriptor checks, not a transformed
whole-engine validation. Different abstract cavity masks still share the same
eight summary values; native reachability of those masks is unverified. Source
axis reindexing changes the declared tangent basis and descriptors. No general
learning benefit or Markov completeness follows.

Later independent review found an additional conditioning limitation. At inward
normal `(1, 1e-8, 0)` with identity affine, rotation roundoff changes the selected
source tangent axis from 1 to 0 at its cutoff, producing a **2.00000000034 mm**
descriptor change. The recorded invariance result applies to the original
regular fixture, not to ill-conditioned axis selection. The separate
[counterexample receipt](../native-spatial-feature-independent-v1/conditioning-counterexample.json)
preserves the inputs; the [review](../native-spatial-feature-independent-v1/README.md)
also confirms that the 0.25 conditional-return gap comes from partial-contact
cost. The proposed eight summaries omit partial-contact history, current tool
and remaining action budget unless those are retained separately. They are not
a Markov-complete state. This is a later documentation finding; the original
report, execution receipts and source archive remain unchanged.

`source-capture.json` binds exact source/test files and the base commit;
`execution-receipt.json` retains exact commands and tool outputs. The verified
`source.tar.gz` preserves those files with repository-relative paths; its file
hash and round-trip verification are in `archive-verification.json`. Extract to
a fresh directory to reproduce the commands, with the recorded local Python
environment and numerical thread limits. The temporary unpacked source copy
was removed only after every archived file matched its captured SHA-256.

The source descriptor lives outside `src/resectionlab`, has no registered input
profile and imports no learner. The probe constructs no policy. Further learned
experiments require a separate declaration and orchestrator approval.
