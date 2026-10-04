# Explicit-reference spatial frame V2

This separate research revision addresses the V1
[conditioning counterexample](../artifacts/native-spatial-feature-independent-v1/conditioning-counterexample.json).
V1, its failed invariance case and every original receipt remain unchanged.
The validation repair is version `synthetic-declared-tangent-frame-v2.1`.
Eleven focused and 30 independent checks pass. The original V2 result and later
failing adversarial evidence are retained separately. There is no production
input profile, patient experiment or learned policy.

The caller supplies a `DeclaredTangent`: a physical reference vector, source hash,
coordinate-frame label and declaration identifier. The frame constructor requires
matching source and coordinate identities for the access geometry. It does not
infer an axis from the source affine and never falls back to another reference.
This checks metadata consistency; it does not authenticate an external source or
provide a universal tangent convention across patients.

Normalize the inward normal and tangent reference after scaling by their largest
component. Use their cross product to measure the tangent sine and construct the
second tangent; a second cross product supplies the first. Require sine greater
than `0.1 + 64 × float64 epsilon`. Values below the conditioning limit or inside
the inclusive roundoff rejection band fail with a receipt containing the actual
reference, normalized normal, measured sine and reason. Accepted frames record
the same inputs and their normalization factor, which is less than 10.

V2.1 validates direct frame construction against the declared reference, normal
and measured sine, then stores immutable computed tuples. It rejects complex
coordinates and nonfinite projected coordinates, source-cell volume or aggregate
summary output. These checks are local to V2; the original V1 functions and
archives are unchanged.

The 0.1 condition is a declared engineering limit, not a clinical angle rule.
The roundoff band is a conservative engineering margin, not a rigorous global
floating-point error interval. It prevents automatic axis switching because
there is no selection branch. Coordinate transformations must transport the
explicit reference alongside points, access center and normal. No unconditional
accuracy or acceptance guarantee at arbitrary floating-point boundaries is
claimed.

The tests target the exact saved proper-rotation counterexample: a declared
well-conditioned Y reference should remain consistent, while the ill-conditioned
X reference should reject both before and after rotation. They also cover the
rejection band, source/frame mismatch, missing references, positive direction
rescaling, translation/RAS-to-LPS, row permutation and source reindexing with a
preserved physical reference. Source reindexing must not silently redefine the
reference.

Candidate coordinates and the eight volume/centroid values reuse V1's pure
descriptor functions. The summary collision stays explicit. Partial-contact
history, current tool, remaining action budget, detailed geometry, topology and
candidate interactions remain missing or compressed. V2 makes no Markov-state
claim and does not resolve those limitations. Input scaling and actor/critic
changes would need separate declarations before any learned experiment.

## Initial measured algebraic result

The 11 focused tests passed in 0.09 seconds with numerical libraries limited to
one thread. The standalone probe took 0.00237 seconds and constructed no simulator,
policy or optimizer; it performed no native transition, gradient or patient load.
This single execution is a verification receipt, not a throughput benchmark.

For the exact saved V1 rotation counterexample, an externally declared Y reference
gives maximum coordinate error **2.22e-16 mm**. Explicitly declaring the near-parallel
X reference instead produces a rejection in both coordinate frames, with measured
sines `1e-8` and `1.00000000165e-8`. There is no fallback. The three tested points
inside the roundoff band reject; sine 0.099 rejects below the engineering limit,
and 0.101 accepts. The exact reference and rejection/acceptance receipts are saved.

The archived alias coordinates still produce distinct six-value candidate rows.
Their historical returns are read from the frozen V1 report and are not recomputed
by this algebra-only probe. The equal-summary/different-mask counterexample remains
equal under V2. Omitted contact memory, tool, budget and geometric structure remain
limitations; this result is not evidence of a learned policy improvement.

## Independent validation and preserved repair

The independent reviewer ran 30 adversarial checks against the preserved initial
V2 source: 14 failed and 16 passed. They exposed directly forged frame bases or
conditions, mutable constructor buffers, overflow from finite coordinates or
volumes, and complex components silently discarded during numeric conversion.
These initial failures remain saved in the independent evidence.

The V2.1 validation repair passes the unchanged 30 independent tests in 0.08s
and the 11 owner checks in 0.06s. Its algebra-only probe took 0.00343s and again
used zero simulators, native transitions, policies, optimizers, gradients or
patient loads. Fourteen explicit comparisons of saved accepted-case numeric
fields, alias coordinates, summary collision and boundary classification are
exactly equal to the initial V2 record. This confirms preservation for those
saved cases; it is not a universal floating-point proof. Source, commands,
receipts and comparison are under `repair-01`, alongside the untouched initial
evidence.

Evidence: [`artifacts/native-spatial-feature-v2`](../artifacts/native-spatial-feature-v2/README.md).
Implementation: `research/native_spatial_features_v2.py`; focused checks:
`tests/test_native_spatial_features_v2.py`; algebra-only probe:
`scripts/probe_native_spatial_features_v2.py`. The probe reads frozen V1 synthetic
coordinates and the independent numerical counterexample, constructs no simulator
or policy, and performs no new native transitions or gradients. Independent
review evidence is saved under
[`native-spatial-feature-v2-independent`](../artifacts/native-spatial-feature-v2-independent/).
No blocking finding remains in this bounded algebra slice. This does not release
any learned or patient experiment.
