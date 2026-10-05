# Independent V2 spatial descriptor review

The repaired `synthetic-declared-tangent-frame-v2.1` passed all **30 independent
algebra-only checks in 0.08 seconds**, using one numerical thread. No simulator,
policy, optimizer, gradient, native transition, or patient was used. The tests
are `tests/test_native_spatial_features_v2_adversarial.py`; source hashes and
the observed invocation are in `repaired-review.json`.

The same tests initially produced **14 failures and 16 passes** against the
owner's preserved initial V2 source. `initial-negative-tests.json` retains the
complete output, source identity, and loader used to test that snapshot.
`initial-adversarial-tests.py` preserves the exact test bytes; they are unchanged
after repair.

Four validation gaps were corrected in V2 only:

- Direct frame construction could report acceptance for a basis, normal, or
  sine inconsistent with the declared reference, including a zero basis.
- Direct construction retained caller-owned numeric containers that could
  mutate the frame after creation.
- Finite extreme coordinates and affine volumes could yield infinite or NaN
  descriptors, including aggregate-volume overflow.
- Complex physical inputs could lose their imaginary components during a
  floating-point cast.

The repaired constructor checks consistency and stores computed immutable
tuples. V2 checks now reject non-real inputs and nonfinite projection or summary
outputs. The review also confirms stable direction normalization from the
smallest positive float64 subnormal to approximately 1e308, transported-reference
behavior through the declared rejection band, reference-hash changes when a
vector changes, and source-cell summaries under a signed/sheared affine with
cyclic source permutation and flips.

V1's numerical source, original report, and retained conditioning counterexample
were checked against the initial V2 input hashes and remain unchanged. The V2
fix does not supply external provenance authentication, a global cross-patient
tangent convention, a rigorous universal floating-point error bound, or a
sufficient Markov state. Physical references must still be transported with the
coordinate frame. No production or patient-study eligibility follows from these
algebraic checks.
