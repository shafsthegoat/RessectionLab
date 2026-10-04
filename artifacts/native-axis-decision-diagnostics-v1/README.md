# Saved decision diagnostic: one completed RAW update

[DIAGNOSTIC.md](DIAGNOSTIC.md) and [diagnostic.json](diagnostic.json) analyze all six prescribed initial/latest selection pairs. They reduce to three distinct observed states, repeated across the two deterministic seeds.

The actor outputs changed: 8, 17 and 12 action rows changed rank, including STOP, while the winning action stayed the same at each step. The top-two logit margins increased in each state, and saved value estimates increased by approximately 0.0299, 0.0318 and 0.0335. This distinguishes a changed learned policy from an unchanged selected route.

No exact repeated 15-feature float32 rows were found among the 26, 24 and 22 legal non-STOP rows in these three states. Exact feature aliasing therefore is not demonstrated on this saved path. This does not rule out aliases elsewhere, information loss in the representation, nonlinear saturation or other explanations; none is established causally by this diagnostic.

Inputs are pinned to the completed pilot's immutable source receipt, actual initial/latest checkpoint bytes and the prior independent saved-tensor forward audit. Pairing fails if the physical action IDs, ordering, raw feature/state arrays, mask or phase differ. There is no softmax, policy forward, simulator call, random draw, gradient, new world or outcome-based choice of a subset.

[verification.json](verification.json) records eleven passing constructed-record tests, the corrected test-only shared-list fixture issue, and exact gzip-only output reproduction. Ordinary concurrent workstation activity was unrestricted; no runtime performance claim is made. Existing pilot reports and original execution receipts were not edited.
