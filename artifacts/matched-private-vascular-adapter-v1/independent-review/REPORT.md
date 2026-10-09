# Independent matched private vascular adapter review

GO for generated-only source integration by root. All 52 small controls pass in 14.71 seconds: 19 adapter controls, 30 existing single-method evaluator regressions, and 3 independent callback-mutation controls. Relevant source hashes were unchanged before and after testing. No model forward/training, patient/model/archive payload or network access occurred; Torch was not imported.

The initial adapter had a concrete cross-method reference substitution bug: the SEARCH callback could replace IL's caller-owned binding and loader after common-reference preflight. A generated reproduction accepted SEARCH with 2 positive cells and IL with 0 in the same completed batch. The repair copies the mappings and captures immutable per-method and common-world fingerprints before private access. It checks those expectations before each later loader/scoring and again at batch completion. Recomputed in-place binding fingerprints cannot authorize a change; late mutation of an already-scored binding invalidates the batch.

All four terminal method slots and complete strategy seals are durable before any private callback. All nominal replay and geometry preflight completes first; the scoring loop makes no planning decisions or nominal replays. Failed/abstained methods retain null slots, real STOP histories remain distinguishable, and private failures preserve the fixed denominator without exporting exception text. Pre-scored outcomes and nonzero/non-integer generated-session counters are rejected.

AST comparison confirms the original geometry and other evaluator definitions are unchanged. The extracted scoring body is identical apart from taking its previously captured expected binding hash. The existing single-method API retains its behavior and passes all 30 regressions. The geometry streaming work remains separate.

This is a generated interface control, not a learned-method or patient experiment. Existing GENERATED_T1 records are explicitly rejected rather than relabelled T1. Real-person/source/role admission remains unimplemented. Cooperative phase limits are not a hard supervisor; reference coverage is not biological completeness, and no clinical-injury or generalized repair claim follows. The boundary protects against accidental misuse, not hostile Python code.

The original outer test receipt records FAIL solely because its broad archive guard blocked three Python standard-library python312.zip import lookups; pytest itself passed all 52 controls. The unchanged original receipt and log are retained. test-classification.json records the explicit runtime-library classification, with zero patient/model/archive/network attempts.

Exact SHA-256 bindings:

- `matched_private_vascular.py`: `148c30d147c4f88f0e725575debb8b50b774e224e80d4b5bf9b18d90465945db`
- `private_vascular_evaluation.py`: `a4ca5a710fa6b8f02251e7d30d5e3fc6bf3602a8f6b2ca9ce7af6e809281856d`
- `test_adapter.py`: `82637681b30a9fa5edcc40e1902377f41ceb343a7c6b4ca367200840718eb258`
- `source.patch`: `4486189366259ff9c3f9de28230433e92ac2b94bf6d7b78bdec1402f2d5a5587`

Machine review: `review.json` SHA-256 `c54b0c22b7951bd1802396b9720592a9f5ac1284f5a138014209405902c1e2e1`. Supporting evidence hashes are included there.
