# Independent review: tiled private vascular contact integration

Decision: **GO for root to apply the exact four-file candidate patch below.** No blocking source or generated-control finding remains. The patch preserves existing private-reference admission and removal gates while replacing contact-cell sets with tile-local unions. It grants no patient admission or new benchmark run.

All reviewer work is ignored-only. No tracked source edits, patient/model payload access, native solver, network activity or full profile rerun occurred. After the corrected six tiny cases terminated, Python/tests were paused for root's trained-model quiet window; this report was then completed as text.

## Exact reviewed candidate

Base private evaluator SHA-256: `a4ca5a710fa6b8f02251e7d30d5e3fc6bf3602a8f6b2ca9ce7af6e809281856d` (root source integration `abbc3e31ef2dd9123e470a31b3701eb375f85424`). Candidate directory: `build/private-vascular-streaming-integration-preparation-v1`.

| Candidate file | SHA-256 |
| --- | --- |
| `stage/src/resectionlab/vascular_contact_streaming.py` | `fa2c09788f002c871575333018d6a11e309b48878045625e16d79fc39d75ebed` |
| `stage/src/resectionlab/private_vascular_evaluation.py` | `df588f2237a2a52b9bdc8e3accbae38a08b54bdc0c6d68abbea980279179d2e8` |
| `stage/tests/test_vascular_contact_streaming.py` | `c23c191bf22181705a2408b3fc4c8880cb2f0aca2393e2c19718e534073a9acf` |
| `stage/tests/test_private_vascular_streaming.py` | `f8fd5458c589b6ab529bba11e056fb70bf24bef0e94fe1b710880e96f5d16791` |
| `source.patch` | `688bd80641fe0584dfc0dd4726aabe6856d51c9f68c1c39674fa07a3a542c101` |
| `candidate-pins.json` | `332cbbe3cbdccce5a4d5733cc2d03a865abc09a767d16a1f8562ce6324e18ffb` |
| `source-delta.json` | `d36a906159e0ae82279f60626f5aa2a432cd0ecd7ea4cb800f5035f68ef9b90b` |
| `RESULT.txt` | `afdf05fd91eb99584147f05eea8674e4518139e900b051c09f7959a2d0781c19` |

The patch adds the reusable geometry module and two generated test files and changes only the private evaluator scoring implementation/import. The matched-method adapter is unchanged. Root should compare the four promoted files to these pins; staged testing changes only the imported evaluator's `ROOT` value for actual repository/build output paths, not its source expression.

## Boundary and geometry findings

Independently inspected the extraction against the measured predecessor and the private evaluator diff. AST controls establish that all existing definitions except the removed dense `_capsule_cells` helper and the changed private scoring body are unchanged. The shared `_evaluate_preflighted_private_vessels` signature, its pre-try checks/receipt creation, exception handler and post-try report persistence remain unchanged. Constants and `ROOT` expression are unchanged.

Generated person/role/source/frame/lineage restrictions, complete durable seals, nominal strategy replay and independent full-history validation still precede private loading. The loaded reference must match the expected immutable binding, seal and spec. The scorer builds unique ordinal IDs for decoded axial sweeps and returns per-sweep records in the original order. It evaluates full-reference-FOV contacts, including proximal shaft exposure beyond nominal support. Duplicate/repeated sweeps remain distinct rows without double-counting whole-tool contacts.

The internal sampler indexes only the already bound reference's mask and coverage arrays. The generic primitive validates a pair of bounded boolean vectors, copies them, rejects positive labels outside coverage and checks cancellation/time after the callback before accumulation. Tile-local shaft/tip/action masks are discarded after counts are accumulated. Cell contacts retain the existing independent segment/box oracle and `1e-10 mm²` tolerance; broad-phase padding only admits additional tile work. Current reference spacing `0.01..100 mm`, frame orthogonality tolerance `1e-9` and finite translated frames remain supported, without the predecessor profile's narrower radius/origin restrictions.

The existing separate `_removed_overlap` and `_counts` bodies are AST-identical, preserving ROI affine mapping, the `1e-8` congruence normalization tolerance, unsupported noncongruent removal status and null unknown outcomes. Private `MAX_REFERENCE_VOXELS=32³`, `MAX_PLANNING_VOXELS=16³`, `MAX_MICROSTEPS=256`, history/horizon restrictions and generated-only admission remain unchanged. The generic primitive's larger grid capability does not expand those private evaluator gates.

Successful reports add `contact_work`, `contact_budget` and `tile_pruning_padding_mm`. Failure remains a saved `evaluation_failed` result with no partial outcomes and no retry. The contact budget uses a 30-second cooperative ceiling while the existing caller's elapsed-time check is passed into every kernel checkpoint; the caller's smaller wall budget remains effective. This does not make a blocking loader or arbitrary generic sampler interruptible.

## Independent controls and preserved reviewer error

First independent command:

```text
.venv/bin/python -B build/private-vascular-streaming-integration-independent-v1/run_review.py
```

This independently reran all **74 existing/author controls** (private evaluator, matched evaluator, new kernel and new integration tests) and **19 independent controls**. The first result was **87 passed, 6 failed in 14.02 seconds**, exit 1. All **90** source snapshots were unchanged. The six failures occurred before geometry execution because the reviewer's fixture passed a NumPy scalar radius while `Capsule` explicitly accepts built-in `int`/`float`. They are reviewer fixture errors, not candidate implementation defects.

The original test source, stdout, exit-1 receipt and snapshots are preserved. Only the independent fixture expression changed from `r*min(spacing)` to `float(r*min(spacing))`. No candidate file changed. Only those six cases were rerun:

```text
.venv/bin/python -B build/private-vascular-streaming-integration-independent-v1/run_corrected_controls.py
```

Result: **6 passed, 13 deselected in 0.59 seconds**, exit 0; all **91** source snapshots unchanged. Thus all **74 existing/author plus 19 independent controls** have passing coverage across the two runs; this is not represented as one uninterrupted 93-test pass. No broader rerun followed the fixture correction.

The independent controls compare full-grid scalar oracle contacts across six uneven-tile, anisotropic, rotated/reflected, far-translated cases, including accepted spacing endpoints; verify per-part/per-action/whole unions and unique sampling; reject five sampler contract violations; test all three work-budget refusals before sampling; test callback deadline refusal before aggregation; verify STOP and wholly outside-FOV behavior; and exercise a zero-radius near-tile-face contact inside the squared tolerance. AST boundary controls and absence of model imports also pass. Numerical threads are 1, pytest plugin autoload/cache writes are disabled, and all arrays are tiny generated fixtures.

The author's separate eight-case saved whole-report parity evidence was also read and its comparison source inspected. Identity, rotated, partial, empty-partial, outside-FOV, noncongruent removal, shifted ROI and STOP reports match the old evaluator after excluding elapsed time and the three additive diagnostics. This author-run parity evidence is corroborating evidence, not an independently rerun whole-report comparison. Its SHA is `8edc9158d429fe714bcfab099bc198d4b7df7dd8f9599bf871bea4e1aa1a29fc`.

## Reviewer evidence and limits

| File in this review directory | SHA-256 |
| --- | --- |
| `test_independent-initial.py` | `843b11c10cfc51335fe021d682d754243777b1838dcca00916086bf842e2495f` |
| `test_independent.py` | `c7f87b1fee28ef9986999d87d05581501c92ae97cd19a06fa0c63846976b3f29` |
| `run_review.py` | `d799425f341fa5679027efb1dba9fbd415fd30fbda77df04d964188726b1743e` |
| `review_worker.py` | `954a43687fa4f2916a8e96879091ee4ca72e3742175133b7bc6f0cb706dda501` |
| `run_corrected_controls.py` | `07e15bdf0cb4e7ac937b5ae7a907cb876f3b22a78f5442289f423fbdce0af0dd` |
| `run-receipt.json` | `5b0d196963798ca5cb432d371717b1f3452405c0690bfebffc9ab0f874cc730a` |
| `pytest-output.txt` | `acc84c20c90b271a2264e09c19c5c5fd134af2eb1445ca7f1ed1f069c791ce02` |
| `source-before.json` / `source-after.json` | `3eaebd832595f1f357389c6f251bc147659ef2fcddb67c0c7e060dfd090cbf1c` |
| `corrected-run-receipt.json` | `ae26489342cfa65446cff5444b938e64db2b36fec64ccac7df16860483c9b1c8` |
| `corrected-pytest-output.txt` | `a2027a2e77f4181f0e185b739a017ce2851da8916e2896c2db08e9b4f31e1b94` |
| `corrected-source-before.json` / `corrected-source-after.json` | `ec31651b6d39c8d6aca0292cf5e2c214378b67f4cf40ff2ca4c7c57b4ba7bae9` |

The measured large-grid performance belongs to the frozen predecessor profile, not this newly extracted module. This integration was tested on tiny generated fixtures only. No measured speedup, real-source QC, registration validity, biological-vessel completeness, injury probability or patient admission follows from this GO. Existing full-strategy/source/frame/lineage and removal requirements remain necessary. The sampler interface is a trusted-caller geometry boundary, not an OS/Python security sandbox.
