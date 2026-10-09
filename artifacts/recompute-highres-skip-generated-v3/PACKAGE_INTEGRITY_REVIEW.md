# Independent integrity review: InstanceNorm negative and recompute-skip V3 packages

**GO for both compact evidence packages.** This is a read-only package integrity and claim review. No model, checkpoint, patient image, or generated full-network inference was used.

| Package | Index SHA-256 | Indexed payloads / bytes | Direct byte-matched copies |
|---|---|---:|---:|
| `build/limited-input-guard-design/inplace-instance-norm-v1-negative-package/` | `0184e9007b5f9434722cfec6d19deb4979b79ba2fed1d142948fb8900908c03f` | 5 / 12,026 | 3 |
| `build/limited-input-guard-design/recompute-highres-skip-v3-generated-package/` | `b9307fdfa0a6631a22ce7d87f65729dc81a8469c3175f0373ad003461a482177` | 17 / 83,412 | 15 |

I independently enumerated each directory. Every indexed path exists exactly once, with its stated size and SHA-256; neither package has an unindexed file or symlink. The direct-copy counts cover frozen source and test files, independent reviews, the V1 mutation counterexample, V3 source and saved worker/supervisor receipts. Each package's own `RESULT.md` and tiny-test transcript are newly assembled payloads and have valid indexed hashes. No `.npy`, weight, patient image, clinical annotation or other bulk binary payload is included. A limited text scan found no private-key block, GitHub token prefix or bearer-token marker.

The norm package keeps the negative conclusion explicit: ordinary generated inputs are close to native InstanceNorm, but a large-offset/low-variance case differed by `1.7628903` with 43 **activation** sign disagreements. The operator cannot establish exclusive ownership from `_base` checks, and model attachment remains on HOLD. It makes no patient or resource-benefit claim.

The recomputation package correctly preserves V1's custom stage-one mutation counterexample, V2/V3 guard evolution, and the bounded generated tiny six-stage result. Its `RESULT.md` says exact parity was asserted inside the reviewed worker, while the raw logit arrays were not retained for independent reload. It does not claim measured RSS saving, speedup, trained checkpoint performance, patient validity or clinical evidence. It also states that the one-off tiny supervisor's uncertain-identity/wait-timeout failure paths are unsuitable for a trained-model controller. The saved child success receipt and the independent audit support only that narrow untrained software-graph conclusion.
