# Independent generated vascular evaluator review

Decision: **GO for this frozen source and generated-control candidate only.** No blocking findings remain. This review does not authorize patient/reference-payload access, model admission or execution, native mechanics, training, deployment, or clinical injury/safety claims.

The reviewer separately inspected the candidate and the existing `research_estimate_planning`, `limited_observation_boundary`, `native_spatial_evaluation`, `functional_events`, independent native-history checker, and batched segment/box interfaces. Existing admission code was not changed. All review writes are under this ignored directory; the candidate was not edited by the reviewer.

## Frozen candidate

| File under `build/private-vascular-evaluator-preparation-v1/` | SHA-256 |
| --- | --- |
| `vascular_evaluator.py` | `30e9efaac9d0f4e311bedf9075362fcd488f1b31f76a3938f4224736d39e2eb7` |
| `test_generated.py` | `11e41868ae6ce8f79c124323f909a4b93ddc4dece94fff8c61268b90adf000cf` |
| `RESULT.txt` | `42853e1d2154f7bf75188367ac049a1b9e86a7e35452b5ceb0ca2bae685072c6` |

These exact hashes were checked before execution and remained unchanged afterward. `source-before.json` and `source-after.json` each contain the same 81-file snapshot, including every Python source under `src/resectionlab`, the reused generated fixture, the three frozen candidate files present at the start, and the independent test/runner. Both snapshot files have SHA-256 `655ce53030517106a32e0fa930ed6472758f288af2571137d5a739d089fd3b7f`. The observed HEAD after testing was `9477cf52e06a93415c03e5329e524167c23685aa`; this is contextual, not a substitute for the ignored-file bindings above.

## Evidence

The reviewer ran the author's 30 tests plus 14 separately written generated controls in a fresh ignored scratch tree, with Python bytecode and pytest plugins/cache disabled and numerical-library thread counts set to one. Result: **44 passed in 6.65 seconds**, process exit 0; harness elapsed 6.785249708918855 seconds. There were no changed snapshot files.

Independent controls compare capsule-cell contacts against the scalar segment/box oracle on rotated anisotropic grids; check forward transform direction, signed permutations and shifted ROI removal; check cubic-millimetre volume for anisotropic congruent cells; reject noncongruent centre rounding; reject parent traversal, symlink parents and FIFO seals without blocking; preserve a failed report when the seal changes during private loading; verify detached immutable reference arrays and transform-manifest mismatch refusal; and exercise a complete STOP-only plan. Audit guards reject subprocess/native launches, network connections, and patient/model array payload opens during the generated tests; the recorded prohibited-action and payload-open lists remained empty. No patient, model or native solver was run.

| Independent artifact | SHA-256 |
| --- | --- |
| `test_independent.py` | `0afaed6099d56b80c9bf8af481031d1928ece311deffc93b4820c0660928bb6e` |
| `run_review.py` | `8f12e7501f38503003db81a952ccfe90d728b72705424ce0b22d0dbfe0d0668f` |
| `run-receipt.json` | `2d033f45ead1f8cdc711ff726fbfe64b8a0734b8f1eaf6037e638df4dcb9e3a4` |
| `pytest-output.txt` | `1b89bc65e091cd341f845c1debf6c921b12be7fe8eb98f8de3afc4cf268813b1` |

The prefreeze review found lexical parent traversal, a stat/read size gap, an unbound generated person declaration, and possible blocking FIFO open. The author corrected these before the final freeze. The final source rejects `..` and symlink paths, uses `O_NOFOLLOW`/`O_NONBLOCK` with descriptor regular-file checks and a maximum-size-plus-one read, binds person identity to the existing planning declaration, and fsyncs the newly created output directory's parent. Generated controls cover the corrections.

After the independent run, the author added source-only evidence files. The reviewer checked all 80 manifest entries against current bytes and sizes; its 79 files shared with the independent snapshot match exactly. The extra evidence-writer helper was inspected but not executed by the reviewer. This supplemental check is saved in `supplemental-evidence-check.json` (SHA-256 `89d86a6558d46fb241c998fdb44791deadf957fe7ff07a506529eaf0a2a77d2d`). It does not change the executing candidate or the independent test result.

| Supplemental file under the candidate directory | SHA-256 |
| --- | --- |
| `candidate-source-manifest.json` | `ada6c01e509d4743b4107239bed4b48a4d215ba0c5740af1257925d5efa8f186` |
| `generated-controls.json` | `38e03f2276c75aa2b40f3b60a45856be22a716276a6e581bc68144bdf701b60c` |
| `save_candidate_evidence.py` | `149557d583273277cf330eb04fb5a55b4bcd4c3d1bd1e23c416cec4b15e3b755` |

## Scope and remaining conditions

- Complete strategy replay and independent full-history geometry finish before the one private loader call. An exclusive, fsynced strategy seal and attempt record exist first. No planner or nominal replay runs after private loading. Loaded reference or seal mismatch yields an explicit failed outcome and no retry.
- Shaft and tip exposure cover the full reference field of view, including positives outside nominal estimated tissue support. Union counts avoid duplicate cells across repeated sweeps. Unknown coverage and outside-FOV motion cannot certify absence; biological vessel-free and clinical injury values remain null.
- Removed overlap is separate from exposure. Its discrete cell mapping requires a signed permutation and integer offset within the reported `1e-8` index tolerance and uses the shifted planning ROI affine. Arbitrary noncongruent grids return unsupported/null overlap. The contact threshold includes the reported `1e-10 mm²` squared-distance tolerance. Neither result is continuous vessel injury or physically validated mechanics.
- Person, source, frame, mask, coverage, transform and lineage identities are declared content bindings. They are not acquisition authentication, an OS/Python sandbox, or admission of real-person material. Real-person, human-review and learned-initializer lineage remains unsupported here.
- Wall limits are cooperative checkpoints. Existing nominal replay and an arbitrary loader are not interruptible by this API; no hard wall/RSS guarantee is made. Process death or a final filesystem failure can leave a reserved/incomplete attempt rather than a terminal report; such an attempt is not success evidence. Any real execution needs its own reviewed supervisor and admission.

Approval applies only to the frozen hashes above. Later source edits or a broader execution scope require a new review.
