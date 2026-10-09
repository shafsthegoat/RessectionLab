# Independent strategy-record review

**GO for the exact source/test patch below. No remaining blocking finding.** Review used only ignored files and small generated fixtures. No tracked edit, patient data, model load/training, native solver, or heavy run occurred.

The patch adds detached JSON export/import around the existing `FrozenResearchPlan`. Export preserves ordered complete physical history, tools, geometry, native coordinates, affine, microsteps, removed/contact cells, and the resulting nominal terminal state. STOP and horizon completion, prior tool state, and cropped ROI coordinates remain explicit. Sparse changes require the exact bound input specification to reconstruct remaining support and connected free space; these are modeled consequences, not observed surgical state.

Both export and import reconstruct a fresh nominal-only world from permitted estimates. Neither takes an evaluator loader or reads withheld target truth. Private outcomes/rewards are excluded from the physical record, while existing unavailable clinical fields remain `None`. A generated private-target swap changes only subsequent evaluator scoring, with the sealed route unchanged. A resealed but false physical history still fails replay. Imported actions and method accounting do not alias mutable input records.

The first candidate accepted a float or Boolean horizon because the legacy plan constructor permits them. The revised new replay helper rejects non-exact integers before export/import. Two added candidate tests cover both entry points for `2.0` and `True`; the legacy constructor remains unchanged. Fresh tests also reject numeric representation changes in terminal/replay counts without relying on Python's Boolean/integer equality.

**105 tests passed in 6.73 seconds:** the existing 85 tests in `test_research_estimate_planning.py` and `test_limited_observation_boundary.py`, the exact patch's 14 new tests, and six independent tests. The patch was applied only to an ignored overlay. Its patched module is byte-identical to the frozen candidate and was loaded under the canonical package name for all regressions. Subprocess creation was blocked during tests. Source/test hashes remained unchanged throughout.

| Reviewed item | SHA-256 |
| --- | --- |
| Baseline tracked module | `1c9d862018a7432915fe94942e7d6cde26167be9bb56d32de7dfe66909b1dd23` |
| Final candidate module | `bd687f2118d89f308d35f34a2cd2fa4beb21173adb4c9c356f178cdd776a04f6` |
| Candidate test source | `fd60fd4ba3d583d589f996e3703197eb1823b6e3262715f791865523304af394` |
| Combined source/test patch | `02bd373bb47161c03ae111f48663fb3d60ebf5846804ed9aef9e4625ee6c3cac` |

Candidate location: `build/limited-observation-next-experiment-v1/candidate.patch`. Evidence in this review directory: `test-output.txt`, `test-status.json`, `candidate-hashes.json`, `patch-apply.txt`, `run_review_tests.py`, and `test_independent.py`. The test status retains passed test IDs and loaded-source hashes.

This is generated DEVELOPMENT recording/replay support. Content hashes protect integrity, not caller authentication or process isolation. Existing production admission is unchanged. Original method costs remain caller-declared, with nominal recording replays separately counted. No patient/generalization, clinical, physical tissue, or surgical feasibility validation follows.
