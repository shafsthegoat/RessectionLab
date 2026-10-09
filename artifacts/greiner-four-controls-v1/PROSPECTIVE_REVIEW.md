# Independent fresh review: remaining three Greiner controls

**Decision: GO for a separate root release of exactly the three never-attempted controls. No blocking finding. This review is not an execution release.** Root must bind the final checkout HEAD, this report, and the exact preparation hash in its separate release. The frozen candidate's preparation and template retain `source_commit: null`; the template remains HOLD.

Reviewed candidate: `build/greiner-native-remaining-three-release-v1`. Review started at HEAD `76cdcc6fef47580dd3b4aa4e08460bb1ca4bdb13`. The final observed HEAD was `1ede7500a1771323b3cf817cdb14906e8fee9bf5` after root's artifact/status commits; all seven tracked executing helpers match their bound hashes at that HEAD. This review does not require root to reuse that observed HEAD if a later final commit has the same bound executing inputs.

## Authorized prospective scope

The launch order is exactly `onearm_shear_dt005` (104 steps), `zeroarm_shear`, `zeroarm_volume`. The spent `onearm_shear_dt010` case is absent from the launch loop and prospective native deck map. There is one permanently reserved attempt directory, one call per case, no retries, sequential calls, and first-negative stopping. No new `attempt-01` directory existed at review completion.

The aggregate allowance is 90 seconds and 24 MiB. Per-case wall allowances are 45/10/10 seconds, output is 8 MiB per case, sampled process-group RSS is 256 MiB, and numerical threads and concurrent native calls are one. The original one-call usage is charged exactly once. No cap increase, coarse rerun, altered mechanics, fit, alternate solver, measured response, patient data, or model training is included.

The checker is byte-identical to the previously approved actual-deck checker. All three prospective adapted decks are byte-identical to the corresponding never-attempted original-release decks. Numerical parameters, time steps, recurrence, mechanical boundary conditions, checker thresholds, stress/reaction checks, residual criteria, and continuum envelopes are unchanged. The refinement comparison consumes the authenticated saved coarse result and the new fine result; it does not launch coarse again. Its convergence ratio remains explicitly diagnostic.

## Original negative and supplementary readout

The original receipt and summary remain `failed_or_incomplete`, with the exact inventory-mismatch failure and `native_provenance_verified: false`. Their bytes, and the original root release, match the copies preserved in `artifacts/greiner-first-control-inventory-negative-v1`.

The supplement authenticates the complete nine-file original attempt (503,011 bytes), exactly one original native attempt, exit code zero, command and thread environment, runtime/profile, original and actual decks, all 174 original audit entries, and 173 currently rehashed original input files. It then replays the unchanged checker over the saved logs and rehashes retained outputs. The replay passes all 53 saved states with zero failures and makes zero new native calls. It neither rewrites the negative nor licenses another coarse attempt.

The retained default plot is `onearm_shear_dt010.xplt`, 40,575 bytes, SHA-256 `86a8153c5dc843aac0acfe21f3ca40bda60c4c09f77634ae372e7885a95a3231`. Its original terminal receipt already contained this binding. The supplement binds the accepted runtime source inventory for `FECore/FEPlotDataStore.cpp`, `FECore/FEAnalysis.cpp`, and `FEBioLib/FEBioModel.cpp`; their default plot behavior and the absence of deck overrides explain the file. Plot bytes are retained and hashed, not used as a material-law acceptance endpoint.

Prospective output acceptance requires the exact input/log/console/receipt inventory plus the expected case-named `.xplt`. Missing, empty, linked, oversized, or additional output fails. Final case and sequence audits detect later output changes, including plot mutation.

## Cleanup failure review

The reused process supervisor catches signaling `PermissionError` and reap `TimeoutExpired` in its cleanup block, preserves their exact type/message in `cleanup_error`, sets a failure reason, and writes a durable failed terminal receipt. The sequence retains that error, sets `native_provenance_verified: false`, and does not launch later cases. It keeps the launched PID and does not record successful clearance, kill, or reap when those operations are unproven.

Fresh tests exercise both cleanup exceptions through the full sequence. A failed signal does not falsely report that waiting occurred. A reap timeout records the attempted wait and leaves the exit code unresolved. These failure paths establish a durable negative; they do not establish that surviving workers have disappeared. Process objects and signaling were fakes throughout this review.

## Independent verification

- All seven writer-supplied candidate hashes matched before review.
- All 197 snapshotted candidate, original, runtime, and reference files retained their exact hashes and sizes after review.
- The complete source-only preparation replay authenticated 115 runtime/profile inputs, ten executing sources, nine constitutive source pairs, all three decks, three default-plot source bindings, the original negative, and the exact frozen supplement.
- All 30 candidate tests and seven fresh independent tests passed: **37 passed in 1.31 seconds**. `subprocess.Popen` was replaced by a rejecting guard during this test run; no real process could be launched by the tests.
- Fresh coverage includes full-sequence cleanup failure persistence and stopping, late plot mutation, linked/oversized expected plots, exact saved-only preparation replay, and one-time combined call/time/output accounting.
- No native solver, model, patient, measured-response, acquisition, or training run occurred. Review writes are confined to this ignored review directory. No tracked edit or commit was made by this reviewer.

Evidence: `tests.txt`, `test-status.json`, `test_independent_review.py`, `writer-hash-verification.json`, `before-hashes.json`, `after-hashes.json`, and `review.json` in this directory.

An inherited reporting detail is nonblocking for the execution limits: `final_active_output_bytes` and `combined_retained_output_bytes` are captured before the last summary serialization. The supervisor separately rescans after that write and enforces the actual 24 MiB aggregate cap. Root should total the finalized retained tree when reporting an exact final byte count. The old 503,011 bytes plus the full new 24 MiB allowance remain below the original 32 MiB ceiling. RSS and active-output monitoring retain their explicitly stated sampling limitations.

## Frozen candidate hashes

| File | SHA-256 |
| --- | --- |
| `supervisor.py` | `b49afc0335f9e5b9e44b11221a85041a6a25aa32a9a33696532fddb09718bb87` |
| `checker.py` | `badc544010749274301d0a528ed04f49385605c5b82068670e7bf26d79b276b6` |
| `saved_coarse.py` | `2baaee70b1a6438e55a211b970803ab605c72d13aaf473e3bed95971f83f23cf` |
| `preparation.json` | `7c2e7542e7c432350d44405b43c16a92be464df4a14d058f584e3fe10224e345` |
| `saved-coarse-supplement.json` | `4367c2d0d12f9975e69363202c88b106bbcd01f689225615268a01ddf9c8f532` |
| `test_supervisor.py` | `554a71ed7e8090d87025822ae6929a1af1f605ed36311fa752c5595b20fb054f` |
| Candidate `REPORT.md` | `0ae08d9cded22d5bd8aefe5dec6ddd4371073c8ed830066985f9db3b6a1e02d1` |

This GO concerns arbitrary-constant numerical software controls only. The source-matched PK2 history and continuum comparison do not establish physical validation, measured-specimen accuracy, patient properties, clinical suitability, or readiness for substantial RL. Literature data schema and measurement chronology remain unresolved.
