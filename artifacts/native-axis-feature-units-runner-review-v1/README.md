# Independent paired axis runner review

The final combined check passed **45 tests** (18 owner + 27 independent) in **12.38 seconds**, with numerical thread limits set to one. This is a test duration, not a performance benchmark; the owner's equivalent check ran concurrently. Two constructed cube runs each performed one optimizer update from two optimization episodes, with complete initial and latest selection panels and six geometry audit receipts per arm. No public patient was loaded or executed.

The independent review owns `tests/test_native_axis_feature_units_review.py` and this evidence directory. Production repairs belong to the runner owner; this review changed no production module, declaration, historical result, or public release gate.

## Preserved negative evidence

- `initial-negatives.log/xml`: four failing regressions in 5.80 seconds. The draft accepted a changed algorithm with stale contract checksum, coherently resealed action units or partial-contact semantics, and an incomplete Adam state inventory. `initial-runner.py.txt` and `initial-review-tests.py.txt` preserve that draft and the exact attack tests.
- `second-review.log/xml`: 18 passes and five failures in 7.49 seconds. Four failures showed completed publication accepted contradictory worker cancellation, peak memory, elapsed time, or enlarged budget. The fifth was a review harness error: attempted mutation of a tuple. `contract-repaired-runner.py.txt` and `expanded-review-tests.py.txt` preserve this stage. The harness now replaces the complete generator vector; no production fix was needed for it.
- `final-combined.log/xml`: all 45 checks pass after the runner repairs. The independent attacks explicitly recompute authority checksums where appropriate so semantic rejection is tested, including all twelve audit receipts and worker resource contradictions.

## Verified boundaries

The runner now checks the whole current learner contract and exact observation semantics, complete finite one-step Adam state, the same paired initial trainable tensor hash and distinct behavior/profile hashes, exact role/generator/seed vectors before calling the trainer, and actual initial checkpoint bytes before the first scored transition or gradient. The last initializer check occurs after construction of fresh Adam; it does not claim an earlier boundary.

The checks retain initial, latest and selected returns separately, retain the initial checkpoint on a tie, join decisions to returned transitions and current action IDs, and require complete six-episode evidence for each arm. Final export cancellation or source failure preserves diagnostic payloads and failure accounting without eligible publication. The completed pair revalidates native audits and checkpoint/history records and refuses resource records contradicting the fixed worker budget. Launcher wall time is reported separately and is not incorrectly constrained by the worker-only cap.

## Reproducibility and limits

`final-combined-execution.json` contains the exact command, environment, checkout revision, source hashes and process duration. `final-tested-inputs.tar.gz` preserves the tested numerical source, runner helpers, manifests and relevant tests/documentation; all archived members match the pre-execution hashes and no captured source changed during the run. Apply this archive over its recorded repository revision for reference artifacts that are deliberately not duplicated here. `review.json` indexes every retained artifact and records the final file scope.

The publication attacks use explicitly constructed test-only launcher/worker authorities around real tiny-run evidence. They validate internal consistency, not authenticity against an adversary who can replace every program and record. Resource boundary failures use copied records; this review did not run a memory-stress or patient timing experiment. Two small integration updates establish neither learning efficacy nor RAW/FEATURE_UNITS superiority. The planned patient comparison and full integrated regression still require the parent's separate source freeze and release. Existing frozen studies remain unchanged.
