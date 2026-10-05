# Preserved real-boundary regression and V2 repair

V1 rejected all patient arms before independent audit because it compared loaded
JSON lists directly with tuple-valued live native metrics. Even STOP exposes the
mismatch at `metrics.crop.origin_voxels` and `metrics.crop.shape`; non-STOP histories
add physical coordinate and microstep tuples. The V1 patient outcomes remain
failed/null, and no patient was rerun during this repair.

`initial-runner.py` preserves the executed V1 numerical source. The unchanged
`regression-tests.py` contains new actual tiny-task checks through the real episode
writer, budget and independent geometry evaluator. Before the repair, both STOP
and non-STOP completion checks failed at the same boundary; the changed-scalar
negative control passed (2 failed, 1 passed, 1.57 seconds). The logs and JUnit retain
that result. This is a small analytical software fixture, not a patient model or
learning experiment.

V2 compares the whole saved/supplied/recaptured metrics through strict canonical
JSON bytes. It normalizes dictionary order and JSON sequence representation only;
all fields, exact numeric values, and nonfinite refusal remain enforced. The
complete owner and existing independent suite then passed 54 checks in 1.50 seconds,
including real STOP/non-STOP independent audit and rejection of a changed saved
scalar. `verification.json` binds those logs, sources, tests and the read-only
patient failure receipt used for diagnosis. Separate protocol review adds further
canonical-value adversaries without reclassifying the old patient results.
