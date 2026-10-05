# V2 JSON-boundary repair review

The actual V1 run remains failed: all 12 attempted arms stopped at the durable-history boundary and zero comparisons were accepted. The earlier integration mocks used JSON-native lists and missed the actual native task's tuple-valued crop and geometry fields. Source commit `981f6c9` and every V1 output remain preserved.

The owner first reproduced the failure with real tiny analytical native tasks, the actual rollout writer and budget guard: STOP and a non-STOP episode both failed before independent audit. That unchanged regression file is pinned here alongside its retained original log (2 failed, 1 passed in 1.57 seconds).

The narrow repair compares **complete canonical JSON bytes** for saved, live and recaptured metrics. Mapping order and tuple/list serialization normalize consistently. No fields are projected out, no numeric tolerance or rounding is introduced, and nonfinite numbers are rejected. The only other production change is the runner's V2 version identifier. Geometry, policy, rewards, patient roles and budgets remain unchanged.

I directly ran the final owner-plus-independent suite: **61 passed in 1.55 seconds**, with log and JUnit retained here. Actual analytical STOP and non-STOP episodes reached accepted native independent audits. Counterfactual controls reject added/deleted fields, one-ULP float changes, altered integers and nested coordinates, and nonfinite values before audit. All earlier resource, source, identity and publication controls remain included.

This review used saved metadata and small analytical fixtures only. It did not read patient arrays or learned checkpoints, replay a patient, fit a model or authorize another run. No V1 outcome has been promoted. `review.json` retains exact input, regression and final source hashes; the earlier review receipts remain unchanged.
