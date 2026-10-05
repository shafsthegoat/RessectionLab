# V2 harness repair: prepared, not executed

V1's actual attempt remains failed under `artifacts/independent-native-batch-history-v1/`. Its first scalar audit passed, but the validation gate compared an empty tuple from `NativeRemovalAudit.to_dict()` with an empty list from saved JSON and stopped. Batch and the final scalar phase were never run.

The new controls instantiate the real audit dataclass from the historical certificate and exercise its JSON serialization roundtrip without constructing a scene or running a geometry check. All three phase variants failed against the unchanged V1 runner; [pre-repair-tests.txt](pre-repair-tests.txt) and [pre-repair.json](pre-repair.json) preserve that evidence.

The repair compares exact canonical JSON bytes. One-ULP volume changes, source changes, rejected audits, and integer/float representation changes still fail. Runner changes are limited to that comparison and selecting/fencing the separate V2 declaration. Numerical sources, scalar default, scene, physical tool, history, observational hooks, phase order and resource limits remain unchanged. The declaration additionally binds the V1 failed receipts.

The repaired owner and existing independent orchestration controls passed **41/41 in 0.34 s**. [repaired.json](repaired.json) binds the exact tested files; [repaired-tests.txt](repaired-tests.txt) is the raw output. These are dataclass/static/mocked controls, not a complete-history timing experiment. V2 requires a separate reviewed commit, immutable archive and root release before execution; no V2 benchmark has been run.
