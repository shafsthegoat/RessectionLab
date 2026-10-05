# Prepared baseline supervision repair

Prepared and unrun. A new explicit root release binding these exact source and
declaration hashes is required before any patient access. V1 and its failed
supervisor receipt remain unchanged; its worker diagnostics are not promoted
to a successful run. No B/V motion or during-US input is permitted.

V1 queried anonymous RSS rows and rejected any zero before distinguishing an
exited worker. A constructed exiting-worker/zero-RSS case reproduces a possible
failure path. **The actual V1 cause is unknown:** failed-query stdout, stderr and
return code were not retained. This repair does not retrospectively classify it.

V2 queries PID, RSS and process state. Each response, return code, parsed PID
rows and direct child `poll()` result is flushed to `resource-probes.jsonl`.
Parent RSS must be present and positive. Missing/zero child RSS fails while
the worker is alive; it is allowed only after direct process polling confirms
exit. An exited worker with nonzero exit status still fails the overall run.
Malformed, duplicate, unexpected-PID, failed, oversized and timed-out queries
fail regardless of whether the worker subsequently exits. Query diagnostics
are bounded to 16 KiB per output stream. Process state alone never establishes
successful exit. Live workers are killed and waited on after monitor failure.

This remains sampled parent-plus-child RSS, not a continuous memory maximum.
A final exit can occur between samples, so successful supervision does not
prove an exact peak bound. The declared 115-second/2-GiB/one-thread caps and
no-automatic-retry rule are unchanged. No inference or solver is involved.

The fixed input allowlist, byte bindings, control authentication, immutable
source guard, proper all-point rigid fit, rank/numeric checks, leave-one-out
calculation and six native-plane render functions are byte-structure identical
to V1. An AST/declaration test verifies this. No threshold, registration
alternative, image plane or parameter changed in response to the diagnostic
residuals. The coordinate justification is copied unchanged.

Twenty-six geometric and mocked supervision checks pass in 0.36 seconds:
known rigid transforms and physical slice sampling; live versus independently
confirmed exit with absent/zero child rows; malformed/failed/timed-out probes;
supervisor termination/exit receipt behavior; unchanged numerical methods.
No real child, patient arrays, coordinates or registration were executed for
these tests. Independent repair review and any separate execution release
remain pending.
