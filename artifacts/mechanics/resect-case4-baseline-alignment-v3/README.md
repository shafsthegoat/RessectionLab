# Prepared bounded termination confirmation

Prepared and unrun on patient data. A new root release is required. V1 and V2
remain frozen failed attempts with saved diagnostics; neither is relabeled.
Only supervision changes. Scientific inputs, proper rigid fit, leave-one-out
calculations, all six image views, access controls and thresholds are unchanged.

V2 recorded a valid `ps` response with child RSS zero and state `?E`, while two
direct polls still reported a live worker. The worker's successful exit became
observable later. That observation motivates one narrow confirmation step;
`?E`, zero RSS or a missing process row alone never proves termination.

For a valid response with positive parent RSS and absent/zero child RSS, V3
checks the direct child status. If still live, it calls `process.wait` once with
a requested timeout no greater than 50 ms or the remaining work deadline.
Confirmed exit is recorded separately from the original decisive sample.
Timeout retains failure and stops the worker; malformed, empty, nonzero-exit,
oversized or failed queries receive no grace. Nonzero worker exit still fails.
Every probe preserves stdout, stderr, PID rows, poll observations, wait budget,
duration and outcome. No query retry or patient rerun is automatic.

The existing 115-second total cap reserves 0.25 seconds for cleanup; queries
and grace use the earlier work deadline. Cleanup wait uses only remaining
total time, with a final elapsed-cap check even when the worker has exited.
Memory overages still fail. RSS remains sampled, with an unmeasured termination
interval: confirmation does not recover a continuous memory peak. The 50 ms limit bounds the **requested timeout**, not guaranteed observed
wall duration. Actual elapsed waits can overshoot under OS scheduling; each
record retains allowed and elapsed seconds, with overshoot given by their
positive difference. A still-live timeout remains failure. The overall
115-second cap includes cleanup and retains a failure on observed overrun.

Twenty-five owner checks passed in 0.57 seconds. They cover rigid geometry,
physical slice sampling, one-shot exit confirmation, live timeout, malformed
queries without grace, remaining-budget accounting, supervisor failure and
cleanup, plus unchanged numerical/access/render source structure and settings.
Two controls launched actual tiny isolated Python processes; only their
deficient `ps` observation was constructed. One exited and was confirmed after
38.915 ms; the other exceeded the requested 50 ms wait (52.828 ms observed),
failed and was killed/reaped. These are process controls, not patient simulation
or evidence that every macOS exit transition will succeed. No patient images,
landmarks, model inference or registration were opened/executed in preparation.

Independent monitor review and its separate evidence follow before any new
patient execution release. B/V motion and the during-US image remain closed.
