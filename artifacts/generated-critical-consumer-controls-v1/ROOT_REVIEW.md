# Positive critical-constraint consumer controls

Root reviewed the generated resolver boundary and promoted the exact test source
SHA256 7b56a6aa89da405927b89099597d203209234d636b32cccc46dcfd2d1f8745d9.
All three canonical tests pass. No production behavior, admission rule or patient
record changed. The two route APIs load a saved generated case, retain the same
requested entry, target, tool and access geometry, then reject those routes when
the injected resolved vessel exclusion occupies the working tissue. Static and
native route results bind the exclusion; stale refinement/run bindings refuse.

The only injected function is resolve_critical_evidence. The generated input
never claims human review and cannot enter the real annotation registry. This
proves the consumers honor a resolved exclusion, not real registry admission,
real patient route accuracy, renderer vessel assets or learned-policy support.
The independent author's audit and root test output are retained alongside.
