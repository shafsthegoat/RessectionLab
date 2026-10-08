# Terminal N36 numerical convergence diagnostic: implementation

October 8, 2026. The separately proposed N36 uniform-mesh diagnostic is implemented
using the existing FEBio framework. Five new files add exact N36 geometry/readout
bounds, the declared experiment and pure controls. All 17 inherited modules remain
unchanged. The N32 negative result and older extrapolation limits remain visible.

Independent implementation review passed 77 pure controls (56 owner and 21
independent) in 0.43 seconds. It verified exact source/declaration closure,
preparation/solve separation, source-bound publication checks, parser/resource
limits and all-state comparison. No preparation, native solve or measured-curve
access occurred in this implementation milestone.

One terminal mesh/solve is allowed: 60 seconds preparation, 1,500 seconds native
solve and 1,800 seconds aggregate solve phase, with declared memory/output caps
and one thread. There is no automatic retry or further uniform refinement.
The prospective endpoint forecast remains fragile; success is not assumed.

Actual execution requires a committed 26-file source archive and phase-specific
release. Solve binds both the accepted preparation result and its publication
check. Result file/directory synchronization and retained-byte scanning are charged;
the small closing receipt's own publication is explicitly excluded. This is not
an operating-system hard real-time guarantee.

A favorable conditional result can only nominate later temporal review.
`spatial_convergence_accepted` and `calibration_released` remain false. Patient
forces, injury thresholds and physical fidelity are not established by this test.
The independent receipt is `verification.json`, SHA-256
`32da494fd4afddc6f12bddb0815235bc95738a0691f854186913b29922b8f210`.
