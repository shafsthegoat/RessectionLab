# Frozen TRAIN gradient diagnostic

One fixed ranking checkpoint reproduced all 29 original TRAIN readouts, with no
optimizer updates or parameter changes. This was a diagnostic of the previously
negative ranking experiment, not a new policy or transfer result.

The complete run took 96.580 seconds including supervision, at 1.377 GB sampled
peak. It made 29 forwards, 79 gradient reads, 5,184 native previews, four TRAIN
source visits and 58 collection/replay steps. Cleanup completed. SELECT/EVAL
remained closed. The independent saved-evidence audit passed 573 checks.

Ranking-only and full-objective gradients would each improve 12 teacher-versus-
chosen motion margins, worsen 12 and leave one zero at this endpoint. Removing
gate/STOP changes none of these signs. Same-state ranking helps 19 of 24 incorrect
choices, but corpus accumulation reverses 12 of those directions. This supports
local shared-parameter/objective interference; it does not establish what an Adam
update would do, prove deficient observations, or justify a universal STOP-loss
change. No learned improvement or physical/clinical validation is claimed.

See [the independent report](independent/REPORT.txt), [full scalar readout](result/gradient-alignment.json)
and [source/evidence index](index.json). Fresh certificate elapsed times changed
the regenerated cache seal; the audit verified all other certificate fields,
original traces, plans, contexts and endpoint observations. It retained this
audit-assumption correction rather than calling it a runtime defect.

The ten generated helper/runner controls passed. A wider baseline and measurement
suite initially gave 190 passes and 11 failures: one historical constructor test
passed a newly added opt-in field to the older API; ten temporary authentication
controls inherited the historical experiment's different Python runtime. Only
the test fixtures were repaired. All 43 tests in those two files then passed,
including explicit rejection of a changed runtime. Across the original suite
and repair, all 201 cases passed; original failure output is retained. Patient
protocols, source mathematics, rewards and archived runtime identities did not
change.

The current software control remains the generated near surface-contact episode:
an initial costly aspiration followed by a non-removing probe, return 0.708,
with full-tool replay and desktop export. It is already tested by the existing
suite. Probe contact supplies no measured sensor information. Search can inspect
the complete permitted nominal world while the actor uses compressed features;
shared available data does not establish equal representation. Private target
invariance is scoped to the tested boundary and does not qualify hidden support
or hazard use upstream of proposal certification.

The next capability is explicit persistent passive tool motion and observation
state, alongside a separate real-observation prediction test. Another optimizer
sweep is not the next milestone.
