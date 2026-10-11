# One targeted final-readout calibration

The completed Q64 experiment regressed at the first decision in both productive
TRAIN cases. Its fitting loss was still falling at update64, and the head badly
underpredicted a logged initial action. An exact-state label-conflict diagnostic
explained only0.176% of residual fitting MSE. The next candidate tests whether
its existing frozen hidden representation supports a better value readout.

One weighted float64 SVD solves a minimum-change correction of only the final
linear readout. The relative cutoff is fixed at1e-8. All original497 recorded
labels and weights remain; STOP is exactlyzero and excluded from gradient/solve
contributions without changing its denominator. Unidentifiable readout directions
retain their old coefficients. The deployed float32 objective must be finite and
non-regressing. Encoder, hidden head, old Q64 and corpus remain untouched. No
new optimizer or encoder forward is needed for calibration.

Eight generated tests pass in1.72s. An independent scalar normal-equation and
structural-nullspace oracle also passes, including the actual generated DTO
wrapper, deployed float32 error and frozen-state checks (1.749s parent). These
verify implementation, not patient policy quality. There is no sweep or tuning.

The next fixed experiment pays one feature reconstruction/cache fill and
reproduces Q64, then compares fresh GREEDY/Q64/CALIBRATED routes on the same four
TRAIN cases. Winning against Q64 alone is insufficient. Promotion requires no
casewise target/return loss or outside/contact increase against greedy and a
strict aggregate route/work gain. The original negative comparison remains
unchanged. Unlogged-action extrapolation and physical validation stay unresolved.
