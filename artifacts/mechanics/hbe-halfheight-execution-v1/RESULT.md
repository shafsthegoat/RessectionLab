# Half-height axial equivalence: passed

All four predefined compression/tension comparisons at N8/N12 passed, each at
all 61 states. Whole-cell extraction, full loading coordinates and full-field
reconstruction preserve the saved full-model result within the original limits.
Independent review replayed every readout and separately reconstructed nodal
motion and raw reactions from text logs.

The complete solve phase took **40.9184 seconds**, with **164,626,432 bytes**
sampled peak memory. Four native calls took 1.925–7.818 seconds each. There
were no retries, remeshing, measured-curve reads or model fitting. The largest
node discrepancy was **4.243e-14 m** against **1.134e-8 m** allowed. All original
numerical per-run checks also passed. See [compact metrics](solve-summary.json),
[original supervision](solve-result.json), [state](solve-state.json) and
[independent review](solve-review.md). Full logs remain at their bound local paths.

This supports a cheaper representation of this particular axial symmetric
model. It does **not** repair the earlier spatial-convergence failure or validate
brain deformation, cutting forces or patient properties. The 1000-Pa modulus
remains a declared numerical gauge; response curves remain closed. Historical
native timing suggests a benefit but the single observations do not establish a
repeatable speedup.

Next: declare and review a separate spatial-refinement experiment using the
unchanged physical probes and convergence budgets. No automatic finer run or
calibration follows this result.
