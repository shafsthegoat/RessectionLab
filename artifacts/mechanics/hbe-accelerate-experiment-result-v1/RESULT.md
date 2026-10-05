# Eighteen completed solves; fixed mesh-convergence gate failed

The single released Accelerate attempt stopped at the pre-calibration aggregate numerical gate. All eighteen reference/scaling solver calls completed with exit 0 and passed their individual numerical checks. Three mesh criteria failed. Both fitted confirmations remain unexecuted. No calibration or held-out access was attempted, and no modulus fit, prediction freeze or physical-validation result was produced.

The immutable execution source was commit `6486dbcfb0395a4c5b9df3f140b08141d0d96fd7`, archive `fd3a72a9…`, with final release `edb91bdb…` and repaired runtime identity `13c4f60c…`. [outcome.json](outcome.json) binds the exact source/archive/release, original raw result, state, aggregate report and supervision receipts. Existing meshes, law, loading, CSV assumptions, solver settings and tolerances were unchanged; no retry or substitute run occurred.

## Failed criteria

The displacement metric is the maximum Euclidean difference at the fixed physical probes over sampled loading states between N8 and N12 meshes. The tension reaction-trend criterion requires the N8→N12 maximum force difference to be smaller than the preceding N4→N8 difference.

| Declared metric | Actual | Required |
|---|---:|---:|
| Compression mesh displacement difference | 22.356144373333107 μm | ≤ 8 μm |
| Tension mesh displacement difference | 23.25768976023929 μm | ≤ 8 μm |
| Tension reaction refinement trend | 0.22774250190189804 mN | < 0.19948362350659735 mN |

Exact SI values, limits, comparison operators and all thirty aggregate observations are retained in [aggregate-gates.json](aggregate-gates.json). Thirteen of sixteen mesh checks, all eight load-step checks and all six stiffness-scaling checks passed. The tension absolute reaction-error check passed; its refinement-trend check failed. Passing individual solves and these other comparisons does not establish adequate mesh convergence.

`reference-numerical.json` was written before the aggregate check raised `ValueError: Numerical error gate failed`. The state therefore has no successfully returned `reference_numerical` binding; the saved report remains preserved and is explicitly bound by this collection. These are saved simulation outputs, not measured specimen responses. This collection did not rerun the primitive or mechanics evaluation.

## Resources and preservation

The recorded supervised worker lasted **396.3033522500191 s**; its experiment state reports **394.0622269170126 s**. Recorded solver durations sum to **338.5155295818113 s**. The slowest call, `torsion_pos:N12:S120:reference`, took **41.28799983300269 s**, below the unchanged 90 s per-call limit. No time, memory or output cap caused termination. Supervisor exit was 1, with no kill reason or cleanup error.

Sampled peak process-group RSS was **253,722,624 bytes** (241.96875 MiB). The output watcher observed **105,300,864 bytes** maximum active output and **997,830,113 bytes** maximum common output, 92.9302% of the unchanged 1 GiB cap. The actual attempt retains **169 files / 914,390,988 bytes**. Sampling may miss brief intermediate peaks; recorded values are not exact continuous maxima.

[raw-output-index.json](raw-output-index.json) binds every raw byte. The whole attempt is now read-only, without compression, removal or content edits. [preservation.json](preservation.json) confirms all 215 launch-baseline inputs, the fourteen archived source files, thirteen original Skyline runtime files, 52 original mesh-preparation files, 37 new deck-preparation files, and both original Skyline attempts (14 and 23 files) remained unchanged. Every per-run retained-file hash matches its recorded receipt, and all eighteen aggregate run objects equal the saved individual readouts.

[access-chronology.json](access-chronology.json) records all four access flags as false, no access ledger, no fitted-input directory and no calibration/fit/prediction/held-out outputs. The sealed measurement ZIP was checked only as a whole-file hash; no member was opened during collection. Further measured validation remains unperformed because the fixed numerical gate did not pass.
