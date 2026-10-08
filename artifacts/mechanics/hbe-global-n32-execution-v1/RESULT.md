# Uniform N32 mechanics: execution passed, accuracy screen failed

One N32 compression calculation completed on the existing HBE_01_03 specimen dimensions, preserving the original cylindrical idealization, bonded plates/free sides, half-height symmetry, Ogden exponent 2, bulk-to-shear ratio and three-field hex8 formulation. The 1,000 Pa shear scale remains a numerical assumption, not a fitted patient property. No measured biological force curve was accessed.

The new 28,233-node / 24,576-cell half mesh reconstructs 53,329 nodes / 49,152 cells. Independent preparation review checked coordinates, topology, reflection and deck assumptions. Preparation took **13.421 seconds** and 608,387,072 bytes sampled process-family RSS. The bare interpreter first failed to import SciPy before preflight or meshing; that failure is preserved. The existing project virtual environment then used the exact pinned Python binary. There was one meshing call and no meshing retry.

The single native solver call took **639.057 seconds**, below its separate 750-second limit. Full solve-phase time was **750.919 seconds**, below 1,200 seconds, including preflight, historical readout replay and final checks; result publication is excluded. Sampled family RSS peaked at 1,798,078,464 bytes below 3 GiB. Total retained output was 513,108,853 bytes below 2 GiB. Sampling can miss short memory peaks. There were no retries or remaining worker/native processes.

| Numerical endpoint | Measured from saved calculation | Declared allowance | Conclusion |
|---|---:|---:|---|
| N24→N32 maximum reaction change | 0.138140 mN | 0.724169 mN | Pass |
| N24→N32 maximum probe displacement change | 1.727312 µm | 8 µm | Pass |
| Latest conditional remaining-force indicator | 0.691702 mN | 0.724169 mN | Narrowly below |
| Two-latest-limit sensitivity envelope | **0.985189 mN** | **0.724169 mN** | **Fail** |

The envelope exceeds allowance at states **49–60**. It is a conditional discretization-model sensitivity measure, not a proved continuum error bound. The endpoint apparent orders are 0.330033, 0.482583 and 0.632920 for N8/12/16, N12/16/24 and N16/24/32. Their changes have not strongly stabilized. The oldest triplet has no admissible positive root at 35 nonrest states; those remain visible. Both latest triplets are eligible at all 60 nonrest states.

Independent saved-output review reproduced all five complete readouts (305 states) and the entire comparison exactly. A separate Brent root calculation reproduced every triplet status, order, limit, envelope and flag; maximum differences were 2.39e−15 in order and 8.23e−16 N in force limits. This verifies numerical reporting, not physical fidelity.

**Spatial acceptance remains false, load-step convergence at the finest mesh remains untested, and material calibration stays closed.** The earlier failed spatial studies are retained unchanged. A single N36 proposal is archived as an unexecuted next diagnostic with explicit time/memory forecasts; no further solver call is authorized by this result. Grading or higher-order elements would require a separately validated convergence family/formulation.

Reproduction bindings are in the preparation/solve releases and baseline receipts. Source is the exact 22-file Git archive from `0e8285d821633d27c9b1f4219ebdde60672844e4` (archive SHA-256 `6632d31aaabb642c338291918ee92b5e2096f375e3a68f2c9c869c4853948f7d`). The local archive, mesh and primitive logs stay under ignored build/output paths. Full replay requires those authenticated dependencies and the pinned local runtime; copying compact receipts alone is not a completed rerun. The original one-shot execution namespace must not be overwritten. No patient-case, clinical accuracy, cutting-force, tissue-injury, training or RL claim follows.

[Diagnostic figure](figure/n32-spatial-diagnostic.png) · [PDF](figure/n32-spatial-diagnostic.pdf). The first figure had a minor text overlap; one declared typography-only redraw corrected it with identical plotted data. Both original local render records remain preserved.

The independent N36 design review finds one terminal uniform-trend check informative, while emphasizing its forecast sensitivity and the older-limit discrepancy. A possible pass would only nominate separate temporal review; it would not repair N32, certify continuum error or release calibration.
