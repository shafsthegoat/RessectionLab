# Case4 curvature candidate: incomplete at the fixed time allowance

The single released curvature-24 / 3 mm minimum candidate did not return a volume mesh. The supervisor terminated its worker at **180.028 s** under the unchanged 180 s / 3 GiB / one-thread allowance. No retry occurred.

The exact recorded reason is `supervision_exception`: the final process-memory sampling command raised `TimeoutExpired` with **0.0117436 s remaining**. The worker exit was **−9**, with no cleanup error or surviving process-group member found afterward. This preserves the actual supervision outcome; it is not a geometric-fidelity rejection.

Gmsh logged 3.648 s for surface classification, 1.482 s for discrete-surface parametrization, and **170.079 s for 1D boundary-curve meshing**. It had entered 2D surface meshing when interrupted. The sampled peak process-group RSS was **590,708,736 bytes**; brief peaks between samples may be missed. These are one attempt's stage observations, not a controlled speed comparison. The earlier HBE solver had terminated, while result-collection I/O could still occur.

One native generation was charged before entering Gmsh. No volume mesh returned, so node/tet counts, complete diagnostic arrays, Jacobian/topology/overlap checks, surface-distance bounds, and volume-error metrics are **unassessed**. The larger count allowance was never demonstrated. The worker and candidate checkpoint files still say `running`; the worker's generation count is `null`. Those interrupted records are preserved verbatim, while the terminal acceptance and supervision receipts report failure.

All **27 source/runtime/release/context bindings** match their before-run hashes. Both prior failed attempts still match their saved output indices. The five raw outputs total **22,993 bytes** and are copied losslessly under `saved-records/`. Only the retained source surface was decoded; there was no image re-extraction, B/V access, solver call, training, source repair, cap increase, or second candidate.

Exact execution source: commit `b8de4416e777b11d31f1b75adbf164522a92eaa3`, using the reviewed read-only ten-file archive. Root release SHA256: `ac226fb7ff677ff9d06e204be7f565a0e9a6b30240e9506444ec2a0f1d54ffe9`. See `execution-baseline.json`, `summary.json`, `post-execution-verification.json`, and `raw-output-index.json`. Independent saved-output review is separate; this record grants no additional execution or solver authority.
