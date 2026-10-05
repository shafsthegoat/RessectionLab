# HBE axial resolution: actual mesh preparation

The separately released preparation completed once and produced the two declared meshes plus four axial input decks. The saved status is `prepared_not_solved`: exactly two Gmsh generation calls, zero specimen solver calls, and no measured-data access. All four solver cases remain `not_executed`. The two co-primary convergence comparisons have not been evaluated.

| Mesh | Nodes | Hex8 cells | Relative volume error | Maximum radial sag / R | Minimum sampled rest scaled Jacobian |
| --- | ---: | ---: | ---: | ---: | ---: |
| N16 | 7,209 | 6,144 | 0.001605606964382066 | 0.0012045438047777763 | 0.7071067811865468 |
| N24 | 23,101 | 20,736 | 0.0007137941770923635 | 0.0005354125290327107 | 0.7071067811865447 |

Both saved generation receipts pass all six declared quality checks, including the unchanged finest-mesh volume and radial-sag limits of 0.005. These values are copied from authenticated actual receipts; independent recomputation of the saved meshes and decks is a separate review. The model, boundaries, load steps, modulus and scientific tolerances remain unchanged. Original Skyline decks and solver-only Accelerate copies are retained.

Supervised elapsed time was **5.689496875042 s**, within the 120 s cap. Worker-body time was 4.668026375002 s. Peak sampled process-group RSS was **285,802,496 bytes** (272.5625 MiB), within 3 GiB. The supervisor recorded exit 0, no timeout, no cleanup error and no retry. Sampling cannot establish every instantaneous memory peak.

The preparation tree contains **28 files / 39,177,774 bytes**. Every file was indexed, made read-only and rehashed after the permission change. Only `outputs/mechanics/hbe-01-03-resolution-v1/mesh-preparation` was frozen; its parent study directory retains its original writable mode for a separately released solve phase. Raw contents were not edited, compressed or removed.

All **168 launch-bound inputs** still match their hashes. The complete previous failed Accelerate experiment also remains unchanged: 169 files / 914,390,988 bytes. The executed source is commit `5b80e3fee2349bc78d3f2ceb9b211efb305a1fc9` with archive SHA `29f5866817a260e49275e51510a1937f432cc1bc40762009025748092e8eefdb`; the actual preparation release is bound in `outcome.json`.

The collector used only saved JSON, byte hashes, file sizes and permissions. Two collector setup mistakes were retained: an unavailable convenience hashing method in the system Python, and an initially incorrect repository ancestor index. Both stopped before raw-file changes or native computation; the successful collection then used streaming hashing and the correct root. Neither was a mesh or solver attempt.

This result establishes preparation evidence only. It does not establish numerical mesh convergence, material fidelity, patient-specific mechanics or clinical adequacy. The separate final solve release has not been issued by this collection.

Evidence: [outcome](outcome.json), [actual quality receipts and case bindings](mesh-quality-summary.json), [complete raw index](raw-output-index.json), [input preservation](preservation.json), [read-only freeze](readonly-freeze.json).
