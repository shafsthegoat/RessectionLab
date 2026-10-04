# PAT16/PAT20 frozen extraction repeatability batch

Declared October 4, 2026; **inference has not started**. Execution is held until
the integrated tests and native-axis timing preflight release their resource
window. The immutable declaration is
[`brain-extraction-pat16-pat20-repeatability-v1.json`](../manifests/experiments/brain-extraction-pat16-pat20-repeatability-v1.json).

The existing `resectionlab.brain_extraction` command accepts these source paths
without edits. It runs no-CSF followed by main when `--compare-main-model` is
set. The batch uses two sequential repetitions per patient, in the order below;
each row launches those two model variants. Only one inference child may run at
a time. Exact command arguments and all input/model hashes are in the declaration.

| Run | Output directory | Source annotation at threshold 0.5 |
| --- | --- | ---: |
| PAT16 repeat 1 | `artifacts/brain-extraction/PAT16-mps-repeatability-v1/r1` | 45,400 voxels |
| PAT16 repeat 2 | `artifacts/brain-extraction/PAT16-mps-repeatability-v1/r2` | Same fixed annotation |
| PAT20 repeat 1 | `artifacts/brain-extraction/PAT20-mps-repeatability-v1/r1` | 12,451 voxels |
| PAT20 repeat 2 | `artifacts/brain-extraction/PAT20-mps-repeatability-v1/r2` | Same fixed annotation |

Both source pairs already passed separate byte, physical-frame, threshold,
saved-case and engineering overlay checks recorded in
[`btc-pat16-independent-qc.json`](btc-pat16-independent-qc.json) and
[`btc-pat20-independent-qc.json`](btc-pat20-independent-qc.json). Their roles are
permanently development. No source arrays or prepared cases will be replaced.

## Unchanged implementation and limits

The wrapper file was hashed independently and is byte-identical to the retained
PAT05 wrapper snapshot:
`702f5df214fd1d25372275bae5333e4c915c34e9cb3745a9fe8d6eb392b74917`.
This is the wrapper identity, separate from the main checkpoint (`37417f80…`),
no-CSF checkpoint (`62bf0113…`), upstream script (`291c253a…`) and marked MPS
runner (`d39cbcba…`). Full hashes and local assets were verified separately.

The isolated inference runtime remains Python 3.12.14, NumPy 2.2.6, Torch 2.14.1,
Surfa 0.6.3, SciPy 1.18.1 and NiBabel 5.4.2. Distribution metadata was checked
without loading a model. Settings remain MPS, two CPU helper threads, 1-mm
border, source-intensity threshold 0.5, and the unchanged erosive intensity
baseline. Automatic CPU fallback and model downloads are disabled.

Each child has the same 300-second timeout and 6-GiB sampled-RSS stopping
threshold. MPS allocations are capped at the lower of 6 GiB or 70% of recommended
device memory. RSS and allocator measurements are samples, not exact total-memory
peaks. Parent preprocessing, runtime probing, rendering and validation are outside
the child timer; eight declared child timeouts sum to 2,400 seconds and do not
constitute a bound on the entire batch. Each runtime probe has its own 30-second
timeout. Existing output directories must not be reused.

On failure, retain the log, named failure record and partial files; mark remaining
runs unexecuted pending review. No hidden retries, changed thresholds, alternative
models or selective exclusion of failures are permitted within this declaration.

## Checks after the timing hold is released

Recheck source, wrapper, model and runtime identity before execution. After each
run, retain the exact wrapper and generated runner, validate native-grid corners,
finite distance values and binary masks, and independently reconstruct the mask
from the distance threshold/component/hole-fill procedure. Compare repeated mask
arrays and distance fields directly and record every discrepancy. Inspect source
annotation inclusion, components, boundary contacts and native overlays.

The existing independent audit helpers support these array checks. Their current
standalone CLI only enumerates PAT05 and PAT28; extending that CLI would be a
separate coordinated change. It is not needed to run the unchanged extractor.

All outputs remain **estimated, review required** and separate from the working
brain mask. Repeatability and annotation inclusion do not establish anatomical
accuracy, a pial surface, cortical access, functional localization or clinical
risk. Repeated runs do not add independent patients. No extraction has been
performed as part of this declaration.
