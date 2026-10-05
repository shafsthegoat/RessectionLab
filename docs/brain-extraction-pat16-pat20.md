# PAT16/PAT20 frozen extraction repeatability batch

Declared October 4, 2026 and executed after the orchestrator released the resource
window. All eight child inferences completed in the declared order with no
failures, retries or retuning. The immutable declaration is
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

## Execution and pending independent review

The complete batch took 83.032 seconds, including parent preparation and repeat
comparison. Individual child times were 5.833–8.033 seconds on this local run.
The maximum sampled child RSS was 683,704,320 bytes; sampled MPS tensor and driver
allocations reached 4,117,935,104 and 5,948,243,968 bytes respectively. These are
the declared observational measurements, not exact combined memory peaks or
population latency estimates.

| Patient | Model | Estimated envelope, mL | Source annotation voxels excluded |
| --- | --- | ---: | ---: |
| PAT16 | No-CSF | 1351.353 | 2,045 of 45,400 |
| PAT16 | Main | 1560.511 | 19 of 45,400 |
| PAT20 | No-CSF | 1203.833 | 413 of 12,451 |
| PAT20 | Main | 1402.249 | 125 of 12,451 |

Both repetitions produced identical native mask arrays and identical predicted
distance arrays for each patient/model combination: zero differing mask voxels
and maximum absolute distance difference 0.0 mm. Every annotation omission is
retained as a review flag. Volumes and inclusion counts above are wrapper-reported
values pending independent array and visual review; repeat equality was measured
directly from the saved arrays.

[`brain-extraction-pat16-pat20-results.json`](brain-extraction-pat16-pat20-results.json)
records all eight child measurements and artifact hashes. The complete execution
receipt is `artifacts/brain-extraction/PAT16-PAT20-repeatability-v1-batch/batch_record.json`,
SHA256 `df8bb5bef8367e1b2d58be1a7d4ec4346481024544e52dd32dd783e7b79e9140`.
The original declaration, executor, exact wrapper snapshot and four CLI logs are
retained beside it. Source/model/runtime hashes and unchanged prepared-case bytes
were checked before each run and after the batch. Working anatomy was not changed.

The separate review will validate native-grid corners, finite distance values,
binary masks and reconstruction from the distance threshold/component/hole-fill
procedure, then inspect annotation inclusion, components, boundary contacts and
native overlays. Its findings will be recorded separately from these execution
results.

The existing independent audit helpers support these array checks. Their current
standalone CLI only enumerates PAT05 and PAT28; extending that CLI would be a
separate coordinated change. It is not needed to run the unchanged extractor.

All outputs remain **estimated, review required** and separate from the working
brain mask. Repeatability and annotation inclusion do not establish anatomical
accuracy, a pial surface, cortical access, functional localization or clinical
risk. Repeated runs do not add independent patients.
