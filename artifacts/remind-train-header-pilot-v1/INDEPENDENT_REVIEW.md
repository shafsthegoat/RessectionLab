# Independent saved-result audit: exact-65 ReMIND header pilot

**PASS_SAVED_HEADER_INVENTORY_ONLY.** The single authorized v2-r3 pilot produced a complete, internally consistent receipt chain. This independent audit read saved JSON and bound software files only: zero patient-payload open attempts, patient-header reopens, pixel decodes, native-solver calls, or model calls. No further pilot ran and no tracked file was edited.

| Evidence | Result |
| --- | --- |
| Scope | ReMIND-002 TRAIN: 64 MR objects plus 1 US object; all 65 expected paths/SHA-256 values/sizes match the frozen plan |
| Read accounting | 102,234,796 verification bytes + 174,465 header bytes = **102,409,261 returned bytes**, across 17,351 reads |
| Durable object records | All 65 intent/result pairs chain exactly; no missing/refused object or unconfirmed read reservation |
| Completion/cleanup | Parent and worker report complete; worker exit 0; parent reaped worker; no cleanup notes or termination request |
| Resources | 2.43485 seconds; worker peak RSS **395,149,312 bytes** (376.84 MiB); 137 files totaling **464,673 bytes** |
| Runtime/source binding | 356 logged module origins independently rehashed against the release; launcher/bootstrap/I/O and promoted source pins match; tracked sources equal source commit `cb1929bf182ab3af1723e457bb2fe74c86c8f69a` |

The root release differs from the reviewed candidate only in its three authorization fields and binds independent review `142e5fe0e39eb7319a052483a7345b4668b9ae98f00ed7d42f1de901d0c0fe20`. Its saved release snapshot matches. Every run file was hash-inventoried, with no unexplained output, symlink, or partial file. Elapsed time, sampled/worker-peak RSS, output size, and each object/aggregate read count remain within the reviewed limits. Cleanup is verified from the saved owned-worker lifecycle evidence; no stale PID/group was signaled during this audit.

The MR inventory reports a classic single-frame **512 × 512 × 64** grid. Saved affine column norms are approximately **0.429688 × 0.429687 × 3.300003 mm**, and the recorded slice-position fit residual is **0.0006611 mm**. I independently recomputed the voxel-cell corner bounds from the affine and checked all 64 sorted source paths. The saved inventory does not contain original orientation, positions, or pixel spacing, so the original plane fit was **not independently repeated**; the residual remains a reviewed-launcher result. Pixel values and anatomical coverage remain unchecked.

The US inventory contains **714 columns × 615 rows × 156 frames**, one shared functional group and 156 per-frame groups, with SOP Class UID `1.2.840.10008.5.1.4.1.1.7.2`. Their contents were not projected into the saved inventory. The current adapter correctly leaves patient-space geometry and spatial-versus-temporal frame interpretation unsupported/unresolved. MR and US FrameOfReferenceUIDs differ; these records establish no cross-modality spatial transform.

Both source-stage hints remain **intraoperative label only**; the US source series description is `US_post_dura`. Acquisition date/time/date-time fields are absent. Raw study/series/content timing does not establish chronology or decision-time availability. Training admission and preoperative-input admission remain false; anatomy QC, anatomical coverage, conversion, and source-pixel checks remain unperformed. A successful header inventory does not change these fields.

Next useful preparation: design generated controls for a minimal US functional-group projection and geometry adapter, including per-frame order, position/orientation/units, and explicit reference-frame linkage, alongside a source-timing provenance contract. Any additional header projection or image/anatomy inspection requires its own exact reviewed release. This result does not authorize cohort expansion. Writer and independent reviewer agree on this scope.

| Bound result | SHA-256 |
| --- | --- |
| Root release | `65c3f90df6e10028ae7c9939ed610cf332e1b368a56ed1235860a239db55aa29` |
| Parent receipt | `ecb7179d7a4b1c295cb98b576fae573c3189d91e9d9f5496671304c17e956132` |
| Worker receipt | `4d2246aa3354fbac4620643b89b5eaee1788cb3082696e67b7322ea7c13eb2ab` |
| MR inventory | `25332ed8c5645426e800b23f4d8e8aafddcd2bd4e45ead9f1c0fc294599da563` |
| US inventory | `16c6cf9f2bd6d9f4f80f6c26ca779b9112da8867910ed883a83a70ac0056c738` |

The adjacent `audit.json` contains all 137 output hashes, all 65 per-object byte deltas, source/runtime checks, raw timing summaries, and explicit limitations. `audit_saved.py` records the reproducible saved-only audit with a patient-payload-open denial hook. Review HEAD was `0df5674f29c7d33131abb93fc9fa1dfd633b1e3e`; source approval remains content-bound to the reviewed pins.
