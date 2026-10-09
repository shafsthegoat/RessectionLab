# Independent saved-result audit: PASS

The one-forward diagnostic matches declaration SHA256 `8463bbf4e244498cc1df6ca2a92df2ed981f64c9622f158aaf7f62e50bd01204`. All 81 source and four input snapshots, selected RL checkpoint byte identity/prior lineage, and every output hash passed. The exact inventory contains 92 indexed files plus its root index: **93 files / 1,813,277 bytes**. This review read saved JSON/NumPy arrays and checkpoint bytes for hashing; it decoded no checkpoint and performed no model forward or native call.

The saved input arrays differ at exactly seven coordinates: `action_geometry[1:8,12]`. Each original 120 mm descriptor becomes that tool's declared original 2.2/12 mm value. All other geometry entries, image/coverage/availability, affine/spacing, state, masks, action IDs/order and provenance remain fixed. Both original and modified observation fingerprints were independently recomputed from the saved arrays and frozen schema. Source-bound runtime checks also report unchanged non-geometry preprocessing tensors and unchanged policy parameters/buffers; those computations were not repeated by this reviewer.

| Saved readout | Actual-long root | Reversed descriptors |
|---|---:|---:|
| STOP logit | −1.8111782 | −1.8111782 |
| STOP minus best movement | +0.4669986 | −6.3032722 |
| Softmax STOP share | 60.9844% | 0.1731338% |

All eight score differences, ranking and softmax arithmetic passed. The four movement candidates shared with the original-length experiment recover their original logits exactly. The highest-ranked modified-input candidate is `NATIVE-SPATIAL:c43127b9ec672e22e768b38f`; it was **not executed**. These results isolate sensitivity to the seven working-length input entries while holding the actual-long candidate inventory fixed. They do not certify fake short descriptors for long tools or demonstrate a repaired policy, improved strategy, calibrated STOP decision or generalization.

The worker records exactly one attempted and completed frozen RL forward, 24 reconstruction previews under the 64-entry cap, and zero executed actions, search, optimizer updates or patient files. Reconstruction took 0.0884 seconds; worker body 0.1165 seconds; supervisor 1.8840 seconds with 311,083,008 bytes sampled peak RSS. Exit zero, confirmed termination, empty cleanup errors and no retry satisfy the 20-second/1 GiB/one-thread release. Imports are included in supervisor time; source snapshots and postflight hashing are outside that timer. RSS sampling can miss transient peaks.

The permitted conclusion is representation sensitivity on one familiar, fully observed six-cell generated fixture. No physical strategy was validated using modified descriptors, and limited-input patient transfer remains untested.

Machine receipt: `audit.json`, SHA256 `d4a21c0fbbe6e5cdd4888ed682778b0f7f3b54e0a75c43b8205b27a0a3e4d451`. `audit_saved.py` reproduces this saved-only audit without importing the policy or native task modules.
