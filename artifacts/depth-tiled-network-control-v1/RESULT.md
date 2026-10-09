# Generated full-network depth-tiling control

The same pretrained 31,197,263-parameter network ran in two fresh CPU processes on one frozen generated input `[1,2,32,32,64]`. Only the second arm's 27 unique Conv3d forward methods used depth tiling; weights, parameter/state/module identities, full-feature-map InstanceNorm and other operations were preserved. Both arms passed unchanged resource guards. Independent inspection reproduced **bit-for-bit equality of all 196,608 FP32 logits**, with no decoded-label differences across 65,536 cells.

Peak sampled RSS fell from **1,212,137,472 B** native to **670,138,368 B** tiled, a **44.7% reduction** in this control. Single forward times were 0.408 s and 0.150 s; those are exploratory measurements from one pair on a shared host, not a general speed claim. Both process groups exited cleanly, with no warning pressure, swap growth, monitor error or download interruption.

This validates one small generated numerical comparison. It does not establish full-128³-patch feasibility, patient segmentation accuracy, RL performance or transfer. The pattern had no logits within ±0.001 of a class threshold, so it provides limited threshold-stress evidence. The earlier larger-input v3 CPU pressure failure remains unchanged; no MPS model ran.

Compact actual receipts, independent reviews and exact source snapshots are preserved here. Arrays and model weights remain outside Git. The next source slice can integrate the tested inference adapter; larger-shape and real-patient use require their own bounded checks.
