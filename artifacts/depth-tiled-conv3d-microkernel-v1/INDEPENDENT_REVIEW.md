# Independent depth-slab Conv3d microkernel audit

**Decision: PASS as a generated FP32 CPU operator proof of concept; no full-model approval.** The saved native and depth-tiled 16→16, 32×48×48 outputs are bytewise equal. The comparison says nothing yet about full GlioMODA network inference, trained-output parity, 128³ resource feasibility, patient segmentation, or clinical quality.

## Operator and architecture check

The tiled operator calculates each output slab's full depth receptive field. For output depth coordinate `j` and kernel offset `r`, both native and tiled calls read global input coordinate `j*stride - padding + r*dilation`; the slab's left and right zero padding covers out-of-bounds coordinates. Height and width are never tiled and retain native Conv3d padding. The code passes the same weight, bias, stride, dilation, and group count to each native tile call, and checks every tile's output geometry. It is CPU/FP32 and inference-only.

The six saved generated tests include pinned-plan-like stride 1 and 2, odd extents, an output tile of one, asymmetric stride/padding, and grouped/dilated geometry. All report exact equality and zero absolute error. This is a useful boundary check, although finite examples do not prove all legal Conv3d configurations. The pinned `3d_fullres` nnU-Net plan specifies Conv3d, 3×3×3 kernels, per-stage strides 1 then 2, and feature widths `[32,64,128,256,320,320]`; the installed `dynamic_network_architectures` `SimpleConvBlocks` source constructs padding `(kernel−1)//2`, dilation 1, and default groups 1. Thus the prototype's stride/padding/dilation/group semantics include the pinned convolutions. The model also has normalization, activations, skip connections, and decoder operations, which this operator test does not exercise.

## Independent saved-array check

I loaded both generated `.npy` outputs independently with pickle disabled. Each is finite `float32` with shape `[1,16,32,48,48]`. They are bytewise equal; maximum absolute difference is 0. Their raw contiguous output SHA-256 is `b857b0f905f731797987ee74d04c9fcdb701f382e9aab850e32d8c423e9015a7`, matching both worker receipts and the comparison. Both workers used the same fixed seed to regenerate input, weight, and bias; neither loaded checkpoint weights or patient arrays.

The two independent processes exited 0. Direct host samples stayed at kernel pressure mask 1 (normal), at least 51%/52% available, and zero swap-used growth; pre- and post-run direct host snapshots were also normal. Sampled peak process RSS was 351,387,648 bytes native and 241,319,936 bytes tiled. Worker `ru_maxrss` high-water readings were 351,289,344 and 241,221,632 bytes, consistent with the sampled values. Native/tiled forward timings were 22.38/10.11 ms, respectively. These are single-process, single-call measurements with different import/process histories and should be labeled exploratory; the 110 MB sampled-peak difference is evidence for this small operator case only, not a network-level savings or speedup. The watchdog checked 1 GiB RSS, 30 s wall time, normal pressure, available ≥45%, swap growth ≤128 MiB, and detached children at approximately 50 ms intervals. It is sampled supervision, not a hard operating-system memory reservation; this actual pair remained well below its caps.

## Frozen evidence

| Item | SHA-256 |
|---|---|
| `depth_tiled_conv3d.py` | `83c9660278d219bae10aa2f0ca7e8d5b21ed948cab639706a1720c27685c098c` |
| `test_generated.py` | `63ac7717d013b6126505eae2dca4d92c53130cc90ad341ae7f29ee7e80673874` |
| `microkernel_worker.py` | `f677e43836275a5d92b6cf08475d6df171fe528cb813892728e8f3c6ecdc8a05` |
| `supervise_microkernel.py` | `078946618aa1a848868ae8e8cc94507912ad60c188a5dccff734ac66d3021974` |
| `generated-parity.json` | `a77c13484572cabbe8635b5accf20ed013840dc8e242b3de57b63a4217c94a46` |
| `microkernel-supervision.json` | `855d37fc785bca2bffeb72b1afcca57683a0a42048d44df82ab6ad5799aa582e` |
| `microkernel-comparison.json` | `58f36849840ffa7c134f017a76722890d19ee7f3762095fbef75ef536e699a2b` |

Preserve the separate v1/v2/v3 whole-model resource negatives. Any full-network control needs its own prospective integration, equivalence criteria, host guard, and review; this audit does not authorize one.
