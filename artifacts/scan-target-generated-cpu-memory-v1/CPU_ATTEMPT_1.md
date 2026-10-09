# GlioMODA pinned nnU-Net generated-input CPU feasibility, v1

Scope: nonpatient software feasibility only. The planned model patch is 128×128×128; the single attempted input was generated zeros with shape `[1,2,64,64,64]`. No patient image, annotation, training, or clinical outcome was used. The GlioMODA wrapper and nnU-Net folder initializer were not called.

## Static preflight

The pinned `3d_fullres` plans build a licensed nnU-Net `PlainConvUNet` with two input channels and three WT/TC/ET sigmoid-region heads. A meta-device build matched all 292 checkpoint state keys and shapes. The network has 108 unique parameter objects / 31,197,263 unique parameter elements, versus 292 named occurrences / 88,624,655 alias-counted elements. There are 88 shared-parameter alias groups. This was a key/shape preflight only, not a value load or forward pass.

CPU convolution selected `_ConvBackend.Slow3d`. A possible full-unfold workspace for one second convolution at the declared 128³ patch is 7,247,757,312 bytes, above the 3 GiB process cap; no full-patch CPU call was attempted. The projected 64³ budget of 2,702,094,652 bytes included an assumed 512 MiB reserve, but was an estimate, not a measured bound.

## Single supervised 64³ attempt

The independently reviewed worker and supervisor SHA-256s were `ff6ca82183dcc9523836cf52c470de023a1efc5deaf23c4bdad9f0dacd8cb79d` and `231c70080ca6b49b7f5384ed698cc4b1e2232e9b9098e415c4a0189161fde6b7`. The worker used only `torch.load(weights_only=True,map_location='cpu')` with four scoped NumPy globals, then intended duplicate-alias equality/identity checks, strict CPU state loading, and one generated forward. The supervisor allowed one attempt under a 3 GiB sampled process-group RSS cap and 120 s wall cap.

The process-group RSS sample reached **3,161,360 KiB** against **3,145,728 KiB** cap (15,632 KiB over), so the supervisor killed the child after **2.746 s**. Exit was `-9`. The process group is now empty. The pinned checkpoint SHA-256 remained `0e29f882310fe8cb076d6cadb982067ef53c6a32231f40ae17d9c173aa4307b3`. There is no result or failure JSON from the worker and **no completed forward output**.

The log contains only the line printed just before the safe checkpoint load. The worker prints nothing between that line and final success, so the saved evidence **does not localize** the memory spike to loading, network construction, alias verification, state copy, or a partially started forward. It cannot establish a passed alias-value check. The previous standalone inspection used the same scoped loader and streaming 8 MiB SHA and completed at 491,008 KiB sampled RSS; the new worker also streams hashes and does not read the 3.3 GB archive into memory. The added network and `_ConvBackend.Slow3d` forward are possible contributors; attributing the cap breach to either without stage markers would overstate the evidence. The projected 64³ budget was too low for the observed process.

Next separately reviewed diagnostic, if pursued: add pre/post stage RSS markers and explicit stdout flushes around the load, alias checks, model copy, and forward; consider a pinned inference-only state extraction that omits optimizer state, with new hash and safe-loading review. Keep the one-attempt v1 receipts and do not relabel a smaller-patch smoke as 128³ readiness or patient accuracy. No retry was made here.

Evidence: `preflight.json`, `generated-smoke-supervision.json`, `generated-smoke.log`, the unchanged checkpoint hash, and independent agent read-only audit. The scripts and receipts are ignored local build artifacts; tracked source and patient splits were unchanged.
