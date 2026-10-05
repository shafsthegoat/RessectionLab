# Real T1 policy scaling diagnostic

The untrained default spatial policy completed four CPU, one-thread inference calls on the creator-verified **TRAIN PAT22 T1** in **6.247 seconds**. No gradient, optimizer step, checkpoint, source annotation, or checkpoint-selection/evaluation patient was used. All parameter hashes remained unchanged. The 18 profiler contract checks passed before execution.

| Native crop | Geometric rays + STOP | Full inference | 3D encoder | Highest observed RSS¹ |
|---|---:|---:|---:|---:|
| 32³ | 16 + 1 | 28.016 ms | 23.727 ms | 429.77 MiB |
| 32³ | 128 + 1 | 30.111 ms | 22.661 ms | 418.36 MiB |
| 64³ | 16 + 1 | 268.849 ms | 248.279 ms | 1173.16 MiB |
| 64³ | 128 + 1 | 255.518 ms | 235.226 ms | 1179.97 MiB |

¹ Maximum of the parent’s sampled RSS and the child’s reported high-water snapshot. They are different measurements; the child snapshot precedes final provenance serialization. Every cell remained below the 1792 MiB sampled stop and 2 GiB rejection limits. No failure or retry occurred.

These are **one un-warmed call per cell**, not latency distributions or evidence that 128 rays are faster. Source verification, NIfTI loading, normalization, cropping and observation construction took another 354–375 ms per fresh child; total child wall time also includes framework import and model initialization. Timers wrap the actual implementation and are nested, so their inclusive durations must not be added together.

The source is 160×256×256 at approximately 1 mm spacing. Crops begin at native voxels `[64,112,112]` and `[48,96,96]`, determined solely from image shape. Each crop keeps its translated native affine and uses the finite full-image mean and population standard deviation from this same case, including background. There is no resampling, clipping, lesion-driven crop, or brain-mask assumption. Structural intensity and initial empty observed cavity are available; target, tissue, motor and language channels remain explicitly unavailable. Ray/access geometry is an engineering query, without anatomical route acceptance.

The 24,331-parameter model’s encoder consumed approximately 75–92% of measured inference time. This supports testing **shared image context plus bounded native-resolution local patches** next: cache a separate static-image branch, update the procedure-state branch explicitly, and retain physical-coordinate/coverage records for every patch. Any such architecture needs fresh equivalence and learning tests. A coarse global image may provide context, while native patches and independent native geometry retain small structures along the complete tool path. Downsampling can erase vessels; a centered crop can omit both the lesion and most anatomy. The full native grid has 40 times the voxels of a 64³ crop; this experiment does not establish whole-volume or training memory fit.

Execution used one Git archive of `b93c9af71e80ee7d63ac116451cc5aa94f38e475`, including the opt-in candidate-critic implementation with its default disabled. All six imported project modules were verified inside that archive. The mutable working checkout was excluded. Source/archive hashes, the root release before execution, every result and the compact receipt are retained under `artifacts/real-spatial-policy-scale-v2/`; the earlier unrun declaration remains unchanged under `artifacts/real-spatial-policy-scale-v1/`.
