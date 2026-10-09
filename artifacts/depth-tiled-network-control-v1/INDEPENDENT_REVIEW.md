# Independent audit: generated native versus depth-tiled CPU network control

**Decision: PASS for this one generated-input numerical control.** The native and depth-tiled full-network CPU runs produced bytewise identical saved FP32 logits under the same frozen checkpoint and generated `[1,2,32,32,64]` input. This shows the substitution preserved this computation; it does not measure patient segmentation accuracy, full 128³ feasibility, or transfer to a held-out patient. The earlier Gate A v3 32×64×64 host-pressure failure is a separate negative and remains intact.

## Independent output and provenance check

I loaded `native/logits.npy` and `tiled/logits.npy` independently with pickle disabled. Both are finite, contiguous `float32` arrays of shape `[1,3,32,32,64]`. All 196,608 values are bytewise equal; maximum absolute error and RMSE are 0. Both raw-array SHA-256 values are `7b5df1e4bc124da3c251b31b3ec6ff80ca34b132cb2fbd9d48a3fe5f9f8b646f`, and each NPY-file digest matches its worker receipt. The frozen comparator reports `control_pass=true`, 0/196,608 out-of-tolerance logits, and 0/65,536 decoded-label disagreements under its prospective `rtol=atol=1e-3` plus exact decoded-label rule.

Both result and supervision receipts bind the same contract SHA-256 `6f962399a271a975a2fea72220304b7ecf0c278ff5645b2ebad41dd66f31f38b`, generated input NPY `1edcafb65d19d721f7a729dc1aed883f24b84bba8f9b4a0f122dc28af0ee1ddf`, and checkpoint `0e29f882310fe8cb076d6cadb982067ef53c6a32231f40ae17d9c173aa4307b3`. I rechecked all contract-pinned runner/adapter sources, plans, dataset and checkpoint hashes after both runs. The worker reports restored scoped safe globals, 88 checkpoint/model parameter alias groups with 184 compared alias pairs, and 31,197,263 unique parameter elements for each arm. The native arm reports no wrapper. The tiled arm reports 27 uniquely wrapped Conv3d modules and preservation of parameter, state-dict storage and module identities. No patient input or MPS execution occurred.

## Resource trace

| Measure | Native | Depth-tiled |
|---|---:|---:|
| Accepted 30-second preflight | yes; seven mask-1 samples | yes; seven mask-1 samples |
| Exit and guard result | 0; accepted, no watchdog/post/monitor error | 0; accepted, no watchdog/post/monitor error |
| Fast samples | 44, all mask 1 | 36, all mask 1 |
| Minimum available memory during run | 41% | 51% |
| Swap-used growth | 0 | 0 |
| Peak sampled process-group RSS | 1,212,137,472 B | 670,138,368 B |
| Forward time (single call) | 0.408 s | 0.150 s |
| Supervisor elapsed time | 2.522 s | 2.086 s |

The sampled RSS difference is 541,999,104 bytes (about 517 MiB, 44.7% of native peak) for this shape. Fast-sample maximum gaps were 0.073/0.061 seconds and slow-inventory maximum ages 0.589/0.487 seconds, below the prospective 0.2/1.5-second limits. Neither arm recorded detached descendants or blocking project overlap; both process groups were empty at audit. Memory and timing came from separate one-shot processes on a shared host, not repeated randomized benchmarking. The source/worker phases place most of the peak difference during forward execution, but this result cannot establish a general speed or memory advantage across shapes and host states.

This particular generated pattern barely exercised segmentation thresholds: whole-tumor logits had 274 positive cells and minimum absolute logit 0.001126; tumor-core and enhancing-tumor logits were entirely negative, and no region had a logit within ±0.001 of zero. Therefore exact decoded-label agreement adds little threshold-stress evidence here, even though full-logit bitwise equality is a stronger numerical observation for this input.

The tested volume is substantially smaller than the model's planned 128³ patch, and the earlier larger generated control hit macOS warning-level pressure. Do not extrapolate the 517 MiB difference or pass status to full 128³, patient data, clinical outcomes, or model accuracy without a separately specified and reviewed test. Preserve all negative receipts and the actual comparator output.
