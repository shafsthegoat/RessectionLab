# Independent audit: generated 64³ tiled layer-memory diagnostic

**Decision: valid completed diagnostic, not full-128³ feasibility.** The single instrumented tiled CPU forward completed under the frozen guard and saved finite output. It used generated input, with no patient scan, native 64³ comparator, segmentation accuracy endpoint, or new full-128³ run. Preserve the earlier full-128³ pressure-stop negative.

I independently loaded the saved logits with pickle disabled: contiguous `float32` `[1,3,64,64,64]`, all finite. File SHA-256 `377bb7a1dc0bea971ab3c2890432d20c2230744b479a5e7605bd824b0a81ddb1` and raw-output SHA-256 `52252550d12d7fc0ddd930c044f58cd6e1bca00606646bbd9780687bce3f2853` match the worker receipt. Post-run contract, worker, supervisor, generated input, tracked adapter and checkpoint bytes match the prospective hashes (`7c2419a2…`, `7a669a61…`, `1f38d6de…`, `c2551f7e…`, `e355495f…`, `0e29f882…`). The worker records restored scoped safe globals, 88 alias groups with 184 compared alias pairs, 27 wrapped unique Conv3d modules with preserved identities, and 31,197,263 unique parameter elements.

The 30-second preflight had seven mask-1 (normal) readings at 56% available and flat swap. The 2.390-second supervised child exited 0 with `accepted_for_feasibility=true` **for this 64³ diagnostic**; no watchdog, post-guard, monitor, slow-inventory, overlap or detached-child error. Forty-one fast samples all had mask 1, at least 55% available, zero swap growth, maximum 0.0605-second sample gap and 0.504-second slow-inventory age. Peak sampled group RSS was 956,301,312 bytes below the 3 GiB cap, with no group-81110 members left afterward. The instrumented forward took 0.575 seconds in one run. Hook libproc sampling totaled 0.00303 seconds excluding JSON flush; neither this nor the separate no-model observer benchmark isolates total runtime perturbation.

The worker log has 110 balanced layer pre/post markers: 46 Conv3d (23 executed; four deep-supervision heads inactive), 44 InstanceNorm3d, 10 ConvTranspose3d, and 10 decoder-stage markers, plus 12 outer phase markers. Key resident-memory observations:

| Boundary | RSS before → after | Change |
|---|---:|---:|
| Encoder stage 0, first Conv3d | 584,187,904 → 618,676,224 B | +34,488,320 B |
| Encoder stage 0, first InstanceNorm3d | 618,676,224 → 652,886,016 B | +34,209,792 B |
| Encoder stage 0, second Conv3d | 652,951,552 → 711,966,720 B | +59,015,168 B |
| Final decoder transpose **after** → stage 4 **before** | 775,864,320 → 842,989,568 B | +67,125,248 B |
| Final decoder stage 4, first tiled Conv3d | 842,989,568 → 956,252,160 B | +113,262,592 B |

The installed decoder source calls `torch.cat((x, skips[-(s+2)]), 1)` between the transpose and stage hook. The final stage input shape changes `[1,32,64,64,64]` → `[1,64,64,64,64]`; a new FP32 concat tensor at that shape contains 67,108,864 bytes, only 16,384 bytes below the measured RSS increase. This is strong localization of a materialized decoder concat at 64³, while still an RSS observation rather than an allocation trace. The first Conv3d/norm in tiny decoder stage 0 (4³) had flat RSS, which helps localize growth toward high-resolution encoder and final decoder operations.

RSS deltas include allocator reuse, retained skip tensors, temporary workspaces, and observation overhead; a flat delta does not mean no allocation. The analogous final decoder concat tensor at 128³ would contain 536,870,912 bytes, but scaling this 64³ trace cannot predict full-128³ peak RSS or prove which allocation caused the earlier host-pressure stop. Any concat-liveness or smaller-tile proposal needs a separately specified numerical-parity and resource test before being treated as an improvement.
