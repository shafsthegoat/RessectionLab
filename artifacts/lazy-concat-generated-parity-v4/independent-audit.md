# Independent actual saved-output audit: generated 64³ lazy concat pair v4

**Disposition: accepted for the narrow generated 64³ numerical comparison.** This is a read-only audit of saved outputs and receipts. I did not load the checkpoint, run the model, inspect patient data, or change a tracked file. It does not establish full-128³ feasibility or patient performance.

## Fixed inputs and acceptance

- Pair contract SHA-256: `1bdf989315cf4d67ce107e18be93fcb6fd90e7f8749504ccbea52252b36ecd3f`. All 15 pinned runner source hashes match current files. Generated input SHA-256: `c2551f7e80714ffa49f2dc300c916d0a75004b9b0c4888f17e40d54f300b4bb4`. Pinned checkpoint SHA-256, verified in both worker results after use: `0e29f882310fe8cb076d6cadb982067ef53c6a32231f40ae17d9c173aa4307b3`. Tiled adapter SHA-256 `e355495f60ee70e579da72522af93620da2bcf940c65dd748cbb54b8bee79f76`; lazy-concat source SHA-256 `ce4d67f816edc13b73f7bfda8bf9af2aa761021473fb4dac47bfac13d7d59f5b`.
- Baseline and lazy supervision each reports `accepted_for_feasibility=true`, exit 0, no watchdog, monitor error, post-guard error, slow-inventory error, detached descendants, or manual-attention flag. Both finalizers report no signaling, no remaining process-group or detached PID, and `residual_possible=false`. Later `ps` found neither child PID 3406 nor 3622; the allowed TractoInferno downloader PID 92434 remained alive. Preflights were accepted; all fast pressure masks were normal (`1`), minimum available memory was 47% baseline and 46% lazy, and swap-used bytes were constant within each arm. Slow inventories contained no blocking overlap and exactly matched the downloader allowance at well below its 512 MiB cap.
- Each output is finite contiguous float32 `[1,3,64,64,64]`; each worker result reports strict alias comparisons (88 groups, 184 compared pairs), scoped safe globals restored, 27 wrapped unique Conv3d modules with parameter/state identities preserved, complete-map InstanceNorm unchanged, dropout disabled, and 110 observed layer markers (27 Conv3d, 22 InstanceNorm3d, five ConvTranspose3d, five decoder stages with pre/post probes). Lazy attachment reports five final decoder stages and depth tile 1. Marker phase order and count match between arms.
- The saved comparator's prospective acceptance was `atol=rtol=0.001` on raw logits plus exact sequential WT/TC/ET decoded labels, with baseline compared to the pinned historical tile-1 result. I independently reloaded the three saved arrays with `allow_pickle=False`, rechecked source/output/reference hashes and the arm acceptance predicates, and recomputed both pair metrics. Baseline, lazy, and historical logits NPY SHA-256 are all `377bb7a1dc0bea971ab3c2890432d20c2230744b479a5e7605bd824b0a81ddb1`: all 786,432 logits are bit-identical, maximum absolute error 0, and decoded-label disagreements 0/262,144 in both comparisons. The closest-to-zero region-logit margins are approximately `0.0000902`, `0.0004919`, and `0.0127115`; no threshold crossings occurred. The saved `comparison.json` SHA-256 is `22b2f110a44f90148c07b8ac1d517d181ce013f19cd796fcd9acb880b92fd5f8`.

## Resource observation and interpretation

| Measure | Baseline | Lazy final concat |
|---|---:|---:|
| Sampled peak process-group RSS | 757,415,936 B | 749,420,544 B |
| Timed forward | 0.537442 s | 0.535711 s |
| Total supervisor elapsed | 2.520082 s | 2.583036 s |
| Fast / slow monitor samples | 44 / 5 | 45 / 5 |

The observed sampled RSS difference is 7,995,392 B (1.06% of baseline). These are one run per arm, with different host baselines and 20 Hz process sampling, so this is not a robust performance estimate. The 2.52/2.58-second numbers are total supervised elapsed, **not** forward times; the 1.73 ms forward difference is similarly too small for a speed claim.

The final decoder trace gives stronger structural evidence than the peak metric. After the final transposed convolution, baseline RSS was 665,452,544 B; at final-stage entry with one concatenated `[1,64,64,64,64]` feature map it was 732,577,792 B, a 67,125,248 B increase. The corresponding lazy arm retained two `[1,32,64,64,64]` inputs and stayed at 655,900,672 B across those markers. A newly materialized float32 concatenation of that shape would contain 67,108,864 B, near the observed baseline step. This supports that the full concatenation was avoided for this generated input. Later full-feature-map convolution/normalization and allocator retention raised lazy RSS to 749,420,544 B, leaving only the 8.0 MB difference at sampled peak. The theoretical 512 MiB concat size at 128³ is an extrapolation, **not** measured full-128³ memory saving or feasibility. Prior full-128³ tile-4 and tile-1 runs remain pressure-aborted negatives.

## Receipt identifiers

- Baseline: `supervision.json` `06516512bafbe7a785fd9ffb27fccd12bc315fe0e3aba2cbb7cebe23942f4218`; `result.json` `9a9a51efb4ca0df599bf6f62501a2328236b49beb3c0e615b6d17b3f13cfd2d4`; `worker.log` `70b0c67ed17961fd8cf317bb7e224a86de787ac8d74df8dedbe573dff3284038`.
- Lazy: `supervision.json` `83c96db19d2061587c27ee00a707ffe3dcdfeab2a72c7bbceecffb7c665fadf3`; `result.json` `98445f1d7b504af851b24fdcd47a617e238b3a03707a671e8aa11d708b3ef8dd`; `worker.log` `e4cef84fa962485950e398267256b8e4189a6d4c27f276fc25ac4d07082b94d0`.
- Historical tile-1 reference result SHA-256 `ae7492549a9b6f90d6684b377c11b5e32711b83a9d3f42f28e9a11257047ca98`; supervision SHA-256 `91396e051668265acbc48f5c7fbdc7a06f66b89f0bf0258a3bc2ec08dddcbd90`.

The next scientific question is whether reducing the final transient allocation changes accepted full-volume resource behavior under a separately frozen, independently reviewed 128³ control. This 64³ observation alone does not authorize or predict that result. No patient scan or observed clinical outcome was used here.
