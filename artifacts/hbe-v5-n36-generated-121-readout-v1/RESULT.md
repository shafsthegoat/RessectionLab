# HBE v5 N36/S120 generated-only 121-frame readout

**Result, 2026-10-09:** One prepared geometry context evaluated **121/121 generated affine compression frames** in a supervised direct-frame loop. Every frame passed the internal fixture criteria. This is a measured readout performance result, **not** a native simulation, measured-force comparison, numerical-convergence result, or physical-validation pass.

The inputs were the frozen v5 compression load schedule and existing SHA-bound N36 specimen half/full meshes and reconstruction mapping. The fixture applied homogeneous `F = diag(1, 1, 1 + d/H)`, with stress and energy density computed from the declared material law and total reaction from the derivative of analytic energy. It spread reactions uniformly over face nodes to satisfy the generated global checks. That distribution is not an FEM equilibrium solution, and the affine field does not establish the specimen's actual free-surface response. Source/adapted decks were deliberately marked `generated-fixture-unbound`.

| Measured item | Result |
| --- | ---: |
| Frame preparation | 3.450 s |
| Direct evaluation of all 121 frames | 116.140 s |
| Child total / supervisor wall | 119.858 s / 120.212 s |
| Peak child RSS / supervisor sampled RSS | 847,396,864 B / 837,353,472 B |
| Largest normalized frame criterion | `9.55e-10` (limit 1) |
| Largest evaluator-versus-analytic energy difference | `8.54e-19 J` |
| Generated full/native work ratios | `1.096e-4` / `1.096e-4` (limit 1) |

The supervisor enforced a **240 s wall limit** and killed at **2.0 GB sampled RSS**, below the requested 2.5 GiB ceiling; neither limit was reached. The child reported its own peak RSS. Sampling does not prove an instantaneous memory ceiling, but both observed measures were far below the guard. There was one run, no retry or cap increase.

This calls `evaluate_prepared_frame` on in-memory generated records. It **does not call the complete saved-log parser**, authenticate native primitive files, run FEBio, read measured HBE response or patient data, or establish agreement with physical measurements. Its `physical_validation_pass` remains null. The next distinct benchmark is the bounded full saved-log reader; the next scientific gate is measured specimen response under released native conditions.

**Exact provenance.** Committed optimized frame source SHA-256: `8431c7ae293befdf75bbccc54ce3481ddbd2e42b70daf1b6f826be96bf68e0d9` (Git commit `ec5f5e4a34db65eafd240d415567db646159bf70`). v5/v4 declarations: `50c5dbc8c45279245a8dcd9b92d16e9499bad6bc61c6dfbd5b5e307af49a71ce` / `85ed9ca8cb1a9048e885f27424678e97f20cb6dd2e2cbb5cfd6dccfe6dafb52b`. N36 full/half/reconstruction files: `6ba012cc042ec0a537a6c61283d444488ccf69de0bc9c385f319a827d69c8e74` / `8e7ad0e636151e007ab9fd8af0d043abfcf889b960c53e726766126691c3c075` / `9e0265495f47ece2d315746f79f40e71f717e0720f6f1621fa9c7f9d2d9eba8f`. Ignored local generated-only runner/supervisor scripts: `74b57b88d7d959ed67403b1adedfb67c2700f69523f4960c560617bb4528f8c7` / `2a355ababf34313f4a7f29066299696260fa0be3b70e7d610a831fe206425086`. Ignored result/supervision JSON: `48ade5c6b240347f9ca164ad0f218e9a06feef1dd118b6860e40bfa572119cd6` / `4a574d2ac1efe751929c76b4538964e50084d6578e8d02c18fdce9e7fca3b03b`.
