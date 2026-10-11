# Complementary cut-union control: saved result

The explicit cumulative cut-union model meets the predefined **representation usefulness criterion**: 13/18 complete sequences versus 11/18 under the retained default model, with no lost completion, no resource refusal, and accepted independent full-tool checks for every nonempty history. The two additional completions are aligned and tilted suction at 0.5 mm. This is a positive result for accumulating complementary cuts in this generated scene. It is not a positive reward, target-volume convergence, clinical result, or SELECT013 route result.

Target removal improves in only one row and is unchanged in the other 17. Reward improves in none: ten rows become more negative and eight are unchanged within 1e-8. All six 0.25 mm target volumes are unchanged. The worker takes 3.59 times the retained baseline time; timings were recorded in separate runs and are descriptive, not a paired speed estimate.

## Saved-only verification

The saved scalar audit joined the declared physical inputs, action sequence, state transitions, row sums, attempted/committed coverage counters, source pins and checker certificates. It used only saved JSON and the Python standard library: no native replay, array loads or new tests. `saved-summary.json` preserves exact per-row rewards, volumes, costs and failures. The raw report retains the complete scalar/action records and certificate results; sparse proof histories were consumed transiently by the separate checker during the original run.

All 18 rows and all 180 prescribed action slots are accounted for: **141 accepted actions, 5 refused actions, 34 unexecuted suffix actions**. There are 15 accepted independent prefix certificates: 13 complete ten-action histories and two partial histories of nine and two actions. Three rows refuse their first action and have no committed history. All refusal states are unchanged, all refused removal credit is zero, and all accepted action volumes/rewards sum to the published row totals. Source case, tools, affine, physical action digest and action coordinates match the baseline for all rows. Preparations remove no analytic target.

The terminal report, prospective receipt and root release contain identical source maps. The tested final runner/declaration/test pins match those maps. The final ten runner controls passed; the earlier parity fixture failures and their diagnosis remain archived, rather than being replaced. They were corrected by JSON-normalizing tuple/list wire representations; the saved numerical values, state hashes and history hashes matched. Root reported 131 canonical checks passing before this ten-control terminal receipt.

Integration revision: `9fc4a50d66c562b5df065175686c3df3e6685ccb`. The only experimental model setting is `CutUnionSpec(max_depth=3, max_tracked_cells=4096, max_leaf_tests_per_preview=4194304)`. The checker receives the same explicit specification. The 18 scenes, tools, apertures, ten physical actions, source labels, reward and 0.0625 mm maximum microstep match the retained baseline.

## What improved, and what did not

| Grid spacing | Baseline complete | Cut-union complete |
|---:|---:|---:|
| 1 mm | 2/6 | 2/6 |
| 0.5 mm | 3/6 | 5/6 |
| 0.25 mm | 6/6 | 6/6 |

At 0.5 mm, aligned suction previously accepted all nine preparations and then refused the target stroke. With the same prescribed actions, its preparations remove 16.6875 mm³ of physical tissue instead of 11.4375 mm³. The final stroke now passes and removes another 1.75 mm³, but **zero analytic target**. Total physical removal is 18.4375 mm³; reward falls from -3.024 to -4.6124.

Tilted suction at 0.5 mm also changes from nine preparations plus a target refusal to all ten cuts accepted. Preparation removal increases from 12.5625 to 15.4375 mm³. The final stroke removes 1.625 mm³, including **0.03485472775 mm³ of analytic target**. Total physical removal is 17.0625 mm³; reward falls from -3.324 to -4.420574327. That target volume is still 92.0% below the exact continuous suction cap, 0.4356341813 mm³.

These complete removals exceed the respective static unions of individually contained whole cells (13.1875 and 14.0625 mm³). That is allowed here: different executed active sweeps can certify complementary subcells before a whole native cell becomes removable. Their histories passed the independent cumulative-coverage and full-tool checks. The saved scalar output does not retain exact failed-blocker cell identities from the baseline, so it supports the fixed-sequence causal effect without a claim that a named blocker cell was the decisive one.

Half-phase suction at 1 mm changes from immediate no-removal refusal to nine accepted preparations. The first five actions add certified coverage only, with zero removed tissue and ordinary action/motion costs. Later preparations remove four native cells, or 4 mm³. This is additional progress, **not a complete sequence**: cut ten still meets remaining-tissue shaft blockage at tip z=2.076470588235294 mm. Its zero baseline reward becomes -1.124.

Four shaft failures persist at the exact same action and recorded microstep endpoint as baseline: aligned aspirator 1 mm, cut 1, z=0.9375; tilted suction 1 mm, cut 1, z=0.8125; tilted aspirator 1 mm, cut 1, z=0.25; tilted aspirator 0.5 mm, cut 3 at (0,1.8,0.9375). These are the preparation blockages previously shown to occur while the physical shaft remains below z=0. Initial occupied native boxes are unchanged by cut union, so it does not repair this outer-voxelization effect. Failure endpoints are not exact first-contact coordinates.

All target values at 0.25 mm are identical to baseline: suction 0.1140625/0.15/0.1155174884 mm³ and aspirator 3.2046875/3.0375/3.161777751 mm³ for aligned/half-phase/tilted grids. Their earlier target shortfalls remain: 65.6–73.8% for suction and 36.8–40.1% for aspirator. More complete sequences do not establish volume convergence.

## Per-row outcomes

Volumes are clipped physical tissue/target volumes. Each row reports only committed actions; a refused suffix is not imputed. Reward uses the unchanged canonical source-cell accounting, including action and motion costs. Tiny roundoff differences below 1e-8 count as unchanged.

| Row | Accepted actions, baseline → union | Target mm³, baseline → union | Reward, baseline → union | Additional physical tissue mm³ |
|---|---:|---:|---:|---:|
| aligned-suction-1mm | 10 → 10 | 0.000000 → 0.000000 | -4.162400 → -4.162400 | 0.000000 |
| aligned-aspirator-1mm | 0 → 0 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 |
| half-phase-suction-1mm | 0 → 9 | 0.000000 → 0.000000 | 0.000000 → -1.124000 | 4.000000 |
| half-phase-aspirator-1mm | 10 → 10 | 0.800000 → 0.800000 | -10.602400 → -10.602400 | 0.000000 |
| tilted-suction-1mm | 0 → 0 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 |
| tilted-aspirator-1mm | 0 → 0 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 |
| aligned-suction-0.5mm | 9 → 10 | 0.000000 → 0.000000 | -3.024000 → -4.612400 | 7.000000 |
| aligned-aspirator-0.5mm | 10 → 10 | 1.587500 → 1.587500 | -16.057400 → -16.357400 | 1.500000 |
| half-phase-suction-0.5mm | 10 → 10 | 0.000000 → 0.000000 | -3.262400 → -3.262400 | 0.000000 |
| half-phase-aspirator-0.5mm | 10 → 10 | 1.300000 → 1.300000 | -14.002400 → -14.002400 | 0.000000 |
| tilted-suction-0.5mm | 9 → 10 | 0.000000 → 0.034855 | -3.324000 → -4.420574 | 4.500000 |
| tilted-aspirator-0.5mm | 2 → 2 | 0.000000 → 0.000000 | -4.472000 → -4.472000 | 0.000000 |
| aligned-suction-0.25mm | 10 → 10 | 0.114063 → 0.114063 | -5.103650 → -5.116150 | 0.062500 |
| aligned-aspirator-0.25mm | 10 → 10 | 3.204688 → 3.204688 | -16.654275 → -16.729275 | 0.375000 |
| half-phase-suction-0.25mm | 10 → 10 | 0.150000 → 0.150000 | -4.494900 → -4.507400 | 0.062500 |
| half-phase-aspirator-0.25mm | 10 → 10 | 3.037500 → 3.037500 | -15.454900 → -15.529900 | 0.375000 |
| tilted-suction-0.25mm | 10 → 10 | 0.115517 → 0.115517 | -4.976904 → -5.045654 | 0.343750 |
| tilted-aspirator-0.25mm | 10 → 10 | 3.161778 → 3.161778 | -16.849517 → -16.893267 | 0.218750 |

Additional tissue does not imply a better objective. In nine of the ten rows with greater removal, target removal is unchanged; the extra credit is outside target. The sole target increase does not offset its additional normal-tissue and action costs. All nonempty experimental sequences have negative total reward. The five coverage-only actions are explicitly charged and do not receive tissue-removal credit.

## Work and bounded execution

Parent exit 0, no stop reason; all rows and all applicable audits complete. Parent wall time is **99.831535 s**, worker **99.416845 s**, below 180/170 s bounds. Peak worker RSS is **471,318,528 bytes** under 2 GiB; parent peak RSS is 27,672,576 bytes. Worker CPU is 98.152227 user + 0.623597 system seconds; parent CPU is 0.146522 user + 0.472887 system seconds. One numerical thread. Final saved progress agrees exactly with the report and has no active native call.

The 146 native calls use **62.680214 s**; independent audits use **34.566842 s**. The retained baseline used 137 calls / 18.099388 native seconds and 8.483284 audit seconds; total worker 27.710535 s and peak RSS 351,191,040 bytes. Descriptive overhead is 3.59× worker time and 1.34× peak RSS. Calls increase by nine because the half-phase 1 mm suction sequence now reaches its tenth action. Separate producer and checker costs both increase; the comparison does not isolate machine variability or normalize the additional accepted work.

Across all returned previews, the model performs **68,307,262 partial-leaf tests** and emits 219,045 sparse witness rows transiently. It certifies 15,344,344 new leaf bits in attempted work: 15,275,960 on committed actions and **68,384 discarded** when a later part of the same preview refuses. Refused previews consume **1,129,021 leaf tests and 2,453 witness rows**. These costs remain included. Leaf-bit totals aggregate different physical grid sizes and include whole-cell fast-path certification; they are computational counts, not an aggregate tissue volume or number of unique physical locations.

The maximum one-preview leaf-test count is 2,171,736 of the 4,194,304 cap; maximum tracked cells 1,236 of 4,096; maximum one-preview sparse witness rows 7,188 of 8,192. No coverage-budget refusal occurs. Cells with incomplete certification remain occupied.

| Spacing | Partial-leaf tests | Attempted new leaf bits | Witness rows | Native seconds | Independent audit seconds |
|---:|---:|---:|---:|---:|---:|
| 1 mm | 3,965,092 | 88,342 | 6,574 | 2.211145 | 1.374461 |
| 0.5 mm | 13,931,174 | 1,428,780 | 40,498 | 10.386381 | 5.784113 |
| 0.25 mm | 50,410,996 | 13,827,222 | 171,973 | 50.082688 | 27.408269 |

Fine-grid rows account for 73.8% of partial-leaf tests. Aspirator accounts for 48,151,402 of all 68,307,262 tests. The fine aspirator audits each take about 6.8–7.0 s, making the independent checker a material, fully counted cost.

| Row | Native seconds | Independent audit seconds | Partial-leaf tests |
|---|---:|---:|---:|
| aligned-suction-1mm | 0.611404 | 0.467728 | 1,096,828 |
| aligned-aspirator-1mm | 0.119532 | 0.000000 | 305,416 |
| half-phase-suction-1mm | 0.464229 | 0.327567 | 736,924 |
| half-phase-aspirator-1mm | 0.873484 | 0.579166 | 1,485,512 |
| tilted-suction-1mm | 0.058288 | 0.000000 | 142,924 |
| tilted-aspirator-1mm | 0.084208 | 0.000000 | 197,488 |
| aligned-suction-0.5mm | 1.168641 | 0.755301 | 1,641,468 |
| aligned-aspirator-0.5mm | 2.999439 | 1.723387 | 3,894,568 |
| half-phase-suction-0.5mm | 1.074168 | 0.645865 | 1,386,576 |
| half-phase-aspirator-0.5mm | 2.842728 | 1.491648 | 3,553,428 |
| tilted-suction-0.5mm | 1.353381 | 0.828626 | 1,883,085 |
| tilted-aspirator-0.5mm | 0.948024 | 0.339285 | 1,572,049 |
| aligned-suction-0.25mm | 4.090506 | 2.158524 | 4,678,828 |
| aligned-aspirator-0.25mm | 12.048970 | 6.797654 | 12,959,356 |
| half-phase-suction-0.25mm | 3.838705 | 2.172212 | 3,838,424 |
| half-phase-aspirator-0.25mm | 11.612757 | 6.793794 | 11,202,532 |
| tilted-suction-0.25mm | 4.783177 | 2.509615 | 4,750,803 |
| tilted-aspirator-0.25mm | 13.708574 | 6.976469 | 12,981,053 |

## Conclusion and limits

Retain the result as a demonstrated complementary-cut representation improvement under an **explicit idealized cut-and-evacuation assumption**. The predeclared sequence-completion criterion passes; reward and volume-accuracy results remain separate and largely negative. The model still uses whole native cells for clearance until coverage is complete, preserves initial outer voxelization, and does not validate actual tissue deformation, cutting or evacuation. This experiment supports maintaining the opt-in representation for further model work; it does not establish a default-model replacement or a patient-policy performance gain.

No learning or patient data entered this test. Neither these 18 rows nor the retained controls attribute the earlier SELECT013 failure principally to representation. The unchanged target volumes and persistent below-surface preparation blockages identify limits that complementary cut union alone does not address.

## Evidence and lean archive

Keep `run-01/report.json` in full: its 398,620 bytes preserve every action, cost and certificate result. Keep launcher, progress and prospective-source records, root release, the three runner files, this result, `saved-summary.json`, and the compact control/promotion records listed in `archive-manifest.json`. Keep failed control receipts/logs and the parity diagnosis alongside the terminal pass. Canonical producer/checker code and their tests are recoverable at the pinned integration commit; do not copy staged/before trees or bulk proof histories. The prior baseline archive is retained separately and referenced by hash, not duplicated.

SHA-256:

- Current report: `3784d71445463bd97ef717ffb0a191852cab0cc75ae85bfe4d26b8c8dff36d42`
- Launcher: `552cabb0aef3117e96deded3c829fb5e6672d5563dda61cfa7c0442d94805d26`
- Progress: `5999e4ee89c3f775d578712bf781bfb704e13b316e0ee5552723fc208a5d10b3`
- Prospective source/declaration: `993daf5f9f09f80e2fd6ea21ccf3aa7171590ad4972868db15d71969c9b220f7`
- Root release: `9343cd2a99719863b5de897f2333e5c7a5a07533862f8c87875408f0a8fd8f54`
- Terminal ten-control receipt: `988d75a3df7062157e63a9b310854e9953ee98a499371ba4785d6de0f676ea13`
- Retained baseline report: `8e40ef8f6c4504c8eedb020687250c59cb4e58f869fe02bf530a9a001761bc9f`
- Saved scalar audit: `5056e8db9e8ebc733a468df453ac776e61dc29ca8fc6b658f7bb757bfe7513e8`
