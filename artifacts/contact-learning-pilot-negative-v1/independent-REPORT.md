# Independent saved-result review: generated contact learning pilot

**PASS for saved-result consistency; negative learned-policy outcome.** The fixed experiment completed, including all 24 teacher slots, 32 IL updates, 32 RL updates, 32 selection episodes and 64 held-out method episodes. Both learned policies chose immediate STOP on every selection and held-out task. SEARCH reached and retained the goal on 12 of 16 held-out tasks. This supports reporting the completed negative result; it does not support a learned-policy improvement, speed/quality advantage, patient-transfer or clinical claim.

Review scope: canonical source inspection and one standard-library saved-file audit only. No project modules were imported, no tensors decoded or models loaded, and no training, inference, evaluation or native replay was run. All writes are in this ignored review directory. The audit hashes checkpoint bytes and reads only their bounded JSON manifests.

## Identity and gates

- Execution commit: `46d5a47928d4749fbd62faed609e3bab8e3e0388`.
- Experiment identity: `sha256:a01e6ba30fbf1e6efabf70f9637d42c0813931624cb50d2d393858029d4fd894`.
- Saved result: `outputs/learning/public-contact-v1/attempt-01/result.json`, SHA-256 `d57a6590c2ecc058429576d9dd0e0714278dbac5fec5029ee22ef7844ce213c1`, equal to the supervisor receipt binding.
- Common initial checkpoint SHA-256: `f96d73659ce8789b7e496c4a5eb000696a0de25dde67e067f1ddedb0f671d376`; initial parameter identity `sha256:e7215950e221045b8e9612e4482837f9b1ff2c52278454b26efc746598cf1487`.
- IL final checkpoint SHA-256: `5295add15d4504db9148a307deaaf503ee8fadbf070a03ffd0968ce7e06ae719`.
- RL final checkpoint SHA-256: `adc27f7bfb3498fb13953d853d4b0e41a0b927f66c3dd93cccd51b60c52424ce`.

All 14 explicitly declared canonical source files match both the execution commit and the audited working bytes. The recorded worker, supervisor, sampler and source-index hashes also match. The immutable protocol literal matches `experiment-freeze.json`, whose semantic digest matches the released experiment. The source declaration is an explicit 14-file binding, not a complete transitive Python/environment closure. The supervisor does not record a terminal source postguard; this audit supplies a later source/commit comparison without claiming proof that no transient source change occurred during execution.

The canonical contract constructs two separate models from the same seed and rejects unequal initial parameter identities. Both saved update chains start from that shared identity, contain exactly 32 sequential one-step receipts, change parameters at every update, and end at their respective final checkpoint identities. The two final byte hashes, parameter identities and lineage digests exactly match `final-checkpoint-freeze.json` and all 48 learned online episode authorship records.

The source performs bounded decoding and final-lineage validation of **both** checkpoints before `_online` begins. The held-out factory requires the resulting evaluation capability before constructing a held-out task. Each held-out episode binds the same final-freeze digest; saved cost phase order places freeze verification before SELECT and MEASUREMENT_EVAL. No checkpoint selection or online optimizer updates are recorded. This is source-path and saved-evidence verification, not a new model execution or independently signed training proof.

## Splits and training accounting

The split is by whole generated layout, including both goals and all descendant states. Recomputing the declared recipe-hash ordering yields 12 TRAIN, four SELECT and eight MEASUREMENT_EVAL layouts with no overlap. Both final lineages use exactly the 24 TRAIN layout/goal pairs. No SELECT or MEASUREMENT_EVAL binding appears in the checked training samples or RL trajectories.

All 24 teachers completed without search/time caps: 16 contacted-and-retained outcomes and eight STOP trajectories, supplying 40 action labels. IL made 128 sampled state/action draws over 32 four-sample updates and used all 40 distinct saved sample tuples. RL collected 128 four-per-update episodes, 164 transitions and two successful training contacts; its 256-transition cap was not reached. Both methods share initialization, architecture, optimizer settings and 32 updates, but they do **not** use equal numbers of transitions or equal offline work: IL also relies on 1,346 teacher search transitions.

The last update receipts report 11.036684 s for IL and 53.772969 s for RL under their 60 s cooperative training budgets. Teacher trace accounting is 40 transitions plus 80 native verification/replay transitions. RL collection accounting is 164 transitions plus 328 native verification/replay transitions. These totals agree with the retained trajectories and cost counters.

## Held-out comparison

Success means `goal_contacted_and_retained`, not simply an episode with status `complete`. Every method uses the same 16 tasks: eight layouts with a surface and a deep goal each. The initial public observation binding matches across all four methods for each task. All 96 total online rows match their separate saved result and episode files, canonical episode identities, histories, strategy seals, metrics, and checkpoint bindings.

| Method | Surface success | Deep success | All held-out success | Mean reward, all 16 | Mean removal, all 16 |
|---|---:|---:|---:|---:|---:|
| STOP | 0/8 | 0/8 | 0/16 | 0 | 0 mm³ |
| SEARCH | 6/8 | 6/8 | 12/16 | 0.32325 | 1.75 mm³ |
| IL | 0/8 | 0/8 | 0/16 | 0 | 0 mm³ |
| RL | 0/8 | 0/8 | 0/16 | 0 | 0 mm³ |

SEARCH's surface mean reward is 0.5025 with 0.875 mm³ mean removal; its deep mean reward is 0.144 with 2.625 mm³ mean removal. All methods retain the goal cell, but STOP/IL/RL never contact it. The learned policies therefore match the STOP floor on these endpoints. SEARCH also succeeds on 5/8 SELECT tasks; both learned policies and STOP succeed on 0/8. SEARCH beats both learned methods on 12 paired held-out tasks and ties on four. No online rows are missing, unresolved or time/call capped; no saved search beam-pruning flags are set.

Use 16 held-out tasks per method, or eight per goal. The 64 held-out method episodes are four paired comparisons, not 64 independent tasks. The underlying independent split unit is eight layouts; the two goals on each layout are correlated. Do not pool SELECT with held-out results or use a success-only denominator. One seed and one fixed endpoint cannot establish general superiority of a training algorithm. There is no untrained-policy evaluation here, so these results alone do not show that training made a policy worse.

## Resources and reporting limits

The parent receipt reports exit zero, 183.463836 s elapsed, 401,571,840 B sampled peak owned-tree RSS over 882 samples, no cleanup errors and no remaining owned PIDs. Limits were 1,500 s total, 1 GiB sampled RSS and one thread, with no automatic retry. Worker-reported wall time is 180.974139 s and process-lifetime peak RSS is 398,966,784 B. The worker's wall timer stops before its final result serialization; parent time also includes startup/finalization. Stage limits are cooperative; sampled RSS can miss transient peaks.

Saved held-out online totals are STOP 2.196183 s, SEARCH 25.450556 s, IL 12.427907 s and RL 12.050205 s. These include per-method source/inventory, planning, execution/replay and method evidence writes; separate checkpoint-load/offline work and final study serialization are outside those per-row totals. The learned policies' lower online time accompanies doing STOP and is not evidence of a useful speed/quality improvement. Planning-only calls (SEARCH 1,038; IL/RL 16 each) exclude other inventory and replay work. Inclusive nested counter times must not be added. The `complete_online_wall_seconds` and `timing_scope` fields exist in aggregate `result.json` but are added after individual row serialization and are absent from the 96 individual row-result files; this is accounted for explicitly by the audit.

## Reproducible saved-only audit

`audit_saved.py` SHA-256: `534ae88bb7a0cee71cfe8c3d71198b39f74faa873f80349d699ff03a976e3e7a`.

`audit.json` SHA-256: `d97ab2c17d76bf905ee79de2ce7fee1af3899a1f1c7db98809c359199f857e9b`.

The audit output contains exact hashes and sizes for 306 saved/source files (40,565,853 B), split membership, checkpoint identities, training counts, every paired row's compact outcome, and the complete supervisor receipt. The saved-only check exited zero in 0.258 s. No model or native rerun is needed to report this saved result.
