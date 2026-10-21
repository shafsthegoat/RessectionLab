# Bounded two-anatomy RL pilot: completed, no learned target-removal gain

The one admitted run completed all 12 genuine REINFORCE/Adam updates (four each for seeds 17, 29 and 43). Every update changed the parameter hash; all three checkpoints match the complete per-seed hash chains. All six trained trajectories, including their common exposure prefixes, were executed again through public engine actions and reproduced their final state, measurements and reward ledgers exactly. The learned deterministic assessments removed **zero additional target on both anatomies**. The policy is not promoted.

This is simulated experience on two already exposed development sources, with both sources participating in optimization and assessment. It provides no held-out, recorded-operative, clinical or statistical efficacy evidence. Privileged source maps selected a frozen common exposure task; action selection thereafter used only the published local observations. The shared native guards remained active.

## Matched assessments

Each row contains all three seeds. Return, tissue quantities and wall time are means per local rollout; command counts are totals across the three rollouts. Target/healthy are additional mm³ after common exposure. Vessel damage is mm²; bleeding is mm³. Collisions count rejected commands with a collision reason; rejected and partial commands remain separate. Zero-target moves are a **proxy**, defined below. Wall time includes episode fork and local assessment, and excludes exposure, fitting and full replay.

| Case | Method | Return | Target | Healthy | Vessel / bleeding | Collision / rejected / partial | Zero-target moves | Wall s | Healthy budget |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| sub06 | Untrained | -0.171478 | 0 | 0 | 0 / 0 | 0 / 5 / 1 | 12 | 2.649 | Met |
| sub06 | Random | -0.026898 | 11.333 | 0 | 0 / 0 | 0 / 1 / 5 | 8 | 2.919 | Met |
| sub06 | Scripted | 0.100800 | 108 | 0 | 0 / 0 | 0 / 0 / 0 | 0 | 4.318 | Met |
| sub06 | Observed greedy local | -0.506033 | 0 | 0 | 0 / 0 | 0 / 15 / 3 | 18 | 3.255 | Met |
| sub06 | Trained | -0.003200 | 0 | 0 | 0 / 0 | 0 / 0 / 0 | 6 | 1.844 | Met |
| UCSF0004 | Untrained | -0.205200 | 0 | 0 | 0 / 0 | 6 / 6 / 0 | 13 | 0.188 | Met |
| UCSF0004 | Random | -0.140267 | 4 | 3 | 0 / 0 | 4 / 4 / 0 | 8 | 0.171 | Met |
| UCSF0004 | Scripted | -0.000400 | 14 | **6** | 0 / 0 | 0 / 0 / 0 | 0 | 0.134 | **Failed all 3 seeds** |
| UCSF0004 | Observed greedy local | -0.606000 | 0 | 0 | 0 / 0 | 18 / 18 / 0 | 18 | 0.182 | Met |
| UCSF0004 | Trained | -0.204800 | 0 | 0 | 0 / 0 | 6 / 6 / 0 | 12 | 0.170 | Met |

Protected damage was also zero in every assessed rollout. The five-mm³ incremental healthy threshold is checked **after** each real action. UCSF scripted reached six mm³ healthy removal on its second action and was stopped, so this arm failed the declared local budget. This post-action check is not a guarantee that every action remains below five mm³. The failed arm is excluded from eligible-baseline comparisons.

The strongest observed baseline satisfying this local budget was the fixed inferior-advance script for sub06 (108 mm³ target, zero healthy and zero rejected commands in each seed). For UCSF, seeded random had the highest mean return among budget-eligible arms and was the only such baseline to remove additional target: target/healthy were 6/4, 0/0 and 6/5 mm³ for seeds 17, 29 and 43. Its four shaft-collision rejections remain a material limitation. These local outcomes do not justify clinical use or a general policy recommendation. The learned assessments improved some invalid-action costs relative to untrained/greedy arms, but did not improve target removal and lost to the eligible case-specific baseline in mean return.

The common setup is retained in cumulative metrics: sub06 removed 27 mm³ target and 710 mm³ bone; UCSF removed 8 mm³ target and 16 mm³ healthy tissue. Setup costs and injury do not disappear when a local policy stops. No assessed trajectory reached native whole-target success.

## Observed policy behavior and fitting evidence

Across six trained assessments, there were 21 commands: three real STOP deactivations (14.3%), zero inspections, six shaft-collision rejections (28.6%), and twelve accepted alternating z-axis moves that removed no additional target. Sub06 seeds 17/43 and UCSF seed 17 stopped immediately; seed 29 alternated z+/z− for six commands on both anatomies; UCSF seed 43 repeated a rejected y− request six times. These records describe the zero-progress result; they do not establish a unique causal explanation.

Training sampled 119 real commands over 24 episodes: 11 STOPs, 11 inspections, 33 rejections, 21 partial outcomes and 65 accepted outcomes. Training removed 104 mm³ additional target in total across independently reset local episodes, while deterministic final assessment removed zero. Exposure and episode resets mean this sum is experience, not a single surgical removal volume. All 12 combined policy-plus-entropy losses and pre-clipping gradient norms were finite. Losses ranged from -40.229431 to -0.365921; pre-clipping norms ranged from 0.098085 to 6.025695, with eight updates clipped at one. Negative objective values are not evidence of useful behavior. Separate policy-loss and entropy scalars were not logged; the actual combined objective, reward terms and every update hash are retained.

A zero-target movement attempt requests a two-mm axis move and obtains zero tumor-removal reward. This includes rejected/partial commands, accepted reversals and healthy-only movement. The CSVs also retain its nominal requested distance (two mm times count). Actual wasted travel is not measured and is not inferred from this proxy. The trained assessments contain 18 such attempts (36 nominal requested mm).

Both preparation Frazier continuations removed zero debris, so the fixed suction tail was omitted equally from all assessment arms. No result here demonstrates learned instrument switching or successful suction.

## Cost and evidence

The fit, matched assessments and physical replays used 350 public transitions in 190.888 wall seconds, with a maximum action latency of 2.019 seconds and process peak RSS of 736,559,104 bytes (about 702.4 MiB). These were below the admitted 900-second, 1,000-transition, 10-second/action and six-GiB limits. Four recorded preparation attempts cost 200 transitions and 222.750 seconds, including the attempt blocked when `imaging.py` changed. The complete recorded runner total is 550 transitions and 413.638 seconds. Earlier manual exploration is separate. There was no repeat fit, sweep, best-seed selection or checkpoint promotion.

- [All 30 seed assessments](all-seed-assessment.csv), [ten case/method comparisons](case-method-comparison.csv), and [12-update curve with losses, gradients and parameter hashes](update-curve.csv).
- [Compact structured summary](summary.json) and [complete trajectories/reward terms](result.json).
- [Post-hoc audit](verification.json), [reproducible read-only audit script](analyze_result.py), and [file/source hash index](evidence-index.json).
- [Final hash-bound preparation](preparation.json), [all preparation costs](preparation-history.json), and [lead-authorized launch gate](launch-gate.json). Preparation SHA: `195327d1222cbd9ac8f8e032f5335d77524cda8522dc739756a5b8d9c17fc823`.

All three checkpoint files, two prefix snapshots and six complete final replay snapshots are retained beside this report and pinned in the index. The audit reads recorded evidence and checkpoints; it performs no new environment transitions or optimizer updates. The runner's physical replays were already included in the measured run budget.
