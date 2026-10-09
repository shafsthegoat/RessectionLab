# Baseline fairness audit v1

Read-only source/result audit, 2026-10-08. Repository HEAD: `eabe64bd8ca9e1d9450b4819f11c9cb1fb859a0a`. No patient images, checkpoint tensors, forwards, training, geometry execution, new experiments, or external sources were opened/run. Existing source tests were inspected, not rerun. Only this ignored build directory was written. Source/result hashes are in `evidence.json`.

## Conclusion

The existing generated native opening task supports a clean within-fixture comparison. It does **not** establish a learned advantage over search: complete search scores 1.100, BC256 scores 1.100, and scratch RL256 scores 1.098. The patient comparison supports only a development result: greedy search beats frozen imitation on all four completed TRAIN cases; representations and model access differ. Keep search as the strong reference and separately test whether learning improves amortized inference or search under a fixed budget.

## What is actually being compared

| Evidence | Matched conditions | Result and attribution limit |
|---|---|---|
| Generated native opening | One six-cell task in a 9×9×7 grid; two tools; horizon two; full tiny image; same nominal/reference target; certified actions; complete-tool native engine; frozen geometric reward | Search 1.100, BC16 0, RL16 −0.896, initial −0.464. BC256 1.100, RL256 1.098. One seed and one training anatomy. Longer fitting establishes capacity, not superiority or transfer. |
| Prepared TRAIN V2 | Same prepared source/access/initial inventory/tools/reward/horizon three; separate arm clones; equal 90 s/468-preview online caps | Search 410.312/789.097/753.905/612.538 versus IL 290.506/672.297/584.914/493.534 on PAT05/22/25/28. PAT16/20 blocked and retained as null. PAT05 trained IL; others are TRAIN development transfer. CNN64 crop versus full nominal search fields. No RL updates. |
| Earlier straight-access planner | Static sampled rigid-tool approaches | `planning.py` computes conditional target accessibility and normal exposure, not sequential contained-cell resection. Do not place its accessibility number in the native-removal score table. |

Evidence: `artifacts/native-opening-{learning,bc-capacity,rl-capacity}-v1/RESULT.md`; `artifacts/prepared-training-planner-comparison-v2/RESULT.md`; `src/resectionlab/planning.py:1`.

BC256 fits all five teacher states using 1,280 loss forwards; its critic remains untrained. RL256 uses 1,024 fresh on-policy episodes, four per update. Search's 20 nominal transitions enumerate 16 terminal sequences; five complete replay/audit demonstrations validate teacher states. RL's final expected stochastic return is 0.546023 and any-target probability 67.99%, despite the near-optimal argmax. Neither policy's softmax is a clinical probability. The RL256 short-tool checkpoint changes from 6.72% to 99.39% STOP preference when only working-length descriptors are replaced with 120 mm: this is a frozen descriptor sensitivity result, not evidence about recertified long-tool actions (`artifacts/native-opening-length-sensitivity-v1/RESULT.md`).

## Exact common contract to freeze

1. **Input track:** Declare either annotation-assisted planning or scan-only inference. Annotation-assisted may openly give the same supplied tumor annotation to all methods; then it tests planning given segmentation. Scan-only requires a TRAIN-fitted/frozen estimator or a separately declared label-independent objective; synthetic intensity thresholds are not MRI estimators. Bind image, support, nominal target, affine, preprocessing, coverage and availability hashes.
2. **Representation:** For a strict same-input ablation, supply the same spatial coverage/resolution to policy and search objective, or give both the same permitted full-volume representation. Report source-matched/full-field search separately if the CNN remains cropped. Do not weaken native collision geometry to fit the crop; share it as an explicit feasibility service. Candidate coordinates and their generation may expose full-field information even when an image channel is hidden.
3. **Action/state contract:** Share the identical proposal provider/version/cap/order, tools, access-selection rule, certified inventory, current tool, cavity update, horizon, STOP/tie semantics and native resolution. The actor DTO has six channels plus coverage/availability, 16 action geometry fields and 10 procedure-state fields. Its mask is only STOP/horizon; upstream native inventory already filters geometry feasibility. These certified actions are model assistance shared by every arm, not perception learned from scratch.
4. **Objective and evaluation:** Freeze weights and outcome definitions before optimization. Keep nominal planning reward separate from evaluator reference reward. All returned plans receive the same independent native replay/audit. Failed, incomplete or unaudited runs retain null outcomes; they do not become zero-score STOP successes.
5. **Learning conditions:** Match architecture and initialization for BC, scratch RL and BC→RL comparisons. Declare the teacher/model access, offline supervision and optimizer budgets. Freeze input normalization and action units; changing those is a new fitted model, not a reinterpretation of old weights.

Evidence: `src/resectionlab/spatial_observations.py:20,30,174,332,380`; `src/resectionlab/native_spatial_task.py:451,499,592`; `src/resectionlab/spatial_policy.py:237`; `docs/real-patient-spatial-training.md:29`.

## Strong permitted-input search baseline

Use the current nominal planning clone and maximize cumulative incremental reward over the **same complete native action sequences**:

`J = sum_t [target_removed_hat_mm3 − 0.2·normal_removed_hat_mm3 − 0.03·nonSTOP − 0.001·complete_tool_path_mm − 0.03·tool_change]`.

Here target membership is the permitted nominal field, normal volume is total contained-cell removal minus that membership, removed cells earn credit once, and STOP adds zero. Constrain actions by the common declared support/access/tool/native engine. The objective is implementable from permitted imaging-derived fields or openly supplied annotations and simulated cavity. It does not optimize neurological outcomes. Functional/vascular unknowns remain unavailable; do not silently make them zero-risk evidence. Do not add a contact-history penalty without adding the corresponding observed memory: the current frozen six-channel actor does not observe retained-contact history.

Keep STOP, uniform legal random, untrained actor, and immediate greedy as diagnostics. The primary search comparator should be bounded multi-step beam search (complete enumeration on tiny fixtures), retain negative access-opening prefixes, and compare STOP at every prefix. Immediate greedy cannot demonstrate a strong sequence-planning baseline on the opening task. Declare beam width, transition mode, time/previews/call limits and incomplete layers; bounded search is not a global optimum.

Add **learned search** by passing the frozen BC or RL policy to the existing `observed_beam_search`: its logits order all legal actions, while the same nominal cumulative reward ranks branches. Its critic is deliberately ignored. Compare identical unguided/guided budgets, proposal inventory and estimator; charge all guidance forwards/hash checks. A gain then means improved exploration under the cap, not a different objective or extra target labels. Full enumeration should recover the same optimum regardless of ordering. Before adding a learned leaf value, separately train/calibrate it on TRAIN trajectories and ablate it; the BC-only critic is not a usable value oracle.

Evidence: `src/resectionlab/native_spatial_task.py:35,508,592,630`; `src/resectionlab/observed_search.py:1,46,94,179,222`; `docs/native-spatial-task.md:21`.

## Leakage checks that matter

| Channel | Required boundary |
|---|---|
| Actual task reward | `NativeSpatialTask._score_record` reads `reference_target`; `planning_clone` substitutes the nominal target. Patient-specific scratch/adaptation in a scan-only test must collect nominal-model rewards, not ordinary evaluator rewards. Training-label supervision on TRAIN is a separately declared offline privilege. Current generated targets coincide, so that fixture cannot expose this leak by itself. |
| Proposals, support, access and crops | Withheld labels must not select endpoints, crop centers, access windows, support expansion or fallback cases. Source nominal/cavity proposals may use the openly permitted nominal field; hiding the channel alone does not produce scan-only planning. Freeze preparation independent of reward ranking. |
| Actor and critic | Six-channel DTO excludes reference labels, per-action true removal sums, rewards and future outcomes. Current actor and candidate-aware critic use this same DTO. Source IDs/provenance are bookkeeping, not learned features. Do not silently substitute the older 15+6 nominal summary-feature policy for the spatial policy. |
| Evaluator metrics and tree | Exact finite-tree expectation and reference removal metrics are assessment only. No outcome-driven checkpoint choice, early success stop, restart, hyperparameter tuning or online adaptation using these values. Distinguish assessment-only tree access from teacher search used for BC. |
| Patient overlap and temporal inputs | Group all visits, crops, derivatives, masks and augmentations by patient; exclude postoperative/future images and unknown-timed molecular/pathology/outcome information from preoperative inputs. Pretrained segmentation overlap is not proven absent. |
| History and world | Exact cavity here is observed simulator state, not recorded surgical observation. Any future uncertain world stays fixed through an episode; planning/selection/evaluation worlds must be distinct. Native task currently rejects nonzero scenario seeds and represents one deterministic world. |

The existing hidden-reference invariance test changes reference labels (including outside support) and checks unchanged DTO/inventory/teacher while actual reward changes (`tests/test_native_spatial_task.py:107`). Extend that criterion to every proposed data adapter, crop/access/proposal path and online-learning callback. Provenance strings alone cannot prove upstream nonleakage (`spatial_observations.py:1`).

## Patient roles and what may be claimed

Locked BTC TRAIN: PAT05/16/20/22/25/28. SELECT: PAT26/27. Reserved later frozen development transfer: PAT29/31. UPenn remains the reserved external-final role; do not use it to repair the present method. `real_patient_learning.py` binds cohort SHA `962d964e…` and checks canonical `BTC:participant_id` identity. SELECT is never a population-gradient source. Role metadata, not any held-out images, was inspected here.

Patient-specific learning is a valid separate mode: freeze the **entire adaptation procedure** on development cases, then restart each test patient from the specified initialization, using only that patient's permitted images/model and the declared budget. Population-frozen measures zero-gradient transfer; population-adapted and patient-scratch measure case-specific optimization. Do not call the latter zero-shot. Repeated rollouts or seeds do not enlarge patient count. A deterministic repeated fixture cannot establish withheld-world robustness. See `MASTER_PLAN.md:498,879` and `docs/real-patient-spatial-training.md:14`.

## Budget and attribution ledger

- Report **cold total** and **reused-case online** time. Charge shared source loading, segmentation/support, access screening and initial candidate inventory explicitly; V2 preparation costs 22.95–26.81 s and 156 previews per completed case before arm clocks. Report inherited preparation of unknown duration as unknown, not zero.
- One uninterrupted online clock covers clone, model planning, policy/hash forwards, candidate preparation, execution/replay, successor inventories and durable terminal export. Independent post-hoc audit is separate but included in total engineering cost. No arm-specific resets or free search calls.
- Record actual native preview entries, nominal transition calls, current-candidate scoring operations, clones, actor/critic forwards, optimizer updates and complete/partial episodes separately. Equal abstract model-call counts do not imply equal work: greedy scores cached certificates without committing every candidate. Native previews also dominate learned inference; actor latency alone is misleading.
- Give equal declared online cap grids plus observed quality/cost curves. BC/RL's offline data and fitting budgets are separately visible, including teacher search, demonstration replay/audit, failed fits, training collection and checkpoint selection. BC256 and RL256 have the same update count but unequal work; this is not equal-compute evidence. Do not double-add overlapping timers.
- Fix hardware/threads; randomize or counterbalance arm order for a speed claim. Current fixed-order single measurements are descriptive. Preserve time/memory failures and the complete patient/method denominator. Cooperative guards need an external process envelope.
- Primary reported gains: BC versus search = amortized teacher imitation; RL versus same initialization/BC = optimization benefit; guided versus unguided search = search efficiency; frozen versus adapted = case adaptation. None is established by a narrower representation, extra labels, a better candidate generator or a larger budget.

Evidence: `src/resectionlab/planning_budget.py:24,120`; `artifacts/native-opening-learning-v1/RESULT.md`; `artifacts/native-opening-rl-capacity-v1/RESULT.md`; `artifacts/prepared-training-planner-comparison-v2/RESULT.md`.

## Small next implementation slice

Before any new large run, make a source-bound comparison declaration containing the common DTO/provider/objective hashes; explicitly distinguish nominal training model from evaluator; include STOP, greedy, multi-step search, frozen BC, scratch RL, BC→RL and frozen-policy-guided search as applicable. Add hidden-reference invariance and common-inventory checks, preserve the existing blocked rows, and profile complete online cost on permitted development input. Fix representation/long-tool compatibility before claiming patient transfer. A missing arm is unexecuted, not a scored failure. The existing generated fixture can verify comparison mechanics; another long fit on that fixture cannot establish the missing generalization evidence.
