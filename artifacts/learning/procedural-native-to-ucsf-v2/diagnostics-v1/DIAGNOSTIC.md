# Procedural transfer: measured limits in the completed v2 development run

This read-only, post hoc diagnostic used the completed v2 records after all six learning runs and all 13 independent native geometry checks passed. It imported the preserved experiment code, checked source/model/checkpoint hashes, and compared saved initial, selected and latest policies on the same observations. It performed four optimization-world transitions: one fixed first cut in each procedural family and the two already saved SEARCH cuts in the patient simulation. No new search, gradients, checkpoint selection, tuning, final-world sampling or stress-world sampling occurred. Checkpoint bytes remained unchanged.

The exact measurements, policy bindings, observations and script hash are in [diagnostic.json](diagnostic.json); the executed source is [diagnostic-script.py](diagnostic-script.py). Runtime source hash: `e1186e12e79263c41e3cdad06112012702b4126a73fd912c14fa1e41d6d8843f`. Diagnostic elapsed time: 5.09 seconds.

## What changed with adaptation

| Seed | Scratch selected return | Adapted selected return | Scratch / adapted updates |
|---|---:|---:|---:|
| 11 | 245.24 | 171.42 | 14 / 12 |
| 23 | 171.42 | 171.42 | 11 / 10 |
| 47 | 245.16 | 245.17 | 12 / 10 |

Frozen procedural initialization returned 139.62; SEARCH and GREEDY each returned 245.24. Every adapted actor changed and improved over its shared initialization, but this three-seed development comparison shows no advantage over scratch. Scratch seed 23 retained its initial checkpoint; all adapted seeds selected updated checkpoints. Wall budgets were matched, while realized samples and updates differed.

## Initial action ranking changes across anatomy domains

| Initial legal-action feature | Procedural families | Patient simulation |
|---|---:|---:|
| Target benefit, mm³ | 2–32 | 15–174 |
| Normal removal, mm³ | 4–52 | 1–11 |
| Insertion distance, mm | 6.5 | 15.5–19.5 |
| Partial normal contact, mm³ | 32–90 | 9–21 |

The shared actor selected a wide tool in both source families. On the patient initial state it ranked `fine:3` highest: logit 0.600, probability 0.303, nominal immediate reward 14.69. The best immediate wide action `wide:0` had logit −0.024, probability 0.162 and nominal reward 171.58. These are policy action probabilities, not clinical probabilities.

This is not an observed STOP or entropy collapse: initial patient STOP probability was 0.083 and entropy was 1.518 nats out of a possible 1.609. In adapted seed 11, sampled optimization batches reached mean return 245.24, yet the best deterministic selection policy returned 171.42. Successful sampled behavior did not reliably become the top-ranked deterministic action sequence within the budget.

Physical scale and composition changed substantially. The shared actor's largest mean absolute first-layer feature contribution was partial contact in the source families (4.88–5.83), but target benefit in the patient state (9.31). Patient wide-action hidden units were 87.5–93.75% saturated at `|tanh| ≥ 0.99`; their mean tanh derivatives were 0.0060–0.0096. Saturation was also common in the source geometries, so these observations support a conditioning concern, not a causal claim that normalization would fix transfer.

## Value representation and update scale

Both procedural initial states and the patient initial state expose the same critic input, `[1, 0, 0, 0, 0, 0]`. The state-only value head therefore predicts the same 0.819 return before adaptation despite differing physical reward scales. Its latest adapted patient predictions remained 1.13–1.20, while observed episode returns often approached 139–245.

Every online update exceeded the global gradient clipping threshold of 5; recorded norms ranged from 117 to 5,509. Actor gradient norms after clipping were always nonzero (2.14–4.96), and actor hashes changed. These records rule out a disconnected or frozen actor as the explanation. They reveal a poorly matched return scale and limited value representation, but do not establish that clipping itself caused the performance gap.

## The stronger planning limit is the fixed candidate set

After the two saved SEARCH cuts, only STOP remains legal even though the three-action horizon has one unused slot. The simulation has removed 249 of 41,919 mm³ of target (0.594%), leaving 41,670 mm³. This is exhaustion of the current fixed candidate inventory, not exhaustion of the time horizon or evidence that the remaining target is safely removable. Policy improvements alone cannot create additional legal proposals under this frozen model.

A subsequent declaration should distinguish expansion of useful, independently checked cavity-frontier proposals from a learning-only ablation of fixed physical feature/return scaling and critic inputs. Both require fresh method comparisons; neither was attempted here. This diagnostic is descriptive evidence from one previously studied structural patient simulation, two procedural source geometries and three optimization seeds. Motor/language function remains unassessed, access is hypothetical, and no clinical efficacy or safety conclusion follows.
