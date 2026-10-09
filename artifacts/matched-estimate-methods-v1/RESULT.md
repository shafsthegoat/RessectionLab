# Frozen planning comparison: generated tool-transfer failure

An independently replayed comparison of search, imitation learning, RL and a learned-policy/search hybrid completed under one fixed protocol. Both full condition suites were sealed before reference evaluation. No training updates or patient data were used.

| Tool geometry | Search | Imitation | RL | Hybrid |
| --- | ---: | ---: | ---: | ---: |
| Original | 1.100 | 1.100 | 1.098 | 1.100 |
| Both working lengths 120 mm | 1.163 | 1.161 | 0.000, STOP | 1.163 |

These are geometric surrogate returns on a fully observed, six-cell generated task. With longer tools, RL stopped despite seven legal movement alternatives, including search's useful solution. Search and hybrid completed without beam pruning or budget truncation. Moving methods removed 2 mm³ modeled target and 4 mm³ modeled normal tissue; these are simulated quantities.

Independent replay verified all eight complete strategies, 15 committed transitions and 141 microsteps. The worker took 2.9182 seconds; supervision took 4.1885 seconds with 313,311,232 bytes sampled peak RSS and clean termination. Single-run timings do not establish deployment speed.

Saved-score decomposition matched four common movements and three newly legal movements. RL's STOP score was unchanged; all movement scores remained lower with the longer tools. STOP softmax was 60.9844% over the actual legal inventory. The earlier 99.39% descriptor-only result used a different inventory. These are model preferences, not clinical probabilities or calibrated uncertainty. This decomposition does not establish which input feature caused the failure.

[Independent result audit](reviews/postrun/REPORT.md), [saved-score decomposition](logit-decomposition.json), and [exact package inventory](package-sha256.json) preserve the result. The archive contains all 111 original files, including two metadata inventories omitted from the original basename-filtered index; the accounting correction is preserved. Model weights and patient records are excluded.

This result diagnoses frozen RL tool transfer. It does not test the primary privileged-training, limited-input unseen-patient hypothesis or establish physical or clinical validity. A proposed one-forward feature intervention remains separate and unexecuted here.
