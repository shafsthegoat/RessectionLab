# Retired documentation · November 26, 2026

Repository upkeep follows the [maintenance policy](../CONTRIBUTING.md). The files below
were removed from the tracked tree because they were superseded, gave outdated
instructions to agents, or recorded one-off runs that nothing linked to.
They are ordinary Git deletions; every byte remains in history.

Last commit containing all of them: `9b079d283f7ee3ba61986de9ff83e837489d7517`
(branch `shaf/interactive-operative-practice-20261121`, November 25, 2026).

```sh
git show 9b079d283f7e:<path>            # read one file
git checkout 9b079d283f7e -- <path>      # restore it into the working tree
```

Restore a file only when a current workflow needs it again, and link it from
[AGENTS.md](../AGENTS.md) or a maintained guide when you do.

## Superseded steering prompts and agent handoffs

Pasted October/November `/goal` and steering messages and the original implementing-agent brief. Two target the deleted `codex/patient-specific-planner` branch; the handoff still names Slicer/PySide6 choices and a missing `DATASET_MANIFEST.json`. Current direction lives in [current status](current-status.md) and the [decision-system objective](decision-system-objective.md). The original research plan remains in [MASTER_PLAN.md](../MASTER_PLAN.md).

| Path | Lines | Title |
| --- | ---: | --- |
| `docs/CODEX_STEERING_PROMPT.md` | 62 | /goal — Deliver probabilistic glioma route planning, then demonstrate useful learning |
| `docs/MEDIVIS_SUPERGOAL_NEUROLOGICAL_HARM.md` | 3,544 | /goal — Transform RessectionLab into a MEDiVIS-caliber patient-specific surgical strategy, neurological-harm, robustness, and learning research platform |
| `IMPLEMENTING_AGENT_HANDOFF.md` | 109 | Implementing-agent handoff |
| `docs/HYBRID_WORLD_MODEL_STEERING.md` | 476 | RessectionLab: evidence-grounded hybrid world model and RL steering |
| `docs/INTEGRATION_FIRST_MULTIMODAL_STEERING.md` | 1,130 | RessectionLab: Integration-First, Multimodal Surgical Simulation and RL Steering Directive |
| `docs/SUPERGOAL_REAL_OBSERVATIONS.md` | 624 | SUPERGOAL: real-data-driven glioma surgery planning, learning and rehearsal |
| `docs/CORE_IDEA_VALIDATION_STEERING.md` | 45 | Reprioritize RessectionLab around validating the core idea as quickly and credibly as poss |

## Dated status diaries

Per-step narratives that duplicated current status and the experiment log. Their results and negative findings remain in Git history and in [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md).

| Path | Lines | Title |
| --- | ---: | --- |
| `PROJECT_STATUS.md` | 3,180 | Project status |
| `docs/REAL_OBSERVATION_EXECUTION_LEDGER.md` | 1,962 | Real-observation execution ledger |

## Superseded cleanup pass

The November 16 support pass, replaced by the [cleanup execution record](cleanup-execution-20261116.md).

| Path | Lines | Title |
| --- | ---: | --- |
| `docs/cleanup-20261116.md` | 155 | Bounded preservation cleanup · November 16, 2026 |
| `docs/cleanup-20261116.json` | 1,024 | (data file) |

## Unlinked one-off experiment notes

Dated results of single runs, preparations and diagnostics (mostly October 4 to 9) that no README, maintained doc, code or test linked to. The code they describe is unchanged. Some are still named inside `artifacts/` provenance records, which keep the original path; restore the file with the command above when you need it.

| Path | Lines | Title |
| --- | ---: | --- |
| `docs/architecture-evidence.md` | 63 | Architecture evidence — October 4, 2026 |
| `docs/brain-extraction-benchmark.json` | 327 | (data file) |
| `docs/brain-extraction-pat16-pat20.md` | 98 | PAT16/PAT20 frozen extraction repeatability batch |
| `docs/btc-pat05-independent-qc.json` | 81 | (data file) |
| `docs/btc-queued-acquisition.md` | 49 | Bounded BTC queue execution |
| `docs/closest-work.md` | 15 | Closest-work audit |
| `docs/development-acquisition-queue.md` | 76 | Next development acquisitions: metadata audit |
| `docs/diffusion-tools.md` | 130 | Optional local diffusion correction tools |
| `docs/experimental-capsule-cache.md` | 70 | Experimental exact capsule-cover cache |
| `docs/hbe-axial-resolution-extension.md` | 26 | Fixed axial mesh-resolution extension |
| `docs/hbe-evaluation-interface.md` | 43 | HBE numerical evaluation interface |
| `docs/hbe-halfheight-equivalence.md` | 24 | HBE axial half-height equivalence |
| `docs/hbe-halfheight-readout.md` | 44 | Half-height numerical readout |
| `docs/hbe-v5-n12-saved-attempt-replay.md` | 50 | N12 saved-attempt numerical replay |
| `docs/hbe-v5-n8-canary-source-only.md` | 11 | HBE v5 first-canary source preparation |
| `docs/hbe-v5-source-binding-gate.md` | 9 | HBE v5 source-deck binding gate |
| `docs/immutable-validation.md` | 146 | Full-suite validation of committed source |
| `docs/independent-geometry-batch-prototype.md` | 97 | Independent segment/box batch prototype |
| `docs/independent-native-batch-comparison.md` | 68 | Prospective single-history comparison |
| `docs/mechanics-nonpatient-sparse-feasibility.md` | 11 | Non-patient tet10 sparse-solver preparation |
| `docs/mechanics-patient-constraints.md` | 114 | Conditional patient displacement: FEBio constraint feasibility |
| `docs/mechanics-patient-mesh-boundary6.md` | 11 | Case4 boundary-6 mesh candidate — source-only preparation |
| `docs/mechanics-patient-mesh-curvature.md` | 29 | One curvature-aware Case4 candidate — preparation only |
| `docs/mechanics-patient-mesh-diagnostic.md` | 11 | One saved Case 4 surface diagnostic |
| `docs/native-axis-cache-paired-probe.md` | 63 | Prospective fixed-trace public cache probe |
| `docs/native-axis-inspection-bridge.md` | 145 | Sidecar contract for neighboring-path inspection |
| `docs/native-axis-performance-review.md` | 140 | Native axis performance review |
| `docs/native-axis-raw-update-pilot.md` | 163 | Prospective pilot: one RAW update on the native axis model |
| `docs/native-axis-scratch-feature-units.md` | 75 | Prospective paired scratch input-unit contrast |
| `docs/native-cache-payload-sizing.md` | 77 | Prospective raw capsule payload sizing |
| `docs/native-cache-query-reconstruction.md` | 62 | Saved cache query reconstruction |
| `docs/native-discretization-control.md` | 54 | Fixed-scene native discretization control |
| `docs/native-ingress-screening.md` | 17 | Initial whole-shaft ingress screen |
| `docs/native-opening-bc-capacity.md` | 51 | Fixed longer imitation fit on the native opening fixture |
| `docs/native-opening-rl-capacity.md` | 62 | Fixed longer scratch RL fit on the native opening fixture |
| `docs/pat05-access-preparation.md` | 15 | PAT05 canonical access metadata |
| `docs/pat25-ingress-access-diagnostic.md` | 31 | PAT25 screened-access preparation diagnostic |
| `docs/prepared-training-planner-comparison.md` | 79 | Prepared TRAIN planner comparison |
| `docs/preprocessing.md` | 150 | Preprocessing and rigid alignment — October 4, 2026 |
| `docs/prior-proposal-persistence.md` | 70 | Registered prior proposals: local persistence and inspection |
| `docs/proposed-feature-unit-mechanism-diagnostic.md` | 52 | Saved-record diagnostic after feature-unit completion |
| `docs/proposed-native-scaling-ablation.md` | 37 | Native actor feature-unit study |
| `docs/real-spatial-search-comparison.md` | 64 | First real-only comparison runner |
| `docs/remind-native-independent-qc.md` | 20 | Independent ReMIND-001 native conversion review |
| `docs/research-estimate-planning.md` | 29 | Research planning from scan-derived estimates |
| `docs/resect-brain-domain-readiness.md` | 78 | Case4 MRI domain readiness |
| `docs/resect-case4-boundary6-fidelity-witness.md` | 15 | Case4 boundary6 saved-geometry fidelity witness (preparation only) |
| `docs/resect-measurement-sensitivity-addendum.md` | 42 | RESECT measurement scales and prospective sensitivity addendum |
| `docs/rhuh-first-image-processing-reconciliation.md` | 15 | First RHUH image: processing and frame reconciliation |
| `docs/rhuh-fixed-plane-qc.md` | 33 | RHUH fixed-plane display: preparation only |
| `docs/rhuh-single-image-inspection.md` | 47 | One RHUH image: bounded offline inspection helper |
| `docs/rl-spatial-encoder-evidence.md` | 89 | Scan-conditioned policy: compact 3D evidence and implementation direction |
| `docs/rl-spatial-encoder-independent-review.json` | 97 | (data file) |
| `docs/scan-diagnostic-runner.md` | 9 | Case4 diagnostic handoff |
| `docs/scan-preprocess-bridge.md` | 11 | Scan-only diagnostic preprocessing bridge |
| `docs/scan-target-adapter.md` | 9 | Scan-only adapter source slice |
| `docs/spatial-multiscale-views.md` | 17 | Permitted multiscale image preparation |
| `docs/training-observation-coverage.md` | 13 | Fixed TRAIN observation coverage |
| `docs/training-observation-grid-binding-v2.md` | 20 | PAT05 exact task-grid binding correction |
| `docs/viewer-interaction-performance.md` | 95 | Viewer interaction performance — 15 October 2026 |
| `docs/vitaldb-recorded-component-intake.md` | 83 | VitalDB recorded component replay |
