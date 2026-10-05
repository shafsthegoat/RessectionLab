# /goal — Deliver probabilistic glioma route planning, then demonstrate useful learning

Continue on `codex/patient-specific-planner` in `shafsthegoat/RessectionLab`. Reconcile this steering note with current HEAD before acting. The review baseline is `5ca02451d63412d074281f8fde5407420a2f0903`; preserve newer work and unrelated changes. Read the three existing specification documents, the current status, `docs/rl-observation-gap.md`, and the completed feature-unit and native-axis pilot reports. Do not rewrite the entire architecture or repeat already completed investigations.

## Actual objective

Deliver an instrument-aware, patient-specific, probabilistic research planner that compares glioma access routes and sequential removal strategies under motor/language anatomical uncertainty. Actual parameter updates are not the product milestone. A spatial observation schema alone is not a trained planner. Search is allowed to be the strongest product mode; maintain a genuine, fairly evaluated RL investigation.

Keep the current Electron interface. Prioritize planning/data integration over additional cosmetic, packaging, or saved-artifact diagnostic work unless a correctness defect blocks this milestone. Keep necessary provenance, geometry checks, cancellation, and regression testing; do not weaken them.

## 1. Finish one probability-aware patient case before expanding the research surface

Use public data and local compute. Consult `DATASET_ACQUISITION_ADDENDUM.md` for exact sources and roles. Preserve existing BTC patient assignments and unopened groups. Try verified UCSF-PDGM v5 diffusion with its matching per-exam rotated gradients; retain preprocessing caveats. Use IDC-hosted ReMIND or UPenn data for appropriate structural/segmentation work when TCIA transfer is blocked. Prefer established processing or valid released derivatives over inventing a new distortion-correction method as a prerequisite.

Create an explicit functional-evidence object containing motor/language arrays, physical transforms, coverage, uncertainty model, source identity, and review status. Carry it through case loading, route search, native refinement, checkpoints, independent evaluation, replay and export. The reviewed desktop refinement path does not forward functional maps or a configurable world generator. Audit hard-exclusion propagation as well.

A provisional research analysis using aligned population priors must remain labeled population-derived and uncalibrated. Engineering QC is not expert approval. Unsupported alignment, absent tracts and absent vascular coverage must never become zero risk. Preserve a clearly separate structural-only mode rather than silently claiming complete functional assessment.

Evaluate complete tool/sequence events over coherent, nonidentical anatomical scenarios. Record event definition, counts, denominator, conditional Monte Carlo interval, generator version, and mean/tail surrogate costs. Zero perturbations with different seeds do not test robustness. Do not sum voxel probabilities or call these postoperative deficit probabilities. Freeze assumptions during each run.

Acceptance: one real case has independently checked alternatives whose reported functional/sensitivity metrics actually respond to evidence, tool and uncertainty changes. Include analytical positive and negative controls; do not force a ranking reversal when the geometry does not warrant one. Preserve rejected candidates and explanations.

## 2. Make the decision problem worth learning

Keep the original bounded configurations as regression fixtures. Integrate the cavity-dependent proposal work into a common nominal-observation interface for search and RL. Add useful exposed-patch and tool/configuration choices without allowing through-tissue travel, unsupported removal or loss of STOP.

Measure action-space coverage and horizon sensitivity before training. Explicitly distinguish static accessibility from valid sequential removal. Investigate the strong source-grid alignment dependence: test subvoxel shifts, revoxelization/orientation and resolution convergence on analytic physical scenes. Rigidly transforming a fixed voxel grid alone does not test discretization bias. Do not alter tool dimensions to manufacture favorable resection numbers.

Keep annotation-assisted and inference-only tracks separate. Supplied reviewable segmentations are legitimate in the former. In the latter, hidden reference masks may influence neither actor inputs nor candidate ordering, filtering or masks. Test this by changing private reference truth while holding permitted observations fixed. Analytical predictions from permitted nominal anatomy are allowed and must be equally available to competitors.

## 3. Improve learning deliberately

Retain the completed fixed-unit result instead of repeating a raw-input sweep. Test a compact spatial/candidate encoder with coverage channels, cavity/contact history, tool geometry, access context and remaining budget. Give the critic the relevant spatial/candidate context too, not only six aggregate fractions. Start with the smallest representation that resolves controlled ambiguous states. Add recurrence only for information genuinely missing from the current observation.

Use nominal-model search as a teacher: compare behavior cloning, search-guided policy/value improvement, scratch RL and unguided search. Expert Iteration and DAgger are useful references below. Pure cloning is imitation learning, not an RL result. Reward-based refinement or search-policy improvement must be named accurately.

Measure actor/critic loss scales and gradients; test justified target normalization without changing reported physical objectives. Preserve STOP. Use a declared curriculum that includes costly openings, delayed benefit, competing functional hazards, varied tools and uncertain observations. Do not make the task favor RL by crippling search.

Profile before longer runs. The expanded-axis pilot spent substantial time in previews and integrity checks for one update. Improve static/dynamic separation, shared computation and regional geometry queries; measure cache hit rates and actual wall time. A faster training backend must be checked against the native reference on full selected histories, not revive the invalid coarse-volume model.

## 4. Evaluate the useful question

Compare SEARCH, policy-only, learned-guided SEARCH and patient adaptation under the same permitted information, objectives, proposals and explicit resource budgets. Report offline cost, online setup, optimization, selection and independent checking separately. Compare quality at matched time and time at matched quality. Keep an untrained initialization control. One optimizer update is a pipeline check, not a substantive training budget.

Measure target/residual volume, normal tissue/contact, functional events and coverage, Pareto quality, constraint violations, expansions, latency and failure/abstention. Use independent patients for generalization claims and genuine nonzero uncertainty for robustness. Preserve current BTC roles. No human-pretraining claim may be made from procedural families. Do not tune on final evaluation.

Fix the outstanding integrated regression failures and obtain a current full pass before release. Add lightweight synthetic-fixture CI. Make one end-to-end workflow work: load/review evidence -> compare routes -> refine/search -> inspect uncertainty -> replay -> save/reopen/export. Do not require population pretraining or clinical calibration before this research milestone can exist.

## 5. Make the planner reusable

Expose a small local planning interface accepting a versioned case, tool catalog, constraints and uncertainty configuration. Return plan IDs, physical frames, full tool poses, predicted events, assumptions, evidence and provenance. Keep the UI a client of this interface. Demonstrate export/round-trip interoperability with an independent imaging viewer. Do not claim Medivis SDK compatibility or affiliation without a documented integration contract.

Deliver executed artifacts and a concise status: what routes can now be compared, which uncertainty is active, where learning helps or fails, and what remains unsupported. Commit and push coherent milestones. Public data acquisition and ordinary free local engineering are authorized; do not incur cloud charges or change billing.

References:
- Expert Iteration: https://arxiv.org/abs/1705.08439
- DAgger: https://proceedings.mlr.press/v15/ross11a.html
- Value-target normalization: https://arxiv.org/abs/1602.07714
- Optional later belief-space planning: https://proceedings.neurips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html
