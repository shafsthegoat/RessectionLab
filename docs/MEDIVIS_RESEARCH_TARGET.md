# Research claims and evidence

The user-supplied [research charter](MEDIVIS_SUPERGOAL_NEUROLOGICAL_HARM.md)
provides hypotheses and scientific boundaries. Its dataset and product statements
require source verification. It does not replace executed evidence, patient roles,
or the prohibition on paid infrastructure without explicit permission. The
charter referenced e477f5e; integration began at 5198947, preserving all later work.

The central target is inspectable patient-specific strategy research that can
measure geometric feasibility, uncertainty and, when supported, functional or
neurological consequences. A geometric tissue-volume score is not neurological
harm. This remains a research prototype with no MEDiVIS affiliation or documented
MEDiVIS API integration.

| Claim and why it matters | Current evidence | Missing evidence / acceptance condition | Comparison | Allowed scope |
|---|---|---|---|---|
| Complete tools and cavity chronology constrain a plan | Native geometry and independent replay on real BTC anatomy; rejected shafts retained | Multi-strategy benchmark against endpoint/centerline/static abstractions | Same anatomy and instrument, simpler geometry | Tested geometric feasibility; not tissue mechanics |
| Learning improves sequential choices | PAT05 matched visited-state imitation 290.51 vs original-only 239.11 | Repeated seeds and separate-patient benefit; same compute/information | Greedy 410.31; two-update RL unchanged at 12.70 | One same-patient imitation signal; not RL superiority |
| Frozen learned choices transfer usefully | Precision repeat: PAT22 672.30 vs greedy 789.10; PAT28 493.53 vs 612.54; PAT25 STOP-only; PAT16/20 blocked; nine episode audits pass | Address action coverage, then freeze evaluation outside training/model selection | Search, diagnostic random; report all five TRAIN cases | Development transfer only; no population generalization claim |
| Uncertainty changes strategy decisions | Existing uncertainty components preserved | Coherent nonzero worlds; measured fragility/ranking changes and source-specific calibration limits | Nominal plan/fixed margins | Model-conditioned sensitivity, not clinical probabilities |
| Functional evidence represents the patient | Evidence classes and availability retained; current learning task lacks function/vessels | Patient-specific measurements or explicitly scoped population priors; coverage and registration validation | Missing-evidence and geometric baselines | Encounter/exposure only unless a separate outcome claim earns support |
| Observed injury predicts observed neurological outcome | RHUH40 simple baseline audited: adding preoperative volume worsened primary Brier by0.00120647; no injury features validated; timing/domain, BTC linkage and63-patient preprocessing limits remain | Justified imaging measurements, endpoint meaning and independent patient-level evaluation | Retain the frozen baseline-function/volume comparator; location only where measured | Observational baseline only; no validated injury-outcome or route-harm model |
| An unperformed strategy predicts neurological deficit | No supporting evidence | Explicit simulator-to-observed injury mapping, distribution support and defensible causal assumptions | Observed-surgery association is insufficient | Unsupported; no paralysis/safety percentage |
| Mechanics predicts measured interaction | Analytical solver controls pass; specimen convergence unresolved | Converged numerical solution plus independent measured force/displacement validation appropriate to claim | Simpler mechanics/interpolation | No validated cutting, patient stiffness or injury prediction |
| Observed anatomy updates invalidate stale plans | RESECT/ReMIND work preserved | Frozen pre-update plans, permitted observed updates and independent replay | Stale plan vs re-evaluated/replanned result | Retrospective update research until tested |
| Desktop exposes supported research | Electron implementation and unrelated UI work preserved | Current packaged snapshot exercising evidence-backed pipeline, performance and visual checks | Reproducible local workflow | Research interface, not autonomous surgery or clinical recommendation |

## MEDiVIS context checked against official pages

Checked October 5, 2026 UTC. [Studio](https://www.medivis.com/studio) already
describes imaging, segmentation, reconstruction, MPR/fusion and saved trajectories.
Our research contribution must be measured beyond those viewer capabilities.
[Cranial Navigation](https://www.medivis.com/navigation/cranial) describes patient
registration, tracked instruments and planned trajectories; this motivates our
coordinate and stale-state checks, without establishing integration.
[Frontier Agents](https://www.medivis.com/frontier/agents) describes models invoking
validated tools. [Frontier Robotics](https://www.medivis.com/frontier/robotics)
describes supervised execution of approved plans. These are product/research
statements from MEDiVIS, not validation of RessectionLab or proof of clinical
performance for any proposed extension.

See [PROJECT_STATUS](../PROJECT_STATUS.md) for actual runs, artifacts and open
issues. Do not promote a claim because code or a software test exists alone.
