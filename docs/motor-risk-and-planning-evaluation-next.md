# Next milestone: motor evidence and matched sequential decisions

Status: proposal after the November 16 structure-aware harm audit. No new
training or functional-injury predictor was implemented. Completed patient
tasks remain zero; generated engineering results are not clinical validation.

## Motor-function evidence before risk prediction

The simulator measures immutable target/residual and structure-specific source
removal/cutting, separate bone and preparation costs, modeled time, supported
mechanical/thermal exposure and declared vascular-interface deficits. These
quantities are uncalibrated exposures. A zero-bleed occlusion can still produce
supply/drainage deficits; those deficits now participate in supported canonical
comparison. Neither a tract label nor those model outputs establish functional
loss or its probability. Motor/language consequences remain unavailable.

The next anatomical deliverable should be a source-bound **motor evidence and
encounter contract**, initially reporting geometric interaction and uncertainty,
not a neurological deficit score. It needs patient-specific cortical motor and
corticospinal evidence, acquisition/model/review provenance, positive/negative/
unknown coverage, registration error and the starting exposure state. Preserve
overlaps and distinguish inferred tractography/fMRI/nTMS proposals from actual
stimulation or monitoring observations. Tractography's predictive limitations
are documented in [Voets et al., 2021](https://pubmed.ncbi.nlm.nih.gov/34196915/).

Future clinical calibration would separately require independently held-out,
patient-linked preoperative motor status, spatial/time-stamped mapping/MEP data,
early postoperative DWI/ADC and follow-up motor outcomes with declared time
points. Ischemic and direct/mechanical pathways must be distinguishable; a
matched clinical study found postoperative diffusion abnormalities associated
with new deficits ([Jakola et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4081783/)).
This motivates collecting those observations; it does not validate the current
vascular model as an ischemia or motor-risk predictor. Missing observations
remain unknown, and no clinical thresholds are proposed here.

UCSF-PDGM-0004 is a practical **qualification-packet** candidate, not an executable
patient task. Existing source/transform checks pass, but cropped/unreviewed
anatomy, missing independent intracranial/entry/work-domain coverage, missing
skull/reliable vascular evidence, explicit acquired-source hold and the full
8,928,000-cell grid block admission. Do not raise the two-million-cell policy
or observation limits without uncropped memory/runtime and replay qualification.
Package full-field review proposals and explicit unknown coverage first;
missing modalities and expert review remain external dependencies. Detailed
case evidence and local estimates stay under ignored `testcases/`.

## One bounded, fair decision experiment

Use generated, independently specified supratentorial layouts first. Freeze a
single observable state contract and source-bound skill inventory for all three
methods: original memberships and coverage, current removed cells, both tools,
declared constraints/budgets and accepted history. Supported dynamic models
also need the same declared parameters, topology and cumulative exposure state
in every observation; otherwise exclude that model from all three methods.
Current graph/RL observation parity is not established. No evaluator-only anatomy,
expected winner or future branch outcome enters a policy observation. Keep
patient developmental data out of held-out evaluation.

Compare (1) greedy one-step admissible target gain, (2) current constrained
multi-step search, and (3) a future learned selector over the **same skills**.
Primitive-only RL versus macro-action search is not this comparison. Include
delayed-benefit preparation, a forbidden shortcut, competing harm budgets and
unreachable residuals. Every skill expands to the same authoritative engine
transitions. All methods use identical precommit constraints and checked
withdrawal/STOP, independent of scalar reward. This follows the separation of
policy reward and modeled action specifications in
[Alshiekh et al., 2017](https://arxiv.org/abs/1708.08611); it is not a guarantee
outside the simulator's qualified model.

Freeze identical episode time/action limits and fluid/debris cleanup criteria
as well. The current generated 113-action plan takes about 101.98 modeled
seconds and retains 236 mm³ modeled debris; default RL allows 60 seconds and
zero endpoint debris. Their default success semantics therefore differ.
Version and physically replay a common endpoint/configuration before any
matched experiment; do not treat shared anatomical budgets as proof of parity.

Begin with a 200-transition timing/admission pilot, then freeze a small protocol:
32 training, 8 validation and 16 held-out generated layouts grouped by geometry
family, with three seeds for any later admitted learning run. Predeclare equal
online physical-call and wall-time tiers from pilot measurements. Account for
search model queries, previews and failed proposals; report training transitions
and compute separately. The current adapter remains evaluation-only: a separate
reviewed training-admission change must precede any training, and existing
mechanics/thermal/vascular RL holds remain in force.

Primary outcomes are original-target treatment/residual within independently
declared neural/structure bounds, successful whole-tool withdrawal/STOP, and
zero committed hard violations. Report neural removal/cuts, bone, preparation,
supported exposures, unavailable evidence, refusal rate and runtime separately;
charge shared prefixes once. Reward alone cannot establish success. RL succeeds
only if it improves held-out delayed-benefit decisions over greedy and offers
a reproducible quality/online-compute tradeoff against constrained search.
Report a negative RL result if it does not. Patient motor-risk accuracy requires
a later independent clinical calibration study and is not an endpoint here.
