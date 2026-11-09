# Decision-system objective and execution contract

The desktop executes `App → useSurgical → Electron IPC → Python surgical_bridge →
SurgicalEngine`. The engine owns material checks, complete-tool swept geometry,
both instruments, declared protection, reviewed task boxes, physical effects and
transaction publication. Geometry shortlists and scalar rewards cannot authorize
an engine refusal. The original reported brainstem run has not been identified;
the replacement regression is explicitly generated engineering geometry.

Four separate concepts currently run:

|Concept|Implementation|Meaning|
|---|---|---|
|Feasibility|`SurgicalEngine`, `surgical_rl_protection`, `surgical_task_domain`, checked staged skills|Hard native constraints plus declared tissue/action/computation limits. Missing anatomy remains unknown.|
|Planning preference|`surgical_staged_planning`, `surgical_sequence_search`, `surgical_two_tool`, `surgical_preparation_planning`|Staged baseline maximizes immediate target mm³. New lookahead retains target/non-target/time Pareto outcomes and recommends lexicographically: more target, less non-target, less modeled time. Two-tool comparison minimizes seconds after its progress gate; preparation comparison charges the whole prefix and suffix.|
|Episode reward|`surgical_rl_rewards.RewardTracker/compute_reward`|Illustrative engineering return, separate from treatment outcome and feasibility. No training occurs in this milestone.|
|Policy training loss|Existing policy-specific BC/PPO fitting code|An optimization objective over observations/returns; no anatomical or outcome certificate. Existing physiological, continuum, heat and vascular RL holds remain.|

Default reward weights are target +1, healthy removal −2, bone −0.05,
deformation −1, bleeding −1 and protected damage −5 per 1,000 mm³ of their
respective source or modeled quantities; cut −1 and vessel damage −4 per
100 mm²; thermal −0.1 per 100 proxy units; near contact −1 per 1,000 mm³·s.
Attempted action and second cost 0.001 each, invalid action 0.1, protected refusal
1. These scales are engineering choices, without clinical calibration.
`SurgicalRLEnv.reset` recreates the tracker and engine. Removal uses unique
source IDs and injury proxies are monotone. A fresh tracker at an arbitrary
checkpoint cannot recover absent historical strain peaks.

`summarize_engine_outcomes(...).sequence`, also exposed as `sequence_outcome`,
is the checked complete-sequence outcome contract used by the new staged output
and desktop breakdown. It validates the exact source/action baseline, counts the
shared prefix once, retains access/preparation/tool-exchange/exit costs, and
separates cumulative, incremental and peak quantities. Native continuum and heat
exposures keep their model units. Instantaneous lattice displacement is not
reported as accumulated injury. Missing motor/language, mechanics or vascular
evidence remains null. Comprehensive swept-shaft contact and calibrated clinical
injury remain unsupported.

Owned categorical target removal is treatment credit. Original target annotation
membership is reported separately, including cells later owned by a protected
or vascular label. Removing all owned target cells cannot erase that residual.
Categorical-only scenes lack original annotation evidence and report it unknown.
Historical task gates keep their declared owned-target definitions; their large
500 mm³ floors do not qualify small synthetic targets for a new study.

Horizon 1 preserves the existing immediate-removal staged controller. The desktop
offers horizon 2 or 3 with beam width 3; the backend permits width 2–4 and at
most 24 work expansions per selection. Native checks and checked exits verify
every retained node. A reserved beam slot retains cavity-changing preparation
with zero immediate target removal. Final choices require positive target gain;
preparation alone never becomes successful treatment. Failed search means a
finite generator/controller or computation limit, not anatomical impossibility.
External radius changes and access/re-entry still use the existing outer schedule;
this milestone does not search their full joint configuration space.

`TransitionMeter` charges previews, discarded probes, checked exits, commits and
replay. Those are computation costs, separate from accepted patient treatment.
The source-geometry cache reuses only immutable label counts and target
coordinates/octants. Dynamic removals, tools, constraints and outcomes are
remeasured. Scene/source/transform edits invalidate it. Tests compare cached and
uncached results and exercise cavity edits. Native execution remains the main
planning expense; no end-to-end speedup is claimed from the geometry cache.

The current generated fixed-radius regression demonstrates clearance followed by
target work, exact replay and withdrawal. It is a mechanism check, not patient
performance or superiority over the strongest full staged controller, which can
solve that coupon using a larger-tip pivot. The five-case baseline, original
incident search, matched performance evidence and remaining work are indexed in
`docs/decision-system-progress.md` and the ignored local evidence directory.
