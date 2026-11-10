# Decision-system objective and execution contract

The desktop executes `App → useSurgical → Electron IPC → Python surgical_bridge →
SurgicalEngine`. The engine owns material checks, complete-tool swept geometry,
both instruments, declared protection, reviewed task boxes, physical effects and
transaction publication. Geometry shortlists and scalar rewards cannot authorize
an engine refusal. The original reported brainstem run has not been identified;
the replacement regression is explicitly generated engineering geometry. The user
clarified that the concern was an implausible access route (possibly posterior or
inferior), not an established named-structure injury. A case-specific cranial
entry window and actual skull removal are now enforced mechanisms; qualifying
those mechanisms on patient anatomy remains open.

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
evidence remains null. Whole-tool capsule checks establish geometric constraints;
calibrated shaft-contact mechanics and clinical injury remain unsupported.

Owned categorical target removal is treatment credit. Original target annotation
membership is reported separately, including cells later owned by a protected
or vascular label. Removing all owned target cells cannot erase that residual.
Categorical-only scenes lack original annotation evidence and report it unknown.
Historical task gates keep their declared owned-target definitions; their large
500 mm³ floors do not qualify small synthetic targets for a new study.

`surgical_patient_task` prospectively freezes a separate source-sized engineering
task: at least 50% of the original target, with owned treatment credit in three
source-centroid RAS octants each contributing at least 5% of its original volume.
The whole-sequence 200 mm³ non-target cap, post-preparation 20 mm³ cap, complete
exit, 240 accepted actions, 4,000 physical calls and 240 seconds remain explicit.
The treatment baseline must extend the exact frozen prefix and precede target
removal; it cannot refund earlier tissue costs. Missing replay or computation
evidence fails its gates. This task is a prospectus/evaluator, not an autonomous
execution admission path. All five current patient tasks remain ineligible.

Patient role claims require a caller-supplied trusted catalog and an exact join
to a predecode TRAIN receipt, original observed-file SHA, native image pixels and
RAS affine. Case metadata cannot select a trusted catalog or grant a reserved
patient a new role. Generated controls are explicitly separate from patient
evidence. The current v1 task policy requires reviewed brainstem coverage even
if a supplied contract has an empty required-structures list. This is an
engineering admission policy, not a verified injury from the reported incident.
A future regional policy would need affirmative source landmarks, the complete
reachable action/effect domain, conservative uncertainty and independent review;
an empty label or a reviewed bounding box cannot establish regional exclusion.

`surgical_protection_transfer` now supplies one explicitly requested geometry
path for a verified full inferior source container. Exact affine-cell support
and an analytic enclosure cover declared continuous rigid stress sets; source
grid boundary/escape is separately unknown and protected. Known interior padding
requires an additional explicit exclusion. These assumed stress bounds do not
measure registration accuracy. The collection adapter still validates its exact
source snapshot, units, registration graph and anatomy review, retains unreviewed
status under research opt-in, and caps native cell allocation. Its default
import algorithm is unchanged. Three acquired-source guard controls verify both
hands' atomic native refusal and exact cold replay; no operative task follows.

Horizon 1 preserves the existing immediate-removal staged controller. The desktop
offers horizon 2 or 3 with beam width 3; the backend permits width 2–4 and at
most 24 work expansions per selection. Native checks and checked exits verify
every retained node. A reserved beam slot retains cavity-changing preparation
with zero immediate target removal. Final choices require positive target gain;
preparation alone never becomes successful treatment. Failed search means a
finite generator/controller or computation limit, not anatomical impossibility.
External radius changes and access/re-entry still use the existing outer schedule;
this milestone does not search their full joint configuration space.

The stock opening skills now provide a reusable bone-only macro: checked shallow
kerf, gradual active roll, active cutting return, then inactive withdrawal. Every
return cut and second contributes to the same ledger. A shaft-collision preview
may propose its actually observed shorter command, but that new command must
pass a fresh complete native preview and commit. Protected/material/domain
refusals are never shortened. Only a completed macro can be retained; stopped
internal prefixes remain unadmitted attempts. The declared proposal family checks
actual straight 0.12 mm waypoints, stock geometry and at most 24 mm depth.
A separate `cranialAccessV1` contract now binds an original source/target, full-
thickness F/K bone scope, entry rectangle/cone, retained-skull anchors, attachment
coverage and exterior workspace. The authoritative engine checks both tools and
every pose, including placement and persistent other-hand actions. Undeclared
entry, inner-table crossing before evacuation and invalid source evidence refuse
atomically. Posterior access remains case-specific, not universally forbidden.

A fixed generic `bone_flap_holder` uses the existing stock forceps envelope. An
exact component resolver requires complete kerf, no face/edge/corner bridge, all
retained skull supported and all declared supported attachments actually released.
The whole flap is swept against anatomy, protection, domains and both tools, with
no whole-flap shaft exemption. Translation uses snapped affine cells; rotation is
unsupported through every pose operation. Native extraction requires complete
exterior passage, then empty-holder withdrawal past the actual exit plane.
`craniotomy.openingComplete` cannot be inferred from removed slivers or slot flags.

`surgical_craniotomy_skills` freezes a separate bone-only policy against the
canonical native source, guard/contract, shared meter and original deadline. It
requires explicit whole-bone burden, zero added target/nonbone removal, complete
checked endpoints and full withdrawal. Only supported, declared flap attachments
may be released. Imported prefix costs need charged native ancestry. Refreezing,
resealed replacement limits or unlogged runtime state cannot reset the budget.

Kerf plus entire flap remain unique non-target source IDs. The original 200 mm³
`SkillLimits` and patient tumor-task caps are unchanged. The generated 285 mm³
control passes only its distinct prospectively declared bone-only policy; the old
200 mm³ policy refuses it. No patient window or full tumor/physics/RL admission is
opened by this mechanism. Patient attachment, source coverage and mechanical
evidence remain required before qualifying the requested patient workflow.

The mutually exclusive `cranialMillingV1` method declares the entire aperture A
and retained skull R separately from the flap contract. It requires reviewed
bone scope, inner boundary, entry/exit and supported retained skull. Fixed stock
drilling uses finite v2 work and the shared physical clock; original source IDs,
actual native ancestry and tagged instrument effects prove every removal. Full
bone removal leaves debris in the field. Only actual Frazier collection with
conserved generated/collected/present/escaped mass, followed by complete native
withdrawal, completes the opening. It never fabricates flap transport or grants
dural/CUSA entry. Unknown source evidence remains a refusal.

`surgical_milling_contact_proposals` supplies bounded, current-cavity stock
drill/suction poses and separate actual-debris suction programs. Its contact
inversion uses the existing exact orthogonal-cell kernel; shear refuses. Source
ordering and generation costs are reported separately, including source-wide
binding work. The fixed-attitude approach/withdrawal union is only a geometry
shortlist. Native staging `place` has endpoint/setup semantics, so its prior
repositioning requires separate caller verification. Every action, finite work,
debris collection, complete original aperture and exit still needs the existing
checked policy. Neither a nominal contact footprint nor a resealed caller proposal
grants admission, resets computation time or refunds a discarded native prefix.
An optional fixed center-plus-eight-quarter-cell point scheme addresses missed
corner contacts without changing tools or source A. The default remains centered.
Every individual inversion counts against a separate explicit 2,048-probe bound;
at most 32 matching suction inversions are additional. An off-center bracket
includes its affine anchor offset, including on long anisotropic cells.

The opt-in `axial_diverse_current_a` schedule chooses existing source cells at
several depths from each active current column, using the same contact inverse.
Immutable source geometry is separate from current membership and fresh collision
checks. Fully cleared columns are excluded before the candidate limit; missing
or duplicate depth slots within active columns still reserve their allotment.
All source preparation and current scans remain inside the original wall scope;
candidate support is never credited as actual removal. The default stays unchanged.

`surgical_joint_milling_opening` turns those proposals into a bounded adaptive
bone-opening skill on the original registered joint limits and meter. It replans
from actual remaining A and tagged debris for at most eight batches. Only the
first unposed setup may use `place`; default staging then uses checked native
withdrawal motions. An optional source-bound multileg plan connects an explicit
global parking pose to a local working pose using native moves. Full-body source,
protection, workspace and other-hand sweeps are checked, including intermediate
rotations. Local retreat is tested against the actual accepted cavity and never
counts as completed withdrawal. This mode keeps the drill until A is gone, then
withdraws globally before exchanging to suction and collecting actual debris.
Full original-A removal, conserved collected debris, the native bone-prefix proof
and exterior tools are required. Dura and tumor permission remain separate.
Final receipt construction, hashing and cancellation checks count against the
original deadline; physical completion cannot override a late refusal.

Joint search accepts a declarative `milling_opening` specification as one skill
expansion. All internal batches share the search policy, meter and goal. The
specification cannot inject policy objects or callbacks. Named pre-work staging
limits and local contact/batch failures permit other candidates to continue;
source, policy or unlogged-history failures remain invalid. This adds backend
inventory support, without automatic source-window generation or new desktop
Generate behavior.

The same checked macro can retain a genuine positive-time, single-endpoint
`BONE_CUTTING_IN_PROGRESS` prefix and repeat the identical stock command within
an explicit bound (default six continuations). Multiwaypoint partials and other
refusals do not receive this continuation. All partial costs remain in the ledger
and shared meter. Wall time starts at meter creation, including any earlier
prefix, discarded probes and idle time; freezing a downstream policy cannot
refund it. These bone-only controls do not change the patient tumor-task caps.

An optional, separately bound `cranialStagesContract` now adds native membrane
preparation after a genuinely completed opening. Its original B/P/W source sets
are disjoint: bone, nonbone membrane preparation, and tumor/brain work. Cutting
leaves the membrane occupied. The stock membrane holder requires the whole
boundary cut and actual attachment release, then moves the complete patch through
checked space and extracts it beyond the true exterior. Released patches remain
collidable. Retained rim/support and whole-tool guards remain enforced. Curved
categorical patches are supported within the declared stock scissor family;
continuum mechanics and clinical calibration remain unknown. Bone clearance alone
still grants no work entry, and standalone milling remains bone-only.

`surgical_joint_cranial_skills` freezes explicit B/P/W caps, action/call/deadline
bounds and one canonical source before any action. All preparation and work remain
in the same unique-source non-target total and shared sequence outcome. Native
history and charged branch ancestry are required for continuation; replacing or
resealing limits cannot refund a prefix. Finite CUSA contact continuations reuse
the existing bounded remainder/clearance generator. Interrupted work retains its
physical costs and remains unadmitted. An assumed post-scalp start explicitly
retains unknown prior exposure cost. These checked programs do not qualify a
patient, promote source roles, alter the old 200/20 policy, or establish planner
superiority; patient source evidence and complete task evaluation remain required.

`surgical_joint_cranial_search` adds a separate bounded search over source-bound
native opening, preparation, configuration and work programs. It retains several
work states and zero-target preparation, checks both-hand exits on separate
branches, and charges failed contacts and discarded probes to the original meter.
The frozen goal precedes every native call: at least half the full original target
and three fixed source-centroid octants each contributing 5%. Only source-owned
target cells earn treatment credit; physical removal of an original annotation
cell later owned by another material remains separately reported. Preparation
cannot shrink the denominator or refund source B/P/W burdens.

Pareto comparison retains separate B/P/W costs, modeled time, exchanges and
available shared mechanical, thermal and vascular consequences. Different evidence
availability is incomparable, never zero risk. The explicit recommendation order
is greater original-target treatment credit, then less whole non-target removal,
then less modeled time. Proposal names and animation duration alone do not create
distinct alternatives. Cancellation or the final readout crossing a bound prevents
plan admission. A local finite-contact controller failure can leave other search
branches available.

This API uses a caller-supplied skill inventory and one fixed source opening
contract; it does not generate another patient window. Horizon one is an immediate-
target ablation, not the strongest applicable patient baseline. The existing
desktop Generate path is not yet wired to this new search. Later export and
verification replay must use the same meter with prospectively reserved calls.
Generated opening/preparation/work results do not qualify patient tasks.

Replay adapters bind derived opening proofs to the exact returned checkpoint's
accepted ancestry. Rewinding drops a future proof; a genuine earlier bone proof
can remain through later membrane/work history. Current stage permission always
derives again from that boundary. Ordinary and persistent observed frames use
the same rule and must match canonical endpoint/receipt identity. The optional
bounded-prefix replay API explicitly refuses this external stage envelope;
default canonical recorded-state replay remains supported.

`TransitionMeter` charges previews, discarded probes, checked exits, commits and
replay. Those are computation costs, separate from accepted patient treatment.
The source-geometry cache reuses only immutable label counts and target
coordinates/octants. Dynamic removals, tools, constraints and outcomes are
remeasured. Scene/source/transform edits invalidate it. Tests compare cached and
uncached results and exercise cavity edits. Native execution remains the main
planning expense; no end-to-end speedup is claimed from the geometry cache.
A duplicate unchanged boundary preview can refuse an already proven removal-limit
violation without repeating the same unsafe query. This is refusal-only; successful
actions and any shortened command still require fresh native checking.

Static protection queries now reuse immutable source IDs/points and an opaque
registered spatial index. Buffer owners, pointers, layouts, source geometry and
index metadata bind the cache; exposed inspection arrays cannot edit the operative
tree. Trusted class dispatch prevents instance helper/query shadows from bypassing
validation. Static queries avoid repeated whole-point finite scans and full index
allocation; moved protected tissue still receives full current-position validation.
Source/guard contracts, physical margins and capsule-distance results are unchanged.
Independent exhaustive scans, mutation controls and native replay check this scope.

One matched instrumented million-cell control reduced native execution from
1.054638 to 0.693960 s; initialization plus control fell 12.5%, with identical
state/actions/calls/quantities. This is one generated control, not patient-planner
throughput or completed treatment evidence. The complete cost and retained draft
failures are indexed in the progress handoff.

The current generated fixed-radius regression demonstrates clearance followed by
target work, exact replay and withdrawal. It is a mechanism check, not patient
performance or superiority over the strongest full staged controller, which can
solve that coupon using a larger-tip pivot. The five-case baseline, original
incident search, matched performance evidence and remaining work are indexed in
`docs/decision-system-progress.md` and the ignored local evidence directory.
