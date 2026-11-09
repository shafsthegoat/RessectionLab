# Decision-system progress and next dependency

Branch: `shaf/decision-system-v1`; baseline `53c49019`. Primary checkout only.
The patient-resection goal remains open. No clinical validation is claimed.
Ignored evidence is under `testcases/decision-system-v1/`; bulk patient files
remain on the external drive. `milestone-results.json` indexes versioned results.

## Implemented

- Source protection survives target overlap. The authoritative engine checks both
  instruments, full swept geometry, withdrawal and supported indirect effects;
  forbidden actions roll back the whole proposal. Reviewed domains also bind
  direct actions and checked skills. Rewards/search cannot override refusals.
- Shared accounting preserves original-target residuals, unique cumulative tissue,
  preparation, exchanges, modeled time, exit and unknown evidence. Prospective
  patient tasks freeze 50% original-target progress in three spatial zones,
  whole 200/work 20 mm³ non-target caps and compute limits. Trusted predecode source
  receipts and exact originals/affines bind roles; metadata cannot grant TRAIN.
- Bounded two/three-skill lookahead retains delayed-benefit preparation and checked
  exits. Finite CUSA work clears at the actual stop and resumes only unreached
  path legs, with the same default eight-continuation budget for immediate and
  lookahead policies. Incomplete skills retain costs without treatment admission.
  Stock v3 uses fixed 1.5 mm geometry; adjustable legacy radii stay labeled legacy.
  A refusal-only optimization removes an identical unsafe boundary query; it never
  reuses a preview to admit changed commands or alters physics/tissue limits.
- The desktop uses one Case → Plan → Compare → Rehearse/edit workflow and draft.
  Edits/reverts stale results. Native cancellation, cards, checkpoint ledger,
  recorded playback and Save/Open work. Zero-target preparation displays only
  with its validated bounded-lookahead receipt; it remains zero treatment credit.

The user clarified that the unlocated incident concerned an implausible entry
route, possibly posterior/inferior, rather than an established brainstem injury.
The new requested scope includes actual skull resection and removal of the flap.

## Full bone-opening mechanism

`cranialAccessV1` now binds source/target, original entry rectangle/cone, intended
flap/kerf IDs, retained skull/anchors, attachment coverage and exterior workspace.
The native gate checks every pose and both hands, including placement, parked
instruments and original crop/model gaps. Before evacuation, tools cannot pass
beyond the exact affine inner table. Unknown coverage remains blocked; a posterior
approach is case-specific rather than globally forbidden.

A fixed generic bone holder resolves the exact full component, including pieces
larger than display summaries. Complete kerf, no face/edge/corner bridge, supported
retained skull and actual release of declared attachments precede grasp. Whole
flap sweeps check anatomy, protection, domains, both tools and coupled updates.
There is no whole-flap shaft exemption. All held pose operations share translation
and rotation guards; the disclosed surrogate uses snapped affine cells and
supports translation only. Released pieces remain in field until actually removed.
Extraction requires the whole flap at its declared exterior; empty-holder
withdrawal must clear the actual exit face, independently of slot flags.

The checked macro freezes an explicit **separate bone-only** budget, native source,
shared meter and deadline. Whole kerf plus flap are charged once. Supported flap
attachments may be released; added target/nonbone removal remains zero. Imported
prefixes need charged native ancestry. Refreezing, resealed limit substitution and
unlogged runtime state cannot reset costs. The old200 mm³ tumor/access caps remain
unchanged. Existing patient, physiological and RL qualifications remain separate.

A generated closed-shell control executes 18 native actions: stock cutting,
periosteal release, grasp, lift, full evacuation and withdrawal. It removes
236 mm³ kerf plus 49 mm³ flap, zero target/brain. The 285 mm³ control passes only its
prospective bone-only policy. Old `SkillLimits(200)` refuses after 199 mm³ accepted
bone when the next preview reaches 236; a 260 mm³ bone policy refuses the entire
49 mm³ flap extraction. These are mechanism controls, not patient completions.

The final bounded generated run costs 104 actual native evaluations across macro,
fresh replay, lifted checkpoint and lifted/final cold reopen, 7.109 seconds and
152.7 MB peak RSS. Its original archive is byte-exact. The prior successful run
is preserved; the repeat was justified by source/budget integrity fixes, with
identical geometry/actions/bounds. Development tests/probes and desktop replay are
additional; no matched speedup or end-to-end patient performance is claimed.
Evidence: `craniotomy-v1/desktop-control-v1/`.

The separate `cranialMillingV1` method now covers full-aperture removal without
claiming a freely detached flap. Source-bound A/R partitions, anchors, local
inner boundary, original entry and full-tool workspace remain hard gates.
Fixed stock drilling uses finite v2 cutting work and shared physical time. The
native ledger keeps all milled bone as debris until actual stock Frazier suction
collects it; full mass conservation and real tool withdrawal are mandatory.
Bone clearance never authorizes dura or intracranial treatment. Legacy flap
snapshots/events retain their original semantics.

The existing checked macro handles only genuine positive-time, single-endpoint
bone-capacity partials, repeating the unchanged command at most six times by
default. Every prefix, continuation, preview and replay remains charged. Its
wall budget now begins at shared-meter creation, closing a prefix-time refund
found by independent review. The old 200 mm³ patient policy remains unchanged.
Generated controls exercise the entire 81 mm³ aperture, actual 81 mm³ debris
collection, zero escaped debris, zero target/brain and real withdrawal; these
remain mechanism evidence. See `craniotomy-v2/` for frozen sources and controls.
The sealed run completes 22 actual actions, including one genuine cutting-work
partial, in 98 native evaluations across macro, exact fresh replay and both
debris-retained/final cold restorations: 20.548 seconds and 169.4 MB peak RSS.
Actual statuses, reasons and state identities replay exactly. Source-only work
was co-running; these measured bounds make no patient speedup claim.

## Patient results and retained failures

All five unchanged TopCoW TRAIN originals 004/006/012/014/018 remain reported.
Independent review checked source planes, units/affines, target recipes, roles
and native replay (`case-qualification/`, `patient-tasks-intact-v3/`). None completes
a meaningful full tumor task. Their five proposed calvarial windows remain unset:
plausible local bone is not attachment, full-aperture or exit qualification.

The strongest retained CTA014 baseline removes15.810416 mm³ target (5.8524%),
272.901532 mm³ cumulative non-target, 280.04 modeled seconds and fully withdraws;
it exceeds the current 200 mm³ cap. Its newer nine-action sliver opening removes
13.977324 mm³ skull, zero brain/target/vessel, with verified exit and cold replay.
The round costs 37 calls/40.40 seconds/1.59 GB; including reused inventory,
61 calls/52.29 seconds. A broader 47-expansion study removes46.972976 mm³ skull
but leaves 201 route-bone cells. Slivers are not a complete craniotomy or treatment
(`patient-search-v1/`).

Authentic CTA014 prefix 44 retains 187.891903 mm³ prior non-target. Both old and
finite-contact policies refuse the next recorded action: preview 222.262373 mm³
total and 34.37047 mm³ new brain. Checked withdrawal and fresh exact 45-action replay
complete within 93 calls/57.529 seconds/1.174 GB. The earlier 94-call wall-budget
failure during replay remains explicitly incomplete. No target is removed
(`contact-continuation-v1/patient-probe/`).

The finite lateral kerf family misses three remaining route-bone cells. A separate
6912-axis source screen per 014/012 finds no cap-compatible meaningful continuation;
cheapest complete geometric paths cost 237.385/244.330 mm³ non-target for 20.9%/10.3%
target. No further native round followed. These are generator limits, not proofs
of anatomical impossibility (`access-family-v2/`).

CT intensity/units/windowing checks found no correction. A justified official-
default SynthSeg pilot was still withheld by independent source review; no mask
was promoted. Missing acquired CTA coverage is not established. Existing structural
MRI alternatives are skull-stripped and lack the full skull/vascular/attachment
package (`task-repair-v2/`; older failures retained in `task-repair-v1/`).

A source-only full-aperture screen now accounts for every supplied bone cell in
each fixed 8 mm disk: 2076/1414/2051/1404/1368 mm³ for 004/006/012/014/018. All
exceed the unchanged 200 mm³ task cap. A constant deepest-column drill envelope
also touches known brain, so categorical bone IDs cannot supply an inner-table
recipe. A frozen 504-record cone screen finds two literal CTA014 full-shaft FOV
fits (only 0.020/0.013 mm margins), neither satisfying conservative sweep and
fixed-aperture bounds. Other sampled fits fail; this finite screen does not prove
continuous anatomical impossibility or that new imaging is required.

Retained meningioma sub-06/sub-13 original CT arrays support 16/33 and 10/33
predeclared upper full-stock-tool pose families inside the released CT grid.
They remain previously exposed development sources, with unqualified target,
critical/dural evidence and registration. Original headers have the same physical
FOV as stored grids; rereading originals does not recover a wider acquisition.
Existing privacy removal remains intact. Header-only official sub-01/02/03 probes
were prospectively sealed as unreserved development candidates, not official
TRAIN: nine released 0.5 mm CT/MR headers, 599,778 transferred bytes, 5.42 seconds,
60 MB RSS and no pixels/native action. Header geometry alone qualifies no case.
See `craniotomy-v2/{source-milling-review,cone-workspace-review,retained-ct-review,
header-source-probe}/` for identities, costs and exact limits.

The predeclared lexical sub-01 pilot then acquired only its three released
CT-BRAIN/CT-BONE/CE-T1W members: 212,819,983 response bytes, 209.05 seconds,
55.8 MB RSS. Full ZIP-member/gzip CRCs, lengths, original header hashes and
observed SHA256s pass; publisher whole-archive MD5 remains unverified. Source-plane
review preserves the existing privacy cut. MRI k0..8 and325..380 are all zero;
header extent or padding alone cannot establish observed exterior workspace.

Dense source MRI review now supports broad inferior-compartment exclusions as a
concrete annotation repair. Sub-01's separately preserved MRI-native container
covers full x/y and k0..240, including extra brain, air and padding; its upper
face is −426.75 mm RAS-z, with an explicitly assumed 8 mm allowance above the
reviewed upper band. Independent source/overlay and full-mask checks confirm
exact recipe/affine/hashes. This is an engineering containment region, not a
precise brainstem segmentation or anatomical volume. It does not qualify a CT
transform, patient task, other critical anatomy or workspace. Equivalent retained
sub-06/sub-13 containers cover full x/y and k0..226, upper faces −441.25/−443.0 mm,
with the same explicit 8 mm allowance. Root independently inspected all dense
packets/actual overlays and streamed all three full masks to verify every included
and excluded plane, original affine/units and source hashes. No native patient
action or original-case change followed (`craniotomy-v2/sub01-source-pilot/`,
`craniotomy-v2/retained-source-qualification-v1/`).

## Source-bound exclusion transfer

The three MRI-native inferior containers now have separate CT-bound derivatives.
Paired source review supports coded identity as conditional engineering geometry;
the old06 bone transform visibly misaligns calvarial context and remains preserved.
No registration is promoted to clinical truth. Assumed stresses are ±5 mm per
RAS axis plus arbitrary-axis3° for01 (enclosed by an L2 radius√3×5 mm), and a
4 mm L2 translation ball with no rotation for06/13. The separate8 mm source
containment allowance is not a measured registration error.

`surgical_protection_transfer` verifies every source cell of the declared slab
and transfers complete affine cells with an analytic continuous rigid-stress
enclosure. It avoids the old tens-of-millions-point source cloud. Source-grid
escape/contact remains unknown and protected. Exact source-zero padding bands
are separately transferred and unioned as unknown; they are never exterior air.
The collection importer opts in explicitly, keeps all original review gates and
unreviewed statuses, and caps native cell allocation before tuple construction.
Its default circumsphere route remains unchanged.

Root checked all six full decoded native/coarse masks, exact OR/unknown subsets,
all cells against an independent matched-axis reference for06/13, and7,820
in-domain forward stress samples per01 grid (9,000 attempted;1,180 outside the
target grid). All seven final source-overlay sheets were inspected. The check
cost11.01 seconds /1.777 GB RSS. A first audit stopped on an arbitrary>8,000
in-domain count assertion; identical samples and its failure remain preserved.
Independent geometry review passes22 controls after correcting a numerical
source-boundary underexclusion; the original21/22 result is retained.

Actual acquired-source native controls now bind the final2 mm exclusions for
01/06/13:1,102,248 /884,780 /706,768 protected cells. At a predeclared empty-material
point inside each exclusion, both stock-tool hands refuse preview and direct
placement with `PROTECTED_STRUCTURE`, atomically preserving the accepted prefix.
Real CaseData Save/Open and fresh/cold native replay preserve exact states,
source/guard identity and original files. Each run costs8 all-phase evaluations;
driver wall15.98 /7.41 /5.92 seconds, peak RSS0.991 /0.859 /0.753 GB. Only configure
is accepted; no physical work, target creation or patient-task admission occurs.
These controls establish source guard binding, not a usable operative window.
Source/model roles, the06 original45,035 mm³ target,13 empty target and all five
original tasks remain unchanged. Evidence: `craniotomy-v3/`.

## Validation and next dependency

The exclusion-transfer milestone passes286 selected backend tests in132.69
seconds, including actual collection import/native refusal and unchanged legacy
protection, patient/domain, milling/flap and checked macro behavior. Independent
geometry checks use1,944 LP intersection tests on oblique/sheared cells. No
frontend changed or additional desktop claim is made by this milestone.

Current backend: 460 distinct selected tests pass across the broad run and focused
compatibility rerun, covering milling, flap, finite-contact/staged work, protection,
domain/material/clock/RL guards and persistence. The initial broad run retained two
failures from the holder catalog extension: an obsolete fifteen-tool test and
historical CUSA-only report reopening. The focused 78-test rerun passes after a
narrow compatibility repair: only the complete preceding catalog without the new
holder is accepted, with all existing tools, physics and clock unchanged. Unknown
catalog hashes and changed CUSA geometry still refuse. New declarations retain
the full current catalog; historical reports acquire no public-plan qualification.

Independent milling review passes 16 native/adversarial controls and 82 checked
macro controls, including actual finite-work prefixes, both hands, mass/exit,
budget ancestry, wall-origin and injury-seal checks. Prior independent flap
coverage and retained failures remain in `craniotomy-v1/`. No patient window is
qualified by these generated tests.

All 1294 desktop tests pass and production build passes. Prior hidden warm/cold
flap replay verifies cutting, attachment release, 49 moved cells at action 16,
evacuation 17, actual withdrawal 18, full 285 mm³ ledger, playback and Save/Open.
Windows remain hidden and originals unchanged. See `craniotomy-v1/` for final
renderer/backend hashes and screenshot sequence. Earlier multiple-patient workflow
checks, cancellation, edit staleness and actual paired-source staged Generate are
retained with their implementation hashes in `contact-continuation-v1/desktop/`;
the three legacy 0007 branches remain unpassed, not completed patient tasks.

Current hidden milling warm-v2 and cold-v2 pass all six actual checkpoints
(0/3/10/13/21/22), returned cutting-partial history, debris quantities, full ledger,
play/pause, Save and fresh reopen. Both remain hidden, with stable backend/source
hashes and renderer `71653cf6…`. Root inspected the actual suction/final captures.
The first harness attempt failed before browser creation on a file-inventory
typo and left an Electron process alive; it was terminated. Early fatal handlers
now suppress native error boxes and exit immediately. A forced startup failure
exits1 in0.532 seconds with zero windows/dialog attempts and no owned process left;
positive cold-v2 passes after this fix. The earlier error-dialog visibility was
not measured. Logs, reproduction commands and screenshots are indexed in
`craniotomy-v2/desktop-control-v1/`.

Next: qualify a patient full-thickness aperture, local inner boundary, retained
support and complete stock-tool workspace. Flap removal additionally needs actual
attachment/release and whole-fragment exit evidence; milling instead needs finite
cutting and actual debris collection. Neither supplies critical-anatomy, target
registration or dural-entry evidence. Freeze the complete bone/brain/target/burden
goals before joint opening/work search. The new conditional source exclusion
transfers supply conservative geometry, not complete anatomical or workspace
qualification. Do not substitute an idealized control
for patient evidence. Old boxes exclude 35.32% of CTA012's
original target and 9.58% of CTA014's; all five targets plus fixed 10 mm buffers fit
acquired CTA FOV. New source-context review is distinct from absent imaging or
anatomical clearance; preserve old benchmarks/bounds rather than expanding them
to admit a failed route. See `craniotomy-v1/access-window-review/`.

An interface audit also found that current dural scissors create a seam while
leaving the membrane occupied. This cannot authorize CUSA passage. A source-bound
cut-and-transport/extract membrane mechanism and separately versioned joint stage
authority are required; bone clearance alone must continue refusing deeper entry.
Joint B/P/W budgets must count all bone, nonbone preparation and brain-work/exit
burdens without changing or refunding the old200/20 policy. A conditional
post-scalp start must be explicit; an intact-head result needs actual scalp work.
See `craniotomy-v3/stage-integration-audit-v1/` for existing native hooks.

Joint patient opening/work search, meaningful completed tasks on three suitable
patients, uncertainty and matched material patient improvement remain open.
Correct STOP, preparation or a generated bone-only pass do not close this goal.

```sh
.venv/bin/python -m pytest -q tests/test_surgical_craniotomy.py \
 tests/test_surgical_bone_transport.py tests/test_surgical_craniotomy_skills.py \
 tests/test_surgical_milling.py tests/test_surgical_milling_native.py
pnpm --dir desktop test
pnpm --dir desktop build
```
