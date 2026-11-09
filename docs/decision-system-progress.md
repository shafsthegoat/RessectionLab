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

## Validation and next dependency

Current backend: 421 selected tests pass, including new native controls, prior
finite-contact/staged work, protection/domain/material/clock/RL guards, physics,
archive and replay. Independent review reproduces 42 contract+17 native tests and
adversarial entry, full-flap, all-held-operation, old200 cap and persistence checks.
It exposed and rechecked source-hash recursion, false withdrawal, non-move rotation,
inner-table, budget resealing and unlogged-prefix errors; failures are retained.

All 1293 desktop tests pass and production build passes. Hidden warm/cold generated
bone replay verifies cutting, attachment release, 49 moved cells at action 16,
evacuation 17, actual withdrawal 18, full 285 mm³ ledger, playback and Save/Open.
Windows remain hidden and originals unchanged. See `craniotomy-v1/` for final
renderer/backend hashes and screenshot sequence. Earlier multiple-patient workflow
checks, cancellation, edit staleness and actual paired-source staged Generate are
retained with their implementation hashes in `contact-continuation-v1/desktop/`;
the three legacy 0007 branches remain unpassed, not completed patient tasks.

Next: source-review a patient cranial window/full-thickness contour, attachments,
retained-skull support, full-tool workspace and flap exit; freeze its complete
bone/brain/target/burden goals before joint opening/work search. Do not substitute
an idealized control for patient evidence. Old boxes exclude 35.32% of CTA012's
original target and 9.58% of CTA014's; all five targets plus fixed 10 mm buffers fit
acquired CTA FOV. New source-context review is distinct from absent imaging or
anatomical clearance; preserve old benchmarks/bounds rather than expanding them
to admit a failed route. See `craniotomy-v1/access-window-review/`.

Joint patient opening/work search, meaningful completed tasks on three suitable
patients, uncertainty and matched material patient improvement remain open.
Correct STOP, preparation or a generated bone-only pass do not close this goal.

```sh
.venv/bin/python -m pytest -q tests/test_surgical_craniotomy.py \
 tests/test_surgical_bone_transport.py tests/test_surgical_craniotomy_skills.py
pnpm --dir desktop test
pnpm --dir desktop build
```
