# Decision-system progress and next dependency

Branch: `shaf/decision-system-v1`; baseline `53c49019`. Existing primary checkout.
The patient-resection goal remains open; no clinical validation is claimed.
Local evidence and full reproduction commands are under ignored
`testcases/decision-system-v1/`; bulk patient files stay on the external drive.

## Implemented and verified

- Authoritative protected/critical source masks survive target overlap. Both tools,
  active effects, shaft, continuous instrument sweep, withdrawal and supported
  indirect effects are checked transactionally. Reviewed task domains also bind
  direct actions and checked skills; search/reward cannot override a refusal.
- Shared sequence accounting preserves original-target residuals, unique cumulative
  tissue removal, preparation, exchanges, modeled time, exits and unknown evidence.
  Source-sized prospective tasks freeze 50% original-target progress plus three
  spatial zones, whole200/work20 mm³ non-target caps and computation limits. Trusted
  predecode receipts, originals and RAS affines bind patient roles; editable case
  metadata cannot convert a reserved patient into TRAIN.
- Bounded two/three-skill lookahead retains delayed-benefit preparation and native
  checked exits. Finite CUSA contact can now clear at the actual stop and resume
  only unreached path legs, up to eight times by default. Immediate and lookahead
  policies get the same capability; explicit zero reproduces single-motion work.
  Partial costs persist, but incomplete skills cannot be selected as treatment.
  Actual stock v3 uses fixed1.5mm geometry; legacy radii stay explicitly legacy.
- A refusal-only shortcut avoids repeating an identical native boundary query once
  its removal-limit violation is certain. It grants no admission and never reuses
  a preview to authorize a changed/clipped command. The unchanged native physics,
  source protection, tissue caps and replay checks remain authoritative.
- Stock opening macros perform shallow kerf, gradual roll, active cutting return
  and checked withdrawal. These remove slivers, **not a complete bone flap**.
- The desktop uses one Case → Plan → Compare → Rehearse/edit workflow and one draft.
  Edits/reverts stale comparisons; recorded results retain source/branch identity.
  Native cancellation, comparison cards, checkpoint ledger, playback and cold
  reopen work. The parser now accepts a zero-target preparation stage only with
  its validated bounded-lookahead receipt; legacy/malformed zero stages still fail.

## Patient results and retained failures

All five TopCoW TRAIN originals004/006/012/014/018 remain unchanged and reported.
Their source planes, units/affines, target recipes, source roles and native replay
were independently checked (`case-qualification/`, `patient-tasks-intact-v3/`).
All five full tumor tasks remain ineligible for missing reviewed critical coverage;
none is counted as completed. The original incident replay was not located. The
user clarified that the concern was an absurd entry route, possibly posterior or
inferior; the affected case/structure remains unverified. New access restrictions
must be case-specific and authoritative, not a global ban on posterior approaches.

The strongest retained CTA014 baseline removes15.810416mm³ target(5.8524%),
272.901532mm³ cumulative non-target,280.04modeledseconds and fully withdraws.
It exceeds the current200mm³ cap. The promoted nine-action opening removes
13.977324mm³ skull and zero brain/target/vessel with complete exit; fresh public
cold replay and hidden desktop reopen agree. Its round costs37nativecalls,
40.40seconds and1.59GB peak; including reused inventory61calls/52.29seconds.
The broader47-expansion study removes46.972976mm³ skull but leaves201 route-bone
cells. Neither result completes meaningful tumor treatment (`patient-search-v1/`).

Finite-contact testing on authentic CTA014 prefix44 retains187.891903mm³ prior
non-target. The next recorded30second action is refused by both policy0 and8:
its unchanged preview reaches222.262373mm³ total and34.37047mm³ new brain.
The checked withdrawal then completes; no target is removed. The final bounded
negative component proof uses93calls,57.529driverseconds and1.174GB peak, including
fresh exact45action replay. The earlier94call attempt exhausted its wall budget
during replay and is preserved as incomplete. Only the refusal-query optimization
justified the second attempt; no matched speedup or positive patient claim follows
(`contact-continuation-v1/patient-probe/`).

Corrected fixed-axis source geometry shows the finite kerf family cannot cover
three remaining route-bone cells. A new6912-axis screen per case014/012 also finds
no cap-compatible meaningful continuation; its cheapest complete geometric paths
cost237.385/244.330mm³ non-target with20.9%/10.3% target. No new native round was
justified. These are generator screens, not anatomical-impossibility proofs
(`patient-search-v1/lateral-opening-v1/`, `access-family-v2/`).

CT intensity, units, normalization and source crop were checked without finding a
repair. A separately justified official-default SynthSeg pilot was still withheld
by independent source-plane review; no mask was promoted. Current CTA coverage
insufficiency is not established. Conservative named-structure coverage remains
unresolved. Existing structural MRI alternatives are skull-stripped and lack the
full skull/vascular/attachment package, so none was promoted as a substitute
(`task-repair-v2/`; prior attempts remain in `task-repair-v1/`).

## Current validation

The current finite-contact/backend suite passes152tests; independent review
passes77overlapping checks plus a real-native post-preview cancellation control.
Generated fixed-v3, finite-capacity, protection, caps, remaining-waypoint and
report-tamper controls are software tests, not patient successes. Earlier safety
before/after and delayed-preparation regressions remain independently reproduced.

All1290desktoptests pass(325main/693renderer/272viewer), as does production build.
Final hidden warm/cold workflows pass on0007and the0005no-route case, including
cancellation, edits, native preview/install, missing-evidence refusal, ledger,
playback, checkpoint return and Save/Open. Actual default-depth staged Generate
on the original paired DEVELOPMENT0007source yields three replay-exact, withdrawn
but unpassed branches(41/44/81mm³ target;135/119/135mm³ non-target), each honestly
labeled legacy. The run costs2589calls/67.63plannerseconds; upfront validation is
additional. An initial successful backend result exposed the zero-preparation UI
bug; the failed display and its2589calls remain charged. A justified rerun and
cold reopen pass on the final renderer. Historical CTA014sliver playback also
passes warm/cold. All windows remained hidden and source archives byte-exact.
See `contact-continuation-v1/desktop/qa-summary.json` for exact hashes, costs,
screenshot sequence and retained failures; `milestone-results.json` indexes all.

## Next substantive dependency

Implement the requested **full skull opening and bone removal**: a source/target-
bound cranial window; complete kerf and attachment state; exact flap component;
whole-flap swept protection/domain/both-tool checks; exterior evacuation; empty-
holder withdrawal; and one unique kerf-plus-flap ledger. Start with declared closed-
shell engineering controls, then require a reviewed patient evidence package.
Read-only design and all five source-bound access proposals are retained in
`craniotomy-v1/`. Every proposed window is still unset. Local calvarial columns
are plausible, but attachments, retained-skull support, complete cut, whole-flap
exit and the intended full aperture are unproven. Full flap volume stays charged;
any separate bone-only task must freeze its own burden before testing.

The old reviewed boxes exclude35.32%of CTA012's target and9.58%of CTA014's target.
All five complete targets plus fixed10mm buffers fit in acquired CTA FOV. Source
review for a new domain is distinct from absent imaging or anatomical clearance;
old bounds are preserved. Do not enlarge a box merely to admit a failed route.

Joint opening/work search, source-qualified full tasks on at least three suitable
patients, uncertainty and matched material patient improvement remain open.
Preserve all five original cases and old benchmarks, unknown anatomy, actual tool
geometry and physics/RL holds. Correct STOP and preparation alone are not treatment.

Reproduce focused current checks:

```sh
.venv/bin/python -m pytest -q tests/test_surgical_contact_continuation.py \
 tests/test_surgical_staged_planning.py tests/test_surgical_sequence_search.py \
 tests/test_surgical_staged_skills.py tests/test_surgical_staged_exit.py \
 tests/test_surgical_patient_task.py tests/test_surgical_patient_role.py
pnpm --dir desktop test
pnpm --dir desktop build
WORKFLOW_OUTPUT="$PWD/testcases/decision-system-v1/NEW-hidden-workflow" \
 desktop/node_modules/.bin/electron desktop/electron/verify-surgical-workflow.cjs
```
