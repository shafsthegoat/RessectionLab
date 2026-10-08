# Real-observation execution ledger

Updated October 8, 2026. The [October 6 supergoal](SUPERGOAL_REAL_OBSERVATIONS.md)
defines the full surgical planning/rehearsal objective, subject to the later
human instructions below. Its [source receipt](../manifests/real_observation_supergoal_source.json)
binds the preserved original text; that historical copy is not rewritten.
This is a compact execution record, not a replacement specification.

## Active human steering, October 8

The user explicitly superseded the earlier real-data-only training restriction:
synthetic data and simulator-generated experience are permitted for RL development
and training where useful. Keep generated experience separate from observed patient
records, record simulator assumptions and model lineage, and evaluate transfer on
held-out real patients and appropriate physical measurements. Generated outcomes
are never clinical evidence. Historical experiments and failed validation remain
unchanged; permission to train does not establish physical fidelity or patient benefit.

Continue eligible acquisition using verified TLS, resumable transfers and checksums.
Finish the queued files continuously, without per-batch review or bookkeeping pauses.
Detailed image, geometry and anatomical reviews form a separate phase. Downloaded
but unreviewed files must remain explicitly unreviewed and cannot satisfy intended-use
QC or spatial admission. Preserve patient splits, source rights and original records.

Parallel workstreams cover acquisition, QC preparation, simulator/RL development,
integration and independent evaluation. Coordinate shared source edits without
interrupting downloads. Compare search, imitation and RL under matched conditions,
retain negative results, and prioritize working end-to-end capabilities and measured
improvements. The original full goal remains open, including mechanical validation.

A [scoped generated opening-task learner](native-opening-learning.md) now admits
explicitly declared simulator experience to three spatial learning APIs. Its
[prepared comparison](../artifacts/native-opening-learning-preparation-v1/RESULT.md)
passes focused and independent controls; an accepted single-episode/gradient profile
confirmed execution before the fixed study. Historical patient/model-loading guards
remain. Old refusal results preserve the earlier behavior; they do not prohibit all
future generated training under the user's later instruction.

For all new commits use `skamal23 <sayemkamal12@gmail.com>` as author and include
`Co-authored-by: shafsthegoat <shafrir.p@gmail.com>`. Preserve existing history.
The user's hourly push request is scheduled as the active thread heartbeat
`push-ressectionlab-progress`: fetch first, push completed reviewed commits normally
to `origin/main`, verify the remote, and preserve unfinished work and running transfers.
No force push, paid infrastructure, billing change or external outreach is authorized.

## Latest generated-learning checkpoint

The [fixed native opening pilot](../artifacts/native-opening-learning-v1/RESULT.md)
completed 16 BC and 16 scratch RL updates from identical initial weights. Search
scored 1.1, BC stopped at zero, and final RL scored −0.896 versus initial −0.464.
Independent saved-output review verified all 278 indexed outputs, actual tensor
changes, all nine comparison arms and 78 accepted simulated histories. The run
took 8.6398 seconds overall with 332,644,352 bytes sampled peak RSS. There were
no retries or cap changes. Positive experiences occurred during training, so
their complete absence cannot explain the failed learning. The fixed 256-update
imitation diagnostic subsequently fits all five labels and matches search at 1.1,
while reproducing the original update16 exactly. Saved-tree arithmetic shows a
small RL16 expected-return gain despite lower target-reaching probability. No
patient or held-out record was used. The [same-model RL capacity run](../artifacts/native-opening-rl-capacity-v1/RESULT.md)
has now completed: 256 updates, 1,024 independently audited training histories,
and exact first-16 reproduction. Final argmax return is 1.098 versus search's
1.100; expected stochastic return is +0.546023. The run took 93.6124 seconds;
geometry and auditing dominate its cost. This supports fixed-task learning only.
Next is a narrow frozen-forward compatibility check on declared TRAIN anatomy,
with no action execution, patient fitting or generalization claim.

## Latest acquisition checkpoint

All [Lausanne TRAIN originals](../artifacts/lausanne-train-acquisition-complete-v1/RESULT.md)
are acquired and byte-verified: 199 people, 210 sessions, 840 files and
10,020,802,851 bytes. The continuous runner finished the final 52 files without
batch review pauses. Separate QC now covers all newly downloaded images: 414/420 original images pass
with six historical T1 conflicts retained. Mask-grid checks pass for 136/144; eight
remain unresolved. No fitting, anatomical approval or spatial admission is implied.

The [full eligible Lausanne annotation intake](../artifacts/lausanne-annotation-full-intake-v1/RESULT.md)
completed all 144 masks from 106 people/117 sessions. Independent content checks
verified all 13,977,015 compressed bytes and 1,134,965 positive voxels. At the
historical processing checkpoint, reference-grid checks passed for 116 masks;
27 awaited references and one retained an unresolved grid. Four metadata exclusions
remain. This completed review does not gate continuous original-image downloads.
Anatomical review, spatial admission and fitting contributions remain unestablished.

The [remaining RESECT TRAIN acquisition](../artifacts/resect-exact-mirror-acquisition-v1/IMPORT_RESULT.md)
is complete: 25 image–mask pairs from 13 people, 50 files and 557,790,240 bytes.
The 24 final masks used an exact-checksum third-party mirror and a locked offline
import; original OSF failures remain. All 890 files in the frozen continuous queue
are now present. Twenty-four RESECT pairs await separate image/label/grid QC;
the previous Case3 QC and all split, missing-label and rights constraints remain.

The additional [104-file original-source queue](../artifacts/resect-train-originals-preparation-v1/RESULT.md)
is committed and running continuously from October 8 at 19:26 UTC. It adds 28 MRI,
17 ultrasound and 59 correspondence files (672,336,857 bytes) from the same 14 TRAIN
people. Published MD5/size and measured local SHA256 are required; no decoding or
admission occurs during transfer. Forty-three owner and seven independent controls
passed. The [N36/S120 solve](../artifacts/hbe-n36-temporal-preparation-v1/EXECUTION.md)
is also running after verified archive/preflight and 13.3256-second preparation;
no result is claimed until terminal publication and independent review.

## Evidence and decisions before the October 8 steering

- Existing planning/ML comparisons remain historical. Search exceeded frozen
  imitation on all four completed development cases; two cases were blocked.
  Those policies learned generated transitions/labels, so neither their weights
  nor their trajectories qualify for the current learning/evaluation pipeline.
- Raw acquired images remain useful. No eligible recorded surgical transition
  cohort has been acquired, and no compliant offline-RL updates have run.
- SynthStrip v1 main/no-CSF weights are known synthetic-trained. Their masks
  cannot regain eligibility through cached files or envelope review. The
  [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9465771/) and locally
  retained source metadata establish this lineage.
- Current exclusion guards stop selected legacy learners, checkpoint loaders,
  archived-runtime diagnostics, synthetic generators and SynthStrip paths. They
  also reject known/unknown model ancestry at brain-support consumption.
  Historical JSON and raw-image inspection remain accessible.
- This is a **bounded exclusion layer**, not complete evidence admission or a
  security boundary. Arbitrary imported NIfTI provenance, archived executables,
  synthetic bundles and all downstream derivations still need a canonical
  source/ancestry admission contract. Do not certify the whole app as compliant.
- Preserve BTC TRAIN 05/16/20/22/25/28, SELECT 26/27, transfer 29/31, UPenn external
  and the UTSW structural-validation role. No inspected record can become an
  untouched evaluation record retroactively.

## Workstreams and next executable steps

| Workstream | Verified state / missing evidence | Next implementation or acquisition |
|---|---|---|
| Data and anatomy | All 840 Lausanne TRAIN original files and 144 masks acquired; 414/420 images and 136/144 mask grids pass separate QC. Six T1 conflicts, eight mask-grid failures and four metadata exclusions remain. All 25 eligible RESECT image–mask pairs are acquired; 24 await QC. | Run separate cache-only RESECT QC, continue any newly qualified acquisition, and perform intended-use anatomical review. Resolve scanner-frame and interscan alignment before spatial admission. Preserve person-level partitions. |
| Learning | BC256 matches search at 1.100; scratch RL256 reaches 1.098 on the same generated task, with 1,024 audited episodes and exact first-16 reproduction. Earlier failed fits remain. Stochastic recovery and transfer remain unproved. | Check two frozen forwards on the unchanged, prospectively admitted PAT05 TRAIN initial observation. Preserve all default learning and support guards, perform no actions or updates, and report distribution changes rather than patient performance. |
| Navigation / integration | Actual sub476 annotation evidence reaches canonical exclusions and desktop inspection. The observed six-B landmark API now advances and reopens immutable states; transport and packaged-startup repairs pass independent checks. Sparse landmarks do not establish dense tissue deformation or clearance. | Add a measurable observation-to-decision comparison once the required support is available. Keep missing clearance null and distinguish observed displacement from modeled interaction. |
| Operative science | Four supplied research reviews are read; proposed mechanisms are not implemented evidence. | Introduce an evidence-timed action/observation timeline with observed events distinct from proposals and model predictions. |
| Exposure / instruments | Generic capsule tools; hypothetical zero-thickness brain opening; no synchronized pair checking or measured skull/tool bundle. | Acquire same-person skull evidence and device metrology; implement synchronized full-pose checks after contracts. |
| Tissue / vascular | N32 improves adjacent agreement but fails the remaining-error envelope at states 49–60. The single N36 solve and independent replay pass all 60 conditional state screens (0.642702 mN envelope < 0.723189 mN allowance). Temporal review is next; spatial acceptance and calibration remain closed. | Prospectively specify one finest-mesh load-step comparison, reporting actual changes against the 0.8317 µN endpoint classification sensitivity as well as the original step criteria. No automatic finer uniform run, relaxed tolerance or calibration release. Retain FEBio; compare SOFA for later interaction only when physical evidence supports that task. |
| Physiology | Actual VitalDB case3/subject2861 DEVELOPMENT import/replay:21,970 pump/NIBP records and7,207 boundaries independently checked;15 focused checks pass. Original arterial task fails. Unshifted native clock supported, recording-start offset and cuff-measurement age unresolved. No validated response or learning. | Fresh-checkout cache preparation passes16independently repeated controls and real-record admission. Add observed-event integration; resolve release-specific native origin. Keep this separate patient's records separate from glioma anatomy, and qualify actual action/endpoint support before learning. |
| Independent evaluation | Reviews retain original failures and verify focused repairs. The opening pilot has 10,726 independent saved-output checks; this verifies execution/accounting, not useful learning or physical fidelity. | Challenge source/role/ancestry, physical assumptions and matched method costs. Generated development tests are permitted and labeled; clinical and transfer claims require their designated real observations or physical measurements. |

## Guard milestone evidence

The [critical-evidence integration](critical-structure-evidence.md) passes 64
combined checks, including actual-source save/reopen and desktop missing-support
refusal, plus the existing 42 policy checks. Independent reviews found additional
consumption and hash-dependent ordering gaps; these are repaired. No generated
patient, fabricated label, simulator trajectory or learning update was used.
The subsequent [actual annotation component](../artifacts/lausanne-sub476-annotation-v1/RESULT.md)
passes 77 combined checks, plus a final 13-check license-binding rerun. Actual
positives reach canonical exclusions and desktop inspection. The source-grid
interpretation, inferred units and false scanner/planning admission flags remain
explicit. Positive surgical route behavior and clinical coverage remain unproved.

The [annotation source audit](critical-annotation-source-audit.md) records a
specific IXI SynthStrip-lineage exclusion; COSTA's restricted dataset and unresolved
release rights/ancestry; and TubeTK's documented clinical MRA/tube pair with
unresolved release metadata. COSTA's separate weights archive is not patient data.
One original Lausanne manual mask was subsequently acquired; the other candidate
labels were not. No existing patient assignment changed.

See [guard receipt](../artifacts/real-observation-policy-v1/RESULT.md).
No patient fixture, model forward pass, optimizer update or simulator episode was
executed for these checks. Missing arguments, protocol requests and hash strings
exercise control flow only. Forty initial checks passed, but independent source
review found two outer failure loggers still able to write before refusal. Their
ordering was repaired and destination-absent regression checks added: **42 pass**.
This verifies the named refusal paths, not physical fidelity or clinical benefit.

## Real source ingestion evidence

The [Lausanne pilot](../artifacts/lausanne-original-pilot-v1/RESULT.md) acquired
36,661,729 bytes from one TRAIN person. The [data-use ledger](../manifests/real_observation_data_use_v1.json)
records zero fitting/RL contributions and the unresolved geometry gate. A
source-review timeout defect was repaired; two focused controls pass.
[TopCoW annotation access](lausanne-original-ingestion.md) is practical but
rights and per-case model-assisted lineage remain unresolved; none acquired.

The [full TRAIN intake acceptance](../artifacts/lausanne-train-intake-v1/RESULT.md)
freezes all 210 sessions; nine initial network failures resolved on metadata
resume. Fourteen focused controls pass after independent review found scope,
termination and provenance defects. The acquisition-free replay verified the
existing person's four source hashes and scalar/header QC; the cache-only
resume launched no worker. Each excluded the other 209 sessions. Neither adds
people, acquired source bytes, learning contributions or spatial admission.

RESECT-SEG source research and the [family role declaration](../manifests/resect-component-cohort-v1.json)
support a prospective visible-cavity component task. Independent metadata review
recomputed all23 IDs, hash assignments and protected Case4 partition. No new
RESECT image/label was opened to select roles. Root confirmed Case3 mask API
size/hash metadata; a separate README request returned HTTP429 and its failure
receipt is retained locally. Earlier source research read the explicit license;
archiving the rights record and exact per-file admission remain pending.
[Measurement follow-up](tissue-mechanics-measurements.md#october-8-real-observation-follow-up)
records label semantics, derivative overlap and the accepted MULTIS run005.

The first [bounded original-image expansion](../artifacts/lausanne-train-intake-v1/acquisition-batch-01-summary.json)
completed sub-001, then closed at its time budget during sub-002. Acquired totals
are two people/two sessions/69,376,670 bytes. The next T1 partial is6,291,456
bytes and has no source/QC admission. All worker processes exited. Root checked
published fixity, receipt identity and source snapshots; no anatomy was accepted.

The second [bounded expansion](../artifacts/lausanne-train-intake-v1/acquisition-batch-02-summary.json)
resumed sub-002 and completed sub-005/006/015 in 521.3543 seconds. Current totals
were six people/six sessions/203,467,291 complete source bytes; 204 sessions remained
incomplete. All six source receipts and published fixity were reverified. These
are acquisition/header/scalar checks, with zero spatial admissions or learning.

The third [bounded expansion](../artifacts/lausanne-train-intake-v1/acquisition-batch-03-summary.json)
completed sub-007/021/022/450 in 527.2644 seconds, adding 133,217,934 bytes.
Closure totals were ten people/ten sessions/40 files/336,685,225 bytes, with
200 sessions deferred by the byte budget. Root and an independent reviewer
reverified all ten source receipts, published fixity and historical code
snapshots. The current edited imaging/core files differ from those retained
execution versions; this does not rewrite the batch's source provenance.
No registration, anatomical acceptance or training contribution is claimed.

The fourth [bounded expansion](../artifacts/lausanne-train-intake-v1/acquisition-batch-04-summary.json)
completed sub030/034/035/036/045 in 599.141 seconds, adding 241,628,307 bytes.
Its 267,911,302-byte attempted-source budget includes the sub077 attempt and
is not measured network transfer. Sub077 exhausted an 18.968-second worker
allowance (exit124); its 5,242,880-byte partial has no acquisition receipt.
All 210 sessions reconcile: ten cached, five completed, one timeout and194
byte-budget deferrals. This is a process failure, not a data-quality finding.

The separate [sub476 intake](../artifacts/lausanne-train-intake-v1/acquisition-batch-05-summary.json)
completed in 60.898 seconds and added 21,050,678 bytes. Selection followed an
exact source-author voxelwise-label crosswalk and frozen TRAIN membership,
before annotation payload inspection. Current totals are16 people/16 sessions,
64 files and599,364,210 bytes;194 sessions remain incomplete. Root and independent
review reverified every source receipt, original-file fixity and retained source
snapshot. All spatial admissions and fitting/IL/RL contributions remain zero.

The sixth [bounded expansion](../artifacts/lausanne-train-intake-v1/acquisition-batch-06-summary.json)
completed sub047/051/052/062, adding 213,831,608 bytes in 598.517 seconds. It
stopped before launching another worker; no new worker failed. Root and an
independent audit reverified all 20 retained receipts (80 files,813,195,818 bytes).
Cumulative unacquired sessions are190. The batch-specific192-not-reverified
list also includes cached sub450/sub476, which it did not revisit before the
time stop; this later audit verified both. The prior sub077 partial stays excluded.

RESECT's official file-version endpoint now resolves revision2 to its explicit
download route. The [bounded pilot](../artifacts/resect-component-admission-v1/RESULT.md)
requires the pinned rights README before image/mask acquisition or QC. Its actual
rights-only request followed one official redirect and stopped on HTTP429 in
0.7040433 seconds; no Retry-After header, scientific payload or automatic retry.
Independent source review accepted the corrected import/watchdog/closure logic;
24 metadata/process controls pass. Anatomy and training admission stay pending.
MULTIS's HTTPS static archive refused connection; the verified HTTPS SVN donor
directory exists, but the exact run005 filename returned404. A four-request
metadata audit found no published run005 checksum. Local hashes alone would
not authenticate HTTP-delivered measurements. The subsequent official SVN Data
listing exposes run027, a different dissection experiment, not accepted
indentation run005. DataCite confirms the HTTP archive as the official DOI
destination but supplies no byte fixity or authenticated alternate. The project
download-index HTTPS attempt timed out. No TLS bypass or scientific payload
access occurred; authenticated run005 source/fixity remains the next dependency.
Another four-request RESECT metadata refresh confirmed the exact revision2
README size and hashes; its official renderer returned HTTP500. The previous
429 download endpoint was not retried. No exact rights text or new scientific
payload was acquired, so that source-specific dependency remains open.

The [seventh](../artifacts/lausanne-train-intake-v1/acquisition-batch-07-summary.json)
and [eighth](../artifacts/lausanne-train-intake-v1/acquisition-batch-08-summary.json)
batches completed13 sessions from12 new people, adding527,758,292 source bytes.
Both closed below600seconds and their268,435,456-byte source-size limits, without
new worker failures. Root and independent audits verified all33 receipts and
originals:32people,132files,1,340,954,110bytes. Sub074's two visits are one person.
Sub077 is now fully verified; the earlier timeout stays in the history.
All167 remaining people/177 sessions remain in the frozen denominator.

The [VitalDB component](../artifacts/vitaldb-recorded-component-v1/RESULT.md)
adds one separately identified DEVELOPMENT person, not a glioma patient or RL
episode. A120second front-end timeout was preserved; the separately declared
official mirror completed6,537,712bytes in15.585seconds with the published hash.
Native headers refuted the live API's arterial-track inventory. A new task,
declared before selected values, uses actual pump and intermittent cuff-monitor
records. All21,970 records exactly match independent decoding; every7,207 event
boundary passes prefix/age/snapshot checks. Initial zero-origin interpretation
was also wrong and is retained as invalid evidence: accepted replay leaves the
native recording-start origin null. Fifteen focused checks pass. No training,
delivered dose, current-pressure inference, physiological fit or clinical result.

The [ninth batch](../artifacts/lausanne-train-intake-v1/acquisition-batch-09-summary.json)
completed six more people/sessions in495.218seconds, adding267,508,864bytes.
Root reverified all39 receipts/originals/source snapshots:38people,156files,
1,608,462,974bytes. Independent audit agrees;161people/171sessions remain.
The next bounded intake is running.

The [cache-preparation review](../artifacts/vitaldb-recorded-component-v1/cache-preparation-independent-review-v1.json)
passes16controls and reproduces21,970records through the unchanged reader from
an isolated fresh cache using authenticated existing bytes. No new remote
transfer, person, observed trajectory or optimizer contribution was added.

Batches10–12 added14sessions from14people; all53receipts/originals/source
snapshots passed root and independent audits. Closure is52people/53sessions/
2,342,939,538bytes. Sub128's timeout resumed successfully; sub454's partial is
excluded and its timeout preserved. The thirteenth intake runs serially with
600seconds/536,870,912source-byte bounds.

RESECT annotation rights are now bound to the exact acquired README. The
reviewed downloader accepts only the official storage bucket/frozen-hash object;
13controls pass after repair of malformed-query exception logging. The actual
Case3 acquisition stopped before bytes with certificate-chain verification
failure. Verification remains enabled; source connectivity is being diagnosed.
No source role or training admission changed.

## Numerical mechanics evidence

The [half-height comparison](../artifacts/mechanics/hbe-halfheight-execution-v1/RESULT.md)
completed four native calls in 40.9184 supervised seconds and 164,626,432 bytes
sampled peak memory. All four compression/tension N8/N12 comparisons passed
61 states each and the unchanged original numerical gates. Independent replay
matched every readout field; a separate NumPy/plain-text reconstruction matched
all-node displacement and summed reflected raw-reaction discrepancies. All 193
baseline input hashes remained exact. No measured response curve was opened.

This is numerical equivalence for the tested symmetric axial branch. The
earlier spatial-convergence failure and full N24 timeout remain, and the
1000-Pa modulus is still a numerical gauge. No tissue-force/material accuracy,
calibration, RL data or finer-mesh acceptance follows. The next spatial study
was separately declared and executed; measured-response partitions stay closed.

The [finer axial study](../artifacts/mechanics/hbe-halfheight-spatial-execution-v1/RESULT.md)
completed three new native cases and exactly replayed five old cases. All new
individual checks passed; the co-primary spatial comparison failed only the
compression N12→16→24 force trend. Fine motion changes are4.799680/3.638568 µm;
all other15 group criteria passed. Total duration365.799 seconds, sampled peak
memory761,856,000 bytes and final output477,486,884 bytes stayed within budget.
Independent replay matched all eight records/488 states without field differences.
Saved-output diagnostics indicate positive but slowly varying convergence order;
estimated remaining error is not measured accuracy. No threshold changed, curve
opened, material fitted or prediction converted into RL experience.

## Remaining delivery gates

The full goal remains active: broad actual TRAIN use and contribution accounting;
eligible recorded offline RL; anatomy update/decision replay (4A); reviewed
diffusion disconnection (4B); skull exposure and synchronized workability (5A);
measured dynamic tool/tissue effects; vascular patency/bleeding (7A/7B); delayed
physiology and rescue; information acquisition (9A), computation allocation (9B)
and explicit branch assumptions (9C); navigation uncertainty (10A); functional
desktop integration and independent real-case acceptance. Current metadata,
guards and historical results do not satisfy those gates.

## October 8: batch13 closure and annotation metadata review

[Batch13](../artifacts/lausanne-train-intake-v1/acquisition-batch-13-summary.json)
added six people/sessions. All59 receipts/originals/source snapshots passed root
and independent checks; audited totals are58people/59sessions/2,611,520,354bytes.
Its599.124second run retained the sub167 timeout. The three cached but unvisited
sessions explain154not-verified-this-batch versus151cumulative remaining.
Batch14 subsequently completed; the later closure is recorded below.

The [annotation metadata review](../artifacts/lausanne-annotation-expansion-v1/RESULT.md)
qualified144TRAIN masks across106people/117sessions (13,977,015prospective bytes).
Four failures remain excluded, including two wrong-patient/session source links.
No new mask payload, label-quality acceptance or optimizer update follows.

## October 8: completed plate-boundary numerical diagnostic

The [saved study](../artifacts/mechanics/hbe-halfheight-boundary-execution-v1/RESULT.md)
completed all three native variants in 739.194 seconds with 1.176 GB sampled RSS.
Independent raw replay reproduced four complete histories (244 states), regional
diagnostics and the comparison exactly. All original source/output hashes and
resource bounds passed. No measured curves or material fitting were used.

Plate refinement has a larger effect than the equally sized interior refinement,
but the conditional plate correction explains only part of the earlier global
change. The old spatial study remains failed. One prospective global N32 check
is being specified; it is not yet implemented or released. Final spatial and
load-step acceptance are required before physical calibration.

## October 8: expanded original-data closure and partial RESECT acquisition

Batches 14–16 acquired 17 more sessions from 16 people. Root and independent
checks agree on **74 people / 76 sessions / 3,496,113,284 source bytes**, including
152 images. The remaining TRAIN denominator is 125 people / 134 sessions.
All original failures are retained; later acquisitions do not alter past batch
closures. Batch17 continues separately with the frozen source and roles.
See the [intake result](../artifacts/lausanne-train-intake-v1/RESULT.md).

The Case3 original ultrasound transfer succeeded after the reviewed native-trust
repair. Its mask was rate-limited, leaving no complete pair or QC admission.
The [partial failure](../artifacts/resect-component-admission-v1/acquisition-native-trust-01/worker-result.json)
and independent review are retained. No component or RL update occurred.

## October 8: real held-out ultrasound correspondence result

The [Case4 study](../artifacts/resect-case4-sparse-update-v1/RESULT.md) completed
separate header, six-B fit/freeze and thirteen-V evaluation stages. All points
were retained. RMS errors are 4.044969 mm without update, 1.020253 mm rigid and
1.104199 mm fixed IDW; independent original-tag/quaternion calculations match
within 8.53e-14 mm. No V-dependent tuning or method promotion occurred.

This advances the measured-correspondence part of Objective 4A. Actual imaging
observations condition the update; this is not preoperative-only inference,
learned surgery, dense retained-tissue accuracy or demonstrated decision benefit.
The historical failures remain. Case4 V is consumed for future adapted methods.
Next: observed-state replay and stale-result invalidation, with unsupported
clearance/cavity decisions remaining unavailable.

## October 8: ninety-person intake and completed Case3 pair QC

[Batch18](../artifacts/lausanne-train-intake-v1/acquisition-batch-18-summary.json)
closes at 90 people, 95 sessions and 4,315,283,541 bytes after root and independent
original/source-snapshot checks. Nineteen sessions from sixteen people were added
in batches17–18. The remaining denominator is 109 people / 115 sessions; the
sub251 timeout remains visible and the nineteenth attempt is separate.

[Case3](../artifacts/resect-component-admission-v1/RESULT.md) now has one complete
real ultrasound/cavity pair. Independent source/grid/scalar checks reproduce
20,005 binary positives and 0.00003068 mm maximum grid difference. Fixed native
slice inspection preserves raw header differences and source/manual annotation
limits. No anatomical acceptance, optimizer updates, training or RL follows.

## October 8: acquired bytes and modality QC are separate

[Batch21](../artifacts/lausanne-train-intake-v1/acquisition-batch-21-summary.json)
closes at 106 people / 111 sessions / 5,086,888,757 acquired source bytes.
Root and independent checks verified every acquired receipt, original byte hash
and source snapshot. The acquired denominator includes sub253’s T1 failure
(`QFORM_SFORM_DISAGREEMENT`); full-pair QC passes for 105 people / 110 sessions.
Its passing TOF is retained separately, without clearing T1 or admitting planning.
All 111TOF checks pass. Acquisition is not fitting: optimizer updates remain zero.
Batch21’s sub283 timeout and older failures remain; later batch22 data are excluded.

[RESECT expansion metadata](../artifacts/resect-train-expansion-v1/RESULT.md)
qualifies 25 image/mask pairs in 13 labeled TRAIN people, retaining three absent label
slots. Another 24 pairs / 48 files / 548,605,591 bytes need exact-whitelist acquisition
and separate QC; new revision-2 routes remain untested. Metadata review passed.
[IBSR](../artifacts/ibsr-source-qualification-v1/RESULT.md) remains unadmitted after
three official metadata checks: README access, label ancestry/domain, participant
overlap and use rights remain unresolved. No new agreement or payload was used.

## October 8: N32 completion retains a negative accuracy result

The [N32 study](../artifacts/mechanics/hbe-global-n32-execution-v1/RESULT.md)
completed with one native call, no retry, and all resource limits respected.
Independent review reproduced 305 states across five readouts and checked all
unequal-spacing force-order calculations with a different root solver. Adjacent
reaction/motion changes now pass. The prespecified two-limit sensitivity envelope
is 0.985189 mN against 0.724169 mN; states 49–60 fail. This conditional estimate is not
a rigorous continuum bound. Spatial acceptance and physical calibration remain
closed, and finest-level temporal verification remains required. No measured
biological curves or patient records were consumed. The N36 proposal is an
unexecuted forecast requiring separate review and resource/source declaration.

The annotation intake is now implemented and independently reviewed: 54 software/
metadata checks pass after a real process-interruption bookkeeping defect was
reproduced and repaired. Public metadata proofs are portable. The first cache-only
real-mask pilot reproduced 193 positive voxels and passed independent source/grid
checks; full-inventory acquisition/QC and actual component fitting remain open.

## October 8: preserve a real annotation format failure before repair

The [first annotation pilots](../artifacts/lausanne-annotation-format-v1/RESULT.md)
separate acquired bytes from usable labels. Sub476 passed cache-only content/grid
checks. Sub022's exact source downloaded successfully, but its first content check
refused a valid private-format NIfTI extension before reading labels. The new
strict framing reader preserves opaque hashes without interpreting those contents;
56 owner and 32 independent controls pass. Independent metadata-only inspection
confirms the actual 592-byte prefix. The original failed receipt and a reviewer
control typo remain recorded. The next executable step is an explicit cache-only
QC repeat, followed by bounded inventory processing; no optimizer update or
clinical/planning admission is supplied by this repair.

The separate sub022 cache-only repeat subsequently completed in 0.538385 seconds
without a GET. Independent streaming reproduced 2,899 positive original voxels,
gzip/fixity checks and exact coded-grid equality with its original TOF. Private
extension hashes and the first failed receipt remain unchanged. Subtype is still
unresolved; the passing source component has no new anatomical/planning or
training admission. Full-inventory content/reference checks are next.

The [RESECT TRAIN importer](../artifacts/resect-train-intake-preparation-v1/RESULT.md)
now passes 75 independent/owner controls. Four initial counterexamples exposed
two acceptance defects: incomplete file lists could report completion, and late
parent bookkeeping could remain accepted. Exact pair/per-file receipt validation
and a final elapsed-time gate repair both. Authentic Case3 compressed-cache
controls and fresh-checkout metadata preflight passed without downloads or
decompression. The first new-pair acquisition, independent source check and
separate label/image QC remain the next steps; all patient roles and rights remain.
