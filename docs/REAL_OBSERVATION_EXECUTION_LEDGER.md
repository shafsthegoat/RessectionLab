# Real-observation execution ledger

Updated October 9, 2026. The [October 6 supergoal](SUPERGOAL_REAL_OBSERVATIONS.md)
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

## Primary transfer hypothesis, October 9

The user's October 9 steering (source SHA-256
`7c1a7d1de504b449376ac8f88a488d8f77d62ce6c622b8c8a6fa2def1c213600`)
asks whether rich privileged simulation training produces **transferable planning
under limited preoperative imaging** on entirely unseen patients, compared with
strong classical and learned-model planners with the same test-time information.
This is a falsifiable primary hypothesis; the within-patient optimization pilot
remains a valid development and application mode. The amended
[protocol](../EXPERIMENT_PROTOCOL.md) freezes the patient cluster, permitted MRI/CT
and derived-feature lineage, deployable action generator/search inputs, full
strategy before private evidence access, independent patient-linked scoring and
separate simulator/anatomical/physical-clinical result levels.

Five [parallel audits](../artifacts/limited-observation-generalization-audit-v1/RESULT.md)
found reusable nominal observations, spatial policy and beam search, alongside a
critical gap: hidden support can still influence action legality and predicted
cavity. A 33-control interface suite and tiny generated private-target swap
passed, but they do not test that support or file-access boundary. Existing
generated learning has one two-step zero-patient admission; the four completed
annotation-assisted TRAIN comparisons favored greedy over frozen imitation and
do not test unseen-patient limited-input transfer. No acquired/admitted cohort
yet combines glioma scans with complete independent vascular, functional and
surgical-interaction truth for the same person. The next source slice is a
fail-closed planning-input/private-reference boundary, not an unqualified
population training sweep. The [first bounded implementation](../artifacts/limited-observation-boundary-v1/RESULT.md)
now seals complete nominal routes before target-only private evaluation on one
exact generated fixture. Sixty focused and 131 independent controls pass; all
ten wider failures reproduce on frozen base as legacy weight-lineage rejects.
It does not admit patients, hide support/hazards, validate clinical outcomes or
establish transfer. The [SynthRAD archive](../artifacts/synthrad-verified-download-v1/RESULT.md)
is byte-verified but awaits separate image and anatomy QC. A bounded existing
manifest review found no further ready declared download; CFB remains deferred
for lack of an authoritative verified-HTTPS payload route.
The [first TRAIN-pair QC preparation](../artifacts/synthrad-first-pair-qc-preparation-v1/RESULT.md)
passes 115 offline controls and independent proof-portability review. It has
not opened a patient image member. The later
[restricted `1BA336` execution](../artifacts/synthrad-first-pair-qc-execution-v1/RESULT.md)
completed mechanical QC within fixed limits; 87 independent receipt checks
passed. It remains `review_pending` for privacy, anatomy, scaling and overlap.
The other 125 TRAIN subjects were not attempted, and no scientific admission
or training update follows.

The [first v3 mechanics execution](../artifacts/hbe-branch-calibration-v3-failure-v1/RESULT.md)
is a terminal negative result under its frozen no-extrapolation protocol. Both
allowed axial curves parsed, but final displacements exceeded the solved domain
by 3.5215 µm in compression and 2.2715 µm in tension. Independent failure and
coordinate audits verified no completed fit, native solve, parameter freeze or
held-out torsion access. Any corrected domain requires a separately versioned
and reviewed protocol; v3 will not be retried or patched in place.
The [v4 preparation](../artifacts/hbe-branch-calibration-v4-preparation-v1/RESULT.md)
binds exact coordinate-derived endpoints and portable review evidence. Its
execution, fitting, and held-out gates remain closed. Independent isolated
controls and focused tests pass, but twelve new same-endpoint references and a
native reconstruction/resource protocol are still required; no physical
calibration or patient interaction was validated.
The [v5 deck-only preparation](../artifacts/hbe-branch-calibration-v5-preparation-v1/RESULT.md)
freezes twelve candidate same-endpoint schedules but no actual deck identity or
native result. Independent review accepted 243 fixture/source controls after a
declared-curve correction. Execution and fit remain refused. This specimen
track does not substitute for patient-anatomy displacement or measured tool
interaction validation.
The [Case4 boundary-6 mesh preparation](../artifacts/mechanics/resect-case4-boundary6-preparation-v1/RESULT.md)
passes 176 source-only controls and preserves all geometry gates. It proposes
one curvature-off 6/24 mm generation after a separate committed-source release;
the [one released generation](../artifacts/mechanics/resect-case4-patient-mesh-boundary6-v4/RESULT.md)
returned a mesh but failed both directed 2 mm surface gates, with sampled
maxima 3.869/2.264 mm. No solve ran. Case4 V outcomes were previously
revealed, limiting any later FEM comparison to retrospective development.
The first catalogued evaluation-role full before/during landmark pair is
Case19, whose payload, frame and eligibility gates remain unopened.
The [one v4 saved-mesh fidelity witness](../artifacts/mechanics/resect-case4-v4-fidelity-witness-execution-v1/RESULT.md)
replayed the failed 2 mm gate without remeshing or solving. Its predeclared
30 mm localization rule failed (best joint-cube capture 25.25%/45.52% versus
80% required in both directions); the estimated Case4 source anatomy and
patient-size solver remain unqualified. This is a negative DEVELOPMENT
engineering result, not observed tissue deformation or clinical evidence.

## Acquisition, QC and mechanics checkpoint, October 9 at 01:18 UTC (historical)

At this checkpoint, the [source-frozen CT/MRI archive transfer](../artifacts/synthrad-archive-runner-v1/RESULT.md)
was active: exact HTTP 200 length/publisher MD5, no scientific decoding. Its 32
offline controls and independent process/hash expiry checks include a repaired
sleep-related campaign deadline defect. The full 14.47 GB archive was running continuously
within persisted attempts and deadlines; partial bytes do not establish fixity or
intended-use QC. Noncommercial rights and all prospective patient roles remain.

The [RESECT QC preparation and first actual attempt](../artifacts/resect-deferred-qc-preparation-v1/RESULT.md)
retain four initially failed receipt controls and their repair, then the first
real execution failure. Case2's inactive qform cannot be reconstructed by the
shared raw-header recorder although its active sform passes the unchanged
geometry rules. No scalar loop completed; 23 further pairs remain unattempted.
The minimum continuation repair passes 23 focused independent controls. The
[second actual batch](../artifacts/resect-deferred-qc-execution-v2/RESULT.md)
finished all 24 new pairs in 19.063570 seconds. All recorded mask checks passed;
all 24 image scalar reviews remain unfinished. Independent review authenticated
288 bindings and verified the saved active transforms and grid corners without
changing failed outcomes. Completing scalar review requires the versioned recorder
compatibility change; source headers, tolerances and admissions remain unchanged.

The [branch-specific calibration implementation](../artifacts/hbe-branch-calibration-preparation-v1/RESULT.md)
is committed as `de420c3`, with 27 owner and 24 independent controls passing.
Its exact 38-file source archive passed metadata preflight in 4.388249 seconds.
The first actual supervised child failed with `Interpreter identity differs`
before any measured response access, fit or native call. The failed attempt is
preserved and the launch environment is being diagnosed. The fixed one-hour
experiment still requires both actual fitted axial confirmations before freeze
and withheld torque access. Physical agreement and patient validity remain open.

The [versioned RESECT QC rerun](../artifacts/resect-deferred-qc-v2-execution-v1/RESULT.md)
now records 24 new structural passes and one inherited pass. Independent saved-output
review verified 498 bindings, all full image scalar loops, binary-mask receipts,
source and before/after fixity. The original v1 recorder failures remain unchanged.
No anatomical, scanner-frame, training or planning admission follows from those
structural checks. A distinct real-anatomy displacement candidate is TRAIN Case3;
its frame, rest-mesh and numerical qualification remain open, and no tool/contact
force measurements are available.

The [runtime replacement diagnosis](../artifacts/hbe-branch-calibration-runtime-drift-v1/RESULT.md)
pins the old and new Python executable hashes. Seven compact original-path failure
and release receipts are preserved for prospective v2 verification. The v2 source
retains the same scientific design and input split. After 82 owner and 15
independent controls, the [exact v2 archive and release](../artifacts/hbe-branch-calibration-runtime-v2-preparation-v1/RESULT.md)
passed a fresh worker-equivalent preflight; one bounded measured attempt is now
complete but failed while parsing an axial CSV column label as a number. The
original schema declared no header; this mismatch stopped before any fit,
fitted deck or native solve. An axial calibration access was attempted, while
the held-out torsion reader stayed sealed. The [independent failure audit and
allowed-axial diagnostic](../artifacts/hbe-branch-calibration-v2-failure-audit-v1/RESULT.md)
show compression's full member was read before the header error; tension was not
reached. The batch ledger cannot resolve member-level reads, so partial access
remains explicit. This negative result is preserved without a retry; a new
version requires the exact checked axial header contract. The
[v3 preparation](../artifacts/hbe-branch-calibration-v3-preparation-v1/RESULT.md)
now changes only the two allowed axial CSV headers plus explicit predecessor
lineage; its owner and guarded independent controls pass. No measured v3 fit,
native solve or held-out torque reveal has occurred. A new root-controlled
release remains the next gate; torsion framing is still uninspected.
The archive download continues with an unverified growing partial file and active
worker and curl processes.

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
The [PAT05 fixed-input diagnostic](../artifacts/pat05-forward-diagnostic-v1/RESULT.md)
completed exactly two frozen forwards in 5.180709 seconds supervised time with
1,549,369,344 bytes sampled peak RSS. All 830 independent metadata/numeric checks
passed. RL256 ranks STOP first of 71 (0.813695 preference); initial ranks it last
(0.009884). These are conditional model preferences, not clinical probabilities.
No action, patient fit or planning endpoint was executed. The input is annotation-
assisted and includes historical simulator-filtered candidates and estimated support.
Generated-to-patient tool lengths, scale, horizon and inventory differ. The
[length-only mechanism test](../artifacts/native-opening-length-sensitivity-v1/RESULT.md)
now completes two fixed forwards on the familiar generated DTO, with no updates.
Four length descriptors alone raise trained STOP preference from 6.7249% to
99.3915%; its STOP score is unchanged while movement scores fall. Initial weights
still favor movement. All 311 saved-output checks pass, and the same five choices
remain. This does not certify altered physical tools or explain every PAT05 shift.
Next prepare real tool geometries within generated contexts and recertify their
action inventories/search baselines; substantial training awaits validated mechanics. The completed PAT05 run is unchanged;
a future index-writer repair includes nested inherited index files, whose bytes
were independently verified despite omission from the original outer index.

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

The additional [104-file original-source queue](../artifacts/resect-train-originals-execution-v1/RESULT.md)
completed continuously in 265.1827 seconds: 28 MRI, 17 ultrasound and 59 correspondence
files (672,336,857 bytes), all independently byte-verified. Combined original coverage
is 70 images/59 landmark files across the same 14 TRAIN people; cavity labels remain
25 masks/13 people. No new roles, scientific QC, fitting or recorded transitions.
The original 890-file queue plus 104 files is 994; 144 Lausanne annotations are separate.
The [N36/S120 solve and independent review passed](../artifacts/hbe-n36-temporal-execution-v1/RESULT.md):
all 121 states and 61 common-state comparisons are verified. Force change is
1.87025e-12 N; probe change is 2.31371e-14 m. No spatial-screen classification changed.
The 1,946.3351-second phase used one native solve and zero remeshing. The original
failures and a reporting-only independent-audit repair remain. The [N24/S120 tension
run](../artifacts/hbe-tension-n24-temporal-execution-v1/RESULT.md) also passes, with
independent exact reconstruction of all 121 states and both sensitivity groups.
Force/probe changes are 6.51507e-12 N / 1.41419e-14 m, with no classification changes;
the full phase took 301.9559 seconds within its 600-second cap. No measured response
was accessed. Next is the declared branch-specific common-scale calibration using
compression N36/S120, tension N24/S120 and torsion ± N12/S120. Two actual fitted
axial confirmations must precede freezing and opening withheld torsion measurements.

## Latest real-source integration checkpoint

The [sub-476 geometry probe](../artifacts/lausanne-critical-geometry-probe-v1/RESULT.md)
completed one actual existing-API check: canonical aneurysm positives produce
FORBIDDEN_COLLISION, with three positive cells and 24 unknown cells in the same
27-cell query. The zero-exclusion comparison is explicitly a software ablation.
Independent saved-output review passes without a new image decode or geometry run.
Source-frame, coverage and false scanner/spatial admission flags are preserved.
No target/access/route or learning record was manufactured. A bound desktop probe
entry is the remaining component integration gap; this is not full planner acceptance.

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


### October 9: freeze paired IXI component intake

The [IXI paired vascular-component intake](../artifacts/ixi-t1-mra-vessel-byte-intake-v1/RESULT.txt) is independently reviewed and frozen before archive-body intake. Metadata identifies 569 paired T1/MRA people, including all 100 vessel-label IDs; all 582 source people retain prospective TRAIN/SELECT/MEASUREMENT_EVAL roles (407/88/87). The authorized three opaque archives total 17.23 GB. Nineteen generated transport controls pass; verified TLS, resumable transfers, checksum provenance and storage reservations are enforced. This is a healthy vascular component, with acquired MRA separated from derived vessel labels. Image/frame/coverage review, patient admission and unknown pretrained overlap remain unresolved. Acquisition will continue without per-batch QC pauses.

Prepared proposals and review evidence are preserved in the artifact directory. Only the documented declaration-status, canonical cohort binding and execution flag changed at freeze; prior patient roles and all payload-access gates are preserved. A separate runtime declaration binds the freeze commit before launch.


### October 9: exact generated inference parity with modest peak reduction

The [generated 64³ lazy-concat comparison](../artifacts/lazy-concat-generated-parity-v4/RESULT.md) completed both baseline and modified forwards. All 786,432 logits and decoded labels are identical to each other and the earlier accepted reference. The final decoder avoids a 67.1 MB allocation, but sampled whole-process peak falls only 7,995,392 bytes (1.06%) in this single ordered pair. Both workers exited cleanly; no robust speed, full128 feasibility or patient-accuracy claim follows. The next engineering test is the separately reviewed full128 lazy arm, with both active downloaders included in its host monitor.


### October 9: staged MR preparation and full generated control

The [staged MR pipeline](../artifacts/remind-staged-full-generated-preparation-v1/RESULT.md) is prepared for a bounded full-size generated test. Metadata validation and selected-series conversion run in separate workers; v2 preserves read accounting even on the two injected supervisor failures. Fourteen generated controls and independent review pass. The next execution tests 59,520 generated metadata records and 64 × 512 × 512 image planes before any actual-patient conversion; no anatomical or training admission follows from this engineering check.


### October 9: integrated sealed-strategy private vascular evaluator

The [private vascular evaluator](../artifacts/private-vascular-evaluator-v1/RESULT.md) is now integrated into the reusable library. It seals and checks the complete strategy before loading an evaluator-owned vessel reference, then measures full shaft/tip exposure and separate removed-cell overlap. Unknown coverage remains unknown. All 125 focused author, independent and related regression tests passed; source snapshots were unchanged. This is generated-only interface validation, with real-person admission and hard process supervision still required before acquired-patient evaluation.


### October 9: paired-data transfer started alongside Tracto

The [IXI intake launch](../artifacts/ixi-t1-mra-vessel-byte-intake-v1/launch/RESULT.txt) is confirmed after the committed role freeze. At 11:49 UTC, 1.67 GB of the first archive was present as unverified resumable partial data; no archive was complete. The concurrent Tracto download had 1,510 verified files totaling 52.26 GB, with zero failed attempts. Both transfers continue; snapshots do not admit image or anatomical use.


### October 9: source-bound mechanics launcher adoption

The [mechanics launcher extension](../artifacts/hbe-v5-n12-nocache-launcher-preparation-v1/RESULT.md) is integrated with 18 passing generated controls and independent cleanup-path validation. It preserves original integrity checks while applying the tested per-file cache hint and accounting for its own work. The next numerical row has not run: the observed 50% available host memory is below its reviewed 55% launch floor. No new native output or reservation exists.


### October 9: full128 pressure negative before lazy decoder

The [full128 inference attempt](../artifacts/full128-lazy-encoder-pressure-negative-v2/RESULT.md) stopped on the unchanged host-pressure guard after 3.00 seconds, with 1.668 GB sampled process-group RSS. Its 25 layer markers stop in the encoder before any decoder operation, so the lazy-concat change was not exercised and no full-size prediction exists. Child cleanup passed, swap use was flat, and both downloaders retained their exact identities and stayed below their allowances. The separate initial supervisor import invocation failure occurred before preflight or model execution. Neither failure establishes patient accuracy or assigns causation for system-wide pressure. Early-encoder allocation is the next measured engineering bottleneck.


### October 9: preserved full MR failure and identity-order repair

The [first full generated MR conversion failure](../artifacts/remind-staged-full-failure-and-order-fix-v1/RESULT.md) is preserved with all 141 reconciled read intents and clean child reaping. The minimal repair makes per-object header summaries identity-ordered while preserving exact physical slice order and values; 16 tiny tests and 16 independent mutation checks pass. The separately authorized full-fixture reuse now completes in 1.95 seconds at a 359,989,248-byte worker peak, pending independent saved-output audit. No real-patient conversion or anatomical admission has occurred.


### October 9: corrected full generated MR pipeline independently passes

The [saved-output result](../artifacts/remind-staged-full-generated-success-v1/RESULT.md) passes all 64 physical-plane checks and 3,327 bounded generated-value samples. All 141 read intents reconcile with zero unknown bytes, both sequential workers are reaped, and the original fixture and ordering negative remain intact. The 1.953-second run peaks at 359,989,248 worker RSS bytes. The next scope is a separately reviewed exact TRAIN-patient conversion; no preoperative, anatomical or training admission is inferred from this synthetic workload. PROJECT_STATUS now consolidates the recent completed slices while this ledger retains their full chronology.


### October 9: all-method sealing before private vascular scoring

The [matched-method adapter](../artifacts/matched-private-vascular-adapter-v1/RESULT.md) now connects existing four-method strategy records to the private evaluator. All complete strategies undergo preflight before any reference callback. An independently reproduced cross-method reference-substitution defect was corrected with captured bindings and final integrity checks. Fifty-two candidate controls and the root integrated 88-test set pass; both the original finding and test-harness errors remain preserved. No learned forward, training, patient admission or real-geometry registration is added.


### October 9: Preserve normalization numerical counterexample before model attachment

The in-place normalization prototype diverged from the native operator by 1.7628903 on a large-offset generated input, with 43 activation sign disagreements. The independently reviewed negative remains excluded from the trained model; mathematical equivalence did not establish numerical or alias safety. See the [preserved evidence](../artifacts/inplace-instance-norm-numerical-negative-v1/RESULT.md).


### October 9: Record bounded six-stage recomputation parity and unresolved memory benefit

The untrained six-stage graph completed in 0.881 seconds with a 226,410,496-byte sampled peak. Reviewed worker code asserted exact baseline/recomputed logits, but no raw arrays were retained for independent reload. Five tiny controls pass; no measured memory reduction or trained-model benefit is established. The original mutable-stage counterexample and unsuitable tiny-controller failure paths remain documented. See the [preserved evidence](../artifacts/recompute-highres-skip-generated-v3/RESULT.md).


### October 9: Validate bounded full-size tiled vessel-contact evaluation

The [independently audited generated profile](../artifacts/private-vascular-streaming-profile-v1/RESULT.txt) evaluates a 256 × 256 × 192 rotated grid in 7.883 seconds with a 41.594 MiB process peak. It considers all 3,072 tiles, prunes 96.582%, preserves repeated-action unions and unknown coverage, and cleans up its single child. This is measured feasibility, not a measured speedup over the unexecuted dense baseline or patient/clinical validation.


### October 9: Verify first bounded real MRI conversion and retain spacing discrepancy

One exact 64-plane ReMIND-002 TRAIN MRI conversion finished in 2.016 seconds at a 438,386,688-byte (418.1 MiB) worker peak. The [saved-output audit](../artifacts/remind-exact64-patient-MR-conversion-result-v1/summary.json) reconciles all 141 read intents, independently reproduces raw-plane geometry and checks NIfTI units/forms, hash and 3,327 bounded samples. A spacing-tag inconsistency remains recorded: SpacingBetweenSlices says 1 mm while image positions imply approximately 3.3 mm. The verified grid uses physical positions/orientations. Worker all-source pixel equality and independent bounded sampling remain distinct. Anatomy, coverage, preoperative timing, MR-US registration and training/planning admission remain unverified or closed.


### October 9: read-only mechanics validation at observed 54% host availability

The [separate hinted probe](../artifacts/hbe-v5-n12-50pct-readonly-v1/RESULT.md) passes independent saved-evidence audit: 14 verified predecessor opens, 2.80 GB call-counted, 5.470 seconds, 178,323,456-byte worker-group peak and clean reaping. Admission was 54%, with every sampled host observation at 53–54% and normal pressure. The protocol name contains 50%, but the result does not validate a 50% start. The original v1 55% admission rule remains unchanged; no native solve, measured-response validation or ordinal reservation occurred.


### October 9: explicit versioned mechanics host policy

The [v2 launcher source](../artifacts/hbe-v5-n12-nocache-v2-source-proposal-v1/RESULT.md) adopts a separately named 54% initial host threshold, informed by the observed 54% read-only start. All 26 generated controls pass independent review. V1 remains at 55%; later 45%/normal checks, numerical sources, caps and single native attempt remain fixed. A prior v1 sidecar now prevents v2 retry; reverse policy switching after any v2 attempt is forbidden. This source change is not a native result or physical validation; exact committed release review and a fresh host reading are still required.


### October 9: Integrate tiled full-tool vascular scoring with preserved evaluation gates

The reusable tiled kernel now replaces dense contact sets in the private evaluator. Root canonical tests pass 74/74 in 14.37 seconds. Independent coverage passes 74 author/existing plus 19 reviewer controls across two runs; six original reviewer fixture type errors and their targeted correction are retained. Eight complete-report comparisons preserve existing outcomes. Generated admission, size limits, full-history sealing and separate removal semantics remain unchanged. Evidence: [integration record](../artifacts/private-vascular-streaming-integration-v1/source-index.json).


### October 9: Prepare separate retrospective actor and private vascular evidence contracts

The metadata helper preserves frozen IXI roles, requires intended-use QC assertions, retains unknown acquisition times and separates immutable T1 actor metadata from private MRA/annotation/registration evidence. Root tests pass 43/43 in 0.30 seconds; 58 independent candidate controls pass. It performs no image QC, authenticates no caller assertion and grants no execution admission. Header-only receipts cannot substitute for anatomy or registration review. [Integration index](../artifacts/ixi-vascular-metadata-preparation-v1/integration-index.json).


### October 9: Preserve exact trained-network recomputation parity and measured tradeoff

Both trained-network forwards on the same generated 64³ input completed with byte-identical logits and decoded labels, independently reloaded from saved arrays. The single ordered hook-free pair shows a 36,487,168-byte (34.80 MiB, 4.80%) lower sampled peak with recomputation, but 16.73 MiB of that difference predates the forward; peak growth differs by only 18.06 MiB. Forward time increases 13.3%. Both children were reaped and exact downloaders remained live. This is limited numerical/resource evidence, not robust performance, full128 feasibility or patient accuracy. The mistyped comparator invocation is retained as a root-transcribed setup refusal. [Result](../artifacts/recompute-parity-64-v2/RESULT.md).


### October 9: selective paired-scan header preparation

The [IXI265 header reader](../artifacts/ixi265-paired-header-preparation-v2/RESULT.txt) is independently checked using 26 generated controls. Selection is deterministic within the existing labeled TRAIN people. Original buffered read-ahead and stale helper-bytecode failures are preserved; v2 uses exact unbuffered member reads and verified source bytes. No actual archive was opened. All three verified archive bindings and a separately reviewed external launcher are still required; header inspection does not establish anatomy, registration, coverage or training admission. Downloads continue independently.


### October 9: full128 recomputation host deferral before model loading

The independently reviewed single-attempt generated release [deferred at host preflight](../artifacts/full128-recompute-host-deferral-v1/RESULT.md). Seven observations over 30.038 seconds showed normal/warning pressure, 38% minimum available memory and an 11-point spread; swap was flat and both exact downloaders remained live. No child, checkpoint load or forward occurred. This is not a model failure and does not trigger the separate during-forward stop rule. The earlier full128 forward failures remain preserved, with no automatic retry or relaxed resource guard.


### October 9: integrate generated limited-input corridor strategy comparison

The [generated bridge](../artifacts/healthy-corridor-bridge-v1/RESULT.md) connects actor-only imaging/support, exhaustive public SEARCH and scripted IL/RL/HYBRID slots, complete insertion/withdrawal histories, all-method sealing and independent private full-tool contact scoring. Root integration passes 150 tests in 30.83 seconds; independent candidate controls pass 47. Duplicate waypoints, resealed suboptimal search and weakened role assertions were reproduced and repaired, with negative records retained. Two-waypoint batches are joint target/tool selection; the first acquired-data comparison requires one fixed target per batch. This is a geometric exposure query, not tissue penetration, removal, mechanics or learned-policy efficacy. Actual patient admission remains closed.


### October 9: native channel-chunk normalization preserves tiny parity without peak benefit

The [bounded generated operator experiment](../artifacts/native-channel-chunk-instancenorm-negative-v1/RESULT.md) passes six tiny parity/guard controls, including the earlier large-offset counterexample, by using native whole-spatial-map normalization per channel. Both saved PyTorch CPU tensor-allocation timelines nevertheless peak at exactly 12,835,648 bytes; this is neither process RSS nor complete native-workspace accounting. The early transient has no established operator attribution. No trained-network attachment, checkpoint read or full128 retry occurred. Ownership in the pinned tiled network remains unproved, so the prototype stays excluded; the earlier hand-written normalization failure is preserved.


### October 9: complete paired evidence acquisition and inspect one TRAIN image header set

All three IXI archives are [byte-verified](../artifacts/ixi-t1-mra-vessel-byte-intake-v1/completion/RESULT.txt), totaling 17,225,109,877 bytes. A truncated ZIP response resumed its final 9,341 bytes without resetting history; no unresolved IXI objects remain. The [reviewed v2 launcher](../artifacts/ixi265-paired-header-launch-v2/RESULT.txt) preserves earlier review-binding and missing-RSS failures and passes 59 generated controls. One actual IXI265 TRAIN attempt then [completed](../artifacts/ixi265-paired-header-result-v1/RESULT.md): 0.6123 supervised seconds, 141,197,312-byte peak, nine group samples, one reaped child and three independently verified selected-copy hashes. The worker accounted for 21,371,048 archive bytes with zero other-person bodies and no voxel arrays. T1/MRA grids differ; MRA/label form differences are numerically tiny but do not establish registration or coverage. Timing, anatomy/voxel QC and planning admission remain open. Tracto live-transfer failures stay separate and preserved.


### October 9: integration-first shared state, actions, replay and source-image display

The [new human steering](INTEGRATION_FIRST_MULTIMODAL_STEERING.md) prioritizes shared runtime integration and coherent tested milestones while preserving independent scientific validation. The [completed integration](../artifacts/integration-first-shared-episode-v1/RESULT.md) connects aspiration, geometric non-removing probe contact, persistent cavity state, policy/search execution and exact full-tool replay. Root canonical tests pass 13 shared backend, nine bridge and six imaging checks; 282 desktop tests and the exact staged desktop production build pass. Five of 142 selected backend regressions fail on both this source and committed baseline, with receipts preserved; they are stale context fixtures, not newly attributed regressions. Independent review audits all 168 generated replay frames. The actual Mac workflow executes both scripted and SEARCH episodes, with contact/removal semantics visible. Public RESECT Case4 T1 and FLAIR display in separate native frames; stale display metadata and coordinate labels found live were repaired. A CPU MPR check supports consistency with source-reported coordinates but does not independently validate anatomical orientation. The final display session ended at its bounded wall cap with child reaped. No registered patient simulation, mechanical accuracy, mixed-mode trained policy or clinical outcome is claimed. Patient image arrays and model weights remain outside Git; downloads continue independently.


### October 9: actual legacy trained-policy execution through the shared transition

The [third bounded attempt](../artifacts/legacy-shared-rollout-smoke-v1/RESULT.md) safely loads the exact existing RL256 checkpoint and performs two actor forwards, zero new updates. Both policy and observed SEARCH use the same initial observation and native transition; saved histories replay and independent accounting passes. Policy reward is 1.098; beam-2 search chooses STOP after nine of 24 allowed transitions, pruning two negative prefixes rather than exhausting its budget. Historical complete-tree search scored 1.100 but used privileged assessment, so it is not a matched deployment-input comparator. No RL superiority follows. The 1.036-second supervised attempt peaks at 237,879,296 sampled RSS bytes and reaps cleanly. V1's strict NumPy safe-loader refusal and V2's tuple/list context-serialization refusal both precede any forward and remain preserved with their executed sources. No patient inputs, mixed-mode policy, physical response data or new training entered this smoke test.


### October 9: restore meaningful context-aware regression fixtures

The [test-only repair](../artifacts/shared-integration-fixture-repair-v1/RESULT.md) fixes all five reproduced baseline fixture failures without changing production code or guards. Both affected canonical files pass 34 tests in 1.37 seconds. Three missing-context negatives remain explicit; stale/masked actions, incomplete episodes and nonfinite rewards now exercise their intended inner validation. The one existing unit-test optimizer update is not a new training job. Original failures remain preserved; no patient/checkpoint reads or scientific performance claim occurred.


### October 9: stronger matched observed search resolves the STOP diagnostic

One [width-4 generated control](../artifacts/shared-search-width4-generated-v1/RESULT.md) uses the same source, native environment and initial permitted observation as the saved RL256 rollout. With all other configured search inputs unchanged, it retains all four initial prefixes, evaluates 15/24 transitions without pruning or caps, and selects a route scoring **1.100**, versus RL's saved 1.098. It removes the same two target and four other-tissue mm³ with a 10 mm rather than 12 mm complete tool path. The earlier width-2 STOP result remains preserved; its narrower beam discarded useful negative prefixes. The new observed-input baseline is separate from the older privileged complete-tree reference. Independent saved-record audit verifies all 22 motion microsteps, state chains and arithmetic without rerunning search. Supervision completes in 1.444 seconds at 235,945,984 sampled RSS bytes with clean reaping. This is a fixed generated-task diagnostic with zero new checkpoint loads, policy forwards or training, not unseen-patient evidence.


### October 9: one saved multimodal workspace with exact generated replay

The [persistence slice](../artifacts/integrated-workspace-persistence-v1/RESULT.md)
now connects immutable source storage, separate native frames, per-image view
state and backend-owned episode replay through native Save and fresh Mac app
reopen. Canonical checks pass 55 backend/imaging and 310 desktop; the exact staged
desktop builds. Independent review checked tampering, old-bundle state clearing,
private-input separation and the complete asset/replay hydration path. Root
exercised generated auxiliary-selected and visible frame-3 replay restarts, then
public RESECT Case4 T1/FLAIR save/reopen. The 29,960,302-byte real bundle retains
both images and unchanged primary identities; no array is committed. Largest
sampled owned UI tree RSS was 1,132,740,608 bytes; all five sessions exited normally.
A live navigation defect was fixed without weakening planning guards. Camera,
contrast/layout persistence, registered fusion, anatomical orientation review,
patient planning admission, mode-aware learned behavior and physical validation
remain open. No policy update or clinical/generalization claim occurred. The next
shared-system slice is post-seal vascular encounter accounting displayed on the
same instrument episode; private annotations remain outside planning inputs.


### October 9 local / October 10 UTC: continuous intake and bounded recovery handoff

The [preserved handoff](../artifacts/tractoinferno-recovery-handoff-v1/RESTORE_AND_REPRODUCE.txt)
contains exact reviewed recovery source, its unreleased template, immutable
54-object proposal, independent evidence and safe restore instructions. Package
manifest SHA-256 is `8230f1126ca43f86b38cb8ae23202ade9b08060fabf40671b5738c9d70269e75`;
38 compact source/text files total 443,716 bytes before the manifest. An isolated
restoration reproduced 32 authored and four independent generated controls. Three
large source metadata indexes and original terminal/journal/partial state remain
explicit external dependencies; this is not a fresh-clone runnable recovery job.

At 00:35:55 UTC, original PID/PGID 92434 remained live with its lock held:
3,420 files /118,465,170,973 bytes verified, 15 exhausted transport records and
39 historical BrokenPipe refusals unchanged. No origin for every historical
BrokenPipe is proved. Remaining-queue bytes plus 100 GiB reserve and 64 GiB output
allowance left approximately 188.49 GB unallocated at that snapshot. Other
inspected declared queues have completion receipts; no further ready queue was
found. No payload was decoded or admitted by this metadata check.

Recovery was not launched. After the original is terminal, the exact adapter
excludes original successes, retains lifetime attempt history and unchanged
partials, then permits at most one additional verified-TLS/range/version/checksum
attempt per selected object. It holds original/predecessor/recovery locks, refuses
changed evidence or occupied journals and never truncates a partial for ignored
Range. Source, filesystem, checksum and TLS refusals remain closed. The ongoing
original queue was neither signaled nor restarted.


### October 9: ordinal-8 tension specimen finally executes within unchanged limits

The [N12/S60 tension result](../artifacts/hbe-v5-tension-n12-nocache-v2-result-v1/RESULT.md)
completed exactly once at source HEAD `0ece80d179e28f183b379ba2298961d0a2fd2c18`.
Root initial availability was 64%/normal; wrapper initial, post-validation and
immediate-native samples were all 63%/normal. Native FEBio ran 10.568 s at a
118.80 MB sampled group peak; separate readout ran 3.080 s at 198.36 MB. Both
exited zero and children were reaped. Full no-cache coverage was 14 preflight and
28 completed eligible opens, preserving the original failed compression receipt
and its distinct saved-output supplement.

Independent read-only stream replay exactly reproduced 61 frames and all
numerical criteria (largest normalized criterion ratio 0.0008950; minimum sampled
Jacobian 0.886877). This does not establish measured force agreement or positivity
everywhere in the continuum. The sidecar includes nine total native calls,
2,113.422 s charged cumulative time and 2,873,709,647 bytes including supplemental
accounting, within the existing study caps. Original rows, numerical settings,
limits and failures are unchanged. No measured-response or patient data were read.

Ordinal 9 remains unreleased. The old chain verifies the native receipt but does
not authenticate the new v2 policy sidecar/envelope or its wrapper-time/sidecar-byte
charges; a narrow source-bound admission adapter is being prepared before the
next native step. Native twelve-row comparison and physical validation remain open.


### October 9: generated vessel encounters reach the shared Mac replay

The [vascular integration](../artifacts/integrated-episode-vascular-v1/RESULT.md)
connects sealed native history, full-tool scoring, bridge/host validation and the
existing Episode panel. Canonical checks pass 69 backend and 333 desktop; worktree
and exact staged builds pass. Root's live Mac test evaluated a reopened scripted
workspace, verified aspiration/probe/initial/STOP and unknown coverage, executed
SEARCH, confirmed stale-report clearing and evaluated the new search history.
It exited normally after 92.80 s at 724.14 MB sampled process-tree peak; GPU and
transient peak memory were not measured. No patient data or model training ran.

Independent review reproduced a mutable-history alias that could change scoring
geometry after preflight while keeping original bindings. The final detached
snapshot and caller/reference guards repair it; the original negative is saved.
Other retained negatives include the initial output-directory refusal, overbroad
test assertion, mistyped canonical test selector and transient author-side Vite
cleanup failure. Root's final canonical run was clean. Primary source, replay,
policy observations/rewards and workspace persistence remain unchanged. The
report is ephemeral; contact is not removal, injury probability or measured
biological clearance. Patient and physical validity remain unestablished.

At 00:54:10 UTC original Tracto worker 92434 still held its lock and had verified
3,691 files / 130,414,590,611 bytes, up 271 files / 11,949,419,638 bytes since the previous
snapshot. Failure counts remained 15 exhausted plus 39 historical BrokenPipe
records. Recovery remains unlaunched; no download was interrupted. The latest
small metadata receipt is build/acquisition-continuity-20261010/
small-20261010T005411582932Z.json (SHA256 d0ce4d5b2ef44c63aea62da8da930e00767806450f05859e2f1f392940e4ccfc).


### October 9: positive generated checks of the existing critical-constraint consumers

Read-only source review confirmed that both route APIs already resolve the
case-owned critical registry internally. The original missing explicit-mask
argument is not an active wiring defect. The [new canonical tests](../artifacts/generated-critical-consumer-controls-v1/ROOT_REVIEW.md)
exercise actual saved-case load, static/native search, critical bindings and
stale refinement through BridgeRuntime. All three pass. Only the resolver is
replaced by a declared generated resolved result, retaining real admission and
without inventing human review. One static and two native candidates retain
requested geometry but become FORBIDDEN_COLLISION under the exclusion.

These are consumer integration controls. Actual positive real registry
admission/route alteration, critical mask/domain rendering and learned planning
remain separate unmet requirements. No patient payload, model inference or
training was used, and production planning code was unchanged.


### October 9: shared aspiration-only transfer adapter enters canonical source

The [integration evidence](../artifacts/legacy-aspiration-projection-integration-v2/RESULT.md)
records 20 passing root canonical tests and 17 independent controls. Both actor
and search see the same STOP/aspiration projection with native action IDs, all
permitted scan/state channels and explicit full-inventory traces. A probe-exposed
history, external probe mutation or drift in the frozen actor/source/model/tool
binding refuses. Production source matches the reviewed candidate exactly.

Root tests used tiny untrained actors only, with zero checkpoints or updates.
The repaired probe-history refusal and initial independent test setup error are
preserved. This prepares generated transfer from the older RL256 task to the
larger desktop fixture; it does not show that transfer worked. Backend-owned
checkpoint admission, live learned selection and honest reopened provenance
remain the next integration slice. No patient or physical validation follows.


### October 9: public indentation methods clarify the physical-data dependency

The [archived protocol follow-up](../artifacts/menichetti-physical-protocol-investigation-v1/README.md)
compares the creator release and public manuscript with the existing response-free
inventory. All five exceptional trial-column counts agree, supporting the frozen
brain-level grouping without independently certifying donor non-overlap. The
nominal stage ramp and hold are documented, but per-trial sample boundaries,
actual indentation/compliance and force polarity/tare/export preprocessing are
not authenticated. Arithmetic agreement with 10,350 rows cannot supply those
boundaries. Apparatus conventions from a different study are not transplanted.

The scoped independent review confirms methods pages 3–4 and the structural
crosscheck. No protected responses, fitted tables, patient records, solver calls
or fitting occurred. The external dependency is a verified protocol/export
manifest; no outreach occurred. Physical validation remains open, and this
negative source search does not prove that unpublished metadata do not exist.


### October 9: ordinal-9 continuation source and portable controls integrated

The [two-source continuation](../artifacts/hbe-v5-ordinal9-continuation-source-v1/RESULT.md)
is promoted unchanged from independent review. It authenticates ordinal 8's exact
v2 sidecar and charges the 1.1413705407176167-second preparation surcharge and
1,048,576-byte reserve once, then persists ordinal 9's own bound supplemental
ledger. Original solver sources, numerical thresholds, predecessor hashes and
aggregate/per-row limits are unchanged. Immediate pre-launch source/release
checks and final sidecar inventory/cap negatives pass.

Root canonical validation passes 11 tests and 25 subtests in 0.29 seconds.
Portable compact fixtures avoid making ignored output a clean-checkout dependency;
a named local historical-path test retains the real source/Git check where present.
No solver or bulk predecessor stream was read. Exact release, read-only no-cache
feasibility and live host admission remain before native execution.

### October 10: N16 tension completes once and independent replay agrees

The [ordinal-9 result](../artifacts/hbe-v5-tension-n16-continuation-result-v1/RESULT.md)
records one 10.512-second native solve and a 4.379-second official readout.
Independent saved-stream replay took 4.373 seconds and matched the complete
readout exactly: 61 frames, 60 steps, all frozen numerical criteria passing.
The native receipt is `d2476c6355a4c23421c829eff4b9a5db1899b4013b73bacf7e2701eda0def144`;
its continuation sidecar is `656dbcdc6edb69a516c68f0e5ff81ab22c2afe4732c7ad74f89efd595435f8eb`.
Ten native rows are complete, with two still unrun. Physical validation remains
null; the calculated reaction is a simulated specimen response.

Initial template topology refusal and first feasibility import refusal are
preserved alongside the corrected, passing 6.177-second feasibility attempt.
The independent replay child passed and wrote its result before the enclosing
audit command failed on a final bookkeeping import. The stale `started`
supervisor status is preserved and explained, rather than silently corrected.
No solver or saved-stream replay was retried. Original receipts and thresholds
remain unchanged; row 10 requires separate continuation admission.

### October 10: native twelve-row comparator integrated without executing it

The [source integration](../artifacts/hbe-v5-native-comparison-integration-v1/RESULT.md)
passes 91 canonical tests. Independent review confirms unchanged shared math,
six whole-result parity controls and the separate comparator import guard.
The original generated-only entry point and native launcher allowlist remain
unchanged. Native admission binds receipts, readouts, original supplemental
evidence and the full chain before and after comparison. This slice ran no
native process or actual bulk verifier. The original nine-row mapping remains
historical; N16 now makes ten complete rows, with two executions and the complete
comparison still outstanding. Measured-force and patient validation remain open.

### October 10: current probe action is dominated under the removal objective

The [shared-engine diagnostic](../artifacts/shared-probe-decision-audit-v1/REPORT.txt)
compares actual generated state arrays and full normalized candidate inventories.
Probing after shallow aspiration changes contact history, tool state and budget,
but not tissue, cavity, free space or anatomical channels. Deleting both probes
from the scripted history preserves the exact final removed-mask hash and raises
return from 1.109 to 1.295. Current width-2 SEARCH returns 3.353 after 24 branch
calls and hits its cap; no optimum is established.

A separate prospective objective was declared before its demonstration: contact
and retain one public goal cell. Shallow aspiration plus probe succeeds; shallow
aspiration alone misses contact, and deeper aspiration destroys the goal. This
is a hypothetical geometric task, not observed anatomy or new sensor information.
No search, learning or reward tuning used that objective. The 368-preview audit
took 1.624 seconds of worker time, with no model, checkpoint or patient access.
Existing checkpoint/desktop integration remains first; a versioned mode/goal-aware
task and actor would be needed before testing useful learned probing.

### October 10: trained aspiration selector connected across backend and desktop

The [source integration](../artifacts/learned-aspiration-desktop-integration-v1/RESULT.md)
passes 84 canonical backend tests and 372 desktop tests. Working and exact staged
desktop builds pass, preserving unrelated App/neighboring work. Independent
review repaired the architecture literal, source/cache substitution, checkpoint
read bounds, parent-disappearance handling and imported accounting qualification.
The final portable worker executes checked installed sibling files, not ignored
proposal code. The old checkpoint context is preserved before transfer projection.

Actual checkpoint loading remains unexecuted in this source slice. The next
bounded live run must retain both actor and matched restricted-search histories,
followed by native replay and saved/imported provenance inspection. No new model
updates, patient access or scientific performance claim follows from passing
transport fixtures; their fabricated identities remain explicitly test-only.

### October 10: actual trained transfer, matched search and fresh Mac reopen

The [live result](../artifacts/learned-aspiration-desktop-live-v1/RESULT.md) now
exercises the actual fixed checkpoint through Electron at source `ed87de4`.
Actor and width-4 aspiration-only SEARCH choose identical aspiration+STOP IDs,
return 3.353, 4/3 mm³ target/other removal and 17 mm full tool path. The actor used
two forwards and zero updates; SEARCH used 14/24 branch calls without hitting a
cap, but pruned prefixes. No optimum, speed or patient-generalization claim is
made. The entire owned worker took 5.0663 seconds at 266,321,920 bytes sampled peak.

Root inspected the live 72-frame tool replay, post-seal generated encounters and
STOP's no-sweep state, then saved and reopened in a fresh app process. The same
episode and final frame returned with honest unverified imported authorship and
computation. Only one actor attempt exists. Both UI sessions exited zero within
their limits; unmodified screenshots are archived. Independent saved replay
matched the full episode, 72 mask hashes and both decision/inventory chains.
Two reviewer bookkeeping recipe mistakes were repaired using saved JSON only;
their original failures are preserved, with no replay retry.

The original acquisition is still live and untouched. Its
[01:46:35 UTC metadata snapshot](../artifacts/acquisition-progress/20261010T014635Z.json)
records 4,466 verified files / 161,200,201,020 bytes, 541 files / 19,918,213,561 bytes
above the prior snapshot. The 15 exhausted and 39 historical BrokenPipe records
remain unresolved; 52 earlier deferred attempts are separately historical.
Capacity is sufficient; recovery remains unlaunched while the original lock is
held. All downloaded payloads remain unreviewed for intended use.
