# Project status

Updated October 9, 2026. The [full surgical planning/rehearsal supergoal](docs/SUPERGOAL_REAL_OBSERVATIONS.md)
remains active, with the [later human steering](docs/REAL_OBSERVATION_EXECUTION_LEDGER.md#active-human-steering-october-8)
superseding its incompatible real-data-only training restriction. Synthetic data
and simulator-generated RL experience are now permitted, separately labeled from
observed patient records, with held-out real-patient and physical validation still
required. The first fixed generated-development comparison retains its negative
result; longer fixed fits now demonstrate learning on that same task. No new
real-patient training or transfer is claimed.

The [generated-fixture limited-observation boundary](artifacts/limited-observation-boundary-v1/RESULT.md)
is committed and has a scope-limited independent GO: 60 focused and 131
additional controls passed. A sealed two-action plan scores 2 versus 0 mm³
private target removed under a target swap without changing its nominal plan.
Ten wider provisional-support failures reproduce on the frozen base and remain
open. This is not patient admission or evidence of transfer.

The October 9 steering makes privileged-simulation training with **limited
preoperative inputs on unseen patients** the primary falsifiable RL hypothesis.
The [research protocol](EXPERIMENT_PROTOCOL.md) and [master plan](MASTER_PLAN.md)
now state the outer patient split, same-information search/learned comparisons,
and independent patient-linked withheld-anatomy scoring. The [parallel source,
data, baseline and method audits](artifacts/limited-observation-generalization-audit-v1/RESULT.md)
found no admitted full benchmark: current spatial support and feasibility share
hidden-world geometry, real learning admission is limited to one zero-patient
generated fixture, and no current cohort joins the required vascular,
functional and mechanical truth. A small target-swap software control passed;
it is not patient transfer evidence.

Eligible downloads will continue without small-batch review pauses; detailed image,
geometry and anatomical QC is a separate phase. Unreviewed data stays unreviewed.
The execution ledger tracks the complete operative goal and open evidence dependencies.
New commits credit `skamal23 <sayemkamal12@gmail.com>` and co-author
`shafsthegoat <shafrir.p@gmail.com>`. An hourly thread heartbeat is active for
normal pushes of completed reviewed commits to `origin/main`.

The [14.47 GB acquired CT/MRI archive transfer](artifacts/synthrad-verified-download-v1/RESULT.md)
completed one resumable attempt with full SHA-256 and publisher MD5 verification.
An unchanged recovery invocation confirmed `all_complete=true` without a second
transfer. One TRAIN CT/MRI pair has now been extracted for restricted structural
QC; the remaining archive members remain unopened by this workflow. Anatomical
review, training and spatial-planning admission are outstanding.
The [bounded first TRAIN-pair QC preparation](artifacts/synthrad-first-pair-qc-preparation-v1/RESULT.md)
now binds exact archive members, tracked source proofs, strict local resource
limits and restricted review-only outputs. Its 115 offline controls and
independent isolated-metadata review pass. No image member was opened during
preparation. The subsequent [one-pair run](artifacts/synthrad-first-pair-qc-execution-v1/RESULT.md)
completed in 1.342 seconds within resource limits and passed 87 independent
receipt checks. It remains `review_pending`: no visual privacy, anatomy,
CT-scaling or CT/MR correspondence clearance, and no training/planning admission.
The [versioned RESECT QC rerun](artifacts/resect-deferred-qc-v2-execution-v1/RESULT.md)
completed all 24 new image–mask pairs in 24.401772 seconds. Full image scalar,
binary-mask and coded-grid checks now pass; an independent saved-output audit
verified 498 bindings without re-decoding patient payload. Together with the
inherited Case3 structural pass, the denominator remains 25 pairs from 13 people,
with three annotation timepoints missing. Historical failed attempts remain
unchanged. Anatomical validity, scanner-frame registration, training and planning
admission remain unestablished.
The [specimen calibration implementation](artifacts/hbe-branch-calibration-preparation-v1/RESULT.md)
passes 27 owner and 24 independent controls. Exact committed-source preflight
passed, but the first supervised execution stopped on an interpreter-identity
mismatch before measured access or native solves. The failed attempt is preserved;
the [runtime-drift diagnosis](artifacts/hbe-branch-calibration-runtime-drift-v1/RESULT.md)
identifies a Python executable replacement between preflight and launch. A new
versioned run with unchanged scientific settings passed 82 owner and 15
independent controls. Its [committed-source preflight and first bounded attempt](artifacts/hbe-branch-calibration-runtime-v2-preparation-v1/RESULT.md)
then stopped on a real CSV header mismatch before fitting or native solves. The
[independent failure audit and allowed-axial header diagnostic](artifacts/hbe-branch-calibration-v2-failure-audit-v1/RESULT.md)
confirm that compression was read but tension and held-out torsion were not.
The [separately versioned v3 correction](artifacts/hbe-branch-calibration-v3-preparation-v1/RESULT.md)
passed 146 owner and 227 guarded independent controls. Its [first authorized,
supervised execution](artifacts/hbe-branch-calibration-v3-failure-v1/RESULT.md)
parsed both allowed axial curves, then stopped before a completed material fit.
Independent review passed 505 controls and located a real coordinate-coverage
mismatch: the last compression and tension rows exceed the frozen solve range by
3.5215 and 2.2715 µm. No native solve or held-out torsion access occurred.
Physical agreement remains unmeasured. V3 is terminal; any larger solve domain
needs a separately versioned, independently reviewed study.
The [v4 load-domain preparation](artifacts/hbe-branch-calibration-v4-preparation-v1/RESULT.md)
now records exact coordinate-derived endpoints and portable evidence bindings.
It passed 215 focused and 52 independent isolated-input controls. Execution,
fitting and held-out torsion remain closed: the proposed six solves are only a
diagnostic design, while twelve fresh same-endpoint references, native primitive
reconstruction and a supervised resource contract are still missing. No new
measurement or physical-fidelity result was produced.
The [v5 pure deck-adapter preparation](artifacts/hbe-branch-calibration-v5-preparation-v1/RESULT.md)
now freezes twelve same-endpoint reference schedules and checks only declared
FEBio load/time transformations. An initial change-inventory error was found
and corrected; 243 focused tests and a fresh-checkout independent review pass.
Actual source decks, full boundary topology, native primitives and resource
release are still unbound; execution, fit and torsion access remain closed.
Independent scope review confirms that this specimen work cannot by itself
validate patient retraction or cutting forces.
The [RESECT Case4 boundary-6 mesh preparation](artifacts/mechanics/resect-case4-boundary6-preparation-v1/RESULT.md)
proposes one 6/24 mm, curvature-off, source-only Gmsh attempt with unchanged
2 mm bidirectional surface and 3% volume gates. All 176 mesh controls pass,
including independent isolated-source checks. The [single released attempt](artifacts/mechanics/resect-case4-patient-mesh-boundary6-v4/RESULT.md)
returned 19,070 nodes and 10,512 tet10 in 17.63 seconds, but failed both
2 mm surface-distance gates (sample maxima 3.869/2.264 mm; full bounds
4.576/3.254 mm). No solver ran. Prior v1 count, v2 surface-fidelity and v3
timeout failures remain. Case4 comparison landmarks were previously revealed,
so any later FEM displacement score here is retrospective development evidence.
A separately frozen patient and sealed outcomes are required for untouched
evaluation.

The records below retain earlier policy and acquisition checkpoints. A
[scoped generated opening-task learner](artifacts/native-opening-learning-preparation-v1/RESULT.md)
is enabled and independently checked. The [fixed comparison](artifacts/native-opening-learning-v1/RESULT.md)
completed 16 imitation and 16 scratch RL updates: exact search scored 1.1,
imitation stopped at 0, and final RL scored −0.896 versus the initial −0.464.
All 78 simulated episodes passed their geometry audits; this does not establish
useful learning or physical fidelity. Legacy patient/model-loading guards remain.
The [five-state diagnostic](artifacts/native-opening-learning-diagnostic-v1/RESULT.md)
shows that BC's average loss improved while all three tool-label states worsened.
RL sampled every terminal route, including successful openings. The [fixed longer
imitation fit](artifacts/native-opening-bc-capacity-v1/RESULT.md) now fits all five
labels and matches search at 1.1 after 256 updates, with exact update-16 reproduction.
This demonstrates training-task capacity only. [Saved-tree arithmetic](artifacts/native-opening-policy-expectation-v1/RESULT.md)
shows RL16 slightly improved expected reward while reducing target-reaching
probability. The [fixed longer scratch RL run](artifacts/native-opening-rl-capacity-v1/RESULT.md)
now scores 1.098 against search's 1.100 after 256 updates and 1,024 audited
training episodes. Expected return improves to +0.546023, but stochastic failures
remain. First-16 trajectories and optimizer state reproduce exactly. This is
capacity on one generated task. The [two-forward PAT05 diagnostic](artifacts/pat05-forward-diagnostic-v1/RESULT.md)
now verifies compatibility with one prepared TRAIN observation: RL256 ranks STOP
first of 71 actions (81.3695% preference), versus last for the initial checkpoint
(0.9884%). No action or planning endpoint ran, so useful patient transfer remains
unestablished. Tool lengths, image/crop scale, horizon and proposal inventory differ
from generated training. The [length-only diagnostic](artifacts/native-opening-length-sensitivity-v1/RESULT.md)
now isolates one sensitivity: changing four length descriptors alone raises trained
STOP preference from 6.7249% to 99.3915%, while initial weights still prefer movement.
All 311 independent saved-output checks pass. These modified descriptors are not
physically recertified tools; useful patient planning remains unestablished.

The [full eligible annotation intake](artifacts/lausanne-annotation-full-intake-v1/RESULT.md)
completed all 144 masks (13,977,015 bytes). Independent content checks passed for
all 144; the historical grid checkpoint is 116 passed, 27 awaiting original
references and one unresolved grid. Four metadata exclusions and the earlier
sub022 failure remain preserved. No anatomical or training admission is implied.

The [continuous download runner](artifacts/continuous-source-acquisition-v1/RESULT.md)
passes 143 focused controls. The [full frozen 890-file queue is now acquired](artifacts/resect-exact-mirror-acquisition-v1/IMPORT_RESULT.md),
including 24 final masks through exact-checksum mirror transport. The old provider
failures remain preserved; its unnecessary cooldown wait was stopped. Acquisition
receipts establish byte identity, with scientific QC handled separately.
All [Lausanne TRAIN originals](artifacts/lausanne-train-acquisition-complete-v1/RESULT.md)
are now complete: 840 files, 199 people, 210 sessions and 10,020,802,851 bytes.
The [separate QC result](artifacts/lausanne-deferred-qc-execution-v1/RESULT.md)
completes all 26 new images and 27 pending references. Across 420 images, 414 pass
and six earlier T1 conflicts remain. Mask-grid checks now pass for 136 of 144;
eight conflicts remain. RESECT now has 25 acquired image–mask pairs from 13 TRAIN
people; their separate versioned structural QC now passes for all 24 new pairs.
Anatomical and spatial
admission remain unestablished. The [single N36 specimen solve](artifacts/hbe-n36-execution-v1/RESULT.md)
now passes its independently reproduced conditional consistency screen:
0.642702 mN envelope versus 0.723189 mN allowance. It supports temporal review
only; spatial acceptance, physical validation and calibration remain unestablished.
The [N36/S120 increment check passed](artifacts/hbe-n36-temporal-execution-v1/RESULT.md).
Independent reconstruction verified all 121 states: force and probe changes are
1.87025e-12 N and 2.31371e-14 m, within unchanged limits, with no spatial-screen
classification changes. The 1,946.3351-second phase completed within its cap.
The [N24/S120 tension check also passed](artifacts/hbe-tension-n24-temporal-execution-v1/RESULT.md):
all 121 states independently reproduce, with force/probe changes 6.51507e-12 N /
1.41419e-14 m and no changed classifications. It completed in 301.9559 seconds.
A branch-specific one-scale fit and two actual fitted axial confirmations remain
next; physical validation and calibration remain unestablished. The additional
[104-file RESECT originals queue is complete](artifacts/resect-train-originals-execution-v1/RESULT.md):
28 MRI, 17 ultrasound and 59 correspondence files from the existing 14 TRAIN people.
All 672,336,857 bytes passed independent size/MD5/SHA256 checks after a continuous
265.1827-second run. Combined originals now comprise 70 images and 59 landmark files;
cavity labels remain 25 masks for 13 people. Scientific QC remains separate. The [TRAIN compatibility contract](artifacts/btc-train-transfer-readiness-v1/contract.json)
is archived; its two frozen forwards completed without actions or optimization.
The [observed six-landmark desktop API](artifacts/observed-landmark-replay-v1/RESULT.md)
now advances acquired phases/points and reopens bound snapshots. Independent review
cleared transport and packaged-startup repairs. Sparse landmarks still provide no
physical clearance or demonstrated surgical decision benefit; those fields stay null.

The [real-annotation geometry probe](artifacts/lausanne-critical-geometry-probe-v1/RESULT.md)
now detects three source-positive aneurysm cells among 27 queried cells using the
canonical exclusion mask; 24 remain unknown. Independent saved-output review passes.
This is a source-frame kernel check; a bound desktop probe entry and full route
acceptance remain open.

Known incompatible learning, model-loading and SynthStrip support paths now
refuse execution; complete artifact admission and provenance enforcement remain
unfinished. The refusal-only suite passes **42 checks**; independent source review
found and verified repair of two failure-logger ordering defects. No patient
processing, model evaluation or optimization ran in this milestone. Historical
results remain unchanged. See [receipt](artifacts/real-observation-policy-v1/RESULT.md).

Lausanne's CC0 source metadata confirms 284 unique people and 296 original paired
T1/TOF sessions. A fixed split now assigns 199 TRAIN / 43 SELECT / 42 measurement
evaluation people, grouping visits and TopCoW copies. One hundred eleven TRAIN original
pairs from 106 people and their sidecars are acquired (5,086,888,757 bytes).
Published byte fixity passes for all acquired files. Full-pair header/scalar QC
passes for 105 people / 110 sessions; sub253 T1 retains a qform/sform conflict.
All 111 TOF images pass their image-level checks; that does not clear the T1 failure. Independent pilot inspection found TOF
sidecar versus NIfTI orientation differs by about 1.70 degrees,
so scanner-frame provenance and interscan registration remain unresolved. Two
source/deadline software controls pass after a timeout-propagation correction.
See [ingestion result](artifacts/lausanne-original-pilot-v1/RESULT.md) and the
[machine-readable use ledger](manifests/real_observation_data_use_v1.json).
The full TRAIN source index is now frozen: 210 sessions / 840 files /
10,020,802,851 bytes, with nine initial metadata failures retained and resolved
on resume. The bounded intake runner passes 14 focused controls. An acquisition-free
pilot replay and a cache-only resume each excluded all other 209 sessions;
they add no new independent people. See [intake acceptance](artifacts/lausanne-train-intake-v1/RESULT.md).
The first 600-second expansion batch completed one additional person, then
timed out on the next with a resumable 6,291,456-byte partial. All workers exited;
The next batch resumed that partial and acquired four more people in 521.3543
seconds. The third batch acquired sub-007/021/022/450 in 527.2644 seconds;
Those batches left 200 TRAIN sessions incomplete. [The third closed batch](artifacts/lausanne-train-intake-v1/acquisition-batch-03-summary.json)
retains its historical source bindings; subsequent code edits are distinct.
The fourth batch completed five further people in 599.141 seconds; sub077 timed
out with a retained 5,242,880-byte partial and no admission. A separate source-
annotation-led intake completed sub476 in 60.898 seconds. Root and independent
byte/receipt audits reverified all sixteen originals and execution snapshots.
[Batch04](artifacts/lausanne-train-intake-v1/acquisition-batch-04-summary.json)
and [batch05](artifacts/lausanne-train-intake-v1/acquisition-batch-05-summary.json)
retain all excluded cases and separate closure totals. Both timeouts are
preserved. The sixth batch added sub047/051/052/062 in 598.517 seconds and
stopped before starting another worker. All 20 source receipts and snapshots
passed root and independent rechecks; 190 TRAIN sessions remained incomplete.
Its closure lists 192 sessions not reverified within that batch, including two
previously acquired sessions that this separate audit verified. See
[batch06](artifacts/lausanne-train-intake-v1/acquisition-batch-06-summary.json).
The seventh and eighth batches added 13 sessions from 12 new people in
559.201 and 508.879 seconds. Sub074's two visits count as one person. All 33
receipts and originals passed root and independent verification; 177 sessions
remain. Sub077's previous partial is now complete and verified; its timeout
record remains unchanged. See [batch07](artifacts/lausanne-train-intake-v1/acquisition-batch-07-summary.json)
and [batch08](artifacts/lausanne-train-intake-v1/acquisition-batch-08-summary.json).
The ninth batch completed six more people/sessions in495.218seconds, adding
267,508,864source bytes. Root and independent review reverified all39 source receipts. Closure totals are38people/39sessions;171sessions remain.
See [batch09](artifacts/lausanne-train-intake-v1/acquisition-batch-09-summary.json).
Batches10–12 added14sessions from14people. All53closure receipts passed root
and independent rechecks;52people/53sessions/2,342,939,538bytes are complete.
Sub128's timeout was resumed successfully; sub454's latest timeout has no
admission and its partial remains. Historical failure records are unchanged.
See [batch12](artifacts/lausanne-train-intake-v1/acquisition-batch-12-summary.json).
Batch13 added six people/sessions; all59 receipts passed independent and root
checks. Closure is58people/59sessions/2,611,520,354bytes;151sessions remain.
Sub167 timed out; its original failure is retained. See [batch13](artifacts/lausanne-train-intake-v1/acquisition-batch-13-summary.json).
Batches 14–16 added 17 sessions from 16 people. All 76 cumulative receipts,
originals and source snapshots passed root and independent checks; 134 sessions
remain. Sub167, sub186 and sub197 completed after earlier timeouts, whose records
remain unchanged. Batch16 retained a sub208 timeout; later batch17 data are
excluded from its closure. Batches 17–18 subsequently added 19 sessions from
16 people; all 95 cumulative receipts passed root and independent checks.
There are 115 sessions remaining after batch18; its sub251 timeout is preserved.
Batches 19–21 added 16 people/sessions. Root and independent byte/receipt audits
verified all 111 acquired sessions, preserving sub253’s T1 failure separately
from acquisition and its passing TOF. There are 99 sessions from 93 people left
to acquire after batch21; its sub283 timeout is preserved. The twenty-second
serial intake continues with 600 seconds and 536,870,912 source bytes.
Full TRAIN acquisition/use and annotation eligibility remain outstanding. No newly compliant
RL training has run: eligible recorded surgical transitions remain missing.
Before further code edits, a fresh fetch again verified remote/local main at
a4659cc; the working branch contained every main commit (zero behind).
All 82 inventoried unfinished desktop/other files were preserved byte-for-byte.

Source-bound critical annotations now connect to static/desktop routes, native
planning, coarse geometric planning and independent native audits. Replay checks
current canonical exclusions; equal-score beam ties no longer depend on hashes
containing future case history. The focused real-source/refusal/clock suite passes
64 checks and TypeScript checking passes. A subsequent actual sub476 manual
aneurysm component preserves 193 positive labels through canonical exclusions,
save/reopen and desktop inspection. Its explicit source-reference normalization
retains the untouched headers and inferred-unit rationale. The combined suite
passes 77 checks; all 13 component checks pass after the final license binding.
Unknown background, missing motor/language and unknown availability remain
visible. This proves retrospective component behavior; scanner-world accuracy,
brain support and surgical route acceptance remain unresolved. See
[component result](artifacts/lausanne-sub476-annotation-v1/RESULT.md) and
[contract and evidence](docs/critical-structure-evidence.md).

RESECT cavity-component roles are now frozen at14 TRAIN/four SELECT/four
evaluation/Case4 protected DEVELOPMENT; an independent metadata review
recomputed membership and preserved Case4's earlier measurement partition.
The original Case3 ultrasound and source cavity mask are now acquired and
independently verified; their basic pair QC passes. The exact rights text is now archived after
the earlier HTTP429 response. The official revision-2 route is now resolved, and the
bounded Case3 intake runner passes 24 combined controls. The historical rights-only attempt
stopped without a retry. A later deliberate official redirect-chain retrieval
verified all1656rights bytes; an exact-object redirect correction passes13checks
and independent review. The subsequent Case3 scientific acquisition failed before any payload bytes
with Python certificate-chain verification; the failure is retained and
certificate verification remains enabled. A reviewed native macOS trust path
then acquired the exact 9,156,131-byte original with published MD5 verification.
The mask source returned HTTP429; no mask, pair QC or training admission follows.
The 17.266-second partial failure and independent review are preserved. A later
bounded resume acquired the exact 28,518-byte mask in 3.957 seconds. Separate
QC found 20,005 binary positive voxels and a 0.00003068 mm maximum grid difference;
independent checks agree. Fixed native-slice visual engineering inspection is
complete; anatomical review, component fitting and planning admission remain open. See
[intake result](artifacts/resect-component-admission-v1/RESULT.md).
The separately frozen Case4 ultrasound study now has an independently verified
real result: thirteen held-out landmark RMS errors are 4.044969 mm without update,
1.020253 mm with rigid correction and 1.104199 mm with fixed IDW. Six observed
intraoperative correspondences condition the models. All points were retained;
independent quaternion/IDW recomputation agrees within 8.53e-14 mm. All three
stages finished under one second each. This is sparse correspondence accuracy,
not dense tissue/force validation or surgical decision acceptance. Case4 stays
DEVELOPMENT; its thirteen validation points are now consumed for future tuning.
See [observed update](artifacts/resect-case4-sparse-update-v1/RESULT.md).

The measured-interaction candidate is creator-accepted MULTIS donor004 run005;
verified HTTPS/fixity and force/pose/surface pairing remain unresolved.

The HBE half-height comparison now passes all four declared axial cases at
61 states each. Independent readout replay and separate nodal/reaction
reconstruction agree. The supervised solve phase took 40.9184 seconds, with
164,626,432 bytes sampled peak memory and four native calls. This establishes
N8/N12 symmetric-branch numerical equivalence only; the earlier spatial
convergence failure, N24 timeout and missing material validation remain.
The subsequent finer axial experiment completed three individually passing
native runs in 365.799 seconds, with 761.9 MB sampled peak memory. Fine displacement
differences are 4.800/3.639 µm (compression/tension), below 8 µm. One compression
reaction-trend comparison failed; the other 15 criteria passed. Independent
raw replay reproduced all eight readouts (488 states) exactly. The failure is
retained; measured curves and material fitting remain closed. Posthoc diagnosis
suggests slow positive-order force convergence, with uncertain remaining error.
See [finer result](artifacts/mechanics/hbe-halfheight-spatial-execution-v1/RESULT.md).
Saved field inspection indicates growing plate-edge concentration with relatively
stable core fields. Independent review supports a separately declared plate-layer
versus matched-size interior-refinement diagnostic. Its scope is local numerical
sensitivity, not proof of a singularity, total continuum accuracy or calibration
release. Implementation passes18controls and independent review; actual pure
preparation completed in25.635seconds with637.3MB sampled peak memory.
Preparation review and all three native solves passed. Independent raw replay
reproduced 244 states and the complete comparison exactly. The primary plate
versus interior contrast reaches 55.308 µN and 0.657 µm; the local plate sequence
has order 2.0447. This is conditional plate sensitivity, not total spatial error.
The solve took 739.194 seconds with 1.176 GB sampled RSS. Global spatial and
load-step convergence remain unresolved; measured curves remain closed. See [next diagnostic](docs/hbe-halfheight-spatial.md#next-diagnostic-after-the-failed-study).

[VitalDB replay](artifacts/vitaldb-recorded-component-v1/RESULT.md) now imports
21,970 actual pump/cuff-monitor records from one separate DEVELOPMENT person.
All records and seven headers match an independent parser; all 7,207 event
boundaries preserve exact timestamp prefixes. Fifteen focused checks pass and
root reproduced the saved CLI snapshot continuation. The original arterial
request failed because those tracks are absent. Native time origin contradicts
the initial release-relative assumption and remains unresolved; replay preserves
unshifted source timestamps with a null origin. Monitor-record age is not cuff
measurement age. No dosing, physiological accuracy, fitting or RL result follows.
Fresh-checkout cache preparation is now implemented:16focused checks pass,
independently repeated. An isolated missing-cache reproduction using authenticated
local bytes admitted all21,970records through the unchanged reader; the actual
offline cache check made zero downloads. Desktop integration remains open.

NFBS supplies source-reviewed brain-extraction labels with CC0 metadata, but
release-specific BEaST augmentation ancestry remains unresolved. LPBA40 would
require a new research agreement. Neither was acquired or admitted; see the
[source audit](docs/critical-annotation-source-audit.md#brain-support-source-follow-up).

## Historical status through October 5

The entries below retain their original decisions and experiment evidence.
Their proposed future simulated-training work is superseded, not authorized.

Updated October 5, 2026. The three revised specification documents remain the
authoritative requirements and have been read completely. This file records
executed work and open gates, not a replacement plan.

**Current priority: credible planning comparisons and neurological-outcome evidence.**
The rapid-validation steering is preserved verbatim in
`docs/CORE_IDEA_VALIDATION_STEERING.md`. It supersedes the earlier requirement to
finish tissue mechanics before beginning geometric learning experiments. UI,
packaging, broad infrastructure, further mechanics runs and model sweeps are
deferred. Preserve all existing patient assignments and negative results.

The subsequently supplied `docs/MEDIVIS_SUPERGOAL_NEUROLOGICAL_HARM.md` is a
research charter, not evidence that its data/model claims already hold. It extends
the target toward neurological outcomes, robustness, changing anatomy and an
inspectable research interface. Geometric non-target volume stays a baseline;
it is not neurological harm. Separate RHUH, BTC and connectivity outcome audits
are complete; their limits are recorded below. No harm model or reward was added. Claims and
hypothesis decisions are tracked in `docs/MEDIVIS_RESEARCH_TARGET.md` and
`docs/INNOVATION_LEDGER.md`. Current useful work and all patient roles remain intact.

The new permitted target-local and whole-source coarse views pass 39 software
checks, with separate fractional coverage and conserved nominal occupancy. One
source-frozen TRAIN observation diagnostic is now archived: four cases attempted,
three completed, PAT05 failed and PAT16/PAT20 retained historical support blocks.
The independent audit accepted the three saved partial records; the overall run
remains incomplete (exit 1). All three legacy crops already contained the full
nominal target. Target-local views alone reduced shaft-centerline visibility;
whole-source views improved it without changing proposal feasibility. No learning
or clinical accuracy improvement was measured. PAT05's wrapper compares a full
20-field runtime record with an eight-field declaration; its actual runtime grid
values were not retained, so equality is unverified. Preserve the failed attempt
unchanged. A V2 correction now authenticates the complete historical record and
retains expected/actual mismatch details; 155 combined offline checks pass. No
corrected patient run has been released. A separate opt-in ingress helper passes
43 analytical checks after two metadata-label binding defects were repaired. It
screens all six exits and selects by physical distance only after any tool-entry
pose is admissible; full motion validation remains mandatory. The existing access
baseline is unchanged. The separately released PAT25 preparation comparison now
completed in 36.754634 supervised seconds, with sampled process-group peak
2,609,086,464 bytes. All six existing exits received 468 static checks; five were
eligible. The fixed distance/axis/sign rule selected axis 2, sign -1. The original
inventory reproduced exactly with 0/78 admissible previews; the selected access
had 78/78, including 39/39 for each tool. Both crops retained 100% of supplied
target mass. All 468 static poses still carry outside-image and normal-tissue
exposure unknowns; deepest-shaft centerline coverage decreased. The frozen
saved-record audit passed on its first run, with 75 bound inputs and all seven raw
outputs unchanged. No tissue was removed and no policy ran. This is a TRAIN
preparation improvement informed by the earlier failure, not independent patient
generalization or a learning gain. See `artifacts/pat25-ingress-access-v1/RESULT.md`
and the unchanged `artifacts/training-observation-coverage-v1/RESULT.md`.

The opt-in access preparation is now reusable without changing the default
shortest-exit constructor. Its shared derivation was moved unchanged; 81 focused
owner, independent and legacy compatibility checks pass. It returns a selected
source without constructing another full inventory or invoking a policy, and
requires the upstream subject-specific source binding. The first four-patient
comparison used it successfully for preparation, with70/76/78/70 initial accepted
previews on PAT05/PAT22/PAT25/PAT28. Only PAT25's access changed; all four crops
retained the supplied target. See `docs/native-access-preparation.md`.

PAT05's historical access metadata now has an explicit adapter: all saved fields
and the executed twenty-field grid remain exact, while previously absent exit
boundaries are labeled newly derived. Its 123 focused controls pass, and actual
PAT05 rederivation now succeeded in that preparation. A separate opt-in planning budget counts
native preview entries across search and replay under one clock; 53 controls pass
after inherited-method cleanup and exact-deadline admission defects were fixed.
The first comparison completed in339.944 seconds but all12 methods failed before
independent evaluation: direct Python equality confused saved JSON lists with
native coordinate tuples. Zero method outcomes were accepted. Both historical
support blocks remain, and exact V1 histories/sources are archived. An independent
saved-record audit confirms the failure disposition and preparation evidence.
The narrow whole-record canonical JSON correction passes61 checks, including
actual tiny native STOP/removal episodes and independent audits. A separately
released V2 completed in 383.646 root seconds with unchanged patients, checkpoint,
task and budgets. All twelve episodes passed independent evaluation; the separate
saved-record audit passed on its first execution. Four comparisons completed out
of six prescribed TRAIN patients, with PAT16/PAT20 still support-blocked. Frozen
imitation versus greedy returns were 290.506/410.312 on PAT05, 672.297/789.097 on
PAT22, 584.914/753.905 on PAT25 and 493.534/612.538 on PAT28. STOP returned zero.
Search wins all four geometric-return comparisons; the frozen policy uses 140
online previews per case versus 264–280 and 16.55–22.35 online seconds versus
30.77–42.62. These are single fixed-order measurements, excluding separately
reported shared preparation and independent audit. The CNN's fixed crop and
search's full nominal fields remain a representation difference. No optimization
occurred; all final policy hashes matched. PAT05 is the training case and the
other cases are inspected development patients, not untouched validation.
All twelve complete metrics/history records match V1 exactly; decision records
also match apart from timing/status fields. The correction changed evaluation
acceptance, not behavior. Original failures remain
unchanged. See `artifacts/prepared-training-planner-comparison-v2/RESULT.md`.

A post-hoc saved-record diagnostic independently reproduces the frozen model's
initial greedy-action ranks: eighth of 77 legal actions on PAT22 and fourth of 71
on PAT28. Action-distribution entropy is 99.715% and 99.687% of the uniform maximum;
executed choices remain deterministic argmax. Later steps account arithmetically
for 91.61% and 83.20% of the total reward gaps, but their states diverge after the
first action, so this is not causal attribution or matched-state regret. The
original five-patient denominator and blocked/STOP-only outcomes remain intact.
See `artifacts/frozen-transfer-decision-diagnostic-v1/RESULT.md`. No new training
or patient-array processing was needed for this diagnostic.

The isolated, signed native public-transfer client passes a version-only smoke
check. The checksum-only helper has 37 passing offline controls; three lifecycle
review failures were repaired and retained. After a separate source-bound release,
one public checksum transfer succeeded in 5.324132 seconds with 61,787 bytes and
parent acceptance. All 12 selected image paths occur in the 720-path index, but
the file gives neither payload lengths nor a checksum algorithm label. That
checksum-only run transferred no patient image.
The combined observation and acquisition integration check passed all 109 focused
tests in 4.81 seconds. See `artifacts/rhuh-checksum-transfer-v1/RESULT.md` and
`artifacts/validation/observation-acquisition-integration-v1/verification.json`.
An independent saved-record audit accepts the received checksum metadata. The
documented link-listing API did not establish compressed target lengths. A
separate prospective single-T1 acquisition experiment has now completed after
its own reviewed preparation, source archive and committed execution release.
The shared transfer helpers passed 86 focused controls, including all 37 original
checksum tests. One invocation received 6,337,221 compressed bytes in 11.450222
supervised seconds; the fixed MD5-over-compressed-bytes candidate matched the
published token. The independent audit reconciled the original bytes, release,
23 archived files and parent/worker receipts. Original bytes remain read-only in
local quarantine. This establishes byte-domain compatibility for one file;
the publisher's algorithm and prior length remain unconfirmed. A separate bounded
format inspection has now completed after 153 software checks and its own committed
release: 240 x 240 x 155 scalar float32, declared 1-mm spacing, all 8,928,000
scaled values finite, valid single-member gzip/NIfTI-1. The finite invertible
qform declares L/P/S voxel axes; sform is absent. This does not establish native
scanner provenance or anatomical registration. Execution took 0.328035 seconds
at the root, with 192,675,840 bytes sampled combined peak memory. The independent
saved-result audit passed without re-decoding. A separately released fixed-plane
display then completed in 1.696 seconds, with sampled worker peak302,055,424 bytes.
Both root and another reviewer inspected the saved views; a separate saved-result
audit reconciled all source/output bindings. The three sampled planes are readable,
but visible tissue approaches/touches the inferior image edge and full anatomical
coverage remains unresolved. No anatomical registration, mask, injury label, case
import or training use was accepted. The original 12-file declaration stays unmet
and unchanged. See `artifacts/rhuh-fixed-plane-qc-v1/RESULT.md`, the earlier format
inspection and acquisition records.

The RHUH public table has 40 patients and 14 recorded postoperative deficits
(six transient, six minor persistent, two major persistent); 26 are recorded `No`.
Exact clinical timing, affected domain, baseline focal deficit and category
definitions are absent. It supports a small observational baseline, not route
risk or new surgery-caused deficit. The public processed imaging inventory has
720 names. One T1 payload now has accepted byte/provenance and binary-format
checks; anatomical alignment, ADC units, segmentation semantics and injury labels
remain unvalidated. BTC has five candidate TRAIN
follow-up pairs in metadata, but individual cross-release linkage remains
unverified; no outcome rows or protected content were opened. The independent
63-patient Figshare release has complete paired matrices and cognitive scores,
but its supplied T0 impairment count is 42 versus 30 in the paper, and connectivity
preprocessing pooled pre/post sessions. These limitations preclude a prospective
imaging-pipeline or route-harm claim. Root independently reproduced the two public
tables' patient counts and outcome counts. See `docs/rhuh-outcome-audit.md`,
`docs/rhuh-postoperative-imaging-audit.md`, `docs/btc-longitudinal-outcome-audit.md`,
`docs/glioma-connectivity-outcome-audit.md` and
`docs/neurological-outcome-inference-limits.md`. All source values stay unchanged.

The isolated RHUH outcome adapter now passes 53 owner/independent controls.
It preserves raw labels and unknown timing, while its permitted preoperative
projection contains only KPS and contrast-enhancing volume and excludes the mixed
source hash. Postoperative-field changes cannot alter that projection or its
identity. The separately committed 40-patient leave-one-out protocol and execution
release have now completed one run: all 9,092 declared logistic fits, 99 label
permutations and 14 positive-patient deletion analyses. Adding enhancing volume
to preoperative KPS worsened the primary Brier score by **+0.00120647**; the combined
model detected only 3/14 recorded deficits at the fixed threshold. The independent
saved-record audit reconstructed all 13,638 predictions across 4,546 folds within
the fixed absolute 1e-12 tolerance, without refitting. All 3,660 bound inputs and
13 raw outputs stayed unchanged. Supervision took 13.897940 seconds with sampled
peak process-group memory of 125.890625 MiB. No external or clinical calibration
validation was performed. These covariates are constant across alternative routes
and cannot rank them; no planner reward was added. See
`artifacts/rhuh-preoperative-baseline-v1/RESULT.md`,
`docs/neurological-outcome-contract.md` and `docs/rhuh-preoperative-baseline.md`.

The task is annotation-assisted target access with at most three decisions
(a certified native tool stroke or STOP). Initial learning used TRAIN patient
PAT05; the same declared task has now been compared on the other five TRAIN
cases with a frozen checkpoint and no adaptation.
Observations contain the permitted preoperative image, supplied target annotation,
provisional tissue support, observed cavity and tool geometry. Supplied annotation
is an input in this track; this is not scan-only inference. Success requires
independently checked positive target removal, with absolute target and other
tissue volumes and full-target fraction reported. It is not whole-target removal,
validated tissue damage, surgical usefulness or patient generalization.

The first executed reachability check completed in 26.21 seconds without an
environment change: one initial motion then STOP removed 175.0006 mm³ of target
and 121.0004 mm³ of other tissue, with geometric reward 150.7035. Independent
full-tool/frontier/containment checks passed, with zero unsupported volume. This
is 1.5301% of the 11,437.0397 mm³ supplied target. The current action catalog does
not establish full-target coverage. No optimizer update occurred in this check.
The first complete learning comparison then ran in 205.84 seconds supervised
(203.77 seconds in the worker), peaking at 1.91 GB. The existing 30,827-parameter
spatial policy received two real REINFORCE updates from four complete on-policy
PAT05 episodes. All ten completed episodes passed independent full-tool and
reward/tissue-accounting checks, with zero invalid action attempts. An independent
saved-record calculation also agreed. Weights changed, but the fixed-latest
argmax plan did not: target 17.0001 mm³, other tissue 20.0001 mm³ and geometric
return 12.7030 both before and after training. This is a negative learning result,
not evidence that the algorithm can never learn.

Random legal returns were 169.3756, -25.0131 and 246.5479. Same-horizon greedy
search completed in 17.40 seconds plus 20.09 seconds execution/audit, with the
shared 8.18-second preparation reported separately. Its accepted plan removed
493.0017 mm³ of target and 412.0014 mm³ of other tissue, return 410.3124. Target
fraction is 4.311%; this is not full resection. Search is stronger under this
fixed geometric objective. The network's subsecond decision time excludes the
required candidate/tool previews and cannot establish an end-to-end speed gain.
See `artifacts/pat05-real-geometric-learning-v1/RESULT.md` and the saved curve.

The separate fixed eight-update imitation experiment also completed: 63.94
seconds supervised, 1.90 GB peak, the same original untrained weights and the
same three-action task. The teacher was replayed exactly and independently
checked before training. The final policy changed its actions and reached
return 139.8975 (+127.1944 over initial), with 173.0006 mm³ target and
164.0006 mm³ other tissue removed. Its plan independently passed; this is a
same-patient imitation gain, not an RL or transfer gain. Teacher cross-entropy
fell from 4.140585 to 4.097359, while the policy still trails search by 270.4149.

The first learned action returns 147.3035, close to search's 150.7035. Its second
and third actions each remove 18 mm³ of other tissue and no target, returning
-3.7010 and -3.7050. Those later choices account for 98.74% of the total gap.
This localized the failure without proving its cause. See
`artifacts/pat05-real-geometric-imitation-v1/RESULT.md` and its comparison plot.

The subsequent matched visited-state experiment completed in 160.40 seconds
supervised (158.02 seconds worker), at 1.86 GB peak. Both branches independently
restored exactly the BC8 weights and Adam state, then used eight additional
updates and six examples per update (48 training forwards each). The control
repeated the original three examples; the augmented branch added three visited
states, with five unique observations across its six weighted examples. All
four complete episodes and a separate saved-record calculation passed.

Control return rose to 239.1058 (286.0010 mm³ target / 233.0008 mm³ other tissue).
Augmented return reached 290.5060 (345.0012 / 271.0009 mm³), a matched difference
of +51.4002. Both chose the same first action; augmentation improved the later
choices and all three of its actions had positive incremental return. This is
support for added state examples in one deterministic pair, not replicated
causal evidence or an RL/generalization gain. Absolute other-tissue removal
increased by 38 mm³; the result is not a clinical safety improvement. Search
still leads at 410.3124. Obtaining the new labels cost 7.85 seconds within the
32.29-second collection phase, additional to prior teacher-generation costs.
See `artifacts/pat05-real-visited-imitation-v1/RESULT.md` and its comparison plot.
Eight focused guard/isolation checks passed; the prior integrated five-file
verification passed 81 checks. No SELECT or unopened patient was accessed.

The first comparison on the remaining five TRAIN patients then completed with
zero optimizer updates: 171.484 seconds summed supervised time, peak sampled RSS
1.802 GB. The frozen visited-state checkpoint, greedy search and one diagnostic
random episode used the unchanged task. PAT16/PAT20 retained their 19/125-cell
support conflicts. PAT22's accepted primary pair scored 672.297 for the policy
(697.998 mm³ target / 127.000 other) and 789.097 for greedy (819.998 / 153.000).
PAT25 had all 78 proposals shaft-blocked; every method correctly selected STOP,
with zero removal and zero score. All three prepared cases saw the full supplied
target in the fixed 64³ crop.

PAT22's random episode and both PAT28 primary episodes failed the independent
accounting check. Their outcomes remain null, not retrospectively accepted. Saved
per-action versus aggregate target/other volumes differ by small amounts; source
inspection and analytical regressions identify float32 reduction/product errors.
PAT28's random episode passed. A standalone independent check confirmed six
accepted episodes, three retained failed episodes and two complete primary pairs
out of five prescribed patients, one of them STOP-only. The original source,
records and failure evidence are losslessly archived. See
`artifacts/remaining-training-frozen-spatial-v1/RESULT.md`.

The narrow double-precision repair now passes 77 focused controls and eight
independent precision controls without changing audit tolerance, weights or
geometry. A separately declared precision-only repeat completed in 179.283 seconds,
at 1.585 GB peak sampled worker RSS. All nine episodes across the three prepared
patients passed independent geometry and accounting; PAT16/PAT20 remained blocked.
PAT22 frozen/greedy returns were 672.297/789.097; PAT28 were 493.534/612.538.
PAT25 remained STOP-only with no target access. Three primary pairs were accepted
out of five prescribed patients, with only two exercising non-STOP choices.
Independent saved-record comparison confirms identical actions, observations,
policy outputs and complete geometry/cell histories in both attempts. The largest
return arithmetic change was 0.000026317. The original three rejected outcomes
remain rejected in their original records and lossless archive. Root verification
checked all 105 repeat archive members against local originals and all 12 indexed
report artifacts. See `artifacts/remaining-training-frozen-spatial-float64-v1/RESULT.md`.

Next: diagnose access and proposal coverage using those saved records before
changing the task or training further. Initial accepted proposal envelopes omit
10,186/13,915 target centers on PAT22 and 7,509/10,269 on PAT28; the full supplied
target is visible in the actor crop. These are action-coverage limitations.
The saved-record diagnosis now identifies narrow lateral sampling: no target
center exceeds the accepted axial bounds on PAT22/PAT28. PAT25's central opening
also fails shaft clearance only 0.5 mm inside the selected access. The first-exit
access heuristic never checked entering-shaft clearance or connection to external
free space. The subsequent six-existing-exit diagnostic completed 468 initial
previews in 106.112 seconds supervised, at 960,036,864 bytes peak RSS. Accepted
counts in x−/x+/y−/y+/z− order were 0/78/62/58/78/72. All 78 original failures
occurred before insertion: the backward shaft re-entered estimated tissue behind
the locally free access. The original selected inventory reproduced exactly.
All six outward neighbors were externally connected, so that local condition
alone cannot establish shaft clearance. Two alternative accesses also hide every
target cell from the current actor crop; a third omits 22 cells. Initial preview
availability is not useful policy visibility, target removal or clinical access.

Independent review reconciled all 468 trace records, six inventories and 61 source
hashes. No cuts, policy calls, training or replacement-access selection occurred.
The run has an explicit launch-order deviation: a saved-log whitespace check
stopped the intended commit/archive block, but a subsequent shell line still
started the reviewed source-hash-bound diagnostic. The source archive and commit
were made after completion and are labeled accordingly; no retry hides this
deviation. The evidence supports a working-tree development diagnosis, not a
precommitted immutable-archive launch. See
`artifacts/pat25-six-exit-access-diagnostic-v1/RESULT.md` and
`docs/frozen-transfer-access-diagnosis.md`.
Population training, repeated-seed advantage, protected-patient transfer and
physical or neurological fidelity remain unproved.

A metadata-only population-readiness check found existing structural-evidence
bundles for all six TRAIN and both SELECT cases; no acquisition is needed for
this bounded annotation-assisted track. It did not decode any additional patient
arrays or perform a fresh whole-bundle byte audit. PAT16/PAT20 remain genuine
factory rejections: their frozen main envelopes omit 19/125 annotated target
cells. Keep these TRAIN failures in the denominator; do not intersect labels or
expand support silently. PAT22/PAT25/PAT28 and SELECT PAT26/PAT27 have prior
zero-omission receipts, but still need per-case support/access, native-grid and
proposal-coverage bindings before new comparisons. The later five-case attempt
and precision repeat above provide explicit executed dispositions; the earlier
metadata-only readiness check itself supplied no transfer evidence.

**Deferred mechanics preparation:** half-height source components are committed;
34 preparation checks and 35 readout checks passed (separate focused suites,
not a combined repository regression). No actual half-height mesh or solver run
was released. The metre/millimetre meshing control remains unfinished with one
owner assertion failure and one independently observed nonfinite-input guard
failure. It was not executed natively and must not be described as validated.

**Earlier mechanics checkpoint (preserved):** the repaired optional solver passed all eight
numerical controls and independent saved-output replay. The graded real-patient
mesh failed the unchanged 2 mm surface-fidelity gate and remains excluded from
solves. The fixed specimen experiment completed 18 numerical cases in 396.3 seconds
but failed mesh-convergence gates: axial probe motion differed by 22.36/23.26 µm
against an 8 µm limit, and tension reaction differences did not decrease across
the three meshes. Step and scale checks passed. Calibration and held-out response
curves remain closed; the two fitted cases did not run. The saved-geometry diagnostic reproduced
its prior surface errors: all 732 corners lie on the saved anatomy surface, but
32 straight midsides and 25 face centroids exceed 2 mm. This supports loss of
local shape between corners; no patient mesh or withheld-motion comparison is
accepted. The diagnostic took 2.90 seconds and left all inputs unchanged. See
`artifacts/mechanics/resect-case4-saved-mesh-diagnostic-v1/RESULT.md`.

Independent specimen diagnosis reproduced all 75 endpoint probes within
5.85e-14 m using a separate interpolation calculation. Shared physical nodes
still differ by 25–28 µm, supporting unresolved spatial discretization. The
original failure is retained. A separate four-run N16/N24 axial resolution
extension prepared both meshes in 5.69 seconds: 7,209 and 23,101 nodes. Independent
reconstruction passed both geometry reports and verified the frozen solver
decks. The separately released four-case study then stopped at its second case:
N16 compression passed in 88.59 seconds; N24 compression timed out at the fixed
420-second limit, with 36/60 steps confirmed. Neither tension case nor either
required mesh comparison ran. Supervised elapsed time was 517.52 seconds and
sampled peak RSS 1,491 MiB. All partial outputs and previous evidence are retained;
there was no retry, measured-response access or material fitting. The completed
case reported 60.78 seconds in its linear solver. Saved-mesh checks rejected
exact Cartesian quarter extraction: tiny coordinate departures exceed its
declared tolerance. The separate midheight plane passes that same geometric
criterion on both N8/N12 meshes. A half-height axial equivalence comparison has source preparation only and is
now deferred; no reduced model is accepted or released. See
`artifacts/mechanics/hbe-quarter-eligibility-diagnosis-v1/README.md` and
`artifacts/mechanics/hbe-resolution-experiment-result-v1/RESULT.md`.

One analytical Gmsh control confirmed that the existing discrete-surface path
supports curvature sizing. A joint curvature-24 / minimum-3-mm profile reduced
bidirectional covering bounds from 5.302/4.935 mm to 0.936/0.920 mm on a fixed
ellipsoid triangulation, with node counts increasing from 269 to 2,806. It took
2.795 seconds and used no patient inputs or physics solver. This is a framework
capability check, not patient validation or a patient resource prediction. A new
patient candidate using that profile passed 95 focused preparation checks,
including independent review. Its one actual patient attempt then stopped at
180.028 seconds before returning a volume mesh: boundary-curve meshing alone
took 170.079 seconds. The exact terminal cause is a final RSS-sampling timeout
with 11.74 ms remaining, recorded as a supervision exception. Counts and shape
fidelity are unassessed, not failed geometry tests. All 27 inputs and both prior
attempts remain unchanged; no retry or solver call occurred. See
`artifacts/mechanics/resect-case4-patient-mesh-curvature-v3/RESULT.md` and
`artifacts/mechanics/gmsh-discrete-curvature-capability-v1/RESULT.md`.

**Earlier steering (October 4): real-patient learning and probability-aware planning.**
The user supplied `docs/CODEX_STEERING_PROMPT.md` and
`docs/DATASET_ACQUISITION_ADDENDUM.md`; both were read completely and copied
unchanged from the supplied files. They supersede the proposed procedural
training screen. No synthetic population screen or opening-learning run was
executed. Existing analytical fixtures remain software checks,
not human training data. The immediate milestone is one real patient with
independently checked alternative plans and nonzero, evidence-bound functional
uncertainty. Population training must use existing real patients, preserve the
BTC TRAIN/SELECT/unopened roles, and assess transfer on separate patients.

Annotation-assisted and inference-only tracks stay distinct. Reviewable supplied
segmentations are valid inputs in the former; private references cannot influence
inputs or proposals in the latter. Remaining gates include broader executable
route coverage, a fully exercised patient desktop workflow, and actual
real-patient spatial learning with independent-patient evaluation.
Search may remain the strongest research mode;
no learned-policy superiority is established. The unrun procedural runner also
exposed a weak teacher on its development fixture; that negative finding is
preserved and will not be used to justify an RL advantage.

**Latest executed results.** Source-bound functional evidence now persists through
native refinement, bridge run configuration, checkpoints and replay. The real
UCSF-PDGM-0004 prior bundle was attached, saved and reopened in 9.12 seconds;
the original case and seven prior proposals remain unchanged. This derivative
uses unreviewed population priors with uniform ±2 mm / ±1° sensitivity only.
Atlas sampling covers 1,431,949 of 1,432,027 modeled tissue cells, with 78 unknown;
this does not establish patient functional coverage or registration accuracy.
The structural mirror's equivalence to official TCIA bytes is still unverified.
See `artifacts/functional-evidence-integration-v1/real-case-roundtrip.json`.

The independent full-tool event evaluator passes 68 combined controls. The
executed real-case comparison completed eight frozen panels in 28.984 seconds
(worker peak 2.108 GiB). Both retained native histories passed fresh geometry
checks. At map support 0.5, motor encounters were fine 51/64 versus wide 64/64
under Gaussian SD 1 mm / 1°, changing to 43/64 and 53/64 under 2 mm / 2°.
Mean and tail exposure costs increased with broader uncertainty despite lower
binary counts. Missing channels produced 64 unknown outcomes, never zero risk.
These are two instruments on the same axis, not two independently discovered
surgical approaches; three other rays remain rejected for native execution.
An independent saved-record audit recomputed counts, intervals and cost summaries
across 1,344 outcomes. See `artifacts/functional-sensitivity-realcase-v1/RESULT.md`.
This remains unreviewed population-map sensitivity, with no postoperative deficit
probability, new-patient generalization or complete-resection claim.

The local `evaluateCandidate` operation now seals a selected native history
before revealing its predeclared final worlds. Interrupted evaluation can resume
the same assessment; optimizer training cannot resume after sealing. Reopen,
replay and export verify the same full-sequence events. All 87 focused integration
checks pass; the current facade has only three final worlds. The latest full
Python regression from committed `4cea5c4` passed 2,433 checks in 278.04 seconds,
with six skips and 14 retained warnings. Five skipped DICOM controls passed in
the separate existing acquisition environment (21 checks), for **2,438 unique
passes, one unavailable historical fixture and zero failures**. All 3,778
archived source files and the consulted UCSF bundle stayed unchanged. Sampled
peak process memory was 922.6 MB; no limits were exceeded, repairs made,
dependencies installed or tests retried. This is code verification, not a
clinical or mechanics accuracy result. Receipts:
`artifacts/validation/python-regression-4cea5c4/attempt-01/`.

The fixed-scene discretization control completed all 16 numerical rows. Five
motions were shaft-rejected without state changes; ten accepted histories passed
independent checks and the last audit reached its declared cap. At 1 mm, phase
or orientation alone changed acceptance. Even at 0.125 mm, credited physical
removal remained 13–16% below the analytic reference. Independent audits took
152 seconds versus 17 seconds in native strokes. These are numerical model
limitations, not patient training or clinical accuracy results. See
`artifacts/native-discretization-control-v1/RESULT.md`.

The real spatial adapter now accepts an explicitly bound, provisional research
envelope without promoting expert-review status. The first PAT05 profile from
committed `8aeb3be` failed at the native orthogonal-affine guard in 3.148 seconds,
before candidate previews, policy calls, transitions or optimizer updates.
The original bundle and all 53 numerical source files stayed unchanged; there
was no retry. Its fixed 32³ crop contains 24.36% of the supplied target and also
limits proposal reach. The independently checked, explicitly bounded correction
preserves source images and moves full-cell corners by at most 1.0842e-7 mm.
The separately frozen v2 profile completed in 9.311 seconds at 1.063 GB peak
sampled RSS: 36 initial legal actions, three fixed test actions, three untrained
forwards and no updates. Its full-history geometry audit passed. The shallow
test sequence removed 26.00009 mm³ normal tissue and no target; its actions
were chosen by inventory order, not the policy. Crucially, 10,558 of 11,437
target centers (92.31%) exceed the fixed catalog's distal axial bound. Broader
shared proposals and observed context must be checked before useful learning.
The original v1 failure remains preserved. See
`artifacts/pat05-real-spatial-profile-v2/outcome.json`.

The same frozen SynthStrip main model completed all four additional BTC
development estimates (TRAIN PAT22/PAT25 and SELECT PAT26/PAT27) in 35.024
seconds, without retries, parameter changes or source mutation. All four outputs
were fixed before annotation-overlap checks. All four proposal bundles now save
and reopen with unchanged source arrays, patient roles and planning hashes.
Independent native-grid and distance-to-mask reconstruction checks passed, with
zero omitted annotation cells; 24 sampled planes received nonexpert visual QC.
The estimates remain unreviewed and working support remains absent. No training
result follows from these inferences. PAT29/PAT31 and UPenn remain unopened.
See `artifacts/brain-extraction/BTC-spatial-main-v1-independent-qc/README.md`.

Shared observed-only search and real-patient batch bookkeeping now pass 51
focused checks, including six independent controls. Search preserves costly
opening prefixes, reports incomplete layers, and uses optional actor guidance
only to order legal actions. Bounded beam retention agrees with full-state
materialization across 430 budget/width controls. Independent review reproduced
and repaired two exceptional-exit accounting errors; the original failures are
retained. Six-TRAIN/two-SELECT roles and patient-uniform imitation sampling are
checked without opening images. These helpers performed no patient search or
optimizer updates. Full-field search and cropped actor inputs remain different
representations. See `docs/real-patient-learning.md` and
`artifacts/real-learning-helpers-independent-review-v1/receipt.json`.

An opt-in common nominal/cavity proposal generator now offers exposed openings
and proximal/distal target endpoints across the existing 13 source columns.
It uses permitted annotations/estimates and observed cavity state, keeps engine
target labels zero, and requires a fresh full native preview for every emitted
unique ray. Endpoints are no longer clipped to the actor crop; visibility and
bounded-family coverage remain separate from geometry acceptance. Seven
independent controls pass, including hidden-reference swaps and paid-opening
sequences. Three legacy states/two transitions match the original default
exactly. The matched PAT05 profiles below measure its initial proposal bounds.
See `artifacts/nominal-cavity-independent-review-v1/provider-receipt.json`.

Optional lazy planning transitions now defer successor inventories until a
retained search branch is expanded. Current-action certification, state,
reward and history remain identical; the ordinary policy transition stays
eager. Independent native/search checks pass (61 combined controls), including
both proposal modes, hidden references, cancellation and exact route parity.
Fewer previews were measured on analytic code fixtures; patient latency and
training benefits remain unmeasured. See
`artifacts/native-lazy-planning-review-v1/receipt.json`.

The first complete-history scalar/batch comparison ended as a harness failure
after its initial scalar geometry audit passed. The source/configuration and
97-microstep history matched the prior numerical control, and all 3,762 archived
files stayed unchanged. An in-memory empty tuple and its JSON empty-list
representation caused an exact certificate comparison to fail. Batch and the
second scalar phase never started; no speedup result follows. Original limits
and failed receipts are retained, with no retry. See
`artifacts/independent-native-batch-history-v1/RESULT.md`.

The separately declared V2 repaired only tuple/list certificate serialization
and completed one scalar/batch/scalar comparison: 43.007 / 4.016 / 43.176 s.
All certificates match the historical result and complete trace payloads are
byte-identical (97 prefixes and 679 ordered geometry queries per phase). The
observed audit-time ratio is 10.71–10.75 on this one analytic history, with
303.2 MB peak memory; this is not a patient or universal speedup estimate.
All 3,804 archived files remained unchanged; independent saved-output review
found no discrepancy. The production default remains scalar. This neither
resolves the source-grid removal deficit nor validates tissue mechanics. See
`artifacts/independent-native-batch-history-v2/RESULT.md` and its independent review.

The matched 64³ PAT05 proposal profiles completed in 16.35 s (fixed lattice)
and 31.58 s (nominal/cavity). Both crops contain all 11,437 supplied target
centers; initial legal actions increased from 48 to 70. The broader accepted
sweep bounding box still excludes 9,124 target centers, versus 10,742 for the
fixed lattice: these are geometric bounds, not measured reachable volume.
Both three-action, first-inventory-order histories passed independent geometry
checks but removed no target. They removed 26 and 3 normal source cells,
respectively. These probes performed no search or learning and do not compare
policy efficacy. All source/input hashes and recorded weights stayed unchanged;
the saved-result audit found no blocking discrepancy. See
`artifacts/pat05-real-spatial-profile-v3/outcome.json` and its independent review.

The ensuing bounded PAT05 SEARCH/untrained-policy comparison failed at SEARCH's
180-second internal limit after 227 transitions, with two complete layers and
a partial third. Its provisional nominal score was never replayed or audited;
the untrained policy remained unexecuted. There is no completed method comparison
or validated route from this run. Parent runtime was 191.05 s and sampled peak
memory 1.287 GB, within the separate 300 s/6 GiB limits. All 3,845 archived files,
57 numerical sources and patient bytes remained unchanged. Zero actor forwards
and updates occurred; early failure prevented final parameter hashing, so no
measured weight-equality claim is made. Only 15.68 s of planning was attributed
to previews; the remaining time was not profiled and has no assigned cause.
No retry or geometric sweep followed. The bounded lane is closed as a preserved
negative result while mechanics takes priority. See
`artifacts/pat05-real-spatial-search-comparison-v1/outcome.json`.

The mechanics evidence review found public, measured human ex-vivo specimen
force/displacement and torque/twist curves, plus separate patient image-landmark
resources. FEBio is the selected first finite-strain specimen framework; its
isolated source build and homogeneous numerical verification are recorded below.
Measured specimen response remains unvalidated. The specimen check must precede
a separate real-anatomy displacement proof of concept; neither can establish
surgical action/force response without suitable interaction measurements.
See `docs/tissue-mechanics-frameworks.md`, `docs/tissue-mechanics-measurements.md`
and `docs/tissue-mechanics-validation.md` for the bounded scope and held-out rules.

The creator's CC BY 4.0 HBE release was acquired once: 12,985,030 bytes, with
all provider lengths and MD5 checks verified and local SHA256 receipts retained.
Metadata reconciles 182 specimens from seven donors. The first eligible specimen
is HBE_01_03, with radius 4 mm and creator-inferred height 4.89159 mm. Its entire
donor is permanently development. Third-cycle compression/tension are assigned
to calibration and third-cycle low-amplitude torsion to withheld-mode validation;
all other curves remain sealed. No measured curve values have been opened or
fitted. Independent metadata review passed. These processed ex-vivo data do not
establish patient stiffness, retractor forces or cutting response. See
`docs/mechanics-hbe-acquisition.md` and
`manifests/experiments/hbe-01-03-specimen-roles-v1.json`.

The measured-specimen protocol now fixes one finite-strain material law, one
positive stiffness fit, three mesh levels and withheld torsion prediction.
Analytical FEBio deck/output checks pass 43 controls, and the isolated runtime
driver passes 19 controls; these are software checks, not executed solver or
measurement agreement. The isolated FEBio 4.13 source build completed in 386.7 s
at 664.7 MB sampled process-group peak, with verified private/system linkage and
unchanged inputs. Its version probe reported 4.13.0. Gmsh 4.15.2 also passed
private payload/version checks;
its original checksum-format extraction failure and exact repair are retained.
See `docs/hbe-specimen-mechanics-poc.md` and `docs/mechanics-febio-verification.md`.

The actual five-deck FEBio patch attempt passed once: zero load, rigid
translation, finite stretch, shear and doubled stiffness. Independent review
verified all 25 saved states, signed reactions, stress, energy, reconstructed
Jacobians, force/moment balance, nonlinear residuals and stiffness scaling.
Largest signed-reaction error was 4.50e-14 N; supervised runtime was 0.525 s
with 48.8 MB sampled peak memory. Inputs, source and runtime hashes stayed
unchanged. These are homogeneous analytical controls, not measured tissue
agreement, mesh convergence or patient validation. See
`artifacts/mechanics-febio-patch-run-v1/RESULT.md` and its independent audit.

The separate actual tet10 and multipoint-constraint attempt passed three
controls: finite affine stretch/shear, constrained rigid translation and a small
nonrigid deformation. Independent saved-output review checked all five states
per case, original constraints, stress, energy and all 72 feasible force
directions. Maximum nonrigid constraint error was 2.00e-16 m; maximum reduced
force was 2.10e-11 N and minimum sampled Jacobian was 0.996685. The three
Skyline calls completed in one 0.781 s supervised attempt with 40.1 MB sampled
peak memory, unchanged inputs and no retry. Concurrent work prevents treating
that time as an isolated benchmark. These are analytical software controls;
the patient volume-integral observation operator and anatomical mesh remain
unvalidated. See `artifacts/mechanics-patient-constraints-run-v1/RESULT.md` and
`artifacts/mechanics-patient-constraints-run-review-v1/REVIEW.md`.

The fixed 5 mm reference-volume tent operator is implemented separately from
the earlier nodal-average solver fixtures. It preserves negative quadratic
weights and checks integration refinement, support, original constraint rows,
rigid-mode rank and deterministic elimination without reading destinations.
Thirty-six analytical controls pass, including independent octant integrals
and exact nonzero constraint reproduction. Early sample-cap failures are
preserved; independent review also caught and fixed mutable diagnostic evidence
that was missing from the operator hash. Integration discrepancies are empirical
convergence checks, not certified error bounds. No patient operator or mechanics
solve has run. See `docs/mechanics-patient-observation.md`.

A source/runtime audit found the optional Apple sparse backend is compiled and
linked, but its adapter copies a nonzero-entry count from a column-pointer array
allocated with only columns-plus-one entries. This concrete bounds defect blocks
its use; no defective-backend execution was attempted. Iterative tolerance types
also truncate fractional defaults. The existing verified Skyline path and size
caps remain unchanged. A separate local framework repair is documented but has
not been built or released. See
`artifacts/febio-sparse-backend-static-review-v1/REVIEW.md`.

The narrow sparse-adapter patch passed six repaired-code memory checks under
ASan/UBSan in 2.203 s, with 157.4 MB sampled group memory. Independent saved-file
review confirms exact patched fragment, source/binary identity and all results.
The original diagnostic compile failure and subsequent automatic worker-screening
interruption are preserved; no original faulty binary was run. Full adapter,
factorization, numerical and physical validation are still pending. A new isolated
runtime declaration will bind this reduced positive-only scope. See
`artifacts/febio-accelerate-csc-repair-v1/README.md` and its independent review.

The isolated build driver and runtime-v2 preparation passed 18 focused controls,
but root withheld the build after a further static review found uninitialized
matrix attributes in the upstream adapter. The SDK requires defined transpose,
triangle and reserved/allocation flags; the pointer-only patch does not provide
them. All symbolic-factor option fields are assigned, so that separate structure
needs no correctness change. A minimal matrix-initialization addition and a new
combined-patch declaration are required. No replacement runtime has been built
or executed; the previous positive offset tests and driver review remain valid
within their stated scope. See the runtime-v2 preparation root decision.

The subsequent direct-path ownership audit also found definite numerical-factor
storage loss across repeated stiffness reformations, skipped symbolic cleanup
before a successful first factorization, and a missing implementation-object
delete. The SDK permits some failed numeric objects to retain allocated storage;
those require appropriate cleanup too. These are static ownership findings, not
measured memory growth or a performance result. The combined adapter repair must
track ownership separately from factorization success, keep the existing
SparseFactor algorithm, and verify defined corrected-only lifecycle controls
before the full build. See `artifacts/febio-accelerate-lifecycle-review-v1/`.

The combined one-file correctness patch is now prepared and independently
reviewed. Twenty-three Python/static preparation controls pass; the new
corrected-only fixtures cover four SDK structure-reset profiles and fourteen
defined mock ownership scenarios. The separately released C++ attempt then
passed all eighteen scenarios under ASan/UBSan in 5.459 s at 312.3 MB peak
sampled group memory. The four expected mock-factor failure messages and two
excluded iterative-tolerance compile warnings are preserved; no sanitizer error
was reported. This does not establish actual factorization or physical accuracy.
The runtime-v3 driver refuses configure/build until the accepted result and
independent review are bound. Original runtime, declarations and failed
diagnostics remain unchanged. See `artifacts/febio-accelerate-lifecycle-v1/`.

Independent saved-result review verified both instrumented arm64 binaries,
all eighteen control inputs and exact expected logs without rerunning them.
The final runtime-v3 declaration now binds that accepted evidence; nine narrow
binding checks pass, including rejection of failed or mismatched receipts.
The original prospective declaration is preserved. Isolated configuration then
passed in 8.050 s at 134.5 MB sampled group memory. All 6,164 original acquisition
entries and 1,037 installed entries remained unchanged; the new 2,322-file source
inventory differs only in the declared adapter. The separately bounded build
then completed in 436.342 s at 585.3 MB sampled group memory, with all thirteen
installed arm64 Mach-O files and dependencies bound by parent acceptance. No
new runtime execution or numerical result is claimed. See
`artifacts/febio-accelerate-csc-runtime-v3/RESULT.md`.

The independent saved-build audit rehashed original/new source and installation
inventories and parsed all thirteen actual arm64 binaries' dependency and RPATH
records. They match the accepted build and resolve only to the new libraries,
the exact existing private OpenMP, or system libraries. Its initial overly strict
RPATH-order assertion and correction are preserved. No executable was launched
by this audit; numerical validation remains pending. See
`artifacts/febio-accelerate-runtime-saved-review-v3/review.json`.

The optional backend comparison/access integration passed 127 focused checks
including 14 independent controls. It now verifies the actual accepted build,
complete library inventory and saved eight-control execution/checker evidence,
while retaining the original scientific decks and all twenty specimen-case
roles. Review exposed four evidence-validation defects, which are repaired with
original failures preserved. This is preparation only: runtime/patch identities
remain provisional until the combined adapter repair is verified and built.
No replacement solver, specimen measurement or patient outcome was accessed.
See `docs/hbe-backend-comparison.md` and its independent review.

The comparison integration is now pinned to the final combined adapter repair,
including its nested patch/source identities and five accepted adapter-control
records. Fifty-four focused checks pass in 1.15 s; scientific XML, access rules,
profile name and original specimen experiment remain unchanged. Actual runtime
identity and numerical evidence are still required before a new specimen run.
See `artifacts/mechanics/hbe-backend-combined-repin-review-v1/`.

The separate eight-case numerical-control runner passed 31 software checks in
0.38 s, including independent rejection and evidence-preservation controls.
Its eight prepared decks change only the linear solver; original physics,
time grids and numerical checkers are retained. The initial unreleased draft
declaration is preserved, and no active execution declaration exists yet.
These are runner checks, not new FEM results. The repaired runtime must be
built and accepted before the actual eight controls can run. See
`docs/mechanics-accelerate-controls.md` and its independent review.

The separately released repaired-runtime attempt then passed all eight actual
numerical controls, including five-state stiffness scaling and both constrained
tet10 cases. Every solver log selected Accelerate and only the solver subtree
changed. One supervised sequence took 5.091 s at 45.4 MB peak sampled group
memory (5.783 s including launcher), with eight calls, no retries and unchanged
inputs. Complete analytical records are preserved for independent saved-output
replay. This is numerical verification, not measured tissue or patient accuracy.
See `artifacts/mechanics-accelerate-controls-run-v1/RESULT.md`.

Independent saved-output replay reproduced all eight original checker results
and five-state stiffness scaling exactly, reverified both summaries, 58 run
inputs and 28 archived source files, and made no solver call. Minimum sampled
Jacobian was 0.9966851963; maximum multipoint constraint error was about 2e-16 m.
The 60 raw analytical files are preserved byte-for-byte. This supports the
next fixed specimen experiment, not measured-force or patient acceptance.
See `artifacts/mechanics-accelerate-controls-saved-review-v1/review.json`.

The new specimen preparation binds the accepted runtime and numerical evidence,
an exact fourteen-file committed source archive, and eighteen read-only copies
whose only scientific-deck change is the linear solver. Independent saved-file
review verified all 218 inputs unchanged. Calibration/held-out roles, one-scale
material model, convergence gates, unverified CSV-format assumption and fixed
20-call/90-s-per-call/900-s aggregate limits are preserved. The disabled draft
cannot execute; a separately recorded final release is required. No specimen
solve or measured member was accessed during preparation. See
`artifacts/mechanics/hbe-accelerate-experiment-preparation-v1/README.md`.

A separate patient-mesh candidate helper passed 52 analytical/mocked checks.
It preserves complete bounded coordinate/connectivity/ID diagnostics before
count rejection, and rejects same-length corrupted writes; the original
failing control is retained. Its one prospective Gmsh size field requests a
12 mm boundary and 24 mm interior without changing the source surface or
2 mm/3% fidelity gates. A new 6,000-node/6,000-element preparation cap is not
solver admission. No new patient meshing has run; the original 2,065-node
rejection remains terminal. A bounded source-bound caller is still required.
See `docs/mechanics-patient-mesh-candidate.md` and its independent review.

The separate mesh caller passed 33 focused checks, including independent
resource/publication rejection tests. It binds a seven-file committed source
archive, the exact saved native surface, private Gmsh and QC ancestry. One future
candidate is limited to 180 s, 3 GiB sampled group memory, one numerical thread,
2 MiB diagnostics and bounded logs/output. This is launcher preparation only;
no new patient mesh or solver has run. See
`docs/mechanics-patient-mesh-candidate-launcher.md`.

The single released graded Case4 attempt returned 5,223 nodes and 2,761 tet10
elements in 10.479 s at 647.6 MB sampled group memory. Count, topology, quality,
midside and overlap checks passed, but sampled source-to-mesh and reverse
surface distances reached 5.460 mm and 3.834 mm, exceeding the unchanged 2 mm
limit. Volume difference was 1.208%; it does not override this local failure.
The complete 413,290-byte diagnostic packet is preserved, all 28 input/context
bindings stayed unchanged, and no retry, MRI re-extraction, B/V access or solve
occurred. The patient mesh remains unaccepted. See
`artifacts/mechanics/resect-case4-patient-mesh-graded-v2/RESULT.md`.

Independent saved-mesh review confirmed the complete four-array packet, finite
coordinates, connectivity and unique native IDs, and all raw/output/source
records. It independently rehashed 26 non-image bindings and checked the saved
before/after evidence for all 28; it did not reload the original mask or source
surface arrays or recompute geometry. The mesh remains rejected. The next
diagnostic is to locate the saved discrepancies, because aggregate maxima alone
cannot identify their spatial cause. See
`artifacts/mechanics/resect-case4-patient-mesh-graded-independent-audit-v2/`.

One bounded Gmsh preparation generated the three measured-specimen meshes
(96/768/2,592 hex elements) and 18 fixed loading decks in 3.43 s, with
140.2 MB maximum sampled memory. Independent native-mesh and deck review
passed the geometry/topology/loading checks; finest volume error is 0.2853%
and boundary sag is 0.2141% of specimen radius. Review also found and preserved
a JSON-reload serialization defect: named boundary sections changed order.
A narrow canonical-order repair now reproduces all 18 original deck byte hashes;
no mesh, load or material was changed or regenerated. This is preparation only,
not specimen equilibrium, convergence or measured force/torque agreement.
See `artifacts/mechanics/hbe-01-03-mesh-preparation-v1/` and the independent review.

The first specimen experiment stopped after its first solver call because the
reader rejected a rounded time header. FEBio itself terminated normally in
0.735 s; the complete failed attempt took 4.266 s. Nineteen cases remain
unexecuted, and no measured curve was opened, fitted or evaluated. The pinned
runtime writes nine significant digits in time headers, whereas the reader
incorrectly assumed twelve. All archived sources, runtime, original meshes,
decks and measurement bytes remained unchanged. The 219 pre-execution controls
passed; the real-output failure is retained independently of those tests. See
`artifacts/mechanics/hbe-01-03-experiment-v1/RESULT.md`.

The narrow header-format repair passed 43 focused checks and independent
saved-output replay. All 61 states of the existing coarse compression run now
pass its individual numerical criteria, including 60 nonlinear residual checks.
No solver was rerun, mechanical tolerance changed or measured curve accessed.
This establishes neither mesh convergence nor agreement with tissue measurements;
continuation still requires its own source-bound release. The original parser
and its reproduced failures remain in
`artifacts/mechanics/hbe-time-header-independent-review-v1/`.

The single source-bound specimen continuation stopped at the finest mesh's
unchanged 90 s per-call limit. The original coarse output was reused exactly;
medium compression completed in 35.902 s and its 61-state readout passed an
independent replay. Fine compression was interrupted after seven log-confirmed
convergences; its primitive files contain states 0–8, so the last snapshot does
not establish an eighth converged step. Total calls are three, including the
original reused call; 17 cases remain unexecuted. No measured curves were
opened, fitted or evaluated. Inputs and both original attempts remain unchanged.
The completed medium log attributes 32.4757 of 35.8333 s (90.63%) to the linear
solver. Sampled peak RSS was 283.7 MB; this identifies a runtime bottleneck,
not a memory failure or a measured-tissue accuracy result. A repaired existing
sparse backend needs separate verification before another experiment. See
`artifacts/mechanics/hbe-01-03-continuation-v1/RESULT.md` and
`artifacts/mechanics/hbe-continuation-outcome-independent-review-v1/`.

For the real-anatomy displacement proof of concept, six official RESECT Case4
files (35,068,010 bytes) were acquired and verified. This patient is permanently
development and does not change existing cohort roles. The reviewed access
contract selects six supplied motions using source coordinates only, leaving at
least six destination measurements withheld. Baseline MRI/ultrasound alignment
cannot use the later ultrasound or those outcomes. Baseline header checks
preserve different original T1/FLAIR affines and the oblique ultrasound frame.
Source-only parsing found 19 eligible landmarks and froze six input IDs with
13 withheld outcomes; every destination coordinate remains unopened.
Brain/cavity boundaries are not supplied. The task
is conditional displacement interpolation with declared mechanical assumptions,
not a validated surgical action, retractor force or cutting response. See
`docs/resect-conditional-displacement-poc.md` and the acquisition receipts.
This is a retrospective intraoperative update: the supplied paired motions are
not available from a preoperative scan alone. Automatic motion extraction and
the timing/uncertainty of such deployment observations remain separate work;
the manual correspondences must not become hidden inputs to a preoperative policy.

One frozen main-v1/MPS estimate of the Case4 T1 brain envelope completed in
7.305 s (11.644 s supervised), with no retry, fallback or download. All 59
archived sources, 33 provenance bindings and the original image stayed unchanged.
The wrapper accepted finite native-grid mask/SDT outputs with zero voxel-centre
corner discrepancy. Independent reconstruction and visual checks are recorded
separately below. This estimated envelope supplies no
reviewed pial surface, cavity, cortical access or patient material properties.
No later motion measurements or ultrasound were used. See
`artifacts/mechanics/resect-case4-brain-envelope-v1/completed-inference.md`.

Independent saved-array QC completed once in 16.420 s with 510.0 MB sampled
process-group RSS. Reconstruction matched every voxel; native affine and full
cell corners matched exactly, and save/reopen checks passed. The mask has one
component of 1,186,021 voxels (1186.021 mL) and touches no image-volume face.
Root inspected all three fixed native planes: the contour broadly follows the
cerebral region, but visible inferior/cerebellar structures are excluded. This
is not a verified whole-brain or pial boundary. Root permits preparation of a
bounded mesh of this unchanged estimated domain only; anatomical acceptance,
cortical access, later motion data and actual meshing remain separate gates.
No mask enlargement or landmark-guided correction is permitted. See
`artifacts/mechanics/resect-case4-brain-envelope-independent-qc-v1/`.

The fixed estimated-domain mesh utility passed 37 analytical/mocked checks.
One actual library control on a small cube completed in 1.537 s with 157.4 MB
sampled memory, but lost 8.499% volume against the fixed 3% limit. Both surface
distance bounds passed; this confirms that distance alone misses an important
fidelity error. The failure and raw output remain preserved. The patient mask,
24/20/16 mm target sizes and all rejection limits stayed unchanged for the
separately released patient attempt below. See `docs/mechanics-patient-mesh.md`
and `artifacts/mechanics-patient-mesh-independent-review-v1/`.

The single real Case4 meshing attempt then failed its fixed coarse count cap:
2,065 nodes versus 2,000, with 1,052 volume elements. It took 7.789 s and
658.4 MB sampled group RSS; medium/fine remained unexecuted. All 19 broader
source/input/context bindings stayed unchanged. Quality and fidelity checks did
not run. The v1 sequence retained the native source surface, logs and returned
counts/IDs, but rejected before saving the over-cap volume geometry. This
limitation prevents retrospective fidelity analysis; no retry, mask change,
solver or B/V access occurred. No accepted patient mesh exists. See
`artifacts/mechanics/resect-case4-patient-mesh-v1/RESULT.md` and its saved audit.

Saved-log investigation also found Gmsh's rounded optimization volume about
4.339% below the extracted source-surface volume. This is a warning from the
library log, not an independent assessment of the unsaved rejected mesh. It
supports investigating boundary approximation rather than merely raising a
node cap. One untested boundary/interior size-field approach and bounded raw
diagnostic retention are documented in `docs/mechanics-patient-mesh-followup.md`;
no new mesh, source change, parameter trial or solver was released.

The first baseline alignment worker produced all six declared views and the
15-pair proper rigid fit, but its supervisor failed with `RSS_MONITOR_FAILED`
after 2.536 s. The cause is unconfirmed because the failed monitoring query's
details were not retained. This remains a failed attempt despite worker exit
zero and 627.9 MB maximum sampled combined memory. Independent saved-output
review reproduces raw/fitted/leave-one-out RMS distances of
3.112/1.164/1.295 mm and verifies the original source/frame/view hashes.
Nonexpert inspection of all six views grants no anatomical acceptance.
The original failure and preparation chronology are preserved in
`artifacts/mechanics/resect-case4-baseline-alignment-v1/`.
A separately reviewed V2 supervision repair reproduced all numerical fields
and six images exactly, but also failed in 2.555 s: its final valid system query
reported child RSS zero/state `?E` while direct exit checks still returned
running, followed shortly by exit zero. All 69 monitoring records and the
failed outcome are retained; this does not retrospectively diagnose V1.
Independent saved-record review confirms source and input integrity. Neither
attempt changed the fit, landmarks or thresholds; neither is a successful
supervised run. All B/V motion destinations and the during-resection ultrasound
remain closed.

A third, separately declared monitor handles the brief process-exit transition
with one bounded direct wait. Its single attempt passed in 2.519 s, with
628.8 MB sampled combined RSS. Independent review reproduced all 69 monitoring
records; the requested 50 ms exit wait confirmed exit zero after 3.825 ms.
All numerical fit fields and six image bytes equal both earlier failed runs,
whose records remain unchanged. The existing nonexpert visual inspection can
therefore be reused for those images only. This is a completed supervised
baseline diagnostic, not clinical or anatomical acceptance. No B/V motion or
during-US access occurred. See
`artifacts/mechanics/resect-case4-baseline-alignment-v3/comparison-and-outcome.json`
and its independent saved-result review.

The conditional-displacement comparison helper now freezes three fields from
exactly the same six observed B motions: no shift, proper rigid motion and fixed
inverse-distance-squared interpolation. Complete baseline/FEM artifacts and
source identities must be frozen before the existing joint V source/destination
reveal. Reports distinguish global baseline coverage, common supported points
and explicit exclusions, with millimetre errors and no invented clinical
probabilities. Thirty analytical controls pass, including independent checks
that exposed and repaired sampler-entrypoint, bounded-read and undeclared-input
problems. No actual B/V motion was accessed and no patient comparison exists yet;
FEM location/interpolation and mechanics gates remain separate. See
`docs/patient-displacement-comparison.md`.

The saved tet10 field evaluator now preserves the native MRI/common ultrasound
frame conversion and returns explicit null statuses for unsupported or ambiguous
queries. Independent review found and fixed a tiny-overlap acceptance error and
two malformed source-identity cases; the original failures remain recorded.
All 57 evaluator/comparison checks pass in 0.41 s with six bound files unchanged.
This is software interpolation evidence only. Complete fields still must freeze
before later outcomes are opened; no patient query, field or displacement result
exists. See `docs/mechanics-patient-interpolation.md` and its independent review.

The first real TRAIN PAT22 inference profile completed all four declared cells
in 6.247 seconds with unchanged model parameters. One CPU call took 28–30 ms
for a 32³ crop and 256–269 ms for a 64³ crop, with highest observed memory about
1,180 MiB. These are single cold calls, not accuracy, training-fit or complete
planning timings. The image encoder dominated these calls. An optional critic
now also observes legal tool/candidate context; its default preserves the old
model exactly. It has no measured learning advantage yet. See
`docs/real-spatial-policy-scale.md` and the compact execution receipts.

One official ReMIND development case is acquired: 386 preoperative DICOM files,
59,163,552 bytes, with verified object identities and native pixel roundtrips.
The T1 and T2 have different frame identifiers; their automatic cerebrum and
manual tumor annotations do not establish a common planning frame. Independent
native-coordinate checks passed. A bounded rigid-registration attempt worsened
the separate similarity diagnostic and remains unaccepted. No joint planner
case or training result is claimed. Official UCSF v5 access was also retried:
the provider reported missing server files and listing/transfer errors. No
directional diffusion or matching exam-specific gradients were downloaded.

**Current user priority (October 4): spatial RL and surgical decision learning.**
Interface development, packaging, the pending cache-sizing execution and the
older paired feature-unit patient run are deferred. The user wants a policy
conditioned on a new patient's available scan evidence and tool geometry, with
optional bounded adaptation/search, rather than further interface polish.
The immediate research direction is spatial observations, explicit separation
of observed estimates from hidden reference anatomy, and sequential opening /
removal decisions across varied anatomies. Existing clinical and data-provenance
limits remain in force; more simulator training cannot establish surgical safety.

The earlier 15+6-feature actor is annotation-assisted: exact supplied labels and
simulated cavity state affect both its features and candidate inventory. It has
no image encoder. Missing motor/language flags identify array presence; they do
not establish adequate functional evidence or compel abstention. Completed
procedural pretraining used zero human training patients. Unseen-patient,
scan-only and unfamiliar-tool generalization are unproved. The latest steering
replaces the proposed procedural training screen with real-patient learning.
Private reference truth must stay out of proposals as well as tensors; learned
inference and search must use the same permitted observation contract.

The new scan-conditioned spatial policy is implemented: a 24,331-parameter 3D
encoder scores tool rays and STOP, with a spatial value baseline. Its observation,
policy and coordinate suites pass 77 checks. Independent analytic review corrected
trajectory-length bias in the policy gradient; numerical-coordinate checks also
corrected loss of valid image-edge samples. The separate synthetic opening task
passes 16 owner and 19 independent checks. Exhaustive enumeration of its smallest
fixture gives a unique return of 0.503158 after two costly openings; one-step
greedy stops at zero. This verifies a sequential learning opportunity, not that
the learned policy has solved it. No procedural population screen will run under
the current real-data-only training instruction.

The deferred procedural spatial screen uses scan intensities, observed support/cavity, tool
geometry and procedure state. Private target and functional labels cannot enter
proposals, policy tensors or search lookahead. Its primary reward is explicitly
geometric: functional exposure remains unassessed. The small task removes
tip-intersected cells, distinct from native contained-cell execution. Native
integration is now implemented separately in the real native adapter described
above; vessel/function integration and held-out human generalization remain
open requirements. See `docs/spatial-policy-research.md`,
`docs/spatial-task.md` and their linked negative-result receipts.

The frozen BTC expansion acquired PAT22/PAT25 for population training and
PAT26/PAT27 for checkpoint selection: 72,375,710 new bytes, with source checksums
and finite image arrays verified. Native T1/annotation orientation reconciliation
has at most 0.000858 mm corner discrepancy. These are acquisition/integrity
results; brain envelopes and cortical access remain unreviewed. PAT29/PAT31 stay
unopened. Receipts are in `artifacts/btc-spatial-acquisition-v1/`; no new human
training or planning run has been executed from this batch.
All four cases now also have source-bound portable bundles with exact save/reopen
identity checks, prepared in 21.02 seconds from committed code. Native MRI and
annotation frames, missing modalities, and cohort roles are preserved. Peak
worker memory was 430 MB. Brain masks and cortical access remain unavailable;
preparation is not anatomical approval. Execution receipts are in
`artifacts/btc-spatial-preparation-v1/`.

The read-only inspection bridge and separate linked MRI/3D tool display were
committed after 100 bridge checks and 68 viewer checks, respectively. App host
integration is unfinished and preserved as working changes. Its 10 focused
synthetic checks do not establish an integrated build or native validation.
Known deferred issues include logical cancellation preceding actual worker
completion and mismatched support-acknowledgment conditions. The validated
packaged app below has not been replaced. The previous full Python suite's
12 failures have focused repairs, but a new full-suite pass is still pending.

**Interface direction updated by the user:** Electron + React + TypeScript is
the current Mac UI target. Further PySide/PyQt interface work has stopped. The
validated Qt prototype is preserved as an earlier experiment; Python planning,
learning, provenance and evaluation remain the shared backend. The new renderer
communicates through a narrow local sidecar interface. The first standalone
Electron package has passed real MRI, route comparison and native-save checks.
The full training package has passed actual updates, cancellation/resume and
restart checks. The refreshed native patient UI now passes blocked-route
preflight, exact-stroke training, replay/export, cancellation and full-app
restart/resume. Accessible replay controls were fixed after a native regression
and now keep the image, displayed step and quantities synchronized.

The Electron renderer adds separate, explicitly unreviewed brain-envelope
contours on the original MRI planes. Source/frame/mask checks and lazy loading
pass for both BTC cases: the no-CSF estimates exclude 214 PAT28 annotation voxels
and 538 PAT05 annotation voxels. These are overlap checks, not accuracy scores.
All 97 desktop checks and the production build pass. Native inspection now
passes on both cases, including source restoration, case switching and the
outside-annotation cursor jumps. A stale footer was caught and fixed; final
renderer source `75d7f69b…` uses the previous verified numerical engine. Working
anatomy and access gates stay unchanged. Screenshots and exact package identities
are in `artifacts/electron-structural-proposals-v1/`.

Seven registered motor/language prior proposals now persist in portable cases
with separate values and atlas coverage, exact T1 registration/T1c display source
bindings, and unchanged planning inputs. They remain view-only population
evidence requiring alignment review. Their Electron controls now display one
selected layer with separate atlas coverage, scalar/binary sampling and cursor
values. Source restoration, stale loads and case changes clear the layer.
The preceding 143 desktop checks and production build passed after removing an unsafe
source-image hash cache, correcting float32 atlas boundary sampling and adding
context-aware route-comparison guidance. Native inspection caught overlapping
unloaded welcome text; a regression test and shell correction now pass at both
1460- and 1050-point window widths. That notice-bearing renderer is
`34285015…` from `100865f`, with the unchanged verified `85a92eff…` engine. Its
source, complete engine payload and packaged upstream notices match the captured
bytes. Loaded-case guidance, real search, motor-prior inspection and source
restoration pass; screenshots and receipts are in `artifacts/dependency-notices-v1/`.
The preceding native renderer is `a6bb6b76…` from `346df23`, with 148 passing
desktop checks and a successful production build. Its new Center on annotations
action returns linked MRI slices to the source annotation center while preserving
prior/source mode, A/B routes, opacity and layout. Native checks pass at both
window widths; unloaded and MRI-only cases disable the control. PAT16/PAT20
main/no-CSF estimates, all four omission jumps and source restoration also pass.
All 72 source inputs, 2,809 unchanged engine entries and 292 unchanged notice
entries verify. `artifacts/electron-annotation-center-v1/` retains exact receipts
and screenshots. The engine remains `85a92eff…`; new experimental learning and
cache code are not in this package.

The preceding renderer is `bc63bc2e…` from `fe25f2a`, with 150 passing desktop
checks and a successful production build. Its fixed header retains the loaded
case identifier during evidence-panel scrolling. Actual UCSF → PAT16 → PAT20
switching passes at 1460- and 1050-point widths, with correct full accessible
labels and a centering regression check. All 73 desktop source inputs verify;
the engine and notices remain byte-identical. Root inspected the minimum-width
UCSF and wide PAT20 MRI screenshots. Native testing did not introduce an
extreme-length synthetic identifier. Exact receipts and images are in
`artifacts/electron-case-identity-v1/`.

The current native renderer is `834c7ceb…` from `c09ee50`, with 153 passing
desktop checks and a successful production build. Empty category/A/B selectors
now stay hidden until candidates exist. Native unloaded, pre-search UCSF and
blocked PAT20 states show their existing next-step guidance at both window
widths. Actual UCSF search produced 54 candidates; category switching and A/B
comparison work, with breadcrumb and centering preserved. Its three categories
were nonempty, so the empty-filter boundary is source-tested rather than claimed
as a native observation. All 73 source inputs, 2,809 engine entries and 292
notice entries verify. Root inspected pre-search and populated screenshots.
See `artifacts/electron-route-empty-state-v1/`; the engine remains unchanged.

Two notice-source completeness gaps and distribution signing remain open. An
independent actual GPU probe passes 487 cases, with the earlier failures retained.
The new complete engine and renderer passed native inspection. All seven layers,
covered zero, positive values, outside/incomplete support, minimum-width layouts,
save/reopen and historical checked replay were exercised. The preceding renderer
`762973e2…` from `c194a8b` repeats the corrected sampler and source-restoration
checks with engine `85a92eff…` from the verified `0b4e334` Python snapshot.
Later experimental learning changes are not in this app snapshot. Native
receipts and screenshots are in `artifacts/electron-priors-v1/`.

The first declared procedural-to-patient comparison stopped before adaptation
updates because equivalent typed and JSON world vectors compared differently.
Its raw records and failure are retained; it produced no validated comparison.
The implementation repair preserves every declared world field and adds a
public-case preflight before any training. The separately declared fresh v2
completed with original geometry, rewards, seeds and budgets unchanged. All six
patient learners changed actor weights and all 13 frozen candidates passed
independent native checking. SEARCH/GREEDY scored 245.24; scratch seeds scored
[245.24, 171.42, 245.16], frozen procedural initialization 139.62, and adapted
seeds [171.42, 171.42, 245.17]. This is a negative transfer result on one reused
development patient with zero human pretraining patients. Final/stress worlds
remain closed. The best sequence removes only 249 of 41,919 target mm³ under
hypothetical access; this is not a complete resection plan.

The separately declared actor feature-unit comparison completed from the
immutable `0bffeaa` archive after all 794 tests and the real-case preflight
passed. It compares raw inputs with fixed physical reference units while
preserving the old action inventory, rewards, critic, seeds and budgets. Other
heavy work was paused for timing comparability; ordinary desktop/OS load was not
controlled. All 23 frozen candidates passed native checking and a separate
artifact audit. Fixed units raised the procedural frozen-policy return from
139.62 to 245.24, matching SEARCH. All three scaled adapted runs selected their
unchanged initial checkpoint; they establish no additional adaptation gain.
Scratch seed differences were [0, 0, +0.08], and no arm exceeded SEARCH. The full
launcher took 534.18 seconds, including separately reported offline and audit
costs. Read-only diagnostics confirm reduced actor saturation and persistent
critic state aliasing, without proving clipping-induced learning harm. See
`artifacts/learning/procedural-native-feature-units-v1/RESULT.md` and its
separate independent audit and diagnostic records. This is one reused case,
two procedural families and zero human pretraining patients.

A separate post hoc geometry probe expands the proposal inventory while keeping
the native checker, source, tools and access assumptions fixed. Under the same
three-cut cap, it removes 1,113 target and 62 normal mm³, with 177 mm³ of partial
normal contact. Both native and independent artifact audits pass. Coverage is
still only 2.655%, and this new, restricted parallel-column model is not yet part
of the desktop planner or the frozen learning comparisons.

The proposal rule now has a separate source- and cavity-bound implementation.
Independent review caught changed arrays and live cavity edits that retained
cached identities; all eight original failures are preserved. Repairs now verify
source contents, committed history and all four cavity masks before emitting
proposals. All 66 focused checks and a separate 15-case review pass. This module
does not certify tool clearance or integrate with learning yet. In one actual
case observation, integrity/proposal checks took approximately 0.29–0.30 seconds
per call; cumulative process peak RSS was 1.16 GiB. Initial and post-cut proposal
inventories matched the frozen prototype. These are individual local timings,
not a latency distribution or isolated allocation measurement.

A separate RAW-only simulator adapter now supplies that inventory to the common
search/learner interface while retaining the native cutting engine. Twenty-nine
new analytic checks and 29 compatibility checks pass; the independent review's
19 cases are included in the new checks. Failed-reset integrity, preview-error
accounting and transient-cache defects were caught, repaired and preserved as
negative evidence. The first public preflight stopped before adapter construction
on a configuration-identity mismatch; two audits isolated truthful provenance
text as the only difference. The separately declared V2 completed five patient
episodes from immutable `6930417`: greedy return 1,098.77 and unchanged RAW
initialization 441.60, with both distinct histories independently certified.
A separate artifact audit and 25 adversarial checks reproduce all source-cell
counts and rewards. The run took 149.38 seconds and 1.314-GiB peak worker RSS.
Its 48.19-second initial policy panel exceeds the older 30-second allowance,
though it is not an exact generic-learner cost. No gradients or final/stress
worlds were used. Later decision features/logits were not retained in that
preflight; the completed bounded update pilot records them. The adapter remains outside the desktop,
and existing procedural checkpoints and feature-unit gates remain unchanged.
See `artifacts/preflight/native-axis-v2/RESULT.md`.

A supported optional decision observer now records the actual already-used
policy inputs, logits, value and action before execution. Fifty-four focused
compatibility checks and 30 independent checks pass; recording preserves exact
weights, Adam state, RNG and environment behavior. Its native journal binds to
served observations and distinguishes attempted, executed and returned steps.
A separately declared one-update seed-11 pilot completed from immutable
`f3a0591` after 53 focused runner checks and the 1,150-test integrated sweep.
One actual actor update used two complete optimization episodes; initial and
updated two-world selection both returned 441.60, correctly retaining the
initial checkpoint. Six complete histories have accepted certificates covering
three unique native histories. A separate 35-test artifact audit reconstructs
all 18 saved forwards and verifies all 12 deterministic choices without rerunning
patient simulation or gradients. Six stochastic choices are eligibility-checked,
not RNG-replayed. The full launcher took 221.14 seconds, online work 169.59
seconds, and peak worker RSS was 2.219 GiB. This verifies the update pipeline,
not a learning improvement. No cache or final/stress worlds were used.
Original receipts remain unchanged; lossless gzip and gzip-only report/audit
reproduction pass. See the actual report in
`artifacts/learning/native-axis-raw-update-pilot-v1/report-v1/RESULT.md` and
`artifacts/native-axis-pilot-result-audit/`.
A later saved-decision diagnostic pairs all six selection records into three
distinct states. Chosen actions are unchanged, while 8/17/12 rows change ordinal
rank, including STOP, and all saved values/logits change. No exact 15-feature
aliases occur among the 26/24/22 legal non-STOP rows on this path. Twenty-six
constructed-record checks and an independent arithmetic/hash audit pass; a
duplicate-decision-ID validation gap was fixed without changing actual results.
This performs no new model forward or simulation and does not explain the cause
of the unchanged score. See `artifacts/native-axis-decision-diagnostics-v2/`.
An isolated exact capsule-cover cache prototype passes 78 focused checks and
26 independent checks, with its actual invalidation failures preserved. Synthetic
reference/cached/reference outputs agree, including independent native histories.
The separately declared public four-phase probe then completed from immutable
`43857b5` in 158.29 seconds. Cold/warm caching took 22.896/23.050 seconds versus
21.182/21.670 seconds for the bracketing uncached phases. Both cached phases had
22,364 misses and zero hits at the declared 32-MiB/16,384-entry limits; no speedup
was demonstrated. All four scientific traces are byte-identical and all four
native audits pass. A separate artifact audit reconstructs five tissue/contact
masks at all four states and the 1,098.77 reward from source cells; 34 adversarial
checks pass. Process peak RSS was 2.133 GiB, distinct from cache payload size.
This negative fixed-trace result does not establish full-learning throughput.
Production geometry and the packaged engine stay unchanged. See
`artifacts/native-axis-cache-report-v2/` and
`artifacts/native-axis-cache-result-audit/`.

An isolated synthetic descriptor probe reproduces a known RAW action/state
alias: identical first reward 4.33 but best two-cut returns 39.43 and 39.68.
Access-relative coordinates and cavity/residual centroids distinguish that
pair. Eight owner checks and seven independent checks pass. This is nine tiny
probe transitions, no patient or policy execution and no production profile
change. Independent review retains a roughly 2-mm rotation-rounding failure at
the tangent-selection cutoff. Different masks can still share the summaries;
contact history, tool and remaining budget are omitted unless supplied separately.
The proposed summary is not a complete state or an established learning gain.
See `docs/synthetic-spatial-feature-probe.md` and both independent artifact sets.

The isolated V2.1 descriptor replaces automatic tangent-axis selection with an
explicit source/frame-bound reference and a declared engineering conditioning
check. The saved rotation case has 2.22e-16-mm error for a well-conditioned
reference; a near-parallel reference rejects in both frames. Independent review
found 14 initial constructor/mutation/overflow/complex-input failures, now fixed
with their original evidence preserved. Eleven owner and 30 independent checks
pass; 14 saved numeric comparisons are unchanged. This is algebra-only work
with no patient, simulator, policy or production-profile changes. Lossy state
summaries and the absence of a shared cross-patient reference convention remain
explicit. See `docs/synthetic-spatial-feature-v2.md`.

The frozen extraction procedure has now run on PAT16 and PAT20, with two
repetitions of both model variants per case. All eight child inferences completed
without failures or tuning; repeated masks and distance arrays were identical.
Independent array and visual QC passed, with 28 focused checks. No-CSF/main
estimates omit 2,045/19 of 45,400 PAT16 annotation voxels and 413/125 of 12,451
PAT20 voxels. These flags remain; no target voxels were inserted into the masks.
Working anatomy and portable source cases are unchanged. The estimates still
require anatomical review, and runtime version pins are not historical binary
attestation. Exact execution/QC records are linked from
`docs/brain-extraction-pat16-pat20-independent-review.md`.
Separate portable copies now include both r1 envelope proposals and exact report
provenance. An independent 11.92-second audit and 15 adversarial checks confirm
unchanged source/planning inputs, exact mask persistence and enforced review
gates; focused native app inspection now passes. Predicted distance arrays remain
external diagnostic files. See `docs/brain-extraction-pat16-pat20-bundles.md`.

## Verified starting state

- Repository initially contained only the three specifications and two commits.
  A bounded search found no existing implementation to preserve elsewhere.
- Local development: Apple M5, 16 GiB unified memory, macOS 26.6, Python 3.12.
  Hardware probe and architecture evidence are saved in `docs/`.
- An isolated environment now has Qt/PySide6, VTK, NiBabel, SciPy, PyTorch and
  testing/packaging tools installed. The first arm64 app bundle passed an actual
  frozen-runtime and MRI/3-D window smoke check on this Mac.
- Work is on `codex/patient-specific-planner`. Every new commit uses author
  `skamal23 <sayemkamal12@gmail.com>` and co-author
  `shafsthegoat <shafrir.p@gmail.com>`; commits are pushed incrementally.

## Executed vertical slices

- Immutable patient/provenance contracts, time-aware context filtering, physical
  NIfTI QC/import, preserved source annotations, case revision and JSON/NPZ
  roundtrip are implemented. Future context cannot alter planning seeds.
- UCSF-PDGM-0004 structural mirror bytes are pinned and checked; equivalence to
  official TCIA remains unverified. TCIA v5 fixes diffusion metadata relative to
  the specification's v4. The official transfer service failed during acquisition.
- Creator-source OpenNeuro BTC `sub-PAT28`, ds001226 v5.0.1, has 15 verified
  files including T1, fractional tumor annotation, AP DWI, PA references and
  gradients. Its full-head T1 does not provide a reviewed cortical access mask.
- A second creator-source BTC case, `sub-PAT05`, was selected by a recorded
  metadata-only rule before image acquisition. Seven pinned files total 17.86 MB;
  independent checks agree on all 11,437 threshold-derived annotation voxels.
  The registry has three development groups, two primary-source identities and
  zero final patients at that checkpoint. Unknown clinical timing remains
  excluded from planning.
- Two further creator-source cases, PAT16 and PAT20, were selected and committed
  from metadata before image access, then acquired with pinned source versions,
  lengths and annex checksums. Their structural files total 35.73 MB excluding
  shared release metadata. Independent source, native-frame, annotation and
  save/reopen checks pass: threshold 0.5 derives 45,400 and 12,451 target voxels.
  Full-head access gates remain enforced, with no reviewed brain masks or new
  diffusion. The registry now has five development representations, four verified
  primary-source identities and zero final groups; all 16 source-evidence checks
  and 26 cohort tests pass. This does not complete a five-UCSF or tract-aware
  pilot, and engineering QC is not expert anatomical approval.
- Seven licensed motor/language population maps load with source hashes and
  explicit registration-review contracts. Patient alignment remains unaccepted.
- Full-tip/shaft/swept-envelope route search retains feasible, dominated and
  rejected alternatives with failure evidence. The UCSF route run generated 54
  candidates, 31 geometrically feasible, 12 geometric Pareto. The separate
  checker agreed on all 54 under the same limited structural assumptions.
- The standalone Electron app loaded the real UCSF bundle and rendered linked
  physical MRI, annotation surfaces and two full-tool routes. Route generation
  took 1.16 seconds in one native run; the Mac save dialog preserved the routes,
  comparison and cursor. Its initial imaging/search bundle is 388.6 MB and has
  no Qt/VTK dependency. Accessible volume is distinct from removed volume.
- Actual patient policy-gradient updates, resumable checkpoints, scratch/search
  comparators and isolated world roles execute. Eighteen development runs and
  six budget/exploration follow-ons retained their source snapshots and failures.
  No RL advantage over search was demonstrated.
- Independent native-footprint checking **invalidated every non-STOP UCSF
  coarse-removal candidate**. The coarse simulator removed cells beyond the
  active tip's supported footprint. Those runs remain negative development
  evidence. Their coarse returns are not valid resection performance.
- The corrective native-grid engine and separate checker now agree on a real
  two-stroke sequence: 249 mm³ target and 17 mm³ normal tissue removed, with zero
  unsupported source-cell volume. Three patient-training seeds changed weights;
  one improved from STOP to SEARCH's score, two retained their initial policies.
  SEARCH and greedy search were faster. All nine frozen development candidates
  passed native checks. Missing functional evidence remains unknown.
- PPO at 32 and 128 actual optimizer updates failed to outperform SEARCH on the
  branching development fixture. Exact source snapshots, initial weights,
  transition counts and negative results are preserved.
- Shared pretraining, frozen inference and isolated adaptation now execute with
  explicit cohort exclusions and separate offline costs. The first two-group
  synthetic experiment used eight offline updates and eight online updates per
  seed. SEARCH scored 1.30, scratch [0, 0, 1.24], frozen 0 and adapted [0, 0, 0].
  All ten independent synthetic geometry checks passed; no final worlds opened.
  This is a negative analytic experiment, not a clinical population policy.
- A separately declared procedural-native comparison completed fresh pretraining
  and all six scratch/adapted arms after a retained implementation-only failed
  attempt. Offline training made 32 actual Adam updates across two nonpatient
  families; online runs made 10–14 updates within matched 30-second cooperative
  caps. All actors changed, though scratch seed 23 selected its initial policy.
  No learned arm exceeded SEARCH. All 13 candidates passed eight distinct native
  sequence audits, and a separate artifact audit recomputed scores from source
  cells. Total launcher time was 288.87 seconds, including 33.12 seconds of
  independent validation. These development results do not replace real
  patient-population evaluation; full evidence is retained under
  `artifacts/learning/procedural-native-to-ucsf-v2/`.
- The Electron training bridge executes real updates, resumable cancellation,
  independently checked replay and export. A 30-second patient-learning budget
  yielded the two-stroke sequence in 37.77 seconds total including setup/checks;
  a 5-second run retained STOP. Resume binds checkpoint bytes, source/runtime
  contract and original budgets; unsigned crash recovery is refused.
- Native UI testing exposed a selected-route mismatch: the factory discarded
  the selected entry/target in favor of parallel global-centroid proposals.
  Exact-ray diagnostics also found that none of the original 54 straight routes
  could cut their own corridor with the declared generic instruments. Merely
  substituting the two named native profiles did not fix those original rays.
  Two explicitly different, source-grid-aligned research tool/access candidates
  pass independent checks. Exact selection pass-through now freezes the complete
  entry, target, window and tool. Actionless routes perform zero updates. Original
  routes remain alongside two explicitly different native-action alternatives.
  A real UCSF bridge run made 32 updates, changed the actor and improved nominal
  selection return from 0 to 171.58: 174 target + 11 normal mm³ removed in one
  independently checked stroke. Export, save and restart/recheck passed. This is
  fixed stroke-versus-STOP refinement, not free-form route learning or a controlled
  efficacy benchmark; final worlds remained unopened.
- Raw BTC diagnostic tensor fitting ran on 101,311 voxels; bounded CSA tracking
  produced 176 unlabeled diagnostic paths. Uncorrected DWI is correctly refused
  for tract-aware planning. Correction and alignment are current work.
- Pinned SynthStrip main/no-CSF models ran locally on MPS in roughly 6–7 seconds
  per variant with identical outputs in four runs. No-CSF excludes 214
  source tumor-annotation voxels; main includes them all. Neither is reviewed
  brain/cortex or an accepted cortical-access mask. CPU failure is preserved.
- The same frozen extraction settings ran on preregistered PAT05 in 7.69/6.01
  seconds for no-CSF/main. Independent source/model/output hashes, native geometry
  and distance-map reconstruction passed. No-CSF excludes 538 of 11,437 source
  annotation voxels; main excludes none. Both remain unreviewed proposals, and
  annotation inclusion is not an accuracy metric.
- Structural proposals now save/reopen separately from working anatomy, with
  immutable source/model/run provenance and review-required status. Adding unused
  proposals leaves planning seeds unchanged. Explicit anatomy prohibitions take
  precedence over collection defaults; reviews cannot silently enlarge masks or
  relabel model estimates as observations. Native UI inspection confirms the
  BTC proposal inventory leaves route generation disabled.
- Isolated PyHySCO phantom testing reduced a known-distortion image error from
  9.68% to 0.40%, with 0.395-mm displacement error. The zero-distortion solver
  failure and unsupported BTC phase-encoding geometry remain explicit blockers;
  no patient correction or FSL execution was performed.
- A separate experimental physical-vector solver passes a bounded rotated-pair
  phantom and an independently certified identical-input branch. Image error
  falls from 9.68% to 1.07%, with 0.315-mm field error; coordinate-invariance
  failures in two earlier revisions are retained. All nonzero runs reach their
  40-iteration limit. Thirty-four focused checks and independent artifact/error
  verification passed. This does not promote any patient preprocessing gate.

## Validation record

The latest root integrated run passed **1,150 tests in 215.95 seconds** from
immutable `f3a0591`, with no skips and no changes or additions to any of the
2,904 baseline files. Fourteen warnings comprise ten intentional NumPy metadata
adversaries and four existing DIPY warnings. The preserved archive includes the
new observer/accounting, one-update pilot runner, cache prototype and portable
proposal audits. Its committed release baseline permits one patient update,
two optimization episodes and both complete selection panels, with no cache,
final worlds or stress worlds. Results remain separate from the launch record.
See `artifacts/validation/native-axis-pilot-public-v1/`.

The preceding integrated run passed **959 tests in 196.11 seconds** from
immutable `558b2e3`, with four existing DIPY warnings. All 2,669 baseline
files stayed byte-identical. This includes the axis provider/adapter, tiny RAW
learning, preflight/adversarial checks, notice collector and acquisition changes.
The earlier 914-pass/one-failure attempt is retained: its analytic orchestration
fixture incorrectly mixed the evolving registry with historical frozen inputs.
The test-only repair passed 63 focused checks and retains a real production
byte-drift rejection test. Both full attempts are in
`artifacts/validation/integrated-axis-v1/`. The new zero-gradient public
action-model preflight V1 stopped during configuration validation before
adapter construction, simulated cuts or gradients. Its source stayed unchanged. Two actual configuration-only audits found
identical physical inputs, with only the tissue-support provenance text
differing between factories. The fresh V2 declaration binds the direct helper
without rewriting either provenance description or bypassing identities.
Its immutable `6930417` launch passed 77 focused checks in 23.07 seconds; all
2,710 baseline files remained unchanged. Five subsequent public episodes and
two unique native audits completed; a separate artifact audit passed 25 tamper
checks. All 46 captured source files and original/execution case bundles still
match. Five histories compress losslessly from 39,286,826 to 665,699 bytes;
gzip-only report regeneration is identical, and raw originals remain local.
Executed versus returned transition accounting separately passes 19 analytic
checks, including one real tiny gradient update and interrupted native commits.

The preceding complete run passed **794 tests in 176.14 seconds**, with four
existing DIPY basis-deprecation warnings, from an immutable archive of `0bffeaa`.
Its bytes stayed unchanged throughout testing. The real public-case preflight
also passed both policy profiles and all three seeds with zero gradients before
registered feature-unit execution. Exact identities and results are in
`artifacts/validation/native-feature-units-runner-v1/`. The preceding
743-pass/one-failure sweep is retained: its orchestration fixture needed to
isolate a newly added earlier preflight, while production rejection was correct.
The test-only repair passed focused checks before the earlier 749-pass sweep and
this latest complete validation. The newer standalone provider and axis adapter have separate independent
reviews and are included in the latest integrated attempt.
Structural adversarial checks found
and fixed metadata precedence, mask-enlargement and provenance problems;
their focused checks passed before each incremental commit. Native
adversaries also caught temporal shaft borrowing, mutable preview descriptors
and checkpoint-resume tampering. Test counts describe their recorded snapshots.
The independent native checker improved from 17.638 to 4.310 seconds on the same
saved patient history, with identical geometric certificates and 91 regression
tests. These are local single-run timings, not a latency distribution.
Native initial-geometry reuse lowered median profiled reset time from 0.462 to
0.0259 seconds without changing captured outputs. Three further 30-second runs
completed 14/11/11 updates versus 7/6/5 previously; only seed 47 improved its
selected score. SEARCH remained best. Renderer hydration decreased from a
352-ms median to 134 ms in a warm-cache V8 benchmark; this is not GPU latency.
See `EXPERIMENT_LOG.md` for executed experiments, including negative results.
The Qt prototype passed its initial standalone runtime and GUI checks. Its next
build was stopped following the user's Electron decision. The React/Electron
package passed its own imaging/search checks and 31 main-process boundary and
lifecycle tests. The current snapshot passes 38 main-process, 74 renderer and
31 viewer checks covering source
hydration, evidence status, A/B identity, independently accepted removal overlays
and population-prior sampling. The complete
training bundle has its own source-bound verification receipts. Qt results do
not certify Electron. No second-Mac test is claimed.

The previously verified refinement shell (renderer `5fe94739…`, Python `377284f6…`)
recovers a six-update cancelled checkpoint after app restart and adds 26 updates
under the original contract. Accessibility decrement/increment now changes the
actual checked replay between 0/0/41,919 and 174/11/41,745 target-removed,
normal-removed and residual-target mm³. Matching screenshots document the fix.
Display-normal averaging improves shading without moving any source triangles;
a separate local probe adds 75.5 ms of worker preparation to 332.0 ms of source
mesh preparation. This is a display tradeoff, not a GPU speedup claim.

## New backend inspection boundary

A separate read-only facade now returns the complete initial expanded-axis
proposal and native-preview inventory with exact RAS geometry, full tool and
source bindings, and explicit unknowns. It requires separate neighboring-column
and estimated-support acknowledgments. It performs no transitions or policy
updates and grants no removal or candidate authority. Sixteen owner and 33
independent synthetic checks pass. Review reproduced and repaired stale cached
source identity after callbacks and a valid LPS normalization roundoff refusal;
both failures remain preserved. Existing selected-route readiness is unchanged.
No public inspection or Electron integration is claimed for this slice. See
`docs/native-axis-inspection.md` and its independent review artifacts.

## Completed saved-cache diagnostic

The released saved-only reconstruction completed once from immutable `b0f5635`:
22,364 ordered calls contain 8,812 distinct argument keys and 13,552 first-pass
repeats. A separate audit reproduces every byte/order and reuse/LRU statistic;
33 audit tests pass. The full key set is below the existing 16,384-entry cap,
so increasing that count alone cannot help this fixed trace. Full cover payload
sizes and byte-weighted reuse remain unknown; no larger memory capacity or
speedup is established. External child wall time was 1.108 seconds, with all
16 archived inputs unchanged. Cooperative limits passed, but exact peak RSS
was not retained. No geometry, patient, simulation, policy or gradient ran.
See `artifacts/validation/native-cache-query-keys-v1/RESULT.md` and
`artifacts/native-cache-query-result-audit-v1/`.

## Expanded-axis policy input profiles

The scratch learner now accepts explicit RAW or FEATURE_UNITS profiles through
a closed axis observation contract. Accounting version 2 distinguishes raw
simulator features from the policy's actual transformed inputs; physical models,
registered divisors and six critic inputs remain unchanged. One hundred owner
regression tests and 25 independent checks pass, including real tiny updates and
profile/transfer rejection before optimization. The independent tiny update
changed actor weights but tied initial selection return; this is integration
evidence, not learning improvement. Original timing-assertion and test setup
failures remain retained. Full integrated regression and a new frozen paired
public declaration remain prerequisites. No new public training occurred.
See `docs/native-axis-input-profiles.md` and the two focused review directories.

## Current integrated regression attempt

The immutable `523a72b` attempt completed with **1,470 passing tests and 12
failures in 243.45 seconds**, plus 14 warnings. All baseline files remained
unchanged, with no added source files. Ten diagnostic CLI tests inherited the
full pytest process's earlier peak memory and correctly hit the production
512-MiB guard before reaching their intended assertions. Two historical pilot
auditor tests encountered the newly added axis contract field. Scoped test
isolation and explicit auditor contract-version repairs are in progress; public
experiments and the next bundled engine remain gated. Original logs and source
manifests are retained in `artifacts/validation/axis-profile-integrated-v1/attempt-01/`.
This failed attempt does not replace the preceding 1,150-test passing snapshot.

## Completion gates still open

Native-resolution geometric simulation now works under explicit hypothetical
access assumptions; reviewed anatomy and surgical validity remain open.
Corrected/accepted functional reconstruction, fully integrated comparison and
refinement interaction, broader cohort benchmark and release remain incomplete.
Frozen/adapted initialization has been tested on analytic and procedural-native
fixtures, including transfer to one reused patient development simulation;
patient-population training and external evaluation remain open. Developer ID
signing/notarization and independent physical-Mac
validation remain open. No locked final cohort has been opened. Clinical deficit
probabilities remain unavailable; simulator success cannot establish clinical
readiness. Current evidence supports a local research prototype only.

No cloud infrastructure or billing changes have been made. Source imaging stays
outside Git. Numerical arrays and checkpoints are excluded; compact experiment
records and source snapshots preserve the executed development evidence.

The complete [RESECT TRAIN metadata inventory](artifacts/resect-train-expansion-v1/RESULT.md)
now qualifies 25 original ultrasound/cavity-mask pairs across 13 labeled people
within the frozen 14-person TRAIN cohort. Case11 has no labels; Case15 lacks a
during-resection label. The remaining 24 pairs total 548,605,591 bytes. Independent
inventory review passed; expanded download routes, payload QC and training remain
unexecuted. Annotation restrictions and protected Case4/SELECT/evaluation roles remain.
The separate exact-file TRAIN importer is now prepared and independently reviewed:
75 controls pass after incomplete-receipt and late-completion acceptance defects
were reproduced and repaired. Both negatives are preserved. It attempts one pair
per explicit bounded invocation; source acquisition and image/label QC remain
separate. See [preparation](artifacts/resect-train-intake-preparation-v1/RESULT.md).

[IBSR qualification](artifacts/ibsr-source-qualification-v1/RESULT.md) did not establish
an eligible support-label source: noncommercial terms, README login, unresolved label
ancestry/domain and participant overlap remain. No scientific payload was accessed.

The [uniform N32 mechanics study](artifacts/mechanics/hbe-global-n32-execution-v1/RESULT.md)
completed one native solve in 639.057 seconds (750.919 seconds for the full phase).
Independent review reproduced five complete readouts and all 61-state comparison
metrics. Adjacent-mesh changes pass, but the conditional remaining-force envelope
**0.985189 mN exceeds 0.724169 mN** at states 49–60. Spatial acceptance, finest-mesh
load-step verification and calibration remain closed; no physical force curve was
accessed. A prospective N36 diagnostic needs a separate reviewed declaration.

The reviewed [144-mask intake](artifacts/lausanne-annotation-intake-preparation-v1/RESULT.md)
is committed with all 148 metadata outcomes retained. An independently checked
cache-only sub476 pilot reproduced 193 positives with unchanged original bytes
and its source-specific grid proof. The first live sub022 transfer acquired its
exact 36,551-byte source but failed content QC before counting labels: the
initial parser refused a valid NIfTI extension. A strict, uninterpreted extension
reader now passes 56 owner and 32 independent controls, including a bounded
check of the real 592-byte prefix. The original failure is retained. A separately
recorded cache-only repeat now passes: 2,899 positive voxels and exact original
TOF grid equality, independently reproduced. Full-inventory processing remains
next; no new fitting or planning admission follows. See [format evidence](artifacts/lausanne-annotation-format-v1/RESULT.md).
