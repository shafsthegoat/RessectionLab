# Vascular rendering parity plan

Goal: vessels that are as crisp, clean, connected and responsive as the
supplied Medivis Studio head-and-neck CTA clip, and usable for simulated
surgery. RessectionLab has no Medivis affiliation and ships no Medivis assets.
The clip is a visual reference of a different patient and field of view.

Branch `shaf/medivis-vascular-parity-20261125`, cut from
`origin/shaf/interactive-operative-practice-20261121` at `9b079d28`.
Evidence lives on Shafrir's HDD under `qc/medivis-parity-20261125/phaseN-name/`
([storage ledger](local-storage-ledger.md)). Source scans and labels stay unchanged.

## Diagnosis

The reference path-traces tagged CTA intensities. The image is noisy while the
camera moves and converges in about a second at rest. RessectionLab meshes
binary masks and composites them into a four-sample ray march. That produces
stair-stepped walls, fragments and clutter. Soft model probabilities exist in
the Python collection but never reach the GPU. EDEN TOF has no bone signal, so
its target is a clean intracranial tree.

## Hero cases and fixed views

| Case | Study | Role | Independent vessel annotation |
| --- | --- | --- | --- |
| CTA-A | TopCoW-014 CTA native showcase, with skull | development | TopBrain v1 CT labels |
| CTA-B | TopCoW-018 CTA vessel comparison | held out | TopBrain v1 CT labels |
| TOF-A | EDEN PTNT_4509 staged TOF review | development | none |
| TOF-B | EDEN PTNT_9937 linked TOF review | held out | none |

TopCoW-018 was a frozen verification subject in the earlier annotated CTA
benchmark and was never used to select settings. TopCoW-012 was a selection
subject and is not held out. Published segmentation models may have trained on
any TopCoW case, so "held out" here means held out from this project's tuning.

Six presets are derived from each scan's geometry and a circle-of-Willis
anchor: anterior, left lateral, superior, oblique, skull base (superior view
under an axial cut) and axial clip. `scripts/capture_vascular_parity.py` holds
the definitions and writes the exact cameras beside each capture.

Six further presets are framed like the reference stills: a wide skull
base, a wide axial clip and a coronal-cut anterior on every case, and, on CT
in the path-traced mode only, an opaque-skull left lateral, an opaque-skull
oblique and a sagittal cut. `capture --add` adds presets to an existing
capture of the same study and settings.

## Measurements

`scripts/capture_vascular_parity.py capture|geometry|sheet --label <phase>`
drives the hidden desktop harness and `resectionlab.vascular_parity`.

| Metric | Definition |
| --- | --- |
| Frame time p95 | Start of one drawn interactive frame to the next during a 180-input continuous orbit, one input per display refresh. Frames per input, interactive resolution and GPU timer coverage are reported with it. CPU plus GPU cost is a diagnostic only: the renderer waits for each GPU fence, so cost understates delivery. |
| Settle time | Pointer release to GPU completion of the settled frame, including the app's 150 ms interaction hold. The renderer declares convergence where it can; otherwise the last refinement before the view goes quiet. |
| Floating components | 26-connected components that do not reach a root. Roots are the named basilar, ICA, vertebral, vein of Galen and sinus labels where they exist, otherwise the largest component. |
| Normal turn p95 | Dihedral angle across shared edges of the displayed wall, in physical space. |
| Source deviation | Millimetres, and smallest-spacing voxels, from each wall vertex to the steepest fall of acquired signal along its outward normal, searched to two voxels. A vertex with no clear edge in reach counts as a full-reach miss, and the resolved fraction is reported. |
| Sharpness | Median 10 to 90 percent rise width across the strongest edges of the 3D pane at logical display scale. The cursor crosshair remains in frame. |
| Dice, clDice, Betti | Dice and clDice inside the annotation's bounding region padded by 5 mm. Betti numbers on whole predicted components that enter that region, and, since Phase 2, a patch Betti error: the region is tiled in 16 mm cubes and the absolute b0 and b1 differences per cube are summed, so loops in unrelated noise cannot cancel missing ones. |
| Blind review | A fresh reviewer scores shuffled, unlabeled frames 1 to 5 on crispness, depth, clutter, connectivity, realism and usefulness beside the reference frames. |

## Open decisions

1. **No annotation exists for either EDEN case.** Phase 2 scored the
   labelled TopCoW-018 MRA as the held-out angiography reference and gives
   PTNT_9937 label-free measures only. Shafrir has not yet confirmed this
   substitution.
2. **Which Betti measure decides the Phase 2 gate.** The whole-volume loop
   count on CTA-B still favours vesselFM alone by two loops (9 against 7)
   after both attempts, while the patch Betti error and every other measure
   favour the ensemble. Needs Shafrir's decision; see Phase 2 results.
3. **What counts as a major vessel lost.** Phase 3 treats a major named
   vessel as lost when it is absent from the default view or less than 90
   percent of its reference centerline is shown, and lists anything between
   90 and 100 percent as partly in review. The list is the basilar, ICA, M1,
   M2, A1-A2, third A2, P1-P2 and vertebral arteries and, on CT, the superior
   sagittal sinus, straight sinus and vein of Galen. Small named segments
   (ophthalmic, anterior choroidal, distal M3 pieces) stay in the review
   overlay when no signal-supported path joins them. The threshold and the
   list were set by the coordinator and need Shafrir's agreement.
4. **Venous walls beside bone do not resolve.** The edge model expects lumen
   brighter than its surroundings. Only 14 to 57 percent of sinus wall vertices
   find such an edge on CTA, so the half-voxel deviation gate is meaningful
   for arteries and TOF vessels, and venous deviation is reported unresolved.
5. **Python suite baseline.** At `9b079d28` the full suite has 571 failures
   and 515 errors from data-policy gates and absent local data, none in vessel
   code. The saved list is the reference: phases must add no new failure.
6. **CTA-A blind gate not met after three attempts.** After the second
   failure Shafrir chose to show finer vessels (2026-11-26); see Phase 6
   results. The third review scores the richer tree 3, 4, 3, 3, 3, 4
   (crispness, depth, clutter, connectivity, realism, usefulness) and ranks it
   above the Phase 5 set (4, 4, 4, 3, 3, 3 from the same reviewer): usefulness
   rose, crispness and clutter fell because the thin branches are lumpy and
   beaded. The reviewer's main remaining gap is vessel surface quality.
   Shafrir decides whether to work on how thin branches are drawn (a smooth
   tube model or tag for 1 to 2 voxel vessels), to try a head-and-neck CTA, or
   to accept the result. The path-traced mode stays behind its setting.
7. **Path-traced vessels do not follow a tool press.** In rehearsal the
   path-traced mode draws vessels from a tag volume sent once per case, so the
   wall mesh dents under a tool only in standard mode. Rehearsal should stay in
   standard mode until the tag volume follows the wall; whether that work is
   wanted is Shafrir's call. The surgeon view also needs its own treatment in
   this mode (see Phase 5 results).
8. **Which mode is the default.** Standard mode stays the default. Blind
   reviewers scored the path-traced mode higher on every criterion on both
   hero cases, and it meets the frame budget, but it fails decision 7 in
   rehearsal.
9. **Decided by Shafrir on 2026-11-26.** Every vessel is drawn in one
   colour; arteries and veins need not be told apart. Detached vessel signal
   should join the tree where that is realistic. Two more EDEN cases, chosen
   for the cleanest trees, go through the pipeline instead of the whole
   cohort (about two hours of inference each). These replace the Phase 3
   colour rule and extend the Phase 2 rule that vesselFM is used only where
   it already touches the named tree.
10. **Earlier uncommitted work is stashed.** Twenty-three edited files from the
   previous session (CTA receiver shadows, vessel-rod caching, walkthrough
   documentation) are in `stash@{0}` on the old branch and are not part of
   this branch.

## Phases

| Phase | Scope | Acceptance | Status |
| --- | --- | --- | --- |
| 0 Baseline | Headless harness reproduces presets and metrics for today's app | Evidence saved for all four cases | done 2026-11-25 |
| 1 Cinematic renderer | Soft-label tagging, tricubic intensity, progressive path tracing, hybrid shading, HDR environment; WebGPU versus WebGL2 recorded | Interaction p95 at most 16.7 ms; settle at most 1.5 s on CTA-A; blind crispness and depth at least 4 | frame gates met in the packaged build 2026-11-26; blind crispness and depth are 4 with an opaque skull or a cut and 3 with the see-through skull; kept behind the Volume lighting setting |
| 2 Segmentation | TopBrain 2025 top-three models, ensembled with vesselFM where connected to the named tree; soft probabilities kept; CTA labels never applied to MRA | Better Dice, clDice and Betti than vesselFM alone on the held-out cases | run 2026-11-26; one Betti count not met after two attempts, see results |
| 3 Cleanup | Root-reachable components by default, the rest behind a review toggle; signal-supported bridging; caliber LOD; arteries red, veins blue | Zero floating components by default; no named major vessel lost | gates met on the drawn meshes 2026-11-26; see results |
| 4 Sim geometry | Centerline and radius implicit tubes shared by drawing, collision, rods and tags | Deviation at most 0.5 voxel; no self-intersection; tools displace vessels | gates met on TopCoW 2026-11-26 behind the tube geometry option; see results |
| 5 Integration | Volume, tubes, tools and clipping in the packaged build; films and pull request | All earlier gates; blind scores at least 4 | run 2026-11-26: frame budget met in the packaged build; TOF-A blind gate met; CTA-A blind gate not met after two attempts; volume-lit vessels do not follow a tool press; see results |
| 6 Finer vessels (added after decision 6) | Thin branches followed in the CTA signal; detached model vessels joined along signal; one vessel colour; two more EDEN cases | The Phase 5 blind gates; zero floating components | run 2026-11-26: zero floating pieces on five cases; TOF-A blind gate holds; CTA-A blind gate not met on the third attempt; frame timing not re-measured on an idle machine; see results |

A phase that fails its gate twice keeps its best version behind a toggle and
goes to Shafrir. Gates are not lowered.

## Phase 0 results

Captured 2026-11-25 on the M2 Pro with 16 GB at `9b079d28` plus the harness,
hidden, 1440 x 960 window, 711 x 805 3D pane. The display refreshes at 120 Hz.
Evidence: `qc/medivis-parity-20261125/phase0-baseline/README.md`.

| | CTA-A | CTA-B | TOF-A | TOF-B |
| --- | --- | --- | --- | --- |
| Frame interval p50 / p95 (ms) | 16.7 / 20.2 | 16.6 / 17.0 | 8.3 / 10.3 | 15.1 / 17.6 |
| Frames per input | 0.50 | 0.59 | 0.99 | 0.65 |
| Interactive long edge (px) | 360 | 360 to 540 | 360 to 720 | 360 to 540 |
| Settle (ms), settled samples | 541, 4 | 259, 4 | 166, mesh only | 182, mesh only |
| Sharpness, rise p50 (px), six views | 2.6 to 4.0 | 2.3 to 3.8 | 1.6 to 1.7 | 1.6 to 1.8 |
| Floating components (volume share) | 6 (1%) | 4 (3%) | 2,475 (45%) | 2,486 (55%) |
| Betti b0 / b1 | 9 / 27 | 9 / 42 | 2,476 / 519 | 2,487 / 496 |
| Arterial normal turn p95 | 71.5 deg | 70.5 deg | 47.8 deg | 47.8 deg |
| Arterial source deviation p50 / p95 (voxels) | 0.5 / 2.0 | 0.6 / 2.0 | 0.9 / 2.0 | 0.9 / 2.0 |
| Arterial edge resolved | 91% | 84% | 78% | 79% |
| Blind crispness / depth | 2.75 / 2.0 | 2.75 / 2.0 | 3.0 / 2.25 | 3.0 / 2.25 |
| Blind clutter / connectivity | 3.25 / 2.5 | 3.5 / 2.0 | 1.25 / 2.0 | 1.0 / 2.0 |
| Blind realism / usefulness | 2.0 / 2.75 | 2.0 / 2.25 | 2.0 / 1.5 | 2.0 / 1.25 |

The reference frames measure a 1.8 to 2.0 px rise at the same scale after
H.264 compression, so the CTA views are 1.3 to 2 times softer.

The CTA studies display the TopBrain annotation itself, so their displayed
Dice, clDice and Betti agreement is self-agreement. The segmentation baseline
is vesselFM alone on held-out CTA-B at a 0.5 threshold: Dice 0.26 to 0.27,
clDice 0.37 to 0.38, 334 to 357 surplus components and 7 to 14 missing cycles
across the two cached blends. No annotated held-out MR angiography run exists yet.

Today's CTA view keeps its frame interval near 16.7 ms only by dropping the
interactive image to a 360 px long edge and drawing every second refresh.

## Phase 1 backend choice

The same one-sample path-tracing kernel (delta tracking through a majorant
grid, gradient shading, one shadow ray and one ambient ray, float
accumulation) was timed in WebGL2, WebGPU fragment and WebGPU compute on
TopCoW-014 in a hidden Electron 44 window on the M2 Pro. GPU time p50 per
sample with bone, shadow and ambient rays:

| Size | Filter | WebGL2 | WebGPU fragment | WebGPU compute |
| --- | --- | --- | --- | --- |
| 905 x 1024 | trilinear | 9.3 ms | 13.8 ms | 13.1 ms |
| 905 x 1024 | tricubic | 19.7 ms | 30.4 ms | 26.2 ms |
| 540 x 611 | trilinear | 4.8 ms | 7.9 ms | 5.6 ms |
| 540 x 611 | tricubic | 10.7 ms | 16.4 ms | 11.1 ms |
| 360 x 407 | tricubic | 7.1 ms | 10.2 ms | 5.9 ms |

Converged 256-sample images from the three backends differ by at most one
8-bit level. **Choice: WebGL2.** It is fastest on the settled pane, and it
shares the existing three.js context, scan texture and mesh depth. WebGPU
compute wins only at small sizes and on vessel-only scenes, by a few
milliseconds, and would need a second device, canvas and volume upload.
Source, logs and images: `qc/medivis-parity-20261125/phase1-cinematic/prototype/README.md`.

## Phase 1 results

`volumeRendering.mode: 'cinematic'` ("Volume lighting" in the 3D view) path
traces the acquired scan: delta tracking over a brick majorant grid, cubic
B-spline intensity at rest, soft tags from the label masks choosing vessel
material, a glass-shell see-through skull, a shadowed key light with a short
obscurance bounce, and float accumulation to 1,024 samples with a
variance-guided filter that fades out. Moving frames trace one filtered sample
at a budgeted resolution. Standard remains the default.

Measured 2026-11-25 on the M2 Pro with 16 GB, hidden, on the built-in Retina
panel at 120 Hz with an external 100 Hz monitor attached as the main display.
Evidence: `qc/medivis-parity-20261125/phase1-cinematic/README.md`.

| Gate | Target | CTA-A result |
| --- | --- | --- |
| Interaction frame interval p95 | at most 16.7 ms | 10.6 to 11.4 ms over five runs: met |
| Settle after release | at most 1.5 s | 1.95 to 1.97 s over five runs, 1.42 s in the full capture run: not met |
| Blind crispness and depth | at least 4 | 3.25 and 3.5: not met |

Settle failed on both attempts, so by the rule above the mode stays behind
its setting and Shafrir chose to move on. "Settled" is the first sample count
whose displayed 3D pane is within 40 dB, and 1 percent of pixels beyond 8
levels, of the 1,024-sample image on the anterior, skull-base and axial-clip
views: 72 samples, limited by the cut skull base (64 samples leave 1.02
percent). A rest sample costs 18 to 21 ms of GPU and is delivered about every
25 ms, after the app's 150 ms interaction hold.

| | CTA-A | CTA-B | TOF-A | TOF-B |
| --- | --- | --- | --- | --- |
| Frame interval p50 / p95 (ms) | 8.6 / 11.2 | 8.7 / 10.4 | 8.5 / 11.0 | 9.6 / 11.2 |
| Frames per input | 0.93 | 0.94 | 0.95 | 0.88 |
| Interactive long edge (px) | 240 to 360 | 320 to 360 | 200 to 360 | 360 |
| Settle (ms) at 72 samples | 1,424 | 1,783 | 2,532 | 2,615 |
| Sharpness, rise p50 (px), six views | 2.0 to 3.2 | 2.3 to 3.7 | 1.8 to 2.1 | 1.9 to 2.2 |
| Blind crispness / depth, before to after | 2.75 to 3.25 / 3.0 to 3.5 | 3.0 to 3.0 / 3.0 to 3.5 | 2.5 to 2.5 / 2.25 to 2.25 | 2.5 to 2.5 / 2.25 to 2.25 |
| Blind clutter / connectivity, before to after | 2.25 to 2.75 / 2.75 to 2.75 | 2.5 to 3.0 / 2.5 to 2.75 | 1.25 to 1.25 / 1.75 to 1.75 | 1.0 to 1.0 / 1.75 to 1.75 |
| Blind realism / usefulness, before to after | 2.5 to 3.25 / 2.75 to 3.25 | 2.0 to 3.25 / 2.75 to 2.75 | 1.5 to 1.5 / 1.5 to 1.5 | 1.5 to 1.5 / 1.25 to 1.25 |

The blind reviewer scored sixteen Phase 0 and sixteen Phase 1 frames shuffled
together. Its Phase 0 scores differ from the Phase 0 reviewer's, so the
before-to-after change within this set is the comparison. Cut CTA views
gained the most (crispness and depth 3 to 4); uncut views keep bone haze, and
TOF views did not move because fragments and tangles dominate them.

Rendered lumen connectivity on CTA-A matches the label mask (arteries 8 of 8
components, veins 3 of 3). Thin vessels are beaded where their opacity follows
the binary mask. PNG export in this mode writes the standard image.

## Phase 2 results

The three top TopBrain 2025 submissions (labelled UZH, ARG and KDH here;
Zenodo record 20158639, CC BY 4.0) run natively on the M2 Pro from their
unpacked container archives: no container runtime, Apple MPS, half-precision
networks, one prediction at a time. `scripts/topbrain_native.py` unpacks,
predicts, combines, scores and retains; `resectionlab.vessel_ensemble` holds
the combination rules. CT pipelines run only on CT and MR pipelines only on
MR. Soft artery, vein and vessel probabilities and a named label map are kept
per case. Evidence, licences and deviations from the containers:
`qc/medivis-parity-20261125/phase2-segmentation/README.md`.

All three teams trained on every TopBrain case, including TopCoW-014 and 018,
so both are in-sample for the models. "Held out" means held out from this
project's choices: everything was chosen on TopCoW-014 and applied to 018.

Attempt one pooled the teams by a plain mean. On CTA-B it lost loops (23
against 42 annotated) and missed the Betti part of the gate. Attempt two chose
the pooling rule on TopCoW-014 only, from five candidates: the best
development team as the base (UZH on CT, ARG on MR), the others added only
where they connect to it. Because attempt two was prompted by a miss on 018,
018 is not a clean test of it; TopCoW-001 CT and MR are pre-registered as
confirmation cases and have not been run.

Held-out scores for the frozen attempt-two ensemble. "vesselFM alone" is the
Phase 0 baseline at a 0.5 threshold. "vesselFM tuned" has its threshold and
component filter chosen on TopCoW-014 with the same objective.

| CTA-B (TopCoW-018 CT) | vesselFM alone | vesselFM tuned | Ensemble |
| --- | --- | --- | --- |
| Dice / clDice | 0.266 / 0.373 | 0.452 / 0.530 | 0.910 / 0.978 |
| Component error (b0) | 357 | 8 | 4 |
| Loop count error (b1; 42 annotated) | 7 | 32 | 9 |
| Patch Betti error, b0 / b1 sums | 655 / 50 | 189 / 24 | 27 / 17 |

| TopCoW-018 MRA (stand-in for TOF-B) | vesselFM alone | vesselFM tuned | Ensemble |
| --- | --- | --- | --- |
| Dice / clDice | 0.499 / 0.375 | 0.651 / 0.520 | 0.952 / 0.983 |
| Component error (b0) | 1,448 | 6 | 7 |
| Loop count error (b1; 31 annotated) | 406 | 1 | 16 |
| Patch Betti error, b0 / b1 sums | 2,381 / 146 | 219 / 20 | 33 / 15 |

Gate reading. Against vesselFM alone the ensemble is better on Dice, clDice,
component error and both patch Betti sums on both cases, and on the loop count
on MRA. It is not better on the whole-volume loop count on CTA-B (9 against
7) after two attempts. vesselFM reaches 35 loops there across 366 separate
components, so its count sits near 42 without matching the annotated loops;
the patch error measures that (50 against 17). Against the tuned vesselFM row
the ensemble wins everything on CT and Dice, clDice and patch Betti on MRA,
and loses the two whole-volume counts on MRA (7 against 6, 16 against 1). The
plain mean had the better MRA loop count (0), which the development case did
not predict.

The vesselFM signal is added only where it touches the named tree, and is
kept as its own soft layer. Against these annotations it always lowers Dice
and clDice (CTA-B 0.910 to 0.889 Dice) because the annotations cover named
vessels only; every unnamed vessel counts as an error. Its value is judged on
the display, under the Phase 3 calibre rule.

One team's CT pipeline labels external carotid branches with the vein label
numbers; those classes go to background on the path that feeds the ensemble.
"CT models only on CT" holds per pipeline: two teams ship weights that serve
both modalities.

## Phase 3 results

The default vessel display (`vesselScope: 'rooted'`) shows the components
that reach a named root (ICA, vertebral, basilar; on CT also the vein of
Galen, straight sinus and superior sagittal sinus), joins fragments across
signal-supported gaps of at most 3 mm, and keeps everything else in an amber
review overlay behind "Show detached and thin parts". Nothing is deleted and
source masks are never edited. Arteries and unclassified vessels are red,
veins and sinuses blue. Vessels of 1 mm and wider always show; 0.5 to 1 mm
vessels show within 10 mm of a tumour or planned corridor. Rules, receipts and
limits: [acquired vascular geometry](native-vascular-geometry.md).

The Phase 2 ensemble is wired into the four hero studies as their initial
vessel layers: the named tree, and the connected small-vessel signal as a
child layer. Supplied annotation layers stay in each study, unchanged and
selectable. The child shows only pieces that extend at least two voxels
beyond the tree wall; one-voxel crusts on the wall go to review.

The gate is counted on the meshes the desktop draws, with face connectivity,
by `capture_vascular_parity.py geometry`, which exits non-zero on any
floating piece. Phase 0 counted voxel components of the earlier layers.

| | CTA-A | CTA-B | TOF-A | TOF-B |
| --- | --- | --- | --- | --- |
| Floating components before (Phase 0 layers) | 6 | 4 | 2,475 | 2,486 |
| Drawn mesh pieces / floating, default view | 3 / 0 | 6 / 0 (one inner wall) | 5 / 0 (four inner walls) | 3 / 0 (two inner walls) |
| Voxel components / floating, default view | 3 / 0 | 5 / 0 | 1 / 0 | 1 / 0 |
| Fragments joined across gaps | 0 | 2 (0.96 and 1.45 mm) | 1 (1.64 mm) | 2 (1.20 and 0.60 mm) |
| Named tree components left in review | 12 | 6 | 5 | 6 |
| Small-vessel pieces shown / wall crusts in review | 6 / 215 | 3 / 613 | 63 / 6,479 | 82 / 6,755 |
| Major vessel lost | none | none | none listed | none listed |
| Major vessels partly in review | R-M2 99.5%, L-M2 98.1% of centerline shown | none | R-M2, 60 of 20,403 voxels | none |

TOF has no reference annotation, so its major-vessel check uses the model's
own labels. Named vessels wholly in review: ophthalmic arteries on both CTA
cases, one anterior choroidal piece on CTA-A, and on TOF the maxillary and
middle meningeal branches, which belong to the external carotid and reach no
intracranial root. TopCoW-018 was run more than once while the measurement
was being corrected after review; each run is logged in the evidence folder.

Two independent reviews changed this phase. The first count could only come
out as zero and used corner connectivity that the drawn surface does not
have; the gate now measures meshes. The desktop first opened the CTA studies
on the annotation instead of the model result; wiring now makes the model
result the default. Captures then showed two display faults that numbers had
missed: CTA arteries drawn in the mauve reserved for unclassified vessels,
and the small-vessel layer drawn as about 200 shells around the tree. Both
are fixed.

## Phase 4 results

`resectionlab.vessel_tubes` stores one model per vessel layer
(`centerline-tubes-v2`): a centerline graph from the existing source-branch
skeleton, and per sample a radius plus angular harmonics up to order six in a
rotation-minimising frame, with end-cap terms at tips. The surface is the
hard union of the fitted segments and node balls, clipped to the acquired
box. Details: [acquired vascular geometry](native-vascular-geometry.md).

The engine option `operative_vascular_geometry = centerline-tubes-v2`
(`vessel_geometry` in `prepare_surgical_environment`, `vesselGeometry` over
IPC; no desktop control sets it yet) derives all four consumers from that
stored model: the drawn and collision wall, rod positions and calibres, the
soft tag volume the path-traced mode reads, and per-vertex artery or vein
class. Which rod sites exist and how they bond still comes from the existing
chain builder. An edit appends to the model's removal record and cuts the
stored field; nothing is refitted, so geometry away from a cut is bit
identical. The desktop's tag volume is sent once per case and is not yet
refreshed after a removal. Existing geometry policies give byte-identical
reports, case hashes, engine snapshots and desktop arrays to the previous
code on the pinned studies.

Attempt one used circular sections with a smooth junction blend and missed
the deviation gate at the 95th percentile (arterial p95 0.46 to 1.35 voxels).
Attempt two added the harmonic sections and was developed on TopCoW-014, then
run on 018. Deviation between the tube wall and the 0.5 level of the frozen
Phase 2 ensemble probability, in smallest-spacing voxels, mean / p95. The
independent review corrected the metric so that source wall with no tube
nearby counts in full; none was found. The model was fitted while the
first-attempt ensemble was current and re-measured, unchanged, on the frozen
one:

| Case | Arteries, tube to source | Arteries, source to tube | Veins, tube to source / source to tube |
| --- | --- | --- | --- |
| TopCoW-014 CTA | 0.09 / 0.26 | 0.07 / 0.23 | 0.10 / 0.29, 0.12 / 0.35 |
| TopCoW-018 CTA | 0.09 / 0.28 | 0.09 / 0.27 | 0.10 / 0.29, 0.12 / 0.34 |
| TopCoW-014 MRA | 0.13 / 0.38 | 0.15 / 0.45 | none |
| TopCoW-018 MRA | 0.14 / 0.40 | 0.16 / 0.47 | none |

In millimetres the arterial p95 is 0.11 to 0.14 on all four. MRA sits closer
to the limit because the limit is half of its 0.30 mm in-plane spacing while
its slices are 0.6 mm apart.

Every wall is closed, manifold, consistently oriented and free of
self-intersections, including coplanar overlaps. Component counts match the
source and no named label is lost; the least-covered label keeps 47 percent
of its voxels inside the tag volume. Loop counts match on 014 CTA and fall by
one to four elsewhere where tubes close holes smaller than their radius.
Meshes have 0.8 to 3.0 million faces; extraction and meshing take 24 to 108 s
and the whole evaluation up to 2.8 GB on the M2 Pro.

Tool contact under the tube option: after each compliant step the rods are
projected so the wall leaves the tool's working shell, checked by the
existing section-motion and wall-sweep rules; a step that cannot be cleared
is partial or rejected and the tool keeps its last clear pose. Over five
strokes with a 1 mm dissector (press, fast stroke, glancing sweep, stroke
along the axis, slow press at a junction) the shell overlaps the displayed
wall by at most 0.018 mm at any accepted step (0.84 to 0.87 mm before), the
working axis stays outside the lumen, and the vessel moves by up to 2.2 mm.
A two-tool pinch stops without either shell entering. The same strokes on
existing walls are unchanged, where only the working axis is a hard limit.

Limits. EDEN studies are not admitted for simulation, so the simulation gate
is shown on TopCoW only. `capsule_clearance` is a lower bound whose mesh
tolerance of one grid step rests on measurement, not proof; the engine
collides against the wall triangles. The fitted sections follow lumps in the
model boundary; displayed normals are smoothed for shading only.

## Path-traced mode after the look pass

After the Phase 3 captures were compared with the reference stills, the
path-traced mode ("Volume lighting") changed, and only that mode: standard
stays the default.

- Bone: a `cinematicBone` setting ("Bone" beside Volume lighting on CT).
  Automatic uses solid, lit bone at any cut or with an opaque skull display,
  and a see-through shell otherwise; the shell keeps its rim and highlight
  and 30 percent of its diffuse light, which removes most of the haze over
  uncut views. Bone tint follows scan density.
- Vessels: an orange-red artery and a blue vein of matching value, a wet
  highlight from a base and a coat lobe evaluated at the key light, and lumen
  normals from a two-voxel stencil at rest so thin vessels shade as tubes.
- Translucent mesh layers are composited into each rest sample before it
  joins the running mean, and silhouette normals are turned edge-on; both
  removed dark dotted rims around vessels.
- A saved native view may now carry the path-traced mode and the bone choice.

Three capture presets framed like the reference stills were added beside the
six fixed ones: a wide skull base, a wide axial clip and an anterior view cut
coronally. Frame timing and blind review of this mode are measured in Phase 5.

## Phase 5 results

Evidence: `qc/medivis-parity-20261125/phase5-integration/` on the personal
drive (captures, sheets, films, the packaged-build check under `integrated/`,
both blind reviews under `blind-review/`), and `RESULT.md` beside the phase
folders.

**Frame budget.** `node electron/package-macos.cjs` built the app from
`3bace51c` (ad hoc signature, not notarized). Measured hidden in that app on
the M2 Pro, three runs each, path-traced mode:

| Scene | Frame interval p95 (ms) | Settle (ms, 72 samples) |
| --- | --- | --- |
| CTA-A default connected view | 10.8, 10.5, 10.9 | 1,375, 1,417, 1,370 |
| Rehearsal crop with tube wall, two tools and an axial cutaway | 10.0, 9.8, 10.0 | 900, 874, 900 |

Both are inside 16.7 ms and 1.5 s. In the development tree CTA-B settles in
1.81 s and both TOF cases in 1.39 to 1.45 s; the gate names CTA-A. Standard
mode draws a frame in 7 to 10 ms and presents on alternate 120 Hz refreshes
(interval p95 17.1 to 18.5 ms), as in Phase 0.

**Blind review.** A fresh reviewer scored shuffled, unlabelled sets of Phase
0, Phase 3 (standard mode) and the final path-traced captures beside the
reference frames. Attempt one used the fixed presets with the transparent
skull display. Attempt two showed the CTA-A final set in the reference clip's
own framings (opaque skull, coronal, sagittal and axial cuts) and told the
reviewer five facts about the data and views; the TOF images did not change.
No renderer code changed between attempts.

| Final set | Crispness | Depth | Clutter | Connectivity | Realism | Usefulness |
| --- | --- | --- | --- | --- | --- | --- |
| CTA-A, attempt one / two | 3 / 4 | 3 / 4 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |
| TOF-A, attempt one / two | 3 / 4 | 3 / 4 | 4 / 4 | 3 / 4 | 3 / 4 | 3 / 4 |
| CTA-A standard mode | 2 / 3 | 2 / 2 | 3 / 3 | 3 / 3 | 1 / 2 | 3 / 3 |
| TOF-A standard mode | 2 / 2 | 2 / 3 | 4 / 4 | 3 / 4 | 2 / 3 | 3 / 3 |
| CTA-A Phase 0 | 2 / 3 | 2 / 2 | 3 / 3 | 2 / 3 | 1 / 2 | 2 / 3 |
| TOF-A Phase 0 | 2 / 3 | 2 / 3 | 1 / 1 | 2 / 2 | 1 / 2 | 1 / 2 |

The TOF-A gate (clutter and connectivity at least 4) is met on attempt two in
both modes; its connectivity rose with unchanged images, so the rise belongs
to the note that clipped views cut the tree and to a different reviewer. The
CTA-A gate (at least 4 on all six) is not met after two attempts. Remaining
differences named by the reviewers: a sparse tree with few distal branches,
a few small detached pieces in cut views, stair-stepped cut faces, the clip
indicator showing at pane corners, and slightly lumpy vessel walls. This goes
to Shafrir as open decision 6; the bar is unchanged.

**Integrated scene.** In the packaged app the external anatomy view shows the
path-traced volume, the tube wall, a dissector and an aspirator, and an axial
cutaway together inside the frame budget. The check found three faults:

- Path-traced vessels stay still under a tool press (open decision 7). In
  standard mode the wall dents away from the tip, by at most 0.56 mm on this
  crop before the contact limit refuses further motion.
- In standard mode the first displaced frame recoloured the tube wall to its
  layer colour, because rebuilt layers drop the tag volume that marked a tube
  wall. The prepared surface now carries that mark itself. Fixed after the
  packaged build and confirmed in the development tree: the wall stays
  orange through three presses.
- The surgeon view in the path-traced mode draws the crop as an opaque block.
  Not fixed.

**Display verification.** `verify-vessel-display.cjs` passes hidden on a
disposable copy of CTA-A: connected tree by default, amber review overlay on
the toggle, more vessel area in the full source, scope kept through save and
reopen, manifest unchanged. Getting there fixed one app fault and two verifier
faults: settings reported by the viewer carried the display-only entries of
derived vessel parts into the saved view, which the engine then refused; the
verifier acted while a reopen was still in flight; and its pixel rule did not
count the colour of an unclassified CTA layer. On this study the
skull-supported scope shows no vessels, for the supplied annotations too,
because the study has no skull coverage estimate.

## Phase 6 results: finer vessels

Evidence: `qc/medivis-parity-20261125/phase6-finer-vessels/README.md`.

**What was hiding vessels.** Not the calibre rule: on the two CTAs it hid 0
and 1 voxels. The named models stop where a vessel narrows to one or two
voxels, and the scored vesselFM addition (0.7 and 0.9, components touching the
tree only) had been chosen against annotations of named vessels, so it left
out unnamed branches. On CT the vesselFM score outside the tree is also about
one third bone.

**What changed.**

- *Fine branches on CT.* `resectionlab.vessel_fine_growth` follows thin bright
  tubes outward from the tree in the acquired CTA and returns them as a
  separate display-only layer ("Image-derived fine branches"). Rule, frozen on
  CTA-A and applied unchanged to CTA-B, in
  [acquired vascular geometry](native-vascular-geometry.md). Never computed
  for MR.
- *Detached model vessels.* The drawn vesselFM layer now holds every component
  at the display thresholds, connected or not, so the existing
  signal-supported bridging (gaps of at most 3 mm) can join detached pieces;
  the rest stay in the amber review overlay. On CT the thresholds are 0.5 with
  voxels at or above 700 HU left out. On MR 0.5 was tried and rejected: on
  TOF-A it joined a noisy mass at the face and skull base to the tree, so MR
  keeps 0.7 and 0.9. Longer gaps were measured on TOF-A (at 0.5, before the
  wall rule): 118 pieces joined at 3 mm, 149 at 6 mm and 169 at 10 mm, for 2
  and 3 percent more shown voxels and 1.6 and 4.6 times the preparation time,
  so the 3 mm rule was kept.
- *Vessel detail.* A **Vessel detail** control: "With fine branches" (default
  on CT) or "Major vessels only" (default on MR; the Phase 3 calibre rule, and
  the fine-branch layer hidden whole).
- *One colour.* All vessels are one red in both render paths.
- *Reopening.* The connected display is stored beside the study
  (`display-cache/`), so it is prepared once: about 25 to 30 s for the three
  CT layers on an idle machine, then read back.
- *EDEN.* `scripts/topbrain_native.py --eden PTNT_#### run PTNT_####-mr`
  registers a restricted case from its prepared study and runs the three
  teams, the frozen ensemble and the display candidates. PTNT_4503 and
  PTNT_7086 were chosen from the cohort's 2026-11-23 captures as the cleanest
  trees (distinct tubes, few stray pieces, no posterior slab).

| | CTA-A | CTA-B | TOF-A | TOF-B | PTNT_4503 |
| --- | --- | --- | --- | --- | --- |
| Voxels in the default view, Phase 3 → Phase 6 | 84,082 → 101,864 | 73,575 → 83,190 | 377,752 → 379,545 | 353,797 → 354,172 | 434,529 |
| Fine branches grown / shown | 28,598 / 20,662 | 12,272 / 6,406 | not computed (MR) | not computed (MR) | not computed (MR) |
| Detached model pieces joined along signal | 4 | 2 | 14 | 5 | 10 |
| Detached model pieces left in review | 457 | 480 | 806 | 856 | 483 |
| Floating mesh pieces in the default view | 0 of 23 | 0 of 11 | 0 of 5 | 0 of 3 | 0 of 4 |
| Major named vessel lost | none | none | none listed | none listed | none listed |
| Loops / cavities of the shown voxels, Phase 3 → Phase 6 | 27 / 1 → 200 / 22 | 33 / 1 → 107 / 8 | 150 / 6 unchanged | 199 / 2 unchanged | 128 / 2 |
| Dice / clDice against the named-vessel annotation, Phase 3 → Phase 6 | 0.875 / 0.902 → 0.786 / 0.585 | 0.888 / 0.935 → 0.829 / 0.795 | no annotation | no annotation | no annotation |
| Frame interval p95 / settle, volume lighting, **while model inference ran on the same machine** | 10.7 ms / 1.62 s | 10.1 ms / 1.93 s | 10.7 ms / 1.41 s | 10.1 ms / 1.42 s | 10.5 ms / 1.68 s |

Reading the table. On CT the default view holds a fifth more vessel on the
development case and an eighth more on the held-out case. On TOF almost
nothing changes: the trees were already dense, and nearly every detached
model piece is too small or too far from the tree to have signal support, so
"joining fragments" adds 5 to 14 pieces per case. The fine branches cost
topology and agreement with the annotation: thin branches that touch form
loops, and every unnamed branch counts as an error against an annotation of
named vessels only. The Phase 2 gate is measured on the ensemble alone and is
unchanged. The settle times above are not gate measurements: they were taken
under load, and the Phase 1 gate (1.5 s on CTA-A) has to be re-measured on an
idle machine with the richer tree.

**Blind review, third attempt.** Same protocol; CTA-A sets were Phase 0, the
Phase 5 final set and the Phase 6 set, the last two in the same eight
reference framings; TOF-A sets were Phase 0, Phase 5 and Phase 6.

| | Crispness | Depth | Clutter | Connectivity | Realism | Usefulness |
| --- | --- | --- | --- | --- | --- | --- |
| CTA-A Phase 6 | 3 | 4 | 3 | 3 | 3 | 4 |
| CTA-A Phase 5, same reviewer | 4 | 4 | 4 | 3 | 3 | 3 |
| TOF-A Phase 6 | 4 | 3 | 4 | 4 | 3 | 4 |
| TOF-A Phase 5, same reviewer | 4 | 3 | 4 | 4 | 3 | 4 |

The reviewer ranked the Phase 6 CTA-A set first ("wins on vessel coverage")
and named its gap to the reference as vessel surface quality: lumpy, beaded
thin segments and scattered small stubs. The CTA-A gate is not met. The TOF-A
gate (clutter and connectivity at least 4) holds; the reviewer could not tell
the Phase 5 and Phase 6 TOF-A sets apart.

**Coordinator's reading of the captures** (`visual-assessments.md`). With an
opaque skull and a cut, CTA-A now shows what the clip shows: a dense connected
red tree over ivory bone, cortical branches and veins up to the sagittal
sinus. Still short of it: thin branches are ragged, the uncut see-through
anterior view is a tangle at the midline, and the opaque uncut views show no
vessel at all because this scan has no neck. CTA-B gains less than CTA-A.

**Not done or not verified.** Frame timing on an idle machine and in the
packaged build (the build in `desktop/release/` predates this phase).
Standard-mode captures of Phase 6. The rehearsal tube model and the vessel
rods still come from the named tree; fine branches are display only.
PTNT_7086's first run stopped on non-finite network output while a hidden
capture shared the GPU and was restarted; its capture is added when it
finishes.

## Handoff, 2026-11-26

State of the gates:

| Phase | Gate | State |
| --- | --- | --- |
| 1 | Interaction p95 at most 16.7 ms, settle at most 1.5 s on CTA-A | met (10.9 ms, 1.42 s worst of three packaged runs) |
| 1 | Blind crispness and depth at least 4 | met with an opaque skull or a cut (4, 4); 3, 3 with the see-through skull |
| 2 | Better Dice, clDice and Betti than vesselFM alone on the held-out cases | Dice, clDice, component count and patch Betti met on both; whole-volume loop count on CTA-B not met (9 against 7); TOF-B scored on a TopCoW MRA stand-in |
| 3 | Zero floating components by default; no named major vessel lost | met on the drawn meshes of all four cases |
| 4 | Deviation at most 0.5 voxel; no self-intersection; tools displace vessels | met on TopCoW behind the tube geometry option, in standard mode |
| 5 | Phase 1 frame budget with volume, tubes, tools and clipping in the packaged build | met |
| 5 | Blind at least 4 on all criteria for CTA-A | not met after three attempts (4, 4, 3, 3, 3, 3; then 3, 4, 3, 3, 3, 4 with the finer vessels of Phase 6) |
| 5 | Blind at least 4 on clutter and connectivity for TOF-A | met on the second and third attempts (4, 4) |
| 5 | All earlier gates hold | not in the path-traced mode during rehearsal: vessels do not follow a tool press |

What ships by default: the connected, rooted vessel tree from the model
ensemble in standard mode, in one vessel colour, with the review toggle; on CT
with the image-derived fine branches (Vessel detail). Behind settings: the
path-traced mode (Volume lighting) with its Bone choice, and the tube geometry
option for rehearsal cases.

Not done: the pre-registered TopCoW-001 confirmation of the Phase 2 ensemble;
EDEN cases in the simulator (not admitted for simulation); a tag volume that
follows cuts and displacement; the surgeon view in the path-traced mode.

Open decisions 1 to 8 above need Shafrir. This branch also carries
`shaf/interactive-operative-practice-20261121` and
`shaf/rehearsal-performance-20261121`, merged in at Shafrir's request; the
pull request to `main` itself was not merged.

