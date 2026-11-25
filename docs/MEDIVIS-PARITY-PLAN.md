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
| Dice, clDice, Betti | Dice and clDice inside the annotation's bounding region padded by 5 mm. Betti numbers on whole predicted components that enter that region. |
| Blind review | A fresh reviewer scores shuffled, unlabeled frames 1 to 5 on crispness, depth, clutter, connectivity, realism and usefulness beside the reference frames. |

## Open decisions

1. **No annotation exists for either EDEN case.** The Phase 2 gate asks for
   better Dice, clDice and Betti on TOF-B, which cannot be computed there.
   Proposed: score the labelled TopCoW-018 MRA as the held-out angiography
   reference, and judge PTNT_9937 on floating components, Betti counts and
   blind review. Needs Shafrir's agreement before Phase 2 closes.
2. **The TopBrain dockers need a container runtime.** None is installed, the
   five images total 36.7 GB compressed, and the internal disk has 21 GB free.
   Runtime storage must sit on the HDD. Installing a runtime needs Shafrir's
   approval. The record is CC BY 4.0 with a citation request.
3. **The annotation itself has unrooted named vessels.** TopCoW-014 has six
   and TopCoW-018 four components (M2, M3, choroidal and ophthalmic segments)
   that do not touch a root label. "Zero floating components" and "no named
   vessel lost" conflict there unless signal-supported bridging joins them.
   Phase 3 will report floating components against the annotation's own
   count, and bridged versus hidden named segments separately.
4. **Venous walls beside bone do not resolve.** The edge model expects lumen
   brighter than its surroundings. Only 14 to 57 percent of sinus wall vertices
   find such an edge on CTA, so the half-voxel deviation gate is meaningful
   for arteries and TOF vessels, and venous deviation is reported unresolved.
5. **Python suite baseline.** At `9b079d28` the full suite has 571 failures
   and 515 errors from data-policy gates and absent local data, none in vessel
   code. The saved list is the reference: phases must add no new failure.
6. **Earlier uncommitted work is stashed.** Twenty-three edited files from the
   previous session (CTA receiver shadows, vessel-rod caching, walkthrough
   documentation) are in `stash@{0}` on the old branch and are not part of
   this branch.

## Phases

| Phase | Scope | Acceptance | Status |
| --- | --- | --- | --- |
| 0 Baseline | Headless harness reproduces presets and metrics for today's app | Evidence saved for all four cases | done 2026-11-25 |
| 1 Cinematic renderer | Soft-label tagging, tricubic intensity, progressive path tracing, hybrid shading, HDR environment; WebGPU versus WebGL2 recorded | Interaction p95 at most 16.7 ms; settle at most 1.5 s on CTA-A; blind crispness and depth at least 4 | not started |
| 2 Segmentation | TopBrain 2025 top-three models, ensembled with vesselFM where connected to the named tree; soft probabilities kept; CTA labels never applied to MRA | Better Dice, clDice and Betti than vesselFM alone on the held-out cases | not started |
| 3 Cleanup | Root-reachable components by default, the rest behind a review toggle; signal-supported bridging; caliber LOD; arteries red, veins blue | Zero floating components by default; no named major vessel lost | not started |
| 4 Sim geometry | Centerline and radius implicit tubes shared by drawing, collision, rods and tags | Deviation at most 0.5 voxel; no self-intersection; tools displace vessels | not started |
| 5 Integration | Volume, tubes, tools and clipping in the packaged build; films and pull request | All earlier gates; blind scores at least 4 | not started |

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
