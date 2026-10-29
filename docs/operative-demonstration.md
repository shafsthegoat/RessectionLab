# Source-bound operative demonstration

This continues the existing rehearsal. The microscope uses opaque source-owned
surfaces, an independent fixed camera, operative material/lighting, and true
geometry shadows. Illustrative colours, roughness and restrained albedo variation
add no vessels, sulci, membranes, geometry, or acquired colour evidence. Linked
MRI keeps the original label colours. Native source resolution remains 1 mm;
optical magnification does not increase physical resolution.

The original weak demo had both insufficient work and a CUSA rear-shoulder
mismatch. Revision 2 corrects that declared proxy explicitly while preserving
old replay hashes and the unchanged 2 mm³/s finite capacity. A checked overlapping
raster followed by central deepening now removes 349 unique owned cells:
219 mm³ target and 130 mm³ brain. The prior authored exposure already removed
560 mm³ target and 4,940 mm³ brain; it is not counted as new finite CUSA work.
Target residual after the new sequence is 18,589 mm³ of 19,368 mm³ initial target.
This is a local demonstration, not a complete resection or validated access route.
The input supplies brain and target; skull, full vessels and functional anatomy
remain unavailable. The camera never creates an opening.

Frazier is explicitly active during left CUSA apply actions. Those actions
remove target while collecting 10.36 mm³ of modeled irrigation; overall 12 mm³
is collected, with no generated bleeding or loose CUSA debris. Later right-hand
moves show checked independent motion but collect no additional fluid because
that field has already cleared. The blue equivalent-volume glyphs represent
modeled cell contents, not a reconstructed liquid surface. Partial actions stop
while native work accumulates; an inserted-tool exchange is refused.

The same final prefix branches into supine, semisitting and left lateral, each
with one second of settling. All keep 6 mm³ irrigation but have different fluid
centroids and local displacement. Maximum modeled shift is approximately
0.02771 / 0.02945 / 0.02741 mm respectively. Semisitting uses the declared
300 mm heart-to-site vector, yielding estimated vertical height 259.81 mm and
static-column offset −2700.71 Pa (−20.257 mmHg); these are not measured vitals.
Camera changes leave the complete native state unchanged. See the existing
[position model](patient-position-model.md) for assumptions and unknowns.

## Local reproduction

The source images and raw captures are intentionally excluded from Git. Exact
local preparation and evidence are under:

`/Users/shaf/coding_projects/RessectionLab/testcases/outputs/operative-visual-20261028/`

- `operative-v4/operative-rehearsal.ressectionlab`: source-bound 120-action episode
  and three 121-action posture branches; initial cursor 16.
- `operative-v4/preparation.json`: canonical timings, statuses, per-action volumes,
  source identities, posture effects and replay result.
- `operative-v4/independent-accounting.json`: independent original-label cell
  counts, instantaneous contact membership, conserved ledgers and matched tails.
- `visual-gate-v2/`: inspected fixed-camera before, four intermediates, after,
  tools-clear, scrub-restored and inspection-only native GPU images. This earlier
  episode has the same removal geometry and a longer authored dwell time.
- `backend-relevant.log`: 402 surgical, replay and persistence tests passed.
- `desktop-full.log`: 1,108 Electron-main, renderer and viewer tests passed.
- `backend-full.log`: broader run interrupted after unrelated legacy training
  failures and absent local source fixtures; this is not a passing full suite.

Prepare again from the already exposed case, choosing a new ignored directory:

```sh
.venv/bin/python scripts/prepare_operative_demonstration.py \
  testcases/outputs/surgeon-physics-20261028/exposed-example-v5/surgeon-rehearsal.ressectionlab \
  testcases/outputs/operative-visual-20261028/another-preparation
```

Capture and verify with the actual GPU Electron application and native engine:

```sh
OPERATIVE_CASE="$PWD/testcases/outputs/operative-visual-20261028/operative-v4/operative-rehearsal.ressectionlab" \
OPERATIVE_OUTPUT="$PWD/testcases/outputs/operative-visual-20261028/another-verification" \
OPERATIVE_RECORD=1 \
desktop/node_modules/.bin/electron desktop/electron/verify-operative-field.cjs
```

The harness always uses `--automation-hidden` and `stayHidden` captures. It refuses
output overwrites and records whether any window was shown. Source input bytes
must remain unchanged. The recording samples only actual GPU-visible native
states whose surface, HUD and timeline identities agree; it does not interpolate
anatomy or speed up removal physics. Playback may buffer to prepare exact surfaces.

## Measured display cost

A five-state GPU profile of the same source (`smoothing-profile-v1/report.json`)
measured brain surface preparation at 6.21–7.72 seconds per changed state, of
which 6.16–7.45 seconds was global display fairing. Each such state remeshed only
one local chunk. Tumor preparation was 0.74–1.16 seconds. GPU submission-to-fence
poll upper bounds had median 29.1 ms and P95 110 ms; these are not GPU timer-query
measurements or guaranteed FPS. The slow path is whole-layer conditioning after
local mesh edits, rather than source-cell physics or a need for a fake animation.

With the existing Smooth 3D surfaces control off, brain preparation for the same
sampled states was 71–100 ms and tumor 30–37 ms. The exact voxel terraces are
visually coarser. Both display choices preserve the same physical episode; the
main evidence uses smoothing for cavity readability and reports its buffering.
The native complete 120-action preparation takes 33.23 wall seconds for
235.69 simulated seconds without collecting/displaying all intermediate states.

The material selector was switched through source labels and illustrative
material without changing native state, saved, and reopened in a fresh Electron
process (`material-cold-reopen-v1/report.json`). That process also confirmed an
inserted exchange fails with a plain-language explanation and unchanged anatomy.

## Open the delivered episode

```sh
./scripts/launch_poc.sh --secondary-display --case \
  /Users/shaf/coding_projects/RessectionLab/testcases/cases/UCSF-PDGM-0007-operative-demo.ressectionlab
```

Choose **Rehearse → Play replay**. The saved example starts at boundary 16 with
both tools inserted and the fixed microscope already selected. Select
`position-supine`, `position-semisitting` or `position-left_lateral` to compare
the matched tails, then open **Patient position and gravity**.

The final `integration-v1/report.json` passed real native automatic replay,
pause/resume, exact scrubbing, inspection transparency, refused exchange,
external/surgeon switching, the three posture branches, camera invariance,
Undo/Redo, Branch, an actual new prone action and native save. Its 255 sampled
GPU images span native 6.8–233.0 s; all retained frame identities agree with the
surface and HUD. Playback took 1152.567 wall seconds, including capturing;
2,923 frame submissions had CPU median 0.4 ms and P95 11.7 ms. This is buffered
replay rather than a real-time claim. The final cold process
`final-cold-reopen-v1/report.json` restored the same native state, both poses,
optical camera, cursor and material; no verifier window was ever visible.

The delivered local movie is
`integration-v1/operative-rehearsal.mp4`, with source-state/volume/timing manifest
`integration-v1/recording-manifest.json`. It uses actual accepted GPU captures,
4× native intervals, and static before/after holds; frames are held without
interpolating tissue. Both source MRI and labels remain unchanged.

Before, intermediate and after native captures are in the same directory:
`01-before-field.png`, `02-progress-01-field.png`, `02-progress-02-field.png`,
`03-after-field.png` and `03b-after-tools-clear-field.png`. Complete app captures
include MRI and the research assumptions; `05-inspection-only.png` labels the
see-through view explicitly. No raw patient image or video is committed.
