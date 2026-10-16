# First executable patient-source rehearsal

The October 16 first release gate is demonstrated in the actual Electron app.
This closes the narrow software proof for Gates 1–2; it does not complete the
patient-corpus, anatomical-accuracy or physiological-validation objective.

The branch starts from merged PR #12 (`cef9705`). Original unfinished work in the
other checkout is preserved. The source MRI/CT/MRA and all model weights stay
outside Git. No patient images were uploaded, no new terms accepted, no training
campaign run and no protected SELECT/UPENN payload opened for this milestone.

## What operates in Electron

Open a genuine glioma MRI, acquired CT skull case or acquired TOF angiography
case from the library. The source record shows native grid/frame, archived
originals, model/checkpoint identity and rights. Layer visibility/opacity,
orbit/zoom, cutaway and linked MRI/CT/MRA slices work. Save/reopen retains source
arrays, masks, originals and view settings.

On UCSF-PDGM-0004, choose **Rehearse → Prepare rehearsal**. The post-exposure
research state has two persistent full-length instruments on one clock:
aspiration removes conservatively contacted tissue; field management accumulates
contact and reduces a dimensionless generated-debris proxy without removing
tissue. Inspect, reassess, motion, exchange and STOP use the same authoritative
Python state. The initial exposure is an explicit hypothesis, separately
accounted from new removal and never called an observed cavity or safe approach.

The manual demonstration committed two 1 mm aspiration actions, then a
field-management action. **Compare two continuations** executed both strategies
from that exact nonempty snapshot:

| Quantity | Short pass + field management | Deeper pass + reassess |
| --- | ---: | ---: |
| Target removed | 55 mm³ | 139 mm³ |
| Non-target removed | 9 mm³ | 9 mm³ |
| Residual target | 22,498 mm³ | 22,414 mm³ |
| Field burden | 0 model units | 0.252 model units |
| Modeled anatomical conflicts | 0 | 0 |
| Shared simulated elapsed time | 9 s | 9 s |
| Initial assumed tissue exposure | 12,628 mm³ | 12,628 mm³ |
| Vascular events / motor / language | Unknown | Unknown |

Play, pause, slider scrubbing, tool selection, the committed timeline and both
branch views were operated through native computer use. New removal and
field-contact overlays appear in the same 3D/MPR scene; acquired scalar data and
source annotations remain unchanged. The native Save dialog wrote a separate
review bundle. After quitting Electron (process exit 0), a fresh process reopened
it with the short branch, frame 19/19, field manager and paused playback restored.

## Real inference evidence reused and verified

| Patient source | Saved inference | Recorded runtime / peak RSS | Limits |
| --- | --- | --- | --- |
| UCSF-PDGM-0004 | SynthStrip extraction; MONAI BraTS 0.5.4 on registered T1/T1ce/T2/FLAIR | 4.21 s / 0.90 GiB; 7.23 s / 1.29 GiB | Pinned preprocessed mirror; prior skull stripping and known BraTS training overlap. No acquired skull/angiography/function. |
| SynthRAD2023-1BA336 | TotalSegmentator CT brain and skull, fast 3 mm | 9.46 s / 3.44 GiB | Acquired CT on a separate person; cropped/defaced coverage and skull surface uncertainty. |
| Lausanne-sub-000-ses-20110101 | vesselFM base on acquired TOF-MRA | 81.34 s / 1.82 GiB | Regional coverage, extracranial signal and false positives; T1 registration unresolved. Research/noncommercial checkpoint. |

These are real retained inference records, not new timings from this milestone.
The independent artifact audit rehashed weights, inputs and outputs, verified
source grids/affines and preserved all cached artifacts. Current Electron
inspection and save/reopen covered all three named patients, never a fusion of
different people. The development tumor overlap is not external accuracy.

## Independent checks and retained failures

- Runtime controls: source/frame binding, two-tool and full-shaft swept collision,
  irreversible removal/contact, synchronized motion, clock, atomic refusal,
  exact branching/replay, accounting and metadata-only protected-role refusal.
- An independent cell/capsule oracle checks every recorded real-source frame,
  target/non-target conservation, contacts and field burden. Private-reference
  and future-metadata interventions do not change deployed observations.
- Actual host `Sidecar` protocol test uses two fresh Python processes and checks
  create/step/compare/save/shutdown/reopen plus the frontend adapter and replay
  masks. Stale states, injected source/authority fields and renderer-supplied
  authoritative histories are refused. Model imports/subprocesses are blocked
  during that check and none are attempted.
- [GUI-save audit](gui-save-audit.json) independently replays the actual
  Electron-created bundles. Named source arrays, provenance, auxiliary sources
  and all 32 embedded original files are exact. The UCSF serializer reorders
  numbered NPZ entries; the complete NPY byte multiset and named arrays remain
  exact. The original overly strict audit failure remains in local evidence.
- The first actual Prepare action exposed a missing JavaScript Sidecar operation
  whitelist. The fix and failure are retained in the protocol report. Renderer
  inspection then exposed fixed residual opacity and ignored cutaways; derived
  meshes now follow source appearance, clipping and bounded smoothing, with
  tests before/during/after asynchronous surface arrival. Full instruments stay
  visible for geometry review.
- Full desktop tests passed, including 123 main-process checks; the final viewer
  run passed 133 checks. The TypeScript/production build passed. The first full
  renderer run found two new-hook fixture omissions and a Vite temporary-cache
  cleanup race; both were repaired and the full desktop suite rerun successfully.

Reproduce with `scripts/launch_poc.sh --case <reviewed bundle>`. The independent
opt-in verifiers are `tests/first_release_qa.py`,
`tests/first_release_sidecar.mjs` and `tests/first_release_gui_save.py`; use this
checkout's `src` on `PYTHONPATH` when sharing an older virtual environment.
Local demonstration bundles are under `outputs/poc-rehearsal/`; original audit
reports, logs and actual screenshots are under `outputs/poc-first-qa/`.
[UI manifest](ui-demonstration.json) binds screenshots and observed actions by
hash. Images, weights and patient archives are intentionally not in Git.

## What remains open

The runtime is a bounded fixed-axis post-exposure research model. It has no
calibrated forces, brain shift, fluid mass/flow, thermal injury, bleeding-source
control, vessel sealing, perfusion or neurological outcome model. Field manager
is deliberately not called a coagulator. Missing vessels/function are unknown;
zero modeled conflicts is not evidence of surgical safety. A 3D camera orbit
is not an operative microscope observation model.

Native patient collections require an adapter before using this rehearsal;
the current episode consumes canonical research case bundles. Real population
accuracy, full-operation fidelity and publication readiness remain unproven.
The [operative workflow](../../docs/operative-workflow-model.md) maps full surgery
and its OR context to source-linked observations and future validation, keeping
all 14 phases visible without pretending this episode already implements them.

Next: run the resumable bundle-qualification pipeline on existing eligible
TRAIN/development people, register/QC one publisher-described healthy MRI/MRA
host, then run one provenance-bound biophysical virtual-glioma pilot. Only after
that pilot qualifies should the planned real/virtual diversity tranche expand.
