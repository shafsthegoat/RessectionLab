# Integrated acquired-anatomy rehearsal

The October 18, 2026 protocol uses the existing instrument engine, native patient
bundle, Electron Rehearse panel and `SurgicalRLEnv`. Acquired images and source
annotations are distinct from scripted simulated actions and model-derived
responses. Anatomical and clinical validation remain absent.

## Saved episodes and reproduction

From the existing primary checkout:

```sh
.venv/bin/python scripts/verify_integrated_rehearsal.py
./scripts/launch_poc.sh --case "$PWD/outputs/integrated-rehearsal-v1/sub06.ressectionlab"
```

Choose **Rehearse** and retain the saved environment. Its `main` and
`ready-for-cusa` branches share the same supplied source anatomy and protection
contract. The [frozen protocol](../artifacts/integrated-rehearsal-v1/protocol.json)
pins both input files and records every commanded action, including blocked
approaches. Reproduction requires those already acquired local bundles and
protection manifests; no acquisition or model inference is launched.

On the same-person `sub-06` CT/MRI derivative, eleven work-limited drill strokes
remove 710 mm³ of supplied bone. Tool exchanges preserve that opening. CUSA
creates a small target cavity, then a retractor holds one source target cell
while the other CUSA removes another 3 mm³. The final state has 57 mm³ of target
removed, one retained held source cell, and approximately 0.667 mm maximum
modeled displacement. Healthy soft-tissue removal is zero on this scripted
prefix; bone removal is counted separately. These numbers describe the numerical
mechanisms, not a recommended access route or a safe resection.

The first wider approach stops at incompatible material, and an exchange while
inserted is rejected. A prior attempted final state removed the held cell too;
that failed retained-tissue criterion is preserved in the
[independent finding](../artifacts/integrated-rehearsal-v1/telemetry-empty-hold-trial.json).
An empty hold record is never accepted as retained tissue.

The identical API also runs the acquired `UCSF-PDGM-0004` MRI derivative. Its
scripted exposure removes 8 mm³ target and 16 mm³ other soft tissue. A separate
drill approach stops with `UNSUPPORTED_MATERIAL`; skull remains unavailable.
Missing dura, attachment topology, complete vascular evidence and functional
maps remain unknown. Both source identities are development cases, and no
reserved final evaluation patient is used.

The [native replay receipts](../artifacts/integrated-rehearsal-v1/case-verification.json)
check source image/label identity and fresh-process native restoration. The
[actual desktop verification](../artifacts/integrated-rehearsal-v1/desktop-verification.json)
records drill preview/commit, Undo/Redo, the final two-hand actions, a new
second-case drill rejection, native Save dialogs and fresh Electron openings.
In the reopened retained-hold episode, choose **Translate tool** and drag a 3D
arrow to edit a proposed pose; **Cancel** restores the recorded pose without
advancing the shared clock. This was exercised directly in the viewport.
The [independent telemetry audit](../artifacts/integrated-rehearsal-v1/telemetry-verification.json)
additionally verifies retained tissue before and after the CUSA removal
microsteps, RAS-mm geometry and source mask identity.

The separate `sub06-vascular.ressectionlab` continuation preserves an earlier
target-58 development prefix and is explicitly distinct from the corrected
retained-hold demonstration. Its 22 further accepted actions create a modeled
wound, control its outlet with bipolar, aspirate modeled fluid and irrigate.
Five damaged vessel faces remain nonpatent. See the
[vascular receipt](../artifacts/integrated-rehearsal-v1/vascular-verification.json)
and [independent telemetry check](../artifacts/integrated-rehearsal-v1/telemetry-vascular-verification.json).
Generated fluid, field clearance and vessel patency are separate model outputs;
none estimates patient blood loss or global perfusion.

The first larger vascular save exceeded the reader's 16 MiB manifest limit
because of JSON indentation. Its recovery preserves every JSON value and array
byte. New native saves use compact JSON and reject oversized metadata before
replacing an existing destination. This preserves the existing open-file budget.

## Protection and measured cost

[Shared engine protection](shared-surgical-guards.md) enforces the same supplied
exclusion masks for desktop and RL actions, including subsequent actions after
native Save/Open. Preview, branch and reset preserve that contract. The UI names
strict, relaxed or unconfigured research mode; supplied masks never establish
anatomical completeness.

On a fixed acquired input, engine step median/p95 changes from 2,024/2,322 ms
before the measured hashing optimization to 1,410/1,555 ms afterward. Full RL
`step_action` currently measures 1,455/1,548 ms; a matching historical full-wrapper
baseline was not recorded. Five repeats describe this workload only. Patient
CPU render preparation measures 172/248 ms for a changed state and 0.76/1.23 ms
for a cached state; this excludes GPU presentation, IPC and image decoding.
See the [render receipt](../artifacts/integrated-rehearsal-v1/render-timing.json).

## Referenced behavior-measurement paper

We referenced **Yilmaz et al., “Continuous monitoring of surgical bimanual
expertise using deep neural networks in virtual reality simulation,” npj Digital
Medicine 5, 54 (2022)**,
[doi:10.1038/s41746-022-00596-8](https://doi.org/10.1038/s41746-022-00596-8).
Its interpretable bimanual features and 0.2-second cadence informed the
[telemetry implementation](surgical-bimanual-telemetry.md). The paper, released
analysis code and sample CSV were consulted with rights/format checks documented
there. No external code, expertise model, invented expert labels or patient risk
calibration is transferred.

The local inspectable outputs are
`outputs/integrated-rehearsal-v1/telemetry/telemetry.html`, `.json` and `.csv`.
They preserve actual microstep timestamps, both poses and activations, motion,
removal, fluid and events, with unknown measured forces and clinical injury.
They support descriptive manual/script/search/RL labels without claiming those
labels establish comparable tasks or surgical expertise.

The [fixed-seed policy comparison](../artifacts/integrated-rehearsal-v1/telemetry-policy-comparison.json)
exports actual recorded script, observed-greedy-local and trained trajectories
from seed 17 on the identical sub-06 exposure prefix. Its local interactive
report is `outputs/integrated-rehearsal-v1/telemetry-policy-comparison/telemetry.html`.
This descriptive replay adds no policy choices or optimizer updates. Its 61
verification calls and 50.934 seconds are separate from the pilot's runner cost.

The bounded [learning pilot](surgical-pilot.md) uses the existing RL wrapper and
the same verified execution engine. Its report retains all seeds and baselines,
actual optimizer updates, setup harm, wall time and blocking conditions. Useful
learned two-instrument control remains an unmet criterion when the supplied task
has no supported second-instrument effect.

The [completed pilot result](../artifacts/surgical-pilot-v1/RESULT.md) records
12 real updates over three seeds and six exact final replays. Learned assessment
adds no target removal on either anatomy; no checkpoint is promoted. Sub-06's
scripted baseline removes 108 additional mm³ with zero healthy removal. UCSF's
scripted arm removes 6 mm³ healthy tissue, exceeding the local 5 mm³ threshold,
and is explicitly a budget failure. Useful second-instrument learning is blocked
by the zero-effect Frazier probes in the declared post-exposure task. These are
development results on two anatomies used in fitting, not evidence of transfer.

Final verification passes 270 focused Python tests and the full desktop test
suite; the production desktop build also passes. The actual desktop native saves
are `outputs/integrated-rehearsal-v1/desktop-bimanual.ressectionlab` and
`outputs/integrated-rehearsal-v1/case-two.ressectionlab`. Fresh-process reads of
both reproduce their expected states; the former is identical to the corrected
scripted checkpoint, including retained tissue.

## Deterministic full-case rehearsal

The Rehearse workbench now exposes Primary resection and Vascular injury / rescue
directors. They observe the shared engine, propose bounded continuous actions,
preview them, and commit only matching public transitions. The current patient
route uses the supplied sub-06 source grid and matching exclusion evidence.
Superior skull openings, fine frontiers, interior target pockets, second-hand
aspiration and checked withdrawal retain complete instrument/shaft collisions.
The director refuses healthy/unknown tissue removal and revisits joined openings
only while a bounded previous pass made target progress. Its configured CUSA
working aperture can reach 6 mm radius in the generic research model; this is
an uncalibrated model parameter, not evidence of clinical tool feasibility.
Scalp, dura, attachments and complete functional/vascular anatomy remain UNKNOWN.

Before its endpoint, the primary director can revisit the established assisting
Frazier ray for at most three positive field stations and ten seconds of dwell
per station. Both hands first deactivate, release and withdraw; preparation,
approach, aspiration and final withdrawal use the public action API. Overlapping
fields are freshly measured at the actual tool pose, and recorded aspiration
uses actual fluid/debris ledger changes. A blocked approach or nonprogressing
dwell ends this bounded pass; retained field material remains in the ledger.

An explicit directional fallback can be configured at a stopped primary
residual endpoint after both tools are released and withdrawn. Native axis 0/1
proposals preserve original cell identities and require the same public guards.
The director records the preceding access pass, then performs continuous checked
exterior transit and a remote orientation change before approaching the new
opening. A rejected or partial transit stops at its actual state. These bounded
source-column opportunities are uncalibrated engineering candidates; they do
not establish clinical approach feasibility or global target reachability.

Pause computation ends automatic scheduling after the current bounded slice.
Replay observes actual engine microsteps at the existing 0.2-second cadence;
intermediate frames are read-only and their final boundary must equal verified
replay. Return to a recorded boundary before branching or resuming. Save pauses
both computation and playback and synchronizes the canonical editable state.
Fluid glyphs show per-cell ledger volumes; outlet reservoirs remain separate.

Observed frames can be prepared during the same descriptive telemetry replay.
Each action's exact outcome and endpoint must match canonical admission before
its frames receive an authenticated local cache entry. The cache binds source,
complete branch, history and loaded producer implementation; corrupt, missing or
obsolete entries fall back to actual observed replay. Fresh-process frame reads
can therefore avoid recomputing dense tool paths during playback. This cache is
disposable and does not replace the native action log or independent verification.

Long full-case native archives use bounded lossless JSON compression. A disposable
local authenticated cache is published only after complete canonical replay,
source/implementation checks and checkpoint equality. Missing or invalid seals
fall back to full replay. Cold admission is cancellable and has a separate time
budget from ordinary instrument actions. Patient episodes and acceptance receipts
belong under ignored `testcases/outputs/`; no patient result is implied by this
implementation description.

Repeated saves reuse compressed source-array payloads only for identical views
backed by immutable bytes. The bounded cache retains weak references rather
than source arrays; mutable storage and changed array layouts are re-encoded.
Each save still rebuilds its artifacts, manifests and workspace identities.

For display-constrained sessions, launch with `--secondary-display`. The app
selects a nonprimary display before creating its window, refuses launch when
none exists, and hides the window if that display disconnects or becomes primary.
