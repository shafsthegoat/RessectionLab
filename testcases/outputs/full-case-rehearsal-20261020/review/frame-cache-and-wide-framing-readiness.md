# Read-only playback readiness review

Scope: saved native JSON/checkpoint records and source inspection only. No patient engine was constructed, no physics or GUI ran, and no production file changed. The live process continued independently. Data estimate captured the saved primary at 1,656 actions / 3,120.741 simulated seconds; it is not the final acceptance artifact.

## Frame disk capacity

- Expected cadence-plus-boundary frame count: approximately **17,260–18,915**, derived from recorded elapsed time and requested action durations. Exact count remains an observed-replay result.
- Exact endpoint gzip-1 field sizes using the store's encoding: `flapComponents` 188,663 B; `drillingV2` 132,021 B; `removed` 120,610 B; `debris` 36,661 B. Stable fields deduplicate, while even a changed cooling scalar causes the whole drilling field to acquire a different blob.
- Across six recorded checkpoints, duration-weighted endpoint-size scenarios are **1.566 GiB** if drilling state varies per frame, or **3.624 GiB** if every non-flap field varies. Applying the latest non-flap size to all frames gives **5.116 GiB**. These are scenarios of unseen intermediate states, not measured cache bytes or rigorous bounds; reuse can reduce them substantially.
- The 512 MiB default is not a defensible complete-history readiness assumption. Existing 64 MiB per-action reservation means `_room(reserve, pending)` can delete the current committed branch entry once approximately 448 MiB is retained, losing all prior frames together. Cumulative publication count does not prove retention.
- Practical bounded configuration: **8 GiB observed-frame store**, while retaining 64 MiB RAM / 128 MiB transient JSON / 64 MiB per-action staging. This gives margin above the source-record scenarios and remains finite. The shared verified-cache owner's current constructor ceiling is 512 MiB, so simply passing a larger observed-cache argument currently fails. Root assigned the scoped configuration and current-entry retention repair separately.
- Final acceptance must record actual same-pass bytes, every action manifest still present, every authenticated frame/boundary readable in a fresh process without physics, and actual GUI behavior. No estimate replaces those checks. The separate canonical verified-boundary cache must also successfully publish before cache-only telemetry admission.

Detailed captured sizes, intervals, source SHA and limitations: `frame-disk-readiness-source-records.json`. Reproducible source-only reader: `frame_disk_readiness.py`.

## Helper readiness

`accept_native.py` performs one independent force-verified canonical restore and seals per-action boundary states after receipt/source assertions. It does not collect observed microstep frames. `export_observed_telemetry.py --prepare-frames` then collects frames during the same required descriptive telemetry replay; it requires publication count equal to actions **and** every action manifest present before export. `verify_every_cached_frame.py` independently reads all authenticated fields/frames, checks canonical endpoints, and uses profiling to refuse public physics calls. These are appropriately distinct checks; none asserts GUI verification.

## Wide framing

The native surgical path is `setSurgical` / `surgicalGroup` (`VolumeRenderer.ts:1069`). Generic `fitCamera('instruments')` unions `recordedTools`, `inspectionTools`, or generic `tools`, plus generic replay bounds (`:1721`); it never unions `surgicalGroup`. The camera-mode buttons are hidden unless `workbenchTab==='research'` (`App.tsx:1760`). Therefore generic instrument fitting does **not** establish visibility of both native surgical bodies in Rehearse. The saved primary currently has an empty `imagingState.states.primary.viewer` object, so it has no explicit wide camera.

Smallest no-production-code route: before freezing the final native SHA, save an explicit `imagingState.states.primary.viewer.camera` (`position`, `target`, `up`) fitted conservatively to (a) desired source/target context and (b) both instruments' complete envelopes at saved initial slots and all recorded requested path poses. Requested poses are used only to bound the camera, never as a claim of actual observed motion. Use a conservative orientation-independent margin enclosing shaft/handle geometry and recorded radius/jaw parameters, and fit both horizontal and vertical FOV at the narrowest supported pane aspect. Bodies extend backward along local -Z; a tip-only bound is insufficient. `restoreCamera` validates/restores this setting and suppresses the asynchronous anatomy auto-fit.

On initial open this gives a wide overview; the existing explicit **Frame operative field** remains the detail route. This saved-camera approach does not add a one-click return to the wide overview after detail framing; orbit/dolly remain available. If that explicit return is required, the smallest later display-only control is a **Frame anatomy and tools** action that unions the actual native surgicalGroup/body bounds. No auto-framing on scrub, source mutation, interpolated motion, or engine changes are needed.
