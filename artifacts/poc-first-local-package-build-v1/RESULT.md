# Current local Mac research preview

The Apple Silicon app is built and operated from:

`desktop/release/RessectionLab-darwin-arm64/RessectionLab.app`

It contains the current Electron workbench and numerical engine from source
commit `78edfe44b7c91daed1877be9d4c0d3830d36a8fd`. It opens existing portable
case bundles and remembers a native-selected source-checked case library.
Patient volumes, saved review copies, model weights and the app binary remain
outside Git. The app is left open on the saved UCSF rehearsal for review.

This is a **local research preview**. It does not establish anatomical accuracy,
physiological calibration, clinical suitability or public-distribution approval.

## Build and preserved first failure

The first backend build completed compilation but failed strict notice capture:
the installed, included `psutil==7.2.2` was missing from the captured lock. The
[actual closure audit](dependency-recovery-preparation/preparation.json) checked
30 included distributions and 5,981 source files. The correction adds exactly
that version pin; no installed dependency, exclusion or notice guard changed.
All failed files were preserved by same-filesystem rename, with every member
verified before and after. The original failure receipt and charged bytes remain.

The separately released recovery succeeds in 48.736 seconds; app packaging
succeeds in 8.558 seconds using the existing checksum-verified Electron 44.5.1
archive. There is no download or automatic retry. The cumulative build accounting
is 6,525,685,722 bytes, including the failed attempt, below the unchanged 8 GiB
limit. Neither stage hits a resource stop. These are sampled stop limits, not OS
hard caps. The release/supervision JSON files retain exact inputs and measurements.

Both outer-app deep/strict verification and a separate strict verification of
the bundled executable pass. The engine architecture is arm64. Signing is local
ad hoc; the app is not notarized.

| Final binding | SHA256 |
| --- | --- |
| Bundled engine executable | `bd8e656abce7e9b9e07aade58f3104cd7d2874634acef39b12fdb61201db306b` |
| `app.asar` | `7ded25f4150ebe6360fcc1e686448e9765491ca9dd2c33d850e061b35ab46359` |
| Packaged desktop/engine input manifest | `aeb2f77676eec57b6737f550d83766e481d92976d08ceaa2168eddc3c9f91386` |
| Embedded Python source manifest | `5396845ef6a780e6f223ea46787dc4e2e325287ed44956ca45ad9fd24eca6df0` |

## Exact packaged-engine check

The [actual audit](../first-release-frozen-sidecar-run-v2/compact-summary.json)
passes in 18.218 seconds. It executes the emitted engine through host modules
extracted from this exact `app.asar`, from neutral working directories with a
cleaned launch environment and the real inherited HOME. It verifies every emitted
engine file against the package inventory: **zero byte differences**. It does
not claim an internal OS sandbox or absence of uninstrumented internal activity.

The test verifies 35 UCSF binary assets, two distinct live instruments, stale and
privileged-input refusals, exact-prefix branches, STOP, native save and a fresh
engine reopen. It also verifies 17 IXI assets, four tissue/UNKNOWN layers exactly
partitioning the original brain, the eight exact UNKNOWN cells, and separate
SynthRAD CT/MRI selection and persistence. All 18 / 37 / 5 archived original-file
identities remain exact across these three separate source bundles. Those are
archive-member counts, not acquired scan or patient counts.

The [first actual audit failure](../first-release-frozen-sidecar-run-v1/compact-summary.json)
remains failed and preserved. An
[independent reconstruction](../first-release-frozen-sidecar-run-v1/independent-json-order-forensics.json)
proves that only JSON object-key order caused its replay-hash mismatch: zero
replay value, type or ordered-array changes. Reconstructing the entire fresh
reply reproduces its recorded response hash. The correction changes only the
auditor's episode/comparison persistence checks: deep structural equality plus
a recursively key-sorted digest. Raw request fingerprints, package bytes,
scientific actions and source pins remain unchanged. All 45 generated controls
pass, including mutations of values, types, array order and signed zero.

The actual audit's sampled peak family/supervisor RSS is 3,816,112,128 bytes,
below its unchanged 4 GiB limit. Both owned children exit. Its cumulative
1,343,816,993 charged bytes include the first failed audit's 457,424,758 bytes.

## Actual native interface

The [GUI evidence index](../poc-first-local-package-gui-v1/evidence.json) binds
15 screenshots, full accessibility records and process/library evidence under
`outputs/poc-first-local-package-gui-v1`. Interaction used the real packaged app.

- Native library selection, cancellation and invalid-empty-folder rejection work.
  The valid folder remains selected and survives a full app-family shutdown.
  The persisted catalog contains 12 source IDs. Recent saved-version counts can
  change; they are not patient counts. Reported parietal search returns two cases.
- UCSF aspiration removes tissue; field management reduces the modeled field
  without additional removal. Same-state alternatives show 34 / 118 mm³ target
  removal and 9 / 9 mm³ non-target removal for this GUI action sequence. These
  generated research outcomes differ from other demonstrations' action sequences.
- Play/pause, start/end scrubbing, both branch replays, native save and reopening
  work. The saved deeper branch restores frame 21 and field-manager selection.
- IXI displays all four estimated/UNKNOWN layers, changes gray-like opacity from
  0.48 to 0.47, enables cutaway and switches back to original MRI.
- Native Open loads acquired SynthRAD CT with brain/skull estimates. Its separate
  T1 MRI displays without CT overlays. A separate native save and confirmed fresh
  app process restore that MRI selection. Every prior app-family process exits,
  every subsequent PID changes, and library-setting bytes remain unchanged.

The [independent UCSF native-save audit](../poc-first-local-package-gui-independent-v1/saved-ucsf-independent-qa.json)
verifies all 39 logical array members, all 18 original files, native affines,
source metadata and the saved rehearsal/view. The
[paired-imaging native-save audit](../poc-first-local-package-gui-independent-v1/saved-synthrad-independent-qa.json)
verifies all eight logical array members and five originals. Only the intended
view selection/session hash and two numerically identical cursor spellings
change; the selected MRI retains zero layers and no accepted overlay/fusion.

## Remaining limits

No new inference, acquisition, accepted registration, healthy-host review or
training campaign occurs during packaging. Existing model estimates remain
unreviewed; absent vessels/function remain unknown. The fixed
[Case3 measurement](../poc-resect-case3-image-registration-v1/RESULT.md) remains
negative for its image-only rigid candidate; packaging changes no accuracy claim.

Public distribution still needs the project license choice, verified CPython
static-vendor notices, the missing upstream root notice for
`react-remove-scroll-bar@2.3.8`, and signing/notarization work. The empty-folder
rejection currently includes an Electron transport-error prefix in its wording.
Advanced experimental workflows are outside this release's acceptance scope.
The larger publication/high-accuracy goal remains incomplete.
