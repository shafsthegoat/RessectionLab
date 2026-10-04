# Electron Mac app packaging

The current desktop target is Electron, React and Three.js. The older Qt build
remains a reference artifact; it is not the current desktop architecture.

## Reproduce locally

Use Python 3.12 with the repository's numerical dependencies and PyInstaller,
and Node 24 with pnpm. First complete the Python environment setup in the
[local development guide](../README.md#local-development). The real-imaging
verifier below requires `outputs/cases/UCSF-PDGM-0004.ressectionlab`, which is
excluded from Git. Prepare that public structural-mirror fixture from the
repository root before running `verify-sidecar.cjs`:

```sh
.venv/bin/python scripts/acquire_public_case.py --inspect-nifti
.venv/bin/python scripts/prepare_case.py
```

The [acquisition guide](data_acquisition.md) records its provenance and limits;
equivalence to official TCIA bytes remains unverified. This fixture is required
by the real-imaging verifier, while the app's synthetic fixture can be opened
without downloading imaging. Then, from the repository root:

```sh
cd desktop
pnpm install --frozen-lockfile
pnpm exec electron --version
pnpm test
pnpm build
cd ..
.venv/bin/python scripts/build_electron_sidecar.py
node desktop/electron/verify-sidecar.cjs --frozen
node desktop/electron/verify-training.cjs --frozen
cd desktop
pnpm package:mac
cd ..
node desktop/electron/verify-sidecar.cjs --bundle
node desktop/electron/verify-training.cjs --bundle
```

The output is `desktop/release/RessectionLab-darwin-arm64/RessectionLab.app`.
The packaged app uses only its bundled numerical engine. The development app
uses `.venv/bin/python` and `src/` from the checkout. No loopback server is
required for packaged operation. A browser-only preview at port 5173 is a
separate development convenience. Preview patient arrays are not bundled.

The native launch accepts `--case /absolute/path/to/case.ressectionlab` or
`--demo`. Omitting both starts an empty workspace. Developer ID signing and
Apple notarization have not been performed; the build uses local ad-hoc signing.
This is an Apple Silicon research build, not a clinically validated release.
Historical scientific report regeneration and audit reconstruction have separate
input requirements; see [artifact reproducibility](artifact-reproducibility.md).

## Build and scientific scope

`build_electron_sidecar.py` freezes an immutable Python source snapshot. Its
report records the exact source file hashes, revision, dirty paths and capture
time. `package-macos.cjs` independently captures the renderer and main-process
inputs, typechecks and builds that captured copy, and embeds its manifest.
The engine is copied with relative symlinks preserved. Electron Packager's
generic extra-resource copy rewrote these links to absolute paths during the
first experiment; the explicit final copy avoids that external dependency.
Packaging verifies the new app in staging before archiving and replacing the
previous build. A failed replacement build leaves the working app available.

The current builder includes Torch and the captured Python source files used
by the learning contract integrity checks. Frozen-engine validation exercises
actual gradient updates, independent selection replay, export, restart and
modified-report rejection. An initial frozen test safely refused training
because its integrity check could not find the raw `learning.py`; including
the captured package sources fixed that packaging omission. Raw package sources come from the same immutable
snapshot as the executable; they are not read from the live checkout.

The first validated engine supported imaging, complete-instrument route search,
evidence inspection and workspace persistence. Torch, Qt, VTK, DIPY and plotting
libraries were excluded in that initial baseline. The subsequent numerical
build includes Torch; Qt, VTK, DIPY and plotting libraries remain excluded.
Route accessibility stays distinct from removal, and unsupported clinical
probabilities remain absent. Imported saved metrics require current evaluation;
matching image hashes alone do not certify arbitrary saved route artifacts.

## Local trust boundary

The renderer has no Node access. Context isolation and sandboxing are enabled;
only the main frame at the expected file URL may invoke IPC. Main-owned native
dialogs grant input and output paths. Named preload methods expose only the
supported research operations; there is no arbitrary filesystem or shell API.
Window creation, navigation and permission requests are denied. A restrictive
content policy is applied in both the native session and the HTML.

The numerical engine communicates through JSON lines over pipes. MRI and mask
arrays use private per-session files. The renderer receives opaque asset IDs;
the main process validates containment, layout, size and SHA-256 before reading.
Cancellation names an outstanding request. Case hashes bind geometry operations
to an immutable cached case. Research checkpoints use an app-owned run directory.

## Recorded validation

`artifacts/electron-packaged-validation.json` records the first real native
window inspection. `artifacts/electron-sidecar-{source,frozen,packaged}.json`
record actual Python process and binary-transfer checks on UCSF-PDGM-0004.
The first packaged app rendered the 240×240×155 MRI and annotations, generated
54 route candidates in 1.16 seconds, showed two alternatives and saved the
comparison using the macOS dialog. Its size was 388,588,500 bytes, with 17
internal symlinks and no external or broken links. Deep strict signature
verification passed. These are local measurements, not general benchmarks.

`artifacts/electron-training-frozen.json` records the later frozen numerical
check: a five-second optimization budget produced nine gradient updates and
an independently accepted synthetic selection replay. The replay contained
39 mm³ modeled target removal and 2 mm³ modeled non-target removal. Full
process startup, training, restart and verification took 25.09 seconds on this
Mac. This fixture tests the executable path; it is not evidence of clinical
efficacy or superiority over search.

`artifacts/electron-cancellation-frozen.json` records a separate cancellation
and restart test: cancellation after two updates was acknowledged in 1.6 ms;
a fresh process preserved the original budget and contract, performed 14 more
updates and produced an independently accepted selection replay. The checkpoint
snapshot may retain status `running`; the signed run manifest and terminal
JSONL event record the cancellation. No final-evaluation operation was invoked.

Finder launches write local diagnostic logs to the application logs directory.
No logs or patient data are transmitted.

The 31 main-process tests cover opaque IDs, checksums, traversal,
symlink replacement, layout limits, trusted frame checks, atomic publication
rollback, bounded logging, structural-import file grants, and matching the
installed Electron version to the captured lockfile. Renderer tests
independently enforce imaging dtype, shape, coordinate and volume contracts.

`verify-cancellation.cjs --bundle` targets the engine inside the actual `.app`
and writes `artifacts/electron-cancellation-packaged.json`. Both modes record
the executable and embedded source identities; packaged mode also records the
renderer archive hash. The verifier rejects changes to these identities between
cancellation and restart. The first packaged run passed in 13.93 seconds:
cancellation after two real updates was acknowledged in 1.98 ms; restart and
resume performed 14 additional updates under the original contract. Its receipt
identifies Python source digest `ddec6f6f…` and renderer archive `98acd274…`.
These results do not apply automatically to later builds.

`artifacts/electron-full-native-validation.json` preserves the complete app's
negative UCSF route-conditioned training result: 32 updates left the actor
unchanged and selected STOP, with zero modeled removal. Native replay, source
view restoration and the macOS export dialog worked. A later geometry audit
identified a changed entry point and no feasible initial cutting action for the
original route/tool configurations. This evidence is retained as a failure of
that modeled configuration, not relabeled as a successful removal experiment.

`artifacts/electron-btc-inventory-validation.json` records source-app inspection
of a second, full-head case. Two whole-brain envelope estimates remain pending
review, the no-CSF annotation mismatch is disclosed, and route generation stays
disabled. Structural import uses three native selections (source MRI, proposed
mask, provenance report); cancelling any selection leaves the case untouched.
Import stores evidence without certifying cortex or promoting the estimate into
working anatomy. This source inspection is separate from packaged validation.

The `artifacts/electron-refinement-v2/` receipts identify the subsequent arm64
build with Python source `377284f6…` and the unique Electron bundle identifier
`org.ressectionlab.electron`. The historical Qt app retains its earlier identity;
the Electron application keeps its existing local research-run directory.
The numerical snapshot built in 80.20 seconds; the first updated shell built in
10.91 seconds. All 41 bundled symlinks were internal and valid, and local deep
strict signature verification passed.

In the actual native interface, the original generic route was blocked before
learning. An explicitly generated native-axis alternative, with its own tool
and 6 mm hypothetical access window, exposed one legal initial cutting action.
A seed-11 run reached 32 updates and independently accepted one fixed stroke:
174 mm³ modeled target removal, 11 mm³ modeled normal removal, and 41,745 mm³
residual target. This is a STOP-versus-declared-stroke optimization with fixed
entry, target, window and instrument. It does not establish a free-form learned
trajectory, a clinical outcome probability, or superiority over search.

Native export and source restoration passed. A second native run was cancelled
after four visible updates; its signed checkpoint retained six updates after
in-flight work settled. All 56 route alternatives
and the selected comparison were saved through the native dialog. The initial
replay-slider accessibility test exposed a label/certified-replay mismatch.
`native-workflow-before-slider-fix.json` deliberately preserves that partial
result; later UI validation must identify the rebuilt renderer separately.

All three development verifiers accept `--report-dir <directory>` so subsequent
build receipts can coexist. The v2 packaged engine checks passed: real MRI/search
and persistence in 6.72 seconds, synthetic training/replay in 8.59 seconds, and
signed cancellation/restart/resume in 13.79 seconds. Their machine-local timings
are verification measurements rather than standardized performance benchmarks.

The final renderer capture `5fe94739…` rebuilt in 10.37 seconds and retains
exactly the same numerical engine bytes as the earlier v2 protocol receipts.
`native-workflow-final.json` identifies the renderer archive and executable
separately. After a full native-app restart, resuming the cancelled run restored
its six saved updates and completed 26 more, reaching 32 with the original
signed request and world contract preserved. Independent selection accepted the
same 174/11 mm³ modeled removal. Native accessibility decrement and increment
now load the certified zero- and one-stroke states: slider, viewer banner and
volume totals agree. Returning to the source annotations also passed, recorded
in `source-restored-final.jpg`. The earlier export and original-route rejection
belong to the explicitly identified pre-fix renderer receipt. No final-evaluation
operation or clinical probability was introduced.

The matched `source-baseline.jpg` and `source-normals-after.jpg` images document
a display-only shading adjustment. It averages surface normals at coincident
vertices; positions, triangles, source-cell quantities and the original voxel
steps remain unchanged. A transient pending-replay state completed before a
native screenshot could capture it; focused renderer tests cover that queue
and its pending/applied labels.

One renderer rebuild failed while fetching Electron checksum metadata despite
a cached archive; the prior app stayed available. The builder now passes the
official installed Electron package's checksums to its supported downloader,
which validates cache hits and downloads without weakening TLS or integrity
checks. An independent no-network test verified the actual cached arm64 archive;
corrupt archives and absent checksums were refused. A subsequent packaging-only
guard also requires the installed version to match the captured pnpm v9 root
lockfile and fails on unsupported or ambiguous entries. That guard was added
after the final app capture; it independently accepted that exact capture's
44.5.1 lockfile, and all 31 main-process tests passed. The validated app itself
was not rebuilt for this build-tool-only change. Local ad hoc signing passed;
distribution signing and notarization remain unperformed.


`artifacts/electron-structural-proposals-v1/` records the next renderer-only
slice. Both PAT28 and PAT05 display the main and no-CSF structural proposals as
lavender contours on the three original MRI planes. The labels remain
review-required and view-only; neither estimate enables route generation or
certifies cortical access. Native inspection reproduces 214 and 538 source
annotation voxels outside the no-CSF estimates. The respective linked cursors
move to the source-derived RAS points, shown at one decimal place as
(-14.0, 9.3, 64.3) and (-41.2, 27.3, 3.2) mm. Clearing and case switching remove
the contour; cancelling the native case dialog preserves the existing view.

That initial capture (`4aba8275…`, 11.33 seconds) exposed a stale footer after
clearing, although its banner and MRI contour correctly disappeared. The
correction was packaged from an isolated checkout of commit `e70edc1`, with all
59 captured desktop inputs independently checked against that exact commit.
The final capture `75d7f69b…` took 12.04 seconds, retained the same validated
numerical engine, and archived the previous app before replacement. Native
retesting confirmed the corrected source-restoration footer and reopened the
existing signed UCSF selection replay: 174 mm³ modeled target removal,
11 mm³ normal removal and 41,745 mm³ residual target. No new training or final
evaluation ran. Initial and final renderer identities have separate receipts.

The native BTC fixtures deliberately lack reviewed planning support and accepted
runs, so a same-case proposal-to-accepted-replay transition was not exercised.
The viewer explicitly suppresses estimates during replay; the native check
confirmed that no estimate survived a case switch into the existing UCSF replay.
These visual checks establish display behavior, not anatomical accuracy. Later
population-prior and layout work is absent from this exact packaged snapshot.


`artifacts/electron-mri-layout-v1/` identifies the exact `8c7c6f7` renderer
capture (`214afa02…`). All 61 desktop inputs matched the committed source; the
build took 10.39 seconds and retained the previous numerical engine. Native
PAT05 and PAT28 checks covered equal 2×2 MRI review, expanded axial/coronal/
sagittal views, and restoration at 1460- and 1050-point window widths. Original
MRI orientation labels, estimate contours, source target volumes and linked
cursor positions remained consistent. A focused axial image advanced from
3.2 to 4.2 mm with one Up key; restoring the layout preserved the new physical
cursor. The separate 3D plane selector and toggle also preserved that cursor.
Controls stayed visible at minimum width. The wide-layout Fit 3D label wraps
vertically in this capture; its later no-wrap source fix is not retroactively
included in this receipt. No new training or population-prior bundle loading
was performed with the retained engine.


`artifacts/electron-priors-v1/` records the seven-layer population-prior slice.
The numerical engine was rebuilt from the immutable `0b4e334` capture
`afd24365…`, with every captured Python input checked against that commit.
The standalone executable is `85a92eff…`; it understands prior bundles and the
completed-selection replay gate. Its build took 83.14 seconds. Later changes
to policy inputs and dependency locks are absent from this engine.

The first prior renderer, also from `0b4e334`, was protocol-tested but was not
inspected through the native interface. A mutable-MRI digest cache finding was
corrected before native checks. The exact `5c57ddb` capture `45668499…` then
passed native selection of all seven real UCSF prior layers, covered zero and
nonzero samples, binary mask membership, missing field-of-view and incomplete
interpolation support. The maps remain view-only, alignment-review-required,
and excluded from route scoring. Patient-specific function and language
dominance remain unknown. Source annotations, MRI bytes and physical cursor
coordinates were preserved.

Native checks at 1460- and 1050-point window widths covered MRI review,
expanded views and source restoration. The Fit 3D control remains accessible;
its text no longer wraps vertically. A native save and reopen retained all
seven prior arrays and the linked cursor. A separate frozen-sidecar comparison
verified unchanged case/planning hashes, source-array hashes, prior-array
hashes and annotation volumes. The development verifier
`desktop/electron/verify-priors.cjs --bundle --prepare-cursors` reproduces the
protocol checks and saves separate view-position fixtures without changing the
original case or its evidence. These fixtures change only the saved cursor.

The representative screenshot landmark was selected by an explicit display
rule: the largest covered motor-map value among source voxels with MRI
intensity at least the median positive intensity. At RAS (-173, 137, 112) mm,
the actual atlas value is 0.99692738, displayed as 0.9969, with source anatomy
visible in all three planes. This is a display landmark, not validation of
motor function. The small-positive and unknown-support regression landmarks
are also retained in the receipts.

An independent GPU audit subsequently found oblique-grid floating-point
boundary disagreement. The final renderer was rebuilt from exact commit
`c194a8b`: capture `762973e2…`, archive `a8da5a06…`, 67 source inputs checked
against the commit, 10.96 seconds. It retains the identical `85a92eff…`
numerical executable. All source hashes used by the independent 487-fixture
WebGL2 audit match this capture. Native focused checks repeated the actual
motor value, covered zero, outside field of view, incomplete support, expanded
MRI and source clearing. No renderer warning, shader error or renderer crash
was observed. `native-workflow-final.json` records this scope; the earlier
`native-workflow-cache-fixed-provisional.json` preserves the broader checks
and their separate renderer identity. The older files named
`capture-verification-final.json` and `app-build-final.json` identify the
intermediate cache correction; the `*-gpu-final.json` files identify the
published app.

The new engine also independently accepted the historical completed selection
replay: 174 mm³ target and 11 mm³ normal removal, with 41,745 mm³ residual.
Accessibility decrement restored step zero at 0/0/41,919 mm³; source restoration
passed. No training or final evaluation was invoked. The enriched prior case
has a different immutable hash and no accepted run, so a same-case
prior-to-replay transition was not exercised; switching to the original case
cleared the prior before its stored replay was opened. These checks establish
software behavior, not anatomical accuracy or clinical readiness. Local deep
strict signature verification passed; distribution signing and notarization
remain unperformed.

The current notice-bearing application uses exact renderer commit `100865f`,
with capture `34285015…` and archive `b99ba3da…`. Its 143 aggregate desktop
checks and production build passed. The complete numerical payload remains
identical to the verified `0b4e334` engine: executable `85a92eff…`, original
source digest `afd24365…`. No later experimental learner was captured.

The actual `.app` now contains `Contents/Resources/ThirdPartyNotices`, outside
`app.asar`. All 292 notice entries and 2,809 numerical payload entries match
the immutable capture after local ad hoc signing. The inventories cover 28
actual frozen Python distribution owners, 35 renderer runtime packages and
complete Electron/Chromium upstream notices. See
[dependency-notices.md](dependency-notices.md) for the collection prerequisites
and two unresolved source-completeness gaps. Notice capture does not establish
complete distribution compliance, and notarization remains unperformed.

`artifacts/dependency-notices-v1/` retains both builds and their identities.
The initial `bee43c9` native check confirmed context-specific unloaded,
estimated-support, blocked full-head and cleared-selection guidance. Real
UCSF search evaluated 54 candidates; clearing both choices prompted selecting
a route. PAT05 full-head route search remained disabled pending reviewed
support. Motor atlas inspection at RAS (-173, 137, 112) mm displayed 0.9969
with patient function unknown, at 1460- and 1050-point window widths. Source
restoration cleared the overlay and preserved the cursor and annotations.
That check also found overlapping unloaded welcome text; the partial receipt
and screenshot preserve the failure.

The corrected `100865f` app repeated native startup at both widths, the
no-case to actual-MRI transition, motor inspection and source restoration.
The welcome now appears once. No renderer warning, shader error or crash was
observed. `native-workflow-final.json` and
`app-capture-verification-final.json` record the focused passing scope. The
historical seven-map/native-replay evidence remains associated with its
original package; it was not represented as a full repetition here. No new
patient training or final evaluation was invoked. Previous application
bundles remain recoverable through the recorded publication paths.
