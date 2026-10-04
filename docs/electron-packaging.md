# Electron Mac app packaging

The current desktop target is Electron, React and Three.js. The older Qt build
remains a reference artifact; it is not the current desktop architecture.

## Reproduce locally

Use Python 3.12 with the repository's numerical dependencies and PyInstaller,
and Node 24 with pnpm. From the repository root:

```sh
cd desktop
pnpm install --frozen-lockfile
pnpm exec electron --version
pnpm test:main
pnpm test:renderer
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

The 26 main-process tests cover opaque IDs, checksums, traversal,
symlink replacement, layout limits, trusted frame checks, atomic publication
rollback, bounded logging, and structural-import file grants. Renderer tests
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
after four visible updates and retained its checkpoint; all 56 route alternatives
and the selected comparison were saved through the native dialog. The initial
replay-slider accessibility test exposed a label/certified-replay mismatch.
`native-workflow-before-slider-fix.json` deliberately preserves that partial
result; later UI validation must identify the rebuilt renderer separately.

All three development verifiers accept `--report-dir <directory>` so subsequent
build receipts can coexist. The v2 packaged engine checks passed: real MRI/search
and persistence in 6.72 seconds, synthetic training/replay in 8.59 seconds, and
signed cancellation/restart/resume in 13.79 seconds. Their machine-local timings
are verification measurements rather than standardized performance benchmarks.
