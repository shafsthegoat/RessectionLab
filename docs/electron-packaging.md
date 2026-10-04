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
cd desktop
pnpm package:mac
cd ..
node desktop/electron/verify-sidecar.cjs --bundle
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

The first validated engine supports imaging, complete-instrument route search,
evidence inspection and workspace persistence. Torch, Qt, VTK, DIPY and plotting
libraries are excluded in this baseline. Training controls remain unavailable
until a subsequent engine's actual training and independent replay are verified.
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

The 20 main-process adversarial tests cover opaque IDs, checksums, traversal,
symlink replacement, layout limits and trusted frame checks. Renderer tests
independently enforce imaging dtype, shape, coordinate and volume contracts.
