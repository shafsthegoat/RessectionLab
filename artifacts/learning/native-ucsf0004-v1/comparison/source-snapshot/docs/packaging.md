# Native macOS packaging

The desktop bundle targets Apple Silicon (`arm64`). It includes its Python, Qt,
VTK, numerical and policy runtimes, so the packaged application does not depend
on a source checkout or an activated Python environment. This is a local research
build. External Developer ID signing, notarization, Intel Macs, and a second
physical Mac remain separate release checks.

## Build and run

From the repository root, prepare a native arm64 Python 3.12 environment with the
recorded dependency versions, then build:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/build_macos.py
open dist/RessectionLab.app
```

If the environment is already installed, run only the build command. Re-run the
environment probe after changing dependency versions. The lock records the
tested local resolution; a different operating system/architecture needs its own
verified resolution.

The builder generates the original vector icon, freezes the app, checks its ad hoc
signature, arm64 executable, internal symlinks, and native dependency references,
then launches the frozen runtime and the desktop
smoke test from an unrelated temporary directory. Python, Qt, and dynamic-library
environment overrides are removed for those checks. The desktop test creates a
synthetic case and an application screenshot. It does not establish anatomical
or clinical validity.

For a development launch, use `scripts/launch_app.sh --source --demo`. Without
`--source`, the launcher uses `dist/RessectionLab.app` when it exists and otherwise
runs the development app. The launcher preserves file paths passed as arguments.

`--clean` rebuilds generated PyInstaller analysis. `--verify-only` runs checks on
the existing bundle. `--no-gui-smoke` explicitly skips the rendered desktop check
when no interactive macOS session is available; it must not be reported as a
passed GUI test.

## Reproducibility and diagnostics

Generated artifacts remain outside source control:

| Artifact | Purpose |
| --- | --- |
| `dist/RessectionLab.app` | Application bundle |
| `build/macos/build.log` | Full freezer log and warnings |
| `build/macos/build-manifest.json` | Revision, source digest, versions, timing, platform |
| `build/macos/runtime-check.json` | Frozen dependency and arithmetic checks |
| `build/macos/gui-smoke.json` | Desktop smoke evidence |
| `build/macos/gui-smoke.png` | Captured application window |
| `build/macos/verification.json` | Signature, architecture, size, launch and render results |
| `build/macos/attempts/` | Prior success/failure reports retained before a new attempt |
| `build/input-snapshot/` | Captured input trees and their original revision/dirty-state manifests |

The builder captures source/resources, packaging code, license notices and lock
files into a separate input tree. It checks that the live source remained stable
during that short copy, marks captured files read-only, then builds entirely from
the snapshot. Any change to the snapshot during freezing rejects the build.
Other work can continue in the original checkout; its changes are reported
separately and do not enter the running build.

The bundle carries its original revision, dirty-state metadata, snapshot digest
and per-file source hashes. The frozen check reads the included package source
files, validates their hashes, and compares the embedded snapshot checksum with
the build record. This confirms source provenance without claiming that an
uncommitted snapshot is a clean Git release. Builds are not claimed to be
bit-for-bit reproducible; native signatures and timestamps can differ.

A new attempt invalidates prior green verification immediately and records a
failure if any check fails. Earlier evidence is retained in the attempt archive.

Finder launches retain stdout/stderr in
`~/Library/Logs/RessectionLab/desktop.log`, with one rotated prior log after 5 MB.
The bundle retains dependency metadata and package-provided license files under
its resources. Source data are never bundled automatically. Licensing of imported
patient data and model weights remains distinct from the software license.

The icon in `packaging/icon.svg` is an original vector illustration created for
this project. It contains no copied Medivis assets or visual trademarks.

## Packaging choices

An onedir `.app` avoids extracting large imaging/model runtimes at every launch.
The spec uses PyInstaller's Qt and per-module VTK hooks, explicit render backend
imports, and one Qt binding. It does not collect every VTK module. Architecture
and signing behavior follow the official
[PyInstaller macOS documentation](https://pyinstaller.org/en/stable/feature-notes.html#macos-multi-arch-support).
Collection follows the official
[hook documentation](https://pyinstaller.org/en/stable/hooks.html).

The optional DIPY/FURY adapter imports the legacy `vtk` facade, which can pull in
the whole VTK wheel even when FURY is unavailable. The build excludes that unused
facade and FURY, while retaining the application's explicit `vtkmodules` imports
and DIPY diagnostic reconstruction. The runtime check executes an Adam gradient
update, an analytic tensor fit, and bounded probabilistic tracking. It checks
numerical execution and packaging, not whether a tract represents motor/language
function. FURY-based generic VTK tract-file I/O is outside this packaged workflow.

The minimum system metadata is macOS 14.0, a conservative initial release target;
this is a compatibility declaration, not evidence that every supported OS version
has been tested. Consult the saved verification manifest for the actual test host.

## Local validation record, October 4, 2026

The first development bundle passed native dependency checks but its GUI smoke
stalled in the pre-fix Cocoa render loop and was terminated. Its failed result is
retained in `build/macos/attempts/`. After the desktop rendering correction, the
rebuilt app passed from a clean launch environment:

| Measurement | Observed result |
| --- | --- |
| Build host | Apple M5, macOS 26.6, Python 3.12.14 |
| Build duration | 162.568 seconds |
| Application size | 1,280,444,879 bytes |
| Native dependency verification | 661 files; 776 internal symlinks |
| Frozen runtime check | 7.863 seconds |
| Synthetic linked MPR and actual 3-D render smoke | Passed in 4.763 seconds |
| Signature and architecture | Ad hoc signature verified; arm64 |

These timings include one run of each named check, not latency percentiles. The
source was being updated during both development builds; neither is a frozen
release. The source version of the expanded tensor/tracking/gradient smoke passed.
The reduced-bundle experiment and its frozen numerical checks await the snapshot
rebuild; no size reduction is claimed yet.
