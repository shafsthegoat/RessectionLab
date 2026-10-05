# Dependency notices in the Mac app

The notice collector preserves upstream text and records its origin and SHA-256.
It does not select a license for RessectionLab or decide whether all distribution
obligations have been satisfied. Missing source material remains explicit.

Packages built with this source contain
`RessectionLab.app/Contents/Resources/ThirdPartyNotices/`, outside `app.asar`.
This directory stays with the application when the `.app` alone is copied:

- `README.txt` and `inventory.json` describe the scope and unresolved items.
- `electron/LICENSE` and `electron/LICENSES.chromium.html` retain the complete
  Electron and Chromium notices. The builder compares them with the texts from
  the checksum-verified Electron archive used for that package.
- `renderer/` retains package-provided license texts and package metadata;
  `renderer-inventory.json` records the production dependency and runtime peer
  closure. This is a conservative package inventory, not a claim that every file
  in those packages was emitted by Vite. Build tools and type-only peers are
  excluded.
- `numerical-engine/` contains the frozen Python inventory, upstream package
  notices, CPython's installed notice, and runtime build-configuration evidence.
  The collection includes complete upstream license directories, such as
  h5py's HDF5/LZF notices and Pillow's native-library notices, while excluding
  incidental Python source and bytecode from import packages named `licenses`.

In Finder, **Show Package Contents** on the application reveals `Contents`, then
`Resources/ThirdPartyNotices`. No network access is needed to read the copied
texts. The build manifest hashes the complete notice tree; the builder verifies
the copied bytes before signing. Missing inventories, mismatched engine bytes,
altered notices, path traversal and external notice symlinks are rejected.

## Reproducing an inventory

New numerical builds run `packaging/python_notices.py` after PyInstaller and
stage the result beside the engine at `desktop/sidecar/notices/`. The collector
uses the matching immutable source snapshot and PyInstaller work tables. It
checks actual embedded module membership, compares installed source compiled
code against the frozen code without executing it, and maps collected sources
to their installed distribution RECORD owners. Every selected distribution
version must match the captured requirements lock. It also hashes the actual
collected payload, including relocated native libraries. Native versions are
attributed from the captured lock and collection table; relocated/signed
libraries are not claimed to be byte-identical to their original wheels.

For an already validated engine, retain that engine and capture notices into a
new directory using its original work tables and source snapshot:

```sh
.venv/bin/python packaging/python_notices.py \
  --engine desktop/sidecar/ressectionlab-engine/ressectionlab-engine \
  --work build/electron-sidecar/work/ressectionlab-engine \
  --snapshot build/electron-sidecar/inputs/20261004T103548.141696Z \
  --output build/dependency-notices/new-python-capture
```

The timestamp above identifies the historical `0b4e334` engine and is not a
portable default. Those ignored build inputs must still be available to audit
that exact executable. A fresh checkout should create its own coherent engine
snapshot. After reviewing the generated inventory, stage a copy at
`desktop/sidecar/notices/`; the application builder refuses a notice inventory
whose executable or collected payload hashes differ. It copies notices beside
the engine, without changing the frozen executable. Ordinary renderer packaging
then captures JavaScript and Electron notices from installed upstream files.

## Current source capture and remaining gaps

`artifacts/dependency-notices-v1/capture-validation.json` records a capture for
the verified `85a92eff…` engine: 5,428 embedded Python module entries, 5,420
matching source code objects (the remaining entries are namespace packages),
2,809 collected payload files, and 28 distribution owners. The collector copies
217 Python notice/metadata/build-configuration files, 69 renderer
notice/metadata files for 35 runtime packages, and both complete Electron texts.
SciPy, nibabel, scikit-image, NumPy, Torch, Pillow, h5py, PyInstaller, setuptools
and their package-provided vendor material are represented. Unexpected frozen
members such as pytest and matplotlib support modules are inventoried from
the actual archive rather than omitted based on intended dependency exclusions.

Two source-completeness gaps remain:

1. `react-remove-scroll-bar@2.3.8` is present in the renderer and declares MIT in
   its metadata, but its installed package contains no license text. The npm
   registry identifies upstream commit `b3b1287…`; the exact commit's license
   lookup returned 404, its GitHub tree lookup returned 422, and the `v2.3.8`
   package lookup returned 404. The independent source-check receipt retains
   those observations. No license template or sibling package's copyright
   notice was substituted.
2. The installed CPython 3.12.14 runtime provides its own `LICENSE.txt`, but no
   complete vendor notice/version manifest for the statically linked runtime.
   Its build configuration references external libraries; those flags alone do
   not establish exact upstream versions or complete applicable notices.
   This requires the provenance and notice material of that specific runtime
   build. Licenses from unrelated installed binaries are not substitutes.

The inventories mark notice-source completeness as **incomplete**. Capturing
available texts does not resolve these gaps or establish release-ready license
compliance. Patient data, pretrained weights and external research tools have
separate provenance and terms; they are not included by this software collector.

Seven JavaScript and five Python focused tests cover closure selection,
missing-text reporting, exact engine/payload binding, source-code mismatch,
upstream-byte preservation and path boundaries. The initial source capture
left the prior app unchanged, recorded in `unchanged-native-baseline.json`.

The first actual notice-bearing app was built from exact commit `bee43c9`.
Its 142 desktop tests and production build passed. The package verification
receipt `app-capture-verification.json` checks all 292 notice-tree entries
inside the application against the immutable capture, and all 2,809 numerical
payload entries against the retained engine. The engine executable remains
`85a92eff…`; its original source revision remains `0b4e334`. Local ad hoc deep
strict signature verification passed. The native receipt
`native-workflow-bee43c9.json` records successful loaded-case guidance,
minimum-width motor MRI inspection and source restoration, plus a visible
duplicate unloaded welcome that required a subsequent renderer correction.
These observations do not close the two source-completeness gaps above.

The corrected native app uses exact renderer commit `100865f`: capture
`34285015…`, archive `b99ba3da…`, with all 71 source inputs checked against
that commit. All 143 aggregate desktop tests and the production build passed.
The full notice and numerical payload trees are byte-identical to the first
notice-bearing package. `app-capture-verification-final.json` binds those
contents to the actual signed application; `native-workflow-final.json`
records the single welcome at normal/minimum widths, real-case loading,
representative motor atlas inspection and source restoration. The previous
partial receipt and overlapping-text screenshots remain unchanged. No new
patient training or final evaluation was run.
