# Current engine and historical Qt dependencies

The active application is Electron/React/TypeScript with a Python numerical
engine. `requirements-lock.txt` retains 51 exact installed versions for that
engine plus research, test, and build tools. It does not install Qt or VTK.
No retained package version was changed.

`requirements-legacy-qt-lock.txt` includes the default file and adds only the
five packages required by the historical interface:

| Package | Existing pinned version |
| --- | --- |
| PySide6 | 6.11.2 |
| PySide6_Addons | 6.11.2 |
| PySide6_Essentials | 6.11.2 |
| shiboken6 | 6.11.2 |
| vtk | 9.7.1 |

All 56 pins were checked against installed distribution metadata. The current
`pyproject.toml` already declares Qt/VTK only under the optional `legacy-qt`
extra and was left unchanged. The installed editable `resectionlab` metadata
predates that source change and still advertises unconditional Qt/VTK
requirements. It was not regenerated during this audit. A subsequent ordinary
editable install reads the current project declarations; the README installs
the chosen pinned lock first and uses `--no-deps` for the editable project.
No other installed, non-legacy distribution had an enabled default requirement
on these five packages.

## Import and packaging evidence

Static inspection of Python source, scripts, tests, and packaging entry points
found direct toolkit imports confined to the historical UI modules,
`scripts/build_macos.py`, and `packaging/macos_entry.py`. The old
`scripts/launch_app.sh` launches that historical application. The reusable
annotation, refinement, and physical-reslicing helpers under `app/` do not
require a toolkit; importing `resectionlab.app` itself is lazy.

`scripts/build_electron_sidecar.py` already explicitly excludes PySide6, PyQt6,
VTK and vtkmodules. Existing packaged-build evidence is described in
[electron-packaging.md](electron-packaging.md). The currently present frozen
engine payload was also scanned for toolkit directories/libraries, with no
matches. Retained historical source files are separate from runtime toolkit
binaries. No new app build was performed for this lock-file change.

To test source behavior without modifying the shared environment, fresh Python
subprocesses loaded a temporary import blocker that makes all five toolkit
module families unavailable, including to subprocesses launched by tests:

- All 33 numerical/research modules and toolkit-independent `app` helpers
  imported successfully; no legacy toolkit module loaded.
- Default test collection completed with 780 tests collected in the final probe.
- Thirteen focused tests passed, covering MRI transfer, route save/reopen,
  JSON-line engine startup/shutdown, physical reslicing, annotation and replay.
- The historical Qt worker module skipped explicitly when the toolkit was
  blocked. Its three tests passed in the existing environment with Qt available.

The reproducible probe, command outputs, exact file hashes, distribution
metadata and payload-scan result are under
`artifacts/dependency-lock-separation/`. `tests/test_app_workers.py` now declares
its optional toolkit requirement before importing the worker.

These checks are not a clean-environment installation or a dependency-resolution
test against a package index. No package was downloaded, upgraded, uninstalled,
or changed in the shared environment. Other platforms and Python versions have
not been validated by this audit. SynthStrip retains its separately documented
isolated NumPy compatibility runtime; this lock separation does not change it.
