# Current engine and historical Qt dependencies

The active application is Electron/React/TypeScript with a Python numerical
engine. `requirements-lock.txt` retains 52 exact installed versions for that
engine plus research, test, and build tools. It includes VTK 9.7.0 for native
surface preparation, matching the existing `anatomy` extra and the actual runtime.
Qt stays optional. No installed package was changed by this correction.

`requirements-legacy-qt-lock.txt` includes the default file and adds only the
four Qt packages required by the historical interface; VTK comes from the shared
engine lock:

| Package | Existing pinned version |
| --- | --- |
| PySide6 | 6.11.2 |
| PySide6_Addons | 6.11.2 |
| PySide6_Essentials | 6.11.2 |
| shiboken6 | 6.11.2 |

The original lock audit checked the separated pins against distribution metadata.
Cold native-study packaging subsequently verified the installed VTK runtime as
9.7.0; the stale optional-only 9.7.1 pin was replaced with that shared pin.
`pyproject.toml` declares VTK in the `anatomy` extra and Qt in `legacy-qt`
and was left unchanged. The installed editable `resectionlab` metadata
predates that source change and still advertises unconditional Qt/VTK
requirements. It was not regenerated during this audit. A subsequent ordinary
editable install reads the current project declarations; the README installs
the chosen pinned lock first and uses `--no-deps` for the editable project.
No other installed, non-legacy distribution had an enabled default requirement
on these five packages.

## Import and packaging evidence

The historical UI modules, `scripts/build_macos.py`, and `packaging/macos_entry.py`
use the legacy display toolkits. Native anatomy surface preparation additionally
uses VTK data/filter bindings through direct `vtkmodules` imports; source runs
that prepare uncached meshes need VTK even though the Electron interface uses
Three.js. The old
`scripts/launch_app.sh` launches that historical application. The reusable
annotation, refinement, and physical-reslicing helpers under `app/` do not
require a toolkit; importing `resectionlab.app` itself is lazy.

`scripts/build_electron_sidecar.py` includes the VTK geometry dependency closure
for cold native-study opening, while excluding PySide6, PyQt6 and the `vtk`
umbrella import. Packaging evidence and the current runtime scope are described
in [electron-packaging.md](electron-packaging.md). The earlier lock separation
was checked against a frozen payload without toolkit binaries; that historical
scan does not describe the current geometry-enabled bundle.

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
