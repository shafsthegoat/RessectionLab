# Integrated Python regression — 4cea5c4

Released source: `4cea5c4371bea158b5e5fb7db9042dbb3c0eed32`.
The complete Git archive contains 3,778 files, 184,289,439 bytes and 141 test
modules. Only its `tests/` directory was collected; working edits, untracked
UI/procedural files and prior test copies under artifacts were excluded.

The main environment passed **2,433 tests**, skipped six and retained 14
warnings in 278.04 seconds (279.734 seconds including isolation/watchdog).
The separate existing IDC environment passed all 21 acquisition/DICOM controls
in 1.11 seconds (1.581 seconds including isolation/watchdog), covering the
five optional DICOM skips. The combined result is **2,438 unique passes,
one remaining skip and no failures**. The remaining skip is the unchanged
procedural test's absent ignored development fixture. No data was added to
erase that skip. The explicit already-consulted UCSF bundle was available to
the separate environment-aware native configuration test.

All 3,778 source hashes and the bound UCSF file hash remained unchanged,
with zero files added to the snapshot. Runtime receipts contain dependency
versions, exact arguments and imported project-module paths; no mutable
`resectionlab` source was imported. The 14 warnings are ten NumPy shape/dtype
assignment deprecations in unit fixtures and four DIPY legacy-basis pending
deprecations. Tests were not edited, retried or marked expected-failure.

Execution used isolated Python startup, explicit snapshot/dependency paths,
offscreen Qt, disabled pytest plugin autoload/cache and one numerical thread.
Processes ran sequentially with 900-second main / 120-second IDC wall caps
and a sampled 4 GiB aggregate process-family RSS cap. A dedicated process
group permits terminating descendants on a cap failure. Small process controls
first verified normal completion, timeout and memory termination. No cap was
reached: sampled main peak was 922,599,424 bytes; IDC peak was 70,221,824 bytes.
Sampling does not establish an unsampled absolute peak.

`execute_regression.py` is the exact driver. `root-release.json` binds its
archive, complete per-file inventory, interpreter paths and existing fixture.
The tar and source tree remain ignored under `build/python-regression-4cea5c4/`.
Logs, JUnit durations, progress, runtime receipts and the summary are retained
here. No installs, patient training, FEM solve, image inference or sealed
patient access occurred. Analytic test fixtures are software validation and
are not real-data training or clinical validation evidence.
