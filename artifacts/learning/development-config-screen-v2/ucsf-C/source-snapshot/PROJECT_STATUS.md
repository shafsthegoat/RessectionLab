# Project status

Updated October 4, 2026. The three revised specification documents remain the
authoritative requirements and have been read completely. This file records
executed work and open gates, not a replacement plan.

## Verified starting state

- Repository initially contained only the three specifications and two commits.
  A bounded search found no existing implementation to preserve elsewhere.
- Local development: Apple M5, 16 GiB unified memory, macOS 26.6, Python 3.12.
  Hardware probe and architecture evidence are saved in `docs/`.
- An isolated environment now has Qt/PySide6, VTK, NiBabel, SciPy, PyTorch and
  testing/packaging tools installed. A runnable app is not yet verified.
- Work is on `codex/patient-specific-planner`. Every new commit uses author
  `skamal23 <sayemkamal12@gmail.com>` and co-author
  `shafsthegoat <shafrir.p@gmail.com>`; commits are pushed incrementally.

## Work underway

- Targeted public-case acquisition and source/license verification. Current TCIA
  v5 corrects diffusion headers and gradients relative to v4 named in the specs;
  the acquisition record will preserve that discrepancy and the exact release.
- Native linked MRI/3-D viewer, import/QC/persistence, whole-tool route geometry,
  search alternatives and independent evaluation contracts.
- Connected sequential removal and actual patient-specific policy updates after
  the route slice, with separate optimization/selection/evaluation worlds.

## Completion gates still open

No milestone is declared complete from dependency installation or written tests.
The real public case, visual coordinate inspection, executed geometry/replay
tests, learning experiments, independent evaluation, Mac packaging, broader
cohort benchmark and release remain unverified or incomplete. Population policy
arms have not been trained/evaluated. Clinical deficit probabilities remain
unavailable; simulator success cannot establish clinical readiness.

No cloud infrastructure or billing changes have been made. Source imaging stays
outside Git. The first test target is a real aligned case plus analytic fixtures
that distinguish tip clearance from shaft clearance.
