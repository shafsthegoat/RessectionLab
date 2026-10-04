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
  testing/packaging tools installed. The first arm64 app bundle passed an actual
  frozen-runtime and MRI/3-D window smoke check on this Mac.
- Work is on `codex/patient-specific-planner`. Every new commit uses author
  `skamal23 <sayemkamal12@gmail.com>` and co-author
  `shafsthegoat <shafrir.p@gmail.com>`; commits are pushed incrementally.

## Executed vertical slices

- Immutable patient/provenance contracts, time-aware context filtering, physical
  NIfTI QC/import, preserved source annotations, case revision and JSON/NPZ
  roundtrip are implemented. Future context cannot alter planning seeds.
- UCSF-PDGM-0004 structural mirror bytes are pinned and checked; equivalence to
  official TCIA remains unverified. TCIA v5 fixes diffusion metadata relative to
  the specification's v4. The official transfer service failed during acquisition.
- Creator-source OpenNeuro BTC `sub-PAT28`, ds001226 v5.0.1, has 15 verified
  files including T1, fractional tumor annotation, AP DWI, PA references and
  gradients. Its full-head T1 does not provide a reviewed cortical access mask.
- Seven licensed motor/language population maps load with source hashes and
  explicit registration-review contracts. Patient alignment remains unaccepted.
- Full-tip/shaft/swept-envelope route search retains feasible, dominated and
  rejected alternatives with failure evidence. The UCSF route run generated 54
  candidates, 31 geometrically feasible, 12 geometric Pareto. The separate
  checker agreed on all 54 under the same limited structural assumptions.
- The packaged app loaded the real UCSF bundle and rendered linked physical MRI,
  annotation surfaces and a full-tool route. Actual UI route generation took
  1.32 seconds in one run. Accessible volume is distinct from removed volume.
  Route selection accessibility and comparison controls are being improved.
- Actual patient policy-gradient updates, resumable checkpoints, scratch/search
  comparators and isolated world roles execute. Eighteen development runs and
  six budget/exploration follow-ons retained their source snapshots and failures.
  No RL advantage over search was demonstrated.
- Independent native-footprint checking **invalidated every non-STOP UCSF
  coarse-removal candidate**. The coarse simulator removed cells beyond the
  active tip's supported footprint. Those runs remain negative development
  evidence; a native-grid connected-removal engine and independent checker are
  under construction. Their coarse returns are not valid resection performance.
- Raw BTC diagnostic tensor fitting ran on 101,311 voxels; bounded CSA tracking
  produced 176 unlabeled diagnostic paths. Uncorrected DWI is correctly refused
  for tract-aware planning. Correction and alignment are current work.

## Validation record

An independent earlier integrated suite passed 168 tests. Additional diffusion,
runner and UI tests have since been added. The latest root targeted run passed
18 tests and exposed one shared-working-tree snapshot race in a runner test;
the owner is isolating its fixture while retaining production source guards.
See `EXPERIMENT_LOG.md` for executed experiments, including negative results.
The app passed its initial standalone runtime and GUI checks; a smaller bundle
with stronger numerical checks is being built. No second-Mac test is claimed.

## Completion gates still open

Native-resolution legal patient resection, corrected/accepted functional
reconstruction, intuitive comparison/refinement interaction, broader cohort
benchmark and release remain incomplete. Population frozen/adapted arms have
not been trained/evaluated. Signing/notarization and independent physical-Mac
validation remain open. No locked final cohort has been opened. Clinical deficit
probabilities remain unavailable; simulator success cannot establish clinical
readiness. Current evidence supports a local research prototype only.

No cloud infrastructure or billing changes have been made. Source imaging stays
outside Git. Numerical arrays and checkpoints are excluded; compact experiment
records and source snapshots preserve the executed development evidence.
