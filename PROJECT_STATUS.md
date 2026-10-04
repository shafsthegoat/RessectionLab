# Project status

Updated October 4, 2026. The three revised specification documents remain the
authoritative requirements and have been read completely. This file records
executed work and open gates, not a replacement plan.

**Interface direction updated by the user:** Electron + React + TypeScript is
the current Mac UI target. Further PySide/PyQt interface work has stopped. The
validated Qt prototype is preserved as an earlier experiment; Python planning,
learning, provenance and evaluation remain the shared backend. The new renderer
communicates through a narrow local sidecar interface. The first standalone
Electron package has passed real MRI, route comparison and native-save checks;
patient-training controls and source-native replay are the next integrated build.

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
- The standalone Electron app loaded the real UCSF bundle and rendered linked
  physical MRI, annotation surfaces and two full-tool routes. Route generation
  took 1.16 seconds in one native run; the Mac save dialog preserved the routes,
  comparison and cursor. Its initial imaging/search bundle is 388.6 MB and has
  no Qt/VTK dependency. Accessible volume is distinct from removed volume.
- Actual patient policy-gradient updates, resumable checkpoints, scratch/search
  comparators and isolated world roles execute. Eighteen development runs and
  six budget/exploration follow-ons retained their source snapshots and failures.
  No RL advantage over search was demonstrated.
- Independent native-footprint checking **invalidated every non-STOP UCSF
  coarse-removal candidate**. The coarse simulator removed cells beyond the
  active tip's supported footprint. Those runs remain negative development
  evidence. Their coarse returns are not valid resection performance.
- The corrective native-grid engine and separate checker now agree on a real
  two-stroke sequence: 249 mm³ target and 17 mm³ normal tissue removed, with zero
  unsupported source-cell volume. Three patient-training seeds changed weights;
  one improved from STOP to SEARCH's score, two retained their initial policies.
  SEARCH and greedy search were faster. All nine frozen development candidates
  passed native checks. Missing functional evidence remains unknown.
- PPO at 32 and 128 actual optimizer updates failed to outperform SEARCH on the
  branching development fixture. Exact source snapshots, initial weights,
  transition counts and negative results are preserved.
- The Electron training bridge executes real updates, resumable cancellation,
  independently checked replay and export. A 30-second patient-learning budget
  yielded the two-stroke sequence in 37.77 seconds total including setup/checks;
  a 5-second run retained STOP. Resume binds checkpoint bytes, source/runtime
  contract and original budgets; unsigned crash recovery is refused.
- Raw BTC diagnostic tensor fitting ran on 101,311 voxels; bounded CSA tracking
  produced 176 unlabeled diagnostic paths. Uncorrected DWI is correctly refused
  for tract-aware planning. Correction and alignment are current work.
- Pinned SynthStrip main/no-CSF models ran locally on MPS in roughly 6–7 seconds
  per variant with identical outputs in three repeats. No-CSF excludes 214
  source tumor-annotation voxels; main includes them all. Neither is reviewed
  brain/cortex or an accepted cortical-access mask. CPU failure is preserved.

## Validation record

The latest root integrated run passed 417 tests and exposed one test-isolation
failure: a subprocess import check incorrectly inspected the parent process.
That test now checks the actual engine process. The subsequent focused bridge,
native-refinement and independent adversarial suite passed all 37 tests. A fresh
integrated run remains due after the current UI slice. Four existing DIPY
basis-deprecation warnings remain. Native adversaries caught and fixed temporal
shaft borrowing, mutable preview descriptors and checkpoint-resume tampering.
The independent native checker improved from 17.638 to 4.310 seconds on the same
saved patient history, with identical geometric certificates and 91 regression
tests. These are local single-run timings, not a latency distribution.
See `EXPERIMENT_LOG.md` for executed experiments, including negative results.
The Qt prototype passed its initial standalone runtime and GUI checks. Its next
build was stopped following the user's Electron decision. The React/Electron
package passed its own imaging/search checks and 20 main-process boundary tests;
the React renderer passed eight source-hydration and seven viewer-geometry tests.
The next build includes the separately tested training engine. Qt results do not
certify Electron. No second-Mac test is claimed.

## Completion gates still open

Native-resolution geometric simulation now works under explicit hypothetical
access assumptions; reviewed anatomy and surgical validity remain open.
Corrected/accepted functional reconstruction, fully integrated comparison and
refinement interaction, broader cohort benchmark and release remain incomplete.
Population frozen/adapted arms have not been trained/evaluated. Developer ID
signing/notarization and independent physical-Mac
validation remain open. No locked final cohort has been opened. Clinical deficit
probabilities remain unavailable; simulator success cannot establish clinical
readiness. Current evidence supports a local research prototype only.

No cloud infrastructure or billing changes have been made. Source imaging stays
outside Git. Numerical arrays and checkpoints are excluded; compact experiment
records and source snapshots preserve the executed development evidence.
