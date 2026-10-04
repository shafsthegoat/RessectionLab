# Project status

Updated October 4, 2026. The three revised specification documents remain the
authoritative requirements and have been read completely. This file records
executed work and open gates, not a replacement plan.

**Interface direction updated by the user:** Electron + React + TypeScript is
the current Mac UI target. Further PySide/PyQt interface work has stopped. The
validated Qt prototype is preserved as an earlier experiment; Python planning,
learning, provenance and evaluation remain the shared backend. The new renderer
communicates through a narrow local sidecar interface. The first standalone
Electron package has passed real MRI, route comparison and native-save checks.
The full training package has passed actual updates, cancellation/resume and
restart checks. The refreshed native patient UI now passes blocked-route
preflight, exact-stroke training, replay/export, cancellation and full-app
restart/resume. Accessible replay controls were fixed after a native regression
and now keep the image, displayed step and quantities synchronized.

The Electron renderer adds separate, explicitly unreviewed brain-envelope
contours on the original MRI planes. Source/frame/mask checks and lazy loading
pass for both BTC cases: the no-CSF estimates exclude 214 PAT28 annotation voxels
and 538 PAT05 annotation voxels. These are overlap checks, not accuracy scores.
All 97 desktop checks and the production build pass. Native inspection now
passes on both cases, including source restoration, case switching and the
outside-annotation cursor jumps. A stale footer was caught and fixed; final
renderer source `75d7f69b…` uses the previous verified numerical engine. Working
anatomy and access gates stay unchanged. Screenshots and exact package identities
are in `artifacts/electron-structural-proposals-v1/`.

Seven registered motor/language prior proposals now persist in portable cases
with separate values and atlas coverage, exact T1 registration/T1c display source
bindings, and unchanged planning inputs. They remain view-only population
evidence requiring alignment review. Their Electron controls now display one
selected layer with separate atlas coverage, scalar/binary sampling and cursor
values. Source restoration, stale loads and case changes clear the layer.
All 131 desktop checks and the production build pass after removing an unsafe
source-image hash cache and correcting float32 atlas boundary sampling. An
independent actual GPU probe passes 487 cases, with the earlier failures retained.
The new complete engine and renderer passed native inspection. All seven layers,
covered zero, positive values, outside/incomplete support, minimum-width layouts,
save/reopen and historical checked replay were exercised. Final renderer
`762973e2…` from `c194a8b` repeats the corrected sampler and source-restoration
checks with engine `85a92eff…` from the verified `0b4e334` Python snapshot.
Later experimental learning changes are not in this app snapshot. Native
receipts and screenshots are in `artifacts/electron-priors-v1/`.

The first declared procedural-to-patient comparison stopped before adaptation
updates because equivalent typed and JSON world vectors compared differently.
Its raw records and failure are retained; it produced no validated comparison.
The implementation repair preserves every declared world field and adds a
public-case preflight before any training. The separately declared fresh v2
completed with original geometry, rewards, seeds and budgets unchanged. All six
patient learners changed actor weights and all 13 frozen candidates passed
independent native checking. SEARCH/GREEDY scored 245.24; scratch seeds scored
[245.24, 171.42, 245.16], frozen procedural initialization 139.62, and adapted
seeds [171.42, 171.42, 245.17]. This is a negative transfer result on one reused
development patient with zero human pretraining patients. Final/stress worlds
remain closed. The best sequence removes only 249 of 41,919 target mm³ under
hypothetical access; this is not a complete resection plan.

A separate post hoc geometry probe expands the proposal inventory while keeping
the native checker, source, tools and access assumptions fixed. Under the same
three-cut cap, it removes 1,113 target and 62 normal mm³, with 177 mm³ of partial
normal contact. Both native and independent artifact audits pass. Coverage is
still only 2.655%, and this new, restricted parallel-column model is not yet part
of the desktop planner or the frozen learning comparisons.

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
- A second creator-source BTC case, `sub-PAT05`, was selected by a recorded
  metadata-only rule before image acquisition. Seven pinned files total 17.86 MB;
  independent checks agree on all 11,437 threshold-derived annotation voxels.
  The registry has three development groups, two primary-source identities and
  zero final patients. Unknown clinical timing remains excluded from planning.
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
- Shared pretraining, frozen inference and isolated adaptation now execute with
  explicit cohort exclusions and separate offline costs. The first two-group
  synthetic experiment used eight offline updates and eight online updates per
  seed. SEARCH scored 1.30, scratch [0, 0, 1.24], frozen 0 and adapted [0, 0, 0].
  All ten independent synthetic geometry checks passed; no final worlds opened.
  This is a negative analytic experiment, not a clinical population policy.
- A separately declared procedural-native comparison completed fresh pretraining
  and all six scratch/adapted arms after a retained implementation-only failed
  attempt. Offline training made 32 actual Adam updates across two nonpatient
  families; online runs made 10–14 updates within matched 30-second cooperative
  caps. All actors changed, though scratch seed 23 selected its initial policy.
  No learned arm exceeded SEARCH. All 13 candidates passed eight distinct native
  sequence audits, and a separate artifact audit recomputed scores from source
  cells. Total launcher time was 288.87 seconds, including 33.12 seconds of
  independent validation. These development results do not replace real
  patient-population evaluation; full evidence is retained under
  `artifacts/learning/procedural-native-to-ucsf-v2/`.
- The Electron training bridge executes real updates, resumable cancellation,
  independently checked replay and export. A 30-second patient-learning budget
  yielded the two-stroke sequence in 37.77 seconds total including setup/checks;
  a 5-second run retained STOP. Resume binds checkpoint bytes, source/runtime
  contract and original budgets; unsigned crash recovery is refused.
- Native UI testing exposed a selected-route mismatch: the factory discarded
  the selected entry/target in favor of parallel global-centroid proposals.
  Exact-ray diagnostics also found that none of the original 54 straight routes
  could cut their own corridor with the declared generic instruments. Merely
  substituting the two named native profiles did not fix those original rays.
  Two explicitly different, source-grid-aligned research tool/access candidates
  pass independent checks. Exact selection pass-through now freezes the complete
  entry, target, window and tool. Actionless routes perform zero updates. Original
  routes remain alongside two explicitly different native-action alternatives.
  A real UCSF bridge run made 32 updates, changed the actor and improved nominal
  selection return from 0 to 171.58: 174 target + 11 normal mm³ removed in one
  independently checked stroke. Export, save and restart/recheck passed. This is
  fixed stroke-versus-STOP refinement, not free-form route learning or a controlled
  efficacy benchmark; final worlds remained unopened.
- Raw BTC diagnostic tensor fitting ran on 101,311 voxels; bounded CSA tracking
  produced 176 unlabeled diagnostic paths. Uncorrected DWI is correctly refused
  for tract-aware planning. Correction and alignment are current work.
- Pinned SynthStrip main/no-CSF models ran locally on MPS in roughly 6–7 seconds
  per variant with identical outputs in four runs. No-CSF excludes 214
  source tumor-annotation voxels; main includes them all. Neither is reviewed
  brain/cortex or an accepted cortical-access mask. CPU failure is preserved.
- The same frozen extraction settings ran on preregistered PAT05 in 7.69/6.01
  seconds for no-CSF/main. Independent source/model/output hashes, native geometry
  and distance-map reconstruction passed. No-CSF excludes 538 of 11,437 source
  annotation voxels; main excludes none. Both remain unreviewed proposals, and
  annotation inclusion is not an accuracy metric.
- Structural proposals now save/reopen separately from working anatomy, with
  immutable source/model/run provenance and review-required status. Adding unused
  proposals leaves planning seeds unchanged. Explicit anatomy prohibitions take
  precedence over collection defaults; reviews cannot silently enlarge masks or
  relabel model estimates as observations. Native UI inspection confirms the
  BTC proposal inventory leaves route generation disabled.
- Isolated PyHySCO phantom testing reduced a known-distortion image error from
  9.68% to 0.40%, with 0.395-mm displacement error. The zero-distortion solver
  failure and unsupported BTC phase-encoding geometry remain explicit blockers;
  no patient correction or FSL execution was performed.
- A separate experimental physical-vector solver passes a bounded rotated-pair
  phantom and an independently certified identical-input branch. Image error
  falls from 9.68% to 1.07%, with 0.315-mm field error; coordinate-invariance
  failures in two earlier revisions are retained. All nonzero runs reach their
  40-iteration limit. Thirty-four focused checks and independent artifact/error
  verification passed. This does not promote any patient preprocessing gate.

## Validation record

The latest root integrated run passed **794 tests in 176.14 seconds**, with four
existing DIPY basis-deprecation warnings, from an immutable archive of `0bffeaa`.
Its bytes stayed unchanged throughout testing. The real public-case preflight
also passed both policy profiles and all three seeds with zero gradients before
registered feature-unit execution. Exact identities and results are in
`artifacts/validation/native-feature-units-runner-v1/`. The preceding
743-pass/one-failure sweep is retained: its orchestration fixture needed to
isolate a newly added earlier preflight, while production rejection was correct.
The test-only repair passed focused checks before the earlier 749-pass sweep and
this latest complete validation. The new standalone proposal module is outside
this archived test snapshot and is being independently reviewed.
Structural adversarial checks found
and fixed metadata precedence, mask-enlargement and provenance problems;
their focused checks passed before each incremental commit. Native
adversaries also caught temporal shaft borrowing, mutable preview descriptors
and checkpoint-resume tampering. Test counts describe their recorded snapshots.
The independent native checker improved from 17.638 to 4.310 seconds on the same
saved patient history, with identical geometric certificates and 91 regression
tests. These are local single-run timings, not a latency distribution.
Native initial-geometry reuse lowered median profiled reset time from 0.462 to
0.0259 seconds without changing captured outputs. Three further 30-second runs
completed 14/11/11 updates versus 7/6/5 previously; only seed 47 improved its
selected score. SEARCH remained best. Renderer hydration decreased from a
352-ms median to 134 ms in a warm-cache V8 benchmark; this is not GPU latency.
See `EXPERIMENT_LOG.md` for executed experiments, including negative results.
The Qt prototype passed its initial standalone runtime and GUI checks. Its next
build was stopped following the user's Electron decision. The React/Electron
package passed its own imaging/search checks and 31 main-process boundary and
lifecycle tests. The current 69 renderer tests and 31 viewer tests cover source
hydration, evidence status, A/B identity, independently accepted removal overlays
and population-prior sampling. The complete
training bundle has its own source-bound verification receipts. Qt results do
not certify Electron. No second-Mac test is claimed.

The previously verified refinement shell (renderer `5fe94739…`, Python `377284f6…`)
recovers a six-update cancelled checkpoint after app restart and adds 26 updates
under the original contract. Accessibility decrement/increment now changes the
actual checked replay between 0/0/41,919 and 174/11/41,745 target-removed,
normal-removed and residual-target mm³. Matching screenshots document the fix.
Display-normal averaging improves shading without moving any source triangles;
a separate local probe adds 75.5 ms of worker preparation to 332.0 ms of source
mesh preparation. This is a display tradeoff, not a GPU speedup claim.

## Completion gates still open

Native-resolution geometric simulation now works under explicit hypothetical
access assumptions; reviewed anatomy and surgical validity remain open.
Corrected/accepted functional reconstruction, fully integrated comparison and
refinement interaction, broader cohort benchmark and release remain incomplete.
Frozen/adapted initialization has been tested on analytic and procedural-native
fixtures, including transfer to one reused patient development simulation;
patient-population training and external evaluation remain open. Developer ID
signing/notarization and independent physical-Mac
validation remain open. No locked final cohort has been opened. Clinical deficit
probabilities remain unavailable; simulator success cannot establish clinical
readiness. Current evidence supports a local research prototype only.

No cloud infrastructure or billing changes have been made. Source imaging stays
outside Git. Numerical arrays and checkpoints are excluded; compact experiment
records and source snapshots preserve the executed development evidence.
