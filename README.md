# RessectionLab

The [simulator qualification workflow](docs/simulator-qualification.md) now
checks source evidence, exact replay, access-inclusive removal and lattice
force/energy behavior before physiological planning. It exposes grid-dependent
mechanics and keeps measured-response and large-scale RL gates open.
An [opt-in voxel continuum](docs/voxel-continuum.md) adds source-owned mechanics,
stress relaxation, mechanical exposure and checked retraction/resection replay;
its patient material and injury claims remain unvalidated.
An [opt-in vascular graph](docs/vascular-graph.md) separates compression,
occlusion, transection, wound sealing and declared supply/drainage signals.
Matched generated maneuvers pass independent numerical checks; patient-specific
vascular evaluation and physiological RL remain on hold.
The [acquired vascular geometry checks](docs/native-vascular-geometry.md) now
retain original CTA voxel IDs, distinguish veins and sinuses, and certify a
separate H₁ representative graph while exposing missing and ambiguous anatomy.

The [full case rehearsal collaborator handoff](artifacts/full-case-rehearsal-20261020/README.md)
includes the saved meningioma source and substantial primary resection, same-case
vascular rescue, checkpoints, action histories, audit evidence and portable macOS
setup/launch/check commands on `codex/poc-first-rehearsal`. It removed 81.34% of
the owned target before a budget-limited stop. Playback is very slow; full final
GUI playthrough and frame-audit limitations are explicitly retained. These
selected data files are a tracked exception to the usual local `testcases/` rule.

The [Research Planning desktop integration](docs/research-planning.md) now
compares observation-only greedy and a frozen learned continuation in the
existing source-bound instrument workbench, with checked native replay and
save/reopen. The UCSF development result remains negative: greedy hits a shaft
collision and the learner stops; neither adds target removal. This does not
close the admitted TRAIN glioma or hybrid/search benchmark gates.

The [bounded two-instrument task](docs/coordinated-burring.md) now learns to
schedule shallow bone burring and debris aspiration in the unchanged simulator.
All three Q-learning seeds match the successful script; either instrument alone
fails. This is fixed-motion scheduling on one development anatomy, with native
replay and strict safeguards preserved, not clinical or transfer validation.

The healthy IXI multimodal workflow preserves acquired T1, MRA and diffusion
sources, measured rigid registration estimates and diagnostic tensor scalars.
Use `./scripts/launch_ixi_multimodal.sh anatomy` or `simulator` for the prepared
local study. The canonical simulator retains partial vessel exclusion and native
source views; tumor-removal RL refuses this case because it has no tumor target.
Preparation and verification are in `scripts/prepare_ixi_multimodal.py` and
`scripts/verify_ixi_multimodal.py`; local source inventories and QC stay in the
Git-ignored `testcases/` directory. No tract or functional protection is inferred
from uncorrected diffusion.

The [anatomy visual/RL integration check](artifacts/anatomy-integration-v3/RESULT.txt)
adds topology-preserving vessel-margin previews, clearer source-derived rendering,
and a fixed-policy Electron comparison with exact shared-engine state matches.
Launch the prepared local examples with `./scripts/launch_anatomy_fidelity.sh mri`,
`cta`, or `rehearsal`. No inference or training runs when opening these examples.

The [October 18 integrated instrument rehearsal](docs/integrated-rehearsal.md)
adds source-pinned CT/MRI drilling followed by a retained retractor hold and
CUSA removal, second-case MRI interoperability, shared persisted protection,
measured action costs and inspectable bimanual telemetry. We referenced
[Yilmaz et al. (2022)](https://doi.org/10.1038/s41746-022-00596-8) for the
behavior-measurement framework. The [bounded learning pilot](docs/surgical-pilot.md)
keeps its actual optimizer updates, baselines, costs and limitations separate
from anatomical or clinical validation.

Current work follows the [October 16 first-release gate ledger](docs/poc-first-gate-ledger.md)
and [patient corpus construction plan](docs/patient-corpus-plan.md): actual
anatomical reconstruction, then one stateful two-tool rehearsal in Electron,
then diverse real patients and explicitly labeled virtual gliomas. The merged
native workbench from PR #12 is the integration base.

The [local Apple Silicon app](artifacts/poc-first-local-package-build-v1/RESULT.md)
now runs the current workbench with its bundled numerical engine. Actual app
checks cover the remembered case-library folder, two-tool rehearsal, branches,
play/pause/scrub, tissue cutaways, separate CT/MRI views and native save/restart.
The packaged-engine audit preserves every tested source array and original file.
This is a local research preview; anatomical accuracy, physiological calibration,
and public-distribution licensing/notarization remain open.

The [first executable patient-source rehearsal](artifacts/poc-first-rehearsal-v1/RESULT.md)
now passes the actual Electron and independent save/replay checks. **Rehearse**
provides two persistent tools, a shared action timeline, play/pause/scrub and two
continuations from the same state. Source anatomy remains unchanged. Current
physics and initial exposure are explicit research assumptions; diverse-case
qualification and anatomical/physiological validation remain in progress.

The [first nine-person corpus qualification](artifacts/patient-corpus-first-tranche-v1/RESULT.md)
now provides resumable source checks and JSON/SQLite search. A bounded, pinned FK
adapter passes independent numerical and source-admission checks; generated
geometry remains distinct from observed tumors and physiological validation.
The [IXI265 integration demonstration](artifacts/ixi265-virtual-pilot-v1/RESULT.md)
also opens fresh brain/vessel estimates and an explicitly generated lesion overlay
in Electron, with native branch save/reopen. Tissue-conditioned growth and
anatomical review remain open.

The [ten-person catalog release](artifacts/patient-corpus-ten-person-v1/RESULT.md)
now adds IXI through a separate frozen release and exposes all ten source cases
in **Cases**, searchable by name, dataset and available anatomy. Catalog opens
verify exact bundle checksums. Actual Electron checks cover BTC discovery/open
and separate IXI vessel-model inspection; source checking does not accept anatomy
as truth or count generated variants as new people.

Two further [acquired SynthRAD CT/MRI cases](artifacts/poc-synthrad-bc-integration-v1/RESULT.md)
now open with fresh brain/skull estimates and pass native save/fresh-process
reopen checks. Their original scans and separate MRI views are preserved. They
remain unreviewed source/model candidates outside the frozen ten-person catalog.

The [separate twelve-source release](artifacts/patient-corpus-twelve-recovery-v1/RESULT.md)
now adds those two cases to **Cases**, preserving the frozen nine/ten releases.
Actual Electron checks show 12 source identities, three skull cases and native
CT/MRI switching. Independent checks preserve earlier receipts and source bytes.
Reviewed tumor-location diversity and anatomical accuracy remain open.

The [IXI tissue-estimate workspace](artifacts/poc-ixi265-tissue-display-package-v1/RESULT.md)
now shows separate tissue estimates and explicit UNKNOWN voxels in actual
Electron. Layer controls, cutaway, original-image switching and native save/fresh
reopen work; the acquired image and earlier references remain unchanged.

The [six BTC source reports](artifacts/btc-six-source-location-evidence-v1/RESULT.md)
now add location and side to **Cases** search. Actual Electron checks find the
expected left-frontal/parietal cases and open the matching acquired scan.
Labels say “Dataset / paper report · review pending”; reviewed anatomy remains
UNKNOWN and unrelated saved versions receive no inferred location.

The [fixed RESECT ultrasound comparison](artifacts/poc-resect-case3-image-registration-v1/RESULT.md)
now measures real paired-image alignment: on all 17 landmarks of one TRAIN
person, the image-only rigid fit increases RMS error from 2.029 to 2.087 mm.
The negative result, unchanged baseline and initial file-format refusal remain
preserved. Improved image similarity does not establish anatomical accuracy.

A local macOS anatomy workbench for acquired patient imaging, saved segmentations,
and linked 3D/source-image inspection.

The **October 14 POC A reset** established Cases / Data / 3D Workspace as the
anatomy foundation. Three separate public cases demonstrate glioma MRI, head CT,
and TOF-MRA with actual locally inferred masks. Original downloaded files,
physical coordinates, registration declarations, and viewing settings are saved
together. Opening a case or changing layers does not run inference.

On the prepared Mac, launch the actual Electron application from this repository:

```sh
./scripts/launch_poc.sh
```

The last opened case is restored. See the [POC A workbench guide](docs/poc-a-workbench.md)
for the walkthrough, model records, acceptance results, and remaining gaps.
The [full surgical planning/rehearsal supergoal](docs/SUPERGOAL_REAL_OBSERVATIONS.md),
[execution ledger](docs/REAL_OBSERVATION_EXECUTION_LEDGER.md),
[master plan](MASTER_PLAN.md), and [handoff](IMPLEMENTING_AGENT_HANDOFF.md)
remain historical context. Simulation and RL work resumes only after the anatomy
gate; this milestone does not establish surgical or clinical validity.

The active Mac application uses **Electron, React and TypeScript**. The Python
engine handles physical geometry, evidence and independent validation. Legacy
patient/policy entry points retain their existing guards. A [scoped generated
opening-task learner](docs/native-opening-learning.md) now supports an explicit
development context. Its [first fixed comparison](artifacts/native-opening-learning-v1/RESULT.md)
completed with a negative learning result: complete search outperformed both
learned policies. The earlier Qt prototype is retained as a tested
historical reference. Application source is in `desktop/`; the
[desktop workflow guide](docs/desktop-workflow.md) covers opening imaging,
inspecting evidence, comparing routes, refinement, replay and local saving.

The [earlier simulation development app](artifacts/integration-first-shared-episode-v1/RESULT.md)
has been exercised on this Mac with generated sequential aspiration/probe episodes,
exact full-tool replay, and public RESECT T1/FLAIR display in separate native frames.
The [saved workspace](artifacts/integrated-workspace-persistence-v1/RESULT.md) now
retains those images, source frames, individual view settings and generated replay
position across restarts. Additional images remain display-only; opening or saving
them does not register them or make them policy inputs. A legacy trained actor runs through the shared backend, while
the desktop episode selector currently offers scripted and SEARCH execution.
The [replay-linked vessel evaluator](artifacts/integrated-episode-vascular-v1/RESULT.md)
now reports generated annotation encounters and unknown coverage for the exact
recorded full-tool paths, separately from removal and clinical injury.
The [matched generated search control](artifacts/shared-search-width4-generated-v1/RESULT.md)
slightly outperforms that actor on one fixed task. No patient generalization is established.
This revised desktop has not been repackaged; see [Electron packaging](docs/electron-packaging.md)
for building and validating each complete app snapshot.

## Local development

The initial tested dependency environment is Apple Silicon, macOS 26.6 and
Python 3.12. Use an isolated environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest tests/test_real_observation_policy.py -q
```

The command above checks the installed historical policy refusals. Generated
development tests may verify software and simulator behavior, but do not establish
real-patient transfer or physical fidelity. Each new real-case slice needs its own
source-bound checks.

`requirements-lock.txt` pins the current numerical engine, research tools,
tests and build tools without Qt or VTK. The active Electron app does not need
either toolkit. Versions were retained from the installed development
environment; a new clean-environment installation of this separated lock has not
been performed. See [dependency validation](docs/dependency-lock-separation.md)
for the import checks and their limits. Mac packaging remains a separate
acceptance gate for each complete app snapshot.

Only when reproducing the historical Qt prototype, additionally use:

```sh
.venv/bin/python -m pip install -r requirements-legacy-qt-lock.txt
```

That optional file includes the default lock plus the five pinned Qt/VTK
packages. The older `scripts/build_macos.py` and `scripts/launch_app.sh` use
that historical interface; the current Electron commands are below. Historical
Qt worker tests skip when the optional toolkit is absent.

For the current desktop interface, install Node and pnpm, then run:

```sh
cd desktop
pnpm install --frozen-lockfile
pnpm test
pnpm build
pnpm start
```

The development app uses the repository's isolated Python environment. The
packaged app bundles its own numerical engine. `pnpm dev` serves an explicitly
read-only browser preview when an ignored public-case export is available;
native open/save and scientific operations run in Electron.

## Public development cases

The creator-source BTC case can be acquired and prepared with:

```sh
.venv/bin/python scripts/acquire_btc_case.py
.venv/bin/python scripts/prepare_btc_case.py --annotation-threshold 0.5
```

This writes `outputs/cases/BTC-sub-PAT28.ressectionlab`. Its T1 includes the full
head, so hypothetical route generation remains blocked without reviewed
research support. A reviewed brain envelope alone does not certify cortical
access. The source annotation is fractional; 0.5 is a recorded
research threshold, not a clinical probability. Unknown-timed context is withheld.

For the limited UCSF structural mirror used in viewer/route development:

```sh
.venv/bin/python scripts/acquire_public_case.py --inspect-nifti
.venv/bin/python scripts/prepare_case.py
```

The resulting bundle is `outputs/cases/UCSF-PDGM-0004.ressectionlab`. Its source
equivalence remains unverified. See the acquisition guides in `docs/` for
pinned releases, licenses, checksums, coordinates and missing evidence.

## Evidence and interpretation

- Source imaging, supplied annotations, estimated anatomy, population priors and
  simulator-generated experience have separate provenance.
- Route accessibility is distinct from tissue removed in a legal simulated
  sequence. All geometry uses a declared physical frame and millimeters.
- Motor/language proximity and modeled structural encounters are research
  surrogates. Clinical neurological-deficit probabilities are unavailable.
- Missing diffusion, functional mapping, vessels or skull anatomy remain
  unassessed. Public MRI plus a plausible visualization does not establish
  surgical safety or clinical readiness.
- Learning records must distinguish observed patient experience from generated
  simulator experience and search-generated imitation labels. Record model
  ancestry and simulator assumptions; keep hidden simulator properties out of
  deployment inputs. Claims about recorded offline RL require actual actions,
  successor observations, timing, censoring and supported observed endpoints.
  Search remains a required comparator and may be the better planner.

## Data and privacy

Raw public imaging, local case bundles and checkpoints are excluded from Git.
Download only named cases under verified terms and retain source checksums.
Mirrored data is identified as such; a mirror checksum does not prove byte
equivalence to an authoritative release. The application is local-first; no
cloud processing or paid infrastructure is required.

The interface is original. Medivis supplies workflow inspiration only; no
affiliation, compatibility, endorsement or device clearance is implied.
