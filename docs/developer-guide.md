# RessectionLab developer guide

This is the technical companion to the [project overview](../README.md): setup,
data acquisition, evidence rules and where to start in the docs.


RessectionLab is a local-first research workbench for patient-specific
neurosurgical planning and rehearsal. It follows one patient from acquired scans
and supplied annotations through reconstruction, source-slice inspection,
approach comparison and 3D rehearsal in **Cases → Scans → Anatomy → Plan →
Rehearse**. Every input needs an anatomical contribution or an exclusion reason.
`SurgicalEngine` is authoritative: executed replay shows the instrument actions
and tissue changes it recorded, while the geometric preview only illustrates a
selected route. Search and learned policies are compared under the same inputs,
actions and constraints. This is an experimental research platform, not a
clinically validated medical device.

**Start here:**

- [Current status and working entry points](current-status.md): architecture,
  current demonstrations, verification and evidence limits.
- [Desktop workflow](desktop-workflow.md): scan-to-rehearsal walkthrough,
  practice controls, replay and reproduction commands.
- [AGENTS.md](../AGENTS.md): a topic index of the docs, code and data locations for
  coding agents and new contributors.
- [Maintenance policy](../CONTRIBUTING.md): keep current-use material, retire
  obsolete code/docs/artifacts and archive useful history outside the tracked tree.

In **Rehearse**, play the recorded instrument motion and tissue changes together
or export the observed engine frames as a movie. The generated cranial example
supports a separate editable practice episode after verified opening.
[Practice controls and starting case](desktop-workflow.md#practice-from-an-accepted-engine-state).
The selected acquired glioma route ends at an intact STOP before access, and
full acquired-patient completion remains zero.
[See the demonstration and its evidence limits](current-status.md#continuous-operative-rehearsal).

Additional datasets, models and historical archives may be on Shafrir's optional
HDD. The [local storage ledger](local-storage-ledger.md) records its location
and contents. This storage is unavailable to Sayem Kamal and other collaborators.

Earlier milestones (the October POC workbench, patient corpus releases,
integrated rehearsal, learning pilots and their dated artifacts) are summarized
in [current status](current-status.md), the
[experiment log](../EXPERIMENT_LOG.md) and the per-topic guides indexed in
[AGENTS.md](../AGENTS.md). Superseded steering prompts, handoffs and status diaries
are listed with restore commands in the
[retired documentation index](retired-docs-20261126.md). The
[cleanup execution record](cleanup-execution-20261116.md) documents the
archived historical data and verified restoration.

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
tests and build tools without Qt. It includes VTK 9.7.0 for native anatomy
surface preparation, matching the existing `anatomy` extra and installed runtime.
The Electron interface uses Three.js; the packaged numerical engine includes
the VTK geometry modules without Qt. Versions were
retained from the installed development
environment; a new clean-environment installation of this separated lock has not
been performed. See [dependency validation](dependency-lock-separation.md)
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

The [EDEN2020 segmented synthetic-cohort guide](eden2020-healthy-synthetic.md)
describes 15 healthy baselines and 120 organic-tumor projects, their segmentation
layers, preparation commands and verified desktop workflow. The prepared files
are optional assets on Shafrir's personal HDD; cloning this repository does not
provide them. They are DEVELOPMENT engineering rehearsals with estimated
anatomy, not clinically validated cases or admitted training data.

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
