# RessectionLab

A local macOS research workspace for inspecting patient-specific glioma access
routes and testing simulated resection strategies.

Development is in progress. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for what
has actually been run. The October 4, 2026 [master plan](MASTER_PLAN.md),
[annotated references](ANNOTATED_REFERENCES.md), and
[implementation handoff](IMPLEMENTING_AGENT_HANDOFF.md) define the project.

The active Mac application uses **Electron, React and TypeScript**. The Python
engine handles physical geometry, patient-specific training, evidence and
independent validation. The earlier Qt prototype is retained as a tested
historical reference. Application source is in `desktop/`; the
[desktop workflow guide](docs/desktop-workflow.md) covers opening imaging,
inspecting evidence, comparing routes, refinement, replay and local saving.

The standalone Electron app has passed real MRI loading, full-tool route
comparison and native workspace-save checks on this Mac. Its Python training
bridge separately passes actual patient-update and independent native-replay
checks. See [Electron packaging](docs/electron-packaging.md) for building and
validating each complete app snapshot.

## Local development

The initial tested dependency environment is Apple Silicon, macOS 26.6 and
Python 3.12. Use an isolated environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest
```

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
pnpm test:main
pnpm test:renderer
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
- Patient-specific learning updates a policy under frozen simulator assumptions.
  Optimization, checkpoint selection and independent evaluation use separate
  worlds. Search remains a required comparator and may be the better planner.

## Data and privacy

Raw public imaging, local case bundles and checkpoints are excluded from Git.
Download only named cases under verified terms and retain source checksums.
Mirrored data is identified as such; a mirror checksum does not prove byte
equivalence to an authoritative release. The application is local-first; no
cloud processing or paid infrastructure is required.

The interface is original. Medivis supplies workflow inspiration only; no
affiliation, compatibility, endorsement or device clearance is implied.
