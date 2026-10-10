# RessectionLab

A local macOS research workspace for inspecting patient-specific glioma access
routes and testing simulated resection strategies.

Development is in progress. The [full surgical planning/rehearsal supergoal](docs/SUPERGOAL_REAL_OBSERVATIONS.md)
and [October 8 human steering](docs/REAL_OBSERVATION_EXECUTION_LEDGER.md#active-human-steering-october-8)
govern the work, with the later [integration-first directive](docs/INTEGRATION_FIRST_MULTIMODAL_STEERING.md)
requiring shared runtime and desktop integration. The steering permits separately labeled synthetic and
simulator-generated experience for RL development/training, with transfer evaluated
on held-out real patients and appropriate physical measurements.
See the [execution ledger](docs/REAL_OBSERVATION_EXECUTION_LEDGER.md) and
[PROJECT_STATUS.md](PROJECT_STATUS.md) for executed work and remaining gates.
The [master plan](MASTER_PLAN.md), [references](ANNOTATED_REFERENCES.md) and
[handoff](IMPLEMENTING_AGENT_HANDOFF.md) remain historical implementation context.

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

The [current development app](artifacts/integration-first-shared-episode-v1/RESULT.md)
has been exercised on this Mac with generated sequential aspiration/probe episodes,
exact full-tool replay, and public RESECT T1/FLAIR display in separate native frames.
The [saved workspace](artifacts/integrated-workspace-persistence-v1/RESULT.md) now
retains those images, source frames, individual view settings and generated replay
position across restarts. Additional images remain display-only; opening or saving
them does not register them or make them policy inputs. A legacy trained actor runs through the shared backend, while
the desktop episode selector currently offers scripted and SEARCH execution.
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
