# Agent guide and documentation index

Start here if you are a coding agent or a new contributor. This file says where
things live. [Current status](docs/current-status.md) says what works now, and
[CONTRIBUTING.md](CONTRIBUTING.md) sets the maintenance policy. Keep this index
current when you add, rename or retire a maintained guide.

## Project in one paragraph

RessectionLab is a local-first research workbench for patient-specific
neurosurgical planning and rehearsal: **Cases → Scans → Anatomy → Plan →
Rehearse**. The Electron/React/TypeScript desktop (`desktop/`) talks to a Python
engine (`src/resectionlab/`) through a sidecar, with an optional pybind11 C++
backend. `SurgicalEngine` is authoritative for actions, tissue state, replay and
save/reopen. Search and learned policies must share observations, actions,
constraints and evaluation. It is a research prototype, not a medical device.

## Rules that apply to every change

- Follow [CONTRIBUTING.md](CONTRIBUTING.md): update the existing guide instead of
  adding a new status report, handoff or per-step diary; retire superseded
  material in the same change.
- Keep provenance separate: source imaging, supplied annotations, model estimates,
  population priors and simulator-generated experience. Missing evidence stays
  UNKNOWN, never zero risk. See [README › Evidence and interpretation](README.md#evidence-and-interpretation).
- Geometry uses a declared physical frame in millimeters (RAS+ in the desktop;
  LPS sources convert once). See [viewer contracts](desktop/src/viewer/README.md).
- Do not reopen guarded legacy planning/training paths when reproducing earlier work.
- For visible desktop changes, follow the
  [UI principles and review checklist](desktop/src/README.md#desktop-ui-principles)
  and inspect the real app at 1440 × 900 and 1050 × 720.
- Patient data, models and many receipts live on the owner's optional HDD, not in
  Git. See the [local storage ledger](docs/local-storage-ledger.md). Ordinary
  development must not require that drive; report missing inputs explicitly.

## Build and test

| Task | Command or guide |
| --- | --- |
| Python environment | [README › Local development](README.md#local-development) (`requirements-lock.txt`, `pip install -e . --no-deps`) |
| Desktop | `pnpm --dir desktop install --frozen-lockfile`, `pnpm --dir desktop test`, `pnpm --dir desktop build`, `pnpm --dir desktop start` |
| Python tests | `.venv/bin/python -m pytest tests/<focused files> -q`; run the subsystem you touched |
| C++ backend | [C++ acceleration record](docs/cpp-acceleration.md) |
| Packaging | [Electron packaging](docs/electron-packaging.md), [dependency notices](docs/dependency-notices.md), [dependency locks](docs/dependency-lock-separation.md) |
| Reproducing reports | [Artifact reproducibility](docs/artifact-reproducibility.md), [rehearsal support scripts](scripts/rehearsal_support/README.md) |

## Code map

| Path | Contents |
| --- | --- |
| `src/resectionlab/surgical_engine.py` | Authoritative engine: actions, tissue state, replay, snapshots |
| `src/resectionlab/surgical_*.py` | Instruments, physics, planning, cranial/joint search, procedure policy, RL wrapper (`surgical_rl_env.py`) |
| `src/resectionlab/surgical_native_*.py`, `native_backend.py`, `geometry_native.cpp` | Optional compiled backend and its control layer |
| `src/resectionlab/anatomy_*.py`, `patient_*.py`, `scan_*.py` | Scan intake, reconstruction, patient studies |
| `src/resectionlab/desktop_bridge.py` | Python sidecar entry point used by `desktop/electron/sidecar.cjs` |
| `src/resectionlab/app/` | Historical Qt prototype; `app/reslice.py` is still used by registration |
| `desktop/src/` | React renderer; component contracts in [desktop/src/README.md](desktop/src/README.md) |
| `desktop/src/viewer/` | Three.js viewer, coordinate and rendering conventions in [viewer/README.md](desktop/src/viewer/README.md) |
| `desktop/electron/` | Main process, sidecar, packaging and hidden-app `verify-*.cjs` checks |
| `scripts/` | Acquisition, preparation, benchmark and verification entry points |
| `tests/` | Python tests (the `test_surgical_*` group covers the engine) |
| `manifests/` | Dataset acquisition records, experiment declarations, archive index |
| `artifacts/` | Small committed evidence and result summaries (`RESULT.md`) from dated milestones |
| `testcases/`, `outputs/` | Git-ignored local data and run outputs (a few legacy outputs are tracked) |

## Documentation by topic

**Status, goals and planning**
- [Current status](docs/current-status.md): maintained entry point and evidence limits
- [Decision-system objective](docs/decision-system-objective.md) and [progress](docs/decision-system-progress.md)
- [Supratentorial procedure policy](docs/supratentorial-procedure-policy.md): what an acquired case needs before a full procedure
- [Approach comparison checkpoint](docs/approach-comparison-checkpoint.md): native comparisons and evidence-aware planning
- [Next milestone: motor risk and planning evaluation](docs/motor-risk-and-planning-evaluation-next.md)
- [Research claims and evidence](docs/MEDIVIS_RESEARCH_TARGET.md), [innovation ledger](docs/INNOVATION_LEDGER.md), [environment and model card](docs/environment-card.md)
- Original research plan and references: [MASTER_PLAN.md](MASTER_PLAN.md), [ANNOTATED_REFERENCES.md](ANNOTATED_REFERENCES.md), [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)
- Results history: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md); retired steering and diaries: [retired documentation index](docs/retired-docs-20261126.md)

**Desktop workflow**
- [Using the Mac workspace](docs/desktop-workflow.md): scan-to-rehearsal, practice, replay, reproduction
- [Native patient anatomy workbench](docs/patient-anatomy-workbench.md), [fresh patient reproduction](docs/fresh-patient-reproduction.md)
- [Interactive patient planning](docs/interactive-patient-planning.md), [research planning](docs/research-planning.md), [neighboring paths panel](docs/neighboring-paths-panel.md)
- [Rehearsal performance](docs/rehearsal-performance.md), [surgeon rehearsal](docs/surgeon-rehearsal.md), [operative demonstration](docs/operative-demonstration.md)
- [Vascular rendering parity plan](docs/MEDIVIS-PARITY-PLAN.md): phases, gates and measurements for the vessel display work
- [POC A workbench](docs/poc-a-workbench.md) and [first-release gate ledger](docs/poc-first-gate-ledger.md) (October foundation)

**Data, cohorts and reconstruction**
- [Data rebuild and native study migration](docs/data-rebuild.md), [local storage ledger](docs/local-storage-ledger.md)
- [Dataset research catalog](docs/DATASET_ACQUISITION_ADDENDUM.md), [cohort registry](docs/cohort-registry.md), [patient corpus plan](docs/patient-corpus-plan.md)
- EDEN2020: [segmented synthetic cohort](docs/eden2020-healthy-synthetic.md), [local intake and vessel review](docs/eden2020-integration.md)
- Acquisition records: [public cases](docs/data_acquisition.md), [PAT05](docs/data_acquisition_pat05.md), [PAT16/PAT20](docs/data_acquisition_pat16_pat20.md), [BTC recovery](docs/btc-recovery-workspace.md), [Lausanne](docs/lausanne-original-ingestion.md), [ReMIND-001](docs/remind-001-development-acquisition.md), [RESECT](docs/resect-selective-acquisition.md)
- Brain extraction: [method](docs/brain_extraction.md) and its independent reviews (`docs/brain-extraction-*`)
- Vessels: [acquired vascular geometry](docs/native-vascular-geometry.md), [vascular graph](docs/vascular-graph.md)
- Diffusion and function: [diffusion pipeline](docs/diffusion-pipeline.md), [functional evidence contract](docs/functional_evidence.md), [population-prior sensitivity](docs/functional-evidence-integration.md), [critical structure evidence](docs/critical-structure-evidence.md), [prior registration](docs/prior-registration.md)

**Engine, instruments and physics**
- [Surgical toolbox API](docs/surgical-toolbox-api.md), [environment format](docs/surgical-environment-format.md), [shared guards](docs/shared-surgical-guards.md), [instrument audit](docs/surgeon-tool-audit.md)
- [Surgical physics V2](docs/surgical-physics-v2.md), [simulator qualification](docs/simulator-qualification.md), [CUSA contact model](docs/cusa-contact-model.md), [bipolar heat](docs/bipolar-thermal.md), [planar blade](docs/planar-blade-maneuver.md)
- Continuum mechanics: [voxel continuum](docs/voxel-continuum.md), [donor transfer](docs/continuum-donor-transfer.md), [domain sensitivity](docs/continuum-domain-sensitivity.md), [contact reference](docs/continuum-contact-reference.md), [performance](docs/continuum-performance.md) and the other `docs/continuum-*` records
- Tissue mechanics evidence: [validation contract](docs/tissue-mechanics-validation.md), [measurements](docs/tissue-mechanics-measurements.md), [frameworks](docs/tissue-mechanics-frameworks.md), HBE/FEBio records (`docs/hbe-*`, `docs/mechanics-*`)
- Surface quality: [local surface audits](docs/local-surface-audits.md), [local surface filter](docs/local-surface-filter.md)

**Planning, search and RL**
- [Surgical RL environment](docs/surgical-rl-environment.md), [RL readiness](docs/rl-readiness.md), [bounded RL pilot](docs/surgical-pilot.md)
- [Generated sequential benchmark](docs/generated-sequential-benchmark.md), [sequential milestone results](docs/sequential-milestone-results.md) and [execution record](docs/sequential-milestone-execution.md), [canonical endpoint](docs/canonical-endpoint-control-execution.md)
- [Consequence comparison](docs/consequence-comparison.md), [joint corridor/sector strategies](docs/joint-anatomy-strategies.md), [sector strategies](docs/sector-strategies.md), [coordinated burring](docs/coordinated-burring.md), [planning budget](docs/planning-budget.md)
- Research framing: [RL optimization hypotheses](docs/rl-optimization-hypotheses.md), [clinical inputs and RL utility](docs/rl-clinical-inputs-and-utility.md), [observation gap](docs/rl-observation-gap.md), [neurological outcome contract](docs/neurological-outcome-contract.md) and [limits](docs/neurological-outcome-inference-limits.md)
- Earlier native-task learning: [native simulation](docs/native-simulation.md), [native spatial task](docs/native-spatial-task.md), [native opening learning](docs/native-opening-learning.md), `docs/native-axis-*`

**Clinical and product research**
- [Glioma operative research dossier](docs/glioma-operative-research-dossier.md), [clinical feasibility review](docs/glioma-simulation-clinical-feasibility-review.md), [operative workflow model](docs/operative-workflow-model.md), [patient position model](docs/patient-position-model.md)
- [Medivis project blueprint](docs/medivis-project-blueprint.md), [Medivis patent review](docs/medivis-patent-opportunity-review.md)

Docs not listed here are dated records of a single run or audit. Read them for
their evidence; do not treat their instructions as current.
