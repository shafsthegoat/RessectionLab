# Current status and working entry points

Repository upkeep follows the [maintenance policy](../CONTRIBUTING.md): keep current-use material,
retire obsolete code/docs/artifacts, and archive useful history outside the tracked tree.

As of November 17, 2026, the maintained application is Electron/React/TypeScript
with the Python engine and optional compiled C++ backend. This tree combines
the November 16 structure-aware harm and canonical endpoint work (`b80d2e9e`,
PR #20), repository cleanup and archive restoration support, and the completed
[EDEN2020 segmented synthetic cohort](eden2020-healthy-synthetic.md).

The EDEN collection contains 15 segmented healthy baselines and 120 organic
tumor scenarios, all DEVELOPMENT. Native checks cover all 135 packages; actual
hidden packaged-app checks cover representative synthetic and healthy cases.
Prepared datasets, model assets and detailed receipts remain on Shafrir's
optional personal HDD and are unavailable to collaborators through a clone.
The [storage ledger](local-storage-ledger.md) records discovery and access limits.

Native patient studies now accept scans together in Cases. Prepare anatomy
uses the available modalities and records an outcome for each stage.
Plan & compare binds an explicitly selected lesion to its source and revision.
Hidden desktop runs exercised supplied UCSF glioma labels and a meningioma
model candidate through geometric comparison, portable save and cold reopen.
The migrated CTA opens for source and label inspection. The migrated EDEN
study keeps eight synthetic targets separate from its healthy anatomy.
[Use the native workbench](patient-anatomy-workbench.md#prepare-a-patient-study) ·
[Reproduce the desktop checks](fresh-patient-reproduction.md#desktop-workflow-and-smoke-test) ·
[Resume the versioned migration](data-rebuild.md#native-study-migration).

The research objective remains source Case → opening/preparation → tumor work →
checked exit → Save/cold Open, with alternative feasible approaches. There are
zero qualified patient tumor-task completions. Missing source anatomy,
physiological calibration and physical/RL qualification remain open. Generated
engineering controls do not close those requirements.

The November 18 evidence-aware planning milestone adds an opt-in cortical
access assessment and named preservation decisions. The matched generated
overlap control retained the same treatment and residual with less CST
membership removal. The critical-residual control recorded a supported
boundary and checked withdrawal. [Results and reproduction](approach-comparison-checkpoint.md#evidence-aware-planning-milestone).
The subsequent native integration displays source-bound comparisons for
UCSF-PDGM and EDEN, with explicit reassessment and unavailable procedural
outcomes. [Native glioma evidence](approach-comparison-checkpoint.md#native-glioma-evidence-integration).

| Need | Entry point |
| --- | --- |
| Objective, capability boundaries and next dependencies | [Decision-system objective](decision-system-objective.md), [progress and evidence](decision-system-progress.md), [procedure policy](supratentorial-procedure-policy.md) |
| Current architecture and backend limits | [Desktop architecture/workflow](desktop-workflow.md), [C++ delivery](cpp-acceleration.md), [packaging](electron-packaging.md) |
| Native comparison, generated procedure controls and matched baseline | [Verified results and reproduction](approach-comparison-checkpoint.md) |
| Build and ordinary verification | [README development commands](../README.md#local-development); `pnpm --dir desktop test`, `pnpm --dir desktop build`; focused Python tests for the touched subsystem |
| Independent replay/monitor tooling | [Maintained rehearsal support](../scripts/rehearsal_support/README.md) |
| Source collections and data-root workflow | [Local HDD ledger and access limits](local-storage-ledger.md), [EDEN segmented cohort](eden2020-healthy-synthetic.md), [data rebuild](data-rebuild.md); ignored `testcases/README.md` and HDD `catalog/START-HERE.md` |
| Experimental features | [Neighboring paths integration contract](neighboring-paths-panel.md), [continuum](voxel-continuum.md), [research planning](research-planning.md), [interactive approaches](interactive-patient-planning.md) |
| Historical steering, requirements and negative results | [Master plan](../MASTER_PLAN.md), [status narratives](../PROJECT_STATUS.md), [handoff](../IMPLEMENTING_AGENT_HANDOFF.md), [execution ledger](REAL_OBSERVATION_EXECUTION_LEDGER.md) |

SurgicalEngine remains authoritative. Source ownership/admission, anatomical
exclusions, cumulative harm, vascular consequences, STOP/withdrawal, residual
target and replay/save compatibility remain required across both backends.
Historical disabled planners and the optional Qt prototype remain available;
`app/reslice.py` is also used by current registration. Do not reopen guarded
planning/training paths while reproducing earlier work.

The [cleanup execution record](cleanup-execution-20261116.md) explains the verified
historical archives, checkout reduction, restoration and remaining KEEP decisions.
The [earlier support pass](cleanup-20261116.md) remains dated context. Neither
replaces the research goal.
Earlier status documents retain their dated narratives and requirements.
