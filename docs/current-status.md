# Current status and working entry points

Repository upkeep follows the [maintenance policy](../CONTRIBUTING.md): keep current-use material,
retire obsolete code/docs/artifacts, and archive useful history outside the tracked tree.

As of November 16, 2026, the maintained application is Electron/React/TypeScript
with the Python engine and optional compiled C++ backend. The November 16
structure-aware harm and canonical endpoint work was merged in `b80d2e9e` (PR
#20). The separate healthy-source synthetic-cohort task began at `ae13e285`;
its ongoing datasets are independent of this cleanup.

The research objective remains source Case → opening/preparation → tumor work →
checked exit → Save/cold Open, with alternative feasible approaches. There are
zero qualified patient tumor-task completions. Missing source anatomy,
physiological calibration and physical/RL qualification remain open. Generated
engineering controls do not close those requirements.

| Need | Entry point |
| --- | --- |
| Objective, capability boundaries and next dependencies | [Decision-system objective](decision-system-objective.md), [progress and evidence](decision-system-progress.md), [procedure policy](supratentorial-procedure-policy.md) |
| Current architecture and backend limits | [Desktop architecture/workflow](desktop-workflow.md), [C++ delivery](cpp-acceleration.md), [packaging](electron-packaging.md) |
| Build and ordinary verification | [README development commands](../README.md#local-development); `pnpm --dir desktop test`, `pnpm --dir desktop build`; focused Python tests for the touched subsystem |
| Independent replay/monitor tooling | [Maintained rehearsal support](../scripts/rehearsal_support/README.md) |
| Source collections and data-root workflow | [Data rebuild](data-rebuild.md); ignored `testcases/README.md` and the existing HDD `catalog/START-HERE.md` |
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
