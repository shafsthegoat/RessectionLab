# POC-first release gate ledger

Active user steering: October 16, 2026. The objective includes a finished research
product, patient datasets with diverse tumor locations, and real and synthetic
instances. The source is the user-supplied
`RessectionLab_POC_First_Codex_Supergoal.md` and the supplied October 14 recording.
The missing attached operative guide was explicitly waived by the user; primary
surgical sources are being researched to inform the environment and full-operation
development plan.

## Current source and permissions

Latest remote `main` was fetched and pulled before implementation:
`cef9705bc644f9e5e7b6e4d67e7f96d508a10be6`, the merge of PR #12. Work proceeds
on `codex/poc-first-rehearsal` in an isolated checkout. The original branch and
unfinished local edits are preserved. Existing local data and model caches are
reused; source files remain outside Git. The confirmed author identities are
`skamal23 <sayemkamal12@gmail.com>` and
`shafsthegoat <shafrir.p@gmail.com>`. Coherent sequential commits credit both.

## Gates and evidence required

| Stage | Current state | Evidence required to close it |
| --- | --- | --- |
| 0. Inspect and reuse | Latest merged workbench pulled; source/model/corpus audit completed | Actual source, runtime, reference and failure records; maintain original data/roles. |
| 1. Inferred anatomy in Electron | Software POC passed: three real cases/four retained model runs hash/grid checked; actual current GUI interaction and separate native save/reopen completed | Anatomical accuracy, coverage review and population validation remain open; no clinical gate is implied. |
| 2. Narrow two-tool episode | Software POC passed: actual tool actions, play/pause/scrub, two exact-state branches, native save/fresh process reopen, screenshots and independent source/geometry/accounting checks | Physiological calibration and native-collection adapter remain open; the bounded canonical-bundle episode is operable. |
| 3. Real/virtual corpus diversity | Construction plan frozen; resumable qualification of existing eligible bundles started after Gates 1–2 | Qualified actual patient bundles, reviewed location diversity, one biophysical host pilot, then bounded descendants. Count unique people separately from variants. |
| 4. Mechanics and planning | Existing search and negative PPO results retained | Named physical/observation measurements, independent tests and matched observation/budget comparisons before promoting new learning. |
| 5. Robotics | Deferred until matched trajectories and mature simulator exist | Rights, embodiment, timestamps and frozen-baseline evidence; images/labels alone cannot close this stage. |

The first release gate requires stages 1 and 2. Closing it does not complete the
larger user objective. It permits the corpus and mechanics work to proceed in
the declared order.

## What can operate now

The latest Electron anatomy workbench opens the real UCSF-PDGM-0004 MRI with
source tumor labels, separate MONAI tumor estimates and SynthStrip extraction.
CT skull/brain and TOF vessel outputs are available on separate named patients.
The independent artifact/bridge audit checks exact original bytes, mask cells,
native affines, saved views and unchanged model caches. The new rehearsal uses
that same viewer, source case and persistence path. The [first-release result](../artifacts/poc-first-rehearsal-v1/RESULT.md)
records actual GUI actions, 55/139 mm³ branch outcomes, native save/reopen,
independent checks and retained failures. The [GUI-save audit](../artifacts/poc-first-rehearsal-v1/gui-save-audit.json)
verifies exact named arrays and all 32 embedded original files across three
separate source patients. Rehearsal effects are generated research states, not
recorded surgery. The [operative workflow](operative-workflow-model.md) maps
source-linked full-operation phases and OR observations to future work.

Current cohort evidence and prospective targets are in the
[construction plan](patient-corpus-plan.md). Its inventory records 291 materialized
source-family identities, nine inspected development/TRAIN bundle people and
zero complete same-person brain/tumor/skull/vessel cases. These counts do not
mean 291 reconstructed or anatomically validated patients. Anatomical location
is unknown when only an arbitrary-frame centroid is available.

## Accuracy and publication gate

Publication readiness requires claims with matching evidence, not a test count
or convincing animation. Freeze intended use, source identities, preprocessing,
model/checkpoint lineage, splits, failure denominators and evaluation criteria
before a held-out run. Report patient-level distributions and confidence bounds
for appropriate reference-mask overlap, boundary error, registration landmarks,
vessel topology/coverage and failure cases. Compare only compatible annotation
definitions. Keep protected UPENN and existing cohort roles intact.

The current UCSF tumor overlap check is developmental and has known BraTS
training overlap. It cannot establish external accuracy. Registration software
consistency cannot establish anatomical accuracy, and simulator conservation
cannot establish tissue, vascular or neurological validity. Those qualifications
remain open until measured or reviewed evidence is available.

## Next executable actions

Qualify the first real-case tranche and one publisher-described healthy
vascular host, measure anatomical diversity and run the bounded
biophysical virtual-glioma pilot described in the construction plan. Expand
mechanics along the operative workflow with explicit sources, observations and
validation requirements. Do not restart the failed PPO campaign merely to
produce training activity.
