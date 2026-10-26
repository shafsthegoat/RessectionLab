# Integrated rehearsal development demonstration

The October 26 integration now runs real source import, anatomy inspection,
three geometry-dependent access simulations, frozen learned CUSA continuation,
new frozen inference from the desktop, native 3D playback and save/reopen. This is a bounded engineering demonstration,
not a complete operation or clinical validation. The full research goal remains
open: skull access is unsupported on these skull-stripped MRI cases, and frozen
continuation failed on the second development patient.

## What runs

[Fresh-patient reproduction](fresh-patient-reproduction.md) describes the two
previously unused development cases, pinned source bytes, explicit sequence and
label roles, actual reconstruction, model installation and native study export.
No reserved evaluation case was opened. Source annotations, estimated brain,
generated simulator transitions and model assumptions retain separate provenance.

Generate three approaches from an intact prepared case using one generic command:

```sh
.venv/bin/python scripts/compare_anatomy_access.py \
  --case testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0007/scene/patient.ressectionlab \
  --allow-modeled-brain-corridor \
  --output testcases/outputs/integrated-goal-20261026/my-access-run
```

The common limits are 48 accepted actions per approach, 200 mm³ modeled
non-target removal and 120 seconds aggregate computation. Each direction uses
a different source brain surface entry and oblique corridor toward the supplied
target. Native whole-tool/shaft, other-hand, material and protection guards
admit each action. The explicitly opted-in brain corridor removes selected
brain-estimate tissue and counts it as non-target removal. It never invents a
skull opening, dura, vessel or functional map. Residual spatial summaries,
checked actions, durations, refusals and independent native replays are saved.

| Case | Target removed, three directions · mm³ | Non-target removed · mm³ | Endpoint |
|---|---|---|---|
| UCSF-PDGM-0005 | 8 / 8 / 6 | 180 / 157 / 199 | All tools withdrawn after checked slower recovery |
| UCSF-PDGM-0007 | 14 / 11 / 15 | 135 / 115 / 119 | All tools withdrawn normally |

The current comparisons are 0005 `access-comparison-v6` and 0007
`access-comparison-v4`. Candidate selection requires at least 12 mm separation
between source brain-surface entries and 30° between inward shaft directions.
Their measured minimum pairs are 24.3 mm / 32.5° and 18.7 mm / 36.2° respectively.
Earlier 0007 directions included a near-duplicate pair (4.24 mm / 19.7°); those
receipts remain, and the new selector refuses an insufficiently distinct set.
All routes still use left CUSA with the other hand parked. They do not yet
compare instrument arrangements or resection orders.

These are small task endpoints with substantial residual target, not maximal
resection or clinical access assessments. A 0005 fast withdrawal initially
failed swept-shaft collision. The retained generic recovery previews the exact
return path with finer time subdivision before committing it; it adds modeled
time and removes no extra tissue. Earlier failed receipts remain available.

## Actual learned execution on the fresh source

The continuation task begins at a verified recorded access boundary before its
cleanup, using the original native engine state. No exposure is fabricated.
Its predeclared gate is at least 30 mm³ additional target, at least two admitted
non-STOP actor actions, at most 20 mm³ additional non-target removal, and checked
deactivation/withdrawal. The actor receives nine tool-frame options with 22
features from published current observations and declared task budgets. Source
arrays, previews and future outcomes are excluded from its inputs.

```sh
.venv/bin/python scripts/run_patient_cusa_learning.py \
  --source testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0007/access-comparison-v4/access-comparison.ressectionlab \
  --branch access-superior-access --source-role DEVELOPMENT \
  --output testcases/outputs/integrated-goal-20261026/my-cusa-run
```

The runner first measures the scripted comparator, fits behavior cloning and
tests its frozen live execution. It then performs two on-policy PPO collection
and update rounds, each with six sampled decisions and two optimizer epochs.
Actor-specific before/after hashes prove the actor changed independently of the
critic. The final checkpoint is cold loaded, makes its own observed-state
choices, and its effects are independently replayed. Both BC and PPO native
branches are exported separately with checkpoint, producer implementation,
runtime, physics-profile and fitting-receipt identities.

On fresh 0007, BC and PPO each selected `inward-2`, `inward-2`, STOP: **36 mm³
additional target, 0 mm³ additional non-target**, two admitted strokes and
successful cleanup. The current v6 run used 151 of its declared 200 physical transition
calls, 6.86 seconds wall and 606 MB peak process RSS. The bound includes source
and independent replay plus export verification. It is not GPU presentation
FPS. The initial untrained comparator also passed with a different endpoint;
no superiority over untrained, BC, script or search is established.

To assess a frozen checkpoint without fitting on the second case:

```sh
.venv/bin/python scripts/run_patient_cusa_learning.py \
  --source testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0005/access-comparison-v6/access-comparison.ressectionlab \
  --branch access-superior-access --source-role DEVELOPMENT \
  --frozen-model testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0007/cusa-continuation-trained-v6/ppo-policy.pt \
  --output testcases/outputs/integrated-goal-20261026/my-frozen-assessment
```

Both original v4 frozen 0007 BC and PPO failed the prespecified 0005 assessment at shaft
collision, gained no additional target and remained incomplete; the failed
cleanup and accepted actions are retained. No derived successful branch was
manufactured. These are development cases, not held-out transfer validation.

A post-failure development revision, v5, retries shaft refusals at the identical
selected endpoint with slower, explicitly recorded durations. It never bypasses
preview/commit checks or the removal cap. Newly fitted v5 BC and PPO still pass
on 0007. On 0005 both frozen actors now admit one stroke (**10 mm³ target,
2 mm³ non-target**) and choose STOP. Checked withdrawal completes, but the
unchanged 30 mm³/two-action task gate remains **unpassed**. This is an observed
early-stop failure, not successful transfer. All original v4 failures remain.

`scripts/export_patient_cusa_episode.py` cold-replays the v5 negative episodes
and exports explicitly unpassed native branches. The actual Electron check at
`testcases/outputs/integrated-goal-20261026/electron-cusa-0005-unpassed-v1/report.json`
played the PPO branch from action 35 through 38 and saved it. The UI shows the
failed task gate separately from completed cleanup. Inspect that preserved case:

```sh
./scripts/launch_poc.sh --case \
  "$PWD/testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0005/cusa-unpassed-native-v1/unpassed-learned-continuations.ressectionlab"
```

Checkpoints require their sealed implementation and runtime. Later policy-source
revisions require new fitting and versioned receipts; do not relabel an older
checkpoint as compatible. The local package documentation distinguishes the
immutable historical v4 snapshot from the current v6 policy/source package.

The v6 runner also binds the archived patient and pinned mirror revision to
the pre-pixel DEVELOPMENT declaration. A CLI role flag alone cannot admit an
unlisted or reserved case. Both v6 frozen 0005 assessments retain the same
10/2 mm³ abstention on the new comparison and now directly export native
unpassed branches, with independent replay and save/reopen (157/200 calls).
A bounded public-preview teacher also missed the original target gate; its
limited local option search does not prove every continuation infeasible.

The separate prepared meningioma rescue experiment retains the original PPO
0/2 negative result. The new observation contract preserves named route-local
fluid information, and BC-initialized PPO succeeds for three fixed training
seeds after 64 actual option interactions per seed. Cold checkpoint, observation,
action, reward and full native-state audits pass for all three. That experiment
uses one identical authored injury, inherits successful BC and does not show
added RL benefit or three-patient performance. Matched 64-interaction search
results and all failures remain separate inspectable receipts.

The prepared-rescue Electron check also plays action 51 through 55 and saves
the archive (`electron-rescue-policy-v5/report.json`). A read-only prefix save
preserves its stored action log without starting full editable replay admission.
It does not promote the inspected cursor to an editable state; full verification
is still required for planning or new instrument actions. Earlier save-timeout
receipts remain available.

## Inspect and play in Electron

Open the combined learned case:

```sh
./scripts/launch_poc.sh --case \
  "$PWD/testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0007/cusa-continuation-trained-v6/learned-continuations.ressectionlab"
```

In **Rehearse**, **Research Planning** compares the three real endpoints and
opens their branches. **Learned CUSA continuation after selected access**
distinguishes BC from PPO, shows additional modeled removal and checkpoint
identities, and opens the learned starting boundary. **Play replay** animates
actual authoritative instrument poses and anatomy changes; it performs no new
inference. The runner above is the real training and policy execution path.
Timeline scrubbing, pause, source-preserving save and reopen use the existing
shared native engine infrastructure.

To execute a policy anew, open either current access comparison, use
**Run a learned continuation**, choose the approach, click **Run frozen CUSA
checkpoint**, and select the compatible v6 `ppo-policy.pt`. The app snapshots
the current workspace, runs actual frozen observed-state inference without
optimizer updates, verifies the trajectory, and opens the saved outcome.
The input file is preserved. **Cancel frozen policy execution** stops the run;
failed display admission restores the input snapshot. Output evidence is under
`testcases/outputs/cusa-desktop/`. Completed cleanup does not promote a failed
task. The action currently requires the local checkout and matching runtime.

The real Electron verifier accepts `access`, `learned` (prepared rescue),
`cusa` (recorded playback), `cusa-execute` (new inference) and `cusa-cancel`
modes, writes only new ignored output directories, preserves its input,
and checks actual playback across multiple recorded action boundaries:

```sh
pnpm -C desktop build
INTEGRATED_RESEARCH_MODE=cusa-execute \
INTEGRATED_RESEARCH_CASE="$PWD/testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0007/access-comparison-v4/access-comparison.ressectionlab" \
INTEGRATED_RESEARCH_CHECKPOINT="$PWD/testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0007/cusa-continuation-trained-v6/ppo-policy.pt" \
INTEGRATED_RESEARCH_OUTPUT="$PWD/testcases/outputs/integrated-goal-20261026/my-electron-run" \
desktop/node_modules/.bin/electron desktop/electron/verify-integrated-research.cjs
```

Source/native/scene verification and actual Electron import/reconstruct/prepare/
save/reopen passed on both fresh cases. Access comparison and fresh PPO playback
have real Electron receipts. The desktop test suites pass; relevant backend
ingestion, access, observed-policy and source-binding tests pass. Eleven older
`test_desktop_bridge.py` fixtures still call intentionally disabled legacy
synthetic/training entrypoints; those failures are reported, not unguarded merely
to make a broad suite green.

Actual new-inference Electron checks passed on both current cases:
`electron-cusa-execute-0007-v1/report.json` records 36/0 mm³ and playback 9→13;
`electron-cusa-execute-0005-v1/report.json` records unpassed 10/2 mm³ and playback
35→38. Both saved the new native episode without altering source or checkpoint.
The inference/replay/export runs took 2.71 s / 54 physical calls and 4.21 s /
157 calls respectively; these are computation measurements, not render FPS.
`electron-cusa-cancel-v1/report.json` confirms actual UI cancellation retained
the source and admitted no learned branch.

Missing skull, complete vessels, function and patient-calibrated mechanics remain
explicitly unavailable. Neurological-deficit and perfusion probabilities are not
reported. A [local reproduction package](integrated-demo-local-package.md) now verifies
all bundled hashes, native branches and cold PPO execution from a relocated
source snapshot on the matching host runtime. The next substantive strategy
step is a bounded comparison of two target-sector resection orders after a
checked corridor, retaining failed reachability and the existing baseline.
The second-case learned gate and full-operation anatomy requirements remain open.
