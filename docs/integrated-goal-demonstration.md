# Integrated rehearsal development demonstration

The October 26 integration now runs real source import, anatomy inspection,
three geometry-dependent access simulations, frozen learned CUSA continuation,
native 3D playback and save/reopen. This is a bounded engineering demonstration,
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
| UCSF-PDGM-0007 | 14 / 11 / 7 | 135 / 115 / 126 | All tools withdrawn normally |

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
  --source testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0007/access-comparison-v1/access-comparison.ressectionlab \
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
successful cleanup. The v4 run used 151 of its declared 160 physical transition
calls, 7.06 seconds wall and 621 MB peak process RSS. The bound includes source
and independent replay plus export verification. It is not GPU presentation
FPS. The initial untrained comparator also passed with a different endpoint;
no superiority over untrained, BC, script or search is established.

To assess a frozen checkpoint without fitting on the second case:

```sh
.venv/bin/python scripts/run_patient_cusa_learning.py \
  --source testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0005/access-comparison-v4/access-comparison.ressectionlab \
  --branch access-superior-access --source-role DEVELOPMENT \
  --frozen-model testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0007/cusa-continuation-trained-v4/ppo-policy.pt \
  --output testcases/outputs/integrated-goal-20261026/my-frozen-assessment
```

Both frozen 0007 BC and PPO failed this prespecified 0005 assessment at shaft
collision, gained no additional target and remained incomplete; the failed
cleanup and accepted actions are retained. No derived successful branch was
manufactured. These are development cases, not held-out transfer validation.

The separate prepared meningioma rescue experiment retains the original PPO
0/2 negative result. The new observation contract preserves named route-local
fluid information, and BC-initialized PPO succeeds for three fixed training
seeds after 64 actual option interactions per seed. Cold checkpoint, observation,
action, reward and full native-state audits pass for all three. That experiment
uses one identical authored injury, inherits successful BC and does not show
added RL benefit or three-patient performance. Matched 64-interaction search
results and all failures remain separate inspectable receipts.

## Inspect and play in Electron

Open the combined learned case:

```sh
./scripts/launch_poc.sh --case \
  "$PWD/testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0007/cusa-continuation-trained-v4/learned-continuations.ressectionlab"
```

In **Rehearse**, **Research Planning** compares the three real endpoints and
opens their branches. **Learned CUSA continuation after selected access**
distinguishes BC from PPO, shows additional modeled removal and checkpoint
identities, and opens the learned starting boundary. **Play replay** animates
actual authoritative instrument poses and anatomy changes; it performs no new
inference. The runner above is the real training and policy execution path.
Timeline scrubbing, pause, source-preserving save and reopen use the existing
shared native engine infrastructure.

The real Electron verifier accepts `access`, `learned` (prepared rescue), and
`cusa` modes, writes only a new ignored output directory, preserves its input,
and checks actual playback across multiple recorded action boundaries:

```sh
pnpm -C desktop build
INTEGRATED_RESEARCH_MODE=cusa \
INTEGRATED_RESEARCH_CASE="$PWD/testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0007/cusa-continuation-trained-v4/learned-continuations.ressectionlab" \
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

Missing skull, complete vessels, function and patient-calibrated mechanics remain
explicitly unavailable. Neurological-deficit and perfusion probabilities are not
reported. A [local reproduction package](integrated-demo-local-package.md) now verifies
all bundled hashes, native branches and cold PPO execution from a relocated
source snapshot on the matching host runtime. The next implementation gate is
a robust checked continuation across the second case's geometry.
