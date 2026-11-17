# Native approach comparison and procedure experiment

## Resume

Work in the primary checkout on `shaf/next-changes-20261117`. The starting revision is
`aa41441f`. The verified external mount is `/Volumes/shafrir` with volume UUID
`9CC3623C-E38B-3EEE-B6BB-B9A06C193700` and data-root marker ID
`756c4d3b-aabf-4837-a354-52aa500af3f9`. Evidence for this experiment belongs in
ignored `testcases/approach-comparison-goal-v1/` or an indexed external location.

Check code and data identities before continuing the first unfinished milestone.

```sh
git status --short --branch
git rev-parse HEAD
export RESSECTIONLAB_DATA_ROOT='/Volumes/shafrir/old coding projects/RessectionLabData'
.venv/bin/python -c 'from resectionlab.data_root import checked_root; print(checked_root())'
```

## Milestones

| Milestone | State | Evidence |
| --- | --- | --- |
| Native comparison | Six default comparisons and the 1 mm CTRL_1277 check cold-verified | `eden-source-verification.json` binds the masks. CTRL_0933 was the first sorted remaining DEVELOPMENT host after CTRL_1277 and CTRL_0866. The first profile measured 129 approaches and 34 scenarios per finalist in 23.48 s with 1.29 GiB peak RSS. |
| Generated procedure | Cold engine replay and hidden Electron verified | `procedure-attempt-02/report.json` records the two checked STOPs, source quantities and the inferior/posterior native protection refusal. The superior approach treated 54 of 108 mm³, retained 54 mm³ and exited with checked STOP and zero source neural or bone removal. `procedure-ui-04/report.json` and its captures show the accepted replay and the renderer's live refusal on unchanged state. |
| Matched baseline | 40 of 40 rows complete and replay-verified | `baseline-v1/report.json` records five conditions from four generated groups, four methods and two matched budgets. The campaign charged 1,384.37 s, including two interrupted attempt reservations, and counted 1,355 physical requests in the completed rows. |
| Gated learning | Closed at the reward/evaluation gate | Zero optimizer updates. The overlap control reached the same final state and reward with 23 greedy requests and 40 search requests at low budget. The reward leaves paid search previews out, so it cannot test the frozen request-efficiency question. A request-aware objective must pass that exact overlap control before any pilot. |

## Hidden EDEN verification

The original CTRL_1277 receipts are under the prepared collection's
`verification/` directory. Use `CTRL_1277-01-v3`, `CTRL_1277-08-v2`, and
`CTRL_1277-cold-v1`. The reconnected runs use
`testcases/approach-comparison-goal-v1/eden-serial/`. Each run has a
`report.json`, hidden app captures, and a pointer to its workspace.

All three hosts restored both target comparisons in a fresh process. At a
2 mm capsule radius and 15 mm entry patch, the six tasks returned zero
suggestions. CTRL_1277 variant 01 reached the sensitivity stage, where all
eight finalists were vetoed. The other five tasks had no nominal survivors.
The receipts retain the candidate ledgers, both input arms, the shortfalls,
and the separate manual geometry and sensitivity checks.

CTRL_1277 variant 01 at a 1 mm capsule radius retained three distinct
alternatives. The target, entry patch, guards and sensitivity settings match
the 2 mm run. Of 129 samples, 113 met the nominal guards and all eight
shortlisted candidates survived 34 declared scenarios. Each arm nominated three
routes and the shared reference vetoed none. The captures and exact cold reopen
are in `CTRL_1277-01-radius1-serial` and `CTRL_1277-radius1-cold-serial` under
`eden-serial/`. Anatomical coverage remains incomplete. These are geometric
measurements on healthy source anatomy with synthetic targets.

The 1 mm hidden run took 49.6 s from launch to receipt, including 12.81 s
for suggestions, 7.85 s for the paired comparison and 0.17 s for Save.
Its cold Cases-to-Plan check took 3.32 s. The sampled combined Electron and
Python peak was 1.72 GiB for the 1 mm run and 3.20 GiB across all completed
EDEN runs. The final passes used a single job at a time. The CTRL_1277
working copy took 141.60 s to stage at the throttled rate.

Use one study at a time. Build the desktop, verify the mounted source root,
and stage a local workspace when it is absent. This path keeps working
comparisons and temporary renderer transfers on the internal disk. The
prepared imaging is read at up to 16 MiB/s, without rebuilding it.
Use a fresh output label for each verification attempt.

```sh
pnpm --dir desktop build
export RESSECTIONLAB_DATA_ROOT='/Volumes/shafrir/old coding projects/RessectionLabData'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export EDEN_VERIFICATION_ROOT="$PWD/testcases/approach-comparison-goal-v1/eden-repro"
host=CTRL_0933
prepared=$(.venv/bin/python -c 'from resectionlab.data_root import checked_root; print(checked_root(expected_uuid="9CC3623C-E38B-3EEE-B6BB-B9A06C193700") / "prepared/eden2020-approach-comparison-goal-v1")')
export EDEN_STUDY="$prepared/$host/patient-anatomy.json"
workspace="$EDEN_VERIFICATION_ROOT/workspaces/$host"
mkdir -p "$workspace"
rsync -rt --bwlimit=16384 --exclude='._*' "$prepared/$host/arrays" "$prepared/$host/originals" "$prepared/$host/revisions" "$EDEN_STUDY" "$workspace/"
export EDEN_WORKSPACE="$workspace/patient-anatomy.json"
cmp "$EDEN_STUDY" "$EDEN_WORKSPACE"
electron=desktop/node_modules/.bin/electron
verifier=desktop/electron/verify-eden-comparison.cjs
EDEN_TARGET=synthetic-target EDEN_RUN="$host-01-repro" EDEN_EDIT=1 "$electron" "$verifier"
EDEN_TARGET=synthetic-target-08 EDEN_RUN="$host-08-repro" "$electron" "$verifier"
EDEN_TARGET=synthetic-target-08 EDEN_RUN="$host-cold-repro" EDEN_EXPECTED="$EDEN_VERIFICATION_ROOT/$host-08-repro/report.json" EDEN_OTHER_EXPECTED="$EDEN_VERIFICATION_ROOT/$host-01-repro/report.json" "$electron" "$verifier"
```

For the smaller-capsule check, stage CTRL_1277 in a separate local verification
root with the same preparation commands. Then run these two phases.

```sh
EDEN_TARGET=synthetic-target EDEN_CORRIDOR_RADIUS=1 EDEN_RUN=CTRL_1277-01-radius1-repro "$electron" "$verifier"
EDEN_TARGET=synthetic-target EDEN_RUN=CTRL_1277-radius1-cold-repro EDEN_EXPECTED="$EDEN_VERIFICATION_ROOT/CTRL_1277-01-radius1-repro/report.json" "$electron" "$verifier"
```

## Baseline results and recovery

The frozen protocol identity is
`sha256:84732c04c3048761ea6972f7b9f82ffccc9d2af46a0408351d42bb33f7678f1a`.
The final report SHA-256 is
`67accb0b99d3c93dbb90f2592f8cc1ccff7d851dbc90983c0f05cdbdd2c3c4d0`.
The profile used 181.87 s and 132 MiB peak RSS. The reconnected execution and
interruption recovery used one simulation worker.

| High-budget condition | Fixed narrow then greedy | Conditional | Greedy | Bounded search |
| --- | --- | --- | --- | --- |
| Gated work, 243 mm³ original | 243 mm³ / 48 requests | 243 mm³ / 13 requests | 27 mm³ / 20 requests | 243 mm³ / 95 requests |
| Off-axis, narrow prefix, 189 mm³ original | 162 mm³ / 44 requests | 189 mm³ / 14 requests | 162 mm³ / 44 requests | 189 mm³ / 81 requests |
| Permitted overlap, 189 mm³ original | 81 mm³ / 24 requests | 81 mm³ / 11 requests | 81 mm³ / 23 requests | 81 mm³ / 76 requests |

The first two conditions above had zero source neural removal. On the overlap
control all four arms removed 54 mm³ each of neural, motor and CST membership
and retained 108 mm³ of target. The conditional arm used the fewest requests
there and recorded three execution refusals. Other methods recorded their
refusals during previews. See each receipt for separate harm dimensions,
modeled time, preview and execution costs, and withdrawal.

The main runner reuses committed rows. The recovery command retries only
infrastructure-interrupted slots once and retains the original interruption
records and reservations, with unchanged caps and source hashes.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -u scripts/run_approach_procedure_baselines.py --profile testcases/approach-comparison-goal-v1/baseline-profile.log --output testcases/approach-comparison-goal-v1/baseline-v1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -u scripts/resume_approach_baseline_interruptions.py --profile testcases/approach-comparison-goal-v1/baseline-profile.log --output testcases/approach-comparison-goal-v1/baseline-v1
```

Acquired-scene entry, instrument work and exit under `surgicalProcedurePolicyV1`
remain pending. EDEN lacks reviewed complete negative coverage of critical
structures, independent working entry and access evidence, and tissue/effect
evidence across the whole two-hand action domain.

## Generated procedure reproduction

From the primary checkout, use new output directories for new attempts. The
post-access program exercises the shared procedure policy, a deliberate tool
preparation, a two-hand state, target work, and physical withdrawal with checked
STOP. The accepted endpoint retains residual target outside its work compartment.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/demonstrate_approach_procedure.py --output testcases/approach-comparison-goal-v1/procedure-repro
PROCEDURE_DEMO_REPORT=testcases/approach-comparison-goal-v1/procedure-repro/report.json PROCEDURE_DEMO_OUTPUT=testcases/approach-comparison-goal-v1/procedure-ui-repro desktop/node_modules/.bin/electron desktop/electron/verify-approach-procedure.cjs
```

The source is acquired healthy anatomy with separately generated geometry. The
Mac is an M2 Pro with 16 GiB physical memory and 10 CPU cores. The installed
numerical backend selected C++ with one native thread. Its identity is
`304dea4eae193865a3fcb7e87a288ba58f4812174bfa70dcd7c6e9367e719e78`.
The procedure-aware resident executor remains ineligible, so the Python
engine owns procedure decisions.
