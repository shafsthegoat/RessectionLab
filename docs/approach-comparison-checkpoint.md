# Native approach comparison and procedure experiment

The verified workflow is in commit `e6549a66` on `shaf/next-changes-20261117`.
Local evidence is under ignored `testcases/approach-comparison-goal-v1/`.
The prepared source is on the [personal data drive](local-storage-ledger.md).
Its `archive/desktop-builds-20261117/README.md` indexes the retired
working-branch package and build snapshot, with checked restore instructions.

| Milestone | State | Evidence |
| --- | --- | --- |
| Native comparison | Six default comparisons and the 1 mm CTRL_1277 check cold-verified | `eden-source-verification.json` binds the masks. CTRL_0933 was the first sorted remaining DEVELOPMENT host after CTRL_1277 and CTRL_0866. The first profile measured 129 approaches and 34 scenarios per finalist in 23.48 s with 1.29 GiB peak RSS. |
| Generated procedure | Cold engine replay and hidden Electron verified | `procedure-attempt-02/report.json` records the two checked STOPs, source quantities and the inferior/posterior native protection refusal. The superior approach treated 54 of 108 mm³, retained 54 mm³ and exited with checked STOP and zero source neural or bone removal. `procedure-ui-04/report.json` and its captures show the accepted replay and the renderer's live refusal on unchanged state. |
| Generated baseline | 40 of 40 executed traces replay-verified | `baseline-v1/report.json` records five development conditions from four generated groups, four methods and two declared budget tiers. The completed rows used 1,355 physical requests. The 1,384.37 s campaign charge includes two interruption reservations. |
| Gated learning | Closed at the reward/evaluation gate | This campaign made zero optimizer updates. Low-budget greedy and search reached the same overlap state and reward with 23 and 40 billed requests respectively. |

## Hidden EDEN verification

Each verification directory contains `report.json` and hidden app captures.
The original CTRL_1277 receipts are under the prepared collection's
`verification/` directory. Use `CTRL_1277-01-v3`, `CTRL_1277-08-v2`, and
`CTRL_1277-cold-v1`. The other receipts and workspaces are under
`testcases/approach-comparison-goal-v1/eden-serial/`.

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

Saved workspaces retain their exact manifests, plans and view settings.
Identical imaging copies were reclaimed. Before opening a saved workspace,
restage its `arrays/`, `originals/`, and `revisions/` from the matching
prepared study using the copy step below. The default CTRL_1277 workspace is
`$prepared/verification/CTRL_1277-01-v1/workspace`. The other saved workspaces
are under `eden-serial/workspaces/`. For recovery of local metadata, the drive
holds a checksum-verified copy under
`archive/closed-evidence/eden-approach-comparison-goal-v1-sidecars/`.

For new verification runs, use a fresh output label and one workspace at a time.
Keep the local workspace and renderer transfers on the internal disk and
throttle the read of prepared imaging to 16 MiB/s.

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

## Pre-merge EDEN geometry audit

All six 2 mm tests put a supplied FLAIR streamline inside the spherical end
cap at the target. Five hit at nominal registration. CTRL_1277 target 01 hits
when the FLAIR source shifts 1 mm superior in `registration-09` and
`combined-09`. Those scenarios have the same effect on the streamline.
Boundary padding applies to voxel labels. The independent measurement uses
finite segments from the prepared tract arrays in patient RAS, with native
source binding and registration checked and streamlines kept separate.

| 2 mm task | Nominal guards | Contact inside the fixed target cap |
| --- | --- | --- |
| CTRL_1277 01 | 29 rejected (aslant 22, arcuate 11, both 4). 100 eligible. Eight scenario vetoes and 92 outside the shortlist. | Shifted arcuate 1.786 mm from the target centre. All 100 eligible entries independently fail that scenario. |
| CTRL_1277 08 | 129 rejected. Every candidate hits IFOF and optic radiation. | Optic radiation 0.488 mm and IFOF 1.235 mm |
| CTRL_0866 01 | 129 rejected. Every candidate hits cingulum and IFOF. | IFOF 1.409 mm and cingulum 1.791 mm |
| CTRL_0866 08 | 129 rejected. Every candidate hits IFOF and optic radiation. | Optic radiation 0.210 mm |
| CTRL_0933 01 | 129 rejected. Every candidate hits arcuate, aslant and IFOF. | Aslant 0.326 mm, arcuate 0.896 mm and IFOF 1.154 mm |
| CTRL_0933 08 | 129 rejected. Every candidate hits arcuate. 126 also hit supplied TOF arterial cells. | Arcuate 0.890 mm |

Each ledger evaluated all 129 entries. No entry failed the geometry, superior-entry,
maximum-length, reference-field or selected-target checks. The tasks specify
no additional per-layer exclusions. The prepared entrance is a hypothetical
disk oriented from a local superior estimated head-surface point toward the
fixed target. Cortical and skull access suitability remains unassessed. The
sampled entries all lie at least 28.68 mm above their respective targets.
The disk has one central sample and eight rings 1.875 mm apart with 16
evenly spaced bearings each. The distance from any disk point to its nearest
sample is at most 3.09 mm. Finer sampling can matter for narrower feasible
regions in other tasks. Every route with the same target, radius and
scenario retains the endpoint cap, so changing the patch orientation,
superior-entry scope, candidate order or shortlist budget cannot remove these
six obstructions. The 4 mm entry and 3° axis separations already allow
three distinct candidates in the 1 mm control.

No capsule has full TOF field coverage. The FLAIR field also lacks full coverage
for all entries in CTRL_0866 target 08 and seven entries in CTRL_0933 target 01.
Positive contacts persist in the observed part of a partial field.
CT-measured skull, dura, functional mapping, complete vascular anatomy,
SWI-derived venous labels and independently measured registration accuracy
remain unavailable. Because the near-orthogonal T1 label grid is evaluated
with conservative sheared-cell bounds, six secondary vein readings in
CTRL_1277 target 08 and two each in CTRL_0866 targets 01 and 08 stay unresolved.
CTRL_0866 target 08 has eight confirmed TOF contacts and one historical
TOF clearance-bound-only rejection removed by the exact orthogonal-cell check.
Each of these candidates already fails a known tract contact. The rich and
restricted arms sample the same geometry. The restricted arm lacks the supplied
FLAIR tracts and nominates three routes in each 2 mm task, all refused by the
shared reference.

The final reassessment matched the original anatomy revision, all 129
candidate geometries and all 34 scenario inputs for each task. Counts below
list nominally eligible candidates, shortlisted candidates, scenario vetoes
and suggestions in that order. The archived app timings include its request
overhead. The final timings measure direct backend calls in a fresh
subprocess per task, with suggestion first and the paired comparison reusing
its warmed geometry cache.

| Task | Archived counts | Final counts | Archived suggest / compare (s) | Final backend suggest / compare (s) |
| --- | --- | --- | --- | --- |
| CTRL_1277 01, 2 mm | 100 / 8 / 8 / 0 | 100 / 8 / 8 / 0 | 12.614 / 8.807 | 18.625 / 11.952 |
| CTRL_1277 08, 2 mm | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 11.809 / 15.232 | 18.293 / 15.827 |
| CTRL_0866 01, 2 mm | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 6.661 / 6.306 | 12.120 / 8.401 |
| CTRL_0866 08, 2 mm | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 7.936 / 7.916 | 10.877 / 11.456 |
| CTRL_0933 01, 2 mm | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 7.747 / 8.057 | 12.723 / 11.083 |
| CTRL_0933 08, 2 mm | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 7.475 / 7.532 | 10.141 / 10.376 |
| CTRL_1277 01, 1 mm | 113 / 8 / 0 / 3 | 113 / 8 / 0 / 3 | 12.810 / 7.851 | 19.361 / 10.197 |

All eight 1 mm finalists and all three nominations from each comparison arm
report incomplete coverage. Three distinct static suggestions remain. The
serial reassessment took 187.57 s and peaked at 1.41 GiB per process. The
per-case measurements and fingerprints are in the ignored audit directory's
`backend-reassessment-v4-final/summary.json` and referenced reports. The final
summary SHA-256 is `5045a2c549e9e09e84d62099438e3a6035d279adcfae8e1e4fedc2c9dfa8ec32`.

The assessment repair proves orthogonal clearance against supplied foreground
cells when a circumsphere bound is inconclusive. Coverage includes the requested
margin, and an explicit exclusion with incomplete coverage is refused. A target
endpoint requires support from its backing acquisition. The traversal upper
bound retains cells when target acquisition fails to cover the capsule. Known
contact, unresolved contact and incomplete evidence remain visible. Older
measurements become stale under the new assessment version and can be rerun.
The scenario repair covers all six signed axes for every independent
registration root. An analytic three-root fixture that previously missed a
contact at a worst assessed +0.516 mm now detects it at −0.200 mm. The EDEN
two-root perturbations remain identical. Tail values use the adverse direction
of each metric and require complete measurements of that metric. They describe
equally weighted stress probes with no calibrated clinical probabilities.

Reproduce the archive classification, independently verified tract distances
and serial backend reassessment with the original prepared studies mounted.
Choose fresh output names when repeating the audit.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
evidence=testcases/approach-comparison-goal-v1
audit=testcases/approach-comparison-audit-v1
mkdir -p "$audit"
.venv/bin/python scripts/audit_eden_approach_shortlist.py --prepared "$evidence/eden-studies" --serial "$evidence/eden-serial" --output "$audit/archived-tract-witnesses-final.json" --tract-witnesses
.venv/bin/python scripts/reassess_eden_approaches.py --prepared "$evidence/eden-studies" --serial "$evidence/eden-serial" --output "$audit/backend-reassessment-v4-final"
```

To continue an interrupted reassessment, repeat its command with `--resume`
and the same output directory.

### Pre-merge app evidence

The hidden M2 Pro app opened CTRL_1277 from Cases and displayed source T1,
the supplied anatomy overlays and the missing-anatomy list from the saved
revision. Both 2 mm variants saved justified abstentions. At 1 mm, target 01
saved three static alternatives and reported incomplete reference coverage
for every nomination in both arms. Cold processes restored source identity,
exact plan and view hashes, paired reports, route coordinates and the
selected route. The sampled combined Electron and Python peak was 3.28 GiB.

The ignored `testcases/approach-comparison-audit-v1/eden-ui-v4/` directory holds
the save receipts under `CTRL_1277-01-radius2`, `CTRL_1277-08-radius2` and
`CTRL_1277-01-radius1`. The final cold captures and receipts are under
`CTRL_1277-default-cold-final` and `CTRL_1277-radius1-cold-final`. In the first,
`00b-unavailable-anatomy.png` shows the manifest disclosures and
`other-target-abstention.png` shows the specific arcuate veto and untested
nominal entries. In the second, `reopened.png` shows the three route overlays
and incomplete-evidence cards. The matching saved plan and view sidecars are
in the identically named cold directories without the `-final` suffix.
Use the hidden verification commands above with CTRL_1277 to reproduce them.
To open the exact archived state, restage the prepared source and copy those
two sidecars into the private workspace before the cold check. The temporary
copy of the source imaging was reclaimed after verification.

## Baseline results and recovery

The frozen protocol identity is
`sha256:84732c04c3048761ea6972f7b9f82ffccc9d2af46a0408351d42bb33f7678f1a`.
The final report SHA-256 is
`67accb0b99d3c93dbb90f2592f8cc1ccff7d851dbc90983c0f05cdbdd2c3c4d0`.
All 140 protocol source hashes match commit `e6549a66407c27f887d8d55fae26ccc0145675af`.
The freeze-time HEAD `f6af9575` predates the bound runner and benchmark versions.
The audit also matched the installed C++ binary, profile and all 40 completed
row hashes. It reconciled the stored receipts without re-executing the episodes.
The ignored `testcases/approach-comparison-audit-v1/` directory contains
`audit_frozen_baseline.py` and `frozen-baseline-audit.json`.

| High-budget condition | Fixed narrow then greedy | Conditional | Greedy | Bounded search |
| --- | --- | --- | --- | --- |
| Gated work, 243 mm³ original | 243 mm³ / 48 requests | 243 mm³ / 13 requests | 27 mm³ / 20 requests | 243 mm³ / 95 requests |
| Critical residual, 216 mm³ original | 135 mm³ / 40 requests | 135 mm³ / 10 requests | 27 mm³ / 20 requests | 135 mm³ / 84 requests |
| Off-axis, wide start, 189 mm³ original | 162 mm³ / 44 requests | 189 mm³ / 13 requests | 27 mm³ / 20 requests | 108 mm³ / 54 requests |
| Off-axis, narrow prefix, 189 mm³ original | 162 mm³ / 44 requests | 189 mm³ / 14 requests | 162 mm³ / 44 requests | 189 mm³ / 81 requests |
| Permitted overlap, 189 mm³ original | 81 mm³ / 24 requests | 81 mm³ / 11 requests | 81 mm³ / 23 requests | 81 mm³ / 76 requests |

The conditional controller met or exceeded high-budget search on the frozen
treatment, named harm and withdrawal criteria in four conditions. On gated work
and off-axis narrow prefix, both methods executed the same number of requests
(13 and 14 respectively). The savings came entirely from avoiding previews.
On permitted overlap, all arms retained 108 mm³ of target and removed 54 mm³
each of neural and motor-cortex membership. High-budget search removed zero
CST membership, while the three other arms each removed 54 mm³. The conditional
arm's 11 requests include three refusals and a physical withdrawal at checked STOP.
High-budget search on the off-axis wide-start condition hit its wall limit
and recorded 91.92 s online against a 90 s cap.

Each condition shares a generated source scene, finite inventory, guarded
transitions and endpoint across arms. The fixed-narrow arm always performs
a charged initial 0.4 mm exchange. The other arms share that prescribed exchange
only in the narrow-prefix condition. The conditional controller reads the
supplied anatomy and exact current removed-cell state. Its contact estimates
for both radii use geometry queries whose time is included in online wall time.
Greedy and search also query guarded future transitions, charged as preview
requests. The learned interface receives a reduced feature matrix that omits
the conditional controller's exact alternate-radius contact estimates.
The conditional final state and request count repeat between tiers for each
condition. Both tiers reuse the same generated development geometry and seed.

The ledger has 1,017 speculative and 338 execution requests, including 139
preview refusals and six execution refusals. Each physical
withdrawal is billed. Logical STOPs and whole-tool exterior checks without
motion use wall time, with 26 preview STOPs and 40 execution STOPs recorded
separately. The verifier compared native and primitive-adapter execution and
the final state, endpoint and attempt counts in 676 separate physical requests.
Preview branches are outside that replay scope.

At low budget, greedy and search each executed three physical requests on
overlap, with 20 and 37 additional preview requests respectively. Both reached
the same state and reward of `0.939`. The episode return uses execution attempts
and modeled effects, leaving paid previews out. It also lacks separate named
structure harm terms. The overlap result leaves room for a harm-aware decision
rule. Evidence for learning would require held-out parent groups with consequential
multi-step choices, a harm-aware conditional comparator with the same observations,
and billed-request comparisons at matched treatment, source harm and checked
withdrawal.

The main runner reuses committed rows. The recovery command retries only
infrastructure-interrupted slots once and retains the original interruption
records and reservations, with unchanged caps and source hashes.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -u scripts/run_approach_procedure_baselines.py --profile testcases/approach-comparison-goal-v1/baseline-profile.log --output testcases/approach-comparison-goal-v1/baseline-v1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -u scripts/resume_approach_baseline_interruptions.py --profile testcases/approach-comparison-goal-v1/baseline-profile.log --output testcases/approach-comparison-goal-v1/baseline-v1
```

Acquired-scene entry, instrument work and exit under `surgicalProcedurePolicyV1`
remain refused. EDEN lacks reviewed complete negative coverage of critical
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

Tests used an M2 Pro with 16 GiB RAM and one C++ numerical thread. The backend
identity is `304dea4eae193865a3fcb7e87a288ba58f4812174bfa70dcd7c6e9367e719e78`.
The Python engine owns procedure decisions.

The pre-merge recheck is in
`testcases/approach-comparison-audit-v1/procedure-recheck/report.json`. The
generated post-access control treated 54 of 108 mm³ and passed whole-tool
withdrawal with residual target. The inferior/posterior attempt returned
`PROTECTED_STRUCTURE` with zero target treatment. Both branches replayed
their action histories and checked STOP.

### Pre-merge regression commands

Run one command at a time from the primary checkout. The archived baseline
audit reads the frozen receipts. The generated procedure reproduction above
writes a separate receipt.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m pytest -q tests/test_anatomy_approaches.py tests/test_native_approach_sensitivity.py \
  tests/test_geometry.py tests/test_cpp_geometry.py tests/test_anatomy_alignment.py tests/test_patient_anatomy.py \
  tests/test_anatomy_study_workflow.py tests/test_eden2020_anatomy.py tests/test_surgical_procedure_policy.py \
  tests/test_surgical_procedure_comparison.py tests/test_native_procedure_incompatibility.py \
  tests/test_surgical_rl_procedure_stop_codec.py tests/test_surgical_rl_rewards.py tests/test_frozen_surgical_cache_paths.py
.venv/bin/python -m pytest -q tests/test_workspace_persistence.py tests/test_anatomy_import.py \
  tests/test_build_electron_sidecar.py tests/test_surgical_sequential_task.py tests/test_fixed_preparation_control.py
python3 testcases/approach-comparison-audit-v1/audit_frozen_baseline.py
(cd desktop && node --test --test-concurrency=1 electron/*.test.cjs)
(cd desktop && node --experimental-strip-types --test --test-concurrency=1 tests/*.test.mjs)
(cd desktop && node --experimental-strip-types --test --test-concurrency=1 src/viewer/*.test.mjs)
pnpm --dir desktop build
```
