# A fixed intervention confirms an overvalued tool switch

The learned continuation estimate chooses a worse action at one exposed
ReMIND-045 TRAIN state. After the same eight movements, its preferred switch
predicts a 1.270777 advantage over the best immediate action. Fresh complete
replay instead measures a 0.060553 disadvantage under the unchanged simulator
reward. Greedy search remains the retained planner; this experiment diagnoses
the learned model and does not promote a new policy.

Both branches finish with twelve movements plus STOP, remove the same 51 target
cells (48.637450 mm³), remove no outside-target tissue and have the same
202.179206 mm³ cumulative contact upper bound. Choosing the immediate-best
action at decision nine avoids two switches and 0.553126 mm of travel. The
fixed intervention recovers two-thirds of the prior 0.091106 deficit, while
remaining 0.030553 below greedy's 48.181963 full return.

| Fixed 045 branch | Learned choice | Immediate-best intervention |
|---|---:|---:|
| Remaining return, including the ninth action | 4.528046 | 4.588599 |
| Full simulated return | 48.090857 | 48.151410 |
| Tool switches | 3 | 1 |
| Complete-tool travel, mm | 96.593776 | 96.040650 |
| Target cells / movements | 51 / 12 | 51 / 12 |

Only the ninth choice changes. The same frozen continuation policy controls
the rest of both branches, with the same anatomy, aperture, tool geometry,
candidate rules, reward and absolute 24-action horizon. The full action lists
were sealed before independent replay. This establishes the effect of one
intervention under that continuation policy, not the optimal continuation of
either action or performance of a general immediate-best replacement.

## What the failure identifies

The predicted tail advantage is 1.301331, overriding an immediate cost of
0.030553. Actual subsequent return is another 0.03 worse, so the predicted
difference misses by 1.331331. This directly confirms the previously suspected
overvaluation at this common state; lower training error had not fixed ranking.
No additional fit or parameter sweep follows from this result.

Both terminal catalogs contain 130 slots: 39 duplicates and 91 distinct emitted
proposals. All 91 fail the full-shaft check against remaining native tissue;
none is certified. The recorded catalogs are taken after choices are fixed and
are diagnostic outputs. They establish exhaustion of this candidate family,
not physical inaccessibility or absence of another surgical route.

The next experiment tests the existing corrected obstruction-centered clearing
actions at the retained twelve-move terminal frontier. It preserves the physical
prefix, fixed aperture, instruments, reward, checks and total 24-action budget,
then compares the original STOP with a bounded preparation suffix. Useful
improvement requires additional target access and positive net incremental
return after all preparation costs. A legal clearing action or an unlocked ray
alone is insufficient. The expanded catalog is a declared input/action change.

## Verification and retained execution failure

Both complete fresh routes pass the independent full-tool geometric checker.
The retained branch reproduces every saved compact history and decision field
exactly after JSON representation normalization. An independent saved-result
auditor passes 1,366 checks, covering seals, state/observation joins, choices,
full outcomes, terminal catalogs, accounting and clean termination. The earlier
715-check recovery-point audit and eleven original runner controls are retained.

The original run completed and sealed both plans, then failed during replay
because Python coordinate tuples were compared directly with JSON lists. Its
first fresh replay projection was not published. That failed attempt remains
immutable. A separate replay-only recovery uses the exact two plans, saves each
new certificate before parity checks, and compares all fields as canonical JSON
without omitting values or adding numerical tolerance. Both replays then pass.
It loads no model, checkpoint or feature cache and performs no replanning.

Eight focused recovery controls pass, including the old tuple/list defect and
refusal of changed coordinates, rewards, state digests and missing fields. Two
control setup failures remain recorded: a root bootstrap-path setup error, then
a control hook misclassifying interpreter standard-library ZIP probes as data.
The narrow hook repair permits only the interpreter-declared read-only archive;
scientific payloads remain prohibited in that import control. Neither failure
performed a native replay or training update.

## Cost and scientific boundary

| Actual execution | Failed original pair | Replay-only recovery | Combined |
|---|---:|---:|---:|
| Native previews | 3,458 | 2,457 | 5,915 |
| Frozen encoder forwards | 24 | 0 | 24 |
| Parent time, seconds | 67.317 | 52.854 | 120.171 |
| Worker time, seconds | 66.723 | 52.466 | 119.189 |
| Peak process-group RSS, bytes | 1,426,227,200 | 656,637,952 | — |

The original pair also pays for reproducing the 64-update head, direct readout
and tail solve. Recovery adds zero optimizer updates, model calls or new plans;
its 26 transitions are the two thirteen-action replays. Earlier cache failure,
direct calibration and tail comparison remain paid, bringing the referenced
sequence to 90,905 previews, 678 encoder forwards and 2,061.424 parent seconds.
The older original Q64 experiment remains an additional historical cost.
Inclusive phases are nested and must not be summed again. Timings are single
instrumented executions, not repeated latency estimates.

These are generated interactions on annotation-assisted real TRAIN anatomy,
with less than 1% of the full target removed. They are not observed surgery,
scan-only generalization, cutting-force validation or neurological safety
evidence. SELECT/EVAL, patient splits and physical-measurement claims remain
unchanged. No new dataset, dependency, model collection or paid service was used.

## Evidence and reproduction

`inventory.json` maps compact source, controls and result copies to their exact
local origins and hashes. It includes the failed source/run, both sealed plans,
separate recovery source/run, original and recovery auditors, control failures
and successful controls. The prepared original pair auditor was not run as a
successful pair audit; `recovery-audit-01.json` is the terminal result.

Patient arrays, model weights and the authenticated feature cache remain outside
Git. Their existing manifests and hashes identify local prerequisites. Restore
archived source/evidence to the recorded `build/` paths without overwriting
existing work to reproduce saved checks. `independent/audit_recovery.py` takes
the exact recovery report, launcher and source-index hashes from its receipt
and a new exclusive output path. It reads saved JSON/source only and does not
replace the fresh geometric replay or supply physical validation.

`storage.json` records measured extension growth and free space after this
compact archive. The completed shared stage remains within its 8 MiB allowance;
the next frontier preparation has a separate bounded reservation.
