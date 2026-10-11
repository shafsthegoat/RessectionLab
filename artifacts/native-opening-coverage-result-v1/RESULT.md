# Opening decisions: benefit over return-only, tie with existing heuristic

Retaining a costly preparation action improves the existing generated opening
task. The finer partial-evacuation signal adds **no demonstrated planning benefit**
over the retained depth-and-whole-volume heuristic. Keep the simpler heuristic
as the decision baseline; keep partial evacuation as an opt-in modeled quantity.

| Matched arm | Search transitions | Tool actions | Reward | Final target / outside target, mm³ | Outside volume before first whole-target removal, mm³ |
|---|---:|---:|---:|---:|---:|
| Return-only beam | 16 | 3 | 1.039 | 2 / 4 | 3.625 |
| Certified evacuation lane | 10 | 2 | 1.098 | 2 / 4 | 4.000 |
| Existing depth-and-whole-volume lane, follow-up pair | 10 | 2 | 1.098 | 2 / 4 | 4.000 |
| Certified evacuation lane, follow-up pair | 10 | 2 | 1.098 | 2 / 4 | 4.000 |

The initial pair saves six search transitions (37.5%) and one action. Its reward
gain is exactly one avoided action and one avoided tool switch, less the cost of
an extra millimeter of tool travel. Path length **increases from 11 to 12 mm**;
preparation outside volume **increases by 0.375 mm³ (10.34%)**. Final removed
geometry is identical. Neither shorter physical travel nor less tissue removal
is established. The fixed gate passes through lower search cost, not lower
preparation volume.

The first useful opening has negative immediate reward. Reserving a lane for
preparation keeps it available for the following target-bearing action. However,
this opening already removes more whole cells and reaches greater depth. The
follow-up pair tests that explanation directly: old and new lanes produce equal
complete histories, outcomes and search-transition counts. Its useful-gain gate
is **false**, as declared for a tie. Reject attributing the improvement to finer
volume representation. No further variation was tried on this fixture.

## Controlled comparison and independent checks

Both pairs use the same six-cell, scan-derived generated anatomy, two generic
tools, terminal-microstep closure, beam width 2, horizon 8, and 200-transition /
120-second arm limits. Only the retention setting changes. Rewards, full-tool
geometry, occupied-parent shaft clearance and STOP selection are unchanged.
The opening estimate is bound to public anatomy, model, committed state and
executed action; it exposes no private target or future observation.

All four selected routes pass independent full-tool and executed-witness checks.
Separate saved-only auditors reconstruct leaf unions, whole-cell rewards,
partial volumes, preparation cost and the declared gates: **762 and 718 recorded
checks**, both PASS. The ablation confirms equality of the full selected histories.
This is mathematical/geometry checking against independent implementations,
not physical measurement. The prior analytic-volume comparison remains separate.

The integrated source passed 144 controls with 26 optional historical-source
checks skipped; eight new runner controls all pass. The second runner's Python
files are byte-identical to the tested first runner. Both owned executions finish
with exit 0, verified cleanup and no remaining child process. An earlier preflight
refusal occurred before any worker because of two pytest convenience symlinks;
only those unused temporary links were removed, and the refusal record is retained.

## Cost, reproducibility and next work

The first pair costs 1.838 s parent time, 288 native previews and 81,543,168 B
sampled peak RSS. The ablation costs 1.646 s, 252 previews and 80,445,440 B peak.
Together they execute 540 previews and 65 task transitions, including setup,
selected-route replay and fixed positive controls. Native preview time totals
1.106 s. The independent saved audits take 0.062 and 0.042 s with no native calls.
No training, actor calls, new data, weights or dependencies enter either pair.
Fixed-order single pairs do not support a general latency claim; nested timing
scopes overlap and must not be added.

[The source/evidence index](ARCHIVE_INDEX.json) binds the compact copied records
to canonical commit `83df385f5b21fd11281d52363871e4df0fcacff6`.
[First audit](independent-audit/REPORT.txt) and
[attribution audit](ablation-audit/REPORT.txt) preserve the tradeoffs and negative
result. Runner READMEs provide exact source-freeze and bounded execution commands;
reproduction needs fresh output paths and explicit resource allocation.

This is one nonanatomical generated task with idealized geometric evacuation.
It establishes no learned-policy gain, patient transfer, force response,
neurological safety or biological cutting fidelity. The next model experiment
must address the existing TRAIN learning failure, using complete supported
continuations and fixed patient roles. Adding this scalar to an actor or training
longer on the same toy task is not justified by these results.
