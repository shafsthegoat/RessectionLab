# Residual blocker correction: more feasible actions, one local unlock

On the authenticated stalled patient state, the revised rule offers 22 extra
actions with 9 feasible, versus the retained rule's 32 with 3 feasible. It
removes an incorrect eligibility gate and avoids clearly impossible aperture
pairs before spending the fixed candidate budget. The original 117 base rays,
source, world, rewards and all 47 historical actions remain exact.

Three independently tested preparations each remove one modeled non-target
cell, about 1 mm³, with zero target gain. Their incremental returns are
−0.267, −0.269 and −0.269. The third opens one previously blocked original
exposed-opening ray; the other two open none. The newly feasible ray is only
previewed, removes another non-target cell and has negative incremental return
−0.243. All 52 nominal-target rays remain blocked. No useful complete route,
RL improvement or new-patient generalization is established.

This narrows the next failure. Two preparations remove complementary blockers
in the same target corridor. Individually, each leaves the other obstruction;
some deeper target rays then encounter another blocking layer. The 16-cell
selection cap also omits 17 eligible tool pairs across 11 cells, including the
previously identified central obstruction. No emitted feasible move removes
that obstruction. Six other feasible extras were not branched within the
declared three-branch budget, so this is not an exhaustive route test.

The next targeted change is to select and continue preparations around a
shared target corridor's dependencies, preserving the same public inputs,
candidate limits, reward and full-tool checks. A saved-data check already
rejects simple target-ray frequency as sufficient: the central blocker still
ranks eighteenth for sixteen slots. A full fresh useful route remains the
acceptance criterion; neither a combined preparation sequence nor another
horizon increase is assumed to work.

Validation and cost:

- 119 canonical controls and one generated causal helper pass.
- Independent saved audit verifies all 47 historical actions, all 351 branch
  re-previews and four independent geometry certificates. Full prefix/prep
  histories match their certificates; deterministic native replay is separate.
- 6,445 previews, 53 native commits and one STOP; four independent geometry
  checks take 5.116 s. Parent runtime is 93.637 s, sampled peak memory
  553,926,656 B and output 15,756,307 B. Cleanup is complete.
- Zero fitting, learned forwards, checkpoint loads, new data or dependencies.
  SELECT013 remains development-exposed; 037 is held and EVAL067 closed.

These checks validate the declared geometric simulator against an independent
implementation on real anatomy. They do not validate cutting forces, tissue
deformation, living-brain interaction or neurological safety. The simulated
occupancy assumption and unassessed external workspace remain explicit.

Runtime HEAD: `42fe588907626de402731a4f588d9936437a7a00`.
Result SHA256: `532581e440afdf656c8a3f24ed730e30beba45470588f9e5a337932e97ffb12c`.
Receipt SHA256: `cd744522216e3e6bf27fed44c1c5be67c225a4cd0f8d5e8aa4ad477dfbab213d`.
Independent saved audit: `24fa703b318f972baa8e79a3be577e23cfb001fc82801597075c13757810cd52`.
The compact archive stores the original history once and exact preparation
deltas; root verified lossless reconstruction. Full repeated output remains
local with hashes. Patient arrays and weights are excluded from Git.

At 11:02:32 UTC, extension growth is 189.438 MiB with 332.713 GiB free, before
this 2.03 MiB compact archive and later Git/source changes. No dataset pilot
is needed for the demonstrated action-selection failure.
